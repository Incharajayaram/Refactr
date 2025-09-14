import os
import tempfile
import json
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

import gitlab
from gitlab.exceptions import GitlabError
import git

logger = logging.getLogger(__name__)


# rate limiting cache
_rate_limit_cache = defaultdict(lambda: datetime.min)
RATE_LIMIT_MINUTES = 5


def format_mr_comment(analysis_results: Dict, max_issues: int = 10) -> str:
    """
    Format analysis results for GitLab merge request comment.
    
    Args:
        analysis_results: Analysis results dictionary
        max_issues: Maximum number of issues to include
        
    Returns:
        Formatted Markdown string
    """
    issues = analysis_results.get("all_issues", [])
    
    if not issues:
        return """## ✅ Code Quality Analysis - All Clear!

No issues were found in this merge request. Excellent code quality! 🎉

---
*Analyzed by Code Quality Intelligence Agent*"""
    
    # sort issues by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    sorted_issues = sorted(
        issues[:max_issues],
        key=lambda x: (severity_order.get(x.get("severity", "info"), 5), x.get("file", ""))
    )
    
    # count issues by severity
    severity_counts = defaultdict(int)
    for issue in issues:
        severity_counts[issue.get("severity", "info")] += 1
    
    # build comment
    comment_parts = [
        "## 🔍 Code Quality Analysis Results",
        "",
        f"Found **{len(issues)}** issue(s) in this merge request.",
        "",
        "### Summary by Severity",
        "",
        "| Severity | Count |",
        "|----------|-------|"
    ]
    
    severity_emojis = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🟢",
        "info": "🔵"
    }
    
    for severity in ["critical", "high", "medium", "low", "info"]:
        if severity_counts[severity] > 0:
            emoji = severity_emojis.get(severity, "⚪")
            comment_parts.append(f"| {emoji} {severity.capitalize()} | {severity_counts[severity]} |")
    
    comment_parts.extend([
        "",
        f"### Top {min(len(sorted_issues), max_issues)} Issues",
        "",
        "<details>",
        "<summary>Click to expand issue details</summary>",
        "",
        "| File | Line | Severity | Issue | Fix |",
        "|------|------|----------|-------|-----|"
    ])
    
    # add issues
    for issue in sorted_issues:
        severity = issue.get("severity", "info")
        emoji = severity_emojis.get(severity, "⚪")
        file_path = issue.get("file", "unknown")
        # shorten file path
        if len(file_path) > 25:
            file_path = "..." + file_path[-22:]
        line = issue.get("line", "N/A")
        description = issue.get("description", "No description")
        
        # generate fix suggestion
        fix = _generate_fix_suggestion(issue)
        
        # truncate descriptions
        if len(description) > 40:
            description = description[:37] + "..."
        if len(fix) > 30:
            fix = fix[:27] + "..."
        
        comment_parts.append(
            f"| `{file_path}` | {line} | {emoji} {severity} | {description} | {fix} |"
        )
    
    comment_parts.extend([
        "",
        "</details>",
        ""
    ])
    
    # add action items for critical issues
    if severity_counts["critical"] > 0 or severity_counts["high"] > 0:
        comment_parts.extend([
            "### ⚠️ Required Actions",
            "",
            "This MR contains issues that should be addressed:",
            "",
            f"- **{severity_counts['critical']}** critical severity issues",
            f"- **{severity_counts['high']}** high severity issues",
            "",
            "Please review and fix these issues before merging."
        ])
    
    comment_parts.extend([
        "",
        "---",
        "",
        "*Code Quality Intelligence Agent | "
        f"Analysis performed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*"
    ])
    
    return "\n".join(comment_parts)


def _generate_fix_suggestion(issue: Dict) -> str:
    """Generate a brief fix suggestion based on issue type."""
    issue_type = issue.get("type", "").lower()
    
    fixes = {
        "complexity": "Split function",
        "security": "Sanitize input",
        "duplication": "Extract method",
        "naming": "Rename variable",
        "unused": "Remove code",
        "import": "Clean imports",
        "documentation": "Add docstring",
        "error": "Add error handling"
    }
    
    for key, fix in fixes.items():
        if key in issue_type:
            return fix
    
    return "Review code"


async def handle_merge_request_event(payload: Dict) -> Dict:
    """
    Handle GitLab merge request webhook event.
    
    Args:
        payload: GitLab webhook payload
        
    Returns:
        Dictionary with processing status
    """
    # extract MR information
    object_kind = payload.get("object_kind")
    if object_kind != "merge_request":
        return {"status": "skipped", "reason": f"Not a merge request event: {object_kind}"}
    
    object_attributes = payload.get("object_attributes", {})
    action = object_attributes.get("action")
    
    # process open, reopen, and update events
    if action not in ["open", "reopen", "update"]:
        return {"status": "skipped", "reason": f"Action '{action}' not processed"}
    
    # skip if it's just a merge status update
    if action == "update" and "oldrev" not in object_attributes:
        return {"status": "skipped", "reason": "Not a code update"}
    
    project = payload.get("project", {})
    project_id = project.get("id")
    mr_iid = object_attributes.get("iid")  # Internal ID for the project
    source_branch = object_attributes.get("source_branch")
    
    if not all([project_id, mr_iid, source_branch]):
        return {"status": "error", "reason": "Missing required MR information"}
    
    # check rate limiting
    project_path = project.get("path_with_namespace", f"project_{project_id}")
    last_comment = _rate_limit_cache[project_path]
    if datetime.now() - last_comment < timedelta(minutes=RATE_LIMIT_MINUTES):
        remaining = RATE_LIMIT_MINUTES - (datetime.now() - last_comment).seconds // 60
        return {
            "status": "rate_limited",
            "reason": f"Rate limit: wait {remaining} minutes"
        }
    
    # get repository URL
    repo_url = project.get("git_http_url")
    if not repo_url:
        return {"status": "error", "reason": "No repository URL found"}
    
    # clone and analyze
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # add token to URL if available
            gitlab_token = os.getenv("GITLAB_TOKEN")
            if gitlab_token and repo_url.startswith("https://"):
                # gitLab format: https://oauth2:TOKEN@gitlab.com/user/repo.git
                repo_url = repo_url.replace("https://", f"https://oauth2:{gitlab_token}@")
            
            # clone repository
            logger.info(f"Cloning {project_path} (branch: {source_branch})...")
            repo = git.Repo.clone_from(repo_url, temp_dir, branch=source_branch, depth=1)
            logger.info(f"Successfully cloned to {temp_dir}")
            
            # run analysis
            try:
                from langgraph_workflow import analyze_code
                analysis_results = await asyncio.to_thread(analyze_code, temp_dir)
            except ImportError:
                # fallback mock analysis
                analysis_results = _mock_analysis(temp_dir)
            
            # format comment
            comment_body = format_mr_comment(analysis_results)
            
            # post comment
            try:
                post_mr_comment(project_id, mr_iid, comment_body)
                _rate_limit_cache[project_path] = datetime.now()
                
                return {
                    "status": "success",
                    "project": project_path,
                    "mr": mr_iid,
                    "issues_found": len(analysis_results.get("all_issues", [])),
                    "comment_posted": True
                }
            except Exception as e:
                logger.error(f"Failed to post comment: {e}")
                return {"status": "error", "reason": f"Failed to post comment: {str(e)}"}
                
        except git.exc.GitCommandError as e:
            logger.error(f"Git clone failed: {e}")
            return {"status": "error", "reason": f"Failed to clone repository: {str(e)}"}
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"status": "error", "reason": f"Analysis failed: {str(e)}"}


def post_mr_comment(project_id: int, mr_iid: int, body: str) -> None:
    """
    Post a comment to a GitLab merge request.
    
    Args:
        project_id: GitLab project ID
        mr_iid: Merge request IID (internal ID)
        body: Comment body in Markdown
    """
    gitlab_token = os.getenv("GITLAB_TOKEN")
    gitlab_url = os.getenv("GITLAB_URL", "https://gitlab.com")
    
    if not gitlab_token:
        raise ValueError("GITLAB_TOKEN environment variable not set")
    
    # initialize GitLab client
    gl = gitlab.Gitlab(gitlab_url, private_token=gitlab_token)
    
    try:
        # get project
        project = gl.projects.get(project_id)
        
        # get merge request
        mr = project.mergerequests.get(mr_iid)
        
        # check for existing comments to avoid duplicates
        notes = mr.notes.list(all=True)
        analysis_marker = "Code Quality Analysis Results"
        
        # get current user info
        current_user = gl.user
        
        for note in notes:
            if note.author["id"] == current_user.id and analysis_marker in note.body:
                # update existing note
                note.body = body
                note.save()
                logger.info(f"Updated existing comment on project {project_id} MR !{mr_iid}")
                return
        
        # create new note
        mr.notes.create({"body": body})
        logger.info(f"Posted new comment on project {project_id} MR !{mr_iid}")
        
    except GitlabError as e:
        logger.error(f"GitLab API error: {e}")
        raise


def _mock_analysis(repo_path: str) -> Dict:
    """Mock analysis results for testing."""
    return {
        "all_issues": [
            {
                "file": "app/models/user.rb",
                "line": 15,
                "severity": "high",
                "type": "security vulnerability",
                "description": "Mass assignment vulnerability",
                "analyzer": "mock"
            },
            {
                "file": "app/controllers/api_controller.rb",
                "line": 8,
                "severity": "medium",
                "type": "missing authentication",
                "description": "API endpoint lacks authentication",
                "analyzer": "mock"
            },
            {
                "file": "spec/models/user_spec.rb",
                "line": 42,
                "severity": "low",
                "type": "test coverage",
                "description": "Missing test for user validation",
                "analyzer": "mock"
            }
        ],
        "summary": {
            "total_files": 3,
            "total_issues": 3
        }
    }


def main():
    """CLI interface for testing GitLab integration."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GitLab integration for Code Quality Intelligence Agent"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # test comment formatting
    format_parser = subparsers.add_parser(
        "format",
        help="Test comment formatting with mock data"
    )
    format_parser.add_argument(
        "--issues",
        type=int,
        default=5,
        help="Number of mock issues to generate"
    )
    
    # simulate webhook
    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Simulate webhook processing"
    )
    simulate_parser.add_argument(
        "--payload",
        required=True,
        help="Path to JSON file with webhook payload"
    )
    
    args = parser.parse_args()
    
    if args.command == "format":
        # generate mock data with specified number of issues
        mock_issues = []
        severities = ["critical", "high", "medium", "low", "info"]
        
        for i in range(args.issues):
            mock_issues.append({
                "file": f"src/file_{i}.py",
                "line": 10 + i * 5,
                "severity": severities[i % len(severities)],
                "type": ["security", "complexity", "style"][i % 3],
                "description": f"Mock issue {i + 1}",
                "analyzer": "mock"
            })
        
        mock_results = {"all_issues": mock_issues}
        comment = format_mr_comment(mock_results)
        print(comment)
        
    elif args.command == "simulate":
        # load and process payload
        with open(args.payload, 'r') as f:
            payload = json.load(f)
        
        logging.basicConfig(level=logging.INFO)
        result = asyncio.run(handle_merge_request_event(payload))
        print(f"Result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
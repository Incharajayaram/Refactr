import os
import hmac
import hashlib
import tempfile
import shutil
import json
import argparse
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

from github import Github, GithubException
from github.PullRequest import PullRequest
import git

logger = logging.getLogger(__name__)


# rate limiting cache - stores last comment time per repo
_rate_limit_cache = defaultdict(lambda: datetime.min)
RATE_LIMIT_MINUTES = 5  # Minimum time between comments on same repo


def verify_github_signature(request_body: bytes, header_signature: str) -> bool:
    """
    Verify GitHub webhook signature for security.
    
    Args:
        request_body: Raw request body bytes
        header_signature: X-Hub-Signature-256 header value
        
    Returns:
        True if signature is valid, False otherwise
    """
    webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET", "")
    if not webhook_secret:
        logger.warning("GITHUB_WEBHOOK_SECRET not set, skipping signature verification")
        return True
    
    if not header_signature:
        return False
    
    # gitHub sends the signature in format "sha256=<signature>"
    if not header_signature.startswith("sha256="):
        return False
    
    expected_signature = header_signature.split("=", 1)[1]
    
    # calculate HMAC
    mac = hmac.new(
        webhook_secret.encode(),
        request_body,
        hashlib.sha256
    )
    calculated_signature = mac.hexdigest()
    
    # compare signatures securely
    return hmac.compare_digest(calculated_signature, expected_signature)


def format_analysis_summary(analysis_results: Dict, max_issues: int = 10) -> str:
    """
    Format analysis results into a compact Markdown summary for PR comment.
    
    Args:
        analysis_results: Analysis results dictionary
        max_issues: Maximum number of issues to include
        
    Returns:
        Formatted Markdown string
    """
    issues = analysis_results.get("all_issues", [])
    
    if not issues:
        return """## ✅ Code Quality Analysis - All Clear!

No issues were found in this pull request. Great job maintaining code quality! 🎉

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
    
    # build summary
    summary_parts = [
        "## 🔍 Code Quality Analysis Results",
        "",
        f"Found **{len(issues)}** issue(s) in this pull request.",
        "",
        "### Summary by Severity",
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
            summary_parts.append(f"| {emoji} {severity.capitalize()} | {severity_counts[severity]} |")
    
    summary_parts.extend([
        "",
        f"### Top {min(len(sorted_issues), max_issues)} Issues",
        "",
        "| Severity | File | Line | Issue | Suggestion |",
        "|----------|------|------|-------|------------|"
    ])
    
    # add top issues
    for issue in sorted_issues:
        severity = issue.get("severity", "info")
        emoji = severity_emojis.get(severity, "⚪")
        file_path = issue.get("file", "unknown")
        # shorten file path for readability
        if len(file_path) > 30:
            file_path = "..." + file_path[-27:]
        line = issue.get("line", "N/A")
        issue_type = issue.get("type", "Unknown")
        description = issue.get("description", "No description")
        
        # generate suggestion based on issue type
        suggestion = _generate_suggestion(issue)
        
        # truncate long descriptions
        if len(description) > 50:
            description = description[:47] + "..."
        if len(suggestion) > 40:
            suggestion = suggestion[:37] + "..."
        
        summary_parts.append(
            f"| {emoji} {severity} | `{file_path}` | {line} | {description} | {suggestion} |"
        )
    
    # add recommendations
    if severity_counts["critical"] > 0 or severity_counts["high"] > 0:
        summary_parts.extend([
            "",
            "### ⚠️ Action Required",
            "",
            f"This PR contains **{severity_counts['critical']}** critical and "
            f"**{severity_counts['high']}** high severity issues that should be addressed before merging.",
            "",
            "Please review the issues above and consider fixing them to improve code quality."
        ])
    
    summary_parts.extend([
        "",
        "---",
        "",
        "*This analysis was performed automatically by the [Code Quality Intelligence Agent](https://github.com/your-org/code-quality-agent). "
        "Please report any false positives or issues.*",
        "",
        f"*Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*"
    ])
    
    return "\n".join(summary_parts)


def _generate_suggestion(issue: Dict) -> str:
    """Generate a brief suggestion based on issue type."""
    issue_type = issue.get("type", "").lower()
    
    suggestions = {
        "complexity": "Refactor into smaller functions",
        "security": "Review security best practices",
        "duplication": "Extract common code",
        "naming": "Use descriptive names",
        "unused": "Remove unused code",
        "import": "Organize imports",
        "documentation": "Add documentation",
        "error": "Handle error cases",
        "performance": "Optimize for performance"
    }
    
    for key, suggestion in suggestions.items():
        if key in issue_type:
            return suggestion
    
    return "Review and fix"


async def handle_pull_request_event(payload: Dict) -> Dict:
    """
    Handle GitHub pull request webhook event.
    
    Args:
        payload: GitHub webhook payload
        
    Returns:
        Dictionary with processing status
    """
    # extract PR information
    action = payload.get("action")
    pull_request = payload.get("pull_request", {})
    repository = payload.get("repository", {})
    
    # only process opened, synchronize (new commits), or reopened events
    if action not in ["opened", "synchronize", "reopened"]:
        return {"status": "skipped", "reason": f"Action '{action}' not processed"}
    
    owner = repository.get("owner", {}).get("login")
    repo_name = repository.get("name")
    pr_number = pull_request.get("number")
    
    if not all([owner, repo_name, pr_number]):
        return {"status": "error", "reason": "Missing required PR information"}
    
    # check rate limiting
    repo_key = f"{owner}/{repo_name}"
    last_comment = _rate_limit_cache[repo_key]
    if datetime.now() - last_comment < timedelta(minutes=RATE_LIMIT_MINUTES):
        remaining = RATE_LIMIT_MINUTES - (datetime.now() - last_comment).seconds // 60
        return {
            "status": "rate_limited",
            "reason": f"Rate limit: wait {remaining} minutes before next comment"
        }
    
    # get clone URL (prefer HTTPS with token)
    head = pull_request.get("head", {})
    clone_url = head.get("repo", {}).get("clone_url")
    branch = head.get("ref")
    
    if not clone_url:
        return {"status": "error", "reason": "No clone URL found"}
    
    # clone and analyze
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # clone the repository
            logger.info(f"Cloning {clone_url} (branch: {branch})...")
            
            # add token to clone URL if available
            github_token = os.getenv("GITHUB_TOKEN")
            if github_token and clone_url.startswith("https://"):
                clone_url = clone_url.replace("https://", f"https://{github_token}@")
            
            repo = git.Repo.clone_from(clone_url, temp_dir, branch=branch, depth=1)
            logger.info(f"Successfully cloned to {temp_dir}")
            
            # run analysis
            try:
                from langgraph_workflow import analyze_code
                analysis_results = await asyncio.to_thread(analyze_code, temp_dir)
            except ImportError:
                # fallback mock analysis for testing
                analysis_results = _mock_analysis(temp_dir)
            
            # format and post comment
            comment_body = format_analysis_summary(analysis_results)
            
            # post comment to PR
            try:
                post_pr_comment(owner, repo_name, pr_number, comment_body)
                _rate_limit_cache[repo_key] = datetime.now()
                
                return {
                    "status": "success",
                    "pr": f"{owner}/{repo_name}#{pr_number}",
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


def post_pr_comment(owner: str, repo: str, pr_number: int, body: str) -> None:
    """
    Post a comment to a GitHub pull request.
    
    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: Pull request number
        body: Comment body in Markdown
    """
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("GITHUB_TOKEN environment variable not set")
    
    # initialize GitHub client
    g = Github(github_token)
    
    try:
        # get repository
        repository = g.get_repo(f"{owner}/{repo}")
        
        # get pull request
        pull_request = repository.get_pull(pr_number)
        
        # check if we've already commented (avoid duplicates)
        bot_login = g.get_user().login
        existing_comments = pull_request.get_issue_comments()
        
        # look for existing analysis comment
        analysis_marker = "Code Quality Analysis"
        for comment in existing_comments:
            if comment.user.login == bot_login and analysis_marker in comment.body:
                # update existing comment instead of creating new one
                comment.edit(body)
                logger.info(f"Updated existing comment on {owner}/{repo}#{pr_number}")
                return
        
        # post new comment
        pull_request.create_issue_comment(body)
        logger.info(f"Posted new comment on {owner}/{repo}#{pr_number}")
        
    except GithubException as e:
        logger.error(f"GitHub API error: {e}")
        raise


def _mock_analysis(repo_path: str) -> Dict:
    """Mock analysis results for testing."""
    return {
        "all_issues": [
            {
                "file": "src/main.py",
                "line": 42,
                "severity": "high",
                "type": "complexity",
                "description": "Function 'process_data' has complexity of 15",
                "analyzer": "mock"
            },
            {
                "file": "src/utils.py",
                "line": 23,
                "severity": "medium",
                "type": "security",
                "description": "Potential SQL injection vulnerability",
                "analyzer": "mock"
            },
            {
                "file": "tests/test_main.py",
                "line": 10,
                "severity": "low",
                "type": "unused import",
                "description": "Unused import 'sys'",
                "analyzer": "mock"
            }
        ],
        "summary": {
            "total_files": 3,
            "total_issues": 3
        }
    }


def main():
    """CLI interface for testing GitHub integration."""
    parser = argparse.ArgumentParser(
        description="GitHub integration for Code Quality Intelligence Agent"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # simulate command
    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Simulate webhook processing with sample payload"
    )
    simulate_parser.add_argument(
        "--payload",
        required=True,
        help="Path to JSON file with webhook payload"
    )
    simulate_parser.add_argument(
        "--skip-post",
        action="store_true",
        help="Skip posting comment (dry run)"
    )
    
    args = parser.parse_args()
    
    if args.command == "simulate":
        # load payload
        with open(args.payload, 'r') as f:
            payload = json.load(f)
        
        # set up logging
        logging.basicConfig(level=logging.INFO)
        
        # process webhook
        if args.skip_post:
            # override post function for dry run
            global post_pr_comment
            original_post = post_pr_comment
            
            def mock_post(owner, repo, pr, body):
                print(f"\n--- Would post to {owner}/{repo}#{pr} ---")
                print(body)
                print("--- End of comment ---\n")
            
            post_pr_comment = mock_post
        
        # run async handler
        result = asyncio.run(handle_pull_request_event(payload))
        print(f"Result: {json.dumps(result, indent=2)}")
        
        if args.skip_post:
            post_pr_comment = original_post


if __name__ == "__main__":
    main()
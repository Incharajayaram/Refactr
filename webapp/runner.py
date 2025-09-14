import os
import json
import shutil
import asyncio
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse

import git
from git import Repo

logger = logging.getLogger(__name__)


class JobStatus:
    """Job status constants."""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


def validate_repo_url(repo_url: str) -> bool:
    """
    Validate that the repository URL is properly formatted.
    
    Args:
        repo_url: The repository URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        parsed = urlparse(repo_url)
        return parsed.scheme in ('http', 'https') and parsed.netloc
    except Exception:
        return False


def construct_clone_url(repo_url: str, token: Optional[str] = None) -> str:
    """
    Construct a clone URL with authentication token if provided.
    
    Args:
        repo_url: The original repository URL
        token: Optional authentication token
        
    Returns:
        The clone URL with embedded token if provided
    """
    if not token:
        return repo_url
    
    # parse the URL
    parsed = urlparse(repo_url)
    
    # for GitHub URLs, embed the token
    if 'github.com' in parsed.netloc:
        # format: https://TOKEN@github.com/owner/repo.git
        return f"{parsed.scheme}://{token}@{parsed.netloc}{parsed.path}"
    
    return repo_url


def local_llm_summary(report: Dict[str, Any]) -> str:
    """
    Generate a natural language summary of the code analysis report.
    This is a placeholder that would call a local LLM in production.
    
    Args:
        report: The analysis report dictionary
        
    Returns:
        A natural language summary
    """
    try:
        # try to import and use the actual LLM function if available
        from langgraph_workflow import local_llm_call
        return local_llm_call(f"Summarize this code analysis report: {json.dumps(report)}")
    except ImportError:
        # fallback to placeholder summary
        total_files = report.get('total_files_analyzed', 0)
        issues = report.get('total_issues', 0)
        
        summary = f"Analysis complete for {total_files} files. "
        
        if issues == 0:
            summary += "No significant issues found. The codebase appears to be well-maintained."
        elif issues < 10:
            summary += f"Found {issues} minor issues that should be addressed."
        else:
            summary += f"Found {issues} issues requiring attention. Consider prioritizing security vulnerabilities and high-complexity functions."
        
        return summary


async def run_job(
    job_id: str,
    repo_url: str,
    branch: str = "main",
    token: Optional[str] = None
) -> Tuple[bool, Any]:
    """
    Run a repository analysis job.
    
    Args:
        job_id: Unique job identifier
        repo_url: Repository URL to analyze
        branch: Branch to analyze (default: main)
        token: Optional authentication token
        
    Returns:
        Tuple of (success: bool, result: dict or error: str)
    """
    job_dir = Path(f"./work/jobs/{job_id}")
    
    try:
        # validate repository URL
        if not validate_repo_url(repo_url):
            return False, "Invalid repository URL"
        
        # create job directory
        job_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created job directory: {job_dir}")
        
        # clone the repository
        clone_url = construct_clone_url(repo_url, token)
        logger.info(f"Cloning repository from {repo_url} (branch: {branch})...")
        
        try:
            repo = Repo.clone_from(
                clone_url,
                job_dir,
                branch=branch,
                depth=1  # Shallow clone for efficiency
            )
            logger.info(f"Successfully cloned repository to {job_dir}")
        except git.exc.GitCommandError as e:
            error_msg = str(e)
            # remove token from error message for security
            if token:
                error_msg = error_msg.replace(token, "***")
            logger.error(f"Git clone failed: {error_msg}")
            return False, f"Failed to clone repository: {error_msg}"
        
        # run the code analysis
        try:
            # import the analyze function
            from langgraph_workflow import analyze_code
            
            logger.info("Starting code analysis...")
            analysis_result = await asyncio.to_thread(analyze_code, str(job_dir))
            
            # add metadata to the report
            report = {
                "job_id": job_id,
                "repository_url": repo_url,
                "branch": branch,
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_results": analysis_result,
                "summary": local_llm_summary(analysis_result)
            }
            
            logger.info(f"Analysis completed successfully for job {job_id}")
            return True, report
            
        except ImportError:
            # fallback if analyze_code is not available
            logger.warning("analyze_code function not available, using mock analysis")
            
            # mock analysis result
            mock_result = {
                "job_id": job_id,
                "repository_url": repo_url,
                "branch": branch,
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_results": {
                    "total_files_analyzed": 42,
                    "total_issues": 7,
                    "languages": ["Python", "JavaScript"],
                    "metrics": {
                        "complexity": {
                            "average": 3.2,
                            "high_complexity_functions": 2
                        },
                        "duplicates": {
                            "duplicate_blocks": 5,
                            "duplicate_lines": 123
                        },
                        "security": {
                            "vulnerabilities": 1,
                            "severity": "medium"
                        }
                    },
                    "recommendations": [
                        "Consider refactoring high-complexity functions",
                        "Remove duplicate code blocks",
                        "Update dependencies with known vulnerabilities"
                    ]
                },
                "summary": "Mock analysis completed. This is a placeholder result."
            }
            
            return True, mock_result
            
    except Exception as e:
        logger.error(f"Unexpected error in job {job_id}: {e}")
        return False, str(e)
    finally:
        # cleanup: Remove the cloned repository
        if job_dir.exists():
            try:
                shutil.rmtree(job_dir)
                logger.info(f"Cleaned up job directory: {job_dir}")
            except Exception as e:
                logger.error(f"Failed to cleanup job directory: {e}")


# example report structure for documentation
EXAMPLE_REPORT_JSON = {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "repository_url": "https://github.com/example/repo",
    "branch": "main",
    "analysis_timestamp": "2024-01-15T10:30:00.000000",
    "analysis_results": {
        "total_files_analyzed": 156,
        "total_issues": 23,
        "languages": {
            "Python": {
                "files": 89,
                "lines": 12450,
                "issues": 15
            },
            "JavaScript": {
                "files": 67,
                "lines": 8930,
                "issues": 8
            }
        },
        "metrics": {
            "complexity": {
                "average": 4.2,
                "distribution": {
                    "low": 120,
                    "medium": 30,
                    "high": 6
                },
                "high_complexity_functions": [
                    {
                        "name": "process_data",
                        "file": "src/data_processor.py",
                        "line": 45,
                        "complexity": 15
                    }
                ]
            },
            "duplicates": {
                "duplicate_blocks": 12,
                "duplicate_lines": 340,
                "files_with_duplicates": [
                    "src/utils.py",
                    "tests/test_utils.py"
                ]
            },
            "security": {
                "total_vulnerabilities": 3,
                "by_severity": {
                    "high": 1,
                    "medium": 2,
                    "low": 0
                },
                "vulnerabilities": [
                    {
                        "type": "SQL Injection",
                        "severity": "high",
                        "file": "src/database.py",
                        "line": 87,
                        "description": "User input directly concatenated in SQL query"
                    }
                ]
            },
            "code_quality": {
                "linting_errors": 45,
                "type_coverage": 78.5,
                "test_coverage": 82.3
            }
        },
        "file_analyses": [
            {
                "path": "src/main.py",
                "language": "Python",
                "metrics": {
                    "lines": 234,
                    "complexity": 5.2,
                    "issues": 3
                },
                "issues": [
                    {
                        "type": "complexity",
                        "line": 45,
                        "severity": "medium",
                        "message": "Function 'handle_request' has complexity of 12"
                    }
                ]
            }
        ],
        "recommendations": [
            {
                "priority": "high",
                "category": "security",
                "recommendation": "Fix SQL injection vulnerability in database.py"
            },
            {
                "priority": "medium",
                "category": "maintainability",
                "recommendation": "Refactor high-complexity functions to improve readability"
            },
            {
                "priority": "low",
                "category": "code_quality",
                "recommendation": "Remove duplicate code blocks to improve maintainability"
            }
        ]
    },
    "summary": "Analysis of 156 files revealed 23 issues. Priority should be given to fixing 1 high-severity security vulnerability. The codebase shows good test coverage (82.3%) but has some complexity hotspots that should be refactored."
}


if __name__ == "__main__":
    # test the runner with a mock job
    import asyncio
    
    async def test_runner():
        success, result = await run_job(
            "test-job-123",
            "https://github.com/python/cpython",
            "main"
        )
        print(f"Success: {success}")
        print(f"Result: {json.dumps(result, indent=2)}")
    
    asyncio.run(test_runner())
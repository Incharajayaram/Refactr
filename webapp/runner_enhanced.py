import os
import asyncio
import subprocess
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
import networkx as nx

from webapp.runner import JobStatus  # Reuse existing JobStatus
from analyzers.python_analyzer import analyze_python
from analyzers.js_analyzer import analyze_javascript
from report.generator import ReportGenerator
from report.visualizations import (
    build_call_graph,
    save_dependency_graph_png,
    generate_hotspot_heatmap,
    render_dependency_network_html,
    compute_file_metrics
)

logger = logging.getLogger(__name__)

# directories
WORK_DIR = Path("./work")
JOBS_DIR = WORK_DIR / "jobs"
REPORTS_DIR = WORK_DIR / "reports"
VISUALS_DIR = WORK_DIR / "visualizations"

# ensure directories exist
for dir_path in [JOBS_DIR, REPORTS_DIR, VISUALS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


async def run_enhanced_job(
    job_id: str,
    repo_url: str,
    branch: str = "main",
    token: Optional[str] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run a complete analysis job with visualizations.
    
    Args:
        job_id: Unique job identifier
        repo_url: Repository URL to analyze
        branch: Branch to analyze
        token: Authentication token for private repos
        
    Returns:
        Tuple of (success, result_or_error)
    """
    job_dir = JOBS_DIR / job_id
    visuals_dir = VISUALS_DIR / job_id
    
    try:
        # ensure directories exist with proper permissions
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        visuals_dir.mkdir(parents=True, exist_ok=True)
        
        # step 1: Clone repository
        logger.info(f"Cloning repository {repo_url} to {job_dir}")
        
        # ensure job directory doesn't exist
        if job_dir.exists():
            shutil.rmtree(job_dir)
        
        # prepare git command
        env = os.environ.copy()
        if token:
            # add token to URL for private repos
            if "github.com" in repo_url:
                repo_url = repo_url.replace("https://", f"https://{token}@")
        
        # disable symlinks for Windows compatibility
        cmd = ["git", "-c", "core.symlinks=false", "clone", "--depth", "1", "--branch", branch, repo_url, str(job_dir)]
        
        # use subprocess.run for Windows compatibility
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            error_msg = result.stderr
            logger.error(f"Git clone failed: {error_msg}")
            return False, f"Failed to clone repository: {error_msg}"
        
        # step 2: Analyze code
        logger.info("Starting code analysis...")
        issues = []
        file_metrics = {}
        
        # analyze Python files
        python_files = list(job_dir.rglob("*.py"))
        for py_file in python_files:
            if any(part in str(py_file) for part in ['.venv', 'venv', '__pycache__']):
                continue
            try:
                file_issues = analyze_python(str(py_file))
                issues.extend(file_issues)
                
                # collect metrics for visualization
                rel_path = py_file.relative_to(job_dir)
                try:
                    lines = len(py_file.read_text(encoding='utf-8').splitlines())
                except UnicodeDecodeError:
                    lines = len(py_file.read_text(encoding='utf-8', errors='ignore').splitlines())
                
                file_metrics[str(rel_path)] = {
                    "language": "Python",
                    "lines": lines,
                    "issues": len([i for i in file_issues if i['file'] == str(py_file)]),
                    "complexity": sum(1 for i in file_issues if i.get('issue_type') == 'high_complexity')
                }
            except Exception as e:
                logger.warning(f"Error analyzing {py_file}: {e}")
        
        # analyze JavaScript files
        js_extensions = ["*.js", "*.jsx", "*.ts", "*.tsx"]
        js_files = []
        for ext in js_extensions:
            js_files.extend(job_dir.rglob(ext))
        
        for js_file in js_files:
            if "node_modules" in str(js_file):
                continue
            try:
                file_issues = analyze_javascript(str(js_file))
                issues.extend(file_issues)
                
                # collect metrics
                rel_path = js_file.relative_to(job_dir)
                try:
                    lines = len(js_file.read_text(encoding='utf-8').splitlines())
                except UnicodeDecodeError:
                    lines = len(js_file.read_text(encoding='utf-8', errors='ignore').splitlines())
                
                file_metrics[str(rel_path)] = {
                    "language": "JavaScript",
                    "lines": lines,
                    "issues": len([i for i in file_issues if i['file'] == str(js_file)]),
                    "complexity": sum(1 for i in file_issues if i.get('issue_type') == 'high_complexity')
                }
            except Exception as e:
                logger.warning(f"Error analyzing {js_file}: {e}")
        
        # step 3: Generate visualizations
        logger.info("Generating visualizations...")
        visualization_paths = {}
        
        try:
            # try SVG visualizations first (most reliable)
            try:
                from report.svg_visualizations import generate_svg_visualizations
                logger.info("Using SVG visualization engine...")
                visualization_paths = generate_svg_visualizations(str(job_dir), file_metrics, visuals_dir)
                logger.info(f"Generated {len(visualization_paths)} SVG visualizations")
            except Exception as e:
                logger.warning(f"SVG visualizations failed: {e}")
                # try better visualizations as fallback
                try:
                    from report.better_visualizations import generate_better_visualizations
                    logger.info("Using enhanced visualization engine...")
                    visualization_paths = generate_better_visualizations(job_dir, file_metrics, visuals_dir)
                    logger.info(f"Generated {len(visualization_paths)} enhanced visualizations")
                except ImportError:
                    # try fast visualizations as fallback
                    try:
                        from report.fast_visualizations import generate_fast_visualizations
                        logger.info("Using fast visualization engine...")
                        visualization_paths = generate_fast_visualizations(job_dir, file_metrics, visuals_dir)
                        logger.info(f"Generated {len(visualization_paths)} fast visualizations")
                    except ImportError:
                        logger.info("Fast visualizations not available, using traditional method...")
                    # fallback to traditional method
                    logger.info("Building dependency graph...")
                    import time
                    
                    start_time = time.time()
                    max_time = 30  # 30 seconds max for visualization
                    
                    try:
                        # for large repos, limit the graph size
                        graph = build_call_graph(str(job_dir))
                        
                        # only proceed if graph is reasonable size
                        if graph.number_of_nodes() > 1000:
                            logger.warning(f"Graph too large ({graph.number_of_nodes()} nodes), skipping visualization")
                        else:
                            # generate dependency graph PNG
                            dep_graph_path = visuals_dir / "dependency_graph.png"
                            save_dependency_graph_png(graph, str(dep_graph_path))
                            visualization_paths["dependency_graph"] = str(dep_graph_path.relative_to(WORK_DIR))
                            
                            # generate interactive dependency network (skip for large graphs)
                            if graph.number_of_nodes() < 500:
                                dep_network_path = visuals_dir / "dependency_network.html"
                                render_dependency_network_html(graph, str(dep_network_path))
                                visualization_paths["dependency_network"] = str(dep_network_path.relative_to(WORK_DIR))
                    except Exception as e:
                        logger.warning(f"Error with dependency graph: {e}")
                    
                    # always try to generate heatmap (it's faster)
                    logger.info("Generating code quality heatmap...")
                    heatmap_path = visuals_dir / "quality_heatmap.png"
                    generate_hotspot_heatmap(file_metrics, str(heatmap_path))
                    visualization_paths["quality_heatmap"] = str(heatmap_path.relative_to(WORK_DIR))
            
        except Exception as e:
            logger.warning(f"Error generating visualizations: {e}")
            # continue without visualizations rather than failing
        
        # step 4: Generate report
        logger.info("Generating report...")
        report_generator = ReportGenerator()
        report_dir = REPORTS_DIR / job_id
        report_dir.mkdir(exist_ok=True)
        
        # generate report (this creates markdown and optionally HTML)
        report_generator.generate_report(issues, str(report_dir))
        
        # step 5: Compile final result
        analysis_summary = {
            "total_files_analyzed": len(file_metrics),
            "total_issues": len(issues),
            "languages": list(set(m["language"] for m in file_metrics.values())),
            "severity_breakdown": {},
            "type_breakdown": {},
            "file_metrics": file_metrics
        }
        
        # count issues by severity and type
        for issue in issues:
            severity = issue.get("severity", "medium")
            issue_type = issue.get("issue_type", "unknown")
            
            analysis_summary["severity_breakdown"][severity] = \
                analysis_summary["severity_breakdown"].get(severity, 0) + 1
            analysis_summary["type_breakdown"][issue_type] = \
                analysis_summary["type_breakdown"].get(issue_type, 0) + 1
        
        report = {
            "job_id": job_id,
            "repository_url": repo_url,
            "branch": branch,
            "analysis_timestamp": datetime.now().isoformat(),
            "summary": analysis_summary,
            "issues": issues,
            "visualizations": visualization_paths,
            "report_files": {
                "markdown": str((report_dir / f"code_quality_report_{report_generator.timestamp.strftime('%Y%m%d_%H%M%S')}.md").relative_to(WORK_DIR)),
                "html": str((report_dir / f"code_quality_report_{report_generator.timestamp.strftime('%Y%m%d_%H%M%S')}.html").relative_to(WORK_DIR)) if (report_dir / f"code_quality_report_{report_generator.timestamp.strftime('%Y%m%d_%H%M%S')}.html").exists() else None
            }
        }
        
        logger.info(f"Analysis completed successfully for job {job_id}")
        return True, report
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error in enhanced job {job_id}: {e}\n{error_details}")
        return False, f"Error: {str(e)}\n\nDetails:\n{error_details}"
    
    finally:
        # cleanup job directory to save space (keep reports and visualizations)
        if job_dir.exists() and not os.getenv("KEEP_JOB_FILES", "false").lower() == "true":
            try:
                shutil.rmtree(job_dir)
                logger.info(f"Cleaned up job directory: {job_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup job directory: {e}")


def get_visualization_path(job_id: str, viz_type: str) -> Optional[Path]:
    """
    Get the path to a specific visualization file.
    
    Args:
        job_id: Job identifier
        viz_type: Type of visualization (dependency_graph, quality_heatmap, etc.)
        
    Returns:
        Path to visualization file or None if not found
    """
    visuals_dir = VISUALS_DIR / job_id
    
    viz_files = {
        "dependency_graph": ["dependency_graph.svg", "dependency_graph.png"],
        "dependency_network": ["dependency_network.svg", "dependency_network.html"],
        "quality_heatmap": ["quality_heatmap.svg", "quality_heatmap.png"]
    }
    
    if viz_type in viz_files:
        # check for each possible file format
        for filename in viz_files[viz_type]:
            viz_path = visuals_dir / filename
            if viz_path.exists():
                return viz_path
    
    return None
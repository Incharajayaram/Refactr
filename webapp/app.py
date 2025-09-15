import os
import json
import uuid
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, validator
import aiofiles

from webapp.runner import run_job, JobStatus
from webapp.runner_enhanced import run_enhanced_job, get_visualization_path
from webapp.qa_routes import router as qa_router
from webapp.webhooks import router as webhook_router
from webapp.qa_ui import router as qa_ui_router
from report.generator import generate_report
import tempfile
from fastapi.responses import FileResponse
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# configuration from environment
ALLOW_OTHER_DOMAINS = os.getenv("ALLOW_OTHER_DOMAINS", "false").lower() == "true"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
KEEP_DAYS = int(os.getenv("KEEP_DAYS", "7"))
WORK_DIR = Path("./work")
JOBS_DIR = WORK_DIR / "jobs"
REPORTS_DIR = WORK_DIR / "reports"
JOBS_META_DIR = WORK_DIR / "jobs_meta"

# ensure directories exist
for dir_path in [JOBS_DIR, REPORTS_DIR, JOBS_META_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# in-memory job registry
jobs_registry: Dict[str, Dict[str, Any]] = {}


class AnalyzeRequest(BaseModel):
    """Request model for repository analysis."""
    repo_url: str
    branch: Optional[str] = "main"
    token: Optional[str] = None
    
    @validator('repo_url')
    def validate_repo_url(cls, v):
        """Validate repository URL."""
        if not ALLOW_OTHER_DOMAINS and "github.com" not in v:
            raise ValueError("Only GitHub repositories are allowed")
        if not v.startswith(("http://", "https://")):
            raise ValueError("Repository URL must start with http:// or https://")
        return v


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None


class AnalyzeResponse(BaseModel):
    """Response model for analyze endpoint."""
    job_id: str
    status: str = "pending"
    message: str = "Analysis job queued successfully"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app."""
    # startup: Load existing job metadata
    port = os.getenv('PORT', '8080')
    logger.info(f"Starting app on port {port}...")
    logger.info("Loading existing job metadata...")
    for meta_file in JOBS_META_DIR.glob("*.json"):
        try:
            with open(meta_file, 'r') as f:
                job_data = json.load(f)
                jobs_registry[job_data['job_id']] = job_data
        except Exception as e:
            logger.error(f"Failed to load job metadata {meta_file}: {e}")
    
    # cleanup old jobs on startup
    cleanup_old_jobs()
    
    yield
    
    # shutdown: Save job metadata
    logger.info("Saving job metadata...")
    for job_id, job_data in jobs_registry.items():
        try:
            meta_file = JOBS_META_DIR / f"{job_id}.json"
            with open(meta_file, 'w') as f:
                json.dump(job_data, f, default=str)
        except Exception as e:
            logger.error(f"Failed to save job metadata {job_id}: {e}")


app = FastAPI(
    title="Code Quality Intelligence Agent",
    description="Analyze GitHub repositories for code quality insights",
    version="1.0.0",
    lifespan=lifespan
)

# add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routers for additional functionality
app.include_router(qa_router)
app.include_router(webhook_router)
app.include_router(qa_ui_router)

# Log startup info
logger.info(f"FastAPI app initialized. Running on port: {os.getenv('PORT', '8080')}")


def cleanup_old_jobs():
    """Clean up job directories older than KEEP_DAYS."""
    try:
        cutoff_date = datetime.now() - timedelta(days=KEEP_DAYS)
        
        # clean up job directories
        for job_dir in JOBS_DIR.iterdir():
            if job_dir.is_dir():
                try:
                    # check modification time
                    mtime = datetime.fromtimestamp(job_dir.stat().st_mtime)
                    if mtime < cutoff_date:
                        shutil.rmtree(job_dir)
                        logger.info(f"Cleaned up old job directory: {job_dir.name}")
                        
                        # remove from registry
                        job_id = job_dir.name
                        if job_id in jobs_registry:
                            del jobs_registry[job_id]
                        
                        # clean up associated files
                        (REPORTS_DIR / f"{job_id}.json").unlink(missing_ok=True)
                        (JOBS_META_DIR / f"{job_id}.json").unlink(missing_ok=True)
                except Exception as e:
                    logger.error(f"Error cleaning up {job_dir}: {e}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


async def update_job_status(job_id: str, status: str, error: Optional[str] = None):
    """Update job status in registry and persist to disk."""
    if job_id in jobs_registry:
        jobs_registry[job_id]["status"] = status
        jobs_registry[job_id]["error"] = error
        
        if status == JobStatus.RUNNING and not jobs_registry[job_id].get("started_at"):
            jobs_registry[job_id]["started_at"] = datetime.now()
        elif status in [JobStatus.DONE, JobStatus.FAILED]:
            jobs_registry[job_id]["finished_at"] = datetime.now()
        
        # persist to disk
        meta_file = JOBS_META_DIR / f"{job_id}.json"
        try:
            async with aiofiles.open(meta_file, 'w') as f:
                await f.write(json.dumps(jobs_registry[job_id], default=str))
        except Exception as e:
            logger.error(f"Failed to persist job status {job_id}: {e}")


async def run_analysis_job(job_id: str, repo_url: str, branch: str, token: Optional[str]):
    """Background task to run repository analysis."""
    try:
        await update_job_status(job_id, JobStatus.RUNNING)
        
        # use provided token or fall back to environment variable
        auth_token = token or GITHUB_TOKEN
        
        # run the enhanced analysis with visualizations
        success, result_or_error = await run_enhanced_job(job_id, repo_url, branch, auth_token)
        
        if success:
            await update_job_status(job_id, JobStatus.DONE)
            # save report
            report_path = REPORTS_DIR / f"{job_id}.json"
            async with aiofiles.open(report_path, 'w') as f:
                await f.write(json.dumps(result_or_error, indent=2))
            jobs_registry[job_id]["report_path"] = str(report_path)
        else:
            await update_job_status(job_id, JobStatus.FAILED, error=str(result_or_error))
            
    except Exception as e:
        logger.error(f"Error in analysis job {job_id}: {e}")
        await update_job_status(job_id, JobStatus.FAILED, error=str(e))


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the home page with repository submission form."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Refactr - AI-Powered Code Analysis</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 2rem;
                background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
                backdrop-filter: blur(4px);
                border: 1px solid rgba(255, 255, 255, 0.18);
            }
            h1 {
                color: #1565c0;
                margin-bottom: 0.5rem;
                font-size: 2.5rem;
                font-weight: 700;
            }
            p {
                color: #546e7a;
                margin-bottom: 2rem;
                font-size: 1.1rem;
            }
            .form-group {
                margin-bottom: 1.25rem;
            }
            label {
                display: block;
                margin-bottom: 0.5rem;
                font-weight: 600;
                color: #37474f;
                font-size: 0.95rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            input, select {
                width: 100%;
                padding: 0.875rem;
                border: 2px solid #e1f5fe;
                border-radius: 8px;
                font-size: 1rem;
                transition: all 0.3s ease;
                background-color: #fafafa;
            }
            input:focus, select:focus {
                outline: none;
                border-color: #2196f3;
                background-color: white;
                box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.1);
            }
            button {
                background: linear-gradient(45deg, #2196f3 30%, #1976d2 90%);
                color: white;
                padding: 0.875rem 2rem;
                border: none;
                border-radius: 25px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                margin-top: 1.5rem;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(33, 150, 243, 0.4);
            }
            .result {
                margin-top: 2rem;
                padding: 1.25rem;
                background-color: #e3f2fd;
                border-radius: 8px;
                display: none;
                border-left: 4px solid #2196f3;
            }
            .error {
                color: #c62828;
                margin-top: 0.5rem;
                font-weight: 500;
            }
            .success {
                color: #2e7d32;
                font-weight: 500;
            }
            .job-status {
                margin-top: 1rem;
                padding: 1.25rem;
                background: linear-gradient(135deg, #e8f5e9 0%, #e1f5fe 100%);
                border-radius: 8px;
                border: 1px solid #b3e5fc;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Refactr</h1>
            <p>AI-powered code quality analysis for modern development teams</p>
            
            <form id="analyzeForm">
                <div class="form-group">
                    <label for="repo_url">Repository URL</label>
                    <input type="url" id="repo_url" name="repo_url" 
                           placeholder="https://github.com/owner/repository" required>
                </div>
                
                <div class="form-group">
                    <label for="branch">Branch (optional)</label>
                    <input type="text" id="branch" name="branch" 
                           placeholder="main" value="main">
                </div>
                
                <div class="form-group">
                    <label for="token">Access Token (optional, for private repos)</label>
                    <input type="password" id="token" name="token" 
                           placeholder="GitHub personal access token">
                </div>
                
                <button type="submit">Analyze Repository</button>
            </form>
            
            <div id="result" class="result"></div>
        </div>
        
        <script>
            const form = document.getElementById('analyzeForm');
            const resultDiv = document.getElementById('result');
            
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = {
                    repo_url: form.repo_url.value,
                    branch: form.branch.value || 'main'
                };
                
                if (form.token.value) {
                    formData.token = form.token.value;
                }
                
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<p>⏳ Submitting analysis request...</p>';
                
                try {
                    const response = await fetch('/api/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(formData)
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        resultDiv.innerHTML = `
                            <p class="success">✅ ${data.message}</p>
                            <p><strong>Job ID:</strong> ${data.job_id}</p>
                            <div class="job-status" id="jobStatus">
                                <p>⏳ Status: ${data.status}</p>
                            </div>
                        `;
                        
                        // Start polling for job status
                        pollJobStatus(data.job_id);
                    } else {
                        resultDiv.innerHTML = `<p class="error">❌ Error: ${data.detail || 'Failed to submit analysis'}</p>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<p class="error">❌ Error: ${error.message}</p>`;
                }
            });
            
            async function pollJobStatus(jobId) {
                const statusDiv = document.getElementById('jobStatus');
                
                const poll = async () => {
                    try {
                        const response = await fetch(`/api/jobs/${jobId}`);
                        const data = await response.json();
                        
                        let statusHtml = `<p>📊 Status: <strong>${data.status}</strong></p>`;
                        
                        if (data.started_at) {
                            statusHtml += `<p>Started: ${new Date(data.started_at).toLocaleString()}</p>`;
                        }
                        
                        if (data.status === 'done') {
                            statusHtml += `<p>Finished: ${new Date(data.finished_at).toLocaleString()}</p>`;
                            statusHtml += `<p>📄 Reports: <a href="/api/report/${jobId}" target="_blank">JSON</a> | <a href="/api/report/${jobId}/html" target="_blank">HTML</a> | <a href="/api/report/${jobId}/pdf" target="_blank">PDF</a></p>`;
                            statusHtml += `<p>📊 Visualizations: <a href="/api/visualizations/${jobId}/dependency_graph" target="_blank">Dependency Graph</a> | <a href="/api/visualizations/${jobId}/dependency_network" target="_blank">Interactive Network</a> | <a href="/api/visualizations/${jobId}/quality_heatmap" target="_blank">Quality Heatmap</a></p>`;
                            statusHtml += `<p>🤖 <a href="#" onclick="startQA('${jobId}'); return false;">Start Q&A Session</a></p>`;
                            statusDiv.innerHTML = statusHtml;
                            // Also check for available visualizations
                            checkVisualizations(jobId);
                            return; // Stop polling
                        } else if (data.status === 'failed') {
                            statusHtml += `<p class="error">Error: ${data.error}</p>`;
                            statusDiv.innerHTML = statusHtml;
                            return; // Stop polling
                        }
                        
                        statusDiv.innerHTML = statusHtml;
                        
                        // Continue polling
                        setTimeout(poll, 2000);
                    } catch (error) {
                        statusDiv.innerHTML = `<p class="error">Error checking status: ${error.message}</p>`;
                    }
                };
                
                // Start polling
                poll();
            }
            
            function startQA(jobId) {
                // Open Q&A interface in new window
                window.open(`/qa/${jobId}`, '_blank');
            }
            
            async function checkVisualizations(jobId) {
                try {
                    const response = await fetch(`/api/visualizations/${jobId}`);
                    if (response.ok) {
                        const data = await response.json();
                        if (data.visualizations.length === 0) {
                            console.log('No visualizations available for this job');
                        }
                    }
                } catch (error) {
                    console.error('Error checking visualizations:', error);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_repository(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit a repository for analysis.
    
    Returns a job ID that can be used to track the analysis progress.
    """
    # generate unique job ID
    job_id = str(uuid.uuid4())
    
    # initialize job in registry
    jobs_registry[job_id] = {
        "job_id": job_id,
        "repo_url": request.repo_url,
        "branch": request.branch,
        "status": JobStatus.PENDING,
        "created_at": datetime.now(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "report_path": None
    }
    
    # add background task
    background_tasks.add_task(
        run_analysis_job,
        job_id,
        request.repo_url,
        request.branch,
        request.token
    )
    
    return AnalyzeResponse(job_id=job_id)


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of an analysis job.
    """
    if job_id not in jobs_registry:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs_registry[job_id]
    return {
        "job_id": job_id,
        "status": job_data["status"],
        "started_at": job_data.get("started_at"),
        "finished_at": job_data.get("finished_at"),
        "error": job_data.get("error"),
        "repo_url": job_data.get("repo_url")  # Include repo_url for Q&A
    }


@app.get("/api/report/{job_id}")
async def get_job_report(job_id: str):
    """
    Get the analysis report for a completed job.
    """
    if job_id not in jobs_registry:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs_registry[job_id]
    
    if job_data["status"] != JobStatus.DONE:
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not complete. Current status: {job_data['status']}"
        )
    
    report_path = Path(job_data.get("report_path", ""))
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found")
    
    try:
        async with aiofiles.open(report_path, 'r') as f:
            content = await f.read()
            report_data = json.loads(content)
        return JSONResponse(content=report_data)
    except Exception as e:
        logger.error(f"Error reading report {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Error reading report")


@app.get("/api/report/{job_id}/html")
async def get_job_report_html(job_id: str):
    """
    Get the analysis report in HTML format.
    """
    if job_id not in jobs_registry:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs_registry[job_id]
    
    if job_data["status"] != JobStatus.DONE:
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not complete. Current status: {job_data['status']}"
        )
    
    # generate HTML report
    report_path = Path(job_data.get("report_path", ""))
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found")
    
    try:
        # read JSON report
        async with aiofiles.open(report_path, 'r') as f:
            content = await f.read()
            report_data = json.loads(content)
        
        # extract issues from the report data
        issues = report_data.get('issues', [])
        
        # generate HTML report
        output_dir = tempfile.mkdtemp()
        
        # create HTML report manually since generate_report creates markdown
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Code Quality Report - {job_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                .summary {{ background: #f0f0f0; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .issue {{ background: #fff; border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 4px; }}
                .high {{ border-left: 4px solid #ff4444; }}
                .medium {{ border-left: 4px solid #ff8800; }}
                .low {{ border-left: 4px solid #ffbb33; }}
                .visualization {{ margin: 20px 0; text-align: center; }}
                .visualization img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Code Quality Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Repository:</strong> {report_data.get('repository_url', 'Unknown')}</p>
                <p><strong>Analysis Date:</strong> {report_data.get('analysis_timestamp', 'Unknown')}</p>
                <p><strong>Total Files Analyzed:</strong> {report_data.get('summary', {}).get('total_files_analyzed', 0)}</p>
                <p><strong>Total Issues:</strong> {report_data.get('summary', {}).get('total_issues', 0)}</p>
            </div>
            
            <h2>Visualizations</h2>
            <div class="visualization">
        """
        
        # add visualization links
        if report_data.get('visualizations'):
            for viz_type, viz_path in report_data['visualizations'].items():
                if viz_type == 'quality_heatmap':
                    html_content += f'<h3>Code Quality Heatmap</h3><img src="/api/visualizations/{job_id}/quality_heatmap" alt="Quality Heatmap"><br>'
        
        html_content += f"""
            </div>
            
            <h2>Issues Found ({len(issues)})</h2>
        """
        
        # add issues
        for issue in issues[:50]:  # Limit to first 50 issues
            severity = issue.get('severity', 'medium')
            html_content += f"""
            <div class="issue {severity}">
                <strong>{issue.get('file', 'Unknown')}:{issue.get('line', '?')}</strong><br>
                <strong>Type:</strong> {issue.get('issue_type', 'Unknown')}<br>
                <strong>Description:</strong> {issue.get('description', 'No description')}<br>
                <strong>Suggestion:</strong> {issue.get('suggestion', 'No suggestion')}
            </div>
            """
        
        if len(issues) > 50:
            html_content += f"<p><i>... and {len(issues) - 50} more issues</i></p>"
        
        html_content += """
        </body>
        </html>
        """
        
        # save HTML file
        html_path = Path(output_dir) / "report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return FileResponse(str(html_path), media_type="text/html")
            
    except Exception as e:
        logger.error(f"Error generating HTML report {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/report/{job_id}/pdf")
async def get_job_report_pdf(job_id: str):
    """
    Get the analysis report in PDF format.
    """
    if job_id not in jobs_registry:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs_registry[job_id]
    
    if job_data["status"] != JobStatus.DONE:
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not complete. Current status: {job_data['status']}"
        )
    
    # first generate HTML, then convert to PDF
    try:
        # read the report data
        report_path = Path(job_data.get("report_path", ""))
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report file not found")
            
        async with aiofiles.open(report_path, 'r') as f:
            content = await f.read()
            report_data = json.loads(content)
        
        # try using different PDF libraries
        output_dir = tempfile.mkdtemp()
        pdf_path = Path(output_dir) / f"report_{job_id}.pdf"
        
        # method 1: Try weasyprint (best for HTML to PDF)
        try:
            try:
                from weasyprint import HTML, CSS
            except (ImportError, OSError) as e:
                # weasyPrint requires system libraries that might not be available
                logger.debug(f"WeasyPrint import failed: {e}")
                raise ImportError("WeasyPrint not available")
            
            # create a simpler HTML for PDF (without external image references)
            html_for_pdf = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    @page {{ size: A4; margin: 2cm; }}
                    body {{ font-family: Arial, sans-serif; font-size: 12pt; line-height: 1.6; }}
                    h1 {{ color: #333; font-size: 24pt; margin-bottom: 20px; }}
                    h2 {{ color: #555; font-size: 18pt; margin-top: 30px; margin-bottom: 15px; }}
                    .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                    .issue {{ border-left: 3px solid #ddd; padding-left: 15px; margin: 15px 0; }}
                    .high {{ border-left-color: #ff4444; }}
                    .medium {{ border-left-color: #ff8800; }}
                    .low {{ border-left-color: #ffbb33; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f5f5f5; font-weight: bold; }}
                </style>
            </head>
            <body>
                <h1>Code Quality Analysis Report</h1>
                
                <div class="summary">
                    <h2>Executive Summary</h2>
                    <p><strong>Repository:</strong> {report_data.get('repository_url', 'Unknown')}</p>
                    <p><strong>Analysis Date:</strong> {report_data.get('analysis_timestamp', 'Unknown')}</p>
                    <p><strong>Branch:</strong> {report_data.get('branch', 'main')}</p>
                </div>
                
                <h2>Analysis Statistics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Files Analyzed</td>
                        <td>{report_data.get('summary', {}).get('total_files_analyzed', 0)}</td>
                    </tr>
                    <tr>
                        <td>Total Issues Found</td>
                        <td>{report_data.get('summary', {}).get('total_issues', 0)}</td>
                    </tr>
                    <tr>
                        <td>Languages</td>
                        <td>{', '.join(report_data.get('summary', {}).get('languages', []))}</td>
                    </tr>
                </table>
                
                <h2>Issue Breakdown</h2>
                <table>
                    <tr>
                        <th>Issue Type</th>
                        <th>Count</th>
                    </tr>
            """
            
            # add issue type breakdown
            for issue_type, count in report_data.get('summary', {}).get('type_breakdown', {}).items():
                html_for_pdf += f"""
                    <tr>
                        <td>{issue_type.replace('_', ' ').title()}</td>
                        <td>{count}</td>
                    </tr>
                """
            
            html_for_pdf += """
                </table>
                
                <h2>Top Issues</h2>
            """
            
            # add top 20 issues
            issues = report_data.get('issues', [])
            for i, issue in enumerate(issues[:20], 1):
                severity = issue.get('severity', 'medium')
                html_for_pdf += f"""
                <div class="issue {severity}">
                    <strong>{i}. {issue.get('file', 'Unknown')}:{issue.get('line', '?')}</strong><br>
                    <strong>Type:</strong> {issue.get('issue_type', 'Unknown').replace('_', ' ').title()}<br>
                    <strong>Description:</strong> {issue.get('description', 'No description')}<br>
                    <strong>Suggestion:</strong> {issue.get('suggestion', 'No suggestion')}
                </div>
                """
            
            if len(issues) > 20:
                html_for_pdf += f"<p><em>... and {len(issues) - 20} more issues. View the full report for complete details.</em></p>"
            
            html_for_pdf += """
            </body>
            </html>
            """
            
            # generate PDF
            HTML(string=html_for_pdf).write_pdf(pdf_path)
            logger.info(f"Generated PDF using weasyprint: {pdf_path}")
            
        except (ImportError, Exception) as e:
            logger.debug(f"WeasyPrint failed: {e}")
            # method 2: Try reportlab (more complex but doesn't need external dependencies)
            try:
                from reportlab.lib import colors
                from reportlab.lib.pagesizes import letter, A4
                from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                
                # create PDF document
                doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
                story = []
                styles = getSampleStyleSheet()
                
                # title
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    textColor=colors.HexColor('#333333')
                )
                story.append(Paragraph("Code Quality Analysis Report", title_style))
                story.append(Spacer(1, 0.3*inch))
                
                # summary
                summary_text = f"""
                <b>Repository:</b> {report_data.get('repository_url', 'Unknown')}<br/>
                <b>Analysis Date:</b> {report_data.get('analysis_timestamp', 'Unknown')}<br/>
                <b>Total Files:</b> {report_data.get('summary', {}).get('total_files_analyzed', 0)}<br/>
                <b>Total Issues:</b> {report_data.get('summary', {}).get('total_issues', 0)}
                """
                story.append(Paragraph(summary_text, styles['Normal']))
                story.append(Spacer(1, 0.2*inch))
                
                # add severity breakdown
                severity_breakdown = report_data.get('summary', {}).get('severity_breakdown', {})
                if severity_breakdown:
                    story.append(Paragraph("<b>Issues by Severity:</b>", styles['Heading2']))
                    for severity, count in severity_breakdown.items():
                        story.append(Paragraph(f"• {severity.title()}: {count}", styles['Normal']))
                    story.append(Spacer(1, 0.2*inch))
                
                # add issues details (limited to prevent huge PDFs)
                issues = report_data.get('issues', [])
                if issues:
                    story.append(Paragraph("<b>Issue Details (First 50):</b>", styles['Heading2']))
                    story.append(Spacer(1, 0.1*inch))
                    
                    # create a simple table for issues
                    issue_data = [['File', 'Line', 'Severity', 'Type', 'Message']]
                    
                    for i, issue in enumerate(issues[:50]):
                        file_path = Path(issue.get('file', 'Unknown')).name
                        line = str(issue.get('line', 'N/A'))
                        severity = issue.get('severity', 'medium')
                        issue_type = issue.get('issue_type', 'unknown')
                        message = issue.get('message', '')[:50] + '...' if len(issue.get('message', '')) > 50 else issue.get('message', '')
                        
                        issue_data.append([file_path[:30], line, severity, issue_type, message])
                    
                    # create table
                    t = Table(issue_data)
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(t)
                    
                    if len(issues) > 50:
                        story.append(Spacer(1, 0.1*inch))
                        story.append(Paragraph(f"<i>... and {len(issues) - 50} more issues</i>", styles['Normal']))
                
                # build PDF
                doc.build(story)
                logger.info(f"Generated PDF using reportlab: {pdf_path}")
                
            except ImportError:
                # method 3: Simple text-based PDF using fpdf
                try:
                    from fpdf import FPDF
                    
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=16)
                    pdf.cell(0, 10, "Code Quality Analysis Report", ln=True, align='C')
                    pdf.ln(10)
                    
                    pdf.set_font("Arial", size=12)
                    pdf.cell(0, 10, f"Repository: {report_data.get('repository_url', 'Unknown')}", ln=True)
                    pdf.cell(0, 10, f"Date: {report_data.get('analysis_timestamp', 'Unknown')}", ln=True)
                    pdf.cell(0, 10, f"Total Issues: {report_data.get('summary', {}).get('total_issues', 0)}", ln=True)
                    
                    pdf.output(str(pdf_path))
                    logger.info(f"Generated PDF using fpdf: {pdf_path}")
                    
                except ImportError:
                    raise HTTPException(
                        status_code=501,
                        detail="PDF generation requires one of: weasyprint, reportlab, or fpdf. Install with: pip install reportlab (easiest option for Windows/WSL)"
                    )
        
        return FileResponse(str(pdf_path), media_type="application/pdf", filename=f"code_quality_report_{job_id}.pdf")
            
    except Exception as e:
        logger.error(f"Error generating PDF report {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs")
async def list_jobs():
    """
    List all analysis jobs with pagination.
    """
    # simple implementation - in production, add proper pagination
    jobs = list(jobs_registry.values())
    jobs.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
    
    return {
        "total": len(jobs),
        "jobs": jobs[:50]  # Return latest 50 jobs
    }


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its associated data.
    """
    if job_id not in jobs_registry:
        raise HTTPException(status_code=404, detail="Job not found")
    
    try:
        # remove job directory
        job_dir = JOBS_DIR / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir)
        
        # remove report
        report_file = REPORTS_DIR / f"{job_id}.json"
        report_file.unlink(missing_ok=True)
        
        # remove metadata
        meta_file = JOBS_META_DIR / f"{job_id}.json"
        meta_file.unlink(missing_ok=True)
        
        # remove from registry
        del jobs_registry[job_id]
        
        return {"message": "Job deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/visualizations/{job_id}/{viz_type}")
async def get_visualization(job_id: str, viz_type: str):
    """
    Get a specific visualization for a job.
    
    viz_type can be: dependency_graph, dependency_network, quality_heatmap
    """
    if job_id not in jobs_registry:
        raise HTTPException(status_code=404, detail="Job not found")
    
    viz_path = get_visualization_path(job_id, viz_type)
    if not viz_path:
        raise HTTPException(status_code=404, detail=f"Visualization '{viz_type}' not found")
    
    # determine media type
    if viz_path.suffix == ".html":
        media_type = "text/html"
    elif viz_path.suffix == ".svg":
        media_type = "image/svg+xml"
    else:
        media_type = "image/png"
    
    return FileResponse(viz_path, media_type=media_type)


@app.get("/api/visualizations/{job_id}")
async def list_visualizations(job_id: str):
    """
    List available visualizations for a job.
    """
    if job_id not in jobs_registry:
        raise HTTPException(status_code=404, detail="Job not found")
    
    available = []
    for viz_type in ["dependency_graph", "dependency_network", "quality_heatmap"]:
        if get_visualization_path(job_id, viz_type):
            available.append({
                "type": viz_type,
                "url": f"/api/visualizations/{job_id}/{viz_type}",
                "description": {
                    "dependency_graph": "Function dependency graph (PNG)",
                    "dependency_network": "Interactive dependency network (HTML)",
                    "quality_heatmap": "Code quality heatmap (PNG)"
                }.get(viz_type, "")
            })
    
    return {"job_id": job_id, "visualizations": available}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "jobs_count": len(jobs_registry),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
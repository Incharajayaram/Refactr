import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl

from retriever.unified_retriever import index_repository, query_code
from qa_interface import CodeQAInterface

logger = logging.getLogger(__name__)

# create directories
QA_INDICES_DIR = Path("work/qa_indices")
QA_INDICES_DIR.mkdir(parents=True, exist_ok=True)

# in-memory job tracking (in production, use Redis or database)
qa_jobs = {}

router = APIRouter(prefix="/api/qa", tags=["Q&A"])


class IndexRequest(BaseModel):
    """Request model for indexing a repository."""
    repository_url: HttpUrl
    job_id: Optional[str] = None  # Use existing analysis job_id if available


class QueryRequest(BaseModel):
    """Request model for querying indexed code."""
    question: str
    job_id: str  # Which index to query
    k: int = 5  # Number of results


class QAJobStatus(BaseModel):
    """Status of a Q&A indexing job."""
    job_id: str
    status: str  # "pending", "indexing", "ready", "failed"
    created_at: datetime
    repository_url: str
    index_path: Optional[str] = None
    error: Optional[str] = None
    chunks_indexed: Optional[int] = None


class QueryResponse(BaseModel):
    """Response from a Q&A query."""
    question: str
    answer: str
    relevant_chunks: int
    response_time: float


def index_repository_background(job_id: str, repo_url: str, repo_path: str):
    """Background task to index a repository."""
    try:
        qa_jobs[job_id]["status"] = "indexing"
        index_path = QA_INDICES_DIR / job_id
        
        # index the repository
        logger.info(f"Indexing repository {repo_url} at {repo_path}")
        index_repository(repo_path, str(index_path))
        
        # count chunks (check FAISS index size)
        faiss_file = index_path / "index.faiss"
        chunks_count = 0
        if faiss_file.exists():
            import faiss
            index = faiss.read_index(str(faiss_file))
            chunks_count = index.ntotal
        
        # update job status
        qa_jobs[job_id].update({
            "status": "ready",
            "index_path": str(index_path),
            "chunks_indexed": chunks_count
        })
        logger.info(f"Successfully indexed {chunks_count} chunks for {repo_url}")
        
    except Exception as e:
        logger.error(f"Failed to index repository: {e}")
        qa_jobs[job_id].update({
            "status": "failed",
            "error": str(e)
        })


@router.post("/index", response_model=QAJobStatus)
async def index_for_qa(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Index a repository for Q&A queries.
    
    This creates embeddings for all code files in the repository
    and stores them in a FAISS index for fast retrieval.
    """
    job_id = request.job_id or str(uuid.uuid4())
    
    # check if already indexed
    if job_id in qa_jobs and qa_jobs[job_id]["status"] == "ready":
        return QAJobStatus(**qa_jobs[job_id])
    
    # the enhanced runner stores repos in work/jobs/{job_id}
    repo_path = Path("work/jobs") / job_id
    if not repo_path.exists():
        # in that case, we can't index anymore
        raise HTTPException(
            status_code=404,
            detail=f"Repository files not found. The analysis job may have cleaned up the files. Try running a new analysis with KEEP_JOB_FILES=true in your .env file."
        )
    
    # create job entry
    qa_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now(),
        "repository_url": str(request.repository_url),
        "index_path": None,
        "error": None,
        "chunks_indexed": None
    }
    
    # start background indexing
    background_tasks.add_task(
        index_repository_background,
        job_id,
        str(request.repository_url),
        str(repo_path)
    )
    
    return QAJobStatus(**qa_jobs[job_id])


@router.get("/status/{job_id}", response_model=QAJobStatus)
async def get_index_status(job_id: str):
    """Get the status of a Q&A indexing job."""
    if job_id not in qa_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return QAJobStatus(**qa_jobs[job_id])


@router.post("/query", response_model=QueryResponse)
async def query_indexed_code(request: QueryRequest):
    """
    Query the indexed codebase with a natural language question.
    
    This uses semantic search to find relevant code chunks
    and generates an answer using the configured LLM.
    """
    import time
    start_time = time.time()
    
    # check if index exists
    if request.job_id not in qa_jobs:
        raise HTTPException(status_code=404, detail="Index not found")
    
    job = qa_jobs[request.job_id]
    if job["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Index not ready. Current status: {job['status']}"
        )
    
    try:
        # query the index
        index_path = job["index_path"]
        relevant_chunks = query_code(request.question, k=request.k, index_path=index_path)
        
        # generate answer using Q&A interface
        qa_interface = CodeQAInterface()
        
        # format prompt with retrieved chunks
        prompt = qa_interface._format_prompt(request.question, relevant_chunks)
        answer = qa_interface._query_claude(prompt)
        
        if not answer:
            answer = qa_interface._placeholder_llm_response(prompt)
        
        response_time = time.time() - start_time
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            relevant_chunks=len(relevant_chunks),
            response_time=response_time
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/index/{job_id}")
async def delete_index(job_id: str):
    """Delete a Q&A index and free up space."""
    if job_id not in qa_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = qa_jobs[job_id]
    
    # delete index files
    if job.get("index_path"):
        index_path = Path(job["index_path"])
        if index_path.exists():
            import shutil
            shutil.rmtree(index_path)
    
    # remove from jobs
    del qa_jobs[job_id]
    
    return {"message": "Index deleted successfully"}


@router.get("/jobs")
async def list_qa_jobs():
    """List all Q&A indexing jobs."""
    return list(qa_jobs.values())
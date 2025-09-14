"""
Unified interface for using RAG indexer with Q&A systems.
This module provides a consistent API for both web and CLI interfaces.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from .rag_indexer import index_repo, query_index, update_index_for_repo

logger = logging.getLogger(__name__)


def index_repository(repo_path: str, index_path: str, chunk_size: int = 200) -> None:
    """
    Index a repository for Q&A using the RAG indexer.
    
    Args:
        repo_path: Path to the repository to index
        index_path: Path where the index will be stored
        chunk_size: Maximum lines per chunk
    """
    logger.info(f"Indexing repository: {repo_path} -> {index_path}")
    
    # Ensure index directory exists
    Path(index_path).mkdir(parents=True, exist_ok=True)
    
    # Use the RAG indexer
    index_repo(repo_path, index_path, chunk_size)
    

def query_code(question: str, k: int = 5, index_path: Optional[str] = None) -> List[str]:
    """
    Query the indexed codebase with a question.
    
    Args:
        question: The query string
        k: Number of results to return
        index_path: Path to the index directory (defaults to ./index)
        
    Returns:
        List of relevant code chunks as strings
    """
    # Default index path
    if index_path is None:
        index_path = "./index"
    
    # Check if index exists
    if not Path(index_path).exists():
        logger.warning(f"Index not found at {index_path}")
        return []
    
    try:
        # Query using RAG indexer
        results = query_index(index_path, question, k=k)
        
        # Extract just the text from results
        code_chunks = []
        for result in results:
            text = result.get('text', '')
            metadata = result.get('metadata', {})
            
            # Format chunk with metadata
            chunk = f"# File: {metadata.get('file_path', 'Unknown')}\n"
            chunk += f"# Lines: {metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}\n"
            chunk += f"# Type: {metadata.get('chunk_type', 'Unknown')}\n"
            if metadata.get('name'):
                chunk += f"# Name: {metadata['name']}\n"
            chunk += "\n" + text
            
            code_chunks.append(chunk)
        
        return code_chunks
        
    except Exception as e:
        logger.error(f"Error querying index: {e}")
        return []


def update_index(repo_path: str, index_path: str) -> None:
    """
    Update the index for a repository (add new/modified files).
    
    Args:
        repo_path: Path to the repository
        index_path: Path to the index directory
    """
    logger.info(f"Updating index for repository: {repo_path}")
    update_index_for_repo(repo_path, index_path)


# Backwards compatibility aliases
index_code = index_repository


def get_index_stats(index_path: str) -> Dict[str, Any]:
    """
    Get statistics about an index.
    
    Args:
        index_path: Path to the index directory
        
    Returns:
        Dictionary with index statistics
    """
    stats = {
        "exists": False,
        "chunks": 0,
        "size_mb": 0
    }
    
    index_dir = Path(index_path)
    if not index_dir.exists():
        return stats
    
    faiss_file = index_dir / "index.faiss"
    if faiss_file.exists():
        stats["exists"] = True
        
        # Get number of chunks
        try:
            import faiss
            index = faiss.read_index(str(faiss_file))
            stats["chunks"] = index.ntotal
        except Exception as e:
            logger.error(f"Error reading FAISS index: {e}")
        
        # Get total size
        total_size = 0
        for file in index_dir.iterdir():
            if file.is_file():
                total_size += file.stat().st_size
        stats["size_mb"] = round(total_size / (1024 * 1024), 2)
    
    return stats
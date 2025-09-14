import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
import logging
from .rag_indexer import RAGIndexer

logger = logging.getLogger(__name__)

# global RAG indexer instance
_rag_indexer: Optional[RAGIndexer] = None
_current_index_path: Optional[str] = None

def get_or_create_indexer(index_path: str = "./index") -> RAGIndexer:
    """Get or create a RAG indexer instance."""
    global _rag_indexer, _current_index_path
    
    if _rag_indexer is None or _current_index_path != index_path:
        # rAGIndexer expects model name, not index path as first parameter
        _rag_indexer = RAGIndexer(model_name='all-MiniLM-L6-v2')
        _current_index_path = index_path
        logger.info(f"Initialized RAG indexer for index at: {index_path}")
    
    return _rag_indexer

def index_code(chunks: List[str], index_path: str = "./index") -> None:
    """
    Index code chunks using the RAG indexer.
    This function provides compatibility with the existing embeddings.py interface.
    
    Args:
        chunks: List of code chunks to index
        index_path: Path to store the index
    """
    # for compatibility, we'll create temporary files and index them
    temp_dir = tempfile.mkdtemp()
    try:
        # write chunks to temporary files
        for i, chunk in enumerate(chunks):
            file_path = os.path.join(temp_dir, f"chunk_{i}.py")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(chunk)
        
        # use the RAG indexer to index the temporary directory
        indexer = get_or_create_indexer(index_path)
        indexer.index_repo(temp_dir, index_path)
        logger.info(f"Indexed {len(chunks)} chunks using RAG indexer")
    finally:
        # clean up temporary directory
        shutil.rmtree(temp_dir)

def query_code(question: str, k: int = 3, index_path: str = "./index") -> List[str]:
    """
    Query the code index for relevant chunks.
    This function provides compatibility with the existing embeddings.py interface.
    
    Args:
        question: The query string
        k: Number of results to return
        index_path: Path to the index
        
    Returns:
        List of relevant code chunks
    """
    indexer = get_or_create_indexer(index_path)
    
    try:
        # use the RAG indexer's query method
        results = indexer.query_index(index_path, question, k=k)
        
        # extract just the code content from results
        code_chunks = []
        for result in results:
            # the result contains 'text' field with the code content
            chunk_with_context = f"# File: {result.get('file_path', 'Unknown')}\n# Lines: {result.get('start_line', '?')}-{result.get('end_line', '?')}\n{result.get('text', '')}"
            code_chunks.append(chunk_with_context)
        
        return code_chunks
    except Exception as e:
        logger.warning(f"Error querying index: {e}")
        return []

def index_repository(repo_path: str, index_path: str = "./index") -> None:
    """
    Index an entire repository using the RAG indexer.
    
    Args:
        repo_path: Path to the repository
        index_path: Path to store the index
    """
    indexer = get_or_create_indexer(index_path)
    
    # use the RAG indexer's native repository indexing
    indexer.index_repo(repo_path, index_path)
    logger.info(f"Indexed repository: {repo_path}")

# for backward compatibility
class CodeEmbeddingSystem:
    """Compatibility wrapper for the old embeddings.py interface."""
    
    def __init__(self, index_path: str = "./index"):
        self.index_path = index_path
        self.indexer = get_or_create_indexer(index_path)
    
    def index_code(self, chunks: List[str]) -> None:
        index_code(chunks, self.index_path)
    
    def query_code(self, question: str, k: int = 3) -> List[str]:
        return query_code(question, k, self.index_path)
    
    def save_index(self, filepath: str = None) -> None:
        # rAG indexer automatically saves
        logger.info("Index is automatically saved by RAG indexer")
    
    def load_index(self, filepath: str = None) -> None:
        # rAG indexer automatically loads
        self.indexer.load_index()
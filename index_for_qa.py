import os
from pathlib import Path
from retriever.embeddings import CodeEmbeddingSystem
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def index_codebase(directory: str = "."):
    """Index all Python files in the directory for Q&A."""
    # initialize the embedding system
    embedding_system = CodeEmbeddingSystem()
    
    # collect all Python files
    path = Path(directory)
    python_files = list(path.rglob("*.py"))
    
    # filter out virtual environments and cache directories
    python_files = [
        f for f in python_files 
        if not any(part in str(f) for part in ['.venv', 'venv', '__pycache__', '.git'])
    ]
    
    logger.info(f"Found {len(python_files)} Python files to index")
    
    # read and chunk all files
    all_chunks = []
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # in production, you'd want more sophisticated chunking
            lines = content.split('\n')
            chunk = []
            current_chunk = []
            
            for line in lines:
                if (line.startswith('def ') or line.startswith('class ')) and current_chunk:
                    # save previous chunk
                    chunk_text = '\n'.join(current_chunk)
                    if len(chunk_text.strip()) > 50:  # Minimum chunk size
                        all_chunks.append(f"# File: {py_file}\n{chunk_text}")
                    current_chunk = [line]
                else:
                    current_chunk.append(line)
            
            # don't forget the last chunk
            if current_chunk:
                chunk_text = '\n'.join(current_chunk)
                if len(chunk_text.strip()) > 50:
                    all_chunks.append(f"# File: {py_file}\n{chunk_text}")
                    
        except Exception as e:
            logger.error(f"Error reading {py_file}: {e}")
    
    logger.info(f"Created {len(all_chunks)} code chunks")
    
    # index all chunks
    if all_chunks:
        embedding_system.index_code(all_chunks)
        
        # save the index
        embedding_system.save_index("code_index.pkl")
        logger.info("Index saved to code_index.pkl")
        logger.info("You can now use the Q&A interface!")
    else:
        logger.error("No code chunks to index!")

if __name__ == "__main__":
    index_codebase(".")
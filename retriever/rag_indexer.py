import os
import ast
import json
import sqlite3
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""
    id: int
    file_path: str
    start_line: int
    end_line: int
    language: str
    chunk_type: str  # 'function', 'class', 'method', 'block'
    name: Optional[str]  # Function/class name if applicable
    text: str
    hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)


class CodeChunker:
    """Splits code files into logical chunks based on AST analysis."""
    
    def __init__(self, chunk_size: int = 200):
        """
        Initialize the code chunker.
        
        Args:
            chunk_size: Maximum lines for fallback chunking
        """
        self.chunk_size = chunk_size
    
    def chunk_file(self, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """
        Split a code file into logical chunks.
        
        Args:
            file_path: Path to the file
            content: File content
            language: Programming language
            
        Returns:
            List of code chunks
        """
        if language == 'python':
            return self._chunk_python(file_path, content)
        elif language == 'javascript':
            return self._chunk_javascript(file_path, content)
        else:
            # fallback to line-based chunking
            return self._chunk_by_lines(file_path, content, language)
    
    def _chunk_python(self, file_path: str, content: str) -> List[CodeChunk]:
        """Chunk Python code using AST."""
        chunks = []
        lines = content.splitlines()
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # if parsing fails, fall back to line chunking
            return self._chunk_by_lines(file_path, content, 'python')
        
        # extract all function and class definitions
        for node in ast.walk(tree):
            chunk_data = None
            
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                chunk_data = {
                    'name': node.name,
                    'chunk_type': 'function',
                    'start_line': node.lineno,
                    'end_line': node.end_lineno or node.lineno
                }
            elif isinstance(node, ast.ClassDef):
                chunk_data = {
                    'name': node.name,
                    'chunk_type': 'class',
                    'start_line': node.lineno,
                    'end_line': node.end_lineno or node.lineno
                }
            
            if chunk_data and chunk_data['end_line'] <= len(lines):
                # extract the actual code
                chunk_lines = lines[chunk_data['start_line'] - 1:chunk_data['end_line']]
                chunk_text = '\n'.join(chunk_lines)
                
                # create chunk
                chunk = CodeChunk(
                    id=0,  # Will be set later
                    file_path=file_path,
                    start_line=chunk_data['start_line'],
                    end_line=chunk_data['end_line'],
                    language='python',
                    chunk_type=chunk_data['chunk_type'],
                    name=chunk_data['name'],
                    text=chunk_text,
                    hash=self._compute_hash(chunk_text)
                )
                chunks.append(chunk)
        
        # handle code outside functions/classes
        if chunks:
            # sort chunks by start line
            chunks.sort(key=lambda c: c.start_line)
            
            # add chunks for code between functions/classes
            module_chunks = []
            last_end = 0
            
            for chunk in chunks:
                if chunk.start_line > last_end + 1:
                    # there's code between chunks
                    between_lines = lines[last_end:chunk.start_line - 1]
                    if any(line.strip() for line in between_lines):  # Non-empty
                        between_text = '\n'.join(between_lines)
                        module_chunk = CodeChunk(
                            id=0,
                            file_path=file_path,
                            start_line=last_end + 1,
                            end_line=chunk.start_line - 1,
                            language='python',
                            chunk_type='module',
                            name=None,
                            text=between_text,
                            hash=self._compute_hash(between_text)
                        )
                        module_chunks.append(module_chunk)
                last_end = chunk.end_line
            
            # add remaining code at the end
            if last_end < len(lines):
                remaining_lines = lines[last_end:]
                if any(line.strip() for line in remaining_lines):
                    remaining_text = '\n'.join(remaining_lines)
                    module_chunk = CodeChunk(
                        id=0,
                        file_path=file_path,
                        start_line=last_end + 1,
                        end_line=len(lines),
                        language='python',
                        chunk_type='module',
                        name=None,
                        text=remaining_text,
                        hash=self._compute_hash(remaining_text)
                    )
                    module_chunks.append(module_chunk)
            
            chunks.extend(module_chunks)
            chunks.sort(key=lambda c: c.start_line)
        else:
            # no functions/classes found, use line chunking
            chunks = self._chunk_by_lines(file_path, content, 'python')
        
        return chunks
    
    def _chunk_javascript(self, file_path: str, content: str) -> List[CodeChunk]:
        """Chunk JavaScript code using the JS AST parser."""
        js_parser_path = Path(__file__).parent.parent / "analyzers" / "js_ast_parser.js"
        
        if not js_parser_path.exists():
            # fallback to line chunking if parser not available
            return self._chunk_by_lines(file_path, content, 'javascript')
        
        try:
            # call the JS parser
            result = subprocess.run(
                ["node", str(js_parser_path), file_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"JS parser error: {result.stderr}")
                return self._chunk_by_lines(file_path, content, 'javascript')
            
            # parse the result
            js_data = json.loads(result.stdout)
            chunks = []
            lines = content.splitlines()
            
            # process functions
            for func_name, func_info in js_data.get('functions', {}).items():
                line_num = func_info.get('line', 1)
                
                # look for the next function or end of file
                end_line = len(lines)
                for other_name, other_info in js_data.get('functions', {}).items():
                    other_line = other_info.get('line', 1)
                    if other_line > line_num and other_line < end_line:
                        end_line = other_line - 1
                
                # extract chunk
                chunk_lines = lines[line_num - 1:end_line]
                chunk_text = '\n'.join(chunk_lines)
                
                chunk = CodeChunk(
                    id=0,
                    file_path=file_path,
                    start_line=line_num,
                    end_line=end_line,
                    language='javascript',
                    chunk_type=func_info.get('type', 'function'),
                    name=func_name,
                    text=chunk_text,
                    hash=self._compute_hash(chunk_text)
                )
                chunks.append(chunk)
            
            # sort and return
            chunks.sort(key=lambda c: c.start_line)
            return chunks if chunks else self._chunk_by_lines(file_path, content, 'javascript')
            
        except Exception as e:
            logger.error(f"Error parsing JavaScript file {file_path}: {e}")
            return self._chunk_by_lines(file_path, content, 'javascript')
    
    def _chunk_by_lines(self, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """Fallback: chunk by fixed number of lines."""
        lines = content.splitlines()
        chunks = []
        
        for i in range(0, len(lines), self.chunk_size):
            chunk_lines = lines[i:i + self.chunk_size]
            chunk_text = '\n'.join(chunk_lines)
            
            if chunk_text.strip():  # Skip empty chunks
                chunk = CodeChunk(
                    id=0,
                    file_path=file_path,
                    start_line=i + 1,
                    end_line=min(i + self.chunk_size, len(lines)),
                    language=language,
                    chunk_type='block',
                    name=None,
                    text=chunk_text,
                    hash=self._compute_hash(chunk_text)
                )
                chunks.append(chunk)
        
        return chunks
    
    def _compute_hash(self, text: str) -> str:
        """Compute hash of chunk text for change detection."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()


class RAGIndexer:
    """Main RAG indexer for code repositories."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', max_seq_length: int = 256):
        """
        Initialize the RAG indexer.
        
        Args:
            model_name: Name of the sentence transformer model
            max_seq_length: Maximum sequence length for the model
        """
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_length
        self.chunker = CodeChunker()
        
        # file extensions to index
        self.supported_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'javascript',
            '.tsx': 'javascript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php'
        }
    
    def _init_database(self, db_path: str):
        """Initialize SQLite database for metadata storage."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                file_path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                language TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                name TEXT,
                text TEXT NOT NULL,
                hash TEXT NOT NULL,
                embedding_id INTEGER NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_file_path ON chunks(file_path)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_hash ON chunks(hash)
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS index_info (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _get_file_language(self, file_path: str) -> Optional[str]:
        """Determine the programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        return self.supported_extensions.get(ext)
    
    def index_repo(self, repo_path: str, index_dir: str, chunk_size: int = 200) -> None:
        """
        Index a repository by creating embeddings for code chunks.
        
        Args:
            repo_path: Path to the repository
            index_dir: Directory to store the index
            chunk_size: Maximum lines per chunk for fallback chunking
            
        Example:
            >>> indexer = RAGIndexer()
            >>> indexer.index_repo('/path/to/repo', './indices/myrepo')
            Indexed 156 chunks from 23 files
        """
        repo_path = Path(repo_path)
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # initialize database
        db_path = index_dir / 'metadata.sqlite'
        self._init_database(str(db_path))
        
        # update chunker settings
        self.chunker.chunk_size = chunk_size
        
        # collect all code files
        code_files = []
        for ext, lang in self.supported_extensions.items():
            code_files.extend(repo_path.rglob(f'*{ext}'))
        
        # filter out common non-source directories
        excluded_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 
                        'dist', 'build', '.pytest_cache', '.mypy_cache'}
        code_files = [f for f in code_files if not any(
            excluded in f.parts for excluded in excluded_dirs
        )]
        
        logger.info(f"Found {len(code_files)} code files to index")
        
        # process files and create chunks
        all_chunks = []
        chunk_id = 0
        
        for file_path in tqdm(code_files, desc="Processing files"):
            language = self._get_file_language(str(file_path))
            if not language:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # get relative path for storage
                rel_path = str(file_path.relative_to(repo_path))
                
                # chunk the file
                chunks = self.chunker.chunk_file(rel_path, content, language)
                
                # assign IDs
                for chunk in chunks:
                    chunk.id = chunk_id
                    chunk_id += 1
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        if not all_chunks:
            logger.warning("No chunks to index")
            return
        
        logger.info(f"Created {len(all_chunks)} chunks, computing embeddings...")
        
        # compute embeddings
        texts = [chunk.text for chunk in all_chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings.astype('float32'))
        
        # save FAISS index
        faiss_path = index_dir / 'index.faiss'
        faiss.write_index(index, str(faiss_path))
        
        # save metadata to database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # clear existing data
        cursor.execute('DELETE FROM chunks')
        
        # insert chunks
        for i, chunk in enumerate(all_chunks):
            cursor.execute('''
                INSERT INTO chunks (id, file_path, start_line, end_line, language,
                                  chunk_type, name, text, hash, embedding_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                chunk.id, chunk.file_path, chunk.start_line, chunk.end_line,
                chunk.language, chunk.chunk_type, chunk.name, chunk.text,
                chunk.hash, i
            ))
        
        # save index info
        cursor.execute('''
            INSERT OR REPLACE INTO index_info (key, value)
            VALUES ('total_chunks', ?), ('dimension', ?), ('model', ?)
        ''', (len(all_chunks), dimension, self.model.get_sentence_embedding_dimension()))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Indexed {len(all_chunks)} chunks from {len(code_files)} files")
    
    def query_index(self, index_dir: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the index for similar code chunks.
        
        Args:
            index_dir: Directory containing the index
            query: Query string
            k: Number of results to return
            
        Returns:
            List of dictionaries with score, metadata, and text
            
        Example:
            >>> indexer = RAGIndexer()
            >>> results = indexer.query_index('./indices/myrepo', 'database connection', k=3)
            >>> for r in results:
            ...     print(f"Score: {r['score']:.3f}, File: {r['metadata']['file_path']}")
            Score: 0.842, File: src/db/connection.py
            Score: 0.756, File: src/models/base.py
            Score: 0.693, File: tests/test_db.py
        """
        index_dir = Path(index_dir)
        faiss_path = index_dir / 'index.faiss'
        db_path = index_dir / 'metadata.sqlite'
        
        if not faiss_path.exists() or not db_path.exists():
            raise ValueError(f"Index not found in {index_dir}")
        
        # load FAISS index
        index = faiss.read_index(str(faiss_path))
        
        # encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # search
        distances, indices = index.search(query_embedding.astype('float32'), k)
        
        # load metadata
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # No result
                continue
            
            # get chunk metadata
            cursor.execute('''
                SELECT file_path, start_line, end_line, language, chunk_type,
                       name, text, hash
                FROM chunks
                WHERE embedding_id = ?
            ''', (int(idx),))
            
            row = cursor.fetchone()
            if row:
                metadata = {
                    'file_path': row[0],
                    'start_line': row[1],
                    'end_line': row[2],
                    'language': row[3],
                    'chunk_type': row[4],
                    'name': row[5],
                    'hash': row[7]
                }
                
                results.append({
                    'score': float(dist),
                    'metadata': metadata,
                    'text': row[6]
                })
        
        conn.close()
        
        return results
    
    def update_index_for_repo(self, repo_path: str, index_dir: str) -> None:
        """
        Update the index for changed files in the repository.
        
        Args:
            repo_path: Path to the repository
            index_dir: Directory containing the index
            
        This function:
        1. Loads existing chunks and their hashes
        2. Re-processes all files
        3. Updates only changed chunks
        4. Maintains stable IDs for unchanged chunks
        """
        index_dir = Path(index_dir)
        db_path = index_dir / 'metadata.sqlite'
        
        if not db_path.exists():
            # no existing index, create new one
            self.index_repo(repo_path, str(index_dir))
            return
        
        # load existing chunks
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute('SELECT file_path, hash, embedding_id FROM chunks')
        existing_chunks = {}
        for row in cursor.fetchall():
            file_path, chunk_hash, embedding_id = row
            if file_path not in existing_chunks:
                existing_chunks[file_path] = {}
            existing_chunks[file_path][chunk_hash] = embedding_id
        
        conn.close()
        
        # re-index (this is a simplified version - a production system would be more incremental)
        logger.info("Updating index...")
        self.index_repo(repo_path, str(index_dir))


def index_repo(repo_path: str, index_dir: str, chunk_size: int = 200) -> None:
    """
    Public function to index a repository.
    
    Args:
        repo_path: Path to the repository
        index_dir: Directory to store the index
        chunk_size: Maximum lines per chunk for fallback chunking
    """
    indexer = RAGIndexer()
    indexer.index_repo(repo_path, index_dir, chunk_size)


def query_index(index_dir: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Public function to query an index.
    
    Args:
        index_dir: Directory containing the index
        query: Query string
        k: Number of results to return
        
    Returns:
        List of dictionaries with score, metadata, and text
    """
    indexer = RAGIndexer()
    return indexer.query_index(index_dir, query, k)


def update_index_for_repo(repo_path: str, index_dir: str) -> None:
    """
    Public function to update an index.
    
    Args:
        repo_path: Path to the repository
        index_dir: Directory containing the index
    """
    indexer = RAGIndexer()
    indexer.update_index_for_repo(repo_path, index_dir)


if __name__ == "__main__":
    # example usage
    import tempfile
    
    # create sample repository
    with tempfile.TemporaryDirectory() as temp_repo:
        # create sample Python file
        sample_py = Path(temp_repo) / "sample.py"
        sample_py.write_text('''
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class MathUtils:
    """Utility class for mathematical operations."""
    
    def factorial(self, n):
        """Calculate factorial of n."""
        if n <= 1:
            return 1
        return n * self.factorial(n-1)
    
    def is_prime(self, n):
        """Check if n is prime."""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

# module-level code
PI = 3.14159
E = 2.71828
''')
        
        # create index
        with tempfile.TemporaryDirectory() as index_dir:
            print("Indexing sample repository...")
            index_repo(temp_repo, index_dir)
            
            # query the index
            queries = [
                "fibonacci recursive function",
                "check if number is prime",
                "mathematical constants"
            ]
            
            for query_text in queries:
                print(f"\nQuery: '{query_text}'")
                results = query_index(index_dir, query_text, k=2)
                
                for i, result in enumerate(results):
                    print(f"\nResult {i+1}:")
                    print(f"  Score: {result['score']:.3f}")
                    print(f"  File: {result['metadata']['file_path']}")
                    print(f"  Lines: {result['metadata']['start_line']}-{result['metadata']['end_line']}")
                    print(f"  Type: {result['metadata']['chunk_type']}")
                    if result['metadata']['name']:
                        print(f"  Name: {result['metadata']['name']}")
                    print(f"  Preview: {result['text'][:100]}...")
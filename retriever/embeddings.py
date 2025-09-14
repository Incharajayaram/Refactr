import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeEmbeddingSystem:
    """Manages code embeddings and similarity search for Q&A."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        """
        Initialize the embedding system.
        
        Args:
            model_name: Name of the sentence-transformer model
            cache_dir: Directory to cache the model (defaults to D:\\StoryWeaver\\model_cache if exists)
        """
        # check if custom cache directory exists
        custom_cache = "D:\\StoryWeaver\\model_cache"
        if cache_dir is None and os.path.exists(custom_cache):
            cache_dir = custom_cache
            logger.info(f"Using existing model cache at: {cache_dir}")
        
        # initialize sentence transformer with cache directory
        try:
            self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            # try to use a smaller model that fits in memory
            alternative_model = "all-MiniLM-L12-v2"
            logger.info(f"Attempting to load alternative model: {alternative_model}")
            self.model = SentenceTransformer(alternative_model, cache_folder=cache_dir)
        
        # initialize FAISS index and storage
        self.index = None
        self.code_chunks = []
        self.embeddings_cache = []
        
        # set batch size based on available memory
        self.batch_size = 16  # Conservative batch size for 8GB RAM
        
    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for text chunks with memory-efficient batching.
        
        Args:
            texts: List of text chunks to embed
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = []
        
        # process in batches to avoid memory issues
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                # move model to GPU if available, but with memory constraints
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                    batch_size=self.batch_size,
                    normalize_embeddings=True  # Normalize for cosine similarity
                )
                embeddings.append(batch_embeddings)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("GPU out of memory, falling back to CPU")
                    # fall back to CPU processing
                    self.model.to('cpu')
                    batch_embeddings = self.model.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=True,
                        batch_size=self.batch_size // 2,  # Smaller batch for CPU
                        normalize_embeddings=True
                    )
                    embeddings.append(batch_embeddings)
                else:
                    raise e
                    
        return np.vstack(embeddings)
    
    def index_code(self, chunks: List[str]) -> None:
        """
        Index code chunks for similarity search.
        
        Args:
            chunks: List of code chunks to index
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return
            
        logger.info(f"Indexing {len(chunks)} code chunks...")
        
        # store the chunks
        self.code_chunks = chunks
        
        # create embeddings
        embeddings = self._create_embeddings(chunks)
        self.embeddings_cache = embeddings
        
        # create FAISS index
        dimension = embeddings.shape[1]
        
        # use IndexFlatIP for inner product (equivalent to cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        
        # add vectors to index
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"Successfully indexed {len(chunks)} chunks")
        
    def query_code(self, question: str, k: int = 3) -> List[str]:
        """
        Query the indexed code chunks for relevant results.
        
        Args:
            question: The question to search for
            k: Number of top results to return
            
        Returns:
            List of most relevant code chunks
        """
        if self.index is None or not self.code_chunks:
            logger.warning("No code has been indexed yet")
            return []
            
        # ensure k doesn't exceed available chunks
        k = min(k, len(self.code_chunks))
        
        # encode the question
        question_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # search for similar chunks
        distances, indices = self.index.search(
            question_embedding.astype('float32'), 
            k
        )
        
        # get the relevant chunks
        relevant_chunks = [self.code_chunks[idx] for idx in indices[0]]
        
        return relevant_chunks
    
    def save_index(self, filepath: str = "code_index.pkl") -> None:
        """Save the index and chunks to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'index': faiss.serialize_index(self.index),
                'chunks': self.code_chunks,
                'embeddings': self.embeddings_cache
            }, f)
        logger.info(f"Index saved to {filepath}")
        
    def load_index(self, filepath: str = "code_index.pkl") -> None:
        """Load the index and chunks from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.index = faiss.deserialize_index(data['index'])
            self.code_chunks = data['chunks']
            self.embeddings_cache = data['embeddings']
        logger.info(f"Index loaded from {filepath}")


# global instance for convenience
_embedding_system = None


def index_code(chunks: List[str]) -> None:
    """
    Index code chunks for similarity search.
    
    Args:
        chunks: List of code chunks to index
    """
    global _embedding_system
    if _embedding_system is None:
        _embedding_system = CodeEmbeddingSystem()
    _embedding_system.index_code(chunks)


def query_code(question: str, k: int = 3) -> List[str]:
    """
    Query the indexed code chunks for relevant results.
    
    Args:
        question: The question to search for
        k: Number of top results to return
        
    Returns:
        List of most relevant code chunks
    """
    global _embedding_system
    if _embedding_system is None:
        _embedding_system = CodeEmbeddingSystem()
    return _embedding_system.query_code(question, k)


if __name__ == "__main__":
    # test the embedding system
    test_chunks = [
        """def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)""",
        
        """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr""",
        
        """class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        
    def insert(self, value):
        if value < self.value:
            if self.left is None:
                self.left = BinaryTree(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = BinaryTree(value)
            else:
                self.right.insert(value)""",
                
        """async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()""",
            
        """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)"""
    ]
    
    print("Testing embeddings system...")
    
    # index the test chunks
    index_code(test_chunks)
    
    # test queries
    test_queries = [
        "How to implement fibonacci?",
        "sorting algorithm implementation",
        "binary tree data structure",
        "async web request"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = query_code(query, k=2)
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(result)
            print("-" * 50)
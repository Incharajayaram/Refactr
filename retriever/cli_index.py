import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

from retriever.rag_indexer import index_repo, query_index, update_index_for_repo

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_query_results(results: List[Dict[str, Any]], verbose: bool = False) -> str:
    """
    Format query results for display.
    
    Args:
        results: List of search results
        verbose: Whether to show full text
        
    Returns:
        Formatted string
    """
    if not results:
        return "No results found."
    
    output = []
    output.append(f"\nFound {len(results)} relevant code chunks:\n")
    
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        score = result['score']
        text = result['text']
        
        output.append(f"{'='*60}")
        output.append(f"Result #{i} (Score: {score:.3f})")
        output.append(f"{'='*60}")
        output.append(f"File: {metadata['file_path']}")
        output.append(f"Lines: {metadata['start_line']}-{metadata['end_line']}")
        output.append(f"Language: {metadata['language']}")
        output.append(f"Type: {metadata['chunk_type']}")
        
        if metadata.get('name'):
            output.append(f"Name: {metadata['name']}")
        
        output.append("")
        
        if verbose:
            output.append("Full Code:")
            output.append("-" * 40)
            output.append(text)
        else:
            # show preview (first 10 lines)
            lines = text.split('\n')[:10]
            output.append("Code Preview:")
            output.append("-" * 40)
            output.append('\n'.join(lines))
            if len(text.split('\n')) > 10:
                output.append("... (truncated)")
        
        output.append("")
    
    return '\n'.join(output)


def cmd_index(args):
    """Handle the 'index' command."""
    repo_path = Path(args.repo_path).resolve()
    index_dir = Path(args.index_dir).resolve()
    
    if not repo_path.exists():
        logger.error(f"Repository path does not exist: {repo_path}")
        return 1
    
    if not repo_path.is_dir():
        logger.error(f"Repository path is not a directory: {repo_path}")
        return 1
    
    logger.info(f"Indexing repository: {repo_path}")
    logger.info(f"Index directory: {index_dir}")
    logger.info(f"Chunk size: {args.chunk_size} lines")
    
    try:
        index_repo(str(repo_path), str(index_dir), args.chunk_size)
        logger.info("Indexing completed successfully!")
        
        # show index statistics
        metadata_path = index_dir / 'metadata.sqlite'
        if metadata_path.exists():
            import sqlite3
            conn = sqlite3.connect(str(metadata_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM chunks')
            total_chunks = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT file_path) FROM chunks')
            total_files = cursor.fetchone()[0]
            
            cursor.execute('SELECT language, COUNT(*) FROM chunks GROUP BY language')
            language_stats = cursor.fetchall()
            
            conn.close()
            
            print(f"\nIndex Statistics:")
            print(f"  Total chunks: {total_chunks}")
            print(f"  Total files: {total_files}")
            print(f"  Languages:")
            for lang, count in language_stats:
                print(f"    - {lang}: {count} chunks")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return 1


def cmd_query(args):
    """Handle the 'query' command."""
    index_dir = Path(args.index_dir).resolve()
    
    if not index_dir.exists():
        logger.error(f"Index directory does not exist: {index_dir}")
        return 1
    
    if not (index_dir / 'index.faiss').exists():
        logger.error(f"No FAISS index found in: {index_dir}")
        logger.error("Please run 'index' command first.")
        return 1
    
    logger.info(f"Querying index: {index_dir}")
    logger.info(f"Query: '{args.query}'")
    logger.info(f"Number of results: {args.k}")
    
    try:
        results = query_index(str(index_dir), args.query, args.k)
        
        if args.json:
            # output as JSON
            print(json.dumps(results, indent=2))
        else:
            # format for human reading
            formatted = format_query_results(results, args.verbose)
            print(formatted)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during query: {e}")
        return 1


def cmd_update(args):
    """Handle the 'update' command."""
    repo_path = Path(args.repo_path).resolve()
    index_dir = Path(args.index_dir).resolve()
    
    if not repo_path.exists():
        logger.error(f"Repository path does not exist: {repo_path}")
        return 1
    
    if not index_dir.exists():
        logger.error(f"Index directory does not exist: {index_dir}")
        return 1
    
    logger.info(f"Updating index for repository: {repo_path}")
    logger.info(f"Index directory: {index_dir}")
    
    try:
        update_index_for_repo(str(repo_path), str(index_dir))
        logger.info("Index updated successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during update: {e}")
        return 1


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='RAG-based code repository indexing and search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # index a repository
  python -m retriever.cli_index index /path/to/repo ./indices/myrepo
  
  # query the index
  python -m retriever.cli_index query ./indices/myrepo "database connection" --k 5
  
  # update existing index
  python -m retriever.cli_index update /path/to/repo ./indices/myrepo
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # index command
    index_parser = subparsers.add_parser(
        'index',
        help='Index a code repository'
    )
    index_parser.add_argument(
        'repo_path',
        help='Path to the repository to index'
    )
    index_parser.add_argument(
        'index_dir',
        help='Directory to store the index'
    )
    index_parser.add_argument(
        '--chunk-size',
        type=int,
        default=200,
        help='Maximum lines per chunk for fallback chunking (default: 200)'
    )
    
    # query command
    query_parser = subparsers.add_parser(
        'query',
        help='Query an indexed repository'
    )
    query_parser.add_argument(
        'index_dir',
        help='Directory containing the index'
    )
    query_parser.add_argument(
        'query',
        help='Query string to search for'
    )
    query_parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of results to return (default: 5)'
    )
    query_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show full code chunks instead of preview'
    )
    query_parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    # update command
    update_parser = subparsers.add_parser(
        'update',
        help='Update an existing index'
    )
    update_parser.add_argument(
        'repo_path',
        help='Path to the repository'
    )
    update_parser.add_argument(
        'index_dir',
        help='Directory containing the index to update'
    )
    
    # parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # execute command
    if args.command == 'index':
        return cmd_index(args)
    elif args.command == 'query':
        return cmd_query(args)
    elif args.command == 'update':
        return cmd_update(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
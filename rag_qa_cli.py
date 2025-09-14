#!/usr/bin/env python3
"""
CLI interface for RAG-based Q&A system.
Uses the RAG indexer to provide intelligent code Q&A.
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever.use_rag_index import index_repository, query_code, get_index_stats
from qa_interface import CodeQAInterface


def main():
    parser = argparse.ArgumentParser(
        description="RAG-based Code Q&A System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index current directory
  python rag_qa_cli.py index

  # Index a specific repository
  python rag_qa_cli.py index --repo /path/to/repo

  # Ask a question (uses default index)
  python rag_qa_cli.py query "How does authentication work?"

  # Ask with more context chunks
  python rag_qa_cli.py query "What are the main API endpoints?" -k 10

  # Start interactive Q&A session
  python rag_qa_cli.py interactive

  # Check index status
  python rag_qa_cli.py status
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index a repository')
    index_parser.add_argument('--repo', default='.', help='Repository path (default: current directory)')
    index_parser.add_argument('--index-dir', default='./index', help='Index directory (default: ./index)')
    index_parser.add_argument('--chunk-size', type=int, default=200, help='Max lines per chunk (default: 200)')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the indexed codebase')
    query_parser.add_argument('question', help='Question to ask about the code')
    query_parser.add_argument('--index-dir', default='./index', help='Index directory (default: ./index)')
    query_parser.add_argument('-k', type=int, default=5, help='Number of results (default: 5)')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive Q&A session')
    interactive_parser.add_argument('--index-dir', default='./index', help='Index directory (default: ./index)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check index status')
    status_parser.add_argument('--index-dir', default='./index', help='Index directory (default: ./index)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'index':
        print(f"[INFO] Indexing repository: {args.repo}")
        print(f"[INFO] Index will be saved to: {args.index_dir}")
        print("-" * 60)
        
        try:
            index_repository(args.repo, args.index_dir, args.chunk_size)
            stats = get_index_stats(args.index_dir)
            print("\n[SUCCESS] Indexing complete!")
            print(f"[INFO] Indexed {stats['chunks']} chunks")
            print(f"[INFO] Index size: {stats['size_mb']} MB")
        except Exception as e:
            print(f"\n[ERROR] Indexing failed: {e}")
            sys.exit(1)
    
    elif args.command == 'query':
        # Check if index exists
        if not Path(args.index_dir).exists():
            print(f"[ERROR] Index not found at {args.index_dir}")
            print("[INFO] Run 'python rag_qa_cli.py index' first")
            sys.exit(1)
        
        print(f"[QUERY] {args.question}")
        print("-" * 60)
        
        # Create Q&A interface with custom index
        qa = CodeQAInterface()
        
        # Override query_code to use specified index
        import qa_interface
        original_query = qa_interface.query_code
        qa_interface.query_code = lambda q, k=5: query_code(q, k=k, index_path=args.index_dir)
        
        try:
            answer = qa.answer_question(args.question)
            print("\n[ANSWER]")
            print(answer)
        except Exception as e:
            print(f"\n[ERROR] Query failed: {e}")
            sys.exit(1)
        finally:
            # Restore original
            qa_interface.query_code = original_query
    
    elif args.command == 'interactive':
        # Check if index exists
        if not Path(args.index_dir).exists():
            print(f"[ERROR] Index not found at {args.index_dir}")
            print("[INFO] Run 'python rag_qa_cli.py index' first")
            sys.exit(1)
        
        print("[INFO] Starting interactive Q&A session")
        print(f"[INFO] Using index at: {args.index_dir}")
        stats = get_index_stats(args.index_dir)
        print(f"[INFO] Index has {stats['chunks']} chunks")
        print("-" * 60)
        
        # Create Q&A interface with custom index
        qa = CodeQAInterface()
        
        # Override query_code to use specified index
        import qa_interface
        original_query = qa_interface.query_code
        qa_interface.query_code = lambda q, k=5: query_code(q, k=k, index_path=args.index_dir)
        
        try:
            qa.run_interactive_session()
        finally:
            # Restore original
            qa_interface.query_code = original_query
    
    elif args.command == 'status':
        stats = get_index_stats(args.index_dir)
        
        print(f"[INDEX STATUS] {args.index_dir}")
        print("-" * 60)
        
        if stats['exists']:
            print(f"[OK] Index exists")
            print(f"[INFO] Chunks: {stats['chunks']}")
            print(f"[INFO] Size: {stats['size_mb']} MB")
        else:
            print(f"[ERROR] No index found at {args.index_dir}")
            print("[INFO] Run 'python rag_qa_cli.py index' to create one")


if __name__ == "__main__":
    main()
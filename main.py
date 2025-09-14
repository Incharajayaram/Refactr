import argparse
import os
import sys
from pathlib import Path
from typing import Set, List


def detect_languages(path: Path) -> Set[str]:
    """
    Detect programming languages present in the given directory or file.
    
    Args:
        path: Path to analyze (can be file or directory)
        
    Returns:
        Set of detected programming languages
    """
    language_extensions = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.mjs': 'JavaScript',
        '.jsx': 'JavaScript',
        '.ts': 'TypeScript',
        '.tsx': 'TypeScript',
        '.java': 'Java',
        '.cpp': 'C++',
        '.c': 'C',
        '.cs': 'C#',
        '.rb': 'Ruby',
        '.go': 'Go',
        '.rs': 'Rust',
        '.php': 'PHP',
        '.swift': 'Swift',
        '.kt': 'Kotlin'
    }
    
    detected_languages = set()
    
    if path.is_file():
        ext = path.suffix.lower()
        if ext in language_extensions:
            detected_languages.add(language_extensions[ext])
    else:
        # directory analysis
        for file_path in path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in language_extensions:
                    detected_languages.add(language_extensions[ext])
    
    return detected_languages


def run_analysis(path: str) -> None:
    """
    Run code quality analysis on the specified path.
    
    This function will be implemented with actual analysis logic.
    Currently serves as a placeholder for future implementation.
    
    Args:
        path: Path to the code to analyze
    """
    from langgraph_workflow import run_workflow
    
    try:
        # run the complete workflow
        result = run_workflow(path)
        
        print(f"\n✅ Analysis Complete!")
        print(f"📊 Issues Found: {result['issues_found']}")
        
        if result.get('report_path'):
            print(f"📄 Report saved to: {result['report_path']}")
        
        if result.get('error'):
            print(f"⚠️  Error: {result['error']}")
        
        # if there's an answer (Q&A was run)
        if result.get('answer'):
            print(f"\n💡 Answer to your question:")
            print(result['answer'])
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise


def analyze_command(args: argparse.Namespace) -> None:
    """
    Handle the 'analyze' subcommand.
    
    Args:
        args: Parsed command line arguments
    """
    path = Path(args.path).resolve()
    
    # validate path exists
    if not path.exists():
        print(f"Error: Path '{path}' does not exist.")
        sys.exit(1)
    
    print(f"Starting code quality analysis for: {path}")
    print("-" * 60)
    
    # detect languages
    detected_languages = detect_languages(path)
    
    if detected_languages:
        print(f" Detected languages: {', '.join(sorted(detected_languages))}")
    else:
        print(" No supported programming languages detected.")
        sys.exit(1)
    
    print("\n Running analysis...")
    
    # call the analysis function
    run_analysis(str(path))
    
    # print completion message
    print("\n Analysis complete!")
    print(f"Code quality report has been generated for {path.name}")


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command line argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog='code_quality_agent',
        description='A powerful CLI tool for analyzing code quality across multiple programming languages.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze ./src
  %(prog)s analyze /path/to/project
  %(prog)s analyze main.py
        """
    )
    
    # add version argument
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    # create subparsers for commands
    subparsers = parser.add_subparsers(
        title='commands',
        description='Available commands',
        dest='command',
        required=True,
        help='Command to execute'
    )
    
    # add 'analyze' subcommand
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze code quality for the specified path',
        description='Perform comprehensive code quality analysis on files or directories.'
    )
    
    analyze_parser.add_argument(
        'path',
        type=str,
        help='Path to the code to analyze (file or directory)'
    )
    
    analyze_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    analyze_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for the analysis report',
        default=None
    )
    
    # set the function to call for this subcommand
    analyze_parser.set_defaults(func=analyze_command)
    
    return parser


def main() -> None:
    """
    Main entry point for the code_quality_agent CLI tool.
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # call the appropriate function based on the command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
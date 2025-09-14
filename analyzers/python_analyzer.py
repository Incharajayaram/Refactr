import ast
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import sys


class ComplexityVisitor(ast.NodeVisitor):
    """
    AST visitor to calculate cyclomatic complexity of functions and methods.
    
    Cyclomatic complexity is calculated as:
    1 + number of decision points (if, elif, for, while, except, with, assert, comprehensions)
    """
    
    def __init__(self):
        self.complexity = 1
        self.name = ""
        
    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        if node.orelse and isinstance(node.orelse[0], ast.If):
            self.complexity += 1
        self.generic_visit(node)
        
    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_With(self, node: ast.With) -> None:
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_Assert(self, node: ast.Assert) -> None:
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if isinstance(node.op, ast.And):
            self.complexity += len(node.values) - 1
        self.generic_visit(node)
        
    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.complexity += sum(1 for _ in node.generators)
        self.generic_visit(node)
        
    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.complexity += sum(1 for _ in node.generators)
        self.generic_visit(node)
        
    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.complexity += sum(1 for _ in node.generators)
        self.generic_visit(node)
        
    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.complexity += sum(1 for _ in node.generators)
        self.generic_visit(node)


def calculate_cyclomatic_complexity(func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
    """
    Calculate the cyclomatic complexity of a function or method.
    
    Args:
        func_node: AST node representing a function or method
        
    Returns:
        Cyclomatic complexity score
    """
    visitor = ComplexityVisitor()
    visitor.visit(func_node)
    return visitor.complexity


def get_function_length(func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
    """
    Calculate the number of lines in a function (excluding docstrings and comments).
    
    Args:
        func_node: AST node representing a function or method
        
    Returns:
        Number of lines in the function body
    """
    if not func_node.body:
        return 0
        
    # skip docstring if present
    start_idx = 0
    if (len(func_node.body) > 0 and 
        isinstance(func_node.body[0], ast.Expr) and
        isinstance(func_node.body[0].value, ast.Constant) and
        isinstance(func_node.body[0].value.value, str)):
        start_idx = 1
        
    if start_idx >= len(func_node.body):
        return 0
        
    # calculate line range
    first_stmt = func_node.body[start_idx]
    last_stmt = func_node.body[-1]
    
    # get the end line of the last statement
    end_line = getattr(last_stmt, 'end_lineno', last_stmt.lineno)
    
    return end_line - first_stmt.lineno + 1


def analyze_security_with_bandit(file_path: str) -> List[Dict[str, Any]]:
    """
    Run Bandit security analysis on a Python file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        List of security issues found
    """
    issues = []
    
    try:
        import bandit
        from bandit.core import config
        from bandit.core import manager
        
        # create Bandit manager
        conf = config.BanditConfig()
        b_mgr = manager.BanditManager(conf, 'file')
        
        # discover and run tests
        b_mgr.discover_files([file_path])
        b_mgr.run_tests()
        
        # extract issues
        for issue in b_mgr.get_issue_list():
            issues.append({
                'file': file_path,
                'line': issue.lineno,
                'issue_type': f'security_{issue.test_id}',
                'description': f"Security issue: {issue.text}",
                'suggestion': issue.text + ". Consider using safer alternatives.",
                'severity': issue.severity,
                'confidence': issue.confidence
            })
            
    except ImportError:
        # bandit not installed, skip security analysis
        pass
    except Exception as e:
        # log error but don't fail the entire analysis
        print(f"Warning: Bandit analysis failed for {file_path}: {str(e)}", file=sys.stderr)
        
    return issues


def analyze_python_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Analyze a single Python file for code quality issues.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        List of issues found in the file
    """
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # parse the file
        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError as e:
            issues.append({
                'file': file_path,
                'line': e.lineno or 1,
                'issue_type': 'syntax_error',
                'description': f"Syntax error: {str(e)}",
                'suggestion': "Fix the syntax error before running analysis."
            })
            return issues
            
        # analyze each function/method in the file
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                func_line = node.lineno
                
                # check cyclomatic complexity
                complexity = calculate_cyclomatic_complexity(node)
                if complexity > 10:
                    issues.append({
                        'file': file_path,
                        'line': func_line,
                        'issue_type': 'high_complexity',
                        'description': f"Function '{func_name}' has high cyclomatic complexity ({complexity})",
                        'suggestion': "Consider breaking this function into smaller, more focused functions."
                    })
                
                # check function length
                func_length = get_function_length(node)
                if func_length > 50:
                    issues.append({
                        'file': file_path,
                        'line': func_line,
                        'issue_type': 'long_function',
                        'description': f"Function '{func_name}' is too long ({func_length} lines)",
                        'suggestion': "Consider splitting this function into smaller, more manageable pieces."
                    })
                    
        # run security analysis if Bandit is available
        security_issues = analyze_security_with_bandit(file_path)
        issues.extend(security_issues)
        
    except Exception as e:
        issues.append({
            'file': file_path,
            'line': 1,
            'issue_type': 'analysis_error',
            'description': f"Error analyzing file: {str(e)}",
            'suggestion': "Ensure the file is readable and contains valid Python code."
        })
        
    return issues


def analyze_python(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Analyze Python code quality for a file or directory.
    
    This function serves as the main entry point for Python analysis,
    handling both individual files and directories.
    
    Args:
        path: Path to a Python file or directory containing Python files
        
    Returns:
        List of dictionaries containing analysis results with keys:
        - file: Path to the file
        - line: Line number where issue was found
        - issue_type: Type of issue (high_complexity, long_function, security_*, etc.)
        - description: Human-readable description of the issue
        - suggestion: Recommendation for fixing the issue
    """
    path = Path(path)
    issues = []
    
    if not path.exists():
        return [{
            'file': str(path),
            'line': 1,
            'issue_type': 'file_not_found',
            'description': f"Path '{path}' does not exist",
            'suggestion': "Provide a valid path to a Python file or directory."
        }]
    
    # handle single file
    if path.is_file():
        if path.suffix == '.py':
            return analyze_python_file(str(path))
        else:
            return [{
                'file': str(path),
                'line': 1,
                'issue_type': 'not_python_file',
                'description': f"File '{path}' is not a Python file",
                'suggestion': "Provide a Python file with .py extension."
            }]
    
    # handle directory
    if path.is_dir():
        python_files = list(path.rglob('*.py'))
        
        if not python_files:
            return [{
                'file': str(path),
                'line': 1,
                'issue_type': 'no_python_files',
                'description': f"No Python files found in '{path}'",
                'suggestion': "Ensure the directory contains Python files."
            }]
        
        # analyze each Python file
        for py_file in python_files:
            # skip common directories that should be ignored
            parts = py_file.parts
            if any(part in ['.venv', 'venv', '__pycache__', '.git', 'node_modules'] for part in parts):
                continue
                
            file_issues = analyze_python_file(str(py_file))
            issues.extend(file_issues)
    
    return issues


# example usage and testing
if __name__ == "__main__":
    # test with current directory or provided path
    test_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print(f"Analyzing Python code in: {test_path}")
    print("-" * 60)
    
    results = analyze_python(test_path)
    
    if results:
        for issue in results:
            print(f"\n{issue['file']}:{issue['line']}")
            print(f"  Issue: {issue['issue_type']}")
            print(f"  Description: {issue['description']}")
            print(f"  Suggestion: {issue['suggestion']}")
    else:
        print("No issues found!")
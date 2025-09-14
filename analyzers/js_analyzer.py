import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Union, Optional


# javaScript analysis script that will be executed via Node.js
JS_ANALYZER_SCRIPT = '''
const fs = require('fs');
const path = require('path');
const esprima = require('esprima');

class JSAnalyzer {
    constructor() {
        this.issues = [];
        this.declaredVars = new Map();
        this.usedVars = new Set();
        this.currentFile = '';
    }

    analyzeFile(filePath) {
        this.issues = [];
        this.declaredVars.clear();
        this.usedVars.clear();
        this.currentFile = filePath;
        
        try {
            const code = fs.readFileSync(filePath, 'utf8');
            const ast = esprima.parseScript(code, {
                loc: true,
                range: true,
                tolerant: true
            });
            
            this.walkAST(ast, 0, []);
            this.checkUnusedVariables();
            
            return this.issues;
        } catch (error) {
            return [{
                file: filePath,
                line: 1,
                issue_type: 'parse_error',
                description: `JavaScript parse error: ${error.message}`,
                suggestion: 'Fix syntax errors in the file before analysis.'
            }];
        }
    }

    walkAST(node, loopDepth = 0, scopeStack = []) {
        if (!node) return;

        switch (node.type) {
            case 'FunctionDeclaration':
            case 'FunctionExpression':
            case 'ArrowFunctionExpression':
                this.checkFunctionLength(node);
                // Create new scope for function
                scopeStack.push(new Set());
                break;
                
            case 'VariableDeclaration':
                this.trackVariableDeclaration(node);
                break;
                
            case 'Identifier':
                this.trackVariableUsage(node, scopeStack);
                break;
                
            case 'ForStatement':
            case 'ForInStatement':
            case 'ForOfStatement':
            case 'WhileStatement':
            case 'DoWhileStatement':
                loopDepth++;
                if (loopDepth > 2) {
                    this.addIssue(
                        node.loc.start.line,
                        'deep_nesting',
                        `Nested loops detected (${loopDepth} levels deep)`,
                        'Refactor to reduce nesting. Consider extracting inner loops into separate functions.'
                    );
                }
                break;
        }

        // Recursively walk child nodes
        for (const key in node) {
            if (node[key] && typeof node[key] === 'object') {
                if (Array.isArray(node[key])) {
                    for (const child of node[key]) {
                        this.walkAST(child, loopDepth, scopeStack);
                    }
                } else if (node[key].type) {
                    this.walkAST(node[key], loopDepth, scopeStack);
                }
            }
        }

        // Clean up loop depth and scope
        if (['ForStatement', 'ForInStatement', 'ForOfStatement', 
             'WhileStatement', 'DoWhileStatement'].includes(node.type)) {
            loopDepth--;
        }
        
        if (['FunctionDeclaration', 'FunctionExpression', 
             'ArrowFunctionExpression'].includes(node.type)) {
            scopeStack.pop();
        }
    }

    checkFunctionLength(node) {
        if (!node.loc) return;
        
        const startLine = node.loc.start.line;
        const endLine = node.loc.end.line;
        const functionLength = endLine - startLine + 1;
        
        if (functionLength > 50) {
            const functionName = node.id ? node.id.name : '<anonymous>';
            this.addIssue(
                startLine,
                'long_function',
                `Function '${functionName}' is too long (${functionLength} lines)`,
                'Break this function into smaller, more focused functions.'
            );
        }
    }

    trackVariableDeclaration(node) {
        for (const declaration of node.declarations) {
            if (declaration.id.type === 'Identifier') {
                this.declaredVars.set(declaration.id.name, {
                    line: node.loc.start.line,
                    kind: node.kind
                });
            } else if (declaration.id.type === 'ObjectPattern') {
                // Handle destructuring
                this.extractIdentifiersFromPattern(declaration.id).forEach(name => {
                    this.declaredVars.set(name, {
                        line: node.loc.start.line,
                        kind: node.kind
                    });
                });
            } else if (declaration.id.type === 'ArrayPattern') {
                // Handle array destructuring
                this.extractIdentifiersFromPattern(declaration.id).forEach(name => {
                    this.declaredVars.set(name, {
                        line: node.loc.start.line,
                        kind: node.kind
                    });
                });
            }
        }
    }

    extractIdentifiersFromPattern(pattern) {
        const identifiers = [];
        
        if (pattern.type === 'Identifier') {
            identifiers.push(pattern.name);
        } else if (pattern.type === 'ObjectPattern') {
            for (const prop of pattern.properties) {
                if (prop.value.type === 'Identifier') {
                    identifiers.push(prop.value.name);
                } else if (prop.value.type === 'ObjectPattern' || 
                          prop.value.type === 'ArrayPattern') {
                    identifiers.push(...this.extractIdentifiersFromPattern(prop.value));
                }
            }
        } else if (pattern.type === 'ArrayPattern') {
            for (const element of pattern.elements) {
                if (element && element.type === 'Identifier') {
                    identifiers.push(element.name);
                } else if (element && (element.type === 'ObjectPattern' || 
                          element.type === 'ArrayPattern')) {
                    identifiers.push(...this.extractIdentifiersFromPattern(element));
                }
            }
        }
        
        return identifiers;
    }

    trackVariableUsage(node, scopeStack) {
        // Skip if it's a property access (e.g., obj.property)
        if (node.parent && node.parent.type === 'MemberExpression' && 
            node.parent.property === node) {
            return;
        }
        
        // Skip function names in declarations
        if (node.parent && node.parent.type === 'FunctionDeclaration' && 
            node.parent.id === node) {
            return;
        }
        
        this.usedVars.add(node.name);
    }

    checkUnusedVariables() {
        for (const [varName, varInfo] of this.declaredVars) {
            if (!this.usedVars.has(varName)) {
                // Skip some common exceptions
                if (varName.startsWith('_') || varName === 'React') {
                    continue;
                }
                
                this.addIssue(
                    varInfo.line,
                    'unused_variable',
                    `Variable '${varName}' is declared but never used`,
                    'Remove unused variable or use it in your code.'
                );
            }
        }
    }

    addIssue(line, issueType, description, suggestion) {
        this.issues.push({
            file: this.currentFile,
            line: line,
            issue_type: issueType,
            description: description,
            suggestion: suggestion
        });
    }
}

// Main execution
const filePath = process.argv[2];
const analyzer = new JSAnalyzer();
const issues = analyzer.analyzeFile(filePath);
console.log(JSON.stringify(issues));
'''


def check_node_and_esprima() -> bool:
    """
    Check if Node.js and esprima are available.
    
    Returns:
        True if both are available, False otherwise
    """
    try:
        # check Node.js
        result = subprocess.run(['node', '--version'], 
                              capture_output=True, text=True, check=True)
        
        # check esprima
        check_script = "console.log(require('esprima').version)"
        result = subprocess.run(['node', '-e', check_script], 
                              capture_output=True, text=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_esprima() -> bool:
    """
    Attempt to install esprima using npm.
    
    Returns:
        True if installation succeeded, False otherwise
    """
    try:
        print("Installing esprima... This may take a moment.", file=sys.stderr)
        subprocess.run(['npm', 'install', '-g', 'esprima'], 
                      capture_output=True, text=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def analyze_js_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Analyze a single JavaScript file for code quality issues.
    
    Args:
        file_path: Path to the JavaScript file
        
    Returns:
        List of issues found in the file
    """
    issues = []
    
    # create temporary file for the analyzer script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        f.write(JS_ANALYZER_SCRIPT)
        analyzer_script_path = f.name
    
    try:
        # run the analyzer
        result = subprocess.run(
            ['node', analyzer_script_path, file_path],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        if result.returncode == 0 and result.stdout:
            try:
                issues = json.loads(result.stdout)
            except json.JSONDecodeError:
                issues.append({
                    'file': file_path,
                    'line': 1,
                    'issue_type': 'analysis_error',
                    'description': 'Failed to parse analysis results',
                    'suggestion': 'Check that the file contains valid JavaScript.'
                })
        else:
            error_msg = result.stderr or 'Unknown error'
            issues.append({
                'file': file_path,
                'line': 1,
                'issue_type': 'analysis_error',
                'description': f'Analysis failed: {error_msg}',
                'suggestion': 'Ensure the file contains valid JavaScript code.'
            })
            
    except subprocess.TimeoutExpired:
        issues.append({
            'file': file_path,
            'line': 1,
            'issue_type': 'timeout',
            'description': 'Analysis timed out',
            'suggestion': 'The file may be too large or complex to analyze.'
        })
    except Exception as e:
        issues.append({
            'file': file_path,
            'line': 1,
            'issue_type': 'analysis_error',
            'description': f'Unexpected error: {str(e)}',
            'suggestion': 'Check system configuration and dependencies.'
        })
    finally:
        # clean up temporary file
        try:
            os.unlink(analyzer_script_path)
        except:
            pass
    
    return issues


def analyze_javascript(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Analyze JavaScript code quality for a file or directory.
    
    This function serves as the main entry point for JavaScript analysis,
    handling both individual files and directories.
    
    Args:
        path: Path to a JavaScript file or directory containing JavaScript files
        
    Returns:
        List of dictionaries containing analysis results with keys:
        - file: Path to the file
        - line: Line number where issue was found
        - issue_type: Type of issue (long_function, deep_nesting, unused_variable, etc.)
        - description: Human-readable description of the issue
        - suggestion: Recommendation for fixing the issue
    """
    path = Path(path)
    issues = []
    
    # check dependencies
    if not check_node_and_esprima():
        # try to install esprima
        if not install_esprima():
            return [{
                'file': str(path),
                'line': 1,
                'issue_type': 'missing_dependencies',
                'description': 'Node.js and/or esprima not found',
                'suggestion': 'Install Node.js and run: npm install -g esprima'
            }]
    
    if not path.exists():
        return [{
            'file': str(path),
            'line': 1,
            'issue_type': 'file_not_found',
            'description': f"Path '{path}' does not exist",
            'suggestion': "Provide a valid path to a JavaScript file or directory."
        }]
    
    # handle single file
    if path.is_file():
        if path.suffix in ['.js', '.jsx', '.mjs']:
            return analyze_js_file(str(path))
        else:
            return [{
                'file': str(path),
                'line': 1,
                'issue_type': 'not_javascript_file',
                'description': f"File '{path}' is not a JavaScript file",
                'suggestion': "Provide a JavaScript file with .js, .jsx, or .mjs extension."
            }]
    
    # handle directory
    if path.is_dir():
        js_extensions = ['*.js', '*.jsx', '*.mjs']
        js_files = []
        
        for ext in js_extensions:
            js_files.extend(path.rglob(ext))
        
        if not js_files:
            return [{
                'file': str(path),
                'line': 1,
                'issue_type': 'no_javascript_files',
                'description': f"No JavaScript files found in '{path}'",
                'suggestion': "Ensure the directory contains JavaScript files."
            }]
        
        # analyze each JavaScript file
        for js_file in js_files:
            # skip common directories that should be ignored
            parts = js_file.parts
            if any(part in ['node_modules', '.git', 'dist', 'build', 
                           'coverage', '.next', '.nuxt'] for part in parts):
                continue
            
            file_issues = analyze_js_file(str(js_file))
            issues.extend(file_issues)
    
    return issues


# example usage and testing
if __name__ == "__main__":
    # test with current directory or provided path
    test_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print(f"Analyzing JavaScript code in: {test_path}")
    print("-" * 60)
    
    results = analyze_javascript(test_path)
    
    if results:
        for issue in results:
            print(f"\n{issue['file']}:{issue['line']}")
            print(f"  Issue: {issue['issue_type']}")
            print(f"  Description: {issue['description']}")
            print(f"  Suggestion: {issue['suggestion']}")
    else:
        print("No issues found!")
#!/usr/bin/env node
/**
 * JavaScript AST Parser for extracting function definitions and calls.
 * Used by the Python visualization module to analyze JavaScript code structure.
 */

const fs = require('fs');
const path = require('path');
const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;

/**
 * Parse a JavaScript file and extract function definitions and calls.
 * 
 * @param {string} filePath - Path to the JavaScript file
 * @returns {Object} - Object containing functions and calls
 */
function parseJavaScriptFile(filePath) {
    const result = {
        functions: {},
        calls: []
    };

    try {
        // Read file content
        const code = fs.readFileSync(filePath, 'utf8');
        
        // Parse the code
        const ast = parser.parse(code, {
            sourceType: 'unambiguous',
            plugins: [
                'jsx',
                'typescript',
                'decorators-legacy',
                'classProperties',
                'asyncGenerators',
                'dynamicImport'
            ]
        });

        // Track current scope
        let currentFunction = null;
        const functionStack = [];

        // Traverse the AST
        traverse(ast, {
            // Function declarations
            FunctionDeclaration(path) {
                const name = path.node.id ? path.node.id.name : '<anonymous>';
                result.functions[name] = {
                    line: path.node.loc.start.line,
                    type: 'function'
                };
            },

            // Function expressions and arrow functions
            FunctionExpression(path) {
                // Try to get the name from variable assignment
                let name = '<anonymous>';
                if (path.node.id) {
                    name = path.node.id.name;
                } else if (path.parent.type === 'VariableDeclarator' && path.parent.id) {
                    name = path.parent.id.name;
                } else if (path.parent.type === 'AssignmentExpression' && path.parent.left.type === 'Identifier') {
                    name = path.parent.left.name;
                }
                
                if (name !== '<anonymous>') {
                    result.functions[name] = {
                        line: path.node.loc.start.line,
                        type: 'expression'
                    };
                }
            },

            ArrowFunctionExpression(path) {
                let name = '<anonymous>';
                if (path.parent.type === 'VariableDeclarator' && path.parent.id) {
                    name = path.parent.id.name;
                } else if (path.parent.type === 'AssignmentExpression' && path.parent.left.type === 'Identifier') {
                    name = path.parent.left.name;
                }
                
                if (name !== '<anonymous>') {
                    result.functions[name] = {
                        line: path.node.loc.start.line,
                        type: 'arrow'
                    };
                }
            },

            // Class methods
            ClassMethod(path) {
                const className = path.parent.parent.id ? path.parent.parent.id.name : '<anonymous>';
                const methodName = path.node.key.name;
                const fullName = `${className}.${methodName}`;
                
                result.functions[fullName] = {
                    line: path.node.loc.start.line,
                    type: 'method'
                };
            },

            // Track entering functions for call attribution
            'FunctionDeclaration|FunctionExpression|ArrowFunctionExpression|ClassMethod': {
                enter(path) {
                    let name = null;
                    
                    if (path.node.type === 'FunctionDeclaration' && path.node.id) {
                        name = path.node.id.name;
                    } else if (path.node.type === 'ClassMethod') {
                        const className = path.parent.parent.id ? path.parent.parent.id.name : '<anonymous>';
                        name = `${className}.${path.node.key.name}`;
                    } else if (path.parent.type === 'VariableDeclarator' && path.parent.id) {
                        name = path.parent.id.name;
                    }
                    
                    if (name) {
                        functionStack.push(name);
                        currentFunction = name;
                    }
                },
                exit(path) {
                    functionStack.pop();
                    currentFunction = functionStack.length > 0 ? functionStack[functionStack.length - 1] : null;
                }
            },

            // Call expressions
            CallExpression(path) {
                if (currentFunction) {
                    let calleeName = null;
                    
                    if (path.node.callee.type === 'Identifier') {
                        calleeName = path.node.callee.name;
                    } else if (path.node.callee.type === 'MemberExpression') {
                        // Handle method calls like obj.method()
                        if (path.node.callee.property.type === 'Identifier') {
                            if (path.node.callee.object.type === 'Identifier') {
                                calleeName = `${path.node.callee.object.name}.${path.node.callee.property.name}`;
                            } else {
                                calleeName = path.node.callee.property.name;
                            }
                        }
                    }
                    
                    if (calleeName) {
                        result.calls.push({
                            caller: currentFunction,
                            callee: calleeName,
                            line: path.node.loc.start.line
                        });
                    }
                }
            }
        });

    } catch (error) {
        console.error(`Error parsing ${filePath}: ${error.message}`);
        // Return empty result on error
        return result;
    }

    return result;
}

// Main execution
if (require.main === module) {
    const args = process.argv.slice(2);
    
    if (args.length !== 1) {
        console.error('Usage: node js_ast_parser.js <file_path>');
        process.exit(1);
    }

    const filePath = args[0];
    
    if (!fs.existsSync(filePath)) {
        console.error(`File not found: ${filePath}`);
        process.exit(1);
    }

    const result = parseJavaScriptFile(filePath);
    console.log(JSON.stringify(result, null, 2));
}

module.exports = { parseJavaScriptFile };
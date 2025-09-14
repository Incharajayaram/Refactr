import ast
import os
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# optional imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PythonCallAnalyzer(ast.NodeVisitor):
    """AST visitor to extract function definitions and calls from Python code."""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.functions = {}  # name -> (line, column)
        self.calls = []  # [(caller, callee, line)]
        self.current_function = None
        self.class_stack = []
    
    def visit_ClassDef(self, node):
        """Track class definitions for method resolution."""
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()
    
    def visit_FunctionDef(self, node):
        """Track function definitions."""
        if self.class_stack:
            func_name = f"{'.'.join(self.class_stack)}.{node.name}"
        else:
            func_name = node.name
        
        self.functions[func_name] = (node.lineno, node.col_offset)
        
        # track current function for call attribution
        old_function = self.current_function
        self.current_function = func_name
        self.generic_visit(node)
        self.current_function = old_function
    
    visit_AsyncFunctionDef = visit_FunctionDef
    
    def visit_Call(self, node):
        """Track function calls."""
        if self.current_function:
            callee = self._get_call_name(node)
            if callee:
                self.calls.append((
                    self.current_function,
                    callee,
                    node.lineno
                ))
        self.generic_visit(node)
    
    def _get_call_name(self, node) -> Optional[str]:
        """Extract the name of the called function."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            # handle chained attributes like obj.method
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))
        return None


def parse_js_ast(file_path: str) -> Dict[str, Any]:
    """
    Parse JavaScript file using Node.js helper script.
    
    Args:
        file_path: Path to JavaScript file
        
    Returns:
        Dictionary with functions and calls
    """
    js_parser_path = Path(__file__).parent.parent / "analyzers" / "js_ast_parser.js"
    
    if not js_parser_path.exists():
        logger.warning(f"JavaScript parser not found at {js_parser_path}")
        return {"functions": {}, "calls": []}
    
    try:
        result = subprocess.run(
            ["node", str(js_parser_path), file_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            logger.error(f"JS parser error: {result.stderr}")
            return {"functions": {}, "calls": []}
    except Exception as e:
        logger.error(f"Failed to parse JS file {file_path}: {e}")
        return {"functions": {}, "calls": []}


def build_call_graph(repo_path: str) -> nx.DiGraph:
    """
    Build a directed graph of function calls in the repository.
    
    Args:
        repo_path: Path to the repository root
        
    Returns:
        NetworkX directed graph with function nodes and call edges
    """
    G = nx.DiGraph()
    repo_path = Path(repo_path)
    
    # parse Python files
    for py_file in repo_path.rglob("*.py"):
        if any(part.startswith('.') or part in ['venv', '__pycache__', 'node_modules'] 
               for part in py_file.parts):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            module_name = str(py_file.relative_to(repo_path)).replace('/', '.').replace('.py', '')
            
            analyzer = PythonCallAnalyzer(module_name)
            analyzer.visit(tree)
            
            # add nodes for functions
            for func_name, (line, _) in analyzer.functions.items():
                full_name = f"{module_name}.{func_name}"
                G.add_node(full_name, 
                          language='Python',
                          file=str(py_file.relative_to(repo_path)),
                          line=line)
            
            # add edges for calls
            for caller, callee, line in analyzer.calls:
                caller_full = f"{module_name}.{caller}"
                # try to resolve callee to full name
                if callee in analyzer.functions:
                    callee_full = f"{module_name}.{callee}"
                else:
                    callee_full = callee
                
                G.add_edge(caller_full, callee_full, line=line)
                
        except Exception as e:
            logger.error(f"Error parsing Python file {py_file}: {e}")
    
    # parse JavaScript files
    for js_file in repo_path.rglob("*.js"):
        if any(part.startswith('.') or part in ['node_modules', 'dist', 'build'] 
               for part in js_file.parts):
            continue
            
        try:
            js_data = parse_js_ast(str(js_file))
            module_name = str(js_file.relative_to(repo_path)).replace('/', '.').replace('.js', '')
            
            # add nodes for functions
            for func_name, info in js_data.get('functions', {}).items():
                full_name = f"{module_name}.{func_name}"
                G.add_node(full_name,
                          language='JavaScript',
                          file=str(js_file.relative_to(repo_path)),
                          line=info.get('line', 0))
            
            # add edges for calls
            for call in js_data.get('calls', []):
                caller_full = f"{module_name}.{call['caller']}"
                callee = call['callee']
                # try to resolve to full name
                if callee in js_data.get('functions', {}):
                    callee_full = f"{module_name}.{callee}"
                else:
                    callee_full = callee
                
                G.add_edge(caller_full, callee_full, line=call.get('line', 0))
                
        except Exception as e:
            logger.error(f"Error parsing JavaScript file {js_file}: {e}")
    
    return G


def save_dependency_graph_png(G: nx.Graph, out_path: str) -> None:
    """
    Save a dependency graph as PNG with nodes sized by centrality.
    
    Args:
        G: NetworkX graph
        out_path: Output path for PNG file
    """
    if len(G) == 0:
        logger.warning("Empty graph, skipping visualization")
        return
    
    plt.figure(figsize=(14, 10))
    
    # calculate centrality for node sizing
    try:
        centrality = nx.degree_centrality(G)
    except:
        centrality = {node: 1 for node in G.nodes()}
    
    # normalize centrality for node sizes
    if centrality:
        min_cent = min(centrality.values())
        max_cent = max(centrality.values())
        if max_cent > min_cent:
            node_sizes = [300 + 2000 * (centrality.get(node, min_cent) - min_cent) / 
                         (max_cent - min_cent) for node in G.nodes()]
        else:
            node_sizes = [800 for _ in G.nodes()]
    else:
        node_sizes = [800 for _ in G.nodes()]
    
    # color nodes by language
    node_colors = []
    for node in G.nodes():
        attrs = G.nodes[node]
        if attrs.get('language') == 'Python':
            node_colors.append('#3776ab')  # Python blue
        elif attrs.get('language') == 'JavaScript':
            node_colors.append('#f7df1e')  # JavaScript yellow
        else:
            node_colors.append('#808080')  # Gray for unknown
    
    # layout algorithm
    if len(G) < 50:
        pos = nx.spring_layout(G, k=2, iterations=50)
    else:
        pos = nx.kamada_kawai_layout(G)
    
    # draw the graph
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color=node_colors,
                          alpha=0.7)
    
    # draw edges with transparency based on graph size
    edge_alpha = max(0.1, min(0.5, 10 / len(G.edges())))
    nx.draw_networkx_edges(G, pos, alpha=edge_alpha, arrows=True,
                          arrowsize=10, edge_color='gray')
    
    # draw labels for high-centrality nodes only
    if len(G) > 20:
        # only label top 20% most central nodes
        threshold = np.percentile(list(centrality.values()), 80)
        labels = {node: node.split('.')[-1] for node, cent in centrality.items() 
                 if cent >= threshold}
    else:
        labels = {node: node.split('.')[-1] for node in G.nodes()}
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # add legend
    python_patch = mpatches.Patch(color='#3776ab', label='Python')
    js_patch = mpatches.Patch(color='#f7df1e', label='JavaScript')
    plt.legend(handles=[python_patch, js_patch], loc='upper right')
    
    plt.title(f"Function Call Dependency Graph ({len(G)} nodes, {len(G.edges())} edges)", 
              fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # save figure
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved dependency graph to {out_path}")


def compute_file_metrics(repo_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Compute metrics for each file in the repository.
    
    Args:
        repo_path: Path to repository root
        
    Returns:
        Dictionary mapping file paths to their metrics
    """
    from analyzers.python_analyzer import PythonAnalyzer
    from analyzers.js_analyzer import JavaScriptAnalyzer
    
    metrics_by_file = {}
    repo_path = Path(repo_path)
    
    # analyze Python files
    py_analyzer = PythonAnalyzer()
    for py_file in repo_path.rglob("*.py"):
        if any(part.startswith('.') or part in ['venv', '__pycache__'] 
               for part in py_file.parts):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = py_analyzer.analyze_code(content)
            rel_path = str(py_file.relative_to(repo_path))
            
            metrics_by_file[rel_path] = {
                'complexity': analysis.get('average_complexity', 0),
                'lines': analysis.get('total_lines', 0),
                'num_issues': len(analysis.get('issues', [])),
                'functions': analysis.get('function_count', 0),
                'classes': analysis.get('class_count', 0),
                'language': 'Python'
            }
        except Exception as e:
            logger.error(f"Error analyzing {py_file}: {e}")
    
    # analyze JavaScript files
    js_analyzer = JavaScriptAnalyzer()
    for js_file in repo_path.rglob("*.js"):
        if any(part.startswith('.') or part in ['node_modules', 'dist', 'build'] 
               for part in js_file.parts):
            continue
            
        try:
            with open(js_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = js_analyzer.analyze_code(content)
            rel_path = str(js_file.relative_to(repo_path))
            
            metrics_by_file[rel_path] = {
                'complexity': analysis.get('average_complexity', 0),
                'lines': analysis.get('total_lines', 0),
                'num_issues': len(analysis.get('issues', [])),
                'functions': analysis.get('function_count', 0),
                'classes': analysis.get('class_count', 0),
                'language': 'JavaScript'
            }
        except Exception as e:
            logger.error(f"Error analyzing {js_file}: {e}")
    
    return metrics_by_file


def generate_hotspot_heatmap(metrics_by_file: Dict[str, Dict], out_path: str) -> None:
    """
    Generate a heatmap showing complexity hotspots across files.
    
    Args:
        metrics_by_file: Dictionary mapping file paths to metrics
        out_path: Output path for the heatmap PNG
    """
    if not metrics_by_file:
        logger.warning("No metrics data, skipping heatmap")
        return
    
    # prepare data for heatmap
    files = list(metrics_by_file.keys())
    
    # create metrics matrix
    metric_names = ['complexity', 'lines', 'num_issues', 'functions']
    data = []
    
    for file in files:
        metrics = metrics_by_file[file]
        row = []
        for metric in metric_names:
            value = metrics.get(metric, 0)
            # normalize by lines for rate metrics
            if metric in ['complexity', 'num_issues'] and metrics.get('lines', 0) > 0:
                value = value / metrics['lines'] * 100
            row.append(value)
        data.append(row)
    
    # create DataFrame
    df = pd.DataFrame(data, index=files, columns=[
        'Complexity/100LOC', 'Lines', 'Issues/100LOC', 'Functions'
    ])
    
    # normalize columns for better visualization
    df_normalized = df.copy()
    for col in df.columns:
        if df[col].max() > df[col].min():
            df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    # create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(files) * 0.3)),
                                    gridspec_kw={'width_ratios': [1, 3]})
    
    # language distribution
    languages = pd.Series([m.get('language', 'Unknown') for m in metrics_by_file.values()])
    lang_counts = languages.value_counts()
    
    ax1.pie(lang_counts.values, labels=lang_counts.index, autopct='%1.1f%%',
            colors=['#3776ab', '#f7df1e', '#808080'])
    ax1.set_title('Language Distribution')
    
    # use custom colormap - white to red for issues
    cmap = LinearSegmentedColormap.from_list('custom', ['#ffffff', '#ffcccc', '#ff6666', '#cc0000'])
    
    # truncate file names for readability
    display_files = [f if len(f) <= 40 else '...' + f[-37:] for f in files]
    
    sns.heatmap(df_normalized, annot=df.round(1), fmt='g', cmap=cmap,
                yticklabels=display_files, cbar_kws={'label': 'Normalized Score'},
                ax=ax2)
    
    ax2.set_title('Code Quality Heatmap', fontsize=16, pad=20)
    ax2.set_xlabel('Metrics')
    
    plt.tight_layout()
    
    # save figure
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved heatmap to {out_path}")


def init_history_db(db_path: str) -> None:
    """Initialize the history database for trend tracking."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            repo_name TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            metadata TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_repo_metric 
        ON metrics_history(repo_name, metric_name)
    ''')
    
    conn.commit()
    conn.close()


def save_metrics_to_history(db_path: str, repo_name: str, metrics: Dict[str, float]) -> None:
    """Save current metrics to history database."""
    init_history_db(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    
    for metric_name, metric_value in metrics.items():
        cursor.execute('''
            INSERT INTO metrics_history (timestamp, repo_name, metric_name, metric_value)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, repo_name, metric_name, float(metric_value)))
    
    conn.commit()
    conn.close()


def plot_trend_chart(history_db: str, metric: str, out_path: str, 
                    repo_name: Optional[str] = None) -> None:
    """
    Plot historical trend for a specific metric.
    
    Args:
        history_db: Path to SQLite history database
        metric: Metric name to plot
        out_path: Output path for the chart
        repo_name: Optional repository name filter
    """
    if not Path(history_db).exists():
        logger.warning(f"History database not found at {history_db}")
        return
    
    conn = sqlite3.connect(history_db)
    
    query = '''
        SELECT timestamp, repo_name, metric_value 
        FROM metrics_history 
        WHERE metric_name = ?
    '''
    params = [metric]
    
    if repo_name:
        query += ' AND repo_name = ?'
        params.append(repo_name)
    
    query += ' ORDER BY timestamp'
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty:
        logger.warning(f"No data found for metric '{metric}'")
        return
    
    # convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # create plot
    plt.figure(figsize=(12, 6))
    
    if repo_name or df['repo_name'].nunique() == 1:
        # single repository
        plt.plot(df['timestamp'], df['metric_value'], marker='o', linewidth=2)
        plt.fill_between(df['timestamp'], df['metric_value'], alpha=0.3)
    else:
        # multiple repositories
        for repo in df['repo_name'].unique():
            repo_data = df[df['repo_name'] == repo]
            plt.plot(repo_data['timestamp'], repo_data['metric_value'], 
                    marker='o', label=repo, linewidth=2)
    
    plt.title(f'{metric.replace("_", " ").title()} Over Time', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    if not repo_name and df['repo_name'].nunique() > 1:
        plt.legend()
    
    plt.tight_layout()
    
    # save figure
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved trend chart to {out_path}")


def embed_visualizations_in_markdown(report_md_path: str, visuals: List[str]) -> None:
    """
    Embed visualization images into the markdown report.
    
    Args:
        report_md_path: Path to the markdown report file
        visuals: List of image paths to embed
    """
    if not visuals:
        return
    
    # read existing report
    with open(report_md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # add visualizations section
    visuals_section = "\n\n## 📊 Visualizations\n\n"
    
    for visual_path in visuals:
        if not Path(visual_path).exists():
            continue
        
        # determine caption based on filename
        filename = Path(visual_path).name
        if 'dependency' in filename:
            caption = "Function Call Dependency Graph"
        elif 'heatmap' in filename:
            caption = "Code Quality Heatmap"
        elif 'trend' in filename:
            caption = "Historical Trend Analysis"
        else:
            caption = "Code Analysis Visualization"
        
        # make path relative to report location
        rel_path = Path(visual_path).relative_to(Path(report_md_path).parent)
        
        visuals_section += f"### {caption}\n\n"
        visuals_section += f"![{caption}]({rel_path})\n\n"
    
    # append to report
    with open(report_md_path, 'a', encoding='utf-8') as f:
        f.write(visuals_section)
    
    logger.info(f"Embedded {len(visuals)} visualizations in report")


def render_dependency_network_html(G: nx.Graph, out_path: str) -> None:
    """
    Generate an interactive HTML visualization of the dependency network.
    
    Args:
        G: NetworkX graph
        out_path: Output path for HTML file
    """
    if not PYVIS_AVAILABLE:
        logger.warning("pyvis not available, skipping interactive visualization")
        return
    
    if len(G) == 0:
        logger.warning("Empty graph, skipping visualization")
        return
    
    # create pyvis network
    net = Network(height='800px', width='100%', directed=True,
                  bgcolor='#ffffff', font_color='#000000')
    
    # configure physics
    net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=100)
    
    # add nodes with attributes
    for node in G.nodes():
        attrs = G.nodes[node]
        
        # determine color by language
        if attrs.get('language') == 'Python':
            color = '#3776ab'
        elif attrs.get('language') == 'JavaScript':
            color = '#f7df1e'
        else:
            color = '#808080'
        
        # calculate size based on degree
        size = 10 + min(50, G.degree(node) * 5)
        
        # create hover title
        title = f"{node}\n"
        title += f"File: {attrs.get('file', 'Unknown')}\n"
        title += f"Line: {attrs.get('line', 'Unknown')}\n"
        title += f"In-degree: {G.in_degree(node)}\n"
        title += f"Out-degree: {G.out_degree(node)}"
        
        net.add_node(node, label=node.split('.')[-1], color=color,
                     size=size, title=title)
    
    # add edges
    for source, target in G.edges():
        net.add_edge(source, target)
    
    # add custom controls
    net.show_buttons(filter_=['physics'])
    
    # save HTML
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(out_path)
    
    logger.info(f"Saved interactive network to {out_path}")


# example usage and demonstration
if __name__ == "__main__":
    # example metrics for testing
    sample_metrics = {
        "src/main.py": {
            "complexity": 15,
            "lines": 234,
            "num_issues": 3,
            "functions": 8,
            "language": "Python"
        },
        "src/utils.py": {
            "complexity": 8,
            "lines": 156,
            "num_issues": 1,
            "functions": 12,
            "language": "Python"
        },
        "js/app.js": {
            "complexity": 12,
            "lines": 445,
            "num_issues": 5,
            "functions": 15,
            "language": "JavaScript"
        }
    }
    
    # generate sample visualizations
    output_dir = "./work/visuals/sample"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # create sample heatmap
    generate_hotspot_heatmap(sample_metrics, f"{output_dir}/heatmap.png")
    print(f"Generated sample heatmap at {output_dir}/heatmap.png")
    
    # create sample history and trend
    db_path = "./data/history.db"
    Path("./data").mkdir(exist_ok=True)
    
    # add sample historical data
    for i in range(5):
        timestamp = (datetime.now() - timedelta(days=i*7)).isoformat()
        save_metrics_to_history(db_path, "sample_repo", {
            "average_complexity": 10 + i * 0.5,
            "total_issues": 20 - i * 2
        })
    
    plot_trend_chart(db_path, "average_complexity", f"{output_dir}/trend.png")
    print(f"Generated sample trend chart at {output_dir}/trend.png")
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import networkx as nx
from collections import defaultdict, Counter
import numpy as np

# set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


def generate_svg_visualizations(job_dir: str, file_metrics: Dict[str, Dict], visuals_dir: Path) -> Dict[str, str]:
    """Generate simple SVG visualizations using matplotlib."""
    visualization_paths = {}
    
    try:
        # 1. Dependency Graph
        dep_graph_path = visuals_dir / "dependency_graph.svg"
        generate_dependency_graph_svg(job_dir, str(dep_graph_path))
        if dep_graph_path.exists():
            visualization_paths["dependency_graph"] = str(dep_graph_path.relative_to(Path("work")))
        
        # 2. Quality Heatmap
        heatmap_path = visuals_dir / "quality_heatmap.svg"
        generate_quality_heatmap_svg(file_metrics, str(heatmap_path))
        if heatmap_path.exists():
            visualization_paths["quality_heatmap"] = str(heatmap_path.relative_to(Path("work")))
        
        # 3. Issues Distribution (as network replacement)
        network_path = visuals_dir / "dependency_network.svg"
        generate_issues_distribution_svg(file_metrics, str(network_path))
        if network_path.exists():
            visualization_paths["dependency_network"] = str(network_path.relative_to(Path("work")))
            
    except Exception as e:
        print(f"Error generating SVG visualizations: {e}")
    
    return visualization_paths


def generate_dependency_graph_svg(repo_path: str, output_path: str):
    """Generate a simple dependency graph as SVG."""
    try:
        # build a simple import graph
        G = nx.DiGraph()
        dependencies = extract_simple_dependencies(repo_path)
        
        # add edges
        for source, targets in dependencies.items():
            for target in targets:
                if target != source:  # Avoid self-loops
                    G.add_edge(source, target)
        
        # limit graph size for performance
        if G.number_of_nodes() > 50:
            # keep only nodes with highest degree
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:50]
            G = G.subgraph([n[0] for n in top_nodes])
        
        # create figure
        plt.figure(figsize=(12, 8))
        
        if G.number_of_nodes() > 0:
            # use spring layout
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # draw the graph
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                   node_size=500, alpha=0.8)
            nx.draw_networkx_edges(G, pos, edge_color='gray', 
                                   arrows=True, alpha=0.5, arrowsize=20)
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
            
            plt.title(f"Module Dependency Graph ({G.number_of_nodes()} modules)", fontsize=16)
        else:
            plt.text(0.5, 0.5, 'No dependencies found', 
                    ha='center', va='center', fontsize=20)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error generating dependency graph: {e}")
        # create empty plot
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, 'Unable to generate dependency graph', 
                ha='center', va='center', fontsize=16)
        plt.axis('off')
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()


def generate_quality_heatmap_svg(file_metrics: Dict[str, Dict], output_path: str):
    """Generate a quality heatmap as SVG."""
    try:
        if not file_metrics:
            raise ValueError("No file metrics available")
        
        # prepare data
        files = []
        issues = []
        lines = []
        languages = []
        
        for file_path, metrics in file_metrics.items():
            files.append(Path(file_path).name[:30])  # Truncate long names
            issues.append(metrics.get('issues', 0))
            lines.append(metrics.get('lines', 0))
            languages.append(metrics.get('language', 'Unknown'))
        
        # sort by issues
        sorted_indices = sorted(range(len(issues)), key=lambda i: issues[i], reverse=True)
        files = [files[i] for i in sorted_indices][:20]  # Top 20 files
        issues = [issues[i] for i in sorted_indices][:20]
        lines = [lines[i] for i in sorted_indices][:20]
        languages = [languages[i] for i in sorted_indices][:20]
        
        # create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        # bar chart of issues by file
        y_pos = np.arange(len(files))
        colors = ['red' if i > 5 else 'orange' if i > 2 else 'green' for i in issues]
        
        ax1.barh(y_pos, issues, color=colors, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(files)
        ax1.set_xlabel('Number of Issues')
        ax1.set_title('Issues by File (Top 20)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # scatter plot of issues vs lines
        scatter_colors = {'Python': 'blue', 'JavaScript': 'orange', 'Unknown': 'gray'}
        for lang in set(languages):
            mask = [l == lang for l in languages]
            x_vals = [lines[i] for i, m in enumerate(mask) if m]
            y_vals = [issues[i] for i, m in enumerate(mask) if m]
            ax2.scatter(x_vals, y_vals, label=lang, alpha=0.6, s=100,
                       color=scatter_colors.get(lang, 'gray'))
        
        ax2.set_xlabel('Lines of Code')
        ax2.set_ylabel('Number of Issues')
        ax2.set_title('Code Quality Scatter Plot', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Code Quality Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error generating quality heatmap: {e}")
        # create empty plot
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, 'No metrics available for visualization', 
                ha='center', va='center', fontsize=16)
        plt.axis('off')
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()


def generate_issues_distribution_svg(file_metrics: Dict[str, Dict], output_path: str):
    """Generate issues distribution chart as SVG."""
    try:
        if not file_metrics:
            raise ValueError("No file metrics available")
        
        # aggregate data
        total_issues = sum(m.get('issues', 0) for m in file_metrics.values())
        total_files = len(file_metrics)
        
        # count by language
        lang_issues = defaultdict(int)
        lang_files = defaultdict(int)
        
        for metrics in file_metrics.values():
            lang = metrics.get('language', 'Unknown')
            lang_issues[lang] += metrics.get('issues', 0)
            lang_files[lang] += 1
        
        # create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Pie chart of issues by language
        if lang_issues:
            labels = list(lang_issues.keys())
            sizes = list(lang_issues.values())
            colors = plt.cm.Set3(range(len(labels)))
            
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax1.set_title('Issues by Language', fontsize=14)
        
        # 2. Bar chart of files by language
        if lang_files:
            langs = list(lang_files.keys())
            counts = list(lang_files.values())
            
            ax2.bar(langs, counts, color='skyblue', alpha=0.8)
            ax2.set_xlabel('Language')
            ax2.set_ylabel('Number of Files')
            ax2.set_title('Files by Language', fontsize=14)
            ax2.grid(True, alpha=0.3)
        
        # 3. Issues histogram
        all_issues = [m.get('issues', 0) for m in file_metrics.values()]
        if all_issues:
            ax3.hist(all_issues, bins=20, color='coral', alpha=0.8, edgecolor='black')
            ax3.set_xlabel('Number of Issues')
            ax3.set_ylabel('Number of Files')
            ax3.set_title('Distribution of Issues per File', fontsize=14)
            ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax4.axis('off')
        summary_text = f"""
        Analysis Summary
        ================
        
        Total Files Analyzed: {total_files}
        Total Issues Found: {total_issues}
        Average Issues per File: {total_issues/total_files:.2f}
        
        Languages Analyzed:
        """
        for lang, count in lang_files.items():
            avg_issues = lang_issues[lang] / count if count > 0 else 0
            summary_text += f"\n  • {lang}: {count} files, {lang_issues[lang]} issues (avg: {avg_issues:.1f})"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Code Issues Distribution Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error generating issues distribution: {e}")
        # create empty plot
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, 'No data available for visualization', 
                ha='center', va='center', fontsize=16)
        plt.axis('off')
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()


def extract_simple_dependencies(repo_path: str) -> Dict[str, set]:
    """Extract simple Python import dependencies."""
    import os
    dependencies = defaultdict(set)
    repo_path = Path(repo_path)
    
    try:
        for py_file in repo_path.rglob("*.py"):
            if any(part in str(py_file) for part in ['.venv', 'venv', '__pycache__', 'node_modules']):
                continue
            
            try:
                rel_path = py_file.relative_to(repo_path)
                module_name = str(rel_path.with_suffix('')).replace(os.sep, '.')
                
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                # simple import extraction
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        # extract imported module
                        if line.startswith('import '):
                            parts = line[7:].split(',')
                            for part in parts:
                                imp = part.strip().split(' as ')[0].split('.')[0]
                                if imp and not imp.startswith('_'):
                                    dependencies[module_name].add(imp)
                        else:  # from X import Y
                            if ' import ' in line:
                                imp = line.split(' import ')[0][5:].strip().split('.')[0]
                                if imp and not imp.startswith(('_', '.')):
                                    dependencies[module_name].add(imp)
                                    
            except Exception:
                continue
                
    except Exception as e:
        print(f"Error extracting dependencies: {e}")
    
    return dependencies
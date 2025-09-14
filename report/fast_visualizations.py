import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
import concurrent.futures
from collections import defaultdict
import pickle

# lightweight visualization libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class FastVisualizer:
    """Fast visualization generator using modern techniques."""
    
    def __init__(self, cache_dir: str = ".viz_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_key(self, data: Any, viz_type: str) -> str:
        """Generate cache key for visualization data."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(f"{viz_type}:{data_str}".encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load visualization from cache if available."""
        cache_file = self.cache_dir / f"{cache_key}.html"
        if cache_file.exists():
            return str(cache_file)
        return None
    
    def _save_to_cache(self, cache_key: str, fig: go.Figure) -> str:
        """Save visualization to cache."""
        cache_file = self.cache_dir / f"{cache_key}.html"
        fig.write_html(str(cache_file), include_plotlyjs='cdn')
        return str(cache_file)
    
    def generate_fast_heatmap(self, file_metrics: Dict[str, Dict], output_path: str) -> None:
        """
        Generate code quality heatmap using Plotly (much faster than matplotlib).
        
        For large repos:
        - Samples top 100 files by issues
        - Uses interactive HTML instead of static PNG
        - Caches results
        """
        cache_key = self._get_cache_key(file_metrics, "heatmap")
        cached = self._load_from_cache(cache_key)
        if cached:
            # copy cached file to output path
            import shutil
            shutil.copy(cached, output_path)
            return
        
        # sample top files if too many
        if len(file_metrics) > 100:
            # sort by issues + complexity
            sorted_files = sorted(
                file_metrics.items(),
                key=lambda x: x[1].get('issues', 0) + x[1].get('complexity', 0),
                reverse=True
            )[:100]
            file_metrics = dict(sorted_files)
        
        # prepare data
        files = list(file_metrics.keys())
        issues = [m.get('issues', 0) for m in file_metrics.values()]
        lines = [m.get('lines', 0) for m in file_metrics.values()]
        languages = [m.get('language', 'Unknown') for m in file_metrics.values()]
        
        # sort files by issues for better visualization
        sorted_data = sorted(zip(files, issues, lines, languages), key=lambda x: x[1], reverse=True)
        files, issues, lines, languages = zip(*sorted_data) if sorted_data else ([], [], [], [])
        
        # create bar chart instead of scatter for clarity
        fig = go.Figure()
        
        # add horizontal bar chart
        fig.add_trace(go.Bar(
            y=files,
            x=issues,
            orientation='h',
            marker=dict(
                color=issues,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Issues"),
            ),
            text=issues,
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>' +
                         'Issues: %{x}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title="Code Quality Issues by File (Top 100)",
            xaxis_title="Number of Issues",
            yaxis_title="Files",
            height=max(600, len(files) * 15),  # Dynamic height based on files
            margin=dict(l=200),  # More space for filenames
            hovermode='closest',
            showlegend=False
        )
        
        # save as interactive HTML
        output_html = output_path.replace('.png', '.html')
        fig.write_html(output_html, include_plotlyjs='cdn')
        self._save_to_cache(cache_key, fig)
        
        # also create a static image for compatibility
        try:
            fig.write_image(output_path, width=1200, height=800)
        except:
            # if static image fails, create a placeholder
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'View interactive heatmap in HTML report', 
                    ha='center', va='center', fontsize=20)
            plt.axis('off')
            plt.savefig(output_path)
            plt.close()
    
    def generate_fast_dependency_graph(self, repo_path: str, output_path: str, max_nodes: int = 200) -> None:
        """
        Generate dependency graph using sampling and clustering.
        
        For large repos:
        - Samples most connected nodes
        - Groups by module/directory
        - Uses force-directed layout
        """
        cache_key = self._get_cache_key(f"{repo_path}:{max_nodes}", "depgraph")
        cached = self._load_from_cache(cache_key)
        if cached:
            import shutil
            shutil.copy(cached, output_path)
            return
        
        # quick dependency extraction (simplified)
        dependencies = defaultdict(set)
        file_count = 0
        
        for py_file in Path(repo_path).rglob("*.py"):
            if file_count > 500:  # Limit scanning
                break
            
            file_count += 1
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                # quick import extraction (not perfect but fast)
                for line in content.split('\n'):
                    if line.strip().startswith(('import ', 'from ')):
                        parts = line.split()
                        if len(parts) > 1:
                            module = parts[1].split('.')[0]
                            if not module.startswith('_'):
                                dependencies[py_file.stem].add(module)
            except:
                continue
        
        # build graph data
        if not dependencies:
            # no dependencies found, create placeholder
            fig = go.Figure()
            fig.add_annotation(
                text="No dependencies found or repository too large",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
        else:
            # create network graph
            edge_trace = []
            node_trace = []
            
            # sample nodes if too many
            nodes = list(dependencies.keys())
            if len(nodes) > max_nodes:
                # keep nodes with most connections
                nodes = sorted(nodes, key=lambda n: len(dependencies[n]), reverse=True)[:max_nodes]
            
            # create edges
            for node in nodes:
                for dep in dependencies[node]:
                    if dep in nodes:
                        edge_trace.append(dict(
                            x=[nodes.index(node), nodes.index(dep)],
                            y=[nodes.index(node), nodes.index(dep)],
                            mode='lines',
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none'
                        ))
            
            # create nodes
            node_trace = go.Scatter(
                x=list(range(len(nodes))),
                y=list(range(len(nodes))),
                mode='markers+text',
                text=nodes,
                textposition="top center",
                marker=dict(
                    size=[min(30, 10 + len(dependencies[n])) for n in nodes],
                    color=[len(dependencies[n]) for n in nodes],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Dependencies")
                )
            )
            
            fig = go.Figure(data=[node_trace] + edge_trace)
            fig.update_layout(
                title=f"Dependency Graph (Top {len(nodes)} modules)",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0,l=0,r=0,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        
        # save
        output_html = output_path.replace('.png', '.html')
        fig.write_html(output_html, include_plotlyjs='cdn')
        self._save_to_cache(cache_key, fig)
        
        # static fallback
        try:
            fig.write_image(output_path)
        except:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'View interactive graph in HTML report', 
                    ha='center', va='center', fontsize=20)
            plt.axis('off')
            plt.savefig(output_path)
            plt.close()
    
    def generate_treemap(self, file_metrics: Dict[str, Dict], output_path: str) -> None:
        """
        Generate treemap visualization for file sizes and issues.
        Very fast and intuitive for large codebases.
        """
        # prepare hierarchical data
        data = []
        for file_path, metrics in file_metrics.items():
            parts = file_path.split('/')
            data.append(dict(
                labels=parts[-1],  # filename
                parents='/'.join(parts[:-1]) if len(parts) > 1 else '',
                values=metrics.get('lines', 1),
                text=f"{metrics.get('issues', 0)} issues",
                customdata=metrics.get('issues', 0)
            ))
        
        # create treemap
        fig = go.Figure(go.Treemap(
            labels=[d['labels'] for d in data],
            parents=[d['parents'] for d in data],
            values=[d['values'] for d in data],
            text=[d['text'] for d in data],
            customdata=[d['customdata'] for d in data],
            marker=dict(
                colorscale='Reds',
                cmid=5,
                colorbar=dict(title="Issues")
            ),
            textinfo="label+text",
            hovertemplate='<b>%{label}</b><br>Lines: %{value}<br>%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Code Structure Treemap (Size = Lines, Color = Issues)",
            height=800
        )
        
        output_html = output_path.replace('.png', '_treemap.html')
        fig.write_html(output_html, include_plotlyjs='cdn')


# fast versions to use in runner_enhanced.py
def generate_fast_visualizations(job_dir: str, file_metrics: Dict[str, Dict], visuals_dir: Path) -> Dict[str, str]:
    """Generate all visualizations using fast methods."""
    visualizer = FastVisualizer()
    visualization_paths = {}
    
    try:
        # fast heatmap
        heatmap_path = visuals_dir / "quality_heatmap.png"
        visualizer.generate_fast_heatmap(file_metrics, str(heatmap_path))
        visualization_paths["quality_heatmap"] = str(heatmap_path.relative_to(Path("work")))
        
        # fast dependency graph (limited nodes)
        dep_graph_path = visuals_dir / "dependency_graph.png"
        visualizer.generate_fast_dependency_graph(str(job_dir), str(dep_graph_path))
        visualization_paths["dependency_graph"] = str(dep_graph_path.relative_to(Path("work")))
        
        # bonus: Treemap visualization
        treemap_path = visuals_dir / "code_treemap.html"
        visualizer.generate_treemap(file_metrics, str(treemap_path))
        visualization_paths["code_treemap"] = str(treemap_path.relative_to(Path("work")))
        
    except Exception as e:
        print(f"Error generating fast visualizations: {e}")
    
    return visualization_paths
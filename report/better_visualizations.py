import os
import json
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class EnhancedVisualizer:
    """Enhanced visualization generator with better dependency tracking."""
    
    def __init__(self):
        self.internal_modules = set()
        self.external_modules = set()
        
    def _extract_dependencies(self, repo_path: str) -> Dict[str, Set[str]]:
        """Extract dependencies with better module resolution."""
        dependencies = defaultdict(set)
        repo_path = Path(repo_path)
        
        # get all Python modules in the repo (internal modules)
        for py_file in repo_path.rglob("*.py"):
            if any(part in str(py_file) for part in ['.venv', 'venv', '__pycache__', 'node_modules']):
                continue
            # convert file path to module name
            rel_path = py_file.relative_to(repo_path)
            module_name = str(rel_path.with_suffix('')).replace(os.sep, '.')
            self.internal_modules.add(module_name)
        
        # now extract imports
        for py_file in repo_path.rglob("*.py"):
            if any(part in str(py_file) for part in ['.venv', 'venv', '__pycache__', 'node_modules']):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                rel_path = py_file.relative_to(repo_path)
                source_module = str(rel_path.with_suffix('')).replace(os.sep, '.')
                
                for line in content.split('\n'):
                    line = line.strip()
                    
                    # handle different import styles
                    if line.startswith('import '):
                        # import module1, module2
                        modules = line[7:].split(',')
                        for mod in modules:
                            mod = mod.strip().split(' as ')[0]  # Remove aliases
                            if mod and not mod.startswith('_'):
                                dependencies[source_module].add(mod.split('.')[0])
                                
                    elif line.startswith('from '):
                        # from module import ...
                        parts = line.split(' import ')
                        if len(parts) >= 2:
                            mod = parts[0][5:].strip()
                            if mod and not mod.startswith(('_', '.')):
                                base_mod = mod.split('.')[0]
                                dependencies[source_module].add(base_mod)
                                
            except Exception:
                continue
        
        # classify modules as internal or external
        for deps in dependencies.values():
            for dep in deps:
                if not any(dep.startswith(int_mod.split('.')[0]) for int_mod in self.internal_modules):
                    self.external_modules.add(dep)
        
        return dependencies
    
    def generate_enhanced_dependency_graph(self, repo_path: str, output_path: str, max_nodes: int = 100):
        """Generate an enhanced dependency graph with better layout and clarity."""
        dependencies = self._extract_dependencies(repo_path)
        
        if not dependencies:
            # create placeholder
            fig = go.Figure()
            fig.add_annotation(
                text="No dependencies found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
        else:
            # create NetworkX graph for better layout
            G = nx.DiGraph()
            
            # add nodes and edges
            for source, targets in dependencies.items():
                # only show internal dependencies by default
                if source in self.internal_modules or any(source.startswith(m) for m in self.internal_modules):
                    for target in targets:
                        # skip self-references
                        if source != target:
                            # separate internal and external dependencies
                            if target in self.internal_modules or any(target.startswith(m) for m in self.internal_modules):
                                G.add_edge(source, target, type='internal')
                            else:
                                G.add_edge(source, target, type='external')
            
            # limit nodes if too many
            if len(G.nodes()) > max_nodes:
                # keep most connected nodes
                degree_dict = dict(G.degree())
                top_nodes = sorted(degree_dict.keys(), key=lambda x: degree_dict[x], reverse=True)[:max_nodes]
                G = G.subgraph(top_nodes).copy()
            
            # use spring layout for better visualization
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # create Plotly traces
            edge_traces = []
            
            # internal edges (blue)
            internal_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'internal']
            if internal_edges:
                edge_x = []
                edge_y = []
                for edge in internal_edges:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                edge_traces.append(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#007ACC'),
                    hoverinfo='none',
                    mode='lines',
                    name='Internal Dependencies'
                ))
            
            # external edges (gray, dashed)
            external_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'external']
            if external_edges:
                edge_x = []
                edge_y = []
                for edge in external_edges:
                    if edge[0] in pos:  # Check if node exists in layout
                        x0, y0 = pos[edge[0]]
                        # external modules might not have positions
                        if edge[1] in pos:
                            x1, y1 = pos[edge[1]]
                        else:
                            # place external modules on the edge
                            x1, y1 = x0 + 0.2, y0 + 0.2
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                
                edge_traces.append(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888', dash='dash'),
                    hoverinfo='none',
                    mode='lines',
                    name='External Dependencies'
                ))
            
            # node trace
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            
            for node in G.nodes():
                if node in pos:
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    
                    # color based on type
                    if node in self.external_modules:
                        node_color.append('#FF6B6B')  # Red for external
                        node_text.append(f"{node} (external)")
                    else:
                        node_color.append('#4ECDC4')  # Teal for internal
                        in_degree = G.in_degree(node)
                        out_degree = G.out_degree(node)
                        node_text.append(f"{node}<br>In: {in_degree}, Out: {out_degree}")
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                hovertext=node_text,
                textposition="top center",
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    size=10,
                    color=node_color,
                    colorbar=dict(
                        thickness=15,
                        title=dict(text="Module Type", side="right"),
                        xanchor="left"
                    )
                )
            )
            
            # create figure
            fig = go.Figure(data=edge_traces + [node_trace])
            
            fig.update_layout(
                title=dict(text=f"Module Dependency Graph ({len(G.nodes())} modules)", font=dict(size=16)),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
        
        # save as HTML
        output_html = output_path.replace('.png', '.html')
        fig.write_html(output_html, include_plotlyjs='cdn')
        
        # try to save as PNG
        try:
            fig.write_image(output_path, width=1200, height=800)
        except:
            # create a matplotlib fallback
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            if dependencies:
                # simple matplotlib visualization
                nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                       node_size=500, font_size=8, arrows=True)
                plt.title("Dependency Graph")
            else:
                plt.text(0.5, 0.5, 'No dependencies found', 
                        ha='center', va='center', fontsize=20)
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    def generate_enhanced_heatmap(self, file_metrics: Dict[str, Dict], output_path: str):
        """Generate an enhanced heatmap with better visualization."""
        if not file_metrics:
            # create empty visualization
            fig = go.Figure()
            fig.add_annotation(
                text="No files analyzed",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
        else:
            # prepare data - limit to top 100 files by issues
            sorted_files = sorted(
                file_metrics.items(),
                key=lambda x: x[1].get('issues', 0),
                reverse=True
            )[:100]
            
            # create treemap for better space utilization
            labels = []
            parents = []
            values = []
            colors = []
            text_info = []
            
            # group by directory
            dir_issues = defaultdict(int)
            for file_path, metrics in sorted_files:
                parts = file_path.split('/')
                if len(parts) > 1:
                    dir_name = parts[0]
                    dir_issues[dir_name] += metrics.get('issues', 0)
            
            # add directory nodes
            for dir_name, total_issues in dir_issues.items():
                labels.append(dir_name)
                parents.append("")
                values.append(1)  # Equal size for directories
                colors.append(total_issues)
                text_info.append(f"{total_issues} issues")
            
            # add file nodes
            for file_path, metrics in sorted_files:
                parts = file_path.split('/')
                parent = parts[0] if len(parts) > 1 else ""
                file_name = parts[-1]
                
                labels.append(file_name)
                parents.append(parent)
                values.append(metrics.get('lines', 1))
                colors.append(metrics.get('issues', 0))
                text_info.append(f"{metrics.get('issues', 0)} issues, {metrics.get('lines', 0)} lines")
            
            # create treemap
            fig = go.Figure(go.Treemap(
                labels=labels,
                parents=parents,
                values=values,
                text=text_info,
                textinfo="label+text",
                marker=dict(
                    colorscale='Reds',
                    cmid=5,
                    line=dict(width=2, color='white'),
                    colorbar=dict(title="Issues")
                ),
                marker_colorscale='Reds',
                marker_line_width=2,
                marker_line_color='white',
                marker_colors=colors,
                hovertemplate='<b>%{label}</b><br>%{text}<br>Size: %{value}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Code Quality Heatmap (Size = Lines of Code, Color = Issues)",
                margin=dict(t=50, l=0, r=0, b=0),
                height=800
            )
        
        # save as HTML
        output_html = output_path.replace('.png', '.html')
        fig.write_html(output_html, include_plotlyjs='cdn')
        
        # also create a bar chart version for PNG
        if file_metrics:
            # create horizontal bar chart for top 50 files
            top_files = sorted_files[:50]
            files = [f[0] for f in top_files]
            issues = [f[1].get('issues', 0) for f in top_files]
            
            fig_bar = go.Figure(go.Bar(
                y=files,
                x=issues,
                orientation='h',
                marker=dict(
                    color=issues,
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Issues")
                ),
                text=issues,
                textposition='outside'
            ))
            
            fig_bar.update_layout(
                title="Top 50 Files by Issue Count",
                xaxis_title="Number of Issues",
                yaxis_title="Files",
                height=max(600, len(files) * 20),
                margin=dict(l=200)
            )
            
            try:
                fig_bar.write_image(output_path, width=1200, height=max(800, len(files) * 20))
            except:
                # matplotlib fallback
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, max(8, len(files) * 0.3)))
                plt.barh(files, issues, color='red', alpha=0.7)
                plt.xlabel('Number of Issues')
                plt.title('Top Files by Issue Count')
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()


def generate_better_visualizations(job_dir: str, file_metrics: Dict[str, Dict], visuals_dir: Path) -> Dict[str, str]:
    """Generate enhanced visualizations."""
    visualizer = EnhancedVisualizer()
    visualization_paths = {}
    
    try:
        # enhanced dependency graph
        dep_graph_path = visuals_dir / "dependency_graph.png"
        visualizer.generate_enhanced_dependency_graph(str(job_dir), str(dep_graph_path))
        visualization_paths["dependency_graph"] = str(dep_graph_path.relative_to(Path("work")))
        
        # the enhanced dependency graph method already creates an HTML version
        dep_network_path = visuals_dir / "dependency_graph.html"
        if dep_network_path.exists():
            visualization_paths["dependency_network"] = str(dep_network_path.relative_to(Path("work")))
        
        # enhanced heatmap
        heatmap_path = visuals_dir / "quality_heatmap.png"
        visualizer.generate_enhanced_heatmap(file_metrics, str(heatmap_path))
        visualization_paths["quality_heatmap"] = str(heatmap_path.relative_to(Path("work")))
        
    except Exception as e:
        print(f"Error generating enhanced visualizations: {e}")
    
    return visualization_paths
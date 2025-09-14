# add these imports to the existing imports in generator.py
from pathlib import Path
from report.visualizations import (
    build_call_graph,
    save_dependency_graph_png,
    compute_file_metrics,
    generate_hotspot_heatmap,
    plot_trend_chart,
    embed_visualizations_in_markdown,
    save_metrics_to_history,
    render_dependency_network_html
)


# add this method to the ReportGenerator class
def generate_enhanced_report(self, issues: List[Dict[str, Any]], 
                           output_path: str,
                           repo_path: str = None,
                           job_id: str = None) -> str:
    """
    Generate a comprehensive markdown report with visualizations.
    
    Args:
        issues: List of issues from all analyzers
        output_path: Path to save the report
        repo_path: Path to the repository (for generating visualizations)
        job_id: Unique job ID for organizing visualizations
        
    Returns:
        Path to the generated report
    """
    # generate base report using existing method
    report_file = self.generate_report(issues, output_path)
    
    # if repo_path is provided, generate visualizations
    if repo_path and Path(repo_path).exists():
        # create visualization directory
        if job_id:
            visuals_dir = Path("./work/visuals") / job_id
        else:
            visuals_dir = Path(output_path) / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)
        
        generated_visuals = []
        
        try:
            # 1. Generate dependency graph
            print("Building function call graph...")
            call_graph = build_call_graph(repo_path)
            if len(call_graph) > 0:
                dep_graph_path = visuals_dir / "dependency_graph.png"
                save_dependency_graph_png(call_graph, str(dep_graph_path))
                generated_visuals.append(str(dep_graph_path))
                
                # also generate interactive HTML version if pyvis is available
                try:
                    html_path = visuals_dir / "dependency_network.html"
                    render_dependency_network_html(call_graph, str(html_path))
                except:
                    pass
            
            # 2. Generate code quality heatmap
            print("Computing file metrics...")
            file_metrics = compute_file_metrics(repo_path)
            if file_metrics:
                heatmap_path = visuals_dir / "quality_heatmap.png"
                generate_hotspot_heatmap(file_metrics, str(heatmap_path))
                generated_visuals.append(str(heatmap_path))
                
                # save metrics to history for trend tracking
                db_path = "./data/history.db"
                Path("./data").mkdir(exist_ok=True)
                
                # calculate aggregate metrics
                avg_complexity = sum(m.get('complexity', 0) for m in file_metrics.values()) / len(file_metrics) if file_metrics else 0
                total_issues = sum(m.get('num_issues', 0) for m in file_metrics.values())
                
                save_metrics_to_history(
                    db_path,
                    self.project_name,
                    {
                        "average_complexity": avg_complexity,
                        "total_issues": total_issues,
                        "total_files": len(file_metrics)
                    }
                )
                
                # generate trend chart if history exists
                trend_path = visuals_dir / "complexity_trend.png"
                plot_trend_chart(db_path, "average_complexity", str(trend_path), self.project_name)
                if trend_path.exists():
                    generated_visuals.append(str(trend_path))
            
            # 3. Embed visualizations in the markdown report
            if generated_visuals:
                print(f"Embedding {len(generated_visuals)} visualizations...")
                embed_visualizations_in_markdown(report_file, generated_visuals)
        
        except Exception as e:
            print(f"Warning: Could not generate some visualizations: {e}")
    
    return report_file


# update the _generate_visualization_placeholders method to reference actual visualizations
def _generate_visualization_section(self) -> str:
    """Generate visualization section with actual image references."""
    return """## 📊 Visualizations

The following visualizations provide additional insights into the codebase:

### Function Dependency Graph
Shows the relationships between functions and modules in the codebase.

### Code Quality Heatmap
Highlights files with high complexity and issue density.

### Historical Trends
Tracks code quality metrics over time (when available).

*Note: Visualizations are generated when a repository path is provided during analysis.*"""


# example usage showing integration
if __name__ == "__main__":
    import tempfile
    import shutil
    
    # create a sample repository structure for demonstration
    with tempfile.TemporaryDirectory() as temp_repo:
        # create sample Python files
        sample_py = Path(temp_repo) / "sample.py"
        sample_py.write_text('''
def complex_function(a, b, c):
    if a > 0:
        if b > 0:
            if c > 0:
                return helper_function(a, b, c)
    return 0

def helper_function(x, y, z):
    return x + y + z

class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return self.add(a, 0) * b
''')
        
        # sample issues
        sample_issues = [
            {
                "file": "sample.py",
                "line": 3,
                "severity": "high",
                "type": "complexity",
                "description": "Cyclomatic complexity is too high",
                "analyzer": "custom"
            }
        ]
        
        # create generator with enhanced report
        generator = ReportGenerator("Demo Project")
        
        # generate report with visualizations
        with tempfile.TemporaryDirectory() as output_dir:
            # add the enhanced method to the generator instance
            generator.generate_enhanced_report = generate_enhanced_report.__get__(generator, ReportGenerator)
            
            report_path = generator.generate_enhanced_report(
                sample_issues,
                output_dir,
                repo_path=temp_repo,
                job_id="demo_job"
            )
            
            print(f"Enhanced report generated: {report_path}")
            
            # check if visualizations were created
            visuals_dir = Path("./work/visuals/demo_job")
            if visuals_dir.exists():
                visuals = list(visuals_dir.glob("*.png"))
                print(f"Generated {len(visuals)} visualizations")
                
                # clean up
                shutil.rmtree("./work/visuals/demo_job")
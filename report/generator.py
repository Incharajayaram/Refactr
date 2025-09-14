import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import hashlib


class ReportGenerator:
    """Generates markdown reports from code quality analysis results."""
    
    def __init__(self, project_name: str = "Project"):
        """
        Initialize the report generator.
        
        Args:
            project_name: Name of the project being analyzed
        """
        self.project_name = project_name
        self.timestamp = datetime.now()
        
        # define severity levels and their order
        self.severity_levels = ["critical", "high", "medium", "low", "info"]
        self.severity_colors = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
            "info": "🔵"
        }
        
        # define issue type categories
        self.issue_categories = {
            "security": ["security", "vulnerability", "injection", "exposure"],
            "performance": ["performance", "efficiency", "optimization", "memory"],
            "maintainability": ["complexity", "maintainability", "readability", "duplication"],
            "style": ["style", "convention", "formatting", "naming"],
            "bug": ["bug", "error", "exception", "null", "undefined"],
            "documentation": ["documentation", "docstring", "comment"],
            "other": []
        }
    
    def generate_report(self, issues: List[Dict[str, Any]], output_path: str) -> str:
        """
        Generate a comprehensive markdown report from analysis results.
        
        Args:
            issues: List of issues from all analyzers
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        # process and group issues
        grouped_issues = self._group_issues(issues)
        statistics = self._calculate_statistics(issues, grouped_issues)
        
        # generate report sections
        report_content = []
        report_content.append(self._generate_header())
        report_content.append(self._generate_executive_summary(statistics))
        report_content.append(self._generate_summary_table(statistics))
        report_content.append(self._generate_severity_breakdown(statistics))
        report_content.append(self._generate_category_breakdown(statistics))
        report_content.append(self._generate_analyzer_breakdown(statistics))
        report_content.append(self._generate_file_analysis(grouped_issues))
        report_content.append(self._generate_detailed_findings(grouped_issues))
        report_content.append(self._generate_recommendations(statistics, grouped_issues))
        report_content.append(self._generate_visualization_placeholders())
        report_content.append(self._generate_footer())
        
        # write report to file
        report_text = "\n\n".join(report_content)
        report_file = os.path.join(output_path, f"code_quality_report_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_file
    
    def _group_issues(self, issues: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict]]]:
        """Group issues by file and severity."""
        grouped = defaultdict(lambda: defaultdict(list))
        
        for issue in issues:
            file_path = issue.get('file', 'unknown')
            severity = issue.get('severity', 'info').lower()
            grouped[file_path][severity].append(issue)
        
        return dict(grouped)
    
    def _calculate_statistics(self, issues: List[Dict[str, Any]], 
                            grouped_issues: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, Any]:
        """Calculate comprehensive statistics from issues."""
        stats = {
            'total_issues': len(issues),
            'total_files': len(grouped_issues),
            'severity_counts': defaultdict(int),
            'category_counts': defaultdict(int),
            'analyzer_counts': defaultdict(int),
            'file_issue_counts': {},
            'critical_files': [],
            'most_common_issues': defaultdict(int)
        }
        
        # count by severity
        for issue in issues:
            severity = issue.get('severity', 'info').lower()
            stats['severity_counts'][severity] += 1
            
            # count by analyzer
            analyzer = issue.get('analyzer', 'unknown')
            stats['analyzer_counts'][analyzer] += 1
            
            # count by category
            category = self._categorize_issue(issue)
            stats['category_counts'][category] += 1
            
            # track issue types
            issue_type = issue.get('type', 'unknown')
            stats['most_common_issues'][issue_type] += 1
        
        # file-level statistics
        for file_path, severities in grouped_issues.items():
            total_in_file = sum(len(issues) for issues in severities.values())
            stats['file_issue_counts'][file_path] = total_in_file
            
            # identify critical files
            if severities.get('critical') or severities.get('high'):
                critical_count = len(severities.get('critical', [])) + len(severities.get('high', []))
                stats['critical_files'].append((file_path, critical_count))
        
        # sort critical files by issue count
        stats['critical_files'].sort(key=lambda x: x[1], reverse=True)
        
        return stats
    
    def _categorize_issue(self, issue: Dict[str, Any]) -> str:
        """Categorize an issue based on its type and description."""
        issue_type = issue.get('type', '').lower()
        description = issue.get('description', '').lower()
        
        for category, keywords in self.issue_categories.items():
            if any(keyword in issue_type or keyword in description for keyword in keywords):
                return category
        
        return 'other'
    
    def _generate_header(self) -> str:
        """Generate report header."""
        return f"""# Code Quality Report - {self.project_name}

**Generated on:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}  
**Report Version:** 1.0.0

---"""
    
    def _generate_executive_summary(self, stats: Dict[str, Any]) -> str:
        """Generate executive summary section."""
        total_issues = stats['total_issues']
        critical_high = stats['severity_counts']['critical'] + stats['severity_counts']['high']
        
        summary_status = "✅ Good" if critical_high == 0 else "⚠️ Needs Attention" if critical_high < 10 else "❌ Critical"
        
        return f"""## Executive Summary

**Overall Status:** {summary_status}

This report presents a comprehensive analysis of code quality for **{self.project_name}**. 
The analysis identified **{total_issues}** total issues across **{stats['total_files']}** files.

### Key Findings:
- **{critical_high}** critical/high severity issues requiring immediate attention
- **{stats['severity_counts']['medium']}** medium severity issues for planned resolution
- **{len(stats['critical_files'])}** files identified as high-priority for remediation
- Most common issue category: **{max(stats['category_counts'], key=stats['category_counts'].get) if stats['category_counts'] else 'N/A'}**"""
    
    def _generate_summary_table(self, stats: Dict[str, Any]) -> str:
        """Generate summary statistics table."""
        return f"""## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Issues | {stats['total_issues']} |
| Files Analyzed | {stats['total_files']} |
| Critical Issues | {stats['severity_counts']['critical']} |
| High Severity | {stats['severity_counts']['high']} |
| Medium Severity | {stats['severity_counts']['medium']} |
| Low Severity | {stats['severity_counts']['low']} |
| Informational | {stats['severity_counts']['info']} |"""
    
    def _generate_severity_breakdown(self, stats: Dict[str, Any]) -> str:
        """Generate severity breakdown section."""
        content = ["## Issue Severity Distribution"]
        
        for severity in self.severity_levels:
            count = stats['severity_counts'].get(severity, 0)
            if count > 0:
                percentage = (count / stats['total_issues'] * 100) if stats['total_issues'] > 0 else 0
                bar_length = int(percentage / 2)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                
                content.append(f"""
### {self.severity_colors[severity]} {severity.capitalize()} - {count} issues ({percentage:.1f}%)
{bar}""")
        
        return "\n".join(content)
    
    def _generate_category_breakdown(self, stats: Dict[str, Any]) -> str:
        """Generate issue category breakdown."""
        content = ["## Issue Categories"]
        
        if not stats['category_counts']:
            return content[0] + "\n\nNo issues to categorize."
        
        content.append("\n| Category | Count | Percentage |")
        content.append("|----------|--------|------------|")
        
        total = stats['total_issues']
        for category, count in sorted(stats['category_counts'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            content.append(f"| {category.capitalize()} | {count} | {percentage:.1f}% |")
        
        return "\n".join(content)
    
    def _generate_analyzer_breakdown(self, stats: Dict[str, Any]) -> str:
        """Generate analyzer breakdown section."""
        content = ["## Issues by Analyzer"]
        
        if not stats['analyzer_counts']:
            return content[0] + "\n\nNo analyzer data available."
        
        content.append("\n| Analyzer | Issues Found |")
        content.append("|----------|--------------|")
        
        for analyzer, count in sorted(stats['analyzer_counts'].items(), key=lambda x: x[1], reverse=True):
            content.append(f"| {analyzer} | {count} |")
        
        return "\n".join(content)
    
    def _generate_file_analysis(self, grouped_issues: Dict[str, Dict[str, List[Dict]]]) -> str:
        """Generate file-level analysis section."""
        content = ["## File Analysis"]
        
        if not grouped_issues:
            return content[0] + "\n\nNo issues found in any files."
        
        # sort files by total issue count
        file_stats = []
        for file_path, severities in grouped_issues.items():
            total = sum(len(issues) for issues in severities.values())
            critical_high = len(severities.get('critical', [])) + len(severities.get('high', []))
            file_stats.append((file_path, total, critical_high))
        
        file_stats.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        content.append("\n### Most Problematic Files")
        content.append("\n| File | Total Issues | Critical/High |")
        content.append("|------|--------------|---------------|")
        
        for file_path, total, critical_high in file_stats[:10]:  # Top 10 files
            file_name = os.path.basename(file_path)
            content.append(f"| {file_name} | {total} | {critical_high} |")
        
        return "\n".join(content)
    
    def _generate_detailed_findings(self, grouped_issues: Dict[str, Dict[str, List[Dict]]]) -> str:
        """Generate detailed findings section."""
        content = ["## Detailed Findings"]
        
        if not grouped_issues:
            return content[0] + "\n\nNo issues to report."
        
        # sort files by critical issues first, then total issues
        sorted_files = sorted(
            grouped_issues.items(),
            key=lambda x: (
                len(x[1].get('critical', [])) + len(x[1].get('high', [])),
                sum(len(issues) for issues in x[1].values())
            ),
            reverse=True
        )
        
        for file_path, severities in sorted_files:
            content.append(f"\n### 📄 {file_path}")
            
            # add file summary
            total_in_file = sum(len(issues) for issues in severities.values())
            content.append(f"\n**Total issues in file:** {total_in_file}")
            
            # group by severity
            for severity in self.severity_levels:
                if severity in severities and severities[severity]:
                    content.append(f"\n#### {self.severity_colors[severity]} {severity.capitalize()} Issues ({len(severities[severity])})")
                    
                    for issue in severities[severity]:
                        line = issue.get('line', 'N/A')
                        issue_type = issue.get('type', 'Unknown')
                        description = issue.get('description', 'No description')
                        analyzer = issue.get('analyzer', 'unknown')
                        
                        content.append(f"""
- **Line {line}**: {issue_type}
  - {description}
  - *Source: {analyzer}*""")
        
        return "\n".join(content)
    
    def _generate_recommendations(self, stats: Dict[str, Any], grouped_issues: Dict[str, Dict[str, List[Dict]]]) -> str:
        """Generate recommendations section."""
        content = ["## Recommendations"]
        
        # priority recommendations based on severity
        if stats['severity_counts']['critical'] > 0:
            content.append(f"""
### 🔴 Immediate Actions Required
1. Address **{stats['severity_counts']['critical']}** critical issues immediately
2. Review and fix security vulnerabilities in the following files:""")
            
            for file_path, count in stats['critical_files'][:5]:
                content.append(f"   - {file_path} ({count} critical/high issues)")
        
        if stats['severity_counts']['high'] > 0:
            content.append(f"""
### 🟠 High Priority
1. Resolve **{stats['severity_counts']['high']}** high severity issues
2. Focus on files with multiple high-severity issues
3. Consider implementing automated security scanning in CI/CD pipeline""")
        
        # category-specific recommendations
        top_category = max(stats['category_counts'], key=stats['category_counts'].get) if stats['category_counts'] else None
        
        if top_category:
            content.append(f"\n### 📊 Category-Specific Recommendations")
            
            if top_category == "security":
                content.append("""
- Implement security code reviews
- Use static security analysis tools
- Regular dependency vulnerability scanning""")
            elif top_category == "performance":
                content.append("""
- Profile application for bottlenecks
- Optimize database queries
- Implement caching strategies""")
            elif top_category == "maintainability":
                content.append("""
- Refactor complex functions
- Improve code documentation
- Establish coding standards""")
        
        # general recommendations
        content.append("""
### 📋 General Best Practices
1. Establish code review process for all changes
2. Implement pre-commit hooks for code quality checks
3. Set up continuous integration with quality gates
4. Regular team training on secure coding practices
5. Maintain up-to-date documentation""")
        
        return "\n".join(content)
    
    def _generate_visualization_placeholders(self) -> str:
        """Generate placeholders for future visualizations."""
        return """## Visualizations

### 📊 Severity Distribution Chart
*[Placeholder for severity distribution pie chart]*

### 📈 Issues Trend Over Time
*[Placeholder for historical trend line graph]*

### 🗂️ Issue Heatmap by File
*[Placeholder for file heatmap visualization]*

### 🔄 Issue Category Network
*[Placeholder for category relationship diagram]*

> Note: Visualizations will be available in future versions of this report."""
    
    def _generate_footer(self) -> str:
        """Generate report footer."""
        return f"""---

## Report Metadata

- **Generated by:** Code Quality Intelligence Agent
- **Report ID:** {hashlib.md5(f"{self.project_name}_{self.timestamp}".encode()).hexdigest()[:8]}
- **Analysis Date:** {self.timestamp.strftime('%Y-%m-%d')}
- **Report Format:** Markdown v1.0

---

*This report is automatically generated. For questions or improvements, please contact the development team.*"""


def generate_report(issues: List[Dict[str, Any]], 
                   output_dir: str = "reports",
                   project_name: str = "Project") -> str:
    """
    Convenience function to generate a report.
    
    Args:
        issues: List of issues from analyzers
        output_dir: Directory to save the report
        project_name: Name of the project
        
    Returns:
        Path to the generated report
    """
    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # create generator and generate report
    generator = ReportGenerator(project_name)
    return generator.generate_report(issues, output_dir)


# example usage
if __name__ == "__main__":
    # sample data for testing
    sample_issues = [
        {
            "file": "src/main.py",
            "line": 42,
            "severity": "critical",
            "type": "security vulnerability",
            "description": "SQL injection vulnerability detected",
            "analyzer": "bandit"
        },
        {
            "file": "src/main.py",
            "line": 156,
            "severity": "high",
            "type": "complexity",
            "description": "Function complexity is too high (15 > 10)",
            "analyzer": "pylint"
        },
        {
            "file": "src/utils.py",
            "line": 23,
            "severity": "medium",
            "type": "style",
            "description": "Line too long (120 > 79 characters)",
            "analyzer": "flake8"
        },
        {
            "file": "src/utils.py",
            "line": 45,
            "severity": "low",
            "type": "naming convention",
            "description": "Variable name 'x' is too short",
            "analyzer": "pylint"
        }
    ]
    
    # generate report
    report_path = generate_report(sample_issues, "sample_reports", "Sample Project")
    print(f"Report generated: {report_path}")
# Refactr - AI-Powered Code Quality Analysis Agent

Refactr is an intelligent code quality analysis tool that leverages AI to provide comprehensive insights into your codebase. It supports multiple programming languages and offers both CLI and web interfaces for seamless integration into your development workflow.

## 🚀 Features

- **Multi-Language Support**: Analyzes Python and JavaScript/TypeScript codebases
- **AI-Powered Q&A**: Interactive chat interface using Claude AI for code understanding
- **Comprehensive Analysis**: 
  - Code complexity metrics
  - Security vulnerability detection
  - Best practices validation
  - Dependency analysis
- **Multiple Interfaces**:
  - Command-line interface (CLI)
  - Web application with RESTful API
  - Interactive visualizations
- **Report Generation**: Multiple formats (JSON, HTML, PDF)
- **Real-time Analysis**: Webhook support for GitHub/GitLab integration

## 📋 Prerequisites

- Python 3.8 or higher
- Git
- 4GB+ RAM recommended for large repositories

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd code_quality_agent
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements-web.txt  # For web interface
```

4. **Configure environment**
```bash
cp .env.example .env
```

Edit `.env` and add your API key:
```
ANTHROPIC_API_KEY=your_claude_api_key_here
```

## 🎯 Usage

### CLI Interface

**Basic analysis:**
```bash
python main.py analyze https://github.com/user/repo
```

**With options:**
```bash
python main.py analyze https://github.com/user/repo \
  --branch main \
  --output-format html \
  --verbose
```

**Local repository:**
```bash
python main.py analyze /path/to/local/repo
```

### Web Interface

1. **Start the server:**
```bash
uvicorn webapp.app:app --reload
```

2. **Access the UI:**
Open http://localhost:8000 in your browser

3. **API endpoints:**
- `POST /api/analyze` - Submit repository for analysis
- `GET /api/jobs/{job_id}` - Check job status
- `GET /api/report/{job_id}` - Get analysis report
- `POST /api/qa/query` - Query the codebase

### Interactive Q&A

```bash
python qa_interface.py
```

Example questions:
- "How does the authentication system work?"
- "What are the main security concerns in this code?"
- "Explain the data flow in the payment module"

## 📊 Analysis Metrics

Refactr analyzes your code across multiple dimensions:

### Code Quality
- **Cyclomatic Complexity**: Measures code complexity
- **Maintainability Index**: Evaluates code maintainability
- **Code Duplication**: Detects repeated code blocks
- **Test Coverage**: Identifies untested code paths

### Security
- **SQL Injection vulnerabilities**
- **XSS vulnerabilities**
- **Hardcoded credentials**
- **Insecure dependencies**

### Best Practices
- **Naming conventions**
- **Function length**
- **File organization**
- **Documentation coverage**

## 🎨 Visualizations

Refactr generates interactive visualizations:
- **Dependency Graph**: Module relationships
- **Quality Heatmap**: File-level quality metrics
- **Issue Distribution**: Issues by type and severity

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Claude AI API key | Required |
| `GITHUB_TOKEN` | GitHub access token | Optional |
| `KEEP_DAYS` | Days to retain analysis | 7 |
| `ANTHROPIC_MODEL` | Claude model to use | claude-3-haiku-20240307 |

### Analysis Rules

Customize analysis rules in `analyzers/rules/`:
- `python_rules.yaml` - Python-specific rules
- `js_rules.yaml` - JavaScript rules
- `security_rules.yaml` - Security patterns

## 🏗️ Architecture

```
refactr/
├── analyzers/          # Language-specific analyzers
├── core/               # Core analysis engine
├── webapp/             # Web application
├── retriever/          # RAG system for Q&A
├── report/             # Report generation
└── visualizations/     # Chart generation
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Anthropic Claude for AI capabilities
- FastAPI for web framework
- Plotly for visualizations
- Radon for code metrics

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check documentation at `/docs`
- Email: support@refactr.ai
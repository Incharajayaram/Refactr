# Code Quality Intelligence Agent - Web API

A FastAPI-based web service for analyzing GitHub repositories and providing code quality insights.

## Features

- 🚀 **RESTful API** for submitting analysis jobs
- 📊 **Background Processing** for long-running analyses
- 🔍 **Job Status Tracking** with real-time updates
- 🌐 **Web UI** for easy repository submission
- 🔒 **Support for Private Repositories** via GitHub tokens
- 📝 **Detailed JSON Reports** with code metrics and recommendations
- 🧹 **Automatic Cleanup** of old job data

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r webapp/requirements.txt
   ```

2. **Set environment variables (optional):**
   ```bash
   export GITHUB_TOKEN=your_github_token  # For private repos
   export ALLOW_OTHER_DOMAINS=true        # To allow non-GitHub URLs
   export KEEP_DAYS=7                     # Days to keep job data
   ```

## Running the Server

Start the FastAPI server with uvicorn:

```bash
uvicorn webapp.app:app --reload --port 8080
```

The server will be available at `http://localhost:8080`

## API Endpoints

### 1. Submit Analysis Job

**POST** `/api/analyze`

Submit a repository for analysis.

```bash
curl -X POST http://localhost:8080/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "repo_url": "https://github.com/owner/repo",
    "branch": "main",
    "token": "ghp_your_token_here"
  }'
```

**Request Body:**
```json
{
  "repo_url": "https://github.com/owner/repo",
  "branch": "main",              // Optional, defaults to "main"
  "token": "ghp_your_token"      // Optional, for private repos
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Analysis job queued successfully"
}
```

### 2. Check Job Status

**GET** `/api/jobs/{job_id}`

Get the current status of an analysis job.

```bash
curl http://localhost:8080/api/jobs/550e8400-e29b-41d4-a716-446655440000
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",  // pending|running|done|failed
  "started_at": "2024-01-15T10:30:00.000000",
  "finished_at": null,
  "error": null
}
```

### 3. Get Analysis Report

**GET** `/api/report/{job_id}`

Retrieve the analysis report for a completed job.

```bash
curl http://localhost:8080/api/report/550e8400-e29b-41d4-a716-446655440000
```

**Response (Example):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "repository_url": "https://github.com/example/repo",
  "branch": "main",
  "analysis_timestamp": "2024-01-15T10:35:00.000000",
  "analysis_results": {
    "total_files_analyzed": 156,
    "total_issues": 23,
    "languages": {
      "Python": {"files": 89, "lines": 12450, "issues": 15},
      "JavaScript": {"files": 67, "lines": 8930, "issues": 8}
    },
    "metrics": {
      "complexity": {
        "average": 4.2,
        "high_complexity_functions": [...]
      },
      "security": {
        "total_vulnerabilities": 3,
        "vulnerabilities": [...]
      }
    },
    "recommendations": [...]
  },
  "summary": "Analysis of 156 files revealed 23 issues..."
}
```

### 4. Health Check

**GET** `/health`

Check if the service is running.

```bash
curl http://localhost:8080/health
```

## Web UI

Access the web interface at `http://localhost:8080` to:
- Submit repositories for analysis via a user-friendly form
- Track job progress in real-time
- View and download analysis reports

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GITHUB_TOKEN` | GitHub personal access token for private repos | None |
| `ALLOW_OTHER_DOMAINS` | Allow non-GitHub repository URLs | `false` |
| `KEEP_DAYS` | Days to keep job data before cleanup | `7` |

### Security Considerations

1. **Token Security**: Never commit tokens to version control
2. **URL Validation**: Only GitHub URLs are allowed by default
3. **CORS**: Configured to allow cross-origin requests
4. **Cleanup**: Old job data is automatically removed

## Testing

Run the test suite:

```bash
pytest webapp/tests/test_app.py -v
```

Run with coverage:

```bash
pytest webapp/tests/test_app.py --cov=webapp --cov-report=html
```

## Docker Deployment

Build and run with Docker:

```bash
docker build -f docker/Dockerfile -t code-quality-agent .
docker run -p 8080:8080 -e GITHUB_TOKEN=your_token code-quality-agent
```

## Development

### Project Structure

```
webapp/
├── app.py              # FastAPI application
├── runner.py           # Background job logic
├── requirements.txt    # Python dependencies
├── tests/
│   └── test_app.py    # Unit tests
└── README_WEB.md      # This file
```

### Adding New Features

1. **New Endpoints**: Add to `app.py` with proper validation
2. **Background Tasks**: Extend `runner.py` for new analysis types
3. **Tests**: Add corresponding tests in `test_app.py`

## Troubleshooting

### Common Issues

1. **"Job not found"**: Job may have been cleaned up. Check `KEEP_DAYS` setting.
2. **"Failed to clone repository"**: Verify the repository URL and access token.
3. **"Connection refused"**: Ensure the server is running on the correct port.

### Logs

Check server logs for detailed error information:
```bash
uvicorn webapp.app:app --log-level debug
```

## API Rate Limits

- No built-in rate limiting (add nginx/reverse proxy for production)
- GitHub API has its own rate limits for cloning

## License

This project is licensed under the MIT License.
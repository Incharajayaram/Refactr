# Multi-stage Dockerfile for Code Quality Intelligence Agent Web API

# Stage 1: Base image with dependencies
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt ./
COPY webapp/requirements.txt ./webapp/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r webapp/requirements.txt

# Stage 2: Application image
FROM base as app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p work/jobs work/reports work/jobs_meta

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default environment variables (can be overridden)
ENV KEEP_DAYS=7
ENV ALLOW_OTHER_DOMAINS=false

# Run the application
CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8080"]

# Example build and run commands:
# docker build -f docker/Dockerfile -t code-quality-agent .
# docker run -p 8080:8080 -e GITHUB_TOKEN=your_token code-quality-agent

# For development with volume mounting:
# docker run -p 8080:8080 -v $(pwd):/app -e GITHUB_TOKEN=your_token code-quality-agent
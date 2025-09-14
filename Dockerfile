# Multi-stage Dockerfile for Code Quality Intelligence Agent Web API

# Stage 1: Base image with dependencies
FROM python:3.11-slim as base

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (minimal, just runtime)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt ./

# Install Python dependencies with proper build tools
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Application image
FROM base as app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p work/jobs work/reports work/jobs_meta work/visualizations work/qa_indices index

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8080

# Expose port (Render will override this)
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Default environment variables (can be overridden)
ENV KEEP_DAYS=7
ENV ALLOW_OTHER_DOMAINS=false

# Run the application (use PORT env variable)
CMD uvicorn webapp.app:app --host 0.0.0.0 --port $PORT
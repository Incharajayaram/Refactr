#!/bin/bash
# Startup script for Render deployment

echo "Starting Code Quality Agent..."

# Create necessary directories
mkdir -p work/jobs work/reports work/visualizations work/qa_indices index

# Run the web application
echo "Starting FastAPI server..."
cd webapp && uvicorn app:app --host 0.0.0.0 --port $PORT
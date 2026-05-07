# syntax=docker/dockerfile:1
# Unified Dockerfile for Hugging Face Spaces (Backend + Frontend)
FROM python:3.12-slim-bookworm AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install all dependencies at once
RUN uv pip install -r requirements.txt

# Final stage
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    INVENIOAI_API_BASE_URL=http://127.0.0.1:8000

WORKDIR /app

# Copy virtual env
COPY --from=builder /opt/venv /opt/venv

# Copy source code
COPY . .

# Create necessary directories and set permissions for Hugging Face (UID 1000)
RUN mkdir -p /app/uploaded_docs /app/qdrant_storage /app/.cache && \
    chmod -R 777 /app/uploaded_docs /app/qdrant_storage /app/.cache

# Ensure scripts are executable
RUN chmod +x start.sh

# HF Spaces usually uses port 7860
EXPOSE 7860

CMD ["./start.sh"]

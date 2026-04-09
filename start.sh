#!/usr/bin/env bash
set -euo pipefail

# Hugging Face Spaces typically provides PORT=7860.
PORT="${PORT:-7860}"

# Start FastAPI backend (internal-only).
uvicorn app.main:app --host 127.0.0.1 --port 8000 &
UVICORN_PID=$!

cleanup() {
  kill "${UVICORN_PID}" >/dev/null 2>&1 || true
  wait "${UVICORN_PID}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

# Start Streamlit (public).
exec streamlit run frontend/streamlit_app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT}" \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
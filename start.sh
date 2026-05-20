#!/bin/bash
# Unified startup script for InvenioAI (LlamaParse Version)

echo "=========================================="
echo "   🧠 InvenioAI | LlamaParse Startup"
echo "=========================================="

# 2. Activate virtual environment
if [ -d ".venv" ]; then
    echo "[INFO] Activating virtual environment..."
    source .venv/Scripts/activate
fi

# 3. Start FastAPI backend
echo "[INFO] Starting FastAPI backend..."
(cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000) &
BACKEND_PID=$!

# 4. Wait for backend to finish preloading models
echo "[INFO] Waiting for backend to finish preloading models..."
TIMEOUT=300 # 5 minutes max
ELAPSED=0
while ! python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=1)" 2>/dev/null; do
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "[ERROR] FastAPI backend crashed during startup!"
        exit 1
    fi
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "[WARNING] Backend startup timed out after ${TIMEOUT}s. Starting Streamlit anyway..."
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done
echo "[INFO] Backend check finished!"

# 5. Start Streamlit frontend
echo "[INFO] Starting Streamlit frontend..."
streamlit run frontend/streamlit_app.py --server.port 7860 --server.address 0.0.0.0

# Cleanup on exit
cleanup() {
    echo "[INFO] Shutting down..."
    kill $BACKEND_PID 2>/dev/null
    exit
}
trap cleanup SIGINT SIGTERM

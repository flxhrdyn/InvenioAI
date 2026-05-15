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

# 4. Wait for backend
echo "[INFO] Waiting for backend (5s)..."
sleep 5

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

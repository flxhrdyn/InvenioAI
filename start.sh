#!/bin/bash
# Unified startup script for InvenioAI

# 1. Start FastAPI backend in the background
echo "Starting FastAPI backend..."
# We run from the root, but the app is in backend/app. 
# PYTHONPATH must include the backend folder so 'app' can be found.
export PYTHONPATH=$PYTHONPATH:$(pwd)/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# 2. Wait for backend to be ready
echo "Waiting for backend..."
sleep 5

# 3. Start Streamlit frontend
echo "Starting Streamlit frontend..."
streamlit run frontend/streamlit_app.py --server.port 7860 --server.address 0.0.0.0

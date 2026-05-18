@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo    🧠 InvenioAI | Intelligent RAG
echo ==========================================
echo.

:: Set PYTHONPATH so uvicorn can find the 'app' module inside 'backend'
set PYTHONPATH=%PYTHONPATH%;%CD%\backend

:: Check if virtual environment exists and activate it
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else (
    echo [WARN] Virtual environment (.venv) not found. Using global python.
)

:: Start FastAPI backend in a separate window
echo [INFO] Launching Backend (FastAPI) on http://localhost:8000...
start "InvenioAI Backend" cmd /k "set PYTHONPATH=%PYTHONPATH%;%CD%\backend && uvicorn app.main:app --host 0.0.0.0 --port 8000"

:: Wait for backend to initialize and preload models
echo [INFO] Waiting for backend to finish preloading models...
:wait_backend
curl -s http://localhost:8000/ > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    timeout /t 2 /nobreak > nul
    goto wait_backend
)
echo [INFO] Backend is ready!

:: Start Streamlit frontend
echo [INFO] Launching Frontend (Streamlit) on http://localhost:7860...
streamlit run frontend/streamlit_app.py --server.port 7860

echo.
echo [INFO] Done. If you close this window, the Frontend will stop.
echo [INFO] Backend is running in a separate window.
pause

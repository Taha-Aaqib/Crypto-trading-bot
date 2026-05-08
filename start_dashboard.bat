@echo off
REM Start the React Dashboard API server + React dev server
REM Run from the Crypto-trading-bot/ directory

echo [1/2] Starting FastAPI server on http://localhost:8000 ...
start "SMC API Server" cmd /k "venv\Scripts\python.exe -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload"

echo [2/2] Starting React dev server on http://localhost:5173 ...
cd frontend
start "SMC React Dashboard" cmd /k "call ..\venv\Scripts\activate.bat & npx --yes vite"

echo.
echo Dashboard running at http://localhost:5173
echo API docs at      http://localhost:8000/docs

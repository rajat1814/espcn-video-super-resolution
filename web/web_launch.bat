@echo off
setlocal

title ESPCN Super-Resolution Web Server
echo ============================================================================
echo   ESPCN Real-Time Video Super-Resolution and Artifact Suppression Web Server
echo ============================================================================
echo.

REM 1. Check Python installation
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found on your system PATH!
    echo Please install Python 3.10+ from python.org and add it to PATH.
    pause
    exit /b 1
)

REM 2. Determine environment / venv
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment .venv...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment venv...
    call venv\Scripts\activate.bat
) else (
    echo [INFO] Using system Python environment...
)

REM 3. Check and install dependencies if missing
python -c "import fastapi, uvicorn, onnxruntime, cv2, numpy, PIL, websockets" >nul 2>&1
if errorlevel 1 (
    echo [SETUP] Missing required libraries. Installing web server requirements...
    echo This is a one-time setup on fresh machines.
    pip install -r web\requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed. Please check your internet connection.
        pause
        exit /b 1
    )
    echo [SETUP] Installation complete!
) else (
    echo [INFO] All dependencies verified! Launching instantly...
)

echo.
echo ============================================================================
echo   Starting Web Application on http://127.0.0.1:8000
echo   Press Ctrl+C to stop the server.
echo ============================================================================
echo.

REM 4. Open default browser after 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8000"

REM 5. Run Uvicorn Server
python -m uvicorn web.app:app --host 127.0.0.1 --port 8000 --log-level info

pause

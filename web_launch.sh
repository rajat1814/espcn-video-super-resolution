#!/usr/bin/env bash
# ============================================================================
#  ESPCN Real-Time Video Super-Resolution & Artifact Suppression Web Server
# ============================================================================

set -e

echo "============================================================================"
echo "  ESPCN Real-Time Video Super-Resolution & Artifact Suppression Web Server"
echo "============================================================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment (venv)..."
    source venv/bin/activate
fi

# Verify dependencies
if ! python3 -c "import fastapi, uvicorn, onnxruntime, cv2, numpy, PIL, websockets" 2>/dev/null; then
    echo "[SETUP] Missing required libraries. Installing web server requirements..."
    pip install -r web/requirements.txt
fi

echo "[INFO] Starting web application on http://127.0.0.1:8000"

# Open browser
if command -v xdg-open &> /dev/null; then
    (sleep 2 && xdg-open "http://127.0.0.1:8000") &
elif command -v open &> /dev/null; then
    (sleep 2 && open "http://127.0.0.1:8000") &
fi

python3 -m uvicorn web.app:app --host 127.0.0.1 --port 8000 --log-level info

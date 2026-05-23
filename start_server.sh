#!/bin/bash
# Startup script for Locopilot Monitoring System

echo "=================================================="
echo "Locopilot Monitoring System - Startup Script"
echo "=================================================="
echo ""

# Check if virtual environment exists
FRESH_VENV=0
if [ ! -d "venv" ]; then
    echo "[WARN] Virtual environment not found. Creating one..."
    python3 -m venv venv
    FRESH_VENV=1
    echo "[OK] Virtual environment created"
fi

# Activate virtual environment
echo "[SETUP] Activating virtual environment..."
source venv/bin/activate

# Per task 0009: do NOT run `pip install` on every server start. Production
# uses a hash-locked install that is performed once during deploy
# (deploy-gpu.sh or scripts/lock-deps.sh + manual install). The only time
# this script installs deps is when it just created an empty venv above.
#
# Production-shaped environments should set LOCOPILOT_REQUIRE_LOCK=1 — this
# refuses to start (and refuses to bootstrap) if requirements.lock is missing
# or still the PLACEHOLDER. Default behavior (dev) keeps the requirements.txt
# fallback.
LOCK_OK=0
if [ -f "requirements.lock" ] && ! grep -q "PLACEHOLDER" requirements.lock; then
    LOCK_OK=1
fi

if [ "${LOCOPILOT_REQUIRE_LOCK:-0}" = "1" ] && [ "$LOCK_OK" != "1" ]; then
    echo "[FAIL] LOCOPILOT_REQUIRE_LOCK=1 but requirements.lock is missing or still the PLACEHOLDER."
    echo "   Generate a real hashed lock with scripts/lock-deps.sh on a Python 3.12 + cu121 box,"
    echo "   commit it, then redeploy. Refusing to start."
    exit 1
fi

if [ "$FRESH_VENV" = "1" ]; then
    if [ "$LOCK_OK" = "1" ]; then
        echo "[SETUP] Bootstrapping deps from requirements.lock (--require-hashes)..."
        pip install --require-hashes --no-deps -r requirements.lock
    elif [ -f "requirements.txt" ]; then
        echo "[WARN] requirements.lock missing or placeholder — falling back to requirements.txt"
        echo "   Run scripts/lock-deps.sh to generate a hashed lock file."
        pip install -r requirements.txt --quiet
    else
        echo "[FAIL] No requirements.lock or requirements.txt found; aborting."
        exit 1
    fi
fi

# Create necessary directories
echo "[SETUP] Creating directories..."
mkdir -p uploads
mkdir -p locopilot_evidence

# Check if YOLO weights exist
if [ ! -f "yolo26n.pt" ]; then
    echo "[WARN] YOLO weights (yolo26n.pt) not found!"
    echo "   Please download YOLO weights or the system will download automatically on first use."
fi

echo ""
echo "=================================================="
echo "Starting server..."
echo "=================================================="
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "[WARN] .env file not found. Using defaults..."
    echo "   Create a .env file from .env.example for custom configuration."
    echo ""
fi

# Start the server
# Uncomment one of the following:

# Development mode (with auto-reload)
#echo "[START] Starting in DEVELOPMENT mode with auto-reload..."
# uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Production mode (with Gunicorn)
echo "[START] Starting in PRODUCTION mode with Gunicorn..."
gunicorn -c gunicorn_config.py app.main:app

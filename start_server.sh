#!/bin/bash
# Startup script for Locopilot Monitoring System

echo "=================================================="
echo "Locopilot Monitoring System - Startup Script"
echo "=================================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt --quiet

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads
mkdir -p locopilot_evidence

# Check if YOLO weights exist
if [ ! -f "yolo11m.pt" ]; then
    echo "⚠️  YOLO weights (yolo11m.pt) not found!"
    echo "   Please download YOLO weights or the system will download automatically on first use."
fi

echo ""
echo "=================================================="
echo "Starting server..."
echo "=================================================="
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Using defaults..."
    echo "   Create a .env file from .env.example for custom configuration."
    echo ""
fi

# Start the server
# Uncomment one of the following:

# Development mode (with auto-reload)
#echo "🚀 Starting in DEVELOPMENT mode with auto-reload..."
# uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Production mode (with Gunicorn)
echo "🚀 Starting in PRODUCTION mode with Gunicorn..."
gunicorn -c gunicorn_config.py app.main:app


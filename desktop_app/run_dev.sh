#!/bin/bash
# Development run script

set -e

echo "=================================="
echo "Running Locopilot CVVR Desktop (Dev Mode)"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "Error: Please run this script from the desktop_app directory"
    exit 1
fi

# Check if dependencies are installed
if ! python3 -c "import PySide6" 2>/dev/null; then
    echo "Dependencies not found. Installing..."
    pip3 install -r requirements_desktop.txt
fi

# Run the application
echo ""
echo "Starting application..."
python3 -m desktop_app.main


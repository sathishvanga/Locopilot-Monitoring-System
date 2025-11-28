#!/bin/bash
# Script to install backend dependencies for Locopilot CVVR Desktop App

echo "=================================="
echo "Installing Backend Dependencies"
echo "=================================="
echo ""

# Check if we're in a virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
    echo "Installing packages in virtual environment..."
    pip install gunicorn uvicorn fastapi torch ultralytics
    echo ""
    echo "✓ Packages installed successfully!"
    exit 0
fi

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Python version: $PYTHON_VERSION"
echo ""

# Try to install with --user flag first
echo "Attempting to install packages with --user flag..."
if python3 -m pip install --user gunicorn uvicorn fastapi torch ultralytics 2>&1; then
    echo ""
    echo "✓ Packages installed successfully in user directory!"
    exit 0
fi

# If that fails, try with --break-system-packages (macOS Homebrew)
echo ""
echo "Attempting with --break-system-packages flag..."
if python3 -m pip install --break-system-packages gunicorn uvicorn fastapi torch ultralytics 2>&1; then
    echo ""
    echo "✓ Packages installed successfully!"
    exit 0
fi

echo ""
echo "=================================="
echo "Installation Failed"
echo "=================================="
echo ""
echo "Please install the packages manually:"
echo "  python3 -m pip install --user gunicorn uvicorn fastapi torch ultralytics"
echo ""
echo "Or if using Homebrew Python:"
echo "  python3 -m pip install --break-system-packages gunicorn uvicorn fastapi torch ultralytics"
echo ""
exit 1


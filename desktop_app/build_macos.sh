#!/bin/bash
# Build script for macOS

set -e  # Exit on error

echo "=================================="
echo "Building Locopilot CVVR for macOS"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "Error: Please run this script from the desktop_app directory"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo ""
echo "Checking dependencies..."
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Not in a virtual environment"
    echo "Installing dependencies with pip3..."
    pip3 install -r requirements_desktop.txt --break-system-packages 2>/dev/null || pip3 install -r requirements_desktop.txt
else
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
    echo "Ensuring dependencies are installed..."
    pip install -r requirements_desktop.txt
fi

# Check if YOLO model exists
echo ""
echo "Checking for YOLO model..."
YOLO_PATH="../yolo11s.pt"
if [ -f "$YOLO_PATH" ]; then
    YOLO_SIZE=$(du -h "$YOLO_PATH" | cut -f1)
    echo "✓ YOLO model found: $YOLO_PATH ($YOLO_SIZE)"
else
    echo "⚠ YOLO model not found at $YOLO_PATH"
    echo "  The model will be downloaded on first use (may take time)"
    echo "  To include the model in the build:"
    echo "    1. Download yolo11s.pt"
    echo "    2. Place it in: $(cd .. && pwd)/yolo11s.pt"
fi

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p ../uploads
mkdir -p ../locopilot_evidence
echo "✓ Directories created"

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build_config/build build_config/dist

# Build with PyInstaller
echo ""
echo "Building application with PyInstaller..."
cd build_config
pyinstaller build_macos.spec

# Check if build succeeded
if [ -d "dist/LocopilotCVVR.app" ]; then
    # Get app size
    APP_SIZE=$(du -sh dist/LocopilotCVVR.app | cut -f1)
    
    echo ""
    echo "=================================="
    echo "Build successful!"
    echo "=================================="
    echo ""
    echo "Application bundle: build_config/dist/LocopilotCVVR.app"
    echo "Application size: $APP_SIZE"
    echo ""
    echo "The app includes:"
    echo "  ✓ Desktop GUI"
    echo "  ✓ FastAPI backend"
    echo "  ✓ ML models (YOLO, etc.)"
    echo "  ✓ Auto-start backend"
    echo ""
    echo "To run:"
    echo "  open build_config/dist/LocopilotCVVR.app"
    echo ""
    echo "To distribute:"
    echo "  1. Compress: tar -czf LocopilotCVVR.app.tar.gz -C build_config/dist LocopilotCVVR.app"
    echo "  2. Distribute the .tar.gz file"
    echo ""
else
    echo ""
    echo "=================================="
    echo "Build failed!"
    echo "=================================="
    exit 1
fi


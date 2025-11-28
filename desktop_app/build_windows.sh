#!/bin/bash
# Build script for Windows
# Can be run in Git Bash, WSL, or any bash environment on Windows

set -e  # Exit on error

echo "=================================="
echo "Building Locopilot CVVR for Windows"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "Error: Please run this script from the desktop_app directory"
    exit 1
fi

# Detect Python command (try python first, then python3)
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found. Please install Python 3.8 or higher"
    exit 1
fi

# Check Python version
python_version=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo ""
echo "Checking dependencies..."
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Not in a virtual environment"
    echo "Installing dependencies with pip..."
    $PYTHON_CMD -m pip install -r requirements_desktop.txt
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
    # Use appropriate command for file size (works on both Windows Git Bash and WSL)
    if command -v du &> /dev/null; then
        YOLO_SIZE=$(du -h "$YOLO_PATH" 2>/dev/null | cut -f1 || echo "unknown")
    else
        YOLO_SIZE="unknown"
    fi
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
$PYTHON_CMD -m PyInstaller build_windows.spec

# Check if build succeeded (handle both .exe on Windows and no extension on macOS)
EXE_FILE=""
if [ -f "dist/LocopilotCVVR.exe" ]; then
    EXE_FILE="dist/LocopilotCVVR.exe"
    EXE_NAME="LocopilotCVVR.exe"
elif [ -f "dist/LocopilotCVVR" ]; then
    EXE_FILE="dist/LocopilotCVVR"
    EXE_NAME="LocopilotCVVR"
fi

if [ -n "$EXE_FILE" ]; then
    # Get file size (works on both Windows Git Bash and WSL)
    if command -v du &> /dev/null; then
        APP_SIZE=$(du -h "$EXE_FILE" 2>/dev/null | cut -f1 || echo "unknown")
    else
        APP_SIZE="unknown"
    fi
    
    echo ""
    echo "=================================="
    echo "Build successful!"
    echo "=================================="
    echo ""
    echo "Application executable: build_config/$EXE_FILE"
    echo "Application size: $APP_SIZE"
    echo ""
    echo "The app includes:"
    echo "  ✓ Desktop GUI"
    echo "  ✓ FastAPI backend"
    echo "  ✓ ML models (YOLO, etc.)"
    echo "  ✓ Auto-start backend"
    echo ""
    echo "To run:"
    echo "  cd build_config/dist"
    echo "  ./$EXE_NAME"
    echo ""
    if [ "$EXE_NAME" = "LocopilotCVVR.exe" ]; then
        echo "Or double-click LocopilotCVVR.exe in Windows Explorer"
    fi
    echo ""
    echo "To distribute:"
    echo "  1. Compress the executable:"
    echo "     tar -czf LocopilotCVVR-windows.tar.gz -C build_config/dist $EXE_NAME"
    echo "  2. Or use 7-Zip/WinRAR to create a zip file"
    echo "  3. Distribute the compressed file"
    echo ""
else
    echo ""
    echo "=================================="
    echo "Build failed!"
    echo "=================================="
    exit 1
fi


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
echo "Installing dependencies..."
pip3 install -r requirements_desktop.txt

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
    echo ""
    echo "=================================="
    echo "Build successful!"
    echo "=================================="
    echo ""
    echo "Application bundle: build_config/dist/LocopilotCVVR.app"
    echo ""
    echo "To run:"
    echo "  open build_config/dist/LocopilotCVVR.app"
    echo ""
else
    echo ""
    echo "=================================="
    echo "Build failed!"
    echo "=================================="
    exit 1
fi


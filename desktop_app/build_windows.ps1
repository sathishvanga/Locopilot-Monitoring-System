# Build script for Windows (PowerShell)
# Run this script in PowerShell: .\build_windows.ps1

$ErrorActionPreference = "Stop"

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Building Locopilot CVVR for Windows" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

# Check if we're in the right directory
if (-not (Test-Path "main.py")) {
    Write-Host "Error: Please run this script from the desktop_app directory" -ForegroundColor Red
    exit 1
}

# Detect Python command
$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonCmd = "python3"
} else {
    Write-Host "Error: Python not found. Please install Python 3.8 or higher" -ForegroundColor Red
    exit 1
}

# Check Python version
$pythonVersion = & $pythonCmd --version 2>&1
Write-Host "Python version: $pythonVersion"

# Install dependencies
Write-Host ""
Write-Host "Checking dependencies..."
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Warning: Not in a virtual environment" -ForegroundColor Yellow
    Write-Host "Installing dependencies with pip..."
    & $pythonCmd -m pip install -r requirements_desktop.txt
} else {
    Write-Host "✓ Virtual environment detected: $env:VIRTUAL_ENV" -ForegroundColor Green
    Write-Host "Ensuring dependencies are installed..."
    pip install -r requirements_desktop.txt
}

# Check if YOLO model exists
Write-Host ""
Write-Host "Checking for YOLO model..."
$yoloPath = "..\yolo11s.pt"
if (Test-Path $yoloPath) {
    $yoloSize = (Get-Item $yoloPath).Length / 1MB
    $yoloSizeFormatted = "{0:N2} MB" -f $yoloSize
    Write-Host "✓ YOLO model found: $yoloPath ($yoloSizeFormatted)" -ForegroundColor Green
} else {
    Write-Host "⚠ YOLO model not found at $yoloPath" -ForegroundColor Yellow
    Write-Host "  The model will be downloaded on first use (may take time)"
    Write-Host "  To include the model in the build:"
    $parentPath = (Get-Item "..").FullName
    Write-Host "    1. Download yolo11s.pt"
    Write-Host "    2. Place it in: $parentPath\yolo11s.pt"
}

# Create output directories
Write-Host ""
Write-Host "Creating output directories..."
New-Item -ItemType Directory -Force -Path "..\uploads" | Out-Null
New-Item -ItemType Directory -Force -Path "..\locopilot_evidence" | Out-Null
Write-Host "✓ Directories created" -ForegroundColor Green

# Clean previous builds
Write-Host ""
Write-Host "Cleaning previous builds..."
if (Test-Path "build_config\build") {
    Remove-Item -Recurse -Force "build_config\build"
}
if (Test-Path "build_config\dist") {
    Remove-Item -Recurse -Force "build_config\dist"
}

# Build with PyInstaller
Write-Host ""
Write-Host "Building application with PyInstaller..."
Push-Location build_config
& $pythonCmd -m PyInstaller build_windows.spec
Pop-Location

# Check if build succeeded (handle both .exe on Windows and no extension on macOS/WSL)
$exePath = $null
$exeName = $null
if (Test-Path "build_config\dist\LocopilotCVVR.exe") {
    $exePath = "build_config\dist\LocopilotCVVR.exe"
    $exeName = "LocopilotCVVR.exe"
} elseif (Test-Path "build_config\dist\LocopilotCVVR") {
    $exePath = "build_config\dist\LocopilotCVVR"
    $exeName = "LocopilotCVVR"
}

if ($exePath) {
    $appSize = (Get-Item $exePath).Length / 1MB
    $appSizeFormatted = "{0:N2} MB" -f $appSize
    
    Write-Host ""
    Write-Host "==================================" -ForegroundColor Green
    Write-Host "Build successful!" -ForegroundColor Green
    Write-Host "==================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Application executable: $exePath"
    Write-Host "Application size: $appSizeFormatted"
    Write-Host ""
    Write-Host "The app includes:" -ForegroundColor Cyan
    Write-Host "  ✓ Desktop GUI"
    Write-Host "  ✓ FastAPI backend"
    Write-Host "  ✓ ML models (YOLO, etc.)"
    Write-Host "  ✓ Auto-start backend"
    Write-Host ""
    Write-Host "To run:" -ForegroundColor Cyan
    Write-Host "  cd build_config\dist"
    Write-Host "  .\$exeName"
    Write-Host ""
    if ($exeName -eq "LocopilotCVVR.exe") {
        Write-Host "Or double-click LocopilotCVVR.exe in Windows Explorer"
    }
    Write-Host ""
    Write-Host "To distribute:" -ForegroundColor Cyan
    Write-Host "  1. Right-click $exeName"
    Write-Host "  2. Select 'Send to' > 'Compressed (zipped) folder'"
    Write-Host "  3. Or use 7-Zip/WinRAR to create a zip file"
    Write-Host "  4. Distribute the compressed file"
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "==================================" -ForegroundColor Red
    Write-Host "Build failed!" -ForegroundColor Red
    Write-Host "==================================" -ForegroundColor Red
    exit 1
}


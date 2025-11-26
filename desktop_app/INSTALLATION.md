# Installation Guide

## Prerequisites

### macOS

- macOS 10.15 (Catalina) or later
- Python 3.11 or later
- Xcode Command Line Tools (for building)

### Windows

- Windows 10 or later
- Python 3.11 or later (if building from source)

## Installation Methods

### Method 1: Pre-built Binaries (Recommended)

#### macOS

1. Download `LocopilotCVVR.app` from GitHub Releases
2. Move to Applications folder
3. Right-click and select "Open" (first time only, to bypass Gatekeeper)
4. The application will launch

#### Windows

1. Download `LocopilotCVVR.exe` from GitHub Releases
2. Run the executable
3. If Windows Defender SmartScreen appears, click "More info" → "Run anyway"

### Method 2: Build from Source

#### macOS Build

```bash
# 1. Clone the repository
git clone <repository-url>
cd "Locopilot Monitoring System/desktop_app"

# 2. Install Python dependencies
pip3 install -r requirements_desktop.txt

# 3. Build the application
./build_macos.sh

# 4. The app will be created in: build_config/dist/LocopilotCVVR.app
open build_config/dist/LocopilotCVVR.app
```

#### Windows Build (via GitHub Actions)

The Windows build is automated:

1. Push code to GitHub
2. GitHub Actions will build automatically
3. Download from GitHub Releases

#### Manual Windows Build

```cmd
# 1. Clone the repository
git clone <repository-url>
cd "Locopilot Monitoring System\desktop_app"

# 2. Install dependencies
pip install -r requirements_desktop.txt

# 3. Build with PyInstaller
cd build_config
pyinstaller build_windows.spec

# 4. The exe will be in: build_config\dist\LocopilotCVVR.exe
```

### Method 3: Run from Source (Development)

```bash
# 1. Navigate to desktop_app directory
cd desktop_app

# 2. Install dependencies
pip3 install -r requirements_desktop.txt

# 3. Run the application
python3 -m desktop_app.main

# Or use the development script (macOS/Linux)
./run_dev.sh
```

## Configuration

### Environment Variables (Optional)

Create a `.env` file in the desktop_app directory:

```bash
cp .env.example .env
```

Edit `.env` to customize:

- API endpoints
- Timeouts
- Logging levels
- Window size

### Local Backend Setup

The desktop app requires the local FastAPI backend for video processing:

1. Navigate to project root
2. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the backend:
   ```bash
   uvicorn app.main:app --reload
   ```
4. Verify it's running:
   ```bash
   curl http://localhost:8000/health
   ```

## Troubleshooting

### macOS

**"App is damaged" error:**

```bash
xattr -cr /Applications/LocopilotCVVR.app
```

**Permission denied:**

```bash
chmod +x /Applications/LocopilotCVVR.app/Contents/MacOS/LocopilotCVVR
```

### Windows

**Missing DLL errors:**

Install Microsoft Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe

**Antivirus blocking:**

Add exception for LocopilotCVVR.exe in your antivirus software

### Both Platforms

**Backend connection error:**

1. Ensure local FastAPI backend is running on port 8000
2. Check firewall settings
3. Verify `CVVR_LOCAL_BACKEND_URL` in configuration

**Login failure:**

1. Check internet connection
2. Verify API endpoint is accessible
3. Check credentials

## Uninstallation

### macOS

```bash
# Remove application
rm -rf /Applications/LocopilotCVVR.app

# Remove logs and config (optional)
rm -rf ~/Library/Application\ Support/LocopilotCVVR
rm ~/Desktop/desktop_app.log
```

### Windows

```cmd
# Delete the executable
del LocopilotCVVR.exe

# Remove logs (optional)
del desktop_app.log
```

## Support

For issues and questions:
- Email: info@mindcoinservices.com
- Phone: +91-97016 58885


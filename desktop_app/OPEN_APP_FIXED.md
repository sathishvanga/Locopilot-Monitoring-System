# How to Open Locopilot CVVR App

## ✅ Method 1: Use the Launcher Script (Recommended)
```bash
cd "/Users/satishvanga/Desktop/Locopilot Monitoring System/desktop_app"
./launch_app.sh
```

## ✅ Method 2: Direct Execution
```bash
cd "/Users/satishvanga/Desktop/Locopilot Monitoring System/desktop_app"
./build_config/dist/LocopilotCVVR.app/Contents/MacOS/LocopilotCVVR
```

## Method 3: Using `open` Command
```bash
cd "/Users/satishvanga/Desktop/Locopilot Monitoring System/desktop_app"
open build_config/dist/LocopilotCVVR.app
```

**Note:** If `open` doesn't work, use Method 1 or 2 instead.

## Method 4: Finder (Double-Click)
1. Open Finder
2. Navigate to: `Desktop/Locopilot Monitoring System/desktop_app/build_config/dist/`
3. Double-click `LocopilotCVVR.app`
4. If macOS blocks it:
   - Right-click → "Open"
   - Click "Open" in the security dialog

## Method 5: Create Desktop Shortcut
```bash
# Create an alias on Desktop
ln -s "/Users/satishvanga/Desktop/Locopilot Monitoring System/desktop_app/build_config/dist/LocopilotCVVR.app" ~/Desktop/LocopilotCVVR.app
```

Then double-click the shortcut on your Desktop.

## Troubleshooting

### App Doesn't Open with `open` Command
- Use the launcher script instead: `./launch_app.sh`
- Or run directly: `./build_config/dist/LocopilotCVVR.app/Contents/MacOS/LocopilotCVVR`

### macOS Security Warning
```bash
# Remove quarantine attribute
xattr -cr build_config/dist/LocopilotCVVR.app
```

### Check if App is Running
```bash
ps aux | grep LocopilotCVVR | grep -v grep
```

### App Opens But No Window
- Check the Dock for the app icon
- Click the app icon in the Dock to bring it to front
- Check Activity Monitor to see if the process is running

## Quick Start
The easiest way is to use the launcher script:
```bash
cd "/Users/satishvanga/Desktop/Locopilot Monitoring System/desktop_app"
./launch_app.sh
```


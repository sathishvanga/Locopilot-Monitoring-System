# How to Open the Locopilot CVVR App

## Method 1: Direct Execution (Most Reliable)
```bash
cd "/Users/satishvanga/Desktop/Locopilot Monitoring System/desktop_app"
./build_config/dist/LocopilotCVVR.app/Contents/MacOS/LocopilotCVVR
```

## Method 2: Using `open` Command
```bash
cd "/Users/satishvanga/Desktop/Locopilot Monitoring System/desktop_app"
open build_config/dist/LocopilotCVVR.app
```

## Method 3: Finder (Double-Click)
1. Navigate to: `desktop_app/build_config/dist/`
2. Double-click `LocopilotCVVR.app`
3. If macOS blocks it, right-click → "Open" → "Open" in dialog

## Method 4: Create an Alias/Shortcut
```bash
# Create a launcher script
cat > ~/Desktop/LocopilotCVVR.sh << 'EOF'
#!/bin/bash
cd "/Users/satishvanga/Desktop/Locopilot Monitoring System/desktop_app"
./build_config/dist/LocopilotCVVR.app/Contents/MacOS/LocopilotCVVR
EOF

chmod +x ~/Desktop/LocopilotCVVR.sh
```

Then double-click `LocopilotCVVR.sh` on your Desktop.

## Troubleshooting

### App Opens But No Window Appears
- Check if it's in the Dock (look for the app icon)
- Check Activity Monitor to see if the process is running
- Try clicking the app icon in the Dock

### macOS Security Warning
If you see a security warning:
1. Right-click the app → "Open"
2. Click "Open" in the security dialog
3. Or run: `xattr -cr build_config/dist/LocopilotCVVR.app`

### Check if App is Running
```bash
ps aux | grep LocopilotCVVR | grep -v grep
```

### View App Logs
The app logs to `desktop_app.log` in the desktop_app directory.


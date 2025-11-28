#!/bin/bash
# Launcher script for Locopilot CVVR Desktop App

APP_PATH="/Users/satishvanga/Desktop/Locopilot Monitoring System/desktop_app/build_config/dist/LocopilotCVVR.app"

# Check if app exists
if [ ! -d "$APP_PATH" ]; then
    echo "Error: App not found at $APP_PATH"
    exit 1
fi

# Remove quarantine attribute if present
xattr -cr "$APP_PATH" 2>/dev/null

# Launch the app directly
echo "Launching Locopilot CVVR..."
"$APP_PATH/Contents/MacOS/LocopilotCVVR" &

echo "App launched! Check your Dock or windows."


# Fix: macOS App Not Opening

If the app doesn't open when you double-click it, macOS Gatekeeper is likely blocking it because it's not code-signed by Apple.

## Quick Fix: Allow the App to Run

### Option 1: Right-click and Open (Recommended)
1. Right-click on `LocopilotCVVR.app`
2. Select "Open" from the context menu
3. Click "Open" in the security dialog
4. The app will now open normally

### Option 2: Remove Quarantine Attribute
Run this command in Terminal:
```bash
cd "/Users/satishvanga/Desktop/Locopilot Monitoring System/desktop_app"
xattr -cr build_config/dist/LocopilotCVVR.app
```

### Option 3: System Settings
1. Go to System Settings → Privacy & Security
2. Scroll down to "Security"
3. If you see a message about the app being blocked, click "Open Anyway"

## Alternative: Run from Terminal

You can always run the app from Terminal:
```bash
cd "/Users/satishvanga/Desktop/Locopilot Monitoring System/desktop_app"
open build_config/dist/LocopilotCVVR.app
```

Or directly:
```bash
"/Users/satishvanga/Desktop/Locopilot Monitoring System/desktop_app/build_config/dist/LocopilotCVVR.app/Contents/MacOS/LocopilotCVVR"
```

## Code Signing (For Distribution)

For proper distribution, the app should be code-signed with an Apple Developer certificate. This requires:
- Apple Developer account ($99/year)
- Developer ID certificate
- Notarization

For development/testing, the adhoc signing is sufficient.


# Quick Start Guide

Get started with Locopilot CVVR Desktop in 5 minutes!

## Prerequisites

✅ Python 3.11+ installed  
✅ Internet connection  
✅ Local FastAPI backend running (for video processing)

## Step 1: Start the Local Backend

```bash
# Navigate to project root
cd "/Users/satishvanga/Desktop/Locopilot Monitoring System"

# Install backend dependencies (first time only)
pip install -r requirements.txt

# Start the backend
uvicorn app.main:app --reload
```

Leave this terminal running. The backend runs on http://localhost:8000

## Step 2: Run the Desktop App

### Option A: Development Mode (Recommended for testing)

```bash
# Open a new terminal
cd "/Users/satishvanga/Desktop/Locopilot Monitoring System/desktop_app"

# Install dependencies (first time only)
pip install -r requirements_desktop.txt

# Run the app
python3 -m desktop_app.main

# Or use the script:
./run_dev.sh
```

### Option B: Build and Run

```bash
cd "/Users/satishvanga/Desktop/Locopilot Monitoring System/desktop_app"

# Build the app
./build_macos.sh

# Run the built app
open build_config/dist/LocopilotCVVR.app
```

## Step 3: Login

1. The app will open with the login screen
2. Enter credentials:
   - **Mobile Number**: `9705589009`
   - **Password**: `Demo@9009`
   - **Organization**: Demo (already selected)
3. Click **"Log In"**

## Step 4: Upload a Video

1. After login, you'll see the pending trips table
2. Click **"Upload Video"** for any trip
3. Select a video file (e.g., .mp4, .avi, .mov)
4. Wait while the app:
   - ✅ Processes the video locally
   - ✅ Detects activities
   - ✅ Generates evidence clips
   - ✅ Uploads to S3
5. Done! Status changes to **"Completed ✓"**

## Troubleshooting

### "Backend Not Running" Error

**Problem**: Local backend is not started

**Solution**:
```bash
# In a separate terminal
cd "/Users/satishvanga/Desktop/Locopilot Monitoring System"
uvicorn app.main:app --reload
```

Verify it's running:
```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy",...}
```

### Login Failed

**Problem**: Network or credentials issue

**Solution**:
- Check internet connection
- Verify credentials are correct
- Check API is accessible:
  ```bash
  curl https://api.mindcoinapps.com/ai_demo_api/health
  ```

### Import Errors

**Problem**: Missing dependencies

**Solution**:
```bash
cd desktop_app
pip install -r requirements_desktop.txt
```

### Port Already in Use

**Problem**: Port 8000 already taken

**Solution**:
```bash
# Find what's using port 8000
lsof -i :8000

# Kill the process (if it's old backend)
kill -9 <PID>

# Or run on different port
uvicorn app.main:app --port 8001

# Then update desktop app config:
export CVVR_LOCAL_BACKEND_URL=http://localhost:8001
```

## Testing Without Backend

You can test the UI without the backend:

1. Login works (remote API)
2. View trips works (remote API)
3. Upload will fail at processing step (expected)

To test upload, you MUST have the local backend running.

## Development Tips

### Enable Debug Logging

```bash
export CVVR_LOG_LEVEL=DEBUG
export CVVR_DEBUG=True
python3 -m desktop_app.main
```

### Check Logs

```bash
tail -f desktop_app.log
```

### Run Tests

```bash
cd desktop_app
pytest tests/ -v
```

### Clean Build

```bash
cd build_config
rm -rf build dist
pyinstaller build_macos.spec
```

## Video Format Support

Supported formats:
- ✅ MP4 (.mp4)
- ✅ AVI (.avi)
- ✅ MOV (.mov)
- ✅ MKV (.mkv)
- ✅ FLV (.flv)
- ✅ WMV (.wmv)

Max file size: **2GB**

## Expected Workflow

```
1. Start Backend (Terminal 1)
   ↓
2. Start Desktop App (Terminal 2)
   ↓
3. Login
   ↓
4. View Pending Trips
   ↓
5. Click "Upload Video"
   ↓
6. Select Video File
   ↓
7. Processing (20-60 seconds)
   ↓
8. Uploading (30-120 seconds)
   ↓
9. Completed ✓
```

## Next Steps

- Read [README.md](README.md) for full features
- See [INSTALLATION.md](INSTALLATION.md) for production setup
- Check [DESKTOP_APP_SUMMARY.md](../DESKTOP_APP_SUMMARY.md) for technical details

## Need Help?

- **Email**: info@mindcoinservices.com
- **Phone**: +91-97016 58885

---

**Ready to go!** 🚀

Start the backend, run the app, and upload your first video!


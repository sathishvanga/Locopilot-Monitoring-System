# Setup Guide - Locopilot Monitoring System

## 📋 Prerequisites

- **Python**: 3.9 or higher
- **pip**: Latest version
- **System Libraries**: OpenCV dependencies (automatically installed)
- **Storage**: At least 2GB free space for models and videos
- **RAM**: At least 4GB (8GB recommended for video processing)
- **CPU**: Multi-core processor recommended

## 🚀 Installation Steps

### Step 1: Clone or Navigate to Project

```bash
cd "/Users/satishvanga/Desktop/Locopilot Monitoring System"
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

This will install:
- FastAPI (web framework)
- Uvicorn (ASGI server)
- Gunicorn (production server)
- Pydantic (data validation)
- OpenCV (video processing)
- MediaPipe (pose detection)
- Ultralytics YOLO (object detection)
- And all required dependencies

**Installation may take 5-10 minutes depending on your internet speed.**

### Step 4: Download YOLO Weights

The system needs YOLO model weights. They will be downloaded automatically on first use, or you can download manually:

```bash
# Option 1: Let the system download automatically (recommended)
# It will download on first video processing

# Option 2: Download manually
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11s.pt
```

### Step 5: Configure Environment (Optional)

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (optional)
nano .env
```

Default configuration works for most cases.

### Step 6: Create Required Directories

```bash
# These are created automatically, but you can create them manually
mkdir -p uploads
mkdir -p locopilot_evidence
```

## ✅ Verify Installation

### Test 1: Check Dependencies

```bash
python -c "import fastapi, uvicorn, cv2, mediapipe; print('✅ All dependencies installed')"
```

### Test 2: Start the Server

```bash
# Make startup script executable
chmod +x start_server.sh

# Start the server
./start_server.sh
```

You should see:
```
==================================================
Locopilot Monitoring System v1.0.0
==================================================
```

### Test 3: Check API Health

Open a new terminal and run:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "application": "Locopilot Monitoring System",
  "version": "1.0.0"
}
```

### Test 4: Run Test Suite

```bash
python test_api.py
```

This will test all endpoints and ensure everything is working.

### Test 5: Access API Documentation

Open your browser and go to:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

## 🎯 Quick Start

### Option 1: Using the Startup Script

```bash
./start_server.sh
```

This script:
1. Creates virtual environment if needed
2. Installs/updates dependencies
3. Creates required directories
4. Starts the server

### Option 2: Manual Start (Development)

```bash
# Activate virtual environment
source venv/bin/activate

# Start with Uvicorn (auto-reload enabled)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Production Start

```bash
# Activate virtual environment
source venv/bin/activate

# Start with Gunicorn (multiprocessing)
gunicorn -c gunicorn_config.py app.main:app
```

## 📡 Test the API

### Using cURL

```bash
curl -X POST "http://localhost:8000/api/v1/video/process" \
  -F "video=@example_data/latest.mp4" \
  -F "tripId=TRIP-001" \
  -F "crewName=John Doe" \
  -F "crewId=C-001" \
  -F "crewRole=1" \
  -F "useMockDetection=true"
```

### Using Python

```python
import requests

url = "http://localhost:8000/api/v1/video/process"

with open('example_data/latest.mp4', 'rb') as f:
    files = {'video': f}
    data = {
        'tripId': 'TRIP-001',
        'crewName': 'John Doe',
        'crewId': 'C-001',
        'crewRole': 1,
        'useMockDetection': True
    }
    response = requests.post(url, files=files, data=data)
    print(response.json())
```

### Using Swagger UI

1. Go to http://localhost:8000/docs
2. Click on `POST /api/v1/video/process`
3. Click "Try it out"
4. Fill in the parameters
5. Upload a video file
6. Click "Execute"

## 🐳 Docker Setup (Optional)

### Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p uploads locopilot_evidence

# Expose port
EXPOSE 8000

# Run with Gunicorn
CMD ["gunicorn", "-c", "gunicorn_config.py", "app.main:app"]
```

### Build and Run

```bash
# Build Docker image
docker build -t locopilot-api .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/locopilot_evidence:/app/locopilot_evidence \
  -v $(pwd)/uploads:/app/uploads \
  locopilot-api
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# Application
APP_NAME="Locopilot Monitoring System"
DEBUG=false
LOG_LEVEL=INFO

# Server
HOST=0.0.0.0
PORT=8000

# File Upload
MAX_UPLOAD_SIZE=524288000  # 500 MB
UPLOAD_DIR=uploads

# Processing
SAMPLE_FPS=0.5  # 1 frame every 2 seconds
OUTPUT_DIR=locopilot_evidence
SAVE_ANNOTATED_FRAMES=false

# Models
YOLO_WEIGHTS_PRELOAD=yolo11s.pt
PRELOAD_OCR=0

# CORS (for production, specify exact origins)
CORS_ORIGINS=*
```

### Gunicorn Configuration

Edit `gunicorn_config.py` for production tuning:

```python
# Adjust workers based on CPU cores
workers = 4  # Or: max(1, multiprocessing.cpu_count() // 2)

# Adjust timeout for longer videos
timeout = 1200  # 20 minutes

# Adjust max upload size if needed
# (Also update MAX_UPLOAD_SIZE in .env)
```

## 🔍 Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "Port 8000 already in use"

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or change port in .env
echo "PORT=8001" >> .env
```

### Issue: YOLO weights download fails

**Solution:**
```bash
# Download manually
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11s.pt

# Or use a different model
echo "YOLO_WEIGHTS_PRELOAD=yolov8s.pt" >> .env
```

### Issue: Video processing timeout

**Solution:**
```bash
# Increase timeout in gunicorn_config.py
# Edit line: timeout = 1200  # Increase to 20 minutes
```

### Issue: Large video upload fails

**Solution:**
```bash
# Increase max upload size in .env
echo "MAX_UPLOAD_SIZE=1073741824" >> .env  # 1 GB
```

### Issue: OpenCV errors

**Solution:**
```bash
# Reinstall OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-contrib-python>=4.11.0.86
```

### Issue: MediaPipe errors

**Solution:**
```bash
# Reinstall MediaPipe
pip uninstall mediapipe
pip install mediapipe==0.10.21
```

## 📊 Performance Tuning

### For Fast Processing

```env
SAMPLE_FPS=1.0  # Sample every second
SAVE_ANNOTATED_FRAMES=false  # Don't save frames
```

### For High Accuracy

```env
SAMPLE_FPS=2.0  # Sample twice per second
SAVE_ANNOTATED_FRAMES=true  # Save frames for review
```

### For Production

```python
# In gunicorn_config.py
workers = multiprocessing.cpu_count()  # Use all cores
timeout = 1800  # 30 minutes timeout
max_requests = 5000  # Higher throughput
```

## 🔐 Security Hardening (Production)

### 1. Enable HTTPS

```bash
# Use nginx reverse proxy with SSL
sudo apt install nginx certbot python3-certbot-nginx

# Configure SSL certificate
sudo certbot --nginx -d your-domain.com
```

### 2. Restrict CORS

```env
# In .env - only allow specific origins
CORS_ORIGINS=https://your-frontend.com,https://app.your-domain.com
```

### 3. Add Authentication

Add to `app/main.py`:
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.middleware("http")
async def verify_token(request: Request, call_next):
    # Add your authentication logic here
    pass
```

### 4. Rate Limiting

```bash
pip install slowapi

# Add to app/main.py
from slowapi import Limiter, _rate_limit_exceeded_handler
```

### 5. File Validation

Already implemented in `VideoProcessingService.validate_video_file()`

## 📝 Next Steps

1. **Test with real videos** - Process actual videos
2. **Integrate with your application** - Use the API in your frontend
3. **Monitor logs** - Check logs for errors
4. **Scale if needed** - Add more workers or instances
5. **Add authentication** - Secure your API
6. **Deploy to production** - Use Docker/Kubernetes

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **Project Overview**: See `PROJECT_OVERVIEW.md`
- **API Usage Guide**: See `API_USAGE_GUIDE.md`
- **Detailed API Docs**: See `README_API.md`

## 🆘 Getting Help

If you encounter issues:

1. Check the logs for error messages
2. Review the troubleshooting section
3. Ensure all dependencies are installed
4. Test with mock detection first
5. Check the GitHub issues (if applicable)

## ✅ Installation Complete!

You're all set! The API is ready to process videos and detect activities.

**Quick Test:**
```bash
./start_server.sh
python test_api.py
```

**Start Processing:**
```bash
curl -X POST "http://localhost:8000/api/v1/video/process" \
  -F "video=@your_video.mp4" \
  -F "tripId=YOUR-TRIP-ID"
```

🎉 **Happy coding!**


# Locopilot Monitoring System - FastAPI Application

A production-ready FastAPI application for video processing and activity detection with clean MVC architecture.

## 🏗️ Architecture

The application follows a clean **MVC (Model-View-Controller)** architecture:

```
app/
├── models/              # Pydantic schemas and domain models
│   ├── activity_models.py
│   └── video_models.py
├── controllers/         # API route handlers (routers)
│   └── video_controller.py
├── services/           # Business logic and processing
│   ├── video_processing_service.py
│   └── activity_detection_service.py
├── repositories/       # Data persistence and file I/O
│   └── activity_repository.py
├── utils/             # Helper functions and utilities
│   ├── config.py
│   └── logger.py
└── main.py            # FastAPI application entry point
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Run the Application

#### Development Mode (with auto-reload):

```bash
# Using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or using Python
python -m app.main
```

#### Production Mode (with Gunicorn):

```bash
gunicorn -c gunicorn_config.py app.main:app
```

### 4. Access the API

- **Interactive API docs (Swagger)**: http://localhost:8000/docs
- **Alternative docs (ReDoc)**: http://localhost:8000/redoc
- **Health check**: http://localhost:8000/health

## 📡 API Endpoints

### POST `/api/v1/video/process`

Upload and process a video file for activity detection.

**Request:**
- `video` (file): Video file (multipart/form-data)
- `tripId` (string): Unique trip identifier (required)
- `crewName` (string, optional): Crew member name
- `crewId` (string, optional): Crew member ID
- `crewRole` (int, optional): Crew role (1 = primary pilot)
- `useMockDetection` (bool, optional): Use mock detection for testing

**Response:**

```json
{
  "status": "success",
  "message": "Video processed successfully",
  "tripId": "TRIP-20251110-001",
  "videoFilename": "uploaded_video.mp4",
  "runDirectory": "/path/to/locopilot_evidence/run_20251110_143045",
  "activitiesJsonPath": "/path/to/locopilot_evidence/run_20251110_143045/activities.json",
  "activitiesCount": 5,
  "activities": [...],
  "processingTime": 45.67
}
```

**Example cURL:**

```bash
curl -X POST "http://localhost:8000/api/v1/video/process" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@example_data/latest.mp4" \
  -F "tripId=TRIP-20251110-001" \
  -F "crewName=John Doe" \
  -F "crewId=C-001" \
  -F "crewRole=1"
```

**Example Python:**

```python
import requests

url = "http://localhost:8000/api/v1/video/process"

files = {
    'video': open('example_data/latest.mp4', 'rb')
}

data = {
    'tripId': 'TRIP-20251110-001',
    'crewName': 'John Doe',
    'crewId': 'C-001',
    'crewRole': 1,
    'useMockDetection': False
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

### GET `/api/v1/video/status/{run_id}`

Get processing status for a specific run.

**Parameters:**
- `run_id` (string): Run directory name (e.g., `run_20251110_143045`)

**Response:**

```json
{
  "status": "completed",
  "message": "Processing completed",
  "activitiesCount": 5,
  "summary": {
    "total_activities": 5,
    "activity_breakdown": {
      "Using mobile phone": 2,
      "Writing activity detected": 3
    },
    "total_duration": 67.5
  }
}
```

### GET `/api/v1/video/health`

Health check for the video processing service.

**Response:**

```json
{
  "status": "healthy",
  "service": "video-processing",
  "version": "1.0.0",
  "config": {
    "max_upload_size_mb": 500,
    "allowed_extensions": [".mp4", ".avi", ".mov", ".mkv"],
    "sample_fps": 0.5,
    "output_dir": "locopilot_evidence"
  }
}
```

## 🔧 Configuration

Configuration is managed through environment variables. See `.env.example` for all available options.

**Key Configuration Options:**

- `SAMPLE_FPS`: Frame sampling rate (default: 0.5 FPS = 1 frame every 2 seconds)
- `MAX_UPLOAD_SIZE`: Maximum video file size (default: 500 MB)
- `YOLO_WEIGHTS_PRELOAD`: YOLO model weights file (default: yolo11s.pt)
- `OUTPUT_DIR`: Directory for storing evidence and activities (default: locopilot_evidence)

## 🔍 Output Structure

After processing, the output is organized as follows:

```
locopilot_evidence/
└── run_20251110_143045/
    ├── activities.json          # Structured activity data
    ├── clips/                   # Video clips of detected activities
    │   ├── latest_cell_phone_frame00001250_001_clip.mp4
    │   └── latest_cell_phone_frame00001250_001_activity.jpg
    └── frames/                  # Annotated frames (if enabled)
        └── frame_00001250.jpg
```

### activities.json Format

```json
[
  {
    "tripId": "TRIP-123",
    "activityType": 2,
    "des": "Using mobile phone",
    "objectType": "cell phone",
    "fileUrl": "/path/to/video.mp4",
    "fileDuration": "00:10:30",
    "activityStartTime": "125.50",
    "activityEndTime": "132.75",
    "crewName": "John Doe",
    "crewId": "C-001",
    "crewRole": 1,
    "date": "2025-11-10",
    "time": "14:30:45",
    "filename": "latest.mp4",
    "peopleCount": 1,
    "evidence": {"rule": "phone_in_hand"},
    "activityImage": "latest_cell_phone_frame00001250_001_activity.jpg",
    "activityClip": "latest_cell_phone_frame00001250_001_clip.mp4"
  }
]
```

## 🐳 Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run with Gunicorn
CMD ["gunicorn", "-c", "gunicorn_config.py", "app.main:app"]
```

Build and run:

```bash
docker build -t locopilot-api .
docker run -p 8000:8000 -v $(pwd)/locopilot_evidence:/app/locopilot_evidence locopilot-api
```

## 🧪 Testing

### Test with Mock Detection

```bash
curl -X POST "http://localhost:8000/api/v1/video/process" \
  -F "video=@example_data/latest.mp4" \
  -F "tripId=TEST-001" \
  -F "useMockDetection=true"
```

This will use mock detection instead of running the full ML pipeline, useful for testing the API structure.

## 📊 Performance

The Gunicorn configuration is optimized for video processing:

- **Workers**: CPU count / 2 (multiprocessing)
- **Timeout**: 600 seconds (10 minutes for long videos)
- **Preload**: Application preloading for faster startup
- **Max requests**: 2000 requests per worker (with jitter)

## 🔐 Security Considerations

For production deployment:

1. **Enable authentication**: Add authentication middleware
2. **Restrict CORS**: Set specific origins instead of `*`
3. **File validation**: Validate video files thoroughly
4. **Rate limiting**: Add rate limiting for API endpoints
5. **HTTPS**: Deploy behind a reverse proxy with SSL/TLS

## 📝 Logging

Logs are written to stdout/stderr with structured formatting:

```
2025-11-10 14:30:45 - app.main - INFO - POST /api/v1/video/process - Status: 200 - Time: 45.678s
```

Configure log level via `LOG_LEVEL` environment variable.

## 🛠️ Development

### Add New Activity Detection

1. Update `ActivityTypeEnum` in `app/models/activity_models.py`
2. Add detection logic in `app/services/activity_detection_service.py`
3. Update evidence rules and descriptions

### Extend API

1. Create new router in `app/controllers/`
2. Register router in `app/main.py`
3. Add corresponding services and models

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Gunicorn Documentation](https://docs.gunicorn.org/)
- [Uvicorn Documentation](https://www.uvicorn.org/)

## 📄 License

[Your License Here]


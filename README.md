# Locopilot Monitoring System - FastAPI Application

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-ready FastAPI application** for video processing and activity detection with clean MVC architecture.

## 🎯 Features

- ✅ **RESTful API** for video upload and processing
- ✅ **Activity Detection** (cell phone, sleep, writing, packing, group detection)
- ✅ **Structured Output** (`activities.json` with detailed metadata)
- ✅ **Clean MVC Architecture** (Models, Controllers, Services, Repositories)
- ✅ **Production-Ready** (Gunicorn with multiprocessing)
- ✅ **Comprehensive Documentation** (Swagger UI, ReDoc)
- ✅ **Type-Safe** (Pydantic models with validation)
- ✅ **Extensible** (Easy to add new features)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
# Development mode (auto-reload)
./start_server.sh

# Production mode
gunicorn -c gunicorn_config.py app.main:app
```

### 3. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### 4. Process a Video

```bash
curl -X POST "http://localhost:8000/api/v1/video/process" \
  -F "video=@example_data/latest.mp4" \
  -F "tripId=TRIP-001"
```

## 📁 Project Structure

```
app/
├── models/              # Pydantic schemas
├── controllers/         # API endpoints
├── services/           # Business logic
├── repositories/       # Data persistence
└── utils/             # Configuration & logging

gunicorn_config.py      # Production server config
requirements.txt        # Dependencies
start_server.sh        # Startup script
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/video/process` | Upload and process video |
| `GET` | `/api/v1/video/status/{run_id}` | Get processing status |
| `GET` | `/api/v1/video/health` | Service health check |

## 📊 Response Example

```json
{
  "status": "success",
  "tripId": "TRIP-001",
  "activitiesCount": 3,
  "activities": [
    {
      "tripId": "TRIP-001",
      "activityType": 2,
      "des": "Using mobile phone",
      "activityStartTime": "45.50",
      "activityEndTime": "52.75",
      "evidence": {"rule": "phone_in_hand"}
    }
  ],
  "activitiesJsonPath": "/path/to/activities.json"
}
```

## 🔧 Configuration

Create a `.env` file (optional):

```env
PORT=8000
SAMPLE_FPS=0.5
MAX_UPLOAD_SIZE=524288000  # 500 MB
YOLO_WEIGHTS_PRELOAD=yolo11s.pt
```

## 🐍 Usage Examples

### Python

```python
import requests

with open('video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/video/process',
        files={'video': f},
        data={'tripId': 'TRIP-001'}
    )
    print(response.json())
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('video', videoFile);
formData.append('tripId', 'TRIP-001');

fetch('http://localhost:8000/api/v1/video/process', {
    method: 'POST',
    body: formData
}).then(res => res.json()).then(console.log);
```

## 🐳 Docker Deployment

```bash
docker build -t locopilot-api .
docker run -p 8000:8000 locopilot-api
```

## 📚 Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Installation instructions
- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Architecture details
- **[API_USAGE_GUIDE.md](API_USAGE_GUIDE.md)** - Code examples
- **[README_API.md](README_API.md)** - Detailed API docs
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - What was built

## 🧪 Testing

```bash
# Run test suite
python test_api.py

# Test with mock detection
curl -X POST http://localhost:8000/api/v1/video/process \
  -F "video=@example_data/latest.mp4" \
  -F "tripId=TEST-001" \
  -F "useMockDetection=true"
```

## 🏗️ Architecture

**Clean MVC Pattern:**

- **Models** (`app/models/`) - Pydantic schemas for validation
- **Controllers** (`app/controllers/`) - API route handlers
- **Services** (`app/services/`) - Business logic
- **Repositories** (`app/repositories/`) - Data persistence
- **Utils** (`app/utils/`) - Configuration & logging

## 🔥 Production Features

- ✅ **Gunicorn** with multiprocessing (CPU count / 2 workers)
- ✅ **600-second timeout** for long video processing
- ✅ **Request validation** with Pydantic
- ✅ **Error handling** with proper HTTP responses
- ✅ **Structured logging** with configurable levels
- ✅ **CORS support** for cross-origin requests
- ✅ **Health checks** for monitoring
- ✅ **Auto-documentation** with OpenAPI/Swagger

## ⚙️ Gunicorn Configuration

```python
workers = max(1, multiprocessing.cpu_count() // 2)
threads = 1
preload_app = True
timeout = 600
max_requests = 2000
bind = "0.0.0.0:8000"
raw_env = [
    "YOLO_WEIGHTS_PRELOAD=yolo11s.pt",
    "PRELOAD_OCR=1",
]
```

## 📦 Dependencies

- **FastAPI** - Modern web framework
- **Uvicorn/Gunicorn** - ASGI servers
- **Pydantic** - Data validation
- **OpenCV** - Video processing
- **MediaPipe** - Pose detection
- **Ultralytics YOLO** - Object detection

## 🔐 Security

- Input validation with Pydantic
- File type and size validation
- Configurable CORS
- Error message sanitization
- Request logging

## 🎯 Activity Types

| Code | Activity | Description |
|------|----------|-------------|
| 2 | Cell Phone | Using mobile phone |
| 3 | Microsleep | Micro-sleep (5+ seconds) |
| 4 | Sleep | Sleep (30+ seconds) |
| 5 | Writing | Writing activity |
| 6 | Packing Bags | Packing bags activity |
| 7 | Group | More than 2 people |

## 📄 Output Files

```
locopilot_evidence/
└── run_20251110_143045/
    ├── activities.json              # Activity data
    └── clips/                       # Video clips & screenshots
        ├── *_clip.mp4
        └── *_activity.jpg
```

## 🤝 Contributing

1. Follow MVC architecture
2. Add docstrings to all functions
3. Use type hints
4. Update documentation
5. Test your changes

## 📝 License

[Your License Here]

## 🆘 Support

- Check **[SETUP_GUIDE.md](SETUP_GUIDE.md)** for installation help
- See **[API_USAGE_GUIDE.md](API_USAGE_GUIDE.md)** for examples
- Visit http://localhost:8000/docs for interactive docs

## ✨ Key Highlights

- 🎨 **Clean Code** - Well-organized MVC architecture
- 📝 **Type-Safe** - Pydantic models throughout
- 🚀 **Fast** - Async FastAPI with multiprocessing
- 📚 **Documented** - Comprehensive docs and examples
- 🧪 **Tested** - Test suite included
- 🔧 **Configurable** - Environment-based settings
- 🌐 **Production-Ready** - Gunicorn configuration included

---

**Status:** ✅ Production-Ready

**Built with:** FastAPI, Pydantic, Gunicorn, OpenCV, MediaPipe, YOLO

**Last Updated:** November 10, 2025

**Get Started:** `./start_server.sh` → http://localhost:8000/docs


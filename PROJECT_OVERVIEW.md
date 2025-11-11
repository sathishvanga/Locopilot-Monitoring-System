# Locopilot Monitoring System - Project Overview

## 📋 Summary

A **production-ready FastAPI application** for video processing and activity detection, built with clean MVC architecture. The system accepts video uploads, processes them for activity detection (cell phone usage, sleep, writing, etc.), and outputs structured data in `activities.json`.

## 🏗️ Architecture

```
Locopilot Monitoring System/
│
├── app/                          # Main application package
│   ├── __init__.py
│   ├── main.py                   # FastAPI application entry point
│   │
│   ├── models/                   # Pydantic schemas & domain models
│   │   ├── __init__.py
│   │   ├── activity_models.py    # Activity data models
│   │   └── video_models.py       # Request/response models
│   │
│   ├── controllers/              # API route handlers (routers)
│   │   ├── __init__.py
│   │   └── video_controller.py   # Video processing endpoints
│   │
│   ├── services/                 # Business logic
│   │   ├── __init__.py
│   │   ├── video_processing_service.py    # Video processing orchestration
│   │   └── activity_detection_service.py  # Activity detection logic
│   │
│   ├── repositories/             # Data persistence & file I/O
│   │   ├── __init__.py
│   │   └── activity_repository.py         # Activities.json management
│   │
│   └── utils/                    # Utilities & helpers
│       ├── __init__.py
│       ├── config.py             # Configuration management
│       └── logger.py             # Logging utilities
│
├── locopilot_monitor.py          # Original video processing logic (can be integrated)
│
├── gunicorn_config.py            # Gunicorn production configuration
├── start_server.sh               # Server startup script
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment configuration template
│
├── README_API.md                 # Detailed API documentation
├── API_USAGE_GUIDE.md            # Quick API usage guide
└── PROJECT_OVERVIEW.md           # This file

```

## ✨ Key Features

### 1. **Clean MVC Architecture**
- **Models**: Pydantic schemas for validation and serialization
- **Controllers**: RESTful API endpoints with proper error handling
- **Services**: Business logic separated from HTTP layer
- **Repositories**: File I/O and data persistence

### 2. **Production-Ready**
- ✅ Gunicorn with multiprocessing
- ✅ Proper error handling and logging
- ✅ Request validation with Pydantic
- ✅ CORS support
- ✅ Health check endpoints
- ✅ Comprehensive documentation

### 3. **Video Processing**
- Upload videos via REST API
- Activity detection (cell phone, sleep, writing, packing bags, group detection)
- Generate structured `activities.json` output
- Save video clips and screenshots of detected activities

### 4. **Scalable & Extensible**
- Easy to add new activity types
- Mock detection mode for testing
- Configurable via environment variables
- Can integrate with existing ML pipelines

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
# Development mode (with auto-reload)
./start_server.sh

# Or manually
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### 4. Process a Video

```bash
curl -X POST "http://localhost:8000/api/v1/video/process" \
  -F "video=@example_data/latest.mp4" \
  -F "tripId=TRIP-001" \
  -F "crewName=John Doe" \
  -F "crewId=C-001" \
  -F "crewRole=1"
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/video/process` | Upload and process video |
| `GET` | `/api/v1/video/status/{run_id}` | Get processing status |
| `GET` | `/api/v1/video/health` | Service health check |
| `GET` | `/health` | Application health check |
| `GET` | `/docs` | Interactive API documentation |

## 📁 Output Structure

```
locopilot_evidence/
└── run_20251110_143045/
    ├── activities.json                                    # Structured activity data
    ├── clips/
    │   ├── video_cell_phone_frame00001250_001_clip.mp4   # Activity video clips
    │   └── video_cell_phone_frame00001250_001_activity.jpg  # Activity screenshots
    └── frames/                                            # Optional annotated frames
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

## 🔧 Configuration

Configure via `.env` file (copy from `.env.example`):

```env
# Application
APP_NAME="Locopilot Monitoring System"
DEBUG=false
LOG_LEVEL=INFO

# Server
HOST=0.0.0.0
PORT=8000

# Processing
SAMPLE_FPS=0.5                    # Frame sampling rate
MAX_UPLOAD_SIZE=524288000         # 500 MB
YOLO_WEIGHTS_PRELOAD=yolo11s.pt

# Output
OUTPUT_DIR=locopilot_evidence
SAVE_ANNOTATED_FRAMES=false
```

## 🐳 Production Deployment

### With Gunicorn (Recommended)

```bash
gunicorn -c gunicorn_config.py app.main:app
```

**Gunicorn Configuration Highlights:**
- Workers: CPU count / 2 (multiprocessing)
- Timeout: 600 seconds (for long video processing)
- Preload app: True (faster startup)
- Max requests: 2000 per worker

### With Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-c", "gunicorn_config.py", "app.main:app"]
```

Build and run:
```bash
docker build -t locopilot-api .
docker run -p 8000:8000 -v $(pwd)/locopilot_evidence:/app/locopilot_evidence locopilot-api
```

## 🔬 Testing

### Test with Mock Detection (Fast)

```bash
curl -X POST "http://localhost:8000/api/v1/video/process" \
  -F "video=@example_data/latest.mp4" \
  -F "tripId=TEST-001" \
  -F "useMockDetection=true"
```

### Test with Real Detection (Full Pipeline)

```bash
curl -X POST "http://localhost:8000/api/v1/video/process" \
  -F "video=@example_data/latest.mp4" \
  -F "tripId=TEST-002" \
  -F "useMockDetection=false"
```

## 📊 Activity Types

| Code | Type | Description | Evidence Rule |
|------|------|-------------|---------------|
| 2 | Cell Phone | Using mobile phone | phone_in_hand |
| 3 | Microsleep | Micro-sleep (5+ seconds) | eyes_closed_5s_or_pose_indicators |
| 4 | Sleep | Sleep (30+ seconds) | eyes_closed_30s_or_pose_indicators |
| 5 | Writing | Writing activity | hand_near_book |
| 6 | Packing Bags | Packing bags activity | hand_near_backpack |
| 7 | Group Detected | More than 2 people | more_than_2_deduplicated_persons |

## 🛠️ Development Guide

### Add New Activity Type

1. **Update model** (`app/models/activity_models.py`):
```python
class ActivityTypeEnum(IntEnum):
    NEW_ACTIVITY = 8
```

2. **Add detection logic** (`app/services/activity_detection_service.py`):
```python
self.activity_descriptions['new_activity'] = 'New activity detected'
self.evidence_rules['new_activity'] = 'detection_rule'
```

3. **Update detection service** with actual ML logic

### Add New Endpoint

1. Create new controller in `app/controllers/`
2. Register router in `app/main.py`:
```python
from .controllers import new_router
app.include_router(new_router)
```

### Integrate Existing Video Processing

Replace mock detection with real processing in `app/services/activity_detection_service.py`:

```python
def detect_activities_real(self, video_path, trip_id, ...):
    from locopilot_monitor import LocopilotActivityMonitor
    
    monitor = LocopilotActivityMonitor(video_path, ...)
    monitor.trip_id = trip_id
    monitor.process_video()
    
    return monitor.all_activities
```

## 📚 Documentation

- **README_API.md** - Comprehensive API documentation
- **API_USAGE_GUIDE.md** - Quick usage examples in multiple languages
- **Swagger UI** - Interactive API docs at `/docs`
- **ReDoc** - Alternative docs at `/redoc`

## 🔐 Security Considerations

For production:
1. **Add authentication** - JWT or API keys
2. **Restrict CORS** - Specific origins only
3. **Rate limiting** - Prevent abuse
4. **File validation** - Thorough video validation
5. **HTTPS** - Deploy behind reverse proxy
6. **Input sanitization** - Validate all inputs

## 🎯 Best Practices Implemented

✅ **Separation of Concerns** - Clean MVC architecture  
✅ **Type Safety** - Pydantic models with validation  
✅ **Error Handling** - Comprehensive exception handling  
✅ **Logging** - Structured logging throughout  
✅ **Documentation** - Docstrings and API docs  
✅ **Configuration** - Environment-based settings  
✅ **Testing** - Mock detection for testing  
✅ **Scalability** - Multiprocessing with Gunicorn  
✅ **Maintainability** - Modular and extensible  

## 🔄 Integration with Existing Code

The existing `locopilot_monitor.py` can be integrated into the service layer:

```python
# app/services/activity_detection_service.py
from locopilot_monitor import LocopilotActivityMonitor

def detect_activities_real(self, video_path, trip_id, ...):
    monitor = LocopilotActivityMonitor(
        video_path=video_path,
        output_dir=output_dir,
        save_annotated_frames=False,
        sample_fps=sample_fps
    )
    monitor.trip_id = trip_id
    monitor.crew_name = crew_name
    monitor.crew_id = crew_id
    monitor.crew_role = crew_role
    monitor.process_video()
    return monitor.all_activities
```

## 📈 Performance

- **Frame sampling**: 0.5 FPS (configurable) = 60x faster processing
- **Multiprocessing**: Uses CPU count / 2 workers
- **Timeout**: 600 seconds for long videos
- **Request pooling**: 2000 requests per worker

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | `pip install -r requirements.txt` |
| YOLO weights missing | Will auto-download or place `yolo11s.pt` in root |
| Timeout errors | Increase `timeout` in `gunicorn_config.py` |
| Large videos fail | Increase `MAX_UPLOAD_SIZE` in `.env` |
| Port already in use | Change `PORT` in `.env` |

## 📝 License

[Your License Here]

## 🤝 Contributing

1. Follow the MVC architecture
2. Add docstrings to all functions
3. Update tests and documentation
4. Use type hints
5. Follow PEP 8 style guide

## 🎓 Learning Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Gunicorn Documentation](https://docs.gunicorn.org/)

---

**Ready to deploy!** 🚀

For questions or support, refer to the documentation or check the code comments.


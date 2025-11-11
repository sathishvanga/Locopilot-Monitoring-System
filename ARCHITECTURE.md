# System Architecture - Locopilot Monitoring System

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
│  (Browser, Mobile App, External Service, cURL, Python, etc.)    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               │ HTTP/REST
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GUNICORN (WSGI Server)                      │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  │ Worker N │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                  │
│  Workers = max(1, CPU count / 2)                                │
│  Timeout = 600s | Max Requests = 2000                           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FASTAPI APPLICATION                           │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  Middleware Layer                          │  │
│  │  • CORS • Logging • Error Handling • Request Validation  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                               │                                  │
│                               ▼                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    API Routers                            │  │
│  │                                                           │  │
│  │  POST /api/v1/video/process                              │  │
│  │  GET  /api/v1/video/status/{run_id}                      │  │
│  │  GET  /api/v1/video/health                               │  │
│  │  GET  /health                                             │  │
│  │  GET  /docs (Swagger UI)                                  │  │
│  │  GET  /redoc                                              │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         MVC LAYERS                               │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ MODELS (Pydantic Schemas)                                │   │
│  │  • VideoUploadRequest                                    │   │
│  │  • VideoProcessingResponse                               │   │
│  │  • ActivityModel                                         │   │
│  │  • ActivityTypeEnum                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                               │                                  │
│                               ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ CONTROLLERS (Route Handlers)                             │   │
│  │  • video_controller.py                                   │   │
│  │    - process_video()                                     │   │
│  │    - get_processing_status()                             │   │
│  │    - health_check()                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                               │                                  │
│                               ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ SERVICES (Business Logic)                                │   │
│  │                                                          │   │
│  │  VideoProcessingService                                  │   │
│  │   • validate_video_file()                                │   │
│  │   • save_uploaded_video()                                │   │
│  │   • process_video()                                      │   │
│  │                                                          │   │
│  │  ActivityDetectionService                                │   │
│  │   • detect_activities_mock()                             │   │
│  │   • detect_activities_real()                             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                               │                                  │
│                               ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ REPOSITORIES (Data Persistence)                          │   │
│  │                                                          │   │
│  │  ActivityRepository                                      │   │
│  │   • save_activities()                                    │   │
│  │   • load_activities()                                    │   │
│  │   • create_run_directory()                               │   │
│  │   • get_activity_summary()                               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                               │                                  │
│                               ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ UTILS (Helpers)                                          │   │
│  │  • Config (Environment variables)                        │   │
│  │  • Logger (Structured logging)                           │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL DEPENDENCIES                         │
│                                                                  │
│  ┌───────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  OpenCV   │  │MediaPipe │  │   YOLO   │  │LocopilotMon. │  │
│  │   (CV)    │  │  (Pose)  │  │ (Detect) │  │   (Legacy)   │  │
│  └───────────┘  └──────────┘  └──────────┘  └──────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FILE SYSTEM                                 │
│                                                                  │
│  uploads/                     locopilot_evidence/               │
│    └── {tripId}_{timestamp}.mp4  └── run_{timestamp}/           │
│                                      ├── activities.json         │
│                                      └── clips/                  │
│                                          ├── *_clip.mp4          │
│                                          └── *_activity.jpg      │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Request Flow

```
1. Client Request
   │
   ▼
2. Gunicorn Worker receives request
   │
   ▼
3. FastAPI Application
   │
   ├─► Middleware (CORS, Logging, Validation)
   │
   ▼
4. Router/Controller
   │
   ├─► video_controller.process_video()
   │
   ▼
5. Service Layer
   │
   ├─► VideoProcessingService.validate_video_file()
   ├─► VideoProcessingService.save_uploaded_video()
   ├─► ActivityDetectionService.detect_activities()
   │
   ▼
6. Repository Layer
   │
   ├─► ActivityRepository.save_activities()
   ├─► ActivityRepository.create_run_directory()
   │
   ▼
7. File System
   │
   ├─► Write activities.json
   ├─► Save video clips
   ├─► Save screenshots
   │
   ▼
8. Response Generation
   │
   ├─► Create VideoProcessingResponse
   ├─► Serialize with Pydantic
   │
   ▼
9. Return JSON Response to Client
```

## 📦 Component Responsibilities

### Controllers (API Layer)
**Responsibility:** Handle HTTP requests/responses
- Receive incoming requests
- Validate request format
- Call service layer
- Format responses
- Handle HTTP errors

### Services (Business Logic)
**Responsibility:** Orchestrate business operations
- Video validation
- File management
- Activity detection orchestration
- Error handling
- Business rules enforcement

### Repositories (Data Access)
**Responsibility:** Data persistence
- Save/load activities.json
- Manage file I/O
- Directory management
- Data validation

### Models (Data Structures)
**Responsibility:** Data validation and serialization
- Request/response schemas
- Activity models
- Type definitions
- Validation rules

### Utils (Helpers)
**Responsibility:** Cross-cutting concerns
- Configuration management
- Logging
- Common utilities

## 🔐 Security Layers

```
┌─────────────────────────────────────────┐
│         Input Validation                │
│  • File size limits                     │
│  • File type validation                 │
│  • Request validation (Pydantic)        │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         Error Handling                  │
│  • HTTP exception handlers              │
│  • Validation error handlers            │
│  • General exception handlers           │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         Logging & Monitoring            │
│  • Request/response logging             │
│  • Error logging with stack traces      │
│  • Performance metrics                  │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         CORS & Headers                  │
│  • Configurable CORS origins            │
│  • Security headers                     │
│  • Content-Type validation              │
└─────────────────────────────────────────┘
```

## 🔄 Data Flow

```
Video Upload
    │
    ├─► Multipart Form Data
    │   ├─► video: File
    │   ├─► tripId: String
    │   └─► metadata: Form Fields
    │
    ▼
Validation
    │
    ├─► File size check (max 500 MB)
    ├─► File type check (.mp4, .avi, .mov, .mkv)
    └─► Pydantic model validation
    │
    ▼
Processing
    │
    ├─► Save video to uploads/
    ├─► Run activity detection
    │   ├─► Frame sampling (0.5 FPS)
    │   ├─► Object detection (YOLO)
    │   ├─► Pose detection (MediaPipe)
    │   └─► Activity classification
    │
    ▼
Output Generation
    │
    ├─► Create run directory
    ├─► Generate activities.json
    ├─► Save video clips
    └─► Save screenshots
    │
    ▼
Response
    │
    └─► Return JSON with:
        ├─► Processing status
        ├─► Activities list
        ├─► File paths
        └─► Metadata
```

## 📊 Scalability

### Horizontal Scaling
```
┌─────────────────┐
│  Load Balancer  │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│ App 1 │ │ App 2 │
└───┬───┘ └──┬────┘
    │        │
    └────┬───┘
         │
┌────────▼────────┐
│ Shared Storage  │
└─────────────────┘
```

### Vertical Scaling
- Increase worker count (CPU cores)
- Increase memory for larger videos
- Adjust timeout for longer processing

## 🔌 Integration Points

### External Systems
- **Frontend Applications** → REST API
- **Mobile Apps** → REST API
- **Backend Services** → REST API
- **ML Pipelines** → Service Layer
- **Monitoring Tools** → Health Endpoints

### Internal Integration
- Existing `locopilot_monitor.py` can be integrated into service layer
- Modular design allows easy plugin of new detection algorithms
- Repository pattern allows switching storage backends

## 🎯 Design Patterns

1. **MVC Pattern** - Separation of concerns
2. **Repository Pattern** - Data access abstraction
3. **Dependency Injection** - Loose coupling
4. **Factory Pattern** - Object creation (Pydantic models)
5. **Singleton Pattern** - Configuration management
6. **Strategy Pattern** - Mock vs Real detection

## 📈 Performance Optimization

```
┌─────────────────────────────────────────┐
│     Application Level                   │
│  • Frame sampling (60x faster)          │
│  • Async I/O operations                 │
│  • Lazy loading of models               │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│     Server Level                        │
│  • Multiprocessing (Gunicorn)           │
│  • Worker pooling                       │
│  • Request queuing                      │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│     Infrastructure Level                │
│  • Load balancing                       │
│  • Caching                              │
│  • CDN for static files                 │
└─────────────────────────────────────────┘
```

## 🧪 Testing Strategy

```
┌─────────────────────────────────────────┐
│     Unit Tests                          │
│  • Model validation                     │
│  • Service methods                      │
│  • Repository operations                │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│     Integration Tests                   │
│  • API endpoints                        │
│  • Service integration                  │
│  • Database operations                  │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│     End-to-End Tests                    │
│  • Complete request flow                │
│  • File upload/download                 │
│  • Error scenarios                      │
└─────────────────────────────────────────┘
```

---

**Architecture:** Clean MVC with layered separation
**Scalability:** Horizontal and vertical scaling supported
**Maintainability:** Modular design with clear responsibilities
**Extensibility:** Easy to add new features and integrations


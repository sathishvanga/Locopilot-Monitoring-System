# Concurrent Video Processing Architecture with GPU Optimization

## Design Decisions (Confirmed)
- **Queue System:** In-Process `asyncio.Queue` (no external dependencies)
- **Persistence:** In-Memory Only (simplest, jobs lost on restart)
- **API Strategy:** Replace existing sync endpoint with async (breaking change)
- **Target GPU:** 16GB VRAM (RTX 4080/A4000) - supports 3 concurrent videos

## Overview

Transform the Locopilot Monitoring System from synchronous, CPU-only video processing to a concurrent, GPU-accelerated architecture that can process multiple videos simultaneously while efficiently utilizing 16 GB GPU memory.

## Current State Analysis

### Architecture Flow
```
POST /api/video/analyze → Save video → Synchronous processing → Return results (blocks)
```

### Key Limitations
| Issue | Current State | Impact |
|-------|--------------|--------|
| Synchronous API | Blocks until video complete | Cannot process concurrent uploads |
| CPU-only default | `yolo_device="cpu"` | Slow inference (~1.5s/frame) |
| Memory multiplication | Each worker loads models (~1.5GB) | 8 workers = 12GB RAM |
| No job queue | Direct processing | No prioritization or backpressure |
| No GPU memory management | No pool, no OOM handling | Potential crashes |

### Current Models & Memory
- **YOLO11m**: ~500-600 MB GPU memory (object detection)
- **YOLO11m-Pose**: ~500-700 MB GPU memory (pose estimation)
- **MediaPipe FaceMesh**: ~100-150 MB (eye aspect ratio)
- **Total**: ~1.2-1.5 GB for shared models

---

## Proposed Architecture

### High-Level Design
```
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                          │
│                   (Async Endpoints + Job Manager)                   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐
│   Job Queue     │   │  Job Tracker    │   │  Status/Results API     │
│ (asyncio.Queue) │   │ (In-Memory +    │   │  GET /jobs/{id}         │
│                 │   │  File-Backed)   │   │  GET /jobs/{id}/result  │
└────────┬────────┘   └────────┬────────┘   └─────────────────────────┘
         │                     │
         └──────────┬──────────┘
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      GPU Resource Manager                           │
│              (Singleton Model Loader + Memory Pool)                 │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │   YOLO11m   │  │ YOLO11m-Pose│  │  FaceMesh   │  (Shared)       │
│  │   ~600MB    │  │    ~700MB   │  │   ~150MB    │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│                                                                     │
│  Semaphore: max_concurrent_videos = 3                               │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  GPU Worker 1   │   │  GPU Worker 2   │   │  GPU Worker 3   │
│   (Video A)     │   │   (Video B)     │   │   (Video C)     │
│  CUDA Stream 1  │   │  CUDA Stream 2  │   │  CUDA Stream 3  │
└─────────────────┘   └─────────────────┘   └─────────────────┘
```

### Memory Budget (16 GB GPU)
| Component | Memory |
|-----------|--------|
| Shared Models (YOLO + Pose + FaceMesh) | 1.5 GB |
| Video A inference buffers | 3.0 GB |
| Video B inference buffers | 3.0 GB |
| Video C inference buffers | 3.0 GB |
| Fragmentation buffer (20%) | 3.2 GB |
| **Total** | **13.7 GB** |
| **Headroom** | **2.3 GB** |

---

## Key Components

### 1. GPU Resource Manager (Singleton)
- Loads models ONCE at application startup
- Manages GPU memory pool with PyTorch CUDA allocator
- Controls concurrency via asyncio.Semaphore (max 3 videos)
- Handles OOM recovery with batch size reduction

### 2. Job Queue & Manager
- `asyncio.Queue` with bounded size (10 jobs) for backpressure
- In-memory job tracker (no persistence - jobs lost on restart)
- Job states: `PENDING → QUEUED → PROCESSING → COMPLETED/FAILED`
- Progress tracking with percentage completion
- Background worker tasks consume from queue

### 3. Async API Endpoints (Replaces Existing Sync Endpoints)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/video/analyze` | POST | Submit video, returns job_id immediately (ASYNC - breaking change) |
| `/api/video/jobs/{job_id}` | GET | Get job status and progress |
| `/api/video/jobs/{job_id}/result` | GET | Get completed results (activities JSON) |
| `/api/video/jobs/{job_id}/cancel` | POST | Cancel pending/running job |
| `/api/video/queue/status` | GET | Get queue depth and active jobs |
| `/api/gpu/status` | GET | GPU memory usage and active video count |

### 4. Frame Batching Strategy
- Batch size: 8 frames for YOLO inference
- Memory per batch: ~2.5 GB
- Dynamic batch reduction on OOM (8 → 4 → 2 → 1)

---

## Implementation Phases

### Phase 1: GPU-Enabled Singleton Model Loader
**Goal:** Enable GPU inference with shared models

**New Files:**
- `app/services/gpu_resource_manager.py` (~250 lines)

**Modified Files:**
- `app/utils/config.py` - Add GPU configuration
- `app/main.py` - Initialize GPU manager at startup
- `locopilot_monitor.py` - Accept preloaded models parameter
- `app/services/activity_detection_service.py` - Use GPU manager

**Key Implementation:**
```python
# gpu_resource_manager.py
class GPUResourceManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.device = torch.device("cuda:0")
        self.yolo_model = YOLO("yolo11m.pt").to(self.device)
        self.yolo_model.fuse()
        self.pose_model = YOLO("yolo11m-pose.pt").to(self.device)
        self.pose_model.fuse()
        self._semaphore = asyncio.Semaphore(3)
```

### Phase 2: Async Job Queue System
**Goal:** Non-blocking API with job tracking (in-memory only)

**New Files:**
- `app/services/job_manager.py` (~250 lines)
- `app/models/job_models.py` (~100 lines)

**Modified Files:**
- `app/controllers/video_controller.py` - Replace sync `/api/video/analyze` with async (breaking change)

**Key Implementation:**
```python
# job_manager.py
class JobManager:
    def __init__(self, max_queue_size: int = 10, num_workers: int = 3):
        self._queue = asyncio.Queue(maxsize=max_queue_size)
        self._jobs: Dict[str, Job] = {}  # In-memory only
        self._workers: List[asyncio.Task] = []
        self._num_workers = num_workers

    async def submit_job(self, video_path: str, config: dict) -> str:
        job_id = generate_job_id()
        job = Job(id=job_id, video_path=video_path, status=JobStatus.QUEUED)
        self._jobs[job_id] = job
        await self._queue.put(job)
        return job_id

    async def start_workers(self, gpu_manager: GPUResourceManager):
        """Start background worker tasks on app startup"""
        for i in range(self._num_workers):
            task = asyncio.create_task(self._worker_loop(gpu_manager, i))
            self._workers.append(task)
```

### Phase 3: Concurrent Video Processing
**Goal:** Process multiple videos simultaneously

**New Files:**
- `app/services/video_worker.py` (~200 lines)

**Modified Files:**
- `app/services/gpu_resource_manager.py` - Add CUDA streams
- `app/utils/video_multiprocessing.py` - Adapt for GPU sharing

**Key Implementation:**
```python
# video_worker.py
async def process_video_worker(job: Job, gpu_manager: GPUResourceManager):
    async with gpu_manager.acquire_gpu_slot():
        try:
            models = gpu_manager.get_models()
            monitor = LocopilotActivityMonitor(preloaded_models=models)
            activities = await asyncio.to_thread(
                monitor.process_video, job.video_path
            )
            job.result = activities
            job.status = JobStatus.COMPLETED
        except torch.cuda.OutOfMemoryError:
            await handle_oom_recovery(job, gpu_manager)
```

### Phase 4: Memory Optimization & Monitoring
**Goal:** Production hardening and observability

**Modified Files:**
- `app/services/gpu_resource_manager.py` - Add monitoring
- `app/main.py` - Add health checks and cleanup

**Key Implementation:**
```python
# Memory monitoring endpoint
@app.get("/api/gpu/status")
async def get_gpu_status():
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "cached_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "active_videos": gpu_manager.active_count,
        "queue_depth": job_manager.queue_size
    }
```

---

## Configuration Additions

```python
# app/utils/config.py additions

# GPU Settings
gpu_enabled: bool = bool(int(os.getenv("GPU_ENABLED", "1")))
gpu_device: str = os.getenv("GPU_DEVICE", "cuda:0")
gpu_memory_fraction: float = float(os.getenv("GPU_MEMORY_FRACTION", "0.85"))

# Concurrency Settings
max_concurrent_videos: int = int(os.getenv("MAX_CONCURRENT_VIDEOS", "3"))
inference_batch_size: int = int(os.getenv("INFERENCE_BATCH_SIZE", "8"))
job_queue_max_size: int = int(os.getenv("JOB_QUEUE_MAX_SIZE", "10"))

# Memory Management
pytorch_cuda_alloc_conf: str = os.getenv(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True"
)
oom_retry_enabled: bool = bool(int(os.getenv("OOM_RETRY_ENABLED", "1")))
```

---

## Files to Create/Modify

### New Files
| File | Purpose | Est. Lines |
|------|---------|------------|
| `app/services/gpu_resource_manager.py` | Singleton model loader, memory pool, semaphore concurrency | ~250 |
| `app/services/job_manager.py` | In-memory async job queue with worker pool | ~250 |
| `app/services/video_worker.py` | GPU worker task, OOM recovery | ~150 |
| `app/models/job_models.py` | Job state enum, request/response Pydantic models | ~100 |

### Modified Files
| File | Changes |
|------|---------|
| `app/utils/config.py` | Add GPU and concurrency settings |
| `app/main.py` | Initialize GPU manager, job workers at startup |
| `app/controllers/video_controller.py` | Replace sync endpoints with async job-based API |
| `app/services/activity_detection_service.py` | Use GPU manager for models |
| `locopilot_monitor.py` | Accept preloaded GPU models, batch inference |
| `app/utils/video_multiprocessing.py` | Adapt worker_initializer for GPU sharing |

---

## Expected Performance

| Metric | Current (CPU) | Target (GPU, 3 concurrent) |
|--------|---------------|---------------------------|
| Single video throughput | ~0.5x real-time | ~3-5x real-time |
| Concurrent capacity | 1 video | 3 videos |
| API response time | Blocks until complete | <500ms (job submission) |
| Memory efficiency | 12GB (8 workers) | ~14GB (3 videos + shared) |
| GPU utilization | 0% | 70-90% |

---

## Industry References

This architecture incorporates patterns from:
- [NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) - Dynamic batching, concurrent model execution
- [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) - Scalable model serving with batching
- [FastAPI Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/) - Async processing patterns
- [PyTorch CUDA Memory Management](https://docs.pytorch.org/docs/stable/notes/cuda.html) - Memory pool configuration
- [GPU Memory Optimization](https://www.runpod.io/articles/guides/avoid-oom-crashes-for-large-models) - OOM prevention strategies

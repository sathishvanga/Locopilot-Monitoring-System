# Task 0001: GPU-Enabled Singleton Model Loader

## Phase
Phase 1

## Goal
Enable GPU inference with shared models

## New Files
- `app/services/gpu_resource_manager.py` (~250 lines)

## Modified Files
- `app/utils/config.py` - Add GPU configuration
- `app/main.py` - Initialize GPU manager at startup
- `locopilot_monitor.py` - Accept preloaded models parameter
- `app/services/activity_detection_service.py` - Use GPU manager

## Key Implementation
```python
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

## Configuration Additions
```python
# GPU Settings
gpu_enabled: bool = bool(int(os.getenv("GPU_ENABLED", "1")))
gpu_device: str = os.getenv("GPU_DEVICE", "cuda:0")
gpu_memory_fraction: float = float(os.getenv("GPU_MEMORY_FRACTION", "0.85"))

# Concurrency Settings
max_concurrent_videos: int = int(os.getenv("MAX_CONCURRENT_VIDEOS", "3"))
inference_batch_size: int = int(os.getenv("INFERENCE_BATCH_SIZE", "8"))
job_queue_max_size: int = int(os.getenv("JOB_QUEUE_MAX_SIZE", "10"))

# Memory Management
pytorch_cuda_alloc_conf: str = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
oom_retry_enabled: bool = bool(int(os.getenv("OOM_RETRY_ENABLED", "1")))
```

## Acceptance Criteria
- [ ] GPU Resource Manager singleton initializes correctly
- [ ] Models load once at startup and are shared
- [ ] Semaphore limits concurrent video processing to 3
- [ ] Configuration values properly loaded from environment

# Task 0003: Concurrent Video Processing

## Phase
Phase 3

## Goal
Process multiple videos simultaneously

## New Files
- `app/services/video_worker.py` (~200 lines)

## Modified Files
- `app/services/gpu_resource_manager.py` - Add CUDA streams
- `app/utils/video_multiprocessing.py` - Adapt for GPU sharing

## Key Implementation
```python
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

## Memory Budget (16 GB GPU)
| Component | Memory |
|-----------|--------|
| Shared Models (YOLO + Pose + FaceMesh) | 1.5 GB |
| Video A inference buffers | 3.0 GB |
| Video B inference buffers | 3.0 GB |
| Video C inference buffers | 3.0 GB |
| Fragmentation buffer (20%) | 3.2 GB |
| **Total** | **13.7 GB** |

## Frame Batching Strategy
- Batch size: 8 frames for YOLO inference
- Memory per batch: ~2.5 GB
- Dynamic batch reduction on OOM (8 → 4 → 2 → 1)

## Acceptance Criteria
- [ ] Multiple videos can process simultaneously (up to 3)
- [ ] CUDA streams provide parallel execution
- [ ] Workers properly acquire and release GPU slots
- [ ] OOM errors trigger batch size reduction

# Task 0004: Memory Optimization & Monitoring

## Phase
Phase 4

## Goal
Production hardening and observability

## Modified Files
- `app/services/gpu_resource_manager.py` - Add monitoring
- `app/main.py` - Add health checks and cleanup

## Key Implementation
```python
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

## API Endpoint
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/gpu/status` | GET | GPU memory usage and active video count |

## Expected Performance
| Metric | Current (CPU) | Target (GPU, 3 concurrent) |
|--------|---------------|---------------------------|
| Single video throughput | ~0.5x real-time | ~3-5x real-time |
| Concurrent capacity | 1 video | 3 videos |
| API response time | Blocks until complete | <500ms (job submission) |
| Memory efficiency | 12GB (8 workers) | ~14GB (3 videos + shared) |
| GPU utilization | 0% | 70-90% |

## Acceptance Criteria
- [ ] `/api/gpu/status` endpoint returns accurate metrics
- [ ] Health checks detect GPU availability issues
- [ ] Cleanup runs after job completion
- [ ] Graceful shutdown stops workers properly

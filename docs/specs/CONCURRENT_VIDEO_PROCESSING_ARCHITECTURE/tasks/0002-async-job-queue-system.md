# Task 0002: Async Job Queue System

## Phase
Phase 2

## Goal
Non-blocking API with job tracking (in-memory only)

## New Files
- `app/services/job_manager.py` (~250 lines)
- `app/models/job_models.py` (~100 lines)

## Modified Files
- `app/controllers/video_controller.py` - Replace sync `/api/video/analyze` with async (breaking change)

## Key Implementation
```python
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

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/video/analyze` | POST | Submit video, returns job_id immediately (ASYNC - breaking change) |
| `/api/video/jobs/{job_id}` | GET | Get job status and progress |
| `/api/video/jobs/{job_id}/result` | GET | Get completed results (activities JSON) |
| `/api/video/jobs/{job_id}/cancel` | POST | Cancel pending/running job |
| `/api/video/queue/status` | GET | Get queue depth and active jobs |

## Job States
`PENDING → QUEUED → PROCESSING → COMPLETED/FAILED`

## Acceptance Criteria
- [ ] Jobs can be submitted and receive immediate job_id response
- [ ] Job status can be queried with progress percentage
- [ ] Completed job results can be retrieved
- [ ] Jobs can be cancelled when pending or running
- [ ] Queue status endpoint shows depth and active jobs

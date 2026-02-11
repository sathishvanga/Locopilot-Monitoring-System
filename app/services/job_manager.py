"""
Job Manager Service - Async job queue system for video processing

Implements a producer-consumer pattern with asyncio for managing video
processing jobs. Provides job submission, status tracking, cancellation,
and graceful shutdown capabilities.

Usage:
    from app.services.job_manager import job_manager

    # Submit a job
    job_id = await job_manager.submit_job("/path/to/video.mp4", {"trip_id": "TRIP-001"})

    # Get job status
    job = await job_manager.get_job(job_id)

    # Cancel a job
    cancelled = await job_manager.cancel_job(job_id)
"""

import asyncio
import uuid
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional, Callable, Any, Awaitable

from ..models.job_models import Job, JobStatus
from ..utils.logger import get_logger
from ..utils.config import get_settings


logger = get_logger(__name__)
settings = get_settings()


class JobManager:
    """
    Async job queue manager for video processing operations.

    Implements the singleton pattern for application-wide job management.
    Provides a bounded queue with configurable workers for controlled
    resource utilization.

    Features:
    - Bounded queue with backpressure (configurable max size)
    - Multiple async workers for parallel processing
    - Job progress tracking
    - Cancellation support
    - Graceful shutdown

    Note: Jobs are stored in-memory only and will be lost on restart.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        max_queue_size: Optional[int] = None,
        num_workers: Optional[int] = None
    ):
        """
        Initialize the job manager.

        Args:
            max_queue_size: Maximum number of jobs in queue (default: from settings)
            num_workers: Number of concurrent worker tasks (default: from settings)
        """
        if self._initialized:
            return
        self._initialized = True

        # Use settings values if not explicitly provided
        self._max_queue_size = max_queue_size or settings.job_queue_max_size
        self._num_workers = num_workers or settings.job_queue_num_workers
        self._queue: Optional[asyncio.Queue] = None
        self._jobs: Dict[str, Job] = {}
        self._workers: List[asyncio.Task] = []
        self._shutdown = False
        self._process_func: Optional[Callable[[Job], Awaitable[Dict[str, Any]]]] = None
        self._jobs_lock = asyncio.Lock()

        logger.info(
            f"Entering __init__ - max_queue_size={max_queue_size}, "
            f"num_workers={num_workers}"
        )
        logger.info("Exiting __init__ - JobManager singleton created")

    def _ensure_queue(self) -> asyncio.Queue:
        """Ensure queue is initialized in current event loop."""
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self._max_queue_size)
        return self._queue

    async def submit_job(self, video_path: str, config: Dict[str, Any]) -> str:
        """
        Submit a new video processing job to the queue.

        Creates a job with PENDING status, then attempts to add it to
        the processing queue. Returns immediately with the job_id.

        Args:
            video_path: Path to video file for processing
            config: Job configuration parameters

        Returns:
            str: Unique job identifier for tracking

        Raises:
            asyncio.QueueFull: When queue is at capacity (backpressure)
        """
        logger.info(
            f"Entering submit_job - video_path={video_path}, "
            f"config_keys={list(config.keys())}"
        )

        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Create job instance
        job = Job(
            id=job_id,
            video_path=video_path,
            config=config,
            status=JobStatus.PENDING,
            progress=0,
            created_at=datetime.utcnow()
        )

        # Store job in memory
        async with self._jobs_lock:
            self._jobs[job_id] = job

        # Attempt to add to queue (non-blocking)
        queue = self._ensure_queue()
        try:
            queue.put_nowait(job_id)
            job.status = JobStatus.QUEUED
            logger.info(
                f"Exiting submit_job - job_id={job_id}, status=QUEUED, "
                f"queue_size={queue.qsize()}"
            )
        except asyncio.QueueFull:
            logger.warning(
                f"Exiting submit_job - job_id={job_id}, queue_full=True"
            )
            raise

        return job_id

    async def get_job(self, job_id: str) -> Optional[Job]:
        """
        Retrieve a job by its ID.

        Args:
            job_id: Unique job identifier

        Returns:
            Job instance if found, None otherwise
        """
        logger.info(f"Entering get_job - job_id={job_id}")

        async with self._jobs_lock:
            job = self._jobs.get(job_id)

        if job:
            logger.info(
                f"Exiting get_job - job_id={job_id}, status={job.status}, "
                f"progress={job.progress}"
            )
        else:
            logger.info(f"Exiting get_job - job_id={job_id}, found=False")

        return job

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or running job.

        Jobs can only be cancelled if they are in PENDING, QUEUED, or
        PROCESSING status. Completed, failed, or already cancelled jobs
        cannot be cancelled.

        Args:
            job_id: Unique job identifier

        Returns:
            bool: True if job was cancelled, False otherwise
        """
        logger.info(f"Entering cancel_job - job_id={job_id}")

        async with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                logger.info(
                    f"Exiting cancel_job - job_id={job_id}, result=False, "
                    f"reason=job_not_found"
                )
                return False

            # Check if job can be cancelled
            if job.status in (
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED
            ):
                logger.info(
                    f"Exiting cancel_job - job_id={job_id}, result=False, "
                    f"reason=job_already_{job.status.value}"
                )
                return False

            # Mark job as cancelled
            previous_status = job.status
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            job.error = "Job cancelled by user"

        logger.info(
            f"Exiting cancel_job - job_id={job_id}, result=True, "
            f"previous_status={previous_status.value}"
        )
        return True

    async def update_job_progress(self, job_id: str, progress: int) -> bool:
        """
        Update job progress percentage.

        Args:
            job_id: Unique job identifier
            progress: Progress percentage (0-100)

        Returns:
            bool: True if job was updated, False if job not found
        """
        async with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            job.progress = max(0, min(100, progress))
            return True

    async def start_workers(
        self,
        process_func: Callable[[Job], Awaitable[Dict[str, Any]]]
    ) -> None:
        """
        Start background worker tasks.

        Workers pull jobs from the queue and process them using the
        provided function. Multiple workers enable parallel processing.

        Args:
            process_func: Async function that processes a job and returns
                         result dict. Should accept Job and return Dict.
        """
        logger.info(f"Entering start_workers - num_workers={self._num_workers}")

        self._process_func = process_func
        self._shutdown = False

        # Ensure queue exists
        self._ensure_queue()

        # Create worker tasks
        for i in range(self._num_workers):
            worker_task = asyncio.create_task(
                self._worker(i),
                name=f"job_worker_{i}"
            )
            self._workers.append(worker_task)
            logger.info(f"Started worker {i}")

        logger.info(
            f"Exiting start_workers - workers_started={len(self._workers)}"
        )

    async def stop_workers(self) -> None:
        """
        Gracefully stop all workers.

        Sets shutdown flag and waits for workers to finish current jobs.
        """
        logger.info("Entering stop_workers")

        self._shutdown = True

        # Cancel all worker tasks
        for worker in self._workers:
            if not worker.done():
                worker.cancel()

        # Wait for workers to finish (with timeout)
        if self._workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.warning("Worker shutdown timed out after 30 seconds")

        self._workers.clear()
        logger.info("Exiting stop_workers - all workers stopped")

    async def _worker(self, worker_id: int) -> None:
        """
        Worker coroutine that processes jobs from the queue.

        Runs continuously until shutdown, pulling jobs from queue and
        processing them one at a time. Periodically cleans up completed
        jobs to prevent unbounded memory growth.

        Args:
            worker_id: Worker identifier for logging
        """
        logger.info(f"Worker {worker_id} started")
        _cleanup_counter = 0

        while not self._shutdown:
            try:
                # Periodic cleanup of completed jobs (every ~50s)
                _cleanup_counter += 1
                if _cleanup_counter >= 50:
                    _cleanup_counter = 0
                    removed = await self.cleanup_completed_jobs(max_age_seconds=3600)
                    if removed > 0:
                        logger.info(f"Worker {worker_id}: Auto-cleaned {removed} old jobs")

                # Wait for job with timeout (allows checking shutdown flag)
                queue = self._ensure_queue()
                try:
                    job_id = await asyncio.wait_for(
                        queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Get job details
                async with self._jobs_lock:
                    job = self._jobs.get(job_id)

                if not job:
                    logger.warning(
                        f"Worker {worker_id}: Job {job_id} not found in memory"
                    )
                    continue

                # Skip cancelled jobs
                if job.status == JobStatus.CANCELLED:
                    logger.info(
                        f"Worker {worker_id}: Skipping cancelled job {job_id}"
                    )
                    continue

                # Process the job
                await self._process_job(worker_id, job)

            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(
                    f"Worker {worker_id}: Unexpected error - {e}",
                    exc_info=True
                )

        logger.info(f"Worker {worker_id} stopped")

    async def _process_job(self, worker_id: int, job: Job) -> None:
        """
        Process a single job.

        Updates job status, calls process function, and handles results
        or errors.

        Args:
            worker_id: Worker identifier for logging
            job: Job instance to process
        """
        logger.info(
            f"Worker {worker_id}: Processing job {job.id} - "
            f"video={job.video_path}"
        )

        # Update job status to processing
        async with self._jobs_lock:
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.utcnow()

        try:
            # Check for cancellation before processing
            if job.status == JobStatus.CANCELLED:
                logger.info(
                    f"Worker {worker_id}: Job {job.id} cancelled before processing"
                )
                return

            # Call the processing function
            if self._process_func:
                result = await self._process_func(job)

                # Check for cancellation after processing
                async with self._jobs_lock:
                    if job.status == JobStatus.CANCELLED:
                        logger.info(
                            f"Worker {worker_id}: Job {job.id} cancelled during processing"
                        )
                        return

                    job.status = JobStatus.COMPLETED
                    job.progress = 100
                    job.result = result
                    job.completed_at = datetime.utcnow()

                logger.info(
                    f"Worker {worker_id}: Job {job.id} completed successfully"
                )
            else:
                raise RuntimeError("No process function configured")

        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"Worker {worker_id}: Job {job.id} failed - {error_msg}",
                exc_info=True
            )

            async with self._jobs_lock:
                job.status = JobStatus.FAILED
                job.error = error_msg
                job.completed_at = datetime.utcnow()

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue and job statistics.

        Returns:
            Dict with queue depth, active/pending/completed job counts
        """
        logger.info("Entering get_queue_status")

        # Count jobs by status
        status_counts = {status: 0 for status in JobStatus}
        for job in self._jobs.values():
            status_counts[job.status] += 1

        queue = self._ensure_queue()
        queue_depth = queue.qsize()

        result = {
            "queue_depth": queue_depth,
            "active_jobs": status_counts[JobStatus.PROCESSING],
            "pending_jobs": status_counts[JobStatus.PENDING] + status_counts[JobStatus.QUEUED],
            "completed_jobs": status_counts[JobStatus.COMPLETED],
            "failed_jobs": status_counts[JobStatus.FAILED],
            "cancelled_jobs": status_counts[JobStatus.CANCELLED],
            "total_jobs": len(self._jobs),
            "max_queue_size": self._max_queue_size,
            "num_workers": self._num_workers,
            "queue_full": queue_depth >= self._max_queue_size
        }

        logger.info(
            f"Exiting get_queue_status - queue_depth={queue_depth}, "
            f"active={result['active_jobs']}, pending={result['pending_jobs']}"
        )

        return result

    @property
    def queue_size(self) -> int:
        """Get current queue depth."""
        queue = self._ensure_queue()
        return queue.qsize()

    @property
    def is_queue_full(self) -> bool:
        """Check if queue is at capacity."""
        queue = self._ensure_queue()
        return queue.full()

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown

    async def cleanup_completed_jobs(self, max_age_seconds: int = 3600) -> int:
        """
        Remove old completed/failed/cancelled jobs from memory.

        Args:
            max_age_seconds: Maximum age of completed jobs to keep (default: 1 hour)

        Returns:
            int: Number of jobs removed
        """
        logger.info(f"Entering cleanup_completed_jobs - max_age_seconds={max_age_seconds}")

        now = datetime.utcnow()
        removed_count = 0

        async with self._jobs_lock:
            jobs_to_remove = []

            for job_id, job in self._jobs.items():
                if job.status in (
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED
                ):
                    if job.completed_at:
                        age = (now - job.completed_at).total_seconds()
                        if age > max_age_seconds:
                            jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                del self._jobs[job_id]
                removed_count += 1

        logger.info(f"Exiting cleanup_completed_jobs - removed={removed_count}")
        return removed_count


# Module-level singleton instance
job_manager = JobManager()

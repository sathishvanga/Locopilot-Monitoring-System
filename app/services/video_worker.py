"""
Video Worker Service - GPU-accelerated video processing worker for concurrent jobs

Implements worker functions for processing video jobs from the queue using
shared GPU models. Provides:
- GPU slot acquisition via semaphore
- CUDA stream assignment for parallel execution
- OOM (Out of Memory) recovery with batch size reduction
- Progress tracking callbacks
- Job cancellation support

Usage:
    from app.services.video_worker import process_video_job, create_worker_loop

    # Process a single job
    result = await process_video_job(job, gpu_manager, progress_callback)

    # Create worker loop for job manager
    worker = create_worker_loop(job_manager, gpu_manager, worker_id=0)
    await worker()
"""

import asyncio
import os
import time
from typing import Callable, Dict, Any, Optional, List, Awaitable

from ..models.job_models import Job, JobStatus
from ..utils.logger import get_logger
from ..utils.config import get_settings


logger = get_logger(__name__)
settings = get_settings()


# Type alias for progress update callback
ProgressCallback = Callable[[float], Awaitable[None]]


async def process_video_job(
    job: Job,
    gpu_manager: "GPUResourceManager",
    update_progress: Optional[ProgressCallback] = None,
    worker_id: int = 0
) -> Dict[str, Any]:
    """
    Process a single video job using GPU resources.

    Acquires a GPU slot, assigns a CUDA stream, and runs video processing
    with the LocopilotActivityMonitor using preloaded shared models.

    Args:
        job: The Job instance to process
        gpu_manager: GPUResourceManager instance for model access and GPU slots
        update_progress: Optional async callback for progress updates (0-100)
        worker_id: Worker identifier for stream assignment

    Returns:
        Dict[str, Any]: Processing result containing:
            - status: "success" or "error"
            - activities: List of detected activities
            - activities_count: Number of activities
            - processing_time: Time in seconds
            - run_directory: Output directory path

    Raises:
        torch.cuda.OutOfMemoryError: If GPU runs out of memory (triggers recovery)
        Exception: Any other processing errors
    """
    logger.info(
        "Entering process_video_job",
        extra={
            "job_id": job.id,
            "video_path": job.video_path,
            "worker_id": worker_id
        }
    )

    start_time = time.time()
    stream = None
    stream_idx = None  # Initialize before try block to avoid NameError in finally
    activities = []

    async with gpu_manager.acquire_gpu_slot():
        try:
            # Assign CUDA stream for this worker
            stream_idx = gpu_manager.assign_stream(worker_id)
            stream = gpu_manager.get_stream(worker_id)

            logger.info(
                f"GPU slot acquired for job {job.id}",
                extra={"stream_idx": stream_idx, "worker_id": worker_id}
            )

            # Get preloaded models
            models = gpu_manager.get_models()
            if not models:
                raise RuntimeError("Models not loaded - call gpu_manager.load_models() first")

            # Update progress: starting
            if update_progress:
                await update_progress(5.0)

            # Import LocopilotActivityMonitor (avoid circular import)
            # This import is done here to prevent module-level circular dependencies
            from locopilot_monitor import LocopilotActivityMonitor

            # Extract configuration from job
            config = job.config or {}
            trip_id = config.get("trip_id", "UNKNOWN")
            crew_members = config.get("crew_members")
            crew_name = config.get("crew_name", "Unknown Crew")
            crew_id = config.get("crew_id", "C-000")
            crew_role = config.get("crew_role", 1)
            save_clips = config.get("save_clips", True)
            run_dir = config.get("run_dir")

            logger.info(
                f"Processing video for trip {trip_id}",
                extra={
                    "video_path": job.video_path,
                    "save_clips": save_clips,
                    "has_crew_members": crew_members is not None
                }
            )

            # Update progress: initializing monitor
            if update_progress:
                await update_progress(10.0)

            # Create monitor with preloaded models
            # Pass create_run_dir=True if run_dir not provided
            monitor = LocopilotActivityMonitor(
                video_path=job.video_path,
                output_dir=settings.output_dir,
                save_annotated_frames=settings.save_annotated_frames,
                frame_save_interval=settings.frame_save_interval,
                sample_fps=settings.sample_fps,
                run_dir=run_dir,
                create_run_dir=(run_dir is None),
                preloaded_models=models
            )

            # Update progress: starting video processing
            if update_progress:
                await update_progress(15.0)

            # Process video in thread pool (CPU-bound operation)
            # This allows the event loop to remain responsive
            activities = await asyncio.to_thread(
                _process_video_sync,
                monitor,
                update_progress,
                job
            )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Build result
            result = {
                "status": "success",
                "message": "Video processed successfully",
                "trip_id": trip_id,
                "video_path": job.video_path,
                "activities": activities,
                "activities_count": len(activities),
                "processing_time": processing_time,
                "run_directory": monitor.run_dir,
                "worker_id": worker_id
            }

            logger.info(
                "Exiting process_video_job",
                extra={
                    "job_id": job.id,
                    "status": "success",
                    "activities_count": len(activities),
                    "processing_time": round(processing_time, 2)
                }
            )

            return result

        except Exception as e:
            # Check for CUDA OOM error
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str:
                logger.error(
                    f"GPU OOM error for job {job.id}",
                    extra={"error": str(e), "worker_id": worker_id}
                )
                await handle_oom_recovery(job, gpu_manager)
                raise

            logger.error(
                f"Error processing job {job.id}: {e}",
                exc_info=True,
                extra={"worker_id": worker_id}
            )
            raise

        finally:
            # Release stream assignment
            if stream_idx is not None:
                gpu_manager.release_stream(worker_id)


def _process_video_sync(
    monitor: "LocopilotActivityMonitor",
    update_progress: Optional[ProgressCallback],
    job: Job
) -> List[Dict[str, Any]]:
    """
    Synchronous video processing wrapper for asyncio.to_thread.

    This function runs in a thread pool executor and handles the
    CPU-bound video processing work.

    Args:
        monitor: LocopilotActivityMonitor instance with preloaded models
        update_progress: Progress callback (will be called from sync context)
        job: Job being processed (for cancellation checks)

    Returns:
        List[Dict[str, Any]]: List of detected activities
    """
    logger.info(f"Starting sync video processing for job {job.id}")

    try:
        # Process video (this is the main CPU-bound work)
        activities = monitor.process_video()

        logger.info(
            f"Sync processing completed for job {job.id}",
            extra={"activities_count": len(activities)}
        )

        return activities

    except Exception as e:
        logger.error(
            f"Sync processing error for job {job.id}: {e}",
            exc_info=True
        )
        raise


async def handle_oom_recovery(
    job: Job,
    gpu_manager: "GPUResourceManager"
) -> None:
    """
    Handle OOM (Out of Memory) error recovery.

    Clears CUDA cache, reduces batch size, and prepares for retry.
    Called when torch.cuda.OutOfMemoryError is caught during processing.

    Args:
        job: The job that encountered OOM
        gpu_manager: GPUResourceManager for memory management
    """
    logger.warning(
        "Entering handle_oom_recovery",
        extra={"job_id": job.id}
    )

    try:
        import torch

        # Check if CUDA is available
        if torch.cuda.is_available():
            # Clear CUDA memory cache
            torch.cuda.empty_cache()

            # Synchronize to ensure all operations complete
            torch.cuda.synchronize()

            # Log memory state after clearing
            allocated_mb = torch.cuda.memory_allocated() / 1024**2
            cached_mb = torch.cuda.memory_reserved() / 1024**2

            logger.info(
                "CUDA cache cleared",
                extra={
                    "allocated_mb": round(allocated_mb, 2),
                    "cached_mb": round(cached_mb, 2)
                }
            )

        # Reduce batch size via GPU manager
        new_batch_size = gpu_manager.reduce_batch_size()

        logger.warning(
            "Exiting handle_oom_recovery",
            extra={
                "job_id": job.id,
                "new_batch_size": new_batch_size,
                "action": "batch_size_reduced"
            }
        )

    except Exception as e:
        logger.error(
            f"Error during OOM recovery for job {job.id}: {e}",
            exc_info=True
        )


def create_worker_loop(
    job_manager: "JobManager",
    gpu_manager: "GPUResourceManager",
    worker_id: int
) -> Callable[[], Awaitable[None]]:
    """
    Create a worker coroutine that processes jobs from the queue.

    Returns an async function that continuously pulls jobs from the
    job manager's queue and processes them until shutdown is requested.

    Args:
        job_manager: JobManager instance for queue access
        gpu_manager: GPUResourceManager for GPU resources
        worker_id: Unique worker identifier (0-indexed)

    Returns:
        Callable: Async worker function that processes jobs in a loop
    """

    async def worker_loop() -> None:
        """
        Worker loop that processes jobs from the queue.

        Runs continuously until job_manager._shutdown is True.
        Handles job processing, progress updates, and error recovery.
        """
        logger.info(f"Worker {worker_id} starting")

        while not job_manager._shutdown:
            try:
                # Wait for a job with timeout (allows checking shutdown flag)
                try:
                    job_id = await asyncio.wait_for(
                        job_manager._queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # No job available, check shutdown flag and continue
                    continue

                # Get job details
                job = await job_manager.get_job(job_id)
                if not job:
                    logger.warning(
                        f"Worker {worker_id}: Job {job_id} not found"
                    )
                    continue

                # Skip cancelled jobs
                if job.status == JobStatus.CANCELLED:
                    logger.info(
                        f"Worker {worker_id}: Skipping cancelled job {job_id}"
                    )
                    continue

                # Process the job
                logger.info(
                    f"Worker {worker_id}: Processing job {job_id}",
                    extra={"video_path": job.video_path}
                )

                # Define progress update callback
                async def update_progress(progress: float) -> None:
                    await job_manager.update_job_progress(job_id, int(progress))

                try:
                    # Process the video job
                    result = await process_video_job(
                        job=job,
                        gpu_manager=gpu_manager,
                        update_progress=update_progress,
                        worker_id=worker_id
                    )

                    # Mark job as completed
                    async with job_manager._jobs_lock:
                        if job.status != JobStatus.CANCELLED:
                            job.status = JobStatus.COMPLETED
                            job.progress = 100
                            job.result = result

                    logger.info(
                        f"Worker {worker_id}: Job {job_id} completed",
                        extra={"activities_count": result.get("activities_count", 0)}
                    )

                except Exception as e:
                    # Mark job as failed
                    error_msg = str(e)
                    logger.error(
                        f"Worker {worker_id}: Job {job_id} failed - {error_msg}",
                        exc_info=True
                    )

                    async with job_manager._jobs_lock:
                        job.status = JobStatus.FAILED
                        job.error = error_msg

            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id}: Cancelled")
                break

            except Exception as e:
                logger.error(
                    f"Worker {worker_id}: Unexpected error - {e}",
                    exc_info=True
                )
                # Continue processing other jobs

        logger.info(f"Worker {worker_id} stopped")

    return worker_loop


async def create_job_processor(
    gpu_manager: "GPUResourceManager"
) -> Callable[[Job], Awaitable[Dict[str, Any]]]:
    """
    Create a job processor function for use with JobManager.

    Returns an async function compatible with JobManager.start_workers()
    that processes jobs using the GPU manager.

    Args:
        gpu_manager: GPUResourceManager instance

    Returns:
        Callable: Async function that processes a Job and returns result dict
    """
    worker_counter = {"count": 0}

    async def process_job(job: Job) -> Dict[str, Any]:
        """Process a job using GPU resources."""
        # Assign worker ID (round-robin)
        worker_id = worker_counter["count"] % gpu_manager.max_concurrent
        worker_counter["count"] += 1

        # Define progress callback
        async def update_progress(progress: float) -> None:
            job.progress = int(progress)

        return await process_video_job(
            job=job,
            gpu_manager=gpu_manager,
            update_progress=update_progress,
            worker_id=worker_id
        )

    return process_job


class VideoWorkerPool:
    """
    Pool of video processing workers for concurrent job execution.

    Manages multiple worker coroutines that pull from the job queue
    and process videos using shared GPU resources.

    Usage:
        pool = VideoWorkerPool(job_manager, gpu_manager, num_workers=3)
        await pool.start()
        # ... submit jobs ...
        await pool.stop()
    """

    def __init__(
        self,
        job_manager: "JobManager",
        gpu_manager: "GPUResourceManager",
        num_workers: int = 3
    ):
        """
        Initialize the worker pool.

        Args:
            job_manager: JobManager for queue access
            gpu_manager: GPUResourceManager for GPU resources
            num_workers: Number of concurrent workers (default: 3)
        """
        logger.info(
            "Entering VideoWorkerPool.__init__",
            extra={"num_workers": num_workers}
        )

        self._job_manager = job_manager
        self._gpu_manager = gpu_manager
        self._num_workers = num_workers
        self._workers: List[asyncio.Task] = []
        self._running = False

        logger.info(
            "Exiting VideoWorkerPool.__init__",
            extra={"status": "initialized"}
        )

    async def start(self) -> None:
        """
        Start all worker tasks.

        Creates and starts worker coroutines that will process jobs
        from the queue until stop() is called.
        """
        logger.info(
            "Entering VideoWorkerPool.start",
            extra={"num_workers": self._num_workers}
        )

        if self._running:
            logger.warning("Worker pool already running")
            return

        self._running = True

        # Create worker tasks
        for i in range(self._num_workers):
            worker_func = create_worker_loop(
                self._job_manager,
                self._gpu_manager,
                worker_id=i
            )
            task = asyncio.create_task(
                worker_func(),
                name=f"video_worker_{i}"
            )
            self._workers.append(task)

        logger.info(
            "Exiting VideoWorkerPool.start",
            extra={"workers_started": len(self._workers)}
        )

    async def stop(self, timeout: float = 30.0) -> None:
        """
        Stop all worker tasks gracefully.

        Args:
            timeout: Maximum time to wait for workers to finish (seconds)
        """
        logger.info(
            "Entering VideoWorkerPool.stop",
            extra={"num_workers": len(self._workers)}
        )

        self._running = False

        # Signal shutdown to job manager
        self._job_manager._shutdown = True

        # Cancel all worker tasks
        for worker in self._workers:
            if not worker.done():
                worker.cancel()

        # Wait for workers to finish
        if self._workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Worker pool shutdown timed out after {timeout}s"
                )

        self._workers.clear()

        logger.info(
            "Exiting VideoWorkerPool.stop",
            extra={"status": "stopped"}
        )

    @property
    def is_running(self) -> bool:
        """Check if worker pool is running."""
        return self._running

    @property
    def worker_count(self) -> int:
        """Get number of active workers."""
        return len([w for w in self._workers if not w.done()])

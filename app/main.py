"""
Main FastAPI application - Locopilot Monitoring System

A production-ready API for video processing and activity detection.
"""

import os

# Thread configuration for CPU inference optimization
# Allow worker processes to set optimal thread counts
# Conservative default of 2, worker initializer will override based on actual worker count
default_threads = 2  # Safe default, workers will set optimal value
os.environ.setdefault('OMP_NUM_THREADS', str(default_threads))
os.environ.setdefault('MKL_NUM_THREADS', str(default_threads))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(default_threads))
os.environ.setdefault('TORCH_CPP_LOG_LEVEL', 'ERROR')

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .controllers import video_router
from .middleware import LoggingMiddleware
from .utils.logger import setup_logging, get_logger
from .utils.config import get_settings
from .utils.video_multiprocessing import shutdown_shared_pool
from .services.dlq import drain_dlq
from .services.gpu_resource_manager import get_gpu_resource_manager
from .services.job_manager import job_manager
from .services.video_processing_service import VideoProcessingService
from .models.job_models import Job


# Initialize settings and logging
settings = get_settings()
setup_logging(level=settings.log_level)
logger = get_logger("app.main")


def print_startup_banner(gpu_manager=None):
    """Print startup banner to console for user visibility"""
    # Get GPU info
    gpu_device = "CPU only"
    gpu_memory = "N/A"
    if gpu_manager and gpu_manager.is_gpu_available:
        gpu_device = gpu_manager.device_name
        gpu_memory = f"{gpu_manager.total_memory_mb:.0f} MB"

    # Truncate long paths/names for display
    output_dir_display = settings.output_dir[:40] + "..." if len(settings.output_dir) > 43 else settings.output_dir
    upload_dir_display = settings.upload_dir[:40] + "..." if len(settings.upload_dir) > 43 else settings.upload_dir
    gpu_device_display = gpu_device[:40] + "..." if len(gpu_device) > 43 else gpu_device

    banner = f"""
+==================================================================+
|           LOCOPILOT MONITORING SYSTEM                            |
+==================================================================+
|  Version:     {settings.app_version:<50} |
|  Environment: {'Development' if settings.debug else 'Production':<50} |
|  Host:        {settings.host}:{settings.port:<44} |
+------------------------------------------------------------------+
|  Output Dir:  {output_dir_display:<50} |
|  Upload Dir:  {upload_dir_display:<50} |
|  Sample FPS:  {str(settings.sample_fps):<50} |
|  YOLO Model:  {settings.yolo_weights:<50} |
+------------------------------------------------------------------+
|  GPU CONFIGURATION                                               |
+------------------------------------------------------------------+
|  GPU Device:        {gpu_device_display:<44} |
|  GPU Memory:        {gpu_memory:<44} |
|  Max Concurrent:    {str(settings.max_concurrent_videos):<44} |
|  Queue Max Size:    {str(settings.job_queue_max_size):<44} |
+------------------------------------------------------------------+
|  API Docs:    http://{settings.host}:{settings.port}/docs{' ' * (40 - len(str(settings.port)))} |
|  Health:      http://{settings.host}:{settings.port}/health{' ' * (38 - len(str(settings.port)))} |
|  GPU Status:  http://{settings.host}:{settings.port}/api/gpu/status{' ' * (31 - len(str(settings.port)))} |
+==================================================================+
"""
    logger.info(banner)


# Module-level singleton VideoProcessingService instance for the queue
# worker path. The class is itself stateless (its dependencies are
# singletons), so a single instance is fine and avoids constructing one
# per job. Task 0010 will introduce a proper ``get_video_processing_service``
# accessor; until then this inline-once construction matches the spec
# guidance in tasks/0003.
_video_processing_service_for_jobs: VideoProcessingService = VideoProcessingService()


async def process_video_job(job: Job) -> dict:
    """
    Process a video job using the VideoProcessingService.

    This function is called by job queue workers to process submitted jobs.
    It wraps the synchronous video processing in an executor to avoid
    blocking the event loop, and gates the work behind
    ``GPUResourceManager.acquire_gpu_slot()`` so the three queue workers
    (started in :func:`lifespan`) cannot run more than
    ``MAX_CONCURRENT_VIDEOS`` videos against the GPU concurrently.

    Without that gate, three workers + a synchronous /process-and-upload
    request could all hit the 20 GB RTX 4000 Ada at once and OOM
    (Pipeline-1 ~8 GB peak + vLLM ~10 GB resident only leaves ~3 GB
    headroom — see CLAUDE.md "One-worker rule").

    Args:
        job: Job instance with video_path and config

    Returns:
        dict: Processing result with activities and metadata
    """
    logger.info(f"Processing job {job.id} - video: {job.video_path}")

    # Extract config parameters
    config = job.config or {}
    trip_id = config.get("trip_id", f"job_{job.id[:8]}")
    crew_members = config.get("crew_members", {})
    lp_crew_name = config.get("lp_crew_name", "Unknown")
    lp_crew_id = config.get("lp_crew_id", "N/A")
    use_mock = config.get("use_mock_detection", False)
    use_mp = config.get("use_multiprocessing", settings.enable_multiprocessing)
    save_clips = config.get("save_clips", False)

    gpu_manager = get_gpu_resource_manager()

    # Hold a GPU slot for the duration of heavy processing. The queue
    # worker bypasses ``try_enqueue`` (the job_manager queue is the
    # admission gate for this path), so we use ``increment_active``
    # directly rather than ``mark_enqueued_started`` — the latter
    # decrements ``_pending_count`` which would corrupt the controller
    # path's admission accounting if a controller request were also in
    # flight. ``acquire_gpu_slot``'s finally still does the matching
    # decrement on exit (single counter authority for the - side).
    async with gpu_manager.acquire_gpu_slot():
        gpu_manager.increment_active()
        # Run synchronous processing in thread pool to not block event loop
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: _video_processing_service_for_jobs.process_video(
                video_path=job.video_path,
                trip_id=trip_id,
                crew_members=crew_members,
                crew_name=lp_crew_name,
                crew_id=lp_crew_id,
                crew_role=1,
                use_mock_detection=use_mock,
                use_multiprocessing=use_mp,
                save_clips=save_clips
            )
        )

    logger.info(
        f"Job {job.id} completed - activities: {result.get('activitiesCount', 0)}, "
        f"time: {result.get('processingTime', 0):.2f}s"
    )

    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager

    Handles startup and shutdown events for resource initialization
    and cleanup, including GPU manager and job queue workers.
    """
    # Initialize GPU Resource Manager (singleton)
    gpu_manager = get_gpu_resource_manager()

    # Initialize GPU device and configure memory settings
    # This sets up CUDA device, memory fraction, and semaphore for concurrency control
    gpu_manager.initialize()

    # Startup banner (now routed through the canonical logger so it goes
    # through the redaction filter and respects the file-only handler config).
    print_startup_banner(gpu_manager)
    logger.info("[OK] Application started successfully!")
    logger.info(f"[LOG] Logs are being written to: {settings.log_dir}/LocopilotMonitoring.log")
    logger.info("-" * 68)

    # Log GPU startup statistics
    gpu_manager.log_startup_stats()

    # Detailed logging to file
    logger.info("=" * 60)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {'Development' if settings.debug else 'Production'}")
    logger.info(f"Output directory: {settings.output_dir}")
    logger.info(f"Upload directory: {settings.upload_dir}")
    logger.info(f"Sample FPS: {settings.sample_fps}")
    logger.info(f"YOLO weights: {settings.yolo_weights}")
    logger.info(f"Max concurrent videos: {settings.max_concurrent_videos}")
    logger.info(f"Job queue max size: {settings.job_queue_max_size}")
    logger.info("=" * 60)

    # Store GPU manager and job manager in app state for access by endpoints
    app.state.gpu_manager = gpu_manager
    app.state.job_manager = job_manager

    # Start job queue workers
    try:
        num_workers = settings.max_concurrent_videos
        logger.info(f"Starting job queue workers (num_workers={num_workers})")
        await job_manager.start_workers(process_video_job)
        logger.info(f"[QUEUE] Job queue started with {num_workers} workers")
        logger.info(f"Job queue workers started successfully")
    except Exception as e:
        logger.error(f"Failed to start job queue workers: {e}", exc_info=True)
        logger.warning(f"[WARNING] Job queue workers failed to start: {e}")

    # Schedule a one-shot DLQ re-drain in the background so any external-API
    # failures from the previous run get replayed on this boot. The task is
    # fire-and-forget — DLQ writes survive restarts so a crash here is
    # recoverable on the next boot.
    async def _drain_dlq_on_startup() -> None:
        try:
            stats = await asyncio.get_running_loop().run_in_executor(None, drain_dlq)
            if stats and stats.get("drained"):
                logger.info(
                    f"[dlq] Startup drain finished drained={stats['drained']} "
                    f"succeeded={stats['succeeded']} failed={stats['failed']}"
                )
        except Exception as e:
            logger.warning(f"[dlq] Startup drain task failed: {e}", exc_info=True)

    try:
        app.state.dlq_drain_task = asyncio.create_task(_drain_dlq_on_startup())
        logger.info("[dlq] Startup re-drain task scheduled")
    except Exception as e:
        logger.warning(f"[dlq] Failed to schedule startup re-drain task: {e}")

    yield

    # Shutdown
    logger.info("[STOP] Shutting down application...")

    # Cancel the DLQ drain task if it's still in flight on shutdown.
    drain_task = getattr(app.state, "dlq_drain_task", None)
    if drain_task is not None and not drain_task.done():
        drain_task.cancel()
        try:
            await drain_task
        except (asyncio.CancelledError, Exception):
            pass

    # Stop job queue workers
    try:
        logger.info("Stopping job queue workers...")
        await job_manager.stop_workers()
        logger.info("[OK] Job queue workers stopped")
    except Exception as e:
        logger.warning(
            f"Error while stopping job queue workers: {e}",
            exc_info=True,
        )

    # Shutdown GPU Resource Manager (clears memory, releases models)
    try:
        gpu_manager.shutdown()
        logger.info("[OK] GPU Resource Manager shut down")
    except Exception as e:
        logger.warning(f"Error shutting down GPU manager: {e}", exc_info=True)

    # Ensure shared multiprocessing pool is shut down cleanly
    try:
        shutdown_shared_pool(wait=True)
        logger.info("Shared multiprocessing pool shut down")
    except Exception as e:
        logger.warning(
            f"Error while shutting down shared multiprocessing pool: {e}",
            exc_info=True,
        )

    logger.info("[OK] Shutdown complete. Goodbye!")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="""
    ## Locopilot Monitoring System API
    
    A production-ready FastAPI application for video processing and activity detection.
    
    ### Features
    - **Video Upload**: Upload videos for processing
    - **Activity Detection**: Detect various activities (cell phone usage, sleep, writing, etc.)
    - **Structured Output**: Generate activities.json with detailed metadata
    - **Scalable Architecture**: Clean MVC pattern with separate layers
    
    ### Architecture
    - **Models**: Pydantic schemas for validation
    - **Controllers**: API route handlers
    - **Services**: Business logic and ML processing
    - **Repositories**: Data persistence (JSON files)
    - **Utils**: Configuration and logging
    
    ### Usage
    1. Upload a video with tripId via `/api/v1/video/process`
    2. Receive processing results with detected activities
    3. Access activities.json file in the output directory
    """,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Configure CORS - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware for request/response tracking
app.add_middleware(LoggingMiddleware)


# Exception handlers
#
# Security (M-22): All exception handlers guard against leaking internal details
# (file paths, database info, stack traces) in production responses.
# Full error details are always logged internally for debugging.
# In debug mode, detailed errors are returned for development convenience.


def _sanitize_validation_errors(errors: list) -> list:
    """
    Sanitize validation errors for production responses.

    Strips internal context (e.g. raw input values, internal URLs) from
    validation error details, keeping only field location, message, and type.
    """
    sanitized = []
    for err in errors:
        sanitized.append({
            "loc": err.get("loc", []),
            "msg": err.get("msg", "Invalid value"),
            "type": err.get("type", "value_error"),
        })
    return sanitized


def _scrub_validation_errors_for_log(errors: list) -> list:
    """
    Strip the ``input`` key (and ``ctx.input``) from validation errors before
    they are written to the log. Pydantic includes the raw rejected value in
    every error dict by default — for malformed Authorization-style fields
    that means a bearer token would otherwise land in the log file (task 0010).
    """
    scrubbed = []
    for err in errors:
        copy = {k: v for k, v in err.items() if k != "input"}
        ctx = copy.get("ctx")
        if isinstance(ctx, dict) and "input" in ctx:
            copy["ctx"] = {k: v for k, v in ctx.items() if k != "input"}
        scrubbed.append(copy)
    return scrubbed


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with production-safe error messages (M-22, task 0010).

    For 5xx responses we return a single opaque body — ``{"detail":
    "internal_error"}`` — so we never echo a controller-side ``str(e)``
    fragment back to the caller. The full exception is already on the
    server side via ``logger.exception`` at the original raise site; the
    response is intentionally minimal.
    """
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")

    # For server errors (5xx), return a fixed opaque body so we never leak
    # an internal exception message via ``str(e)``-style detail. Debug mode
    # is intentionally NOT given a free pass here — the leak would still
    # occur if DEBUG were ever set in production by mistake.
    if exc.status_code >= 500:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": "internal_error"}
        )

    # For client errors (4xx) and debug mode, return the detail as-is
    # since these are intentional user-facing messages.
    if settings.debug:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "message": exc.detail,
                "error": str(exc.detail),
            }
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with sanitized output (M-22, task 0010).

    The default ``exc.errors()`` payload includes the raw rejected ``input``
    value for every failing field; for an Authorization-shaped header that
    embeds the bearer in the log line. We always strip ``input`` from the
    logged version, regardless of debug mode.
    """
    logger.warning(f"Validation error: {_scrub_validation_errors_for_log(exc.errors())}")

    # In production, sanitize validation errors to remove internal context
    # (e.g. raw input values that may contain sensitive data).
    if settings.debug:
        errors_detail = exc.errors()
    else:
        errors_detail = _sanitize_validation_errors(exc.errors())

    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Validation error",
            "errors": errors_detail,
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions without leaking internals (M-22, task 0010).

    The full exception is logged with traceback. The response body is the
    same opaque ``{"detail": "internal_error"}`` shape used by the 5xx
    HTTP exception handler so all server errors look the same to the caller.
    """
    error_msg = str(exc) if exc else "Unknown error"
    logger.error(f"Unexpected error: {error_msg}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal_error"}
    )


# Register routers
app.include_router(video_router)


# Root endpoint
@app.get(
    "/",
    tags=["root"],
    summary="API Root",
    description="Welcome endpoint with API information"
)
async def root():
    """
    API root endpoint
    
    Returns basic information about the API.
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/v1/video/health"
    }


# Health check endpoint with GPU and queue monitoring
@app.get(
    "/health",
    tags=["health"],
    summary="Application Health Check",
    description="Check overall application health including GPU and queue status"
)
async def health(request: Request):
    """
    Enhanced application health check

    Returns comprehensive health status including:
    - Application info (name, version)
    - GPU status (availability, health, active videos)
    - Job queue status (depth, workers)

    Returns:
        dict: Health status with GPU and queue information
    """
    gpu_manager = getattr(request.app.state, 'gpu_manager', None)
    job_mgr = getattr(request.app.state, 'job_manager', None)

    health_status = {
        "status": "healthy",
        "application": settings.app_name,
        "version": settings.app_version,
    }

    # Add GPU health information
    if gpu_manager:
        is_healthy, message = gpu_manager.check_memory_health()
        health_status["gpu"] = {
            "available": gpu_manager.is_gpu_available,
            "healthy": is_healthy,
            "message": message,
            "active_videos": gpu_manager.active_count,
            "device": gpu_manager.device_name,
        }

        # Set overall status based on GPU health
        if not is_healthy:
            health_status["status"] = "degraded"

    # Add queue information
    if job_mgr:
        queue_status = job_mgr.get_queue_status()
        health_status["queue"] = {
            "depth": queue_status.get("queue_depth", 0),
            "workers_running": queue_status.get("num_workers", 0),
            "active_jobs": queue_status.get("active_jobs", 0),
            "pending_jobs": queue_status.get("pending_jobs", 0),
            "queue_full": queue_status.get("queue_full", False),
        }

        # Mark as degraded if queue is full
        if queue_status.get("queue_full", False):
            health_status["status"] = "degraded"

    return health_status


# GPU Status endpoint for detailed monitoring
@app.get(
    "/api/gpu/status",
    tags=["monitoring"],
    summary="GPU Status",
    description="Get detailed GPU memory usage and processing statistics"
)
async def get_gpu_status(request: Request):
    """
    Get detailed GPU status and memory statistics.

    Returns comprehensive GPU information including:
    - Memory allocation (allocated, cached, free)
    - Memory utilization percentage
    - Device information
    - Active video processing count
    - Job queue depth
    - Configuration limits

    Non-blocking operation suitable for frequent polling.

    Returns:
        dict: GPU and queue status information
    """
    gpu_manager = getattr(request.app.state, 'gpu_manager', None)
    job_mgr = getattr(request.app.state, 'job_manager', None)

    if not gpu_manager:
        return {
            "gpu_available": False,
            "message": "GPU manager not initialized",
            "active_videos": 0,
            "queue_depth": 0,
            "max_concurrent_videos": settings.max_concurrent_videos,
        }

    # Get memory statistics (non-blocking)
    memory_stats = gpu_manager.get_memory_stats()

    # Get queue depth
    queue_depth = 0
    if job_mgr:
        queue_depth = job_mgr.queue_size

    # Build response
    response = {
        **memory_stats,
        "active_videos": gpu_manager.active_count,
        "queue_depth": queue_depth,
        "max_concurrent_videos": settings.max_concurrent_videos,
        "job_queue_max_size": settings.job_queue_max_size,
    }

    # Add health status
    is_healthy, health_message = gpu_manager.check_memory_health()
    response["health"] = {
        "is_healthy": is_healthy,
        "message": health_message,
        "warning_threshold_percent": settings.gpu_memory_warning_threshold,
    }

    return response


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )


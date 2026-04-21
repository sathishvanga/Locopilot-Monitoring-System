"""
Gunicorn configuration for production deployment

This configuration uses multiprocessing for high-performance video processing.
Logging is configured to use file-only output for clean terminal operation.
"""

import multiprocessing
import os
import logging
from logging.handlers import TimedRotatingFileHandler


# Setup gunicorn logger (file-only output)
def _setup_gunicorn_logger():
    """
    Setup a file-only logger for gunicorn lifecycle events.
    Console logging is disabled for clean terminal output.
    """
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("gunicorn.config")
    logger.setLevel(logging.INFO)
    
    # Only add handler if not already present
    if not logger.handlers:
        file_handler = TimedRotatingFileHandler(
            filename=os.path.join(log_dir, "LocopilotMonitoring.log"),
            when="midnight",
            interval=1,
            backupCount=4,
            encoding="utf-8",
            utc=True,
        )
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s,%(msecs)03d [N/A] [N/A] [N/A] [N/A] [%(levelname)s] [%(name)s] [N/A N/A] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Initialize logger
logger = _setup_gunicorn_logger()


# Worker configuration
# ---------------------------------------------------------------------------
# C-1 (task 2.1): pin to a SINGLE gunicorn worker.
#
# GPUResourceManager is a per-process singleton — admission counters
# (``active_count``, ``max_concurrent_videos``) live in module-level state
# that is NOT shared across forked workers. Running N gunicorn workers
# therefore multiplies the effective concurrency cap by N: with
# ``MAX_CONCURRENT_VIDEOS=3`` and ``workers=3`` we would admit up to 9
# simultaneous video jobs on a single 20 GB RTX 4000 Ada GPU (~2-3 GB per
# active slot), reliably triggering CUDA OOM under real load.
#
# Health, status, and admin endpoints are async and handle concurrency via
# the event loop, not additional workers — one uvicorn worker is sufficient
# for the request rates this service sees. For heavier bursts, scale by
# increasing ``max_concurrent_videos`` (up to the GPU memory budget), not by
# adding gunicorn workers.
#
# The ``GUNICORN_WORKERS`` env var is retained as an emergency override in
# case we need to fall back to multi-worker mode (e.g. to isolate a memory
# leak). In production it should remain unset, leaving ``workers = 1``.
#
# See ``tasks/code-review-critical-fixes.md`` (C-1 / 2.1) for the full
# rationale and rollback procedure.
cpu_count = multiprocessing.cpu_count()
gunicorn_workers = int(os.getenv("GUNICORN_WORKERS", "1"))
workers = gunicorn_workers if gunicorn_workers > 0 else 1

threads = 1
worker_class = "uvicorn.workers.UvicornWorker"
logger.info(f"[gunicorn_config.py] CPU count: {cpu_count}")
logger.info(f"[gunicorn_config.py] Workers: {workers} (single-GPU admission model — see C-1)")

# Application preloading
preload_app = True

# Timeouts
# ✅ PRODUCTION: Increased timeout for long videos (15 minutes)
timeout = int(os.getenv("GUNICORN_TIMEOUT", "900"))  # 15 minutes for long video processing
graceful_timeout = 60
keepalive = 5

# Request limits
# ✅ MEMORY FIX: Reduced from 2000 to 100 to force worker restarts and prevent memory accumulation
max_requests = 100
max_requests_jitter = 10

# Binding
bind = "0.0.0.0:8000"

# Logging - direct to file only
accesslog = os.path.join(os.getenv("LOG_DIR", "logs"), "gunicorn_access.log")
errorlog = os.path.join(os.getenv("LOG_DIR", "logs"), "gunicorn_error.log")
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "locopilot-monitor"

# Environment variables
raw_env = [
    f"YOLO_WEIGHTS_PRELOAD={os.getenv('YOLO_WEIGHTS_PRELOAD', 'yolo26n.pt')}",
    f"YOLO_DEVICE={os.getenv('YOLO_DEVICE', 'cpu')}",  # Pass GPU device to workers
]

# Worker lifecycle hooks
def on_starting(server):
    """
    Called just before the master process is initialized
    """
    logger.info("=" * 60)
    logger.info("Starting Locopilot Monitoring System")
    logger.info(f"Workers: {workers}")
    logger.info(f"Threads per worker: {threads}")
    logger.info(f"Timeout: {timeout}s")
    logger.info(f"Bind: {bind}")
    logger.info("=" * 60)


def on_reload(server):
    """
    Called when the server is reloaded
    """
    logger.info("Server reloading...")


def worker_int(worker):
    """
    Called when a worker receives the SIGINT or SIGQUIT signal
    """
    logger.warning(f"Worker {worker.pid} received interrupt signal")


def worker_abort(worker):
    """
    Called when a worker receives the SIGABRT signal
    """
    logger.warning(f"Worker {worker.pid} aborted")


def pre_fork(server, worker):
    """
    Called just before a worker is forked
    """
    pass


def post_fork(server, worker):
    """
    Called after a worker has been forked
    """
    logger.info(f"Worker spawned (pid: {worker.pid})")


def pre_exec(server):
    """
    Called before a new master process is forked
    """
    logger.info("Forking new master process")


def when_ready(server):
    """
    Called just after the server is started
    """
    logger.info("=" * 60)
    logger.info("Locopilot Monitoring System ready to accept requests")
    logger.info(f"Listening on {bind}")
    logger.info("=" * 60)


def worker_exit(server, worker):
    """
    Called just after a worker has been exited
    """
    logger.info(f"Worker {worker.pid} exited")


def nworkers_changed(server, new_value, old_value):
    """
    Called when the number of workers changes
    """
    logger.info(f"Number of workers changed from {old_value} to {new_value}")

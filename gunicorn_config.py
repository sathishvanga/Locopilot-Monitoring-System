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
# ✅ PRODUCTION OPTIMIZED: Environment-aware worker configuration
# Development (11 cores): 2 workers for stability
# Production (12 cores): 3 workers for higher throughput
cpu_count = multiprocessing.cpu_count()
gunicorn_workers = int(os.getenv("GUNICORN_WORKERS", "2"))

# Auto-detect optimal workers for production
if cpu_count >= 12:
    # Production server (12+ cores): 3 workers
    workers = gunicorn_workers if gunicorn_workers > 0 else 3
else:
    # Development machine (<12 cores): 2 workers
    workers = gunicorn_workers if gunicorn_workers > 0 else max(1, min(2, cpu_count // 4))

threads = 1
worker_class = "uvicorn.workers.UvicornWorker"
logger.info(f"[gunicorn_config.py] CPU count: {cpu_count}")
logger.info(f"[gunicorn_config.py] Workers: {workers} (optimized for {'production' if cpu_count >= 12 else 'development'})")

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
    "YOLO_WEIGHTS_PRELOAD=yolo11m.pt",  # YOLOv8m (medium) for faster CPU inference
    "PRELOAD_OCR=1",
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

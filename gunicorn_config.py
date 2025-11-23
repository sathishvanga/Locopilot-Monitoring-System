"""
Gunicorn configuration for production deployment

This configuration uses multiprocessing for high-performance video processing.
"""

import multiprocessing
import os


# Worker configuration
# ✅ MEMORY FIX: Reduced from cpu_count//2 to max 2 workers to prevent memory explosion
workers = max(1, min(2, multiprocessing.cpu_count() // 4))
threads = 1
worker_class = "uvicorn.workers.UvicornWorker"
print(f" [gunicorn_config.py] CPU count: {multiprocessing.cpu_count()}")
print(f" [gunicorn_config.py] Workers: {workers} (limited to 2 max for memory management)")
# Application preloading
preload_app = True

# Timeouts
timeout = 600  # 10 minutes for video processing
graceful_timeout = 30
keepalive = 5

# Request limits
# ✅ MEMORY FIX: Reduced from 2000 to 100 to force worker restarts and prevent memory accumulation
max_requests = 100
max_requests_jitter = 10

# Binding
bind = "0.0.0.0:8000"

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "locopilot-monitor"

# Environment variables
raw_env = [
    "YOLO_WEIGHTS_PRELOAD=yolo11s.pt",
    "PRELOAD_OCR=1",
]

# Worker lifecycle hooks
def on_starting(server):
    """
    Called just before the master process is initialized
    """
    print("=" * 60)
    print("Starting Locopilot Monitoring System")
    print(f"Workers: {workers}")
    print(f"Threads per worker: {threads}")
    print(f"Timeout: {timeout}s")
    print(f"Bind: {bind}")
    print("=" * 60)


def on_reload(server):
    """
    Called when the server is reloaded
    """
    print("Server reloading...")


def worker_int(worker):
    """
    Called when a worker receives the SIGINT or SIGQUIT signal
    """
    print(f"Worker {worker.pid} received interrupt signal")


def worker_abort(worker):
    """
    Called when a worker receives the SIGABRT signal
    """
    print(f"Worker {worker.pid} aborted")


def pre_fork(server, worker):
    """
    Called just before a worker is forked
    """
    pass


def post_fork(server, worker):
    """
    Called after a worker has been forked
    """
    print(f"Worker spawned (pid: {worker.pid})")


def pre_exec(server):
    """
    Called before a new master process is forked
    """
    print("Forking new master process")


def when_ready(server):
    """
    Called just after the server is started
    """
    print("=" * 60)
    print("Locopilot Monitoring System ready to accept requests")
    print(f"Listening on {bind}")
    print("=" * 60)


def worker_exit(server, worker):
    """
    Called just after a worker has been exited
    """
    print(f"Worker {worker.pid} exited")


def nworkers_changed(server, new_value, old_value):
    """
    Called when the number of workers changes
    """
    print(f"Number of workers changed from {old_value} to {new_value}")


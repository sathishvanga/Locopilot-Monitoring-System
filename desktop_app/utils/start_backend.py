"""
Helper script to start the FastAPI backend server
This script is bundled with the app and can be run with system Python
"""
import sys
import os
import multiprocessing
from pathlib import Path

# Add the project root to Python path
if hasattr(sys, '_MEIPASS'):
    # Running from PyInstaller bundle
    project_root = Path(sys._MEIPASS)
else:
    # Running from source
    project_root = Path(__file__).parent.parent.parent

sys.path.insert(0, str(project_root))

# Now import and run uvicorn
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get('BACKEND_PORT', 8000))
    host = os.environ.get('BACKEND_HOST', '127.0.0.1')
    
    # Calculate optimal worker count (similar to gunicorn_config.py)
    # Desktop app: use cpu_count // 2 for better utilization, capped at 4
    cpu_count = multiprocessing.cpu_count()
    max_workers_cap = 4  # Prevent memory issues (each worker loads models)
    workers = max(1, min(cpu_count // 2, max_workers_cap))
    
    print(f"Starting backend with {workers} worker(s) (CPU count: {cpu_count})")
    
    # Run the FastAPI app with workers for optimal CPU utilization
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        workers=workers,  # ✅ CPU OPTIMIZATION: Use multiple workers
        log_level="warning"
    )


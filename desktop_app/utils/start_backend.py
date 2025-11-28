"""
Helper script to start the FastAPI backend server
This script is bundled with the app and can be run with system Python
"""
import sys
import os
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
    
    # Run the FastAPI app
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level="warning"
    )


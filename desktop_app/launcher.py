"""
Launcher script for PyInstaller packaged application

This script sets up the Python path and launches the main application.
It's needed because PyInstaller can't handle relative imports in the main script.
"""

import sys
import os
import multiprocessing as mp
import faulthandler
from pathlib import Path

# Ensure Requests/SSL can verify HTTPS inside the bundled app
try:
    import certifi
    _cafile = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", _cafile)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _cafile)
except Exception:
    pass

# Add the parent directory to Python path to enable package imports
if getattr(sys, 'frozen', False):
    # Running in PyInstaller bundle
    bundle_dir = Path(sys._MEIPASS)
    # Add bundle dir to path so 'desktop_app' package can be imported
    sys.path.insert(0, str(bundle_dir))
else:
    # Running in development
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    sys.path.insert(0, str(project_root))


def main_wrapper():
    """Wrapper to set up multiprocessing and faulthandler before launching app"""
    # Stabilize Qt + multiprocessing on macOS/Linux
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    
    # Enable faulthandler for better crash reporting
    try:
        faulthandler.enable()
    except Exception:
        pass
    
    # Now import and run the main application
    from desktop_app.main import main
    return main()


if __name__ == "__main__":
    sys.exit(main_wrapper())



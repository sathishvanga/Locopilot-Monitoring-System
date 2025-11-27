"""
Launcher script for PyInstaller packaged application

This script sets up the Python path and launches the main application.
It's needed because PyInstaller can't handle relative imports in the main script.
"""

import sys
import os
from pathlib import Path

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

# Now import and run the main application
from desktop_app.main import main

if __name__ == "__main__":
    sys.exit(main())



"""
Backend Manager Service

Manages the lifecycle of the FastAPI backend process for the desktop application.
Automatically starts the backend on app launch and gracefully shuts it down on exit.
"""

import os
import sys
import time
import socket
import subprocess
import signal
from typing import Optional
from pathlib import Path
import requests

from ..utils.logger import get_logger
from ..utils.config import get_settings


logger = get_logger(__name__)
settings = get_settings()


class BackendManager:
    """
    Service for managing the local FastAPI backend process
    
    Handles automatic startup, health checking, and graceful shutdown
    of the backend server process.
    """
    
    def __init__(self):
        """Initialize backend manager"""
        self.backend_process: Optional[subprocess.Popen] = None
        self.backend_started_by_us = False
        logger.info("Backend manager initialized")
        logger.info(f"Running in packaged mode: {self._is_packaged()}")
    
    def _is_packaged(self) -> bool:
        """
        Check if running from PyInstaller bundle
        
        Returns:
            bool: True if running from packaged app, False if running from source
        """
        return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')
    
    def _get_backend_path(self) -> Path:
        """
        Get path to backend code (packaged or development)
        
        Returns:
            Path: Path to the 'app' directory containing backend code
        """
        if self._is_packaged():
            # In packaged macOS app, backend is in Contents/Resources/app
            # sys._MEIPASS can point to Contents/Frameworks, so we need to adjust
            meipass = Path(sys._MEIPASS)
            logger.info(f"sys._MEIPASS: {meipass}")
            
            # Check if we're in Frameworks directory and adjust to Resources
            if meipass.name == 'Frameworks':
                # Navigate to Contents/Resources instead
                backend_path = meipass.parent / 'Resources' / 'app'
            else:
                # Standard path
                backend_path = meipass / 'app'
            
            logger.info(f"Using packaged backend at: {backend_path}")
        else:
            # In development, backend is in project root
            desktop_app_dir = Path(__file__).parent.parent
            backend_path = desktop_app_dir.parent / 'app'
            logger.info(f"Using development backend at: {backend_path}")
        
        return backend_path
    
    def is_backend_running(self) -> bool:
        """
        Check if local FastAPI backend is running
        
        Returns:
            bool: True if backend is running and healthy, False otherwise
        """
        try:
            # Try to connect to the port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', settings.local_backend_port))
            sock.close()
            
            if result == 0:
                # Port is open, check if it's our API with health endpoint
                try:
                    response = requests.get(
                        f"{settings.local_backend_url}/health",
                        timeout=5
                    )
                    return response.status_code == 200
                except:
                    return False
            
            return False
            
        except Exception as e:
            logger.debug(f"Backend check error: {e}")
            return False
    
    def start_backend(self) -> bool:
        """
        Start the FastAPI backend process
        
        If backend is already running, reuses existing instance.
        Otherwise, spawns a new uvicorn subprocess.
        
        Returns:
            bool: True if backend is running (new or existing), False if failed to start
        """
        try:
            # Check if backend is already running
            if self.is_backend_running():
                logger.info("Backend is already running - reusing existing instance")
                self.backend_started_by_us = False
                return True
            
            logger.info("Starting FastAPI backend...")
            
            # Get backend path (works in both packaged and development mode)
            backend_path = self._get_backend_path()
            app_main_path = backend_path / "main.py"
            
            # Verify the app.main module exists
            if not app_main_path.exists():
                logger.error(f"Backend main.py not found at: {app_main_path}")
                logger.error(f"Is packaged: {self._is_packaged()}")
                return False
            
            # Get the parent directory (project root or _MEIPASS)
            project_root = backend_path.parent
            logger.info(f"Backend project root: {project_root}")
            
            # Build uvicorn command
            # In packaged mode, sys.executable may not support spawning processes
            # Try to find system Python as fallback
            python_exe = sys.executable
            
            if self._is_packaged():
                # Try to use system Python instead of the packaged executable
                import shutil
                system_python = shutil.which('python3') or shutil.which('python')
                if system_python:
                    logger.info(f"Using system Python: {system_python}")
                    python_exe = system_python
                else:
                    logger.warning("No system Python found, using packaged executable (may fail)")
            
            cmd = [
                python_exe,
                "-m", "uvicorn",
                "app.main:app",
                "--host", "127.0.0.1",
                "--port", str(settings.local_backend_port),
                "--log-level", "warning"
            ]
            
            # Prepare environment variables
            env = os.environ.copy()
            
            # In packaged mode, ensure Python can find the backend modules
            if self._is_packaged():
                # Add project root to PYTHONPATH so 'app' module can be imported
                if 'PYTHONPATH' in env:
                    env['PYTHONPATH'] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
                else:
                    env['PYTHONPATH'] = str(project_root)
                logger.info(f"Set PYTHONPATH to: {env['PYTHONPATH']}")
            
            # Start backend process with proper working directory and environment
            self.backend_process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                # Prevent subprocess from inheriting signals
                start_new_session=True if os.name != 'nt' else False
            )
            
            logger.info(f"Backend process started with PID: {self.backend_process.pid}")
            self.backend_started_by_us = True
            
            # Wait for backend to become available
            if self._wait_for_startup(timeout=settings.backend_startup_timeout):
                logger.info("Backend started successfully")
                return True
            else:
                logger.warning("Backend failed to start within timeout")
                self._force_stop_backend()
                return False
            
        except FileNotFoundError as e:
            logger.error(f"Failed to start backend - uvicorn not found: {e}")
            logger.error("Please ensure uvicorn is installed: pip install uvicorn")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start backend: {e}", exc_info=True)
            return False
    
    def _wait_for_startup(self, timeout: int = 10) -> bool:
        """
        Wait for backend to become available
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if backend became available, False if timeout
        """
        logger.debug(f"Waiting for backend startup (timeout: {timeout}s)")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_backend_running():
                logger.debug("Backend is ready")
                return True
            time.sleep(0.5)
        
        return False
    
    def stop_backend(self) -> None:
        """
        Gracefully stop the backend process
        
        Only stops the backend if it was started by this manager instance.
        Existing backends (not started by us) are left running.
        """
        try:
            # Only stop if we started it
            if not self.backend_started_by_us:
                logger.info("Backend was not started by us - leaving it running")
                return
            
            if self.backend_process is None:
                logger.debug("No backend process to stop")
                return
            
            # Check if process is still running
            if self.backend_process.poll() is not None:
                logger.debug("Backend process already terminated")
                return
            
            logger.info(f"Stopping backend process (PID: {self.backend_process.pid})")
            
            # Try graceful termination first
            try:
                if os.name == 'nt':  # Windows
                    self.backend_process.terminate()
                else:  # Unix/Linux/macOS
                    os.killpg(os.getpgid(self.backend_process.pid), signal.SIGTERM)
                
                # Wait for process to terminate (max 5 seconds)
                try:
                    self.backend_process.wait(timeout=5)
                    logger.info("Backend stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination failed
                    logger.warning("Backend did not stop gracefully, forcing termination")
                    self._force_stop_backend()
            
            except ProcessLookupError:
                # Process already gone
                logger.debug("Backend process already terminated")
            
        except Exception as e:
            logger.error(f"Error stopping backend: {e}", exc_info=True)
    
    def _force_stop_backend(self) -> None:
        """Force kill the backend process"""
        if self.backend_process is None:
            return
        
        try:
            if os.name == 'nt':  # Windows
                self.backend_process.kill()
            else:  # Unix/Linux/macOS
                os.killpg(os.getpgid(self.backend_process.pid), signal.SIGKILL)
            
            self.backend_process.wait(timeout=2)
            logger.info("Backend process force killed")
        except Exception as e:
            logger.error(f"Error force killing backend: {e}")
    
    def get_backend_status(self) -> dict:
        """
        Get backend status information
        
        Returns:
            dict: Status information including running state, PID, etc.
        """
        is_running = self.is_backend_running()
        
        status = {
            "is_running": is_running,
            "started_by_us": self.backend_started_by_us,
            "process_pid": self.backend_process.pid if self.backend_process else None,
            "backend_url": settings.local_backend_url
        }
        
        return status


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
import threading
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
        self.backend_thread: Optional[threading.Thread] = None
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
            meipass = Path(sys._MEIPASS)
            logger.info(f"sys._MEIPASS: {meipass}")
            logger.info(f"Platform: {sys.platform}")
            
            # Platform-specific path resolution
            if sys.platform == 'darwin':  # macOS
                # In packaged macOS app, backend is in Contents/Resources/app
                # sys._MEIPASS can point to Contents/Frameworks, so we need to adjust
                if meipass.name == 'Frameworks':
                    # Navigate to Contents/Resources instead
                    backend_path = meipass.parent / 'Resources' / 'app'
                else:
                    # Standard path
                    backend_path = meipass / 'app'
            elif sys.platform == 'win32':  # Windows
                # On Windows onefile mode, app directory is directly in _MEIPASS
                backend_path = meipass / 'app'
            else:  # Linux or other
                # Standard path for other platforms
                backend_path = meipass / 'app'
            
            logger.info(f"Using packaged backend at: {backend_path}")
            logger.info(f"Backend path exists: {backend_path.exists()}")
            if backend_path.exists():
                logger.info(f"Backend path contents: {list(backend_path.iterdir())[:5]}")
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
            
            # In packaged mode, run uvicorn programmatically from within the app
            # This avoids requiring system Python or manual installation
            # User doesn't need to know about Python, uvicorn, or any tech stack
            if self._is_packaged():
                logger.info("Starting backend server (packaged mode)...")
                
                # Ensure the app module is in the path
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                
                # Change to project root directory for proper module resolution
                original_cwd = os.getcwd()
                os.chdir(str(project_root))
                
                # Start uvicorn in a daemon thread
                # This runs completely in the background - user never sees it
                self.backend_thread = threading.Thread(
                    target=self._run_uvicorn_in_thread,
                    args=(
                        str(project_root),
                        settings.local_backend_port,
                        "127.0.0.1",
                        original_cwd
                    ),
                    daemon=True,
                    name="BackendServer"
                )
                self.backend_thread.start()
                
                logger.info("Backend server thread started")
                self.backend_started_by_us = True
                
            else:
                # In development mode, use subprocess with current Python
                logger.info("Running uvicorn via subprocess (development mode)...")
                
                cmd = [
                    sys.executable,
                    "-m", "uvicorn",
                    "app.main:app",
                    "--host", "127.0.0.1",
                    "--port", str(settings.local_backend_port),
                    "--log-level", "warning"
                ]
                
                # Start backend process
                self.backend_process = subprocess.Popen(
                    cmd,
                    cwd=str(project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
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
                # Try to read error output for debugging
                if self.backend_process:
                    try:
                        _, stderr = self.backend_process.communicate(timeout=1)
                        if stderr:
                            logger.error(f"Backend stderr: {stderr.decode('utf-8', errors='ignore')}")
                    except:
                        pass
                # Check if thread failed
                if self.backend_thread and not self.backend_thread.is_alive():
                    logger.error("Backend server thread terminated unexpectedly")
                self._force_stop_backend()
                return False
            
        except FileNotFoundError as e:
            logger.error(f"Failed to start backend server: {e}")
            # Don't expose technical details to users - just log internally
            return False
            
        except Exception as e:
            logger.error(f"Failed to start backend: {e}", exc_info=True)
            # Try to read error output if process was created
            if self.backend_process:
                try:
                    _, stderr = self.backend_process.communicate(timeout=1)
                    if stderr:
                        logger.error(f"Backend stderr: {stderr.decode('utf-8', errors='ignore')}")
                except:
                    pass
            # Check if thread failed
            if self.backend_thread and not self.backend_thread.is_alive():
                logger.error("Backend server thread terminated unexpectedly")
            return False
    
    def _run_uvicorn_in_thread(self, project_root: str, port: int, host: str, original_cwd: str) -> None:
        """
        Run uvicorn programmatically in a background thread
        
        This function runs uvicorn in a daemon thread, completely hidden from the user.
        The user never sees Python, uvicorn, or any technical details - it just works.
        
        Args:
            project_root: Path to the project root (where app module is located)
            port: Port number to run the server on
            host: Host address to bind to
            original_cwd: Original working directory to restore if needed
        """
        try:
            # Set up the environment
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            # Change to project root directory for proper module resolution
            os.chdir(project_root)
            
            # Import uvicorn and run the server
            # This runs in the background - completely transparent to the user
            import uvicorn
            
            logger.debug(f"Starting uvicorn server on {host}:{port}")
            
            # Run uvicorn - this will block the thread until server stops
            uvicorn.run(
                "app.main:app",
                host=host,
                port=port,
                log_level="warning",
                access_log=False  # Don't log access requests to keep it quiet
            )
        except Exception as e:
            # Log error but don't expose technical details to user
            import traceback
            error_msg = f"Backend server error: {e}\n{traceback.format_exc()}\n"
            logger.error(error_msg)
            
            # Try to restore original directory
            try:
                os.chdir(original_cwd)
            except:
                pass
    
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
            
            # Handle threading.Thread (packaged mode)
            if self.backend_thread is not None:
                if not self.backend_thread.is_alive():
                    logger.debug("Backend thread already terminated")
                    return
                
                logger.info("Stopping backend server...")
                # For threads, we can't directly terminate uvicorn
                # Uvicorn will stop when the main process exits (daemon thread)
                # But we can try to signal it to stop gracefully
                # The thread will be cleaned up automatically when app exits
                logger.info("Backend server will stop when application exits")
                return
            
            # Handle subprocess.Popen (development mode)
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
        """Force stop the backend"""
        # Handle threading.Thread - daemon threads stop automatically
        if self.backend_thread is not None:
            # Daemon threads are automatically terminated when main process exits
            # We can't forcefully kill a thread, but uvicorn should stop gracefully
            logger.debug("Backend thread will be cleaned up automatically")
            return
        
        # Handle subprocess.Popen
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
        
        # Get PID from either process type
        process_pid = None
        if self.backend_thread is not None:
            # Threads don't have PIDs, use thread identifier
            process_pid = self.backend_thread.ident if self.backend_thread.is_alive() else None
        elif self.backend_process is not None:
            process_pid = self.backend_process.pid if self.backend_process.poll() is None else None
        
        status = {
            "is_running": is_running,
            "started_by_us": self.backend_started_by_us,
            "process_pid": process_pid,
            "backend_url": settings.local_backend_url
        }
        
        return status


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
import shlex
import shutil
from typing import Optional, Dict, Any
from pathlib import Path
import requests

from ..utils.logger import get_logger
from ..utils.config import get_settings
from ..utils.backend_health import check_backend_health
from ..utils.path_utils import is_packaged, get_backend_path, get_gunicorn_config_path, validate_path
from ..utils.constants import (
    DEFAULT_WORKER_COUNT,
    DEFAULT_WORKER_TIMEOUT,
    DEFAULT_BACKEND_PORT
)


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
        return is_packaged()
    
    def _get_backend_path(self) -> Path:
        """
        Get path to backend code (packaged or development)
        
        Returns:
            Path: Path to the 'app' directory containing backend code
            
        Raises:
            FileNotFoundError: If backend path cannot be determined
        """
        return get_backend_path()
    
    def is_backend_running(self) -> bool:
        """
        Check if local FastAPI backend is running
        
        Returns:
            bool: True if backend is running and healthy, False otherwise
        """
        return check_backend_health(
            backend_url=settings.local_backend_url,
            backend_port=settings.local_backend_port,
            timeout=5
        )
    
    def _get_gunicorn_config_path(self, project_root: Path) -> Optional[Path]:
        """
        Get path to gunicorn_config.py (packaged or development)
        
        Args:
            project_root: Project root directory
            
        Returns:
            Path: Path to gunicorn_config.py, or None if not found
        """
        return get_gunicorn_config_path(project_root)
    
    def start_backend(self) -> bool:
        """
        Start the FastAPI backend process using Gunicorn
        
        If backend is already running, reuses existing instance.
        Otherwise, spawns a new gunicorn subprocess with gunicorn_config.py.
        
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
            try:
                backend_path = self._get_backend_path()
            except FileNotFoundError as e:
                logger.error(f"Backend path not found: {e}")
                return False
            
            app_main_path = backend_path / "main.py"
            
            # Verify the app.main module exists
            if not app_main_path.exists():
                logger.error(f"Backend main.py not found at: {app_main_path}")
                logger.error(f"Is packaged: {self._is_packaged()}")
                return False
            
            # Get the parent directory (project root or _MEIPASS)
            project_root = backend_path.parent
            logger.info(f"Backend project root: {project_root}")
            
            # Security: Validate project root path
            try:
                project_root = validate_path(project_root, must_exist=True)
            except (ValueError, FileNotFoundError) as e:
                logger.error(f"Invalid project root: {e}")
                return False
            
            # Build gunicorn command
            # In packaged mode, sys.executable may not support spawning processes
            # Try to find system Python as fallback
            python_exe = sys.executable
            
            if self._is_packaged():
                # Try to use system Python instead of the packaged executable
                import shutil
                system_python = shutil.which('python3') or shutil.which('python')
                
                # Security: Validate system Python path
                if system_python:
                    system_python_real = os.path.realpath(system_python)
                    if not os.path.exists(system_python_real) or not os.access(system_python_real, os.X_OK):
                        logger.error(f"System Python path is invalid or not executable: {system_python_real}")
                        system_python = None
                
                if system_python:
                    logger.info(f"Checking system Python: {system_python}")
                    python_exe = system_python
                    
                    # Check if system Python has required packages
                    try:
                        import subprocess as sp
                        check_cmd = [python_exe, "-c", "import gunicorn, uvicorn, fastapi, torch; print('OK')"]
                        result = sp.run(check_cmd, capture_output=True, timeout=5)
                        if result.returncode != 0:
                            error_output = result.stderr.decode('utf-8', errors='ignore')
                            logger.warning(
                                f"System Python at {python_exe} is missing required packages.\n"
                                f"Error: {error_output}\n"
                                f"Attempting to use bundled Python instead..."
                            )
                            # Fall back to bundled Python (sys.executable)
                            python_exe = sys.executable
                            logger.info(f"Using bundled Python: {python_exe}")
                        else:
                            logger.info(f"System Python has required packages - using: {python_exe}")
                    except Exception as e:
                        logger.warning(f"Could not verify system Python packages: {e}, using bundled Python")
                        python_exe = sys.executable
                else:
                    logger.warning("No system Python found, using bundled executable")
                    python_exe = sys.executable
            
            # Find gunicorn_config.py path
            gunicorn_config_path = self._get_gunicorn_config_path(project_root)
            
            # Validate and sanitize paths before building command
            # Security: Validate python_exe path
            if not os.path.exists(python_exe) and not shutil.which(python_exe):
                logger.error(f"Python executable not found or invalid: {python_exe}")
                return False
            
            # Security: Validate project_root path
            project_root_real = os.path.realpath(str(project_root))
            if not os.path.isdir(project_root_real):
                logger.error(f"Invalid project root directory: {project_root_real}")
                return False
            
            # Build gunicorn command with sanitized paths
            if gunicorn_config_path and gunicorn_config_path.exists():
                # Security: Validate config path is within project root
                config_path_real = os.path.realpath(str(gunicorn_config_path))
                if not config_path_real.startswith(project_root_real + os.sep):
                    logger.error(f"Config path outside project root: {config_path_real}")
                    return False
                
                logger.info(f"Using gunicorn config: {gunicorn_config_path}")
                # Use gunicorn with config file - paths are validated
                cmd = [
                    python_exe,
                    "-m", "gunicorn",
                    "app.main:app",
                    "-c", config_path_real,  # Use realpath
                    "--bind", f"127.0.0.1:{settings.local_backend_port}"
                ]
            else:
                # Fallback: Use gunicorn without config file (with inline settings)
                logger.warning(f"gunicorn_config.py not found, using gunicorn with inline settings")
                logger.warning(f"Searched in: {project_root}")
                cmd = [
                    python_exe,
                    "-m", "gunicorn",
                    "app.main:app",
                    "--bind", f"127.0.0.1:{settings.local_backend_port}",
                    "--workers", str(DEFAULT_WORKER_COUNT),  # Single worker for desktop app
                    "--worker-class", "uvicorn.workers.UvicornWorker",
                    "--timeout", str(DEFAULT_WORKER_TIMEOUT),  # 10 minutes for video processing
                    "--log-level", "warning"
                ]
            
            # Prepare environment variables
            env = os.environ.copy()
            
            # Set GUNICORN_BIND to localhost for desktop app security
            env['GUNICORN_BIND'] = f"127.0.0.1:{settings.local_backend_port}"
            
            # In packaged mode, ensure Python can find the backend modules
            if self._is_packaged():
                # Add project root to PYTHONPATH so 'app' module can be imported
                if 'PYTHONPATH' in env:
                    env['PYTHONPATH'] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
                else:
                    env['PYTHONPATH'] = str(project_root)
                logger.info(f"Set PYTHONPATH to: {env['PYTHONPATH']}")
            
            # Security: Validate all command arguments are safe
            # All paths have been validated above, but double-check
            # Allow system executables (python, python3) and prevent path traversal
            allowed_system_paths = ['/usr', '/opt', '/bin', '/sbin', '/Library']
            for arg in cmd:
                if isinstance(arg, str):
                    # Block path traversal attempts
                    if '..' in arg:
                        logger.error(f"Unsafe command argument detected (path traversal): {arg}")
                        return False
                    # Allow system executables and project paths
                    if arg.startswith('/'):
                        is_system_path = any(arg.startswith(path) for path in allowed_system_paths)
                        is_project_path = arg.startswith(project_root_real)
                        if not (is_system_path or is_project_path):
                            logger.error(f"Unsafe command argument detected: {arg}")
                            return False
            
            # Start backend process with proper working directory and environment
            # Security: Use validated realpath for working directory
            self.backend_process = subprocess.Popen(
                cmd,
                cwd=project_root_real,  # Use validated realpath
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                # Prevent subprocess from inheriting signals
                start_new_session=True if os.name != 'nt' else False
            )
            
            logger.info(f"Backend process started with PID: {self.backend_process.pid}")
            logger.info(f"Command: {' '.join(cmd)}")
            self.backend_started_by_us = True
            
            # Wait for backend to become available
            if self._wait_for_startup(timeout=settings.backend_startup_timeout):
                logger.info("Backend started successfully")
                return True
            else:
                # Check if process crashed
                if self.backend_process.poll() is not None:
                    # Process terminated - read error output
                    try:
                        stderr_output = self.backend_process.stderr.read().decode('utf-8', errors='ignore')
                        stdout_output = self.backend_process.stdout.read().decode('utf-8', errors='ignore')
                        logger.error(f"Backend process crashed with exit code: {self.backend_process.returncode}")
                        if stderr_output:
                            # Log full stderr (not truncated) - it's important for debugging
                            logger.error(f"Backend stderr:\n{stderr_output}")
                        if stdout_output:
                            logger.error(f"Backend stdout:\n{stdout_output}")
                    except Exception as e:
                        logger.error(f"Could not read backend error output: {e}")
                else:
                    logger.warning("Backend process is still running but not responding to health checks")
                    # Try to read any available output
                    try:
                        # Use non-blocking read
                        import select
                        if os.name != 'nt':  # select doesn't work on Windows
                            if select.select([self.backend_process.stderr], [], [], 0)[0]:
                                stderr_output = self.backend_process.stderr.read(500).decode('utf-8', errors='ignore')
                                if stderr_output:
                                    logger.warning(f"Backend stderr (partial): {stderr_output}")
                    except:
                        pass
                
                logger.warning("Backend failed to start within timeout")
                self._force_stop_backend()
                return False
            
        except FileNotFoundError as e:
            logger.error(f"Failed to start backend - gunicorn not found: {e}")
            logger.error("Please ensure gunicorn is installed: pip install gunicorn")
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
        logger.info(f"Waiting for backend startup (timeout: {timeout}s)")
        
        start_time = time.time()
        check_interval = 0.5
        last_log_time = start_time
        
        while time.time() - start_time < timeout:
            # Check if process crashed
            if self.backend_process and self.backend_process.poll() is not None:
                logger.error(f"Backend process terminated unexpectedly with exit code: {self.backend_process.returncode}")
                return False
            
            # Check if backend is responding
            if self.is_backend_running():
                elapsed = time.time() - start_time
                logger.info(f"Backend is ready (started in {elapsed:.1f}s)")
                return True
            
            # Log progress every 2 seconds
            if time.time() - last_log_time >= 2:
                elapsed = time.time() - start_time
                logger.debug(f"Still waiting for backend... ({elapsed:.1f}s / {timeout}s)")
                last_log_time = time.time()
            
            time.sleep(check_interval)
        
        logger.warning(f"Backend did not become available within {timeout}s")
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
    
    def get_backend_status(self) -> Dict[str, Any]:
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


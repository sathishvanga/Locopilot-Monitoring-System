"""
Path resolution utilities for desktop application

Handles path resolution for both packaged (PyInstaller) and development modes.
"""

import os
import sys
from typing import Optional
from pathlib import Path

from .logger import get_logger


logger = get_logger(__name__)


def is_packaged() -> bool:
    """
    Check if running from PyInstaller bundle.
    
    Returns:
        bool: True if running from packaged app, False if running from source
    """
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')


def get_backend_path() -> Path:
    """
    Get path to backend code (packaged or development).
    
    Returns:
        Path: Path to the 'app' directory containing backend code
        
    Raises:
        FileNotFoundError: If backend path cannot be determined
    """
    if is_packaged():
        # In packaged macOS app, backend is in Contents/Resources/app
        # sys._MEIPASS can point to Contents/Frameworks, so we need to adjust
        meipass = Path(sys._MEIPASS)
        logger.info(f"sys._MEIPASS: {meipass}")
        
        # Try multiple possible locations
        possible_paths = [
            meipass / 'app',  # Standard path if _MEIPASS is Resources
            meipass.parent / 'Resources' / 'app',  # If _MEIPASS is Frameworks
            meipass.parent.parent / 'Resources' / 'app',  # If _MEIPASS is deeper
        ]
        
        # Find the first path that exists
        for backend_path in possible_paths:
            if backend_path.exists() and (backend_path / 'main.py').exists():
                logger.info(f"Using packaged backend at: {backend_path}")
                return backend_path
        
        # If none found, log all tried paths
        logger.error(f"Backend not found in expected locations. Tried: {possible_paths}")
        raise FileNotFoundError(f"Backend not found in packaged app. Tried: {possible_paths}")
    else:
        # In development, backend is in project root
        desktop_app_dir = Path(__file__).parent.parent
        backend_path = desktop_app_dir.parent / 'app'
        
        if not backend_path.exists():
            raise FileNotFoundError(f"Backend not found at: {backend_path}")
        
        logger.info(f"Using development backend at: {backend_path}")
        return backend_path


def get_gunicorn_config_path(project_root: Path) -> Optional[Path]:
    """
    Get path to gunicorn_config.py (packaged or development).
    
    Args:
        project_root: Project root directory
        
    Returns:
        Path: Path to gunicorn_config.py, or None if not found
    """
    # In development, config is in project root
    config_path = project_root / "gunicorn_config.py"
    if config_path.exists():
        return config_path
    
    # Try to find actual project root (where gunicorn_config.py should be)
    # Go up from desktop_app directory to find project root
    desktop_app_dir = Path(__file__).parent.parent
    actual_project_root = desktop_app_dir.parent
    config_path = actual_project_root / "gunicorn_config.py"
    if config_path.exists():
        logger.info(f"Found gunicorn_config.py in project root: {config_path}")
        return config_path
    
    # In packaged mode, might be in different locations
    if is_packaged():
        possible_paths = [
            project_root / "gunicorn_config.py",
            Path(sys._MEIPASS) / "gunicorn_config.py",
            Path(sys._MEIPASS).parent / "Resources" / "gunicorn_config.py",
            actual_project_root / "gunicorn_config.py",  # Try actual project root
        ]
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found gunicorn_config.py in packaged location: {path}")
                return path
    
    logger.debug(f"gunicorn_config.py not found. Searched: {project_root}, {actual_project_root}")
    return None


def validate_path(path: Path, must_exist: bool = True) -> Path:
    """
    Validate and resolve a path.
    
    Args:
        path: Path to validate
        must_exist: Whether path must exist
        
    Returns:
        Path: Resolved real path
        
    Raises:
        ValueError: If path is invalid
        FileNotFoundError: If path doesn't exist and must_exist is True
    """
    try:
        real_path = path.resolve()
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid path: {path} - {e}") from e
    
    if must_exist and not real_path.exists():
        raise FileNotFoundError(f"Path does not exist: {real_path}")
    
    return real_path


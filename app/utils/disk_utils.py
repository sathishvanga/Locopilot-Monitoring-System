"""
Disk space utilities for validating sufficient storage before file uploads.

This module provides utilities to check available disk space and ensure
sufficient storage exists for large file uploads (up to 1GB).
"""

import os
import shutil
from typing import Tuple
from app.utils.logger import get_logger

logger = get_logger(__name__)


def get_free_disk_space(directory: str) -> int:
    """
    Get the free disk space available in the specified directory.

    Args:
        directory: Path to check disk space for (directory must exist)

    Returns:
        Free disk space in bytes

    Raises:
        FileNotFoundError: If directory does not exist
        OSError: If unable to get disk usage stats
    """
    try:
        # Ensure directory exists
        if not os.path.exists(directory):
            # Try parent directory if specified directory doesn't exist yet
            parent = os.path.dirname(directory)
            if os.path.exists(parent):
                directory = parent
            else:
                raise FileNotFoundError(f"Directory does not exist: {directory}")

        # Get disk usage statistics
        stat = shutil.disk_usage(directory)
        free_bytes = stat.free

        logger.debug(
            f"Disk space for {directory}: "
            f"Total={stat.total / (1024**3):.2f}GB, "
            f"Used={stat.used / (1024**3):.2f}GB, "
            f"Free={free_bytes / (1024**3):.2f}GB"
        )

        return free_bytes

    except Exception as e:
        logger.error(f"Failed to get disk space for {directory}: {e}", exc_info=True)
        raise


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable string (KB, MB, GB).

    Args:
        bytes_value: Number of bytes

    Returns:
        Human-readable string (e.g., "1.5 GB", "500 MB")
    """
    if bytes_value < 1024:
        return f"{bytes_value} B"
    elif bytes_value < 1024 ** 2:
        return f"{bytes_value / 1024:.1f} KB"
    elif bytes_value < 1024 ** 3:
        return f"{bytes_value / (1024 ** 2):.1f} MB"
    else:
        return f"{bytes_value / (1024 ** 3):.2f} GB"


def check_disk_space_available(
    directory: str,
    required_bytes: int,
    reserve_gb: int = 5
) -> Tuple[bool, str]:
    """
    Check if sufficient disk space is available for a file upload.

    This function ensures that after uploading the file, there will still be
    at least `reserve_gb` GB of free space remaining on the disk.

    Args:
        directory: Directory where file will be uploaded
        required_bytes: Number of bytes required for the upload
        reserve_gb: Minimum GB to keep free after upload (default: 5)

    Returns:
        Tuple of (is_available, error_message)
        - (True, "") if sufficient space available
        - (False, "error message") if insufficient space

    Examples:
        >>> has_space, msg = check_disk_space_available("/tmp", 1024**3, 5)
        >>> if not has_space:
        ...     print(msg)
        "Insufficient disk space. Required: 1.00 GB + 5 GB reserve, Available: 3.50 GB"
    """
    try:
        # Get current free disk space
        free_bytes = get_free_disk_space(directory)

        # Calculate minimum required space (file + reserve)
        reserve_bytes = reserve_gb * 1024 ** 3
        total_required = required_bytes + reserve_bytes

        # Check if enough space available
        if free_bytes < total_required:
            error_msg = (
                f"Insufficient disk space. "
                f"Required: {format_bytes(required_bytes)} + {reserve_gb} GB reserve "
                f"({format_bytes(total_required)} total), "
                f"Available: {format_bytes(free_bytes)}"
            )
            logger.warning(f"❌ Disk space check failed: {error_msg}")
            return False, error_msg

        logger.info(
            f"✅ Disk space check passed: "
            f"{format_bytes(free_bytes)} available, "
            f"{format_bytes(total_required)} required"
        )
        return True, ""

    except Exception as e:
        error_msg = f"Failed to check disk space: {str(e)}"
        logger.error(f"❌ {error_msg}", exc_info=True)
        return False, error_msg


def ensure_directory_exists(directory: str, create_if_missing: bool = True) -> bool:
    """
    Ensure that a directory exists, optionally creating it if missing.

    Args:
        directory: Directory path to check/create
        create_if_missing: If True, create directory if it doesn't exist

    Returns:
        True if directory exists (or was created), False otherwise

    Raises:
        OSError: If directory creation fails
    """
    try:
        if os.path.exists(directory):
            if not os.path.isdir(directory):
                logger.error(f"❌ Path exists but is not a directory: {directory}")
                return False
            return True

        if create_if_missing:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")
            return True
        else:
            logger.warning(f"⚠️ Directory does not exist: {directory}")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to ensure directory exists {directory}: {e}", exc_info=True)
        raise

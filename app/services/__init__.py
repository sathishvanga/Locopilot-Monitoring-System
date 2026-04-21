"""
Services package - Business logic and processing
"""

from typing import Any

from .video_processing_service import VideoProcessingService
from .activity_detection_service import ActivityDetectionService
from .image_preprocessing_service import ImagePreprocessingService
# C-1 (task 2.2): only import the class + accessor eagerly. The legacy module-
# level singleton name ``gpu_resource_manager`` is resolved lazily via the
# ``__getattr__`` hook below so that importing ``app.services`` does not
# instantiate the manager (or initialise CUDA) at import time.
from .gpu_resource_manager import GPUResourceManager, get_gpu_resource_manager
from .job_manager import JobManager, job_manager

__all__ = [
    "VideoProcessingService",
    "ActivityDetectionService",
    "ImagePreprocessingService",
    "GPUResourceManager",
    "get_gpu_resource_manager",
    "gpu_resource_manager",
    "JobManager",
    "job_manager",
]


def __getattr__(name: str) -> Any:
    """Lazy backward-compat shim for ``app.services.gpu_resource_manager``.

    Preserves ``from app.services import gpu_resource_manager`` at the cost
    of deferring singleton construction until first access.
    """
    if name == "gpu_resource_manager":
        return get_gpu_resource_manager()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


"""
Controllers package - API route handlers
"""

from .video_controller import router as video_router
from .v2_video_controller import router as v2_video_router

__all__ = ["video_router", "v2_video_router"]


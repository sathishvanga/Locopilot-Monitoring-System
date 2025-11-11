"""
Services package - Business logic and processing
"""

from .video_processing_service import VideoProcessingService
from .activity_detection_service import ActivityDetectionService

__all__ = ["VideoProcessingService", "ActivityDetectionService"]


"""YOLO model handlers for object and pose detection."""

from app.core.models.yolo_handler import (
    YOLOHandler,
    YOLO_KEYPOINT_INDICES,
    YOLO_HEAD_INDICES,
    YOLO_BODY_INDICES,
    YOLO_MIN_KEYPOINTS,
)

__all__ = [
    'YOLOHandler',
    'YOLO_KEYPOINT_INDICES',
    'YOLO_HEAD_INDICES',
    'YOLO_BODY_INDICES',
    'YOLO_MIN_KEYPOINTS',
]

"""Core utility modules for the application.

This package contains utility functions and classes for common operations
such as geometry calculations, data processing, and helper functions.
"""
from app.core.utils.geometry import (
    calculate_iou,
    _compute_iou,
    bbox_overlap_with_margin,
    deduplicate_person_boxes,
)
from app.core.utils.pose_utils import (
    get_keypoint,
    YOLO_KEYPOINT_INDICES,
    calculate_wrist_distance,
)

__all__ = [
    "calculate_iou",
    "_compute_iou",
    "bbox_overlap_with_margin",
    "deduplicate_person_boxes",
    "get_keypoint",
    "YOLO_KEYPOINT_INDICES",
    "calculate_wrist_distance",
]

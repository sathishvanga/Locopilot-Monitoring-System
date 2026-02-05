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

__all__ = [
    "calculate_iou",
    "_compute_iou",
    "bbox_overlap_with_margin",
    "deduplicate_person_boxes",
]

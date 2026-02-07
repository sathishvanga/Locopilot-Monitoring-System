"""Tracking modules for person and object tracking.

This module provides tracking classes for maintaining temporal
consistency of detected persons and their roles across video frames.
"""

from app.core.tracking.person_tracker import PersonTracker

__all__ = [
    'PersonTracker',
]

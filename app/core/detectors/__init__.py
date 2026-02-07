"""Detectors for activity, gesture, and object recognition.

This module provides specialized detector classes for monitoring
locomotive cab activities and crew coordination.
"""

from app.core.detectors.activity_detector import ActivityDetector
from app.core.detectors.gesture_detector import GestureDetector
from app.core.detectors.mind_diversion_detector import MindDiversionDetector
from app.core.detectors.object_detector import ObjectDetector
from app.core.detectors.sleep_detector import SleepDetector

__all__ = [
    'ActivityDetector',
    'GestureDetector',
    'MindDiversionDetector',
    'ObjectDetector',
    'SleepDetector',
]

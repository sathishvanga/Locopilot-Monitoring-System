"""Core modules for the locopilot application.

This module provides core functionality for:
- Activity state management and tracking (ActivityTracker)
- Activity configuration (ActivityConfig, ActivityState)
- Evidence management (EvidenceManager)
"""
from app.core.activity_tracker import ActivityConfig, ActivityState, ActivityTracker
from app.core.evidence_manager import EvidenceManager

__all__ = [
    'ActivityConfig',
    'ActivityState',
    'ActivityTracker',
    'EvidenceManager',
]

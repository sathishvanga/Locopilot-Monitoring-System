"""
Tests for temporal filtering fixes (2026-02-11).

Tests the 5 fixes applied to temporal_filtering_service.py:
  Fix 1: Post-loop reclassification check
  Fix 3: Writing max duration cap
  Fix 4: Self-reclassification = rejection
  Fix 5: Conflicting activity suppression (packing_bags vs writing)

These tests exercise the temporal filtering logic without requiring
video files, VLM servers, or GPU — they mock cv2/numpy at sys.modules
level and feed synthetic frame_detections.
"""

import importlib
import logging
import os
import sys
import tempfile
from datetime import timedelta
from typing import Dict, List
from unittest.mock import MagicMock, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Mock cv2 and numpy before importing the service module
# ---------------------------------------------------------------------------

_cv2_mock = MagicMock()
_cv2_mock.CAP_PROP_FRAME_COUNT = 7
_cv2_mock.CAP_PROP_FPS = 5

# Create a VideoCapture mock that returns sane metadata
_cap_instance = MagicMock()
_cap_instance.get.side_effect = lambda prop: {7: 100000, 5: 25.0}.get(prop, 0)
_cap_instance.read.return_value = (True, MagicMock())
_cv2_mock.VideoCapture.return_value = _cap_instance

# numpy mock (needed by vlm_verification_service import)
_np_mock = MagicMock()

# Ensure mocks are in sys.modules before any app imports
if 'cv2' not in sys.modules:
    sys.modules['cv2'] = _cv2_mock
if 'numpy' not in sys.modules:
    sys.modules['numpy'] = _np_mock
if 'httpx' not in sys.modules:
    sys.modules['httpx'] = MagicMock()

# Now import the service
from app.services.temporal_filtering_service import TemporalFilteringService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ACTIVITY_THRESHOLDS = {
    'microsleep': {'required_consecutive': 1, 'grace_frames': 10, 'min_duration': 2.0},
    'sleep': {'required_consecutive': 1, 'grace_frames': 10, 'min_duration': 2.0},
    'cell_phone': {'required_consecutive': 1, 'grace_frames': 8, 'min_duration': 0.1},
    'writing': {'required_consecutive': 3, 'grace_frames': 10, 'min_duration': 0.1},
    'packing_bags': {'required_consecutive': 2, 'grace_frames': 5, 'min_duration': 0.0},
    'group_detected': {'required_consecutive': 3, 'grace_frames': 8, 'min_duration': 0.0},
    'lp_hand_gesture': {'required_consecutive': 1, 'grace_frames': 5, 'min_duration': 0.0},
    'alp_hand_gesture': {'required_consecutive': 1, 'grace_frames': 5, 'min_duration': 0.0},
    'mind_diversion': {'required_consecutive': 2, 'grace_frames': 5, 'min_duration': 0.0},
    'eating_drinking': {'required_consecutive': 2, 'grace_frames': 5, 'min_duration': 0.0},
    'no_person_detected': {'required_consecutive': 3, 'grace_frames': 3, 'min_duration': 5.0},
    'alp_not_standing': {'required_consecutive': 2, 'grace_frames': 3, 'min_duration': 0.0},
}

ACTIVITY_TYPE_MAP = {
    'microsleep': 1, 'sleep': 2, 'cell_phone': 3, 'writing': 4,
    'packing_bags': 5, 'group_detected': 6, 'lp_hand_gesture': 7,
    'alp_hand_gesture': 8, 'mind_diversion': 9, 'eating_drinking': 13,
    'no_person_detected': 10, 'alp_not_standing': 11,
}

DESCRIPTIONS = {k: f'{k} detected' for k in ACTIVITY_TYPE_MAP}
EVIDENCE_RULES = {k: f'rule_{k}' for k in ACTIVITY_TYPE_MAP}


def _empty_activities_map() -> Dict[str, bool]:
    return {name: False for name in ACTIVITY_THRESHOLDS}


def _make_detection(frame_idx: int, timestamp_sec: float, **active_activities) -> Dict:
    """Build a single frame detection dict with specified activities active."""
    amap = _empty_activities_map()
    for name, val in active_activities.items():
        if name in amap:
            amap[name] = val
    return {
        'frame_idx': frame_idx,
        'timestamp_sec': timestamp_sec,
        'activities_map': amap,
        'person_roles': {},
        'persons_data_summary': {},
    }


def _make_service(vlm_service=None, settings=None) -> TemporalFilteringService:
    return TemporalFilteringService(
        activity_thresholds=ACTIVITY_THRESHOLDS,
        activity_type_map=ACTIVITY_TYPE_MAP,
        activity_descriptions=DESCRIPTIONS,
        evidence_rules=EVIDENCE_RULES,
        sample_fps=0.5,
        logger=logging.getLogger('test_temporal'),
        vlm_service=vlm_service,
        settings=settings,
    )


def _run(detections: List[Dict], vlm_service=None, settings=None) -> List[Dict]:
    """Run temporal filtering and return detected activities."""
    svc = _make_service(vlm_service=vlm_service, settings=settings)

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        dummy_video = f.name

    try:
        return svc.apply_temporal_filtering(
            frame_detections=detections,
            video_path=dummy_video,
            run_dir=None,
            trip_id='test-trip',
            fps=25.0,
            sample_fps=0.5,
            crew_name='Test',
            crew_id='test-id',
            crew_role=1,
            crew_members={},
            camera_angle=1,
            save_clips=False,
        )
    finally:
        os.unlink(dummy_video)


# ---------------------------------------------------------------------------
# Fix 1: Post-loop reclassification check
# ---------------------------------------------------------------------------

class TestFix1PostLoopReclassification:

    def test_reclassified_sleep_starts_via_post_loop(self):
        """Sleep should start when VLM reclassifies writing→sleep,
        even though sleep comes before writing in dict iteration order."""
        vlm = MagicMock()
        vlm.verify_detection_sync.return_value = (
            False, {'reason': 'reclassify:sleeping', 'confidence': 'high'}
        )

        # 20 frames of writing at 2s intervals
        dets = [_make_detection(i * 50, i * 2.0, writing=True) for i in range(20)]
        result = _run(dets, vlm_service=vlm)

        types = [a['objectType'] for a in result]
        assert any('sleep' in t for t in types), (
            f"Expected sleep via reclassification, got: {types}"
        )

    def test_reclassified_microsleep_starts(self):
        """microsleep should start when VLM reclassifies writing→microsleep."""
        vlm = MagicMock()
        vlm.verify_detection_sync.return_value = (
            False, {'reason': 'reclassify:microsleep', 'confidence': 'high'}
        )

        dets = [_make_detection(i * 50, i * 2.0, writing=True) for i in range(15)]
        result = _run(dets, vlm_service=vlm)

        types = [a['objectType'] for a in result]
        assert any('microsleep' in t for t in types), (
            f"Expected microsleep via reclassification, got: {types}"
        )


# ---------------------------------------------------------------------------
# Fix 3: Writing max duration cap
# ---------------------------------------------------------------------------

class TestFix3WritingMaxDuration:

    def test_writing_capped_at_max_duration(self):
        """Writing >120s should be force-ended, producing multiple activities."""
        settings = MagicMock()
        settings.writing_max_duration = 120.0
        settings.vlm_sleep_screening_enabled = False

        # 200 frames * 2s = 400s of writing
        dets = [_make_detection(i * 50, i * 2.0, writing=True) for i in range(200)]
        result = _run(dets, settings=settings)

        writing = [a for a in result if 'writing' in a['objectType']]
        assert len(writing) >= 2, (
            f"Expected >=2 writing segments from 400s duration, got {len(writing)}"
        )

    def test_writing_under_cap_is_single_activity(self):
        """Writing <120s should produce exactly one activity."""
        settings = MagicMock()
        settings.writing_max_duration = 120.0
        settings.vlm_sleep_screening_enabled = False

        # 30 frames * 2s = 60s
        dets = [_make_detection(i * 50, i * 2.0, writing=True) for i in range(30)]
        result = _run(dets, settings=settings)

        writing = [a for a in result if 'writing' in a['objectType']]
        assert len(writing) == 1, (
            f"Expected 1 writing for 60s duration, got {len(writing)}"
        )

    def test_writing_suppressed_in_frame_after_force_end(self):
        """After force-end, writing should not restart in the same frame."""
        settings = MagicMock()
        settings.writing_max_duration = 10.0  # Very short cap for testing
        settings.vlm_sleep_screening_enabled = False

        # 20 frames * 2s = 40s
        dets = [_make_detection(i * 50, i * 2.0, writing=True) for i in range(20)]
        result = _run(dets, settings=settings)

        writing = [a for a in result if 'writing' in a['objectType']]
        # With 10s cap and 40s of writing, should get multiple segments
        assert len(writing) >= 2, (
            f"Expected multiple writing segments with 10s cap, got {len(writing)}"
        )


# ---------------------------------------------------------------------------
# Fix 4: Self-reclassification = rejection
# ---------------------------------------------------------------------------

class TestFix4SelfReclassification:

    def test_self_reclassify_does_not_start_activity(self):
        """VLM returning reclassify:alp_hand_gesture for alp_hand_gesture
        should NOT confirm the activity."""
        vlm = MagicMock()
        vlm.verify_detection_sync.return_value = (
            False, {'reason': 'reclassify:alp_hand_gesture', 'confidence': 'low'}
        )

        dets = [_make_detection(i * 50, i * 2.0, alp_hand_gesture=True) for i in range(10)]
        result = _run(dets, vlm_service=vlm)

        hg = [a for a in result if 'hand_gesture' in a['objectType']]
        assert len(hg) == 0, (
            f"Self-reclassification should not confirm activity, got {len(hg)} hand_gesture(s)"
        )

    def test_self_reclassify_writing_prevents_start(self):
        """VLM returning reclassify:writing for writing should reject."""
        vlm = MagicMock()
        vlm.verify_detection_sync.return_value = (
            False, {'reason': 'reclassify:writing', 'confidence': 'low'}
        )

        dets = [_make_detection(i * 50, i * 2.0, writing=True) for i in range(20)]
        result = _run(dets, vlm_service=vlm)

        writing = [a for a in result if 'writing' in a['objectType']]
        assert len(writing) == 0, (
            f"Self-reclassified writing should never start, got {len(writing)}"
        )


# ---------------------------------------------------------------------------
# Fix 5: Conflicting activity suppression
# ---------------------------------------------------------------------------

class TestFix5ConflictingSuppression:

    def test_packing_suppresses_writing(self):
        """When both packing_bags and writing are True, only packing starts."""
        dets = [
            _make_detection(i * 50, i * 2.0, writing=True, packing_bags=True)
            for i in range(15)
        ]
        result = _run(dets)

        types = [a['objectType'] for a in result]
        assert any('packing' in t for t in types), f"Expected packing, got: {types}"
        assert not any('writing' == t.replace(' ', '_') for t in types), (
            f"Writing should be suppressed, got: {types}"
        )

    def test_writing_works_alone(self):
        """Writing activates normally without packing_bags."""
        dets = [_make_detection(i * 50, i * 2.0, writing=True) for i in range(10)]
        result = _run(dets)

        types = [a['objectType'] for a in result]
        assert any('writing' in t for t in types), (
            f"Expected writing, got: {types}"
        )

    def test_packing_works_alone(self):
        """packing_bags activates normally without writing."""
        dets = [_make_detection(i * 50, i * 2.0, packing_bags=True) for i in range(10)]
        result = _run(dets)

        types = [a['objectType'] for a in result]
        assert any('packing' in t for t in types), (
            f"Expected packing_bags, got: {types}"
        )

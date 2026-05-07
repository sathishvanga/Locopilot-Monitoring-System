"""Static object suppression filter.

Lifted from ``LocopilotActivityMonitor._update_static_backpack_tracking`` and
``_update_static_phone_tracking`` in ``locopilot_monitor.py`` (T5 of the
refactor plan at ``docs/specs/locopilot-refactor/PLAN.md``).

Both monolith methods implement the identical IoU-NMS-by-frame-count algorithm
to suppress detections that remain stationary across frames (cabin fixtures
misidentified as backpacks, panel instruments misidentified as phones). This
module unifies them into a single ``StaticObjectFilter`` parameterised by
``label`` and ``log_level`` so the existing log strings remain byte-identical
(operators grep ``[STATIC BACKPACK]`` and ``[STATIC PHONE]`` in production).
"""
from __future__ import annotations

import logging
from typing import List, Optional

from app.core.utils.geometry import calculate_iou


class StaticObjectFilter:
    """Tracks bbox stability across frames; objects that remain in the same
    location (IoU >= threshold) for >= min_frames are flagged static and
    filtered out of detection lists.

    Notes:
        - ``label`` is rendered into log lines uppercased (``[STATIC BACKPACK]``
          / ``[STATIC PHONE]``) and lowercased in the human-readable phrase
          (``Suppressed static backpack`` / ``Suppressed static phone``) to
          match the original log strings verbatim.
        - For ``label='phone'`` the suffix ``" — likely panel instrument"``
          (an em-dash) is appended to the suppression message to preserve the
          legacy phone log.
        - ``log_level='debug'`` mirrors the backpack tracker; ``'info'`` mirrors
          the phone tracker.
    """

    def __init__(
        self,
        *,
        label: str,
        iou_threshold: float,
        min_frames: int,
        enabled: bool,
        log_level: str = 'debug',
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.label = label
        self.iou_threshold = iou_threshold
        self.min_frames = min_frames
        self.enabled = enabled
        self.log_level = log_level
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.candidates: list = []

    def _emit(self, message: str) -> None:
        if self.log_level == 'info':
            self.logger.info(message)
        else:
            self.logger.debug(message)

    def filter(self, detections: List) -> List:
        """Update candidate tracking with the current frame's detections and
        return only those that have NOT been classified as static.

        The detection format mirrors the monolith: each entry is either a
        bbox-prefixed sequence (``[x1, y1, x2, y2, ...]``) or a 4-tuple bbox.
        The returned list preserves the original detection objects.
        """
        if not self.enabled or not detections:
            return detections

        current_bboxes = [det[:4] if len(det) >= 4 else det for det in detections]
        new_candidates = []

        for curr_bbox in current_bboxes:
            best_iou = 0.0
            best_idx = -1
            for idx, cand in enumerate(self.candidates):
                iou = calculate_iou(curr_bbox, cand['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= self.iou_threshold and best_idx >= 0:
                # Same object as previous frame — increment counter
                old = self.candidates[best_idx]
                new_candidates.append({
                    'bbox': list(curr_bbox),
                    'frame_count': old['frame_count'] + 1,
                })
            else:
                # New detection — start tracking
                new_candidates.append({
                    'bbox': list(curr_bbox),
                    'frame_count': 1,
                })

        self.candidates = new_candidates

        # Filter: keep only detections that have NOT been static for too long
        filtered = []
        for det in detections:
            bbox = det[:4] if len(det) >= 4 else det
            is_static = False
            matched_cand = None
            for cand in self.candidates:
                iou = calculate_iou(bbox, cand['bbox'])
                if iou >= self.iou_threshold and cand['frame_count'] >= self.min_frames:
                    is_static = True
                    matched_cand = cand
                    break
            if not is_static:
                filtered.append(det)
            else:
                if self.label == 'backpack':
                    self._emit(
                        f"[STATIC BACKPACK] Suppressed static backpack at {list(bbox[:4])} "
                        f"(seen for {matched_cand['frame_count']} frames)"
                    )
                elif self.label == 'phone':
                    self._emit(
                        f"[STATIC PHONE] Suppressed static phone at {[int(x) for x in bbox[:4]]} "
                        f"(seen for {matched_cand['frame_count']} frames — likely panel instrument)"
                    )
                else:
                    # Generic fallback for future labels; keeps the same shape.
                    self._emit(
                        f"[STATIC {self.label.upper()}] Suppressed static {self.label.lower()} at {list(bbox[:4])} "
                        f"(seen for {matched_cand['frame_count']} frames)"
                    )

        return filtered

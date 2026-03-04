"""Backpack Persistence Tracker — bridges short YOLO detection gaps.

YOLO intermittently misses backpack/suitcase detections across frames.
At 0.5 FPS, even one missed frame = 2-second gap, breaking the motion
analysis that needs continuous hand-near-backpack signals.

Strategy:
  1. Track recently-seen backpacks via IoU matching
  2. On a miss, attempt ROI re-detection (crop + CLAHE + upscale + YOLO)
  3. If ROI fails and misses < threshold, inject phantom bbox (last-known position)
  4. Inject recovered/phantom bboxes into detections['backpack'] in-place
     so downstream packing detection sees them with zero code changes
"""

import cv2
import numpy as np
from collections import deque
from typing import Any, Dict, List, Optional

from app.core.utils.geometry import calculate_iou


def _to_int_bbox(bbox) -> List[int]:
    """Ensure bbox coordinates are plain ints (YOLO returns float tensors)."""
    return [int(v) for v in bbox]


class _TrackedBackpack:
    """Internal state for a single tracked backpack."""

    __slots__ = (
        "bbox", "detection_history", "consecutive_misses",
        "last_detected_bbox", "source",
    )

    def __init__(self, bbox: List[int], history_len: int):
        self.bbox: List[int] = _to_int_bbox(bbox)
        self.detection_history: deque = deque(maxlen=history_len)
        self.detection_history.append(True)
        self.consecutive_misses: int = 0
        self.last_detected_bbox: List[int] = _to_int_bbox(bbox)
        self.source: str = "yolo"


class BackpackPersistenceTracker:
    """Bridges short YOLO detection gaps for backpack/suitcase objects."""

    def __init__(
        self,
        settings: Any,
        yolo_handler: Any,
        preprocessing_service: Optional[Any] = None,
        logger: Optional[Any] = None,
    ):
        self.settings = settings
        self.yolo_handler = yolo_handler
        self.preprocessing_service = preprocessing_service
        self.logger = logger

        # Config
        self.enabled: bool = getattr(settings, "backpack_persistence_enabled", True)
        self.history_len: int = getattr(settings, "backpack_persistence_history", 6)
        self.min_detections: int = getattr(settings, "backpack_persistence_min_detections", 2)
        self.max_phantom: int = getattr(settings, "backpack_persistence_max_phantom", 3)
        self.roi_margin: float = getattr(settings, "backpack_persistence_roi_margin", 0.20)
        self.roi_confidence: float = getattr(settings, "backpack_persistence_roi_confidence", 0.20)
        self.iou_threshold: float = getattr(settings, "backpack_persistence_iou_threshold", 0.3)
        self.roi_upscale: int = getattr(settings, "backpack_persistence_roi_upscale", 2)
        self.roi_preprocess: bool = getattr(settings, "backpack_persistence_roi_preprocess", True)

        # State
        self._tracked: List[_TrackedBackpack] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_and_recover(
        self,
        frame: np.ndarray,
        detections: Dict[str, List],
        timestamp_sec: float,
    ) -> None:
        """Main entry point — call after YOLO detection, before packing logic.

        Mutates *detections['backpack']* in-place by injecting recovered or
        phantom bboxes for backpacks that YOLO missed this frame.
        """
        if not self.enabled:
            return

        current_bboxes = detections.get("backpack", [])

        # 1. Match current detections to tracked backpacks
        matched_tracked_indices, matched_current_indices = self._match_current_to_tracked(
            current_bboxes
        )

        # 2. Update matched tracked entries
        for t_idx, c_idx in zip(matched_tracked_indices, matched_current_indices):
            tracked = self._tracked[t_idx]
            tracked.bbox = _to_int_bbox(current_bboxes[c_idx])
            tracked.last_detected_bbox = _to_int_bbox(current_bboxes[c_idx])
            tracked.detection_history.append(True)
            tracked.consecutive_misses = 0
            tracked.source = "yolo"

        # 3. Create new tracked entries for unmatched current detections
        for c_idx in range(len(current_bboxes)):
            if c_idx not in matched_current_indices:
                self._tracked.append(
                    _TrackedBackpack(current_bboxes[c_idx], self.history_len)
                )

        # 4. Handle tracked backpacks that were missed this frame
        recovered_count = 0
        phantom_count = 0
        indices_to_prune = []

        for t_idx in range(len(self._tracked)):
            if t_idx in matched_tracked_indices:
                continue  # Already matched — skip

            tracked = self._tracked[t_idx]
            tracked.detection_history.append(False)
            tracked.consecutive_misses += 1

            # Only attempt recovery if we have enough history
            positive_hits = sum(tracked.detection_history)
            if positive_hits < self.min_detections:
                if tracked.consecutive_misses >= self.max_phantom:
                    indices_to_prune.append(t_idx)
                continue

            # Attempt ROI re-detection
            redetected_bbox = self._attempt_roi_redetection(frame, tracked)
            if redetected_bbox is not None:
                int_bbox = _to_int_bbox(redetected_bbox)
                tracked.bbox = int_bbox
                tracked.last_detected_bbox = int_bbox
                tracked.consecutive_misses = 0
                tracked.source = "roi_redetect"
                detections["backpack"].append(int_bbox)
                recovered_count += 1
            elif tracked.consecutive_misses < self.max_phantom:
                # Inject phantom bbox at last-known position
                phantom_bbox = list(tracked.last_detected_bbox)  # already int from init/update
                tracked.bbox = phantom_bbox
                tracked.source = "phantom"
                detections["backpack"].append(phantom_bbox)
                phantom_count += 1
            else:
                indices_to_prune.append(t_idx)

        # 5. Prune stale tracked entries (iterate in reverse to preserve indices)
        for t_idx in sorted(indices_to_prune, reverse=True):
            self._tracked.pop(t_idx)

        # 6. Log activity
        if (recovered_count > 0 or phantom_count > 0) and self.logger:
            self.logger.debug(
                f"[BACKPACK PERSISTENCE] t={timestamp_sec:.1f}s "
                f"recovered={recovered_count} phantom={phantom_count} "
                f"tracked={len(self._tracked)}"
            )

    def reset(self) -> None:
        """Clear all tracked state (call at video boundaries)."""
        self._tracked.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match_current_to_tracked(
        self, current_bboxes: List[List[int]]
    ) -> tuple:
        """Greedy IoU matching of current detections to tracked backpacks.

        Returns (matched_tracked_indices, matched_current_indices) as sets.
        """
        if not self._tracked or not current_bboxes:
            return set(), set()

        matched_tracked: set = set()
        matched_current: set = set()

        # Build IoU matrix (tracked × current)
        pairs = []
        for t_idx, tracked in enumerate(self._tracked):
            for c_idx, cbbox in enumerate(current_bboxes):
                iou = calculate_iou(tracked.bbox, cbbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, t_idx, c_idx))

        # Greedy: pick highest IoU pairs first
        pairs.sort(key=lambda x: x[0], reverse=True)
        for _, t_idx, c_idx in pairs:
            if t_idx in matched_tracked or c_idx in matched_current:
                continue
            matched_tracked.add(t_idx)
            matched_current.add(c_idx)

        return matched_tracked, matched_current

    def _attempt_roi_redetection(
        self, frame: np.ndarray, tracked: _TrackedBackpack
    ) -> Optional[List[int]]:
        """Crop last-known bbox + margin, preprocess, upscale, run YOLO.

        Returns the re-detected bbox in full-frame coordinates, or None.
        """
        if self.yolo_handler is None or not hasattr(self.yolo_handler, "object_model"):
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = tracked.last_detected_bbox

        # Expand bbox by roi_margin
        bw = x2 - x1
        bh = y2 - y1
        margin_x = int(bw * self.roi_margin)
        margin_y = int(bh * self.roi_margin)

        cx1 = max(0, x1 - margin_x)
        cy1 = max(0, y1 - margin_y)
        cx2 = min(w, x2 + margin_x)
        cy2 = min(h, y2 + margin_y)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        # Preprocess: CLAHE + unsharp mask
        if self.roi_preprocess:
            crop = self._preprocess_crop(crop)

        # Upscale
        if self.roi_upscale > 1:
            crop = cv2.resize(
                crop, None,
                fx=self.roi_upscale, fy=self.roi_upscale,
                interpolation=cv2.INTER_LINEAR,
            )

        # Run YOLO on crop
        try:
            results = self.yolo_handler.object_model(
                crop,
                verbose=False,
                conf=self.roi_confidence,
                imgsz=max(crop.shape[:2]),
                device=self.yolo_handler.device,
            )
        except Exception:
            return None

        if not results or len(results[0].boxes) == 0:
            return None

        # Find best backpack/suitcase detection in crop
        best_bbox = None
        best_conf = 0.0
        model_names = self.yolo_handler.object_model.names

        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model_names.get(cls, "")
            if class_name in ("backpack", "suitcase") and conf > best_conf:
                best_conf = conf
                coords = box.xyxy[0].cpu().numpy()
                best_bbox = coords

        if best_bbox is None:
            return None

        # Map crop coordinates back to full frame
        rx1, ry1, rx2, ry2 = best_bbox
        scale = self.roi_upscale if self.roi_upscale > 1 else 1
        full_x1 = int(cx1 + rx1 / scale)
        full_y1 = int(cy1 + ry1 / scale)
        full_x2 = int(cx1 + rx2 / scale)
        full_y2 = int(cy1 + ry2 / scale)

        return [full_x1, full_y1, full_x2, full_y2]

    @staticmethod
    def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
        """Apply CLAHE + unsharp mask to an ROI crop."""
        # CLAHE on L channel
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Unsharp mask
        blurred = cv2.GaussianBlur(crop, (0, 0), 2.0)
        crop = cv2.addWeighted(crop, 1.5, blurred, -0.5, 0)

        return crop

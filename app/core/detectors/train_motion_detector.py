"""
Train Motion Detection — Vibration + Side-Window Hybrid Approach

Detects train motion by analyzing frame-to-frame pixel jitter on person-masked
cab interior surfaces (vibration signal), optical flow in the side window strip
(scenery signal), and block-wise variance changes (stability signal).

When integrated into the main pipeline, person bounding boxes are passed in
from the existing YOLO detection stage — no separate YOLO model is loaded.
"""

from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np


class TrainMotionDetector:
    """
    Detects train motion using vibration analysis on cab interior
    + side window scenery change.

    Constructor follows the existing detector pattern:
        __init__(self, settings=None, logger=None, sample_fps=0.5)
    """

    def __init__(self, settings=None, logger=None, sample_fps: float = 0.5, camera_angle: int = 1):
        self.settings = settings
        self.logger = logger
        self.sample_fps = sample_fps
        self.camera_angle = camera_angle  # 1=LP side, 2=ALP side

        # Read thresholds from settings (with defaults matching tested values)
        self.vibration_threshold = getattr(settings, 'train_motion_vibration_threshold', 1.0)
        self.vibration_high = getattr(settings, 'train_motion_vibration_high', 3.0)
        self.running_threshold = getattr(settings, 'train_motion_running_threshold', 0.45)
        self.temporal_window = getattr(settings, 'train_motion_temporal_window', 5)
        self.window_flow_threshold = getattr(settings, 'train_motion_window_flow_threshold', 2.0)
        self.weight_vibration = getattr(settings, 'train_motion_weight_vibration', 0.5)
        self.weight_window = getattr(settings, 'train_motion_weight_window', 0.3)
        self.weight_stability = getattr(settings, 'train_motion_weight_stability', 0.2)
        self.person_mask_padding = getattr(settings, 'train_motion_person_mask_padding', 0.10)
        # Percentile at which to clip the vibration diff distribution before
        # averaging. Default 90 → drop the top 10% of pixel diffs (hotspots
        # from shadow sweeps / moved objects) so they don't dominate the mean.
        self.vibration_trim_percentile = float(
            getattr(settings, 'train_motion_vibration_trim_percentile', 90.0)
        )
        # Rolling-median smoothing on vibration_mean: rejects 1–2 frame spikes
        # (person walking, bag handling) so they don't saturate the vib score
        # and anchor the state machine in RUNNING.
        self.vibration_median_window = max(1, int(
            getattr(settings, 'train_motion_vibration_median_window', 5)
        ))
        # How many prior frames of person bboxes to union into the interior
        # mask (0 = current only). At low sample FPS a walking person covers
        # many pixels per sample; one-frame history isn't enough.
        self.person_bbox_history = max(0, int(
            getattr(settings, 'train_motion_person_bbox_history', 2)
        ))
        # Cold-start guard: at the start of each chunk (per-worker fresh
        # detector) the temporal smoother has no history, so a single noisy
        # frame (person moving while seated, bag set down, door opening at a
        # station) can produce vib_mean spikes >> vibration_high and commit
        # RAW=RUNNING. While the detector has fewer than cold_start_frames
        # vibration samples in its buffer, demote RAW=RUNNING to STOPPED
        # unless the side-window optical flow is also elevated — a genuinely
        # running train shows scenery streaming, a station-stop with cab
        # activity does not.
        self.cold_start_require_window_flow = bool(
            getattr(settings, 'train_motion_cold_start_require_window_flow', True)
        )
        # Independent of vibration_median_window so the median smoother can
        # stay narrow (its default is 1, no-op) while the cold-start guard
        # still covers the first ~10s of a chunk at sample_fps=0.5.
        self.cold_start_frames = max(0, int(
            getattr(settings, 'train_motion_cold_start_frames', 5)
        ))
        # Counter of how many frames this detector has seen vibration data for
        # (i.e. after prev_gray was set on frame 0). Used by the cold-start
        # guard instead of len(_vib_history) so the guard works even when
        # vibration_median_window=1 (no-op smoother) collapses the buffer to
        # a single entry after frame 1.
        self._frames_seen = 0
        self.stability_block_size = 16
        self.confidence_threshold = 0.6

        # Select window ROI based on camera angle
        # LP Side (camera_angle=1): window strip on LEFT edge
        # ALP Side (camera_angle=2): window strip on RIGHT edge
        if camera_angle == 2:
            window_roi_str = getattr(settings, 'train_motion_window_roi_alp', '0.88,0.05,1.0,0.85')
        else:
            window_roi_str = getattr(settings, 'train_motion_window_roi_lp', '0.0,0.05,0.12,0.85')
        parts = [float(x.strip()) for x in window_roi_str.split(',')]
        self.window_roi_x1 = parts[0] if len(parts) > 0 else 0.0
        self.window_roi_y1 = parts[1] if len(parts) > 1 else 0.05
        self.window_roi_x2 = parts[2] if len(parts) > 2 else 0.12
        self.window_roi_y2 = parts[3] if len(parts) > 3 else 0.85

        # Extra scenery ROIs: semicolon-separated list of "x1,y1,x2,y2" (normalized).
        # Masked out of interior so bright doorways / secondary windows don't
        # contribute to vibration, but NOT used for the window-flow analysis.
        extra_str = getattr(settings, 'train_motion_extra_mask_rois', '') or ''
        self.extra_mask_rois: List[Tuple[float, float, float, float]] = []
        for roi in extra_str.split(';'):
            roi = roi.strip()
            if not roi:
                continue
            try:
                x1, y1, x2, y2 = [float(v.strip()) for v in roi.split(',')]
                self.extra_mask_rois.append((x1, y1, x2, y2))
            except Exception:
                if logger:
                    logger.warning(f"[TrainMotion] ignoring malformed extra ROI: {roi!r}")

        # State
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_gray_window: Optional[np.ndarray] = None
        self.state_history: deque = deque(maxlen=self.temporal_window)
        self._prev_block_vars: Optional[list] = None
        # Rolling history of prior-frame person bboxes. Each entry is a list of
        # bboxes for one frame. All entries are unioned into the interior mask
        # so the person's full multi-frame trail gets excluded from vibration.
        self.person_bbox_history_buf: deque = deque(maxlen=self.person_bbox_history)
        # Rolling buffer of recent trimmed vibration_mean values, used to compute
        # a median-smoothed vibration_mean before scoring.
        self._vib_history: deque = deque(maxlen=self.vibration_median_window)

    def create_interior_mask(
        self, frame_shape: Tuple[int, int], person_bboxes: List
    ) -> np.ndarray:
        """
        Create mask for STATIC INTERIOR only:
        255 = static interior (keep for vibration analysis)
        0   = person or window (ignore)
        """
        h, w = frame_shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255

        # Mask out persons (with padding)
        pad = self.person_mask_padding
        for bbox in person_bboxes:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            bw, bh = x2 - x1, y2 - y1
            px, py = int(bw * pad), int(bh * pad)
            x1m = max(0, x1 - px)
            y1m = max(0, y1 - py)
            x2m = min(w, x2 + px)
            y2m = min(h, y2 + py)
            mask[y1m:y2m, x1m:x2m] = 0

        # Mask out the side window strip (scenery area — analyzed separately)
        wx1 = int(self.window_roi_x1 * w)
        wy1 = int(self.window_roi_y1 * h)
        wx2 = int(self.window_roi_x2 * w)
        wy2 = int(self.window_roi_y2 * h)
        mask[wy1:wy2, wx1:wx2] = 0

        # Mask out any additional scenery regions (e.g. right-side doorway on
        # LP-camera). These contribute pixel motion but aren't tracked for flow.
        for (ex1, ey1, ex2, ey2) in self.extra_mask_rois:
            ax1 = int(max(0.0, min(1.0, ex1)) * w)
            ay1 = int(max(0.0, min(1.0, ey1)) * h)
            ax2 = int(max(0.0, min(1.0, ex2)) * w)
            ay2 = int(max(0.0, min(1.0, ey2)) * h)
            if ax2 > ax1 and ay2 > ay1:
                mask[ay1:ay2, ax1:ax2] = 0

        # Also mask top 5% (timestamp overlay) and bottom 8%
        mask[:int(h * 0.05), :] = 0
        mask[int(h * 0.92):, :] = 0

        return mask

    def _get_window_roi(self, gray: np.ndarray) -> np.ndarray:
        """Extract the side window ROI from a grayscale frame."""
        h, w = gray.shape[:2]
        x1 = int(self.window_roi_x1 * w)
        y1 = int(self.window_roi_y1 * h)
        x2 = int(self.window_roi_x2 * w)
        y2 = int(self.window_roi_y2 * h)
        return gray[y1:y2, x1:x2]

    def compute_vibration(
        self, gray: np.ndarray, interior_mask: np.ndarray
    ) -> dict:
        """
        Compute frame-to-frame vibration on static interior surfaces.
        """
        result = {
            "vibration_mean": 0.0,
            "vibration_mean_raw": 0.0,
            "vibration_std": 0.0,
            "vibration_max": 0.0,
            "vibration_p95": 0.0,
            "interior_ratio": 0.0,
            "vibration_score": 0.0,
        }

        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return result

        # Absolute frame difference on interior only
        diff = cv2.absdiff(gray, self.prev_gray).astype(np.float32)

        # Get interior pixels only
        interior_pixels = diff[interior_mask == 255]
        total_pixels = gray.shape[0] * gray.shape[1]
        interior_count = len(interior_pixels)
        result["interior_ratio"] = interior_count / total_pixels if total_pixels > 0 else 0

        if interior_count < 100:
            self.prev_gray = gray.copy()
            return result

        # Trim the top (100 - trim_percentile)% of diffs before averaging.
        # Real camera vibration spreads small diffs over most interior pixels,
        # so trimming the tail barely changes the mean. Hotspots (shadow sweeps,
        # papers moved by the person, objects placed on the console) concentrate
        # large diffs in a few hundred pixels — those dominate a raw mean but
        # get clipped here.
        trim_cut = float(np.percentile(interior_pixels, self.vibration_trim_percentile))
        trimmed = interior_pixels[interior_pixels <= trim_cut]
        trimmed_mean = float(np.mean(trimmed)) if trimmed.size > 0 else float(np.mean(interior_pixels))

        # Temporal median smoothing: push the spatial-trimmed mean into a
        # rolling buffer and take the median. A 1–2 frame spike (person walks
        # across the cabin, bag lifted) surrounded by quieter frames gets
        # suppressed; sustained vibration (real train running) still rises.
        self._vib_history.append(trimmed_mean)
        smoothed_mean = float(np.median(self._vib_history))

        result["vibration_mean"] = smoothed_mean
        result["vibration_mean_raw"] = trimmed_mean
        result["vibration_std"] = float(np.std(interior_pixels))
        result["vibration_max"] = float(np.max(interior_pixels))
        result["vibration_p95"] = float(np.percentile(interior_pixels, 95))

        if result["vibration_mean"] >= self.vibration_high:
            result["vibration_score"] = 1.0
        elif result["vibration_mean"] >= self.vibration_threshold:
            result["vibration_score"] = (
                (result["vibration_mean"] - self.vibration_threshold)
                / (self.vibration_high - self.vibration_threshold)
            )
        else:
            result["vibration_score"] = 0.0

        self.prev_gray = gray.copy()
        return result

    def compute_window_flow(self, gray: np.ndarray) -> dict:
        """
        Compute optical flow or frame diff in the side window ROI.
        """
        result = {
            "window_mean_diff": 0.0,
            "window_flow_mag": 0.0,
            "window_score": 0.0,
        }

        window_gray = self._get_window_roi(gray)

        if self.prev_gray_window is None:
            self.prev_gray_window = window_gray.copy()
            return result

        # Frame difference in window
        diff = cv2.absdiff(window_gray, self.prev_gray_window).astype(np.float32)
        result["window_mean_diff"] = float(np.mean(diff))

        # Optical flow in window ROI
        if window_gray.shape[0] > 10 and window_gray.shape[1] > 10:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray_window, window_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            result["window_flow_mag"] = float(np.mean(mag))

        # Score
        threshold = self.window_flow_threshold
        if result["window_flow_mag"] >= threshold * 2:
            result["window_score"] = 1.0
        elif result["window_flow_mag"] >= threshold:
            result["window_score"] = (result["window_flow_mag"] - threshold) / threshold
        else:
            result["window_score"] = 0.0

        self.prev_gray_window = window_gray.copy()
        return result

    def compute_stability(
        self, gray: np.ndarray, interior_mask: np.ndarray
    ) -> dict:
        """
        Compute local variance / block-wise stability of the interior.
        High local variance change -> vibration -> moving.

        Vectorized: reshape the frame into non-overlapping (bs x bs) blocks
        via a single numpy view + ``.var(axis=(2,3))`` instead of a Python
        double-loop with per-block np.var. ~50-100x faster on 1080p frames.
        """
        result = {
            "block_variance_mean": 0.0,
            "stability_score": 0.0,
        }

        bs = self.stability_block_size
        h, w = gray.shape

        # Block grid dimensions (floor to bs-aligned size, matching the
        # loop's `range(0, h - bs, bs)` semantics).
        num_y = max(0, (h - bs) // bs)
        num_x = max(0, (w - bs) // bs)
        if num_y == 0 or num_x == 0:
            self._prev_block_vars = np.empty(0, dtype=np.float32)
            return result

        crop_h = num_y * bs
        crop_w = num_x * bs
        gray_f = gray[:crop_h, :crop_w].astype(np.float32)
        # shape -> (num_y, bs, num_x, bs) -> (num_y, num_x, bs, bs)
        blocks = gray_f.reshape(num_y, bs, num_x, bs).transpose(0, 2, 1, 3)
        block_vars = blocks.var(axis=(2, 3))  # (num_y, num_x)

        mask_blocks = interior_mask[:crop_h, :crop_w].reshape(num_y, bs, num_x, bs).transpose(0, 2, 1, 3)
        # mean per block of the 0/255 mask; >128 == "mostly interior"
        interior_block_mask = mask_blocks.mean(axis=(2, 3)) > 128  # (num_y, num_x)
        curr_vars = block_vars[interior_block_mask].ravel()

        if self._prev_block_vars is None:
            self._prev_block_vars = curr_vars
            return result

        prev_vars = self._prev_block_vars
        if curr_vars.size > 0 and prev_vars.size > 0:
            min_len = min(curr_vars.size, prev_vars.size)
            diffs = np.abs(curr_vars[:min_len] - prev_vars[:min_len])
            result["block_variance_mean"] = float(diffs.mean())

            if result["block_variance_mean"] > 1200:
                result["stability_score"] = 1.0
            elif result["block_variance_mean"] > 800:
                result["stability_score"] = (result["block_variance_mean"] - 800) / 400
            else:
                result["stability_score"] = 0.0

        self._prev_block_vars = curr_vars
        return result

    def get_smoothed_state(self, raw_state: str, raw_confidence: float) -> Tuple[str, float]:
        """Temporal smoothing over recent frames."""
        self.state_history.append((raw_state, raw_confidence))
        if len(self.state_history) < 2:
            return raw_state, raw_confidence

        running_count = sum(1 for s, _ in self.state_history if s == "RUNNING")
        stopped_count = sum(1 for s, _ in self.state_history if s == "STOPPED")
        total = len(self.state_history)

        if running_count / total >= self.confidence_threshold:
            avg_conf = float(np.mean([c for s, c in self.state_history if s == "RUNNING"]))
            return "RUNNING", avg_conf
        elif stopped_count / total >= self.confidence_threshold:
            avg_conf = float(np.mean([c for s, c in self.state_history if s == "STOPPED"]))
            return "STOPPED", avg_conf
        else:
            return "UNCERTAIN", 0.5

    def process_frame(
        self, frame: np.ndarray, person_bboxes: List
    ) -> Tuple[str, float, dict]:
        """
        Full pipeline for one frame.

        Args:
            frame: BGR frame from video
            person_bboxes: list of [x1, y1, x2, y2] bounding boxes
                           (from pipeline's YOLO detection, already available)

        Returns:
            (smoothed_state, smoothed_confidence, diagnostics)
        """
        # Grayscale + light blur to reduce compression noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Create interior mask (excludes persons + window + overlays).
        # Union current + last N frames' person bboxes so motion trails from
        # the person moving across the cabin don't leak into the vibration diff.
        mask_bboxes = list(person_bboxes)
        for prev_bboxes in self.person_bbox_history_buf:
            mask_bboxes.extend(prev_bboxes)
        interior_mask = self.create_interior_mask(frame.shape, mask_bboxes)

        # Vibration analysis on static interior
        vib = self.compute_vibration(gray, interior_mask)

        # Side window flow analysis
        win = self.compute_window_flow(gray)

        # Block stability analysis
        stab = self.compute_stability(gray, interior_mask)

        # Combined score
        combined_score = (
            self.weight_vibration * vib["vibration_score"]
            + self.weight_window * win["window_score"]
            + self.weight_stability * stab["stability_score"]
        )

        # Decision
        cold_start = (
            self.cold_start_require_window_flow
            and self._frames_seen < self.cold_start_frames
        )
        self._frames_seen += 1
        if combined_score >= self.running_threshold:
            if cold_start and win["window_score"] <= 0.0:
                # Vibration alone during cold start can be person motion, not
                # the train. Demote to STOPPED so the gate stays effective at
                # chunk boundaries; the smoother will recover RUNNING quickly
                # once real scenery flow appears.
                raw_state = "STOPPED"
                confidence = 0.5
            else:
                raw_state = "RUNNING"
                confidence = min(1.0, combined_score / self.running_threshold)
        else:
            raw_state = "STOPPED"
            confidence = min(1.0, (self.running_threshold - combined_score) / self.running_threshold)

        # First frame has no previous -> unknown
        if self.prev_gray is None and vib["vibration_mean"] == 0:
            raw_state = "UNKNOWN"
            confidence = 0.0

        # Temporal smoothing
        smoothed_state, smoothed_conf = self.get_smoothed_state(raw_state, confidence)

        diagnostics = {
            "num_persons": len(person_bboxes),
            "interior_ratio": vib["interior_ratio"],
            "vib_mean": vib["vibration_mean"],
            "vib_mean_raw": vib["vibration_mean_raw"],
            "vib_std": vib["vibration_std"],
            "vib_p95": vib["vibration_p95"],
            "vib_score": vib["vibration_score"],
            "win_diff": win["window_mean_diff"],
            "win_flow": win["window_flow_mag"],
            "win_score": win["window_score"],
            "blk_var": stab["block_variance_mean"],
            "stab_score": stab["stability_score"],
            "combined_score": combined_score,
            "raw_state": raw_state,
            "raw_confidence": confidence,
            "smoothed_state": smoothed_state,
            "smoothed_confidence": smoothed_conf,
            "cold_start": cold_start,
        }

        if self.person_bbox_history > 0:
            self.person_bbox_history_buf.append(list(person_bboxes))

        return smoothed_state, smoothed_conf, diagnostics

#!/usr/bin/env python3
"""
Train Motion Detection — Vibration + Side-Window Hybrid Approach
================================================================

Problem: Cab CCTV camera points DOWN at the interior. Outside scenery is
only visible through a tiny strip on the left edge. Pure optical flow on
"background scenery" fails because there's almost no scenery visible.

Solution: Detect motion via TWO complementary signals:

  Signal 1 — VIBRATION (primary):
    Moving train → mechanical vibration → frame-to-frame pixel jitter
    on STATIC interior surfaces (green walls, instruments, seat backs).
    Mask out persons (they move intentionally) and the narrow window strip.
    Compute mean absolute frame difference on the static interior.
    Moving → higher jitter. Stopped → very stable.

  Signal 2 — SIDE WINDOW SCENERY (secondary):
    The thin strip on the left edge shows outside. Compute optical flow
    or frame diff specifically in this narrow ROI. If scenery is changing
    with consistent direction → moving.

  Signal 3 — FRAME STABILITY (tertiary):
    Compute structural similarity (SSIM) between consecutive frames on
    the static interior. Moving → lower SSIM. Stopped → high SSIM.

Usage:
  python test_train_motion.py --video /path/to/video.mp4 [options]
"""

import argparse
import csv
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class MotionConfig:
    """Tunable parameters."""

    # Sampling
    sample_fps: float = 0.5

    # YOLO person masking
    yolo_weights: str = "yolo11l.pt"
    yolo_confidence: float = 0.25
    person_mask_padding: float = 0.10  # 10% bbox expansion

    # Side window ROI (normalized coords — left edge strip)
    window_roi_x1: float = 0.0
    window_roi_y1: float = 0.05
    window_roi_x2: float = 0.12  # narrow left strip
    window_roi_y2: float = 0.85

    # Vibration detection (absolute thresholds — no calibration needed)
    # Stopped baseline: 0.1-0.3 (from stationary video)
    # Running smooth:   2.0-4.0
    # Running rough:    5.0-28.0
    vibration_threshold: float = 1.0   # Mean abs diff above this = vibrating
    vibration_high: float = 3.0        # Definite motion

    # Side window flow
    window_flow_threshold: float = 2.0  # Mean flow magnitude in window ROI

    # SSIM-like stability metric
    stability_block_size: int = 16  # Block size for local variance

    # Combined decision weights
    weight_vibration: float = 0.5
    weight_window: float = 0.3
    weight_stability: float = 0.2

    # Score threshold for RUNNING
    running_threshold: float = 0.45

    # Temporal smoothing
    temporal_window: int = 5
    confidence_threshold: float = 0.6

    # Output
    save_annotated_video: bool = True
    save_csv: bool = True
    output_dir: str = "motion_test_output"


class TrainMotionDetector:
    """
    Detects train motion using vibration analysis on cab interior
    + side window scenery change.
    """

    def __init__(self, config: MotionConfig):
        self.config = config
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_gray_window: Optional[np.ndarray] = None
        self.state_history: deque = deque(maxlen=config.temporal_window)

        # Baseline calibration (first N frames)
        self.calibration_diffs: list = []
        self.baseline_diff: float = 0.0
        self.calibrated: bool = False
        self.calibration_frames: int = 5

        # Load YOLO
        from ultralytics import YOLO
        self.yolo = YOLO(config.yolo_weights)
        print(f"[YOLO] Loaded {config.yolo_weights}")

    def detect_persons(self, frame: np.ndarray) -> List[np.ndarray]:
        """Run YOLO and return person bounding boxes [x1,y1,x2,y2]."""
        results = self.yolo(frame, conf=self.config.yolo_confidence, classes=[0], verbose=False)
        bboxes = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    bboxes.append(xyxy)
        return bboxes

    def create_interior_mask(
        self, frame_shape: Tuple[int, int], person_bboxes: List[np.ndarray]
    ) -> np.ndarray:
        """
        Create mask for STATIC INTERIOR only:
        255 = static interior (keep for vibration analysis)
        0   = person or window (ignore)
        """
        h, w = frame_shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255

        # Mask out persons (with padding)
        pad = self.config.person_mask_padding
        for bbox in person_bboxes:
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            px, py = int(bw * pad), int(bh * pad)
            x1m = max(0, x1 - px)
            y1m = max(0, y1 - py)
            x2m = min(w, x2 + px)
            y2m = min(h, y2 + py)
            mask[y1m:y2m, x1m:x2m] = 0

        # Mask out the side window strip (scenery area — analyzed separately)
        wx1 = int(self.config.window_roi_x1 * w)
        wy1 = int(self.config.window_roi_y1 * h)
        wx2 = int(self.config.window_roi_x2 * w)
        wy2 = int(self.config.window_roi_y2 * h)
        mask[wy1:wy2, wx1:wx2] = 0

        # Also mask top 5% (timestamp overlay) and bottom 5%
        mask[:int(h * 0.05), :] = 0
        mask[int(h * 0.92):, :] = 0

        return mask

    def get_window_roi(self, frame: np.ndarray) -> np.ndarray:
        """Extract the side window ROI."""
        h, w = frame.shape[:2]
        x1 = int(self.config.window_roi_x1 * w)
        y1 = int(self.config.window_roi_y1 * h)
        x2 = int(self.config.window_roi_x2 * w)
        y2 = int(self.config.window_roi_y2 * h)
        return frame[y1:y2, x1:x2]

    def compute_vibration(
        self, gray: np.ndarray, interior_mask: np.ndarray
    ) -> dict:
        """
        Compute frame-to-frame vibration on static interior surfaces.
        Returns vibration metrics.
        """
        result = {
            "vibration_mean": 0.0,
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

        result["vibration_mean"] = float(np.mean(interior_pixels))
        result["vibration_std"] = float(np.std(interior_pixels))
        result["vibration_max"] = float(np.max(interior_pixels))
        result["vibration_p95"] = float(np.percentile(interior_pixels, 95))

        # Use absolute thresholds (no calibration — avoids corruption
        # when first frames are already running)
        effective_threshold = self.config.vibration_threshold

        if result["vibration_mean"] >= self.config.vibration_high:
            result["vibration_score"] = 1.0
        elif result["vibration_mean"] >= effective_threshold:
            result["vibration_score"] = (
                (result["vibration_mean"] - effective_threshold)
                / (self.config.vibration_high - effective_threshold)
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

        window_gray = self.get_window_roi(gray)

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
        threshold = self.config.window_flow_threshold
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
        High local variance change → vibration → moving.
        """
        result = {
            "block_variance_mean": 0.0,
            "stability_score": 0.0,
        }

        bs = self.config.stability_block_size
        h, w = gray.shape

        if not hasattr(self, '_prev_block_vars') or self._prev_block_vars is None:
            # Compute block variances for current frame
            block_vars = []
            for y in range(0, h - bs, bs):
                for x in range(0, w - bs, bs):
                    block_mask = interior_mask[y:y+bs, x:x+bs]
                    if np.mean(block_mask) > 128:  # mostly interior
                        block = gray[y:y+bs, x:x+bs].astype(np.float32)
                        block_vars.append(np.var(block))
            self._prev_block_vars = block_vars
            return result

        # Current block variances
        curr_vars = []
        for y in range(0, h - bs, bs):
            for x in range(0, w - bs, bs):
                block_mask = interior_mask[y:y+bs, x:x+bs]
                if np.mean(block_mask) > 128:
                    block = gray[y:y+bs, x:x+bs].astype(np.float32)
                    curr_vars.append(np.var(block))

        if len(curr_vars) > 0 and len(self._prev_block_vars) > 0:
            min_len = min(len(curr_vars), len(self._prev_block_vars))
            diffs = [abs(curr_vars[i] - self._prev_block_vars[i]) for i in range(min_len)]
            result["block_variance_mean"] = float(np.mean(diffs))

            # Score: higher variance change = more vibration
            # Typical values: 500-1300. Need higher thresholds to discriminate.
            if result["block_variance_mean"] > 1200:
                result["stability_score"] = 1.0
            elif result["block_variance_mean"] > 800:
                result["stability_score"] = (result["block_variance_mean"] - 800) / 400
            else:
                result["stability_score"] = 0.0

        self._prev_block_vars = curr_vars
        return result

    def get_smoothed_state(self, raw_state: str, raw_confidence: float) -> Tuple[str, float]:
        """Temporal smoothing."""
        self.state_history.append((raw_state, raw_confidence))
        if len(self.state_history) < 2:
            return raw_state, raw_confidence

        running_count = sum(1 for s, _ in self.state_history if s == "RUNNING")
        stopped_count = sum(1 for s, _ in self.state_history if s == "STOPPED")
        total = len(self.state_history)

        if running_count / total >= self.config.confidence_threshold:
            avg_conf = np.mean([c for s, c in self.state_history if s == "RUNNING"])
            return "RUNNING", float(avg_conf)
        elif stopped_count / total >= self.config.confidence_threshold:
            avg_conf = np.mean([c for s, c in self.state_history if s == "STOPPED"])
            return "STOPPED", float(avg_conf)
        else:
            return "UNCERTAIN", 0.5

    def process_frame(self, frame: np.ndarray) -> Tuple[str, float, dict]:
        """Full pipeline for one frame."""

        # Step 1: Detect persons
        person_bboxes = self.detect_persons(frame)

        # Step 2: Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Light blur to reduce compression noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Step 3: Create interior mask (excludes persons + window + overlays)
        interior_mask = self.create_interior_mask(frame.shape, person_bboxes)

        # Step 4: Vibration analysis on static interior
        vib = self.compute_vibration(gray, interior_mask)

        # Step 5: Side window flow analysis
        win = self.compute_window_flow(gray)

        # Step 6: Block stability analysis
        stab = self.compute_stability(gray, interior_mask)

        # Step 7: Combined score
        combined_score = (
            self.config.weight_vibration * vib["vibration_score"]
            + self.config.weight_window * win["window_score"]
            + self.config.weight_stability * stab["stability_score"]
        )

        # Decision
        if combined_score >= self.config.running_threshold:
            raw_state = "RUNNING"
            confidence = min(1.0, combined_score / self.config.running_threshold)
        else:
            raw_state = "STOPPED"
            confidence = min(1.0, (self.config.running_threshold - combined_score) / self.config.running_threshold)

        # First frame has no previous → unknown
        if self.prev_gray is None and vib["vibration_mean"] == 0:
            raw_state = "UNKNOWN"
            confidence = 0.0

        # Temporal smoothing
        smoothed_state, smoothed_conf = self.get_smoothed_state(raw_state, confidence)

        diagnostics = {
            "num_persons": len(person_bboxes),
            "person_bboxes": person_bboxes,
            "interior_ratio": vib["interior_ratio"],
            # Vibration
            "vib_mean": vib["vibration_mean"],
            "vib_std": vib["vibration_std"],
            "vib_p95": vib["vibration_p95"],
            "vib_score": vib["vibration_score"],
            # Window
            "win_diff": win["window_mean_diff"],
            "win_flow": win["window_flow_mag"],
            "win_score": win["window_score"],
            # Stability
            "blk_var": stab["block_variance_mean"],
            "stab_score": stab["stability_score"],
            # Combined
            "combined_score": combined_score,
            "raw_state": raw_state,
            "raw_confidence": confidence,
            "smoothed_state": smoothed_state,
            "smoothed_confidence": smoothed_conf,
        }

        return smoothed_state, smoothed_conf, diagnostics


def draw_annotated_frame(
    frame: np.ndarray,
    state: str,
    confidence: float,
    diag: dict,
    timestamp_sec: float,
    frame_idx: int,
    config: MotionConfig,
) -> np.ndarray:
    """Draw diagnostics overlay on frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Draw person bboxes
    for bbox in diag.get("person_bboxes", []):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 200), 2)

    # Draw window ROI (cyan)
    wx1 = int(config.window_roi_x1 * w)
    wy1 = int(config.window_roi_y1 * h)
    wx2 = int(config.window_roi_x2 * w)
    wy2 = int(config.window_roi_y2 * h)
    cv2.rectangle(out, (wx1, wy1), (wx2, wy2), (255, 255, 0), 2)
    cv2.putText(out, "WINDOW", (wx1 + 2, wy1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # State banner
    if state == "RUNNING":
        color = (0, 200, 0)
        icon = ">>> RUNNING"
    elif state == "STOPPED":
        color = (0, 0, 220)
        icon = "[ ] STOPPED"
    else:
        color = (0, 165, 255)
        icon = "??? UNCERTAIN"

    # Background bar at bottom
    cv2.rectangle(out, (0, h - 75), (w, h), (30, 30, 30), -1)

    # State
    cv2.putText(out, f"{icon}  conf={confidence:.2f}", (10, h - 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Metrics line 1
    line1 = (
        f"t={timestamp_sec:.0f}s  f={frame_idx}  "
        f"VIB={diag['vib_mean']:.2f}(s={diag['vib_score']:.2f})  "
        f"WIN={diag['win_flow']:.2f}(s={diag['win_score']:.2f})  "
        f"BLK={diag['blk_var']:.1f}(s={diag['stab_score']:.2f})"
    )
    cv2.putText(out, line1, (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Metrics line 2
    line2 = (
        f"combined={diag['combined_score']:.3f}  "
        f"raw={diag['raw_state']}  persons={diag['num_persons']}  "
        f"interior={diag['interior_ratio']:.0%}"
    )
    cv2.putText(out, line2, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Score bar (visual gauge)
    bar_x = w - 220
    bar_w = 200
    bar_h = 16
    bar_y = h - 68
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill_w = int(bar_w * min(1.0, diag["combined_score"]))
    fill_color = (0, 200, 0) if diag["combined_score"] >= config.running_threshold else (0, 0, 200)
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)
    # Threshold marker
    thresh_x = bar_x + int(bar_w * config.running_threshold)
    cv2.line(out, (thresh_x, bar_y - 2), (thresh_x, bar_y + bar_h + 2), (255, 255, 255), 2)
    cv2.putText(out, "SCORE", (bar_x, bar_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    return out


def run_test(video_path: str, config: MotionConfig, max_frames: int = 0):
    """Process video and output results."""
    os.makedirs(config.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    step = max(1, int(round(fps / config.sample_fps)))
    sampled_count = total_frames // step

    print(f"[VIDEO] {video_path}")
    print(f"  {width}x{height}, {fps:.2f} FPS, {duration:.0f}s ({duration/60:.1f}min)")
    print(f"  Sampling at {config.sample_fps} FPS → step={step}, ~{sampled_count} samples")
    print()

    detector = TrainMotionDetector(config)

    # Video writer
    video_writer = None
    if config.save_annotated_video:
        out_path = os.path.join(config.output_dir, "motion_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(out_path, fourcc, 2.0, (width, height))
        print(f"[OUTPUT] Video → {out_path}")

    # CSV
    csv_file = None
    csv_writer = None
    csv_path = os.path.join(config.output_dir, "motion_results.csv")
    if config.save_csv:
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "sample", "frame", "time_sec",
            "raw_state", "smoothed_state", "confidence",
            "vib_mean", "vib_std", "vib_p95", "vib_score",
            "win_flow", "win_score",
            "blk_var", "stab_score",
            "combined_score", "interior_ratio", "persons",
        ])
        print(f"[OUTPUT] CSV  → {csv_path}")

    print()
    hdr = (f"{'#':>5}  {'Frame':>6}  {'Time':>7}  {'P':>2}  "
           f"{'Vib':>5}  {'VS':>4}  {'Win':>5}  {'WS':>4}  "
           f"{'Blk':>6}  {'BS':>4}  {'Comb':>5}  "
           f"{'Raw':>9}  {'Smooth':>9}  {'Conf':>5}")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    sample_idx = 0
    frame_idx = 0
    t_start = time.time()
    counts = {"RUNNING": 0, "STOPPED": 0, "UNKNOWN": 0, "UNCERTAIN": 0}

    while True:
        if max_frames > 0 and sample_idx >= max_frames:
            break
        if frame_idx >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_idx / fps if fps > 0 else 0
        state, confidence, diag = detector.process_frame(frame)
        counts[state] = counts.get(state, 0) + 1

        # Color codes
        def cc(s):
            colors = {"RUNNING": "\033[92m", "STOPPED": "\033[91m"}
            return f"{colors.get(s, chr(27)+'[93m')}{s:>9}\033[0m"

        print(
            f"{sample_idx:>5}  {frame_idx:>6}  {timestamp_sec:>6.0f}s  "
            f"{diag['num_persons']:>2}  "
            f"{diag['vib_mean']:>5.2f}  {diag['vib_score']:>4.2f}  "
            f"{diag['win_flow']:>5.2f}  {diag['win_score']:>4.2f}  "
            f"{diag['blk_var']:>6.1f}  {diag['stab_score']:>4.2f}  "
            f"{diag['combined_score']:>5.3f}  "
            f"{cc(diag['raw_state'])}  {cc(state)}  {confidence:>5.2f}"
        )

        if video_writer is not None:
            annotated = draw_annotated_frame(
                frame, state, confidence, diag, timestamp_sec, frame_idx, config
            )
            video_writer.write(annotated)

        if csv_writer is not None:
            csv_writer.writerow([
                sample_idx, frame_idx, f"{timestamp_sec:.2f}",
                diag["raw_state"], state, f"{confidence:.3f}",
                f"{diag['vib_mean']:.3f}", f"{diag['vib_std']:.3f}",
                f"{diag['vib_p95']:.3f}", f"{diag['vib_score']:.3f}",
                f"{diag['win_flow']:.3f}", f"{diag['win_score']:.3f}",
                f"{diag['blk_var']:.2f}", f"{diag['stab_score']:.3f}",
                f"{diag['combined_score']:.4f}", f"{diag['interior_ratio']:.3f}",
                diag["num_persons"],
            ])

        sample_idx += 1
        frame_idx += step

    cap.release()
    if video_writer:
        video_writer.release()
    if csv_file:
        csv_file.close()

    elapsed = time.time() - t_start
    total = sum(counts.values())

    print("=" * len(hdr))
    print()
    print(f"[DONE] {sample_idx} samples in {elapsed:.1f}s ({sample_idx/elapsed:.1f} samp/sec)")
    print()
    for s in ["RUNNING", "STOPPED", "UNCERTAIN", "UNKNOWN"]:
        c = counts.get(s, 0)
        if c > 0:
            print(f"  {s:>10}: {c:>5} ({c/total*100:.1f}%)")
    print()
    if config.save_annotated_video:
        print(f"  Video: {os.path.join(config.output_dir, 'motion_annotated.mp4')}")
    if config.save_csv:
        print(f"  CSV:   {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Motion Detection (Vibration+Window)")
    parser.add_argument("--video", required=True)
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--yolo-weights", default="yolo11l.pt")
    parser.add_argument("--vib-threshold", type=float, default=1.0)
    parser.add_argument("--vib-high", type=float, default=3.0)
    parser.add_argument("--win-threshold", type=float, default=2.0)
    parser.add_argument("--running-threshold", type=float, default=0.45)
    parser.add_argument("--temporal-window", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output-dir", default="motion_test_output")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-csv", action="store_true")

    # Window ROI
    parser.add_argument("--window-x1", type=float, default=0.0)
    parser.add_argument("--window-y1", type=float, default=0.05)
    parser.add_argument("--window-x2", type=float, default=0.12)
    parser.add_argument("--window-y2", type=float, default=0.85)

    args = parser.parse_args()

    config = MotionConfig(
        sample_fps=args.sample_fps,
        yolo_weights=args.yolo_weights,
        vibration_threshold=args.vib_threshold,
        vibration_high=args.vib_high,
        window_flow_threshold=args.win_threshold,
        running_threshold=args.running_threshold,
        temporal_window=args.temporal_window,
        output_dir=args.output_dir,
        save_annotated_video=not args.no_video,
        save_csv=not args.no_csv,
        window_roi_x1=args.window_x1,
        window_roi_y1=args.window_y1,
        window_roi_x2=args.window_x2,
        window_roi_y2=args.window_y2,
    )

    run_test(args.video, config, args.max_frames)


if __name__ == "__main__":
    main()

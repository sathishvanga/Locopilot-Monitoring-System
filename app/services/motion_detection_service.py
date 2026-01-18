"""
Motion Detection Service (Stage 0)

This service provides a fast motion-based filtering layer that runs BEFORE the ML detection
pipeline. By detecting static frames early, we can skip expensive model inference on frames
where nothing is happening, significantly improving processing speed.

Algorithm: Frame Differencing with Adaptive Thresholding
- Grayscale conversion + Gaussian blur (noise reduction)
- Frame-to-frame differencing with cv2.absdiff()
- Binary thresholding to detect change regions
- Contour analysis for significant motion regions
- Adaptive threshold calibration from first N frames

Performance Benefits:
- Frame differencing is ~24x faster than ML-based motion detection
- Typical skip rate: 20-50% of frames (depends on video content)
- Processing overhead: ~0.5ms per frame for motion check
"""

import cv2
import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Setup module-level logger
logger = logging.getLogger(__name__)


@dataclass
class MotionMetrics:
    """Stores motion detection results for a frame"""

    motion_score: float = 0.0  # Percentage of frame with motion (0.0 - 1.0)
    motion_area: int = 0  # Total pixels with detected motion
    motion_percentage: float = 0.0  # Percentage of frame area with motion
    contour_count: int = 0  # Number of significant motion regions
    has_significant_motion: bool = True  # Whether frame should be processed
    processing_time_ms: float = 0.0  # Time taken for motion detection


@dataclass
class MotionDetectionStats:
    """Tracks motion detection statistics across the processing session"""

    frames_analyzed: int = 0
    frames_skipped: int = 0
    frames_processed: int = 0
    total_processing_time_ms: float = 0.0
    consecutive_skips: int = 0
    max_consecutive_skips: int = 0

    @property
    def skip_rate(self) -> float:
        """Calculate the skip rate as a percentage"""
        if self.frames_analyzed == 0:
            return 0.0
        return (self.frames_skipped / self.frames_analyzed) * 100

    @property
    def avg_processing_time_ms(self) -> float:
        """Calculate average processing time per frame"""
        if self.frames_analyzed == 0:
            return 0.0
        return self.total_processing_time_ms / self.frames_analyzed


class MotionDetectionService:
    """
    Stage 0 motion detection to skip static frames before ML inference.

    Uses frame differencing with adaptive thresholding to detect motion.
    Static frames are skipped to reduce GPU/model inference calls.

    Configuration:
        motion_threshold: Minimum percentage of frame that must change (default: 0.01 = 1%)
        min_contour_area: Minimum pixel area for a motion region (default: 500)
        blur_kernel_size: Gaussian blur kernel size (default: 5)
        binary_threshold: Threshold for diff image binarization (default: 25)
        adaptive_calibration: Enable adaptive threshold calibration (default: True)
        max_consecutive_skips: Safety limit for consecutive skipped frames (default: 5)
    """

    def __init__(
        self,
        motion_threshold: float = 0.01,
        min_contour_area: int = 500,
        blur_kernel_size: int = 5,
        binary_threshold: int = 25,
        adaptive_calibration: bool = True,
        calibration_frames: int = 10,
        max_consecutive_skips: int = 5
    ):
        """
        Initialize the motion detection service.

        Args:
            motion_threshold: Minimum percentage of frame that must change (0.0-1.0)
            min_contour_area: Minimum pixel area for a motion region to be counted
            blur_kernel_size: Size of Gaussian blur kernel (must be odd)
            binary_threshold: Threshold for converting diff image to binary (0-255)
            adaptive_calibration: Whether to calibrate threshold from first N frames
            calibration_frames: Number of frames to use for calibration
            max_consecutive_skips: Maximum consecutive frames to skip (safety valve)
        """
        self.motion_threshold = motion_threshold
        self.min_contour_area = min_contour_area
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.binary_threshold = binary_threshold
        self.adaptive_calibration = adaptive_calibration
        self.calibration_frames = calibration_frames
        self.max_consecutive_skips = max_consecutive_skips

        # State for frame differencing
        self._prev_frame_gray: Optional[np.ndarray] = None
        self._frame_count: int = 0

        # Adaptive calibration state
        self._calibration_scores: list = []
        self._baseline_motion: float = 0.0
        self._is_calibrated: bool = False

        # Statistics tracking
        self.stats = MotionDetectionStats()

        logger.info(
            f"MotionDetectionService initialized: threshold={motion_threshold:.2%}, "
            f"min_contour={min_contour_area}, blur_kernel={self.blur_kernel_size}, "
            f"binary_threshold={binary_threshold}, adaptive={adaptive_calibration}, "
            f"max_consecutive_skips={max_consecutive_skips}"
        )

    def reset(self):
        """Reset the service state for a new video"""
        self._prev_frame_gray = None
        self._frame_count = 0
        self._calibration_scores = []
        self._baseline_motion = 0.0
        self._is_calibrated = False
        self.stats = MotionDetectionStats()
        logger.debug("MotionDetectionService reset for new video")

    def detect_motion(self, frame: np.ndarray) -> MotionMetrics:
        """
        Detect motion in the current frame compared to the previous frame.

        Args:
            frame: BGR frame from video capture

        Returns:
            MotionMetrics with motion analysis results
        """
        start_time = time.perf_counter()

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur for noise reduction
        gray = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)

        # First frame - always process
        if self._prev_frame_gray is None:
            self._prev_frame_gray = gray
            self._frame_count = 1

            processing_time = (time.perf_counter() - start_time) * 1000
            return MotionMetrics(
                motion_score=1.0,
                motion_area=0,
                motion_percentage=0.0,
                contour_count=0,
                has_significant_motion=True,
                processing_time_ms=processing_time
            )

        # Calculate absolute difference
        diff = cv2.absdiff(self._prev_frame_gray, gray)

        # Apply binary threshold
        _, thresh = cv2.threshold(diff, self.binary_threshold, 255, cv2.THRESH_BINARY)

        # Find contours to count significant motion regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by minimum area
        significant_contours = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]

        # Calculate motion metrics
        motion_area = sum(cv2.contourArea(c) for c in significant_contours)
        frame_area = frame.shape[0] * frame.shape[1]
        motion_percentage = motion_area / frame_area if frame_area > 0 else 0.0

        # Update previous frame
        self._prev_frame_gray = gray
        self._frame_count += 1

        # Adaptive calibration during first N frames
        if self.adaptive_calibration and not self._is_calibrated:
            self._calibration_scores.append(motion_percentage)
            if len(self._calibration_scores) >= self.calibration_frames:
                self._calibrate_baseline()

        # Determine if motion is significant
        effective_threshold = self._get_effective_threshold()
        has_significant_motion = (
            motion_percentage >= effective_threshold or
            len(significant_contours) >= 3  # Multiple motion regions indicate activity
        )

        processing_time = (time.perf_counter() - start_time) * 1000

        return MotionMetrics(
            motion_score=motion_percentage / effective_threshold if effective_threshold > 0 else 1.0,
            motion_area=motion_area,
            motion_percentage=motion_percentage,
            contour_count=len(significant_contours),
            has_significant_motion=has_significant_motion,
            processing_time_ms=processing_time
        )

    def should_skip_frame(self, frame: np.ndarray) -> Tuple[bool, MotionMetrics]:
        """
        Determine if a frame should be skipped based on motion detection.

        This is the main entry point for Stage 0 filtering. It handles:
        - Motion detection
        - Statistics tracking
        - Safety valve (max consecutive skips)

        Args:
            frame: BGR frame from video capture

        Returns:
            Tuple of (should_skip: bool, metrics: MotionMetrics)
        """
        metrics = self.detect_motion(frame)

        # Update statistics
        self.stats.frames_analyzed += 1
        self.stats.total_processing_time_ms += metrics.processing_time_ms

        # Check if we should skip
        should_skip = not metrics.has_significant_motion

        # Safety valve: don't skip more than max_consecutive_skips frames in a row
        if should_skip:
            self.stats.consecutive_skips += 1
            if self.stats.consecutive_skips >= self.max_consecutive_skips:
                # Force processing this frame
                should_skip = False
                logger.debug(
                    f"Safety valve triggered: forcing frame processing after "
                    f"{self.stats.consecutive_skips} consecutive skips"
                )

        if should_skip:
            self.stats.frames_skipped += 1
            self.stats.max_consecutive_skips = max(
                self.stats.max_consecutive_skips,
                self.stats.consecutive_skips
            )
        else:
            self.stats.frames_processed += 1
            self.stats.consecutive_skips = 0

        return should_skip, metrics

    def _calibrate_baseline(self):
        """Calibrate the motion baseline from collected samples"""
        if not self._calibration_scores:
            return

        # Use median + 1 standard deviation as baseline
        # This accounts for camera shake and minor lighting fluctuations
        median_motion = np.median(self._calibration_scores)
        std_motion = np.std(self._calibration_scores)

        # Baseline is typical noise + buffer
        self._baseline_motion = median_motion + std_motion
        self._is_calibrated = True

        logger.info(
            f"Motion baseline calibrated: median={median_motion:.4f}, "
            f"std={std_motion:.4f}, baseline={self._baseline_motion:.4f}"
        )

    def _get_effective_threshold(self) -> float:
        """Get the effective motion threshold (considering calibration)"""
        if self.adaptive_calibration and self._is_calibrated:
            # Use the larger of configured threshold or calibrated baseline
            return max(self.motion_threshold, self._baseline_motion)
        return self.motion_threshold

    def get_statistics_summary(self) -> dict:
        """Get a summary of motion detection statistics"""
        return {
            'frames_analyzed': self.stats.frames_analyzed,
            'frames_skipped': self.stats.frames_skipped,
            'frames_processed': self.stats.frames_processed,
            'skip_rate_percent': round(self.stats.skip_rate, 2),
            'avg_processing_time_ms': round(self.stats.avg_processing_time_ms, 3),
            'total_processing_time_ms': round(self.stats.total_processing_time_ms, 2),
            'max_consecutive_skips': self.stats.max_consecutive_skips,
            'is_calibrated': self._is_calibrated,
            'baseline_motion': round(self._baseline_motion, 4) if self._is_calibrated else None,
            'effective_threshold': round(self._get_effective_threshold(), 4)
        }

    def log_statistics(self, log_level: int = logging.INFO):
        """Log motion detection statistics"""
        stats = self.get_statistics_summary()
        logger.log(
            log_level,
            f"Motion Detection Stats: analyzed={stats['frames_analyzed']}, "
            f"skipped={stats['frames_skipped']} ({stats['skip_rate_percent']:.1f}%), "
            f"avg_time={stats['avg_processing_time_ms']:.2f}ms, "
            f"max_consecutive_skips={stats['max_consecutive_skips']}"
        )

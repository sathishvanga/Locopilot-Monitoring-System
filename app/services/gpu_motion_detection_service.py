"""
GPU-Accelerated Motion Detection Service (Stage 0)

This service provides GPU-accelerated motion detection using PyTorch CUDA,
replacing CPU-based OpenCV operations for significantly faster processing.

GPU Operations:
- Frame resize: torch.nn.functional.interpolate
- Grayscale conversion: torch tensor operations
- Gaussian blur: torch convolution
- Threshold operations: torch tensor operations
- Background subtraction: GPU tensor operations

Performance Benefits:
- Offloads motion detection from CPU to GPU
- Frees CPU for video decoding and other tasks
- Can process frames 3-5x faster on GPU
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

# Setup module-level logger
logger = logging.getLogger(__name__)


@dataclass
class GPUMotionMetrics:
    """Stores motion detection results for a frame"""
    motion_score: float = 0.0
    motion_area: int = 0
    motion_percentage: float = 0.0
    has_significant_motion: bool = True
    processing_time_ms: float = 0.0
    scene_change_detected: bool = False
    is_subsampled_skip: bool = False


@dataclass
class GPUMotionDetectionStats:
    """Tracks motion detection statistics"""
    frames_analyzed: int = 0
    frames_skipped: int = 0
    frames_processed: int = 0
    total_processing_time_ms: float = 0.0
    consecutive_skips: int = 0
    max_consecutive_skips: int = 0
    scene_change_skips: int = 0
    subsampled_skips: int = 0

    @property
    def skip_rate(self) -> float:
        if self.frames_analyzed == 0:
            return 0.0
        return (self.frames_skipped / self.frames_analyzed) * 100

    @property
    def avg_processing_time_ms(self) -> float:
        if self.frames_analyzed == 0:
            return 0.0
        return self.total_processing_time_ms / self.frames_analyzed


class GPUMotionDetectionService:
    """
    GPU-accelerated motion detection using PyTorch CUDA.

    This replaces CPU-based OpenCV operations with GPU tensor operations
    for faster motion detection.
    """

    def __init__(
        self,
        motion_threshold: float = 0.015,
        min_contour_area: int = 1000,
        blur_kernel_size: int = 5,
        binary_threshold: int = 30,
        max_consecutive_skips: int = 20,
        detection_scale: float = 0.35,
        roi_margin: float = 0.08,
        scene_change_threshold: float = 0.15,
        continuous_motion_subsample: int = 2,
        min_continuous_motion_frames: int = 5,
        device: str = "cuda"
    ):
        """
        Initialize GPU motion detection service.

        Args:
            motion_threshold: Minimum percentage of frame that must change (0.0-1.0)
            min_contour_area: Minimum pixel area for motion region
            blur_kernel_size: Size of Gaussian blur kernel (must be odd)
            binary_threshold: Threshold for diff image binarization (0-255)
            max_consecutive_skips: Maximum consecutive frames to skip
            detection_scale: Scale factor for frame downscaling
            roi_margin: Border margin to ignore motion in
            scene_change_threshold: Brightness change threshold
            continuous_motion_subsample: Subsample rate during sustained motion
            min_continuous_motion_frames: Frames before subsampling kicks in
            device: CUDA device to use ("cuda", "cuda:0", etc.)
        """
        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            logger.info(f"GPU Motion Detection using: {torch.cuda.get_device_name(0)}")

        self.motion_threshold = motion_threshold
        self.min_contour_area = min_contour_area
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.binary_threshold = binary_threshold / 255.0  # Normalize to 0-1
        self.max_consecutive_skips = max_consecutive_skips
        self.detection_scale = detection_scale
        self.roi_margin = roi_margin
        self.scene_change_threshold = scene_change_threshold
        self.continuous_motion_subsample = continuous_motion_subsample
        self.min_continuous_motion_frames = min_continuous_motion_frames

        # Create Gaussian blur kernel on GPU
        self.blur_kernel = self._create_gaussian_kernel(self.blur_kernel_size).to(self.device)

        # State tracking
        self._prev_frame: Optional[torch.Tensor] = None
        self._frame_count: int = 0
        self.continuous_motion_frames: int = 0

        # Running average for background model (simple exponential moving average)
        self._background: Optional[torch.Tensor] = None
        self._alpha = 0.01  # Learning rate for background model

        # Statistics tracking
        self.stats = GPUMotionDetectionStats()

        logger.info(
            f"GPUMotionDetectionService initialized: threshold={motion_threshold:.2%}, "
            f"scale={detection_scale}, device={self.device}"
        )

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float = None) -> torch.Tensor:
        """Create a Gaussian blur kernel."""
        if sigma is None:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

        coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        # Create 2D kernel
        kernel = g.unsqueeze(0) * g.unsqueeze(1)
        kernel = kernel / kernel.sum()

        # Shape for conv2d: (out_channels, in_channels, H, W)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        return kernel

    def _to_gpu_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """Convert numpy frame to GPU tensor."""
        # Convert BGR to grayscale using standard weights
        if len(frame.shape) == 3:
            # BGR to grayscale: 0.114*B + 0.587*G + 0.299*R
            frame_float = frame.astype(np.float32) / 255.0
            gray = 0.114 * frame_float[:, :, 0] + 0.587 * frame_float[:, :, 1] + 0.299 * frame_float[:, :, 2]
        else:
            gray = frame.astype(np.float32) / 255.0

        # Convert to tensor and move to GPU
        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor

    def _resize_tensor(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        """Resize tensor using GPU interpolation."""
        if scale == 1.0:
            return tensor
        return F.interpolate(tensor, scale_factor=scale, mode='bilinear', align_corners=False)

    def _apply_gaussian_blur(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur using GPU convolution."""
        # Pad to maintain size
        pad = self.blur_kernel_size // 2
        padded = F.pad(tensor, (pad, pad, pad, pad), mode='reflect')
        blurred = F.conv2d(padded, self.blur_kernel)
        return blurred

    def _apply_roi_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply ROI mask to ignore border regions."""
        if self.roi_margin <= 0:
            return tensor

        _, _, h, w = tensor.shape
        margin_h = int(h * self.roi_margin)
        margin_w = int(w * self.roi_margin)

        # Clone to avoid modifying in place
        masked = tensor.clone()
        masked[:, :, :margin_h, :] = 0  # Top
        masked[:, :, -margin_h:, :] = 0  # Bottom
        masked[:, :, :, :margin_w] = 0  # Left
        masked[:, :, :, -margin_w:] = 0  # Right

        return masked

    def detect_motion(self, frame: np.ndarray) -> GPUMotionMetrics:
        """
        Detect motion in the current frame using GPU acceleration.

        Args:
            frame: Input frame (numpy array, BGR format)

        Returns:
            GPUMotionMetrics with motion detection results
        """
        start_time = time.perf_counter()

        # Convert to GPU tensor
        curr_frame = self._to_gpu_tensor(frame)

        # Resize for faster processing
        curr_small = self._resize_tensor(curr_frame, self.detection_scale)

        # Apply Gaussian blur
        curr_blurred = self._apply_gaussian_blur(curr_small)

        metrics = GPUMotionMetrics()

        # First frame - initialize background
        if self._prev_frame is None or self._background is None:
            self._prev_frame = curr_blurred.clone()
            self._background = curr_blurred.clone()
            self._frame_count = 1
            metrics.processing_time_ms = (time.perf_counter() - start_time) * 1000
            return metrics

        # Scene change detection
        prev_mean = self._prev_frame.mean().item()
        curr_mean = curr_blurred.mean().item()
        if prev_mean > 0:
            brightness_change = abs(curr_mean - prev_mean) / prev_mean
            if brightness_change > self.scene_change_threshold:
                metrics.scene_change_detected = True
                # Update background with new scene
                self._background = curr_blurred.clone()
                self._prev_frame = curr_blurred.clone()
                metrics.processing_time_ms = (time.perf_counter() - start_time) * 1000
                return metrics

        # Compute motion using background subtraction
        diff = torch.abs(curr_blurred - self._background)

        # Apply binary threshold
        motion_mask = (diff > self.binary_threshold).float()

        # Apply ROI mask
        motion_mask = self._apply_roi_mask(motion_mask)

        # Calculate motion metrics
        total_pixels = motion_mask.numel()
        motion_pixels = motion_mask.sum().item()
        motion_percentage = motion_pixels / total_pixels

        # Update background model (exponential moving average)
        self._background = self._alpha * curr_blurred + (1 - self._alpha) * self._background

        # Store for next frame
        self._prev_frame = curr_blurred.clone()
        self._frame_count += 1

        # Calculate scaled motion area (back to original resolution)
        scale_factor = 1.0 / (self.detection_scale ** 2)
        motion_area = int(motion_pixels * scale_factor)

        metrics.motion_score = motion_percentage
        metrics.motion_area = motion_area
        metrics.motion_percentage = motion_percentage
        metrics.has_significant_motion = motion_percentage > self.motion_threshold

        # Continuous motion subsampling
        if metrics.has_significant_motion:
            self.continuous_motion_frames += 1
            if self.continuous_motion_frames >= self.min_continuous_motion_frames:
                if self._frame_count % self.continuous_motion_subsample != 0:
                    metrics.has_significant_motion = False
                    metrics.is_subsampled_skip = True
        else:
            self.continuous_motion_frames = 0

        metrics.processing_time_ms = (time.perf_counter() - start_time) * 1000
        return metrics

    def should_skip_frame(self, frame: np.ndarray) -> Tuple[bool, GPUMotionMetrics]:
        """
        Determine if the frame should be skipped (no significant motion).

        Args:
            frame: Input frame (numpy array)

        Returns:
            Tuple of (should_skip: bool, metrics: GPUMotionMetrics)
        """
        metrics = self.detect_motion(frame)

        # Update statistics
        self.stats.frames_analyzed += 1
        self.stats.total_processing_time_ms += metrics.processing_time_ms

        should_skip = False

        # Skip if scene change detected
        if metrics.scene_change_detected:
            should_skip = True
            self.stats.scene_change_skips += 1

        # Skip if no significant motion
        elif not metrics.has_significant_motion:
            if metrics.is_subsampled_skip:
                should_skip = True
                self.stats.subsampled_skips += 1
            else:
                should_skip = True

        # Safety valve: don't skip too many consecutive frames
        if should_skip:
            self.stats.consecutive_skips += 1
            if self.stats.consecutive_skips >= self.max_consecutive_skips:
                should_skip = False
                self.stats.consecutive_skips = 0
                logger.debug(f"Safety valve triggered after {self.max_consecutive_skips} consecutive skips")
        else:
            self.stats.consecutive_skips = 0

        if should_skip:
            self.stats.frames_skipped += 1
            self.stats.max_consecutive_skips = max(
                self.stats.max_consecutive_skips, self.stats.consecutive_skips
            )
        else:
            self.stats.frames_processed += 1

        return should_skip, metrics

    def reset(self):
        """Reset the service state for a new video."""
        self._prev_frame = None
        self._background = None
        self._frame_count = 0
        self.continuous_motion_frames = 0
        self.stats = GPUMotionDetectionStats()
        logger.debug("GPUMotionDetectionService reset for new video")

    def get_stats_summary(self) -> str:
        """Get a summary of motion detection statistics."""
        return (
            f"analyzed={self.stats.frames_analyzed}, "
            f"skipped={self.stats.frames_skipped} ({self.stats.skip_rate:.1f}%), "
            f"processed={self.stats.frames_processed}, "
            f"avg_time={self.stats.avg_processing_time_ms:.2f}ms"
        )

    def log_statistics(self):
        """Log motion detection statistics (called during cleanup)."""
        logger.info(f"GPU Motion Detection Summary: {self.get_stats_summary()}")

"""
Multiprocessing utilities for video processing

This module implements a shared process pool architecture for parallel video processing
with worker initialization, work partitioning, and progress tracking.
"""

import os
import json
import cv2
import time
import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from dataclasses import dataclass, asdict
import multiprocessing as mp
import gc
import atexit

from .multiprocessing_config import MultiprocessingConfig
from .logger import get_logger
from .config import get_settings
from .json_utils import NumpyEncoder


logger = get_logger(__name__)

# NOTE: Settings are accessed lazily via get_settings() within functions
# to prevent caching before environment variables are fully loaded.
# Do not use a module-level settings variable.

# NumpyEncoder is imported from app.utils.json_utils (the single canonical
# encoder). Earlier this module shipped its own copy that was missing
# np.bool_ in the repository copy; consolidating fixes that drift.


# Global variables for worker processes (initialized once per worker)
_worker_models = None

# Shared global process pool (mimics POC_2 app.boot.get_pool)
_shared_pool: Optional[ProcessPoolExecutor] = None


@dataclass
class FrameRange:
    """Represents a range of frames to process.

    When overlap is enabled (C-02 fix), each chunk except the first extends
    backwards by ``overlap_seconds`` worth of frames so that the worker's
    temporal state (consecutive_detections, sleep state, baseline calibration)
    can warm up before the canonical region begins.

    Attributes:
        start_frame: First frame of the *canonical* region (activities here are kept).
        end_frame: Last frame (exclusive) of the canonical region.
        start_time: Timestamp (seconds) corresponding to start_frame.
        end_time: Timestamp (seconds) corresponding to end_frame.
        expected_sampled_frames: Estimated sampled frames in the canonical region.
        range_id: Monotonically increasing chunk identifier.
        overlap_start_frame: First frame the worker actually processes (may be
            earlier than start_frame for state warm-up).  For the first chunk
            this equals start_frame.
        overlap_start_time: Timestamp (seconds) corresponding to overlap_start_frame.
    """
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    expected_sampled_frames: int
    range_id: int
    overlap_start_frame: int = 0
    overlap_start_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ProcessingState:
    """Tracks processing progress and state"""
    total_expected_frames: int
    processed_frames: int
    completed_ranges: List[int]
    failed_ranges: List[int]
    start_time: float
    last_update_time: float
    done: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingState':
        """Create from dictionary"""
        return cls(**data)

    def get_progress_percentage(self) -> float:
        """Get progress as percentage"""
        if self.total_expected_frames == 0:
            return 0.0
        return (self.processed_frames / self.total_expected_frames) * 100.0


def worker_initializer(config: MultiprocessingConfig):
    """
    Initialize worker process with models and configurations

    This function runs once per worker process at startup to:
    - Prevent Qt/GUI initialization (Windows compatibility)
    - Set thread counts for Torch and OpenCV
    - Disable OpenCV OpenCL
    - Preload heavy models (YOLO26, YOLO26-Pose, MediaPipe FaceMesh, Preprocessing Service)
    - Set environment variables

    Args:
        config: Multiprocessing configuration
    """
    global _worker_models

    try:
        # WINDOWS FIX: Prevent Qt/GUI windows from opening in worker processes
        # Set this FIRST before any imports that might trigger Qt initialization
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        os.environ['DISPLAY'] = ''  # For compatibility on Linux/Unix systems

        # Set environment variables
        config.set_worker_env_vars()

        # Set PyTorch thread count
        torch.set_num_threads(config.torch_threads)

        # Disable OpenCV OpenCL
        if config.disable_opencv_opencl:
            cv2.ocl.setUseOpenCL(False)

        # Set OpenCV thread count
        cv2.setNumThreads(config.opencv_threads)

        logger.info(f"Worker {os.getpid()} initialized: torch_threads={config.torch_threads}, "
                   f"opencv_threads={config.opencv_threads}, opencl_disabled={config.disable_opencv_opencl}")

        # Preload models if enabled (ONCE per worker - significant performance improvement)
        if config.preload_models:
            from ultralytics import YOLO
            import mediapipe as mp
            from app.services.yolo_pose_adapter import YoloPoseAdapter

            # 1. Load YOLO object detection model
            logger.info(f"Worker {os.getpid()} loading YOLO model: {config.yolo_model_path}")
            yolo_model = YOLO(config.yolo_model_path)

            # Move model to GPU if configured
            if config.yolo_device and config.yolo_device != 'cpu':
                # Convert numeric string to int (e.g., "0" -> 0 for GPU device)
                device = int(config.yolo_device) if config.yolo_device.isdigit() else config.yolo_device
                yolo_model.to(device)
                logger.info(f"Worker {os.getpid()} YOLO moved to device: {device}")

            # Phase 3.5 Quick Win A: Fuse Conv+BatchNorm layers for faster inference (15-20% speedup)
            if hasattr(yolo_model.model, 'fuse'):
                yolo_model.fuse()
                logger.info(f"Worker {os.getpid()} YOLO model layers fused for optimized inference")

            # 2. Load YOLO26-Pose model for body pose estimation
            yolo_pose_model_path = config.yolo_pose_model_path
            logger.info(f"Worker {os.getpid()} loading YOLO26-Pose model: {yolo_pose_model_path}")
            yolo_pose_raw = YOLO(yolo_pose_model_path)

            # Move pose model to GPU if configured
            if config.yolo_device and config.yolo_device != 'cpu':
                # Convert numeric string to int (e.g., "0" -> 0 for GPU device)
                device = int(config.yolo_device) if config.yolo_device.isdigit() else config.yolo_device
                yolo_pose_raw.to(device)
                logger.info(f"Worker {os.getpid()} YOLO-Pose moved to device: {device}")

            # Fuse pose model layers as well
            if hasattr(yolo_pose_raw.model, 'fuse'):
                yolo_pose_raw.fuse()
                logger.info(f"Worker {os.getpid()} YOLO-Pose model layers fused")
            yolo_pose = YoloPoseAdapter(
                model_path=yolo_pose_model_path,
                conf_threshold=0.45,
                preloaded_model=yolo_pose_raw
            )

            # 3. Initialize MediaPipe FaceMesh (for Eye Aspect Ratio detection)
            logger.info(f"Worker {os.getpid()} initializing MediaPipe FaceMesh")
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=2,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            # 4. Initialize Image Preprocessing Service
            preprocessing_service = None
            try:
                from app.services.image_preprocessing_service import ImagePreprocessingService
                # Get settings lazily within worker process
                worker_settings = get_settings()
                preprocessing_config = {
                    'enable_image_preprocessing': worker_settings.enable_image_preprocessing,
                    'use_clahe': worker_settings.use_clahe,
                    'use_gamma_correction': worker_settings.use_gamma_correction,
                    'use_unsharp_masking': worker_settings.use_unsharp_masking,
                    'use_noise_reduction': worker_settings.use_noise_reduction,
                    'adaptive_preprocessing': worker_settings.adaptive_preprocessing,
                    'clahe_clip_limit': worker_settings.clahe_clip_limit,
                    'clahe_tile_grid_size': worker_settings.clahe_tile_grid_size,
                    'gamma_value': worker_settings.gamma_value,
                    'unsharp_strength': worker_settings.unsharp_strength,
                    'unsharp_radius': worker_settings.unsharp_radius,
                    'noise_reduction_kernel': worker_settings.noise_reduction_kernel
                }
                preprocessing_service = ImagePreprocessingService(config=preprocessing_config)
                logger.info(f"Worker {os.getpid()} preprocessing service initialized")
            except Exception as e:
                logger.warning(f"Worker {os.getpid()} failed to init preprocessing: {e}")

            _worker_models = {
                'yolo': yolo_model,
                'yolo_pose': yolo_pose,
                'face_mesh': face_mesh,
                'mp_face_mesh': mp_face_mesh,
                'preprocessing_service': preprocessing_service,
            }

            # 5a. Load separate ROI model if configured (stronger model for pose-guided crops)
            if config.yolo_roi_model_path and config.yolo_roi_model_path != config.yolo_model_path:
                logger.info(f"Worker {os.getpid()} loading ROI YOLO model: {config.yolo_roi_model_path}")
                yolo_roi_model = YOLO(config.yolo_roi_model_path)
                if config.yolo_device and config.yolo_device != 'cpu':
                    device = int(config.yolo_device) if config.yolo_device.isdigit() else config.yolo_device
                    yolo_roi_model.to(device)
                if hasattr(yolo_roi_model.model, 'fuse'):
                    yolo_roi_model.fuse()
                _worker_models['yolo_roi'] = yolo_roi_model
            else:
                _worker_models['yolo_roi'] = None

            logger.info(f"Worker {os.getpid()} all models loaded successfully "
                       f"(detection: {config.yolo_model_path})")

            # ------------------------------------------------------------------
            # Task 0007: pre-construct lightweight detector instances once per
            # worker.  None of these hold heavy model weights, but constructors
            # do some bookkeeping (Haar XML loads, dict allocations, threshold
            # reads from settings), and at one monitor rebuild per 15s chunk
            # that's ~120 rebuilds per worker on a 30-minute video.  Store the
            # instances on _worker_models so LocopilotActivityMonitor.__init__
            # can reuse them via preloaded_models.
            #
            # State-reset decision: the monitor explicitly resets detector
            # state at the start of each chunk (see
            # LocopilotActivityMonitor.__init__).  Overlap warm-up (task 0003)
            # is responsible for re-priming the baseline window.  Keeping
            # detectors persistent across chunks would bleed state from one
            # worker invocation into the next, which is undesirable while
            # other per-chunk state (activities dict, consecutive counters)
            # is still reallocated inside the monitor.
            # ------------------------------------------------------------------
            try:
                from app.core.detectors import (
                    SleepDetector,
                    GestureDetector,
                    MindDiversionDetector,
                    ActivityDetector,
                )
                from app.services.yolo_pose_adapter import get_keypoint_by_name

                worker_settings_for_detectors = get_settings()
                sample_fps_cfg = worker_settings_for_detectors.sample_fps
                coordination_window_cfg = (
                    worker_settings_for_detectors.hand_gesture_coordination_window
                )

                _worker_models['sleep_detector'] = SleepDetector(
                    settings=worker_settings_for_detectors,
                    sample_fps=sample_fps_cfg,
                    logger=logger,
                )
                _worker_models['gesture_detector'] = GestureDetector(
                    settings=worker_settings_for_detectors,
                    session_timeout=10.0,
                    coordination_window=coordination_window_cfg,
                    get_keypoint_func=get_keypoint_by_name,
                )
                _worker_models['mind_diversion_detector'] = MindDiversionDetector(
                    settings=worker_settings_for_detectors,
                    logger=logger,
                )
                _worker_models['activity_detector'] = ActivityDetector(
                    settings=worker_settings_for_detectors,
                    get_keypoint_func=get_keypoint_by_name,
                )

                logger.info(
                    f"Worker {os.getpid()} pre-loaded detectors: "
                    "SleepDetector, GestureDetector, MindDiversionDetector, "
                    "ActivityDetector"
                )
            except Exception as detector_err:
                # Detectors are an optimization: if they fail to construct,
                # fall back to per-chunk construction rather than failing the
                # whole worker.  Log prominently so regressions are visible.
                logger.warning(
                    f"Worker {os.getpid()} failed to preload detectors "
                    f"(will fall back to per-chunk construction): {detector_err}",
                    exc_info=True,
                )

    except Exception as e:
        logger.error(f"Worker {os.getpid()} initialization failed: {e}", exc_info=True)
        raise


def get_shared_pool(config: MultiprocessingConfig) -> ProcessPoolExecutor:
    """
    Lazily create and return a shared ProcessPoolExecutor.

    Mirrors POC_2's app.boot.get_pool behavior:
    - Uses spawn context for stability on macOS/Linux.
    - Uses a worker initializer to set Torch/OpenCV threads and preload models.
    """
    global _shared_pool
    if _shared_pool is not None:
        return _shared_pool

    # WINDOWS FIX: Set Qt offscreen mode BEFORE creating process pool
    # This prevents worker processes from creating GUI windows on Windows
    # Must be set in main process before spawning workers
    import sys
    if sys.platform == 'win32':
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        os.environ['DISPLAY'] = ''

    num_workers = config.get_num_workers()

    # Log GPU worker cap when active (C-04 OOM prevention)
    if config.yolo_device and config.yolo_device != 'cpu':
        logger.info(
            f"GPU device '{config.yolo_device}' active: workers capped to "
            f"{config.gpu_max_workers} (GPU_MAX_WORKERS) to prevent OOM "
            f"(each worker loads YOLO + YOLO-Pose onto GPU)"
        )

    ctx = mp.get_context(config.start_method or "spawn")

    _shared_pool = ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=worker_initializer,
        initargs=(config,),
    )
    atexit.register(shutdown_shared_pool)
    logger.info(
        f"Created shared ProcessPoolExecutor(max_workers={num_workers}, "
        f"start_method={config.start_method})"
    )
    return _shared_pool


def shutdown_shared_pool(wait: bool = True) -> None:
    """
    Shutdown the shared process pool (called on application shutdown).

    Mirrors the atexit-based shutdown in POC_2's app.boot.
    """
    global _shared_pool
    try:
        if _shared_pool is not None:
            logger.info("Shutting down shared ProcessPoolExecutor")
            _shared_pool.shutdown(wait=wait, cancel_futures=True)
            _shared_pool = None
    except Exception as e:
        logger.warning(f"Error while shutting down shared pool: {e}", exc_info=True)


def calculate_frame_ranges(
    video_path: str,
    sample_fps: float,
    chunk_duration: float,
    min_chunk_duration: float = 2.0,
    overlap_seconds: float = 12.0,
) -> Tuple[List[FrameRange], int, float]:
    """
    Calculate frame ranges for parallel processing.

    Splits video into fixed-duration chunks for balanced CPU and IO load.

    Each chunk (except the first) includes an overlap region that extends
    backward into the previous chunk's territory.  The overlap allows each
    worker's LocopilotActivityMonitor to warm up its temporal state
    (consecutive_detections, sleep state, baseline calibration) so that
    activities spanning chunk boundaries are not split or lost (C-02 fix).

    The returned FrameRange objects carry both the *canonical* region
    (start_frame / start_time) and the *overlap-inclusive* region
    (overlap_start_frame / overlap_start_time).  The worker processes frames
    from overlap_start_frame to end_frame but only keeps activities whose
    videoStartTime >= start_time.

    Args:
        video_path: Path to video file
        sample_fps: Sampling rate (frames per second)
        chunk_duration: Target duration per chunk in seconds
        min_chunk_duration: Minimum chunk duration in seconds
        overlap_seconds: Seconds of overlap from previous chunk to warm up
            temporal state at chunk boundaries (default 12.0).  The default
            is sized to cover ``sleep_baseline_calibration_window`` and
            ``hand_gesture_coordination_window`` (both ~10s) so pose-based
            sleep and coordination detection are not suppressed at chunk
            seams (ARCH-03).  Set to 0 to disable overlap.

    Returns:
        Tuple of (frame_ranges, total_frames, native_fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / native_fps
    cap.release()

    # Calculate sampling step
    step = max(1, int(round(native_fps / max(1e-6, float(sample_fps)))))

    # Calculate frame ranges based on chunk duration
    frame_ranges = []
    range_id = 0

    current_time = 0.0
    while current_time < video_duration:
        # Calculate chunk end time
        chunk_end_time = min(current_time + chunk_duration, video_duration)
        actual_chunk_duration = chunk_end_time - current_time

        # Skip chunks that are too short (except the last one)
        if actual_chunk_duration < min_chunk_duration and chunk_end_time < video_duration:
            current_time = chunk_end_time
            continue

        # C-02 overlap: extend the processing start of each chunk (except the
        # first) backward by overlap_seconds so the worker can warm up temporal
        # state before reaching frames unique to this chunk.  The canonical
        # start (start_frame / start_time) remains at current_time; only the
        # overlap_start_frame / overlap_start_time are shifted backward.
        if range_id > 0 and overlap_seconds > 0:
            overlap_start_time_val = max(0.0, current_time - overlap_seconds)
        else:
            overlap_start_time_val = current_time

        # Calculate frame indices
        start_frame = int(current_time * native_fps)
        end_frame = int(chunk_end_time * native_fps)
        overlap_start_frame_val = int(overlap_start_time_val * native_fps)

        # Calculate expected sampled frames in the canonical region only
        # (overlap frames are for warm-up and should not inflate progress)
        canonical_frames_in_range = end_frame - start_frame
        expected_samples = max(1, canonical_frames_in_range // step)

        frame_range = FrameRange(
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=current_time,
            end_time=chunk_end_time,
            expected_sampled_frames=expected_samples,
            range_id=range_id,
            overlap_start_frame=overlap_start_frame_val,
            overlap_start_time=overlap_start_time_val,
        )

        frame_ranges.append(frame_range)
        range_id += 1
        current_time = chunk_end_time

    logger.info(f"Video split into {len(frame_ranges)} chunks "
               f"(~{chunk_duration}s each, {overlap_seconds}s overlap, "
               f"{sample_fps} FPS sampling)")

    return frame_ranges, total_frames, native_fps


def _parse_video_start_seconds(time_val) -> float:
    """Parse an activity's videoStartTime to float seconds.

    Handles both float-string values (e.g. ``"12.50"``) and HH:MM:SS
    timestamp strings.  Returns 0.0 on failure.
    """
    if time_val is None:
        return 0.0
    if isinstance(time_val, (int, float)):
        return float(time_val)
    if isinstance(time_val, str):
        # Try float first (most common for videoStartTime)
        try:
            return float(time_val)
        except ValueError:
            pass
        # Fallback: HH:MM:SS
        if ':' in time_val:
            try:
                parts = time_val.split(':')
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2]) if len(parts) > 2 else 0
                return hours * 3600 + minutes * 60 + seconds
            except (ValueError, IndexError):
                pass
    return 0.0


def process_frame_range(
    video_path: str,
    frame_range: FrameRange,
    sample_fps: float,
    trip_id: str,
    crew_members: Dict[str, Dict[str, str]],
    crew_name: str,
    crew_id: str,
    crew_role: int,
    output_dir: str,
    run_dir: str = None,
    save_clips: bool = True,
    video_start_time: str = None,
    camera_angle: int = 1,
    overlap_seconds: float = 0.0
) -> Dict[str, Any]:
    """
    Process a specific frame range (worker task function).

    This function runs in a worker process and processes frames within
    the assigned range independently.  Clips and images can be saved if
    requested.

    C-02 fix: When the FrameRange has an overlap region
    (overlap_start_frame < start_frame), the worker processes frames
    starting from overlap_start_frame so that temporal state
    (consecutive_detections, sleep state, baseline calibration) is warmed
    up.  Activities whose videoStartTime falls before the canonical
    start_time are discarded to avoid duplicates.

    PERFORMANCE: Uses pre-loaded models from _worker_models (initialized
    once per worker) instead of loading models for each task.

    Args:
        video_path: Path to video file
        frame_range: Frame range to process (includes overlap metadata)
        sample_fps: Sampling rate
        trip_id: Trip identifier
        crew_name: Crew member name
        crew_id: Crew member ID
        crew_role: Crew role
        output_dir: Output directory (base directory)
        run_dir: Run directory for saving clips (if None, no clips saved)
        save_clips: Whether to save clips and images (default: True)
        video_start_time: Video recording start time in HH:MM:SS format (optional)
        camera_angle: Camera angle for LP/ALP role assignment (default: 1)
        overlap_seconds: Overlap duration in seconds (passed for logging; the
            actual overlap region is encoded in frame_range).

    Returns:
        Dictionary with detected activities and metadata
    """
    global _worker_models

    # Fail fast if worker initialization did not complete successfully.
    # When _worker_models is None the monitor would silently attempt to
    # reload all models from scratch, causing severe performance degradation
    # instead of surfacing the real init failure.
    if _worker_models is None:
        raise RuntimeError(
            f"Worker {os.getpid()} models not initialized. "
            "worker_initializer() likely failed or was not called. "
            "Cannot process frames without pre-loaded models."
        )

    try:
        from locopilot_monitor import LocopilotActivityMonitor

        worker_id = os.getpid()

        # Determine whether this chunk has an overlap warm-up region
        has_overlap = frame_range.overlap_start_frame < frame_range.start_frame
        actual_processing_start = (
            frame_range.overlap_start_frame if has_overlap else frame_range.start_frame
        )
        actual_processing_start_time = (
            frame_range.overlap_start_time if has_overlap else frame_range.start_time
        )

        logger.info(
            f"Worker {worker_id} processing range {frame_range.range_id}: "
            f"frames {actual_processing_start}-{frame_range.end_frame} "
            f"({actual_processing_start_time:.2f}s - {frame_range.end_time:.2f}s) "
            f"[canonical_start={frame_range.start_frame} ({frame_range.start_time:.2f}s), "
            f"overlap={'yes' if has_overlap else 'no'}, save_clips={save_clips}]"
        )

        # Determine whether annotated frames should be persisted
        # Clips and images are ALWAYS saved (for UI evidence)
        # Frames are only saved when save_clips=True AND save_annotated_frames config is True
        # Get settings lazily within worker process
        worker_settings = get_settings()
        save_frames = save_clips and worker_settings.save_annotated_frames

        # PERFORMANCE: Use pre-loaded models (guaranteed non-None after the check above)
        preloaded = _worker_models
        if preloaded:
            logger.debug(f"Worker {worker_id} using pre-loaded models (fast path)")

        # Always create monitor with run directory to save clips/images
        # Clips and images are essential for UI, so we always create the directory
        if run_dir:
            # Use provided run directory for saving clips and frames
            monitor = LocopilotActivityMonitor(
                video_path=video_path,
                output_dir=output_dir,
                save_annotated_frames=save_frames,  # Only save frames when explicitly requested
                frame_save_interval=worker_settings.frame_save_interval,
                sample_fps=sample_fps,
                run_dir=run_dir,  # Use shared run directory
                create_run_dir=False,  # Don't create new directory
                preloaded_models=preloaded  # Pass pre-loaded models
            )
        else:
            # No run directory provided - should not happen in normal operation
            raise ValueError("run_dir must be provided to save evidence clips and images")

        # Set trip and crew information
        monitor.trip_id = trip_id
        monitor.crew_name = crew_name
        monitor.crew_id = crew_id
        monitor.crew_role = crew_role

        # Set crew members mapping if provided
        if crew_members:
            monitor.crew_members = crew_members

        # Set camera angle for LP/ALP role assignment
        monitor.camera_angle = camera_angle

        # Set video start time for motion rules (when OCR unavailable)
        if video_start_time and hasattr(monitor, 'set_video_start_time'):
            monitor.set_video_start_time(video_start_time)

        # C-02 fix: Process from overlap_start_frame (not start_frame) so that
        # temporal state warms up over the overlap region before the canonical
        # region begins.  Activities are filtered below to discard any whose
        # videoStartTime falls within the overlap-only region.
        activities = monitor.process_video_range(
            start_frame=actual_processing_start,
            end_frame=frame_range.end_frame,
            save_clips=save_clips
        )

        # C-02 fix: Discard activities that started during the overlap warm-up
        # region.  These activities exist only to warm up temporal state; the
        # previous chunk's worker is the canonical owner of that time range.
        if has_overlap and activities:
            canonical_start_sec = frame_range.start_time
            pre_filter_count = len(activities)
            activities = [
                a for a in activities
                if _parse_video_start_seconds(a.get('videoStartTime')) >= canonical_start_sec
            ]
            discarded = pre_filter_count - len(activities)
            if discarded > 0:
                logger.info(
                    f"Worker {worker_id} range {frame_range.range_id}: "
                    f"discarded {discarded} overlap warm-up activities "
                    f"(videoStartTime < {canonical_start_sec:.2f}s)"
                )

        logger.info(f"Worker {worker_id} completed range {frame_range.range_id}: "
                   f"{len(activities)} activities detected")

        # MEMORY FIX: Explicit cleanup (closes MediaPipe, clears buffers, forces GC)
        # This mirrors POC_2's cleanup pattern
        monitor.cleanup()

        return {
            'success': True,
            'range_id': frame_range.range_id,
            'activities': activities,
            'processed_frames': frame_range.expected_sampled_frames,
            'error': None
        }

    except Exception as e:
        logger.error(f"Worker {os.getpid()} failed processing range {frame_range.range_id}: {e}",
                    exc_info=True)
        return {
            'success': False,
            'range_id': frame_range.range_id,
            'activities': [],
            'processed_frames': 0,
            'error': str(e)
        }
    finally:
        # MEMORY FIX: Always cleanup monitor resources in finally block
        # This mirrors POC_2's cleanup pattern
        if 'monitor' in locals():
            try:
                monitor.cleanup()
                del monitor
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup: {cleanup_error}")

        # MEMORY FIX: Force garbage collection
        gc.collect()


class VideoMultiprocessingOrchestrator:
    """
    Orchestrates parallel video processing with process pool.

    This class manages:
    - Shared or dedicated process pool lifecycle
    - Task submission and result collection
    - Progress tracking and persistence
    - Result aggregation
    """

    def __init__(
        self,
        config: Optional[MultiprocessingConfig] = None,
        output_dir: Optional[str] = None,
        use_shared_pool: bool = True,
    ):
        """
        Initialize orchestrator

        Args:
            config: Multiprocessing configuration (uses default if None)
            output_dir: Output directory for results and state
            use_shared_pool: Whether to reuse a global shared pool
                             (recommended: True for server workloads)
        """
        self.config = config or MultiprocessingConfig()
        self.output_dir = output_dir or "locopilot_evidence"
        self.pool: Optional[ProcessPoolExecutor] = None
        self.state: Optional[ProcessingState] = None
        self.use_shared_pool = use_shared_pool

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def initialize_pool(self):
        """Initialize process pool (shared or per-orchestrator)."""
        if self.pool is not None:
            logger.warning("Process pool already initialized")
            return

        num_workers = self.config.get_num_workers()

        # Log GPU worker cap when active (C-04 OOM prevention)
        if self.config.yolo_device and self.config.yolo_device != 'cpu':
            logger.info(
                f"GPU device '{self.config.yolo_device}' active: workers capped to "
                f"{self.config.gpu_max_workers} (GPU_MAX_WORKERS) to prevent OOM "
                f"(each worker loads YOLO + YOLO-Pose onto GPU)"
            )

        logger.info(
            f"Initializing process pool with {num_workers} workers "
            f"(method={self.config.start_method}, shared={self.use_shared_pool})"
        )

        # WINDOWS FIX: Set Qt offscreen mode BEFORE creating process pool
        # This prevents worker processes from creating GUI windows on Windows
        import sys
        if sys.platform == 'win32':
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            os.environ['DISPLAY'] = ''

        if self.use_shared_pool:
            # Reuse a global pool, like POC_2's get_pool()
            self.pool = get_shared_pool(self.config)
        else:
            # Dedicated pool per orchestrator (legacy behavior)
            ctx = mp.get_context(self.config.start_method or "spawn")
            self.pool = ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=ctx,
                initializer=worker_initializer,
                initargs=(self.config,),
            )

        logger.info("Process pool initialized successfully")

    def shutdown_pool(self, wait: bool = True):
        """
        Shutdown process pool.

        For shared-pool mode, this is a no-op; the global pool is shut down
        once at application shutdown via shutdown_shared_pool().
        """
        if self.pool is None:
            return

        if self.use_shared_pool:
            logger.debug("Skipping shutdown of shared process pool from orchestrator")
            return

        logger.info("Shutting down dedicated process pool")
        self.pool.shutdown(wait=wait)
        self.pool = None

    def save_state(self, run_dir: str):
        """Save processing state to disk"""
        if not self.config.enable_result_persistence or self.state is None:
            return

        state_path = os.path.join(run_dir, self.config.state_file_name)
        try:
            with open(state_path, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2, cls=NumpyEncoder)
            logger.debug(f"State saved to {state_path}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self, run_dir: str) -> Optional[ProcessingState]:
        """Load processing state from disk"""
        if not self.config.enable_result_persistence:
            return None

        state_path = os.path.join(run_dir, self.config.state_file_name)
        if not os.path.exists(state_path):
            return None

        try:
            with open(state_path, 'r') as f:
                data = json.load(f)
            state = ProcessingState.from_dict(data)
            logger.info(f"State loaded from {state_path}")
            return state
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None

    def process_video_parallel(
        self,
        video_path: str,
        trip_id: str,
        crew_members: Dict[str, Dict[str, str]] = None,
        crew_name: str = "John Doe",
        crew_id: str = "C-001",
        crew_role: int = 1,
        sample_fps: float = 1.0,
        run_dir: str = None,
        save_clips: bool = True,
        video_start_time: str = None,
        camera_angle: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Process video in parallel using multiple workers.

        C-02 fix: Passes overlap_seconds from config to calculate_frame_ranges
        so that each worker chunk (except the first) includes a warm-up
        region from the previous chunk.  This prevents temporal state
        discontinuity at chunk boundaries.

        Args:
            video_path: Path to video file
            trip_id: Trip identifier
            crew_name: Crew member name
            crew_id: Crew member ID
            crew_role: Crew role
            sample_fps: Sampling rate
            run_dir: Run directory for output
            save_clips: Whether to save video clips and images (default: True)
            video_start_time: Video recording start time in HH:MM:SS format (optional)
            camera_angle: Camera angle for LP/ALP role assignment (1 = LP Side, 2 = ALP Side)

        Returns:
            List of detected activities (merged from all ranges)
        """
        start_time = time.time()

        # Initialize pool if needed
        if self.pool is None:
            self.initialize_pool()

        # Calculate frame ranges with overlap for temporal state warm-up (C-02)
        overlap_seconds = self.config.overlap_seconds
        logger.info(
            f"Calculating frame ranges for parallel processing "
            f"(overlap_seconds={overlap_seconds})"
        )
        frame_ranges, total_frames, native_fps = calculate_frame_ranges(
            video_path=video_path,
            sample_fps=sample_fps,
            chunk_duration=self.config.chunk_duration_seconds,
            min_chunk_duration=self.config.min_chunk_duration_seconds,
            overlap_seconds=overlap_seconds,
        )

        # Calculate total expected sampled frames
        total_expected_frames = sum(fr.expected_sampled_frames for fr in frame_ranges)

        # Initialize processing state
        self.state = ProcessingState(
            total_expected_frames=total_expected_frames,
            processed_frames=0,
            completed_ranges=[],
            failed_ranges=[],
            start_time=start_time,
            last_update_time=start_time,
            done=False
        )

        logger.info(f"Submitting {len(frame_ranges)} tasks to process pool "
                   f"(expected {total_expected_frames} sampled frames, save_clips={save_clips})")

        # Submit tasks to pool
        futures: Dict[Future, FrameRange] = {}

        for frame_range in frame_ranges:
            future = self.pool.submit(
                process_frame_range,
                video_path=video_path,
                frame_range=frame_range,
                sample_fps=sample_fps,
                trip_id=trip_id,
                crew_members=crew_members,
                crew_name=crew_name,
                crew_id=crew_id,
                crew_role=crew_role,
                output_dir=self.output_dir,
                run_dir=run_dir,  # Pass run_dir for clip saving
                save_clips=save_clips,
                video_start_time=video_start_time,  # Pass video start time for motion rules
                camera_angle=camera_angle,
                overlap_seconds=overlap_seconds,  # C-02: pass for logging
            )
            futures[future] = frame_range

        # Collect results as they complete
        all_activities = []

        for future in as_completed(futures):
            frame_range = futures[future]
            # Per-chunk timeout: 10x the actual chunk duration
            chunk_duration = frame_range.end_time - frame_range.start_time
            chunk_timeout = chunk_duration * 10

            try:
                result = future.result(timeout=chunk_timeout)

                if result['success']:
                    # Append activities from this range
                    all_activities.extend(result['activities'])

                    # Update state
                    self.state.processed_frames += result['processed_frames']
                    self.state.completed_ranges.append(result['range_id'])

                    logger.info(f"Range {result['range_id']} completed: "
                               f"{len(result['activities'])} activities, "
                               f"{self.state.get_progress_percentage():.1f}% done")
                else:
                    # Record failure
                    self.state.failed_ranges.append(result['range_id'])
                    logger.error(f"Range {result['range_id']} failed: {result['error']}")

                # Update state timestamp and save
                self.state.last_update_time = time.time()
                if self.config.enable_result_persistence:
                    self.save_state(run_dir)

            except TimeoutError:
                logger.error(
                    f"Range {frame_range.range_id} timed out after {chunk_timeout:.1f}s "
                    f"(chunk_duration={chunk_duration:.1f}s), skipping"
                )
                future.cancel()
                self.state.failed_ranges.append(frame_range.range_id)
            except Exception as e:
                logger.error(f"Failed to get result for range {frame_range.range_id}: {e}",
                           exc_info=True)
                self.state.failed_ranges.append(frame_range.range_id)

        # Mark as done
        self.state.done = True
        self.state.last_update_time = time.time()

        if self.config.enable_result_persistence:
            self.save_state(run_dir)

        # Sort activities by start time (handles both timestamp strings and float seconds)
        def parse_activity_time(time_val):
            """Parse activity time - handles HH:MM:SS strings or float seconds"""
            if time_val is None:
                return 0.0
            if isinstance(time_val, (int, float)):
                return float(time_val)
            if isinstance(time_val, str):
                # Try parsing as HH:MM:SS
                if ':' in time_val:
                    try:
                        parts = time_val.split(':')
                        hours = float(parts[0])
                        minutes = float(parts[1])
                        seconds = float(parts[2]) if len(parts) > 2 else 0
                        return hours * 3600 + minutes * 60 + seconds
                    except (ValueError, IndexError):
                        pass
                # Try parsing as float
                try:
                    return float(time_val)
                except ValueError:
                    pass
            return 0.0

        all_activities.sort(key=lambda x: parse_activity_time(x.get('activityStartTime', 0)))

        # Group overlapping activities of different types into combined records
        from ..services.concurrent_activity_grouping_service import get_concurrent_grouping_service
        concurrent_grouping_service = get_concurrent_grouping_service()
        all_activities = concurrent_grouping_service.group_concurrent_activities(all_activities, run_dir)

        processing_time = time.time() - start_time

        logger.info(f"Parallel processing completed in {processing_time:.2f}s: "
                   f"{len(all_activities)} activities detected, "
                   f"{len(self.state.completed_ranges)} ranges completed, "
                   f"{len(self.state.failed_ranges)} ranges failed")

        # Ensure clips directory exists
        clips_dir = os.path.join(run_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)

        # Save merged activities to main run directory.
        # Route through ActivityRepository so every writer in the codebase
        # uses the same atomic + locked write protocol (Task 0002).
        if len(all_activities) > 0:
            try:
                from ..repositories.activity_repository import ActivityRepository
                activities_json_path = ActivityRepository().save_activities(
                    all_activities, run_dir
                )
                logger.info(f"Saved {len(all_activities)} activities to {activities_json_path}")
            except Exception as e:
                logger.error(f"Failed to save activities.json: {e}")

        # Log clip generation status
        if save_clips:
            # Count generated clips
            clip_count = sum(1 for a in all_activities if a.get('activityClip') and
                           os.path.exists(a.get('activityClip', '')))
            logger.info(f"Generated {clip_count} video clips and images in {clips_dir}")
        else:
            logger.info("Note: Clip generation was disabled (save_clips=False)")

        # MEMORY FIX: Force garbage collection after processing
        gc.collect()

        return all_activities

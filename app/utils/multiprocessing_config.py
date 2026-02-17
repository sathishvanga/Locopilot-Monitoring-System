"""
Multiprocessing configuration for video processing

This module provides configuration for parallel video processing with shared process pools.
"""

import os
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional

# Lazy import helper to avoid circular imports
def _get_settings_chunk_duration() -> float:
    """Get chunk duration from Settings (single source of truth)."""
    from app.utils.config import get_settings
    return get_settings().mp_chunk_duration


def _get_settings_gpu_batch_size() -> int:
    """Get gpu_batch_size from Settings (single source of truth)."""
    from app.utils.config import get_settings
    return get_settings().gpu_batch_size


def _get_settings_gpu_batch_enabled() -> bool:
    """Get gpu_batch_enabled from Settings (single source of truth)."""
    from app.utils.config import get_settings
    return get_settings().gpu_batch_enabled


@dataclass
class MultiprocessingConfig:
    """Configuration for multiprocessing video pipeline"""

    # Process pool settings
    # ✅ PRODUCTION OPTIMIZED: Environment-aware configuration
    # Balance: More workers = better parallelization but more memory usage (each worker loads models)
    max_workers: Optional[int] = None  # None = auto-detect (min(CPU count, max_workers_cap))
    # Local (12-core): 10 workers × 2 threads ≈ 20 logical threads
    # Production (16-core Dell R420): 14 workers × 4 threads ≈ 56 logical threads
    max_workers_cap: int = int(os.getenv("MP_MAX_WORKERS_CAP", "10"))  # Maximum number of worker processes
    start_method: str = "spawn"  # Use 'spawn' for cross-platform stability

    # Work partitioning settings
    # ✅ Default sourced from Settings.mp_chunk_duration (15s) to ensure
    # hand gesture coordination detection works correctly (coordination window is 10s).
    # Override via MP_CHUNK_DURATION env var or by passing chunk_duration_seconds explicitly.
    chunk_duration_seconds: Optional[float] = None  # None = use Settings.mp_chunk_duration
    min_chunk_duration_seconds: float = 2.0  # Minimum chunk duration

    # Overlap settings for temporal state warm-up at chunk boundaries (C-02 fix)
    # Each worker processes overlap_seconds of extra frames from the previous chunk
    # to warm up temporal state (consecutive_detections, sleep state, baseline calibration).
    # Activities detected during the overlap region are discarded; only the canonical
    # chunk's results are kept.  This prevents temporal state discontinuity at boundaries.
    overlap_seconds: float = float(os.getenv("MP_OVERLAP_SECONDS", "2.0"))  # Overlap duration in seconds

    # Worker initialization settings
    # ✅ PRODUCTION OPTIMIZED: Thread counts from environment or defaults
    # Production: 3 threads (Xeon processors benefit from higher threading)
    # Development: 3 threads (balanced for consumer CPUs)
    # M-11: These defaults are overridden at runtime by set_worker_env_vars()
    # which computes threads_per_worker = cores / workers to prevent over-subscription.
    torch_threads: int = int(os.getenv("TORCH_THREADS", "3"))  # Torch thread count per worker (synced at init)
    opencv_threads: int = int(os.getenv("OPENCV_THREADS", "3"))  # OpenCV thread count per worker (synced at init)
    disable_opencv_opencl: bool = True  # Disable OpenCV OpenCL
    
    # Model preloading settings - YOLO11 (latest, better accuracy, faster)
    # Configurable via environment variables: YOLO_WEIGHTS_PRELOAD, YOLO_POSE_WEIGHTS
    preload_models: bool = True  # Preload models in worker initializer
    yolo_model_path: str = os.getenv("YOLO_WEIGHTS_PRELOAD", "yolo11m.pt")  # YOLO model for object detection
    yolo_pose_model_path: str = os.getenv("YOLO_POSE_WEIGHTS", "yolo11m-pose.pt")  # YOLO-Pose for body pose estimation
    yolo_device: str = os.getenv("YOLO_DEVICE", "cpu")  # Device for YOLO inference (cpu, cuda:0, 0)

    # GPU Batch Processing Settings
    # These settings optimize GPU utilization by processing multiple frames at once
    # instead of one frame at a time, keeping the GPU busy and reducing overhead.
    # Defaults sourced from Settings (single source of truth) via __post_init__.
    gpu_batch_size: Optional[int] = None  # None = use Settings.gpu_batch_size (default 8)
    gpu_batch_enabled: Optional[bool] = None  # None = use Settings.gpu_batch_enabled (default True)
    model_cache_dir: Optional[str] = None  # Model cache directory
    
    # Progress settings
    enable_progress_tracking: bool = True  # Enable progress updates
    progress_update_interval: float = 1.0  # Progress update interval in seconds
    
    # Result persistence settings
    enable_result_persistence: bool = True  # Save intermediate results
    state_file_name: str = "processing_state.json"  # State file name
    
    # GPU worker cap: prevents OOM when multiple workers each load YOLO models onto the same GPU.
    # GPU inference is already parallelized via CUDA, so fewer workers suffice.
    gpu_max_workers: int = int(os.getenv("GPU_MAX_WORKERS", "4"))

    def __post_init__(self):
        """Resolve defaults from Settings when not explicitly provided.

        Settings is the single source of truth for all shared configuration.
        It already respects environment variables via pydantic_settings,
        so we do not need separate os.getenv fallbacks here.
        """
        if self.chunk_duration_seconds is None:
            self.chunk_duration_seconds = _get_settings_chunk_duration()
        if self.gpu_batch_size is None:
            self.gpu_batch_size = _get_settings_gpu_batch_size()
        if self.gpu_batch_enabled is None:
            self.gpu_batch_enabled = _get_settings_gpu_batch_enabled()

    def get_num_workers(self) -> int:
        """
        Calculate optimal number of worker processes

        Targets maximum CPU utilization for fastest processing.
        Formula: Uses min(CPU cores, max_workers_cap) to balance performance and memory.
        With 11 workers × 3 threads each = 33 threads total (saturates 11-core system).

        When using GPU (yolo_device != 'cpu'), workers are capped to gpu_max_workers
        (default 4) to prevent GPU OOM. Each worker loads both YOLO and YOLO-Pose
        models onto the GPU (~500MB-1GB each), so 10 workers would exhaust VRAM.

        Returns:
            int: Number of worker processes to use
        """
        if self.max_workers is not None:
            num_workers = min(self.max_workers, self.max_workers_cap)
        else:
            # Auto-detect: min(CPU cores, configured cap)
            # This ensures we don't exceed max_workers_cap while utilizing available cores
            cpu_count = mp.cpu_count()
            num_workers = min(cpu_count, self.max_workers_cap)

        # GPU OOM prevention: cap workers when using GPU device
        # GPU inference is already parallelized via CUDA, so fewer workers suffice.
        if self.yolo_device and self.yolo_device != 'cpu':
            if num_workers > self.gpu_max_workers:
                num_workers = self.gpu_max_workers

        return num_workers
    
    def set_worker_env_vars(self):
        """Set environment variables for worker processes.

        M-11 fix: Synchronize all thread counts through a single calculation
        to prevent over-subscription.  Previously, torch_threads and
        opencv_threads were independent hardcoded values (default 3 each)
        while OMP_NUM_THREADS was computed dynamically, risking
        workers * (torch + opencv + OMP) threads >> CPU cores.

        Now threads_per_worker is computed once and applied to torch,
        opencv, OMP, MKL, and all other threading libraries uniformly.
        """
        # Compute threads_per_worker based on available cores and worker count.
        # Cap at 8 to avoid diminishing returns from excessive threading.
        num_cores = mp.cpu_count()
        num_workers = self.get_num_workers()
        threads_per_worker = max(2, min(8, num_cores // max(1, num_workers)))

        # Synchronize torch and opencv thread counts with the computed value
        # so that total threads (workers * threads_per_worker) stays within
        # the CPU core budget.
        self.torch_threads = threads_per_worker
        self.opencv_threads = threads_per_worker

        os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)
        os.environ['MKL_NUM_THREADS'] = str(threads_per_worker)
        os.environ['OPENBLAS_NUM_THREADS'] = str(threads_per_worker)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(threads_per_worker)
        os.environ['NUMEXPR_NUM_THREADS'] = str(threads_per_worker)
        
        # Set OpenCV settings
        if self.disable_opencv_opencl:
            os.environ['OPENCV_OPENCL_DEVICE'] = 'disabled'
        
        # Set model cache directory if specified
        if self.model_cache_dir:
            os.environ['TORCH_HOME'] = self.model_cache_dir
            os.environ['TRANSFORMERS_CACHE'] = self.model_cache_dir


def get_default_config() -> MultiprocessingConfig:
    """
    Get default multiprocessing configuration
    
    Returns:
        MultiprocessingConfig: Default configuration instance
    """
    return MultiprocessingConfig()


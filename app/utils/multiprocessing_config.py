"""
Multiprocessing configuration for video processing

This module provides configuration for parallel video processing with shared process pools.
"""

import os
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional


@dataclass
class MultiprocessingConfig:
    """Configuration for multiprocessing video pipeline"""

    # Process pool settings
    # ✅ PRODUCTION OPTIMIZED: Environment-aware configuration
    # Balance: More workers = better parallelization but more memory usage (each worker loads models)
    max_workers: Optional[int] = None  # None = auto-detect (min(CPU count, max_workers_cap))
    # Production (12-core): 14 workers × 3 threads ≈ 42 logical threads (max throughput)
    # Development (11-core): 12 workers × 3 threads ≈ 36 logical threads
    max_workers_cap: int = int(os.getenv("MP_MAX_WORKERS_CAP", "12"))  # Maximum number of worker processes
    start_method: str = "spawn"  # Use 'spawn' for cross-platform stability

    # Work partitioning settings
    # ✅ PRODUCTION OPTIMIZED: 5s chunks for 12+ core servers, 6s for development
    # Production: 5s chunks = ~450 chunks (optimal for 14 workers)
    # Development: 6s chunks = ~380 chunks (optimal for 11-12 workers)
    chunk_duration_seconds: float = float(os.getenv("MP_CHUNK_DURATION", "6.0"))  # Configurable chunk duration
    min_chunk_duration_seconds: float = 2.0  # Minimum chunk duration

    # Worker initialization settings
    # ✅ PRODUCTION OPTIMIZED: Thread counts from environment or defaults
    # Production: 3 threads (Xeon processors benefit from higher threading)
    # Development: 3 threads (balanced for consumer CPUs)
    torch_threads: int = int(os.getenv("TORCH_THREADS", "3"))  # Torch thread count per worker
    opencv_threads: int = int(os.getenv("OPENCV_THREADS", "3"))  # OpenCV thread count per worker
    disable_opencv_opencl: bool = True  # Disable OpenCV OpenCL
    
    # Model preloading settings
    preload_models: bool = True  # Preload models in worker initializer
    yolo_model_path: str = "yolov8m.pt"  # Path to YOLO model weights
    model_cache_dir: Optional[str] = None  # Model cache directory
    
    # Progress settings
    enable_progress_tracking: bool = True  # Enable progress updates
    progress_update_interval: float = 1.0  # Progress update interval in seconds
    
    # Result persistence settings
    enable_result_persistence: bool = True  # Save intermediate results
    state_file_name: str = "processing_state.json"  # State file name
    
    def get_num_workers(self) -> int:
        """
        Calculate optimal number of worker processes

        Targets maximum CPU utilization for fastest processing.
        Formula: Uses min(CPU cores, max_workers_cap) to balance performance and memory.
        With 11 workers × 3 threads each = 33 threads total (saturates 11-core system).

        Returns:
            int: Number of worker processes to use
        """
        if self.max_workers is not None:
            return min(self.max_workers, self.max_workers_cap)
        
        # Auto-detect: min(CPU cores, configured cap)
        # This ensures we don't exceed max_workers_cap while utilizing available cores
        cpu_count = mp.cpu_count()
        num_workers = min(cpu_count, self.max_workers_cap)
        
        # For systems with many cores, we still cap at max_workers_cap
        # The combination of workers × threads per worker provides good CPU utilization
        return num_workers
    
    def set_worker_env_vars(self):
        """Set environment variables for worker processes"""
        # Set thread counts
        os.environ['OMP_NUM_THREADS'] = str(self.torch_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.torch_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.torch_threads)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(self.torch_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.torch_threads)
        
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


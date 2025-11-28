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
    # ✅ PERFORMANCE FIX: Increased max_workers_cap from 2 to 6 for better CPU utilization (~60% target)
    # Balance: More workers = better parallelization but more memory usage (each worker loads models)
    max_workers: Optional[int] = None  # None = auto-detect (min(CPU count, max_workers_cap))
    max_workers_cap: int = 6  # Maximum number of worker processes (increased from 2 for better CPU utilization)
    start_method: str = "spawn"  # Use 'spawn' for cross-platform stability
    
    # Work partitioning settings
    chunk_duration_seconds: float = 6.0  # Split video into 6-second chunks
    min_chunk_duration_seconds: float = 2.0  # Minimum chunk duration
    
    # Worker initialization settings
    # ✅ PERFORMANCE FIX: Increased thread counts from 1 to 2 for better CPU utilization
    # With 6 workers × 2 threads = 12 threads total (good for 8-16 core systems)
    torch_threads: int = 2  # Torch thread count per worker (increased from 1)
    opencv_threads: int = 2  # OpenCV thread count per worker (increased from 1)
    disable_opencv_opencl: bool = True  # Disable OpenCV OpenCL
    
    # Model preloading settings
    preload_models: bool = True  # Preload models in worker initializer
    yolo_model_path: str = "yolo11s.pt"  # Path to YOLO model weights
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
        
        Targets ~60% CPU utilization while maintaining memory safety.
        Formula: Uses min(CPU cores, max_workers_cap) to balance performance and memory.
        With 6 workers × 2 threads each = 12 threads total (good for 8-16 core systems).
        
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


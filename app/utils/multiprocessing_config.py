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
    # ✅ MEMORY FIX: Reduced max_workers_cap from 8 to 2 for memory safety
    max_workers: Optional[int] = None  # None = auto-detect (min(CPU count, max_workers_cap))
    max_workers_cap: int = 2  # Maximum number of worker processes (reduced from 8)
    start_method: str = "spawn"  # Use 'spawn' for cross-platform stability
    
    # Work partitioning settings
    chunk_duration_seconds: float = 6.0  # Split video into 6-second chunks
    min_chunk_duration_seconds: float = 2.0  # Minimum chunk duration
    
    # Worker initialization settings
    torch_threads: int = 1  # Torch thread count per worker
    opencv_threads: int = 1  # OpenCV thread count per worker
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
        
        Returns:
            int: Number of worker processes to use
        """
        if self.max_workers is not None:
            return min(self.max_workers, self.max_workers_cap)
        
        # Auto-detect: min(CPU cores, configured cap)
        cpu_count = mp.cpu_count()
        return min(cpu_count, self.max_workers_cap)
    
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


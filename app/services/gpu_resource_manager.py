"""
GPU Resource Manager - Thread-safe singleton for GPU resource management

Provides centralized management of GPU resources including:
- Device initialization (CUDA or CPU fallback)
- YOLO model loading and fusing for inference
- Asyncio semaphore for concurrency control (max 3 concurrent videos)
- GPU memory pool management
- OOM recovery helpers

This module implements lazy initialization - models are not loaded until first request.
"""

import asyncio
import gc
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional, Tuple, Any, Dict, List

import torch

from ..utils.logger import get_logger
from ..utils.config import get_settings


logger = get_logger(__name__)


class GPUResourceManager:
    """
    Thread-safe singleton GPU Resource Manager

    Manages GPU resources for video processing including:
    - Device detection and initialization
    - YOLO model loading with lazy initialization
    - Concurrency control via asyncio semaphore
    - GPU memory management and OOM recovery

    Usage:
        from app.services.gpu_resource_manager import gpu_resource_manager

        # Get shared model instances
        yolo_model, pose_model = gpu_resource_manager.get_models()

        # Use semaphore for concurrency control
        async with gpu_resource_manager.acquire_gpu_slot():
            # Process video with GPU resources
            ...
    """

    _instance: Optional['GPUResourceManager'] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> 'GPUResourceManager':
        """
        Thread-safe singleton instance creation

        Returns:
            GPUResourceManager: Singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize GPU Resource Manager

        Uses lazy initialization - actual model loading is deferred
        until first call to get_models() or initialize()
        """
        if self._initialized:
            return

        self._initialized = True
        self._models_loaded = False
        self._initialization_lock = threading.Lock()

        logger.info("Entering GPUResourceManager.__init__")

        # Get settings
        self._settings = get_settings()

        # Device configuration
        self._device: Optional[torch.device] = None
        self._device_name: str = "not_initialized"
        self._total_memory_mb: float = 0.0
        self._gpu_available: bool = False

        # Model instances - Detection tier (nano - fast, for Stage 1)
        self._detection_yolo_model: Optional[Any] = None
        self._detection_pose_model: Optional[Any] = None

        # Model instances - Voting tier (medium - accurate, for Stage 2)
        self._yolo_model: Optional[Any] = None
        self._pose_model: Optional[Any] = None
        self._face_mesh: Optional[Any] = None
        self._mp_face_mesh: Optional[Any] = None
        self._preprocessing_service: Optional[Any] = None

        # Concurrency control
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._active_count: int = 0
        self._active_count_lock = threading.Lock()

        # GPU memory tracking
        self._peak_memory_mb: float = 0.0
        self._oom_recovery_count: int = 0

        # Batch size management for OOM recovery
        self._current_batch_size: int = self._settings.inference_batch_size
        self._min_batch_size: int = 1

        # CUDA streams for parallel execution
        self._streams: List[Any] = []
        self._stream_assignments: Dict[int, int] = {}
        self._stream_lock = threading.Lock()

        logger.info(
            "Exiting GPUResourceManager.__init__",
            extra={
                "max_concurrent": self._settings.max_concurrent_videos,
                "batch_size": self._current_batch_size,
                "status": "created (lazy initialization pending)"
            }
        )

    def initialize(self) -> bool:
        """
        Initialize GPU device and configure memory settings

        This method is called automatically on first model access,
        but can be called explicitly during application startup.

        Returns:
            bool: True if initialization successful
        """
        logger.info("Entering initialize()")

        with self._initialization_lock:
            if self._device is not None:
                logger.info("Exiting initialize() - already initialized")
                return True

            try:
                # Configure PyTorch CUDA allocator
                cuda_alloc_conf = self._settings.pytorch_cuda_alloc_conf
                if cuda_alloc_conf:
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = cuda_alloc_conf
                    logger.info(f"Set PYTORCH_CUDA_ALLOC_CONF={cuda_alloc_conf}")

                # Check GPU availability
                gpu_enabled = self._settings.gpu_enabled
                cuda_available = torch.cuda.is_available()
                self._gpu_available = gpu_enabled and cuda_available

                if self._gpu_available:
                    # Use configured GPU device
                    device_str = self._settings.gpu_device
                    self._device = torch.device(device_str)
                    self._device_name = torch.cuda.get_device_name(self._device)

                    # Get total memory
                    self._total_memory_mb = (
                        torch.cuda.get_device_properties(self._device).total_memory
                        / (1024**2)
                    )

                    # Configure memory fraction
                    memory_fraction = self._settings.gpu_memory_fraction
                    torch.cuda.set_per_process_memory_fraction(
                        memory_fraction,
                        device=self._device
                    )

                    # Initialize CUDA streams for parallel execution
                    self._initialize_cuda_streams()

                    logger.info(
                        f"GPU initialized: {self._device_name} "
                        f"({self._total_memory_mb:.0f} MB total, "
                        f"{memory_fraction*100:.0f}% allocated)"
                    )
                else:
                    # Fallback to CPU
                    self._device = torch.device("cpu")
                    self._device_name = "CPU"
                    self._total_memory_mb = 0.0

                    if gpu_enabled and not cuda_available:
                        logger.warning(
                            "GPU enabled in config but CUDA not available - "
                            "falling back to CPU"
                        )
                    else:
                        logger.info("Using CPU device (GPU disabled in config)")

                # Initialize semaphore for concurrency control
                max_concurrent = self._settings.max_concurrent_videos
                self._semaphore = asyncio.Semaphore(max_concurrent)
                logger.info(
                    f"Concurrency semaphore initialized: "
                    f"max {max_concurrent} concurrent videos"
                )

                logger.info("Exiting initialize() - success")
                return True

            except Exception as e:
                logger.error(
                    f"GPU initialization failed: {e}",
                    exc_info=True
                )
                # Fallback to CPU on any error
                self._device = torch.device("cpu")
                self._device_name = "CPU (fallback after error)"
                self._total_memory_mb = 0.0
                self._gpu_available = False
                self._semaphore = asyncio.Semaphore(
                    self._settings.max_concurrent_videos
                )
                logger.info("Exiting initialize() - fallback to CPU")
                return False

    def _initialize_cuda_streams(self) -> None:
        """Initialize CUDA streams for parallel video processing"""
        if not self._gpu_available:
            return

        try:
            # Create streams for each concurrent slot
            max_concurrent = self._settings.max_concurrent_videos
            for i in range(max_concurrent):
                stream = torch.cuda.Stream(device=self._device)
                self._streams.append(stream)

            logger.info(f"Created {len(self._streams)} CUDA streams for parallel execution")

        except Exception as e:
            logger.warning(f"Failed to create CUDA streams: {e}")
            self._streams = []

    def load_models(self, force_reload: bool = False) -> bool:
        """
        Load and fuse YOLO models for inference

        Models are loaded lazily on first access. This method is thread-safe
        and will only load models once even if called from multiple threads.

        Args:
            force_reload: If True, reload models even if already loaded

        Returns:
            bool: True if models loaded successfully
        """
        logger.info("Entering load_models()", extra={"force_reload": force_reload})

        with self._initialization_lock:
            if self._models_loaded and not force_reload:
                logger.info("Exiting load_models() - already loaded")
                return True

            # Ensure device is initialized
            if self._device is None:
                self.initialize()

            try:
                from ultralytics import YOLO

                # =========================================================
                # Detection tier (nano - fast, for Stage 1 activity scanning)
                # =========================================================
                detection_yolo_weights = self._settings.yolo_detection_weights
                detection_pose_weights = self._settings.yolo_detection_pose_weights

                logger.info(f"Loading Detection-tier YOLO model (nano): {detection_yolo_weights}")
                start_time = time.time()

                self._detection_yolo_model = YOLO(detection_yolo_weights)
                if self._gpu_available:
                    self._detection_yolo_model.to(self._device)
                    self._detection_yolo_model.fuse()

                detection_yolo_time = time.time() - start_time
                logger.info(
                    f"Detection YOLO (nano) loaded in {detection_yolo_time:.2f}s "
                    f"on {self._device_name}"
                )

                logger.info(f"Loading Detection-tier Pose model (nano): {detection_pose_weights}")
                start_time = time.time()

                self._detection_pose_model = YOLO(detection_pose_weights)
                if self._gpu_available:
                    self._detection_pose_model.to(self._device)
                    self._detection_pose_model.fuse()

                detection_pose_time = time.time() - start_time
                logger.info(
                    f"Detection Pose (nano) loaded in {detection_pose_time:.2f}s "
                    f"on {self._device_name}"
                )

                # =========================================================
                # Voting tier (medium - accurate, for Stage 2 verification)
                # =========================================================
                yolo_weights = self._settings.yolo_weights
                pose_weights = self._settings.yolo_pose_weights

                logger.info(f"Loading Voting-tier YOLO model (medium): {yolo_weights}")
                start_time = time.time()

                # Load object detection model
                self._yolo_model = YOLO(yolo_weights)
                if self._gpu_available:
                    self._yolo_model.to(self._device)
                    # Fuse model for inference (optimizes Conv+BatchNorm layers)
                    self._yolo_model.fuse()

                yolo_load_time = time.time() - start_time
                logger.info(
                    f"Voting YOLO (medium) loaded and fused in {yolo_load_time:.2f}s "
                    f"on {self._device_name}"
                )

                # Load pose estimation model
                logger.info(f"Loading Voting-tier Pose model (medium): {pose_weights}")
                start_time = time.time()

                self._pose_model = YOLO(pose_weights)
                if self._gpu_available:
                    self._pose_model.to(self._device)
                    # Fuse model for inference
                    self._pose_model.fuse()

                pose_load_time = time.time() - start_time
                logger.info(
                    f"Voting Pose (medium) loaded and fused in {pose_load_time:.2f}s "
                    f"on {self._device_name}"
                )

                # Load MediaPipe FaceMesh
                logger.info("Loading MediaPipe FaceMesh")
                try:
                    import mediapipe as mp
                    self._mp_face_mesh = mp.solutions.face_mesh
                    self._face_mesh = self._mp_face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=3,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                except ImportError:
                    logger.warning("MediaPipe not available, skipping FaceMesh")
                    self._mp_face_mesh = None
                    self._face_mesh = None

                # Load preprocessing service
                logger.info("Loading ImagePreprocessingService")
                try:
                    from .image_preprocessing_service import ImagePreprocessingService
                    self._preprocessing_service = ImagePreprocessingService()
                except ImportError:
                    logger.warning("ImagePreprocessingService not available")
                    self._preprocessing_service = None

                # Log GPU memory usage after model loading
                if self._gpu_available:
                    self._log_gpu_memory_stats()

                self._models_loaded = True
                logger.info(
                    f"Exiting load_models() - all models ready on {self._device_name}"
                )
                return True

            except Exception as e:
                logger.error(f"Model loading failed: {e}", exc_info=True)
                self._models_loaded = False
                return False

    def get_models(self) -> Tuple[Any, Any]:
        """
        Get shared YOLO model instances (tuple format)

        Performs lazy initialization if models haven't been loaded yet.

        Returns:
            Tuple[Any, Any]: (yolo_model, pose_model) instances

        Raises:
            RuntimeError: If model loading fails
        """
        logger.info("Entering get_models()")

        if not self._models_loaded:
            success = self.load_models()
            if not success:
                raise RuntimeError("Failed to load YOLO models")

        logger.info(
            f"Exiting get_models() - returning models on {self._device_name}"
        )
        return self._yolo_model, self._pose_model

    def get_models_dict(self) -> Dict[str, Any]:
        """
        Get dictionary of loaded models for use in workers.

        Returns:
            Dict[str, Any]: Dictionary containing model references:
                - yolo: Detection-tier YOLO model (nano - fast)
                - yolo_pose: Detection-tier YOLO pose model (nano - fast)
                - yolo_voting: Voting-tier YOLO model (medium - accurate)
                - yolo_pose_voting: Voting-tier YOLO pose model (medium - accurate)
                - face_mesh: MediaPipe FaceMesh instance
                - mp_face_mesh: MediaPipe FaceMesh class
                - preprocessing_service: Image preprocessing service
        """
        logger.debug("Entering get_models_dict")

        if not self._models_loaded:
            self.load_models()

        models = {
            # Detection tier (nano - fast, for Stage 1)
            'yolo': self._detection_yolo_model,
            'yolo_pose': self._detection_pose_model,
            # Voting tier (medium - accurate, for Stage 2)
            'yolo_voting': self._yolo_model,
            'yolo_pose_voting': self._pose_model,
            # Shared resources
            'face_mesh': self._face_mesh,
            'mp_face_mesh': self._mp_face_mesh,
            'preprocessing_service': self._preprocessing_service
        }

        logger.debug("Exiting get_models_dict", extra={"num_models": len(models)})
        return models

    def get_detection_models(self) -> Tuple[Any, Any]:
        """
        Get detection-tier model instances (nano - fast for Stage 1).

        Returns:
            Tuple[Any, Any]: (detection_yolo_model, detection_pose_model)
        """
        if not self._models_loaded:
            self.load_models()
        return self._detection_yolo_model, self._detection_pose_model

    def get_voting_models(self) -> Tuple[Any, Any]:
        """
        Get voting-tier model instances (medium - accurate for Stage 2).

        Returns:
            Tuple[Any, Any]: (voting_yolo_model, voting_pose_model)
        """
        if not self._models_loaded:
            self.load_models()
        return self._yolo_model, self._pose_model

    def get_yolo_model(self) -> Any:
        """
        Get shared YOLO object detection model

        Returns:
            YOLO model instance for object detection
        """
        yolo_model, _ = self.get_models()
        return yolo_model

    def get_pose_model(self) -> Any:
        """
        Get shared YOLO pose estimation model

        Returns:
            YOLO model instance for pose estimation
        """
        _, pose_model = self.get_models()
        return pose_model

    @asynccontextmanager
    async def acquire_gpu_slot(self):
        """
        Async context manager for acquiring a GPU processing slot

        Limits concurrent video processing to prevent GPU memory exhaustion.

        Usage:
            async with gpu_resource_manager.acquire_gpu_slot():
                # Process video here
                results = model.predict(frame)

        Yields:
            None
        """
        logger.info(
            f"Entering acquire_gpu_slot() - current active: {self._active_count}"
        )

        # Ensure semaphore is initialized
        if self._semaphore is None:
            self.initialize()

        try:
            await self._semaphore.acquire()

            with self._active_count_lock:
                self._active_count += 1
                current_count = self._active_count

            logger.info(
                f"GPU slot acquired - active workers: {current_count}/"
                f"{self._settings.max_concurrent_videos}"
            )

            yield

        finally:
            self._semaphore.release()

            with self._active_count_lock:
                self._active_count -= 1
                current_count = self._active_count

            logger.info(
                f"GPU slot released - active workers: {current_count}/"
                f"{self._settings.max_concurrent_videos}"
            )

    def increment_active(self) -> int:
        """
        Increment active video count when starting a new job.

        Returns:
            int: New active count
        """
        with self._active_count_lock:
            self._active_count += 1
            count = self._active_count

        logger.debug(f"Active video count incremented to {count}")
        return count

    def decrement_active(self) -> int:
        """
        Decrement active video count when a job completes.

        Returns:
            int: New active count
        """
        with self._active_count_lock:
            self._active_count = max(0, self._active_count - 1)
            count = self._active_count

        logger.debug(f"Active video count decremented to {count}")
        return count

    def can_accept_job(self) -> bool:
        """
        Check if the system can accept another video processing job.

        Returns:
            bool: True if under max concurrent limit
        """
        return self._active_count < self._settings.max_concurrent_videos

    def _log_gpu_memory_stats(self) -> None:
        """Log current GPU memory statistics"""
        if self._device is None or self._device.type != "cuda":
            return

        try:
            allocated = torch.cuda.memory_allocated(self._device) / (1024**2)
            reserved = torch.cuda.memory_reserved(self._device) / (1024**2)
            max_allocated = torch.cuda.max_memory_allocated(self._device) / (1024**2)

            self._peak_memory_mb = max(self._peak_memory_mb, allocated)

            logger.info(
                f"GPU Memory - Allocated: {allocated:.1f}MB, "
                f"Reserved: {reserved:.1f}MB, "
                f"Peak: {max_allocated:.1f}MB"
            )
        except Exception as e:
            logger.warning(f"Failed to log GPU memory stats: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current GPU memory statistics.

        Returns:
            dict: GPU memory stats including:
                - gpu_available: Whether GPU is available
                - allocated_mb: Currently allocated memory
                - cached_mb: Memory reserved by caching allocator
                - max_allocated_mb: Peak memory allocation
                - device_name: GPU device name
                - total_memory_mb: Total GPU memory
                - free_mb: Available memory
                - utilization_percent: Memory usage percentage
        """
        logger.debug("Entering get_memory_stats")

        if not self._gpu_available:
            stats = {"gpu_available": False, "device_name": self._device_name}
            logger.debug("Exiting get_memory_stats - no GPU")
            return stats

        try:
            allocated_mb = torch.cuda.memory_allocated(self._device) / (1024**2)
            cached_mb = torch.cuda.memory_reserved(self._device) / (1024**2)
            max_allocated_mb = torch.cuda.max_memory_allocated(self._device) / (1024**2)

            # Calculate free memory and utilization
            free_mb = self._total_memory_mb - cached_mb
            utilization_percent = (
                (cached_mb / self._total_memory_mb * 100)
                if self._total_memory_mb > 0
                else 0
            )

            stats = {
                "gpu_available": True,
                "allocated_mb": round(allocated_mb, 2),
                "cached_mb": round(cached_mb, 2),
                "max_allocated_mb": round(max_allocated_mb, 2),
                "device_name": self._device_name,
                "total_memory_mb": round(self._total_memory_mb, 2),
                "free_mb": round(free_mb, 2),
                "utilization_percent": round(utilization_percent, 2),
            }

            logger.debug(
                f"Exiting get_memory_stats - utilization: {utilization_percent:.1f}%"
            )
            return stats

        except Exception as e:
            logger.error(f"Error getting memory stats: {e}", exc_info=True)
            return {
                "gpu_available": True,
                "error": str(e),
                "device_name": self._device_name,
            }

    def clear_gpu_memory(self) -> None:
        """
        Clear GPU memory caches

        Call this method to free up GPU memory after processing,
        especially useful after OOM errors or between batches.
        """
        logger.info("Entering clear_gpu_memory()")

        if not self._gpu_available:
            logger.info("Exiting clear_gpu_memory() - not using GPU")
            return

        try:
            # Get memory before clearing
            before_cached = torch.cuda.memory_reserved(self._device) / (1024**2)

            # Clear PyTorch cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Run garbage collection
            gc.collect()

            # Get memory after clearing
            after_cached = torch.cuda.memory_reserved(self._device) / (1024**2)
            freed_mb = before_cached - after_cached

            logger.info(
                f"Exiting clear_gpu_memory() - freed {freed_mb:.1f}MB "
                f"(before: {before_cached:.1f}MB, after: {after_cached:.1f}MB)"
            )

        except Exception as e:
            logger.warning(f"Failed to clear GPU memory: {e}")

    def clear_memory_cache(self) -> None:
        """
        Alias for clear_gpu_memory() for backward compatibility.

        Clear CUDA memory cache to free up reserved memory.
        """
        self.clear_gpu_memory()

    def handle_oom_error(self) -> bool:
        """
        Handle Out of Memory error with recovery attempt

        Attempts to recover from GPU OOM by clearing caches and
        optionally reducing batch sizes.

        Returns:
            bool: True if recovery was attempted, False if OOM retry disabled
        """
        logger.info("Entering handle_oom_error()")

        self._oom_recovery_count += 1

        if not self._settings.oom_retry_enabled:
            logger.warning(
                f"OOM error occurred (count: {self._oom_recovery_count}) - "
                "retry disabled, not attempting recovery"
            )
            return False

        logger.warning(
            f"OOM error occurred (count: {self._oom_recovery_count}) - "
            "attempting recovery"
        )

        # Clear GPU memory
        self.clear_gpu_memory()

        # Reduce batch size if possible
        if self._current_batch_size > self._min_batch_size:
            old_batch_size = self._current_batch_size
            self._current_batch_size = max(
                self._min_batch_size,
                self._current_batch_size // 2
            )
            logger.info(
                f"Reduced batch size from {old_batch_size} to {self._current_batch_size}"
            )

        # Log recovery attempt
        logger.info(
            f"Exiting handle_oom_error() - recovery attempted "
            f"(total OOM count: {self._oom_recovery_count})"
        )

        return True

    async def handle_oom_recovery(self) -> None:
        """
        Async version of OOM error recovery.

        Clears CUDA cache, reduces batch size, and synchronizes.
        Should be called when torch.cuda.OutOfMemoryError is caught.
        """
        self.handle_oom_error()

    def get_current_batch_size(self) -> int:
        """
        Get current batch size (may be reduced after OOM recovery)

        Returns:
            int: Current batch size for inference
        """
        return self._current_batch_size

    @property
    def current_batch_size(self) -> int:
        """Get current inference batch size"""
        return self._current_batch_size

    def reset_batch_size(self) -> int:
        """
        Reset batch size to configured default after successful processing.

        Returns:
            int: Reset batch size
        """
        self._current_batch_size = self._settings.inference_batch_size
        logger.info(f"Batch size reset to {self._current_batch_size}")
        return self._current_batch_size

    def reduce_batch_size(self) -> int:
        """
        Reduce batch size for OOM recovery.

        Implements dynamic batch reduction: 8 -> 4 -> 2 -> 1

        Returns:
            int: New batch size
        """
        logger.info(
            "Entering reduce_batch_size",
            extra={"current_batch_size": self._current_batch_size}
        )

        if self._current_batch_size > self._min_batch_size:
            self._current_batch_size = max(
                self._min_batch_size,
                self._current_batch_size // 2
            )

        logger.info(
            "Exiting reduce_batch_size",
            extra={"new_batch_size": self._current_batch_size}
        )
        return self._current_batch_size

    def check_memory_health(self) -> Tuple[bool, str]:
        """
        Check if GPU memory usage is healthy (below 80% threshold).

        Returns:
            tuple[bool, str]: (is_healthy, message)
                - is_healthy: True if memory usage is below 80%
                - message: Human-readable status message
        """
        logger.debug("Entering check_memory_health")

        if not self._gpu_available:
            message = "GPU not available - running in CPU mode"
            logger.debug("Exiting check_memory_health - no GPU")
            return True, message

        try:
            stats = self.get_memory_stats()
            utilization = stats.get("utilization_percent", 0)

            # Health threshold at 80%
            MEMORY_WARNING_THRESHOLD = 80.0

            if utilization >= MEMORY_WARNING_THRESHOLD:
                is_healthy = False
                message = (
                    f"High GPU memory usage: {utilization:.1f}% "
                    f"({stats.get('cached_mb', 0):.0f}/{self._total_memory_mb:.0f} MB)"
                )
                logger.warning(f"GPU memory health warning: {message}")
            else:
                is_healthy = True
                message = (
                    f"GPU memory healthy: {utilization:.1f}% utilization "
                    f"({stats.get('free_mb', 0):.0f} MB free)"
                )

            logger.debug(f"Exiting check_memory_health - healthy: {is_healthy}")
            return is_healthy, message

        except Exception as e:
            logger.error(f"Error checking memory health: {e}", exc_info=True)
            return False, f"Error checking GPU health: {str(e)}"

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the GPU Resource Manager

        Returns:
            Dict with status information including device, memory, and concurrency
        """
        memory_stats = self.get_memory_stats()

        status = {
            "initialized": self._device is not None,
            "models_loaded": self._models_loaded,
            "device": str(self._device) if self._device else "not_initialized",
            "device_name": self._device_name,
            "gpu_available": self._gpu_available,
            "active_count": self._active_count,
            "max_concurrent": self._settings.max_concurrent_videos,
            "oom_recovery_count": self._oom_recovery_count,
            "current_batch_size": self._current_batch_size,
            "model_names": ["yolo", "yolo_pose", "face_mesh"] if self._models_loaded else [],
            "num_cuda_streams": len(self._streams),
            **memory_stats
        }

        return status

    def log_startup_stats(self) -> None:
        """Log GPU statistics at application startup"""
        logger.info("=" * 60)
        logger.info("GPU Resource Manager - Startup Statistics")
        logger.info("-" * 60)

        if self._gpu_available:
            stats = self.get_memory_stats()
            logger.info(f"GPU Device: {self._device_name}")
            logger.info(f"Total Memory: {self._total_memory_mb:.0f} MB")
            logger.info(f"Allocated: {stats.get('allocated_mb', 0):.2f} MB")
            logger.info(f"Cached: {stats.get('cached_mb', 0):.2f} MB")
            logger.info(f"Free: {stats.get('free_mb', 0):.2f} MB")
            logger.info(f"Utilization: {stats.get('utilization_percent', 0):.1f}%")
            logger.info(f"Max Concurrent Videos: {self._settings.max_concurrent_videos}")
            logger.info(f"Inference Batch Size: {self._current_batch_size}")
            logger.info(f"CUDA Streams: {len(self._streams)}")
        else:
            logger.info("GPU Status: Not available (CPU-only mode)")
            logger.info(f"Device: {self._device_name}")
            logger.info(f"Max Concurrent Videos: {self._settings.max_concurrent_videos}")

        logger.info("=" * 60)

    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics for new monitoring period"""
        if not self._gpu_available:
            return

        try:
            torch.cuda.reset_peak_memory_stats(self._device)
            self._peak_memory_mb = 0.0
            logger.info("GPU peak memory statistics reset")
        except Exception as e:
            logger.error(f"Error resetting peak stats: {e}", exc_info=True)

    # =========================================================================
    # Stream Management Methods
    # =========================================================================

    def get_stream(self, worker_id: int) -> Optional[Any]:
        """
        Get CUDA stream for a specific worker.

        Args:
            worker_id: Worker identifier (0-indexed)

        Returns:
            Optional[torch.cuda.Stream]: CUDA stream or None if not available
        """
        if not self._streams:
            return None

        if worker_id < len(self._streams):
            return self._streams[worker_id]

        # Fallback to round-robin if worker_id exceeds stream count
        return self._streams[worker_id % len(self._streams)]

    def assign_stream(self, worker_id: int) -> Optional[int]:
        """
        Assign a CUDA stream to a worker.

        Args:
            worker_id: Worker identifier

        Returns:
            Optional[int]: Assigned stream index or None
        """
        if not self._streams:
            return None

        with self._stream_lock:
            # Find an available stream (simple round-robin)
            stream_idx = worker_id % len(self._streams)
            self._stream_assignments[worker_id] = stream_idx
            logger.debug(
                f"Assigned stream {stream_idx} to worker {worker_id}"
            )
            return stream_idx

    def release_stream(self, worker_id: int) -> None:
        """
        Release a CUDA stream assignment for a worker.

        Args:
            worker_id: Worker identifier
        """
        with self._stream_lock:
            if worker_id in self._stream_assignments:
                del self._stream_assignments[worker_id]
                logger.debug(f"Released stream for worker {worker_id}")

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def active_count(self) -> int:
        """
        Number of currently active GPU processing slots

        Returns:
            int: Count of active workers using GPU resources
        """
        return self._active_count

    @property
    def is_initialized(self) -> bool:
        """
        Check if GPU device has been initialized

        Returns:
            bool: True if device initialization is complete
        """
        return self._device is not None

    @property
    def models_loaded(self) -> bool:
        """Check if models are loaded"""
        return self._models_loaded

    @property
    def device_name(self) -> str:
        """
        Get the name of the initialized device

        Returns:
            str: Device name (GPU model name or "CPU")
        """
        return self._device_name

    @property
    def device(self) -> Optional[torch.device]:
        """
        Get the PyTorch device instance

        Returns:
            torch.device: The initialized device, or None if not initialized
        """
        return self._device

    @property
    def is_gpu_available(self) -> bool:
        """
        Check if GPU is being used

        Returns:
            bool: True if using CUDA device
        """
        return self._gpu_available

    @property
    def total_memory_mb(self) -> float:
        """Get total GPU memory in MB"""
        return self._total_memory_mb

    @property
    def max_concurrent(self) -> int:
        """Get maximum concurrent videos"""
        return self._settings.max_concurrent_videos

    # =========================================================================
    # Cleanup Methods
    # =========================================================================

    def unload_models(self) -> None:
        """
        Unload models and free GPU memory.

        Call during application shutdown.
        """
        logger.info("Entering unload_models")

        try:
            # Close MediaPipe resources
            if self._face_mesh is not None:
                self._face_mesh.close()
                self._face_mesh = None

            # Clear detection tier model references (nano)
            self._detection_yolo_model = None
            self._detection_pose_model = None

            # Clear voting tier model references (medium)
            self._yolo_model = None
            self._pose_model = None
            self._mp_face_mesh = None
            self._preprocessing_service = None

            # Clear CUDA streams
            self._streams = []
            self._stream_assignments = {}

            # Clear GPU memory
            if self._gpu_available:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            self._models_loaded = False

            logger.info("Exiting unload_models", extra={"status": "unloaded"})

        except Exception as e:
            logger.error(f"Error unloading models: {e}", exc_info=True)

    def shutdown(self) -> None:
        """
        Shutdown the GPU Resource Manager

        Clears GPU memory and releases model references.
        Call this during application shutdown.
        """
        logger.info("Entering shutdown()")

        # Unload models
        self.unload_models()

        # Clear GPU memory
        if self._gpu_available:
            self.clear_gpu_memory()

        # Final garbage collection
        gc.collect()

        if self._gpu_available:
            torch.cuda.empty_cache()

        logger.info("Exiting shutdown() - GPU resources released")


# Module-level singleton instance export
gpu_resource_manager = GPUResourceManager()


def get_gpu_resource_manager() -> GPUResourceManager:
    """
    Get the singleton GPU Resource Manager instance.

    Returns:
        GPUResourceManager: The singleton instance
    """
    return gpu_resource_manager

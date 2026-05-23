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
from typing import Optional, Tuple, Any, Dict

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

        # Model instances
        self._yolo_model: Optional[Any] = None
        self._pose_model: Optional[Any] = None
        self._face_mesh: Optional[Any] = None
        self._mp_face_mesh: Optional[Any] = None
        self._preprocessing_service: Optional[Any] = None

        # Concurrency control
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._active_count: int = 0
        self._active_count_lock = threading.Lock()
        # ``_pending_count`` tracks requests that have been admitted by the
        # controller but are still waiting on the semaphore (i.e. active slots
        # are full). Combined with ``_active_count`` this lets the controller
        # enforce a bounded queue — when active+pending >= (max + queue_max),
        # we return 503 instead of admitting unbounded work.
        self._pending_count: int = 0
        self._pending_count_lock = threading.Lock()

        # GPU memory tracking
        self._peak_memory_mb: float = 0.0
        self._oom_recovery_count: int = 0

        # Batch size management for OOM recovery
        self._current_batch_size: int = self._settings.inference_batch_size
        self._min_batch_size: int = 1

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

                # Note: ``self._semaphore`` is intentionally *not* constructed
                # here. ``asyncio.Semaphore`` binds to whichever event loop
                # runs the first ``await`` against it; constructing it in this
                # synchronous path (called from gunicorn startup or test
                # ``setUp``) ties it to whatever loop happens to be current
                # then — and later awaits from a different loop raise
                # ``RuntimeError: <Semaphore [...]> is bound to a different
                # event loop``. Defer creation to ``acquire_gpu_slot()`` where
                # we are guaranteed to be inside the loop that will await it.
                logger.info(
                    f"Concurrency cap configured: max "
                    f"{self._settings.max_concurrent_videos} concurrent videos "
                    f"(semaphore lazy-constructed on first acquire_gpu_slot)"
                )

                logger.info("Exiting initialize() - success")
                return True

            except Exception as e:
                logger.error(
                    f"GPU initialization failed: {e}",
                    exc_info=True
                )
                # Fallback to CPU on any error. Semaphore is still lazy —
                # see note above; do not construct it here either.
                self._device = torch.device("cpu")
                self._device_name = "CPU (fallback after error)"
                self._total_memory_mb = 0.0
                self._gpu_available = False
                logger.info("Exiting initialize() - fallback to CPU")
                return False

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

                # Get model paths from settings
                yolo_weights = self._settings.yolo_weights
                pose_weights = self._settings.yolo_pose_weights

                logger.info(f"Loading YOLO model: {yolo_weights}")
                start_time = time.time()

                # Load object detection model
                self._yolo_model = YOLO(yolo_weights)
                if self._gpu_available:
                    self._yolo_model.to(self._device)
                    # Fuse model for inference (optimizes Conv+BatchNorm layers)
                    self._yolo_model.fuse()

                yolo_load_time = time.time() - start_time
                logger.info(
                    f"YOLO model loaded and fused in {yolo_load_time:.2f}s "
                    f"on {self._device_name}"
                )

                # Load pose estimation model
                logger.info(f"Loading Pose model: {pose_weights}")
                start_time = time.time()

                self._pose_model = YOLO(pose_weights)
                if self._gpu_available:
                    self._pose_model.to(self._device)
                    # Fuse model for inference
                    self._pose_model.fuse()

                pose_load_time = time.time() - start_time
                logger.info(
                    f"Pose model loaded and fused in {pose_load_time:.2f}s "
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
                - yolo: YOLO object detection model
                - yolo_pose: YOLO pose estimation model
                - face_mesh: MediaPipe FaceMesh instance
                - mp_face_mesh: MediaPipe FaceMesh class
                - preprocessing_service: Image preprocessing service
        """
        logger.debug("Entering get_models_dict")

        if not self._models_loaded:
            self.load_models()

        models = {
            'yolo': self._yolo_model,
            'yolo_pose': self._pose_model,
            'face_mesh': self._face_mesh,
            'mp_face_mesh': self._mp_face_mesh,
            'preprocessing_service': self._preprocessing_service
        }

        logger.debug("Exiting get_models_dict", extra={"num_models": len(models)})
        return models

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
        The underlying ``asyncio.Semaphore`` is lazy-constructed on first
        call so the manager can be instantiated synchronously (outside any
        event loop) and then awaited from whichever loop the request lands
        in — ``asyncio.Semaphore`` binds to the loop running the first
        ``await``, so eager construction tied it to the wrong loop.

        Counter authority (Task 0003): this method does *not* increment
        ``_active_count`` on entry. The migration from pending → active
        happens atomically inside :meth:`mark_enqueued_started`, which
        every caller (controller path *and* queue-worker path) is
        expected to invoke once the slot is held. That keeps an admitted
        request in *exactly one* of pending or active at every observable
        moment — never both — which is the invariant
        :meth:`try_enqueue`'s position formula depends on. The matching
        ``_active_count`` decrement on slot release lives in this
        method's finally block.

        Usage (controller path — request went through ``try_enqueue``):
            async with gpu_resource_manager.acquire_gpu_slot():
                # Migrate the request from pending → active so the
                # counters reflect that it is now running rather than
                # waiting. This is the only path that should call
                # ``mark_enqueued_started``.
                gpu_resource_manager.mark_enqueued_started()
                results = model.predict(frame)

        Usage (queue-worker path — bypassed ``try_enqueue``):
            async with gpu_resource_manager.acquire_gpu_slot():
                # No pending entry to consume; just bump active so
                # observability sees the running job. ``mark_enqueued_started``
                # would corrupt ``_pending_count`` for any in-flight
                # controller admission, so use ``increment_active``
                # instead.
                gpu_resource_manager.increment_active()
                results = model.predict(frame)

        Yields:
            None
        """
        logger.info(
            f"Entering acquire_gpu_slot() - current active: {self._active_count}"
        )

        # Lazy-construct the semaphore on first use. We deliberately do NOT
        # call ``self.initialize()`` here for the semaphore alone — that
        # method is for CUDA setup and is unsafe to call from arbitrary
        # event loops. The semaphore itself is cheap to build inline. The
        # check + assignment is guarded by ``_initialization_lock`` so
        # concurrent first-callers from threads/loops can't both create
        # competing semaphores.
        if self._semaphore is None:
            with self._initialization_lock:
                if self._semaphore is None:
                    self._semaphore = asyncio.Semaphore(
                        self._settings.max_concurrent_videos
                    )
                    logger.info(
                        f"Concurrency semaphore lazy-constructed "
                        f"(max {self._settings.max_concurrent_videos})"
                    )

        async with self._semaphore:
            # Post-acquire log line. The "Entering acquire_gpu_slot()" line
            # above logs the *request* — but if the semaphore is contended
            # the value of ``_active_count`` it carries is stale by the time
            # the slot is actually held. This second line is emitted right
            # after the await succeeds so ops can correlate the timeline:
            # request → wait → acquired. Reading ``_active_count`` is racy
            # without the lock, but it's only an observability hint here so
            # we keep the log path lock-free.
            logger.info(
                f"acquire_gpu_slot() slot acquired - active={self._active_count}/"
                f"{self._settings.max_concurrent_videos}"
            )
            try:
                yield
            finally:
                # Symmetric ``_active_count`` decrement. We don't know
                # whether ``mark_enqueued_started`` actually ran (the
                # caller might raise before it does), so clamp at zero
                # to stay robust under partial-failure paths.
                with self._active_count_lock:
                    self._active_count = max(0, self._active_count - 1)
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

    def can_accept_job(self) -> bool:
        """
        Check if the system can accept another video processing job.

        Returns:
            bool: True if under max concurrent limit
        """
        return self._active_count < self._settings.max_concurrent_videos

    def try_enqueue(self) -> tuple[bool, int]:
        """Admission check for a new video job.

        Returns ``(admitted, position)``:
        - ``admitted=True``  → caller may proceed to ``acquire_gpu_slot()``.
          The pending counter is incremented; caller MUST call
          ``mark_enqueued_started()`` right after the slot is acquired so
          pending decrements stay consistent. ``acquire_gpu_slot()`` itself
          owns ``_active_count`` (single counter authority).
        - ``admitted=False`` → queue is full. Caller returns HTTP 503.

        The admissions cap is ``max_concurrent_videos + job_queue_max_size``.
        At steady state this allows up to ``max_concurrent_videos`` active
        jobs plus ``job_queue_max_size`` waiters. Beyond that we shed load
        rather than grow the queue unbounded.

        Counter discipline:
        - ``_pending_count`` counts only requests admitted but not yet
          inside ``acquire_gpu_slot()``'s body. Once the slot is acquired
          and ``mark_enqueued_started()`` runs, the request migrates from
          pending to active.
        - The cap check uses ``_pending_count + _active_count`` (a request
          is in *exactly one* of those buckets, never both — that was the
          double-count bug this method previously had).
        - ``position`` reflects ordinal place in the line including the
          currently running jobs: with one active job and an empty queue,
          a freshly admitted request gets ``position=2`` (one ahead of it,
          itself is the second).
        """
        max_active = int(self._settings.max_concurrent_videos)
        max_queue = int(getattr(self._settings, 'job_queue_max_size', 10))
        with self._pending_count_lock:
            with self._active_count_lock:
                in_system = self._active_count + self._pending_count
                if in_system >= max_active + max_queue:
                    return False, in_system
                self._pending_count += 1
                # Position = "your ordinal in the queue counting active
                # jobs in front of you". One active job + this newly
                # pending request → position 2.
                position = self._active_count + self._pending_count
        return True, position

    def mark_enqueued_started(self) -> None:
        """Migrate a controller-path request from pending to active.

        Called *inside* the ``acquire_gpu_slot()`` ``with`` block, after
        the semaphore has been acquired, by callers that previously went
        through :meth:`try_enqueue`. Decrements ``_pending_count`` and
        increments ``_active_count``, so an admitted request is counted
        in *exactly one* of pending or active at every observable moment
        — that's the single counter authority promise this module makes
        (Task 0003).

        Only the controller / admission-gate path should call this. The
        queue-worker path bypasses ``try_enqueue`` and instead calls
        :meth:`increment_active` directly, because decrementing pending
        without a matching ``try_enqueue`` increment would corrupt the
        admission accounting for any in-flight controller request.

        Pairs with ``try_enqueue()``.

        Lock ordering note: ``_pending_count_lock`` is acquired *before*
        ``_active_count_lock`` (nested), matching ``try_enqueue``'s order.
        This makes the pending→active migration atomic — no observer can
        see the request as neither pending nor active (a one-slot
        under-count window the previous sequential-locks version had).
        Keeping the same lock acquisition order across the module avoids
        deadlock with concurrent ``try_enqueue`` callers.
        """
        with self._pending_count_lock:
            with self._active_count_lock:
                self._pending_count = max(0, self._pending_count - 1)
                self._active_count += 1

    def release_enqueue_on_error(self) -> None:
        """Roll back a pending admission if the caller bailed before the slot
        was acquired (e.g. request body validation failed after admission).
        Safe to call even if ``mark_enqueued_started()`` already ran — the
        counter is clamped at zero."""
        with self._pending_count_lock:
            self._pending_count = max(0, self._pending_count - 1)

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

            # Clear model references
            self._yolo_model = None
            self._pose_model = None
            self._mp_face_mesh = None
            self._preprocessing_service = None

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


# ---------------------------------------------------------------------------
# Lazy singleton access
# ---------------------------------------------------------------------------
# C-1 (task 2.2): the previous eager ``gpu_resource_manager = GPUResourceManager()``
# at module scope created a singleton — and, depending on configuration, could
# touch CUDA — at *import* time. Under gunicorn's ``preload_app = True`` this
# runs in the master process before fork, which is hostile to CUDA: any CUDA
# context initialised pre-fork becomes unusable in the forked worker.
#
# To fix this we instantiate lazily on first call to :func:`get_gpu_resource_manager`
# (``GPUResourceManager.__new__`` itself already implements thread-safe
# double-checked locking via ``_lock``). A module-level ``__getattr__`` shim
# preserves backward compatibility for any caller that still writes
# ``from app.services.gpu_resource_manager import gpu_resource_manager`` — they
# will trigger the same lazy init on first attribute access rather than at
# import time.


def get_gpu_resource_manager() -> GPUResourceManager:
    """
    Get the singleton GPU Resource Manager instance.

    Performs lazy initialisation on first call — importing this module does
    not create a ``GPUResourceManager`` (or any CUDA context).

    Returns:
        GPUResourceManager: The singleton instance
    """
    # ``GPUResourceManager.__new__`` uses a class-level lock for thread-safe
    # singleton creation; simply calling the constructor is sufficient and
    # idempotent.
    return GPUResourceManager()


def __getattr__(name: str) -> Any:
    """Backward-compat shim for the module-level ``gpu_resource_manager`` name.

    Any ``from app.services.gpu_resource_manager import gpu_resource_manager``
    or ``app.services.gpu_resource_manager.gpu_resource_manager`` access routes
    through here and triggers the lazy constructor. Every other attribute
    lookup falls through to the normal ``AttributeError`` so typos still fail
    loudly.
    """
    if name == "gpu_resource_manager":
        return get_gpu_resource_manager()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

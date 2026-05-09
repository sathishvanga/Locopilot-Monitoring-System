"""
Configuration management for the Locopilot Monitoring System

Uses environment variables with sensible defaults for production deployment.
"""

import os
import json
import tempfile
from typing import Optional, List
from functools import lru_cache
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Thread configuration for CPU inference optimization
# Allow worker processes to set optimal thread counts
# Conservative default of 2, worker initializer will override based on actual worker count
default_threads = 2  # Safe default, workers will set optimal value
os.environ.setdefault('OMP_NUM_THREADS', str(default_threads))
os.environ.setdefault('MKL_NUM_THREADS', str(default_threads))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(default_threads))

# Phase 3.5 Quick Win B: OpenCV threading for faster preprocessing (5-10% speedup)
import cv2
opencv_threads = int(os.getenv('OPENCV_THREADS', '4'))  # Match worker thread count
cv2.setNumThreads(opencv_threads)


class Settings(BaseSettings):
    """
    Application settings with environment variable support
    
    All settings can be overridden via environment variables.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application settings
    app_name: str = "Locopilot Monitoring System"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS settings - allow all origins by default
    cors_allowed_origins: List[str] = json.loads(
        os.getenv("CORS_ALLOWED_ORIGINS", '["*"]')
    )

    # Media / status API auth key (C-9). When set, the media and status
    # endpoints require an ``X-API-Key`` header whose value matches this
    # setting via ``hmac.compare_digest``. When unset (None / empty), the
    # auth dependency logs a one-shot warning per process and allows the
    # request through — backward-compatible rollout mode. Flip to required
    # by setting MEDIA_API_KEY in ``.env.production`` once clients have
    # been updated.
    media_api_key: Optional[str] = os.getenv("MEDIA_API_KEY", None)

    # File upload settings
    max_upload_size: int = 5 * 1024 * 1024 * 1024  # 5 GB
    allowed_video_extensions: List[str] = [".mp4", ".avi", ".mov", ".mkv"]
    # Use cross-platform temp directory (works on Windows, macOS, and Linux)
    upload_dir: str = os.getenv("UPLOAD_DIR", os.path.join(tempfile.gettempdir(), "locopilot_uploads"))
    
    # Output settings
    # Convert to absolute path to avoid path resolution issues
    output_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
        "locopilot_evidence"
    )
    save_annotated_frames: bool = True
    frame_save_interval: int = 1
    
    # Video processing settings
    sample_fps: float = 0.5  # Sample at 0.5 FPS (1 frame every 2 seconds)
    
    # Multiprocessing settings
    enable_multiprocessing: bool = bool(int(os.getenv("ENABLE_MULTIPROCESSING", "1")))
    # 15s chunks ensure hand gesture coordination detection works correctly
    # Coordination window is 10s, so 15s chunks capture full coordination sequences
    # Tradeoff: fewer chunks (~118 for 30-min video) but reliable coordination detection
    mp_chunk_duration: float = 15.0  # Chunk duration in seconds (optimized for coordination detection)
    # ARCH-03: overlap window between adjacent worker chunks.  Must be >=
    # max(sleep_baseline_calibration_window, hand_gesture_coordination_window)
    # so pose-based sleep and gesture coordination are not suppressed at seams.
    # Enforced by the _validate_overlap_window model_validator below.
    mp_overlap_seconds: float = float(os.getenv("MP_OVERLAP_SECONDS", "12.0"))
    mp_max_workers: Optional[int] = None  # None = auto-detect (uses min(CPU count, max_workers_cap))
    mp_max_workers_cap: int = 12  # Maximum number of workers (11 cores + slight oversubscription)
    
    # Model settings - YOLO26 (NMS-free, 43% faster CPU inference)
    # Detection: nano for fast bulk frame processing (~57K frames)
    yolo_weights: str = os.getenv("YOLO_WEIGHTS_PRELOAD", "yolo26n.pt")  # YOLO26 nano for object detection
    yolo_pose_weights: str = os.getenv("YOLO_POSE_WEIGHTS", "yolo26n-pose.pt")  # YOLO26 nano-pose for multi-person pose
    yolo_pose_confidence: float = float(os.getenv("YOLO_POSE_CONFIDENCE", "0.45"))  # Pose detection confidence

    # ROI crop detection: small model for pose-guided crop detection around wrists/hips
    # Uses a stronger model on small crops around keypoints for better recall on
    # small objects (cell phones, cups, bottles, books) from overhead CCTV angles.
    # When empty, ROI detection uses the same model as full-frame detection.
    yolo_roi_weights: str = os.getenv("YOLO_ROI_WEIGHTS", "yolo26s.pt")  # YOLO26 small for ROI crop detection
    yolo_roi_confidence: float = float(os.getenv("YOLO_ROI_CONFIDENCE", "0.15"))  # Lower threshold for small object recall

    # Phase 2: Inference optimization settings
    # CHANGED from 416 to 640 for better accuracy on small objects (cell phones)
    yolo_imgsz: int = int(os.getenv("YOLO_IMGSZ", "640"))  # Model input size (640 for better small object detection)
    yolo_device: str = "cpu"  # Device for YOLO inference (cpu, 0 for GPU)

    # GPU Settings - Enable GPU acceleration for video processing
    gpu_enabled: bool = bool(int(os.getenv("GPU_ENABLED", "1")))  # Enable GPU if available
    gpu_device: str = os.getenv("GPU_DEVICE", "cuda:0")  # CUDA device identifier
    gpu_memory_fraction: float = float(os.getenv("GPU_MEMORY_FRACTION", "0.85"))  # Max GPU memory to use (85%)

    # GPU Batch Processing Settings - maximize GPU utilization
    gpu_batch_size: int = int(os.getenv("GPU_BATCH_SIZE", "8"))  # Frames per GPU batch
    gpu_batch_enabled: bool = bool(int(os.getenv("GPU_BATCH_ENABLED", "1")))  # Enable GPU batch processing

    # Concurrency Settings - Control parallel video processing
    max_concurrent_videos: int = int(os.getenv("MAX_CONCURRENT_VIDEOS", "3"))  # Max videos processed simultaneously
    inference_batch_size: int = int(os.getenv("INFERENCE_BATCH_SIZE", "8"))  # Frames per inference batch
    job_queue_max_size: int = int(os.getenv("JOB_QUEUE_MAX_SIZE", "10"))  # Max pending jobs in queue

    # Memory Management - OOM recovery and CUDA allocator settings
    pytorch_cuda_alloc_conf: str = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    oom_retry_enabled: bool = bool(int(os.getenv("OOM_RETRY_ENABLED", "1")))  # Enable OOM recovery

    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: str = os.getenv("LOG_DIR", "logs")  # Directory for log files
    environment: str = os.getenv("ENVIRONMENT", "development")  # production or development
    prod_log_level: str = os.getenv("PROD_LOG_LEVEL", "INFO")
    dev_log_level: str = os.getenv("DEV_LOG_LEVEL", "DEBUG")
    
    # S3 Upload API settings
    s3_upload_api_url: str = os.getenv(
        "S3_UPLOAD_API_URL",
        "https://api.mindcoinapps.com/ai_demo_api/amazonUpload/uploadWithFolder"
    )

    # External API settings (CVVR API)
    # Use {division} placeholder in URL - will be replaced with actual division value at runtime
    cvvr_api_url: str = os.getenv(
        "CVVR_API_URL",
        "https://api.mindcoinapps.com/ai_{division}_api/cvvr/cvvrTripViolations/addUpdateBulk"
    )
    cvvr_api_url_no_events: str = os.getenv(
        "CVVR_API_URL_NO_EVENTS",
        "https://api.mindcoinapps.com/ai_{division}_api/cvvr/cvvrTripViolations/addUpdateBulkNoEvents"
    )
    cvvr_api_default_division: str = os.getenv("CVVR_API_DEFAULT_DIVISION", "ai_demo_api")
    cvvr_api_token: Optional[str] = os.getenv("CVVR_API_TOKEN", None)
    cvvr_api_timeout: int = int(os.getenv("CVVR_API_TIMEOUT", "30"))
    cvvr_api_enabled: bool = bool(int(os.getenv("CVVR_API_ENABLED", "1")))  # Enable by default
    host_url: str = os.getenv("HOST_URL", "https://celebxmedia.info")  # URL for building fileUrl

    # MinIO settings for video downloads
    # SECURITY: Defaults are intentionally empty strings ("fail-closed"). Real
    # credentials must be supplied via the MINIO_ACCESS_KEY / MINIO_SECRET_KEY
    # environment variables (typically loaded from .env / .env.production).
    # The ``_validate_minio_credentials_in_production`` model_validator below
    # enforces that both values are non-empty whenever ENVIRONMENT=production
    # so a misconfigured deploy fails fast at startup instead of silently
    # falling back to a hardcoded literal.
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "mind.snikbtel.uk:9000")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "")
    minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "")
    minio_secure: bool = bool(int(os.getenv("MINIO_SECURE", "1")))
    minio_bucket: str = os.getenv("MINIO_BUCKET", "cvss")

    # Task 0008 — SSRF defense for the ``videoUrl`` form field on
    # ``/api/video/analyze``. Hostnames in this allowlist are the ONLY
    # hosts that ``validate_external_url`` will accept; everything else
    # (including cloud-metadata IPs, RFC1918, localhost) is rejected
    # with 400. ``MINIO_ALLOWED_HOSTS`` may be set as either a JSON
    # array string (``'["mind.snikbtel.uk", "backup.example.com"]'``) or
    # a plain comma-separated host list
    # (``mind.snikbtel.uk,backup.example.com``). Parsing happens in
    # ``parse_minio_allowed_hosts`` below; bad input fails fast at
    # startup with a clear error rather than crashing at class-definition
    # time as it did when this was inlined as ``json.loads(os.getenv(...))``
    # (reviewer finding H4 — empty string or malformed JSON took the
    # whole process down before pydantic could surface a useful error).
    minio_allowed_hosts: List[str] = ["mind.snikbtel.uk"]
    # Companion to ``MINIO_ALLOWED_HOSTS``. When set, an allowlisted host
    # that resolves to a private/loopback IP (e.g. ``gpu.mindcoinapps.com``
    # → ``10.10.0.2`` on the GPU server's internal NIC) is accepted instead
    # of rejected. Default off; flip on only when the MinIO endpoint is on
    # the same private network as this service.
    minio_allow_private_ips: bool = bool(int(os.getenv("MINIO_ALLOW_PRIVATE_IPS", "0")))
    # Hard cap on the number of bytes ``MinioService.download_video`` will
    # accept from a remote URL. A hostile (or misconfigured) server could
    # return a 100 GB stream and exhaust the GPU box's disk; the streaming
    # downloader aborts and unlinks the partial file once this cap is
    # exceeded. Default 5 GiB matches ``max_upload_size`` for parity
    # between the upload and URL-download paths.
    max_external_download_bytes: int = int(
        os.getenv("MAX_EXTERNAL_DOWNLOAD_BYTES", str(5 * 1024 ** 3))
    )
    
    # Image preprocessing settings (for MediaPipe detection enhancement)
    enable_image_preprocessing: bool = bool(int(os.getenv("ENABLE_IMAGE_PREPROCESSING", "1")))  # Enable by default
    use_clahe: bool = bool(int(os.getenv("USE_CLAHE", "1")))  # CLAHE is most effective
    use_gamma_correction: bool = bool(int(os.getenv("USE_GAMMA_CORRECTION", "1")))
    use_unsharp_masking: bool = bool(int(os.getenv("USE_UNSHARP_MASKING", "0")))  # Optional, can add artifacts
    use_noise_reduction: bool = bool(int(os.getenv("USE_NOISE_REDUCTION", "1")))
    adaptive_preprocessing: bool = bool(int(os.getenv("ADAPTIVE_PREPROCESSING", "1")))  # Use quality metrics
    clahe_clip_limit: float = float(os.getenv("CLAHE_CLIP_LIMIT", "1.5"))  # REDUCED from 2.0 (less aggressive CLAHE)
    
    # Parse ``MINIO_ALLOWED_HOSTS`` (Task 0008, reviewer finding H4).
    # ``pydantic-settings`` will pass the raw env string through here
    # ``mode='before'``; we accept JSON arrays first, fall back to
    # comma-split, and raise a clear ``ValueError`` if both fail so a
    # bad config surfaces at startup with a useful message rather than
    # crashing at class-definition with an unrelated ``json.JSONDecodeError``.
    @field_validator('minio_allowed_hosts', mode='before')
    @classmethod
    def parse_minio_allowed_hosts(cls, v):
        """Parse the allowlist from env (JSON list, comma-split, or default)."""
        # Field default (None / unset / already a Python list).
        if v is None:
            return ["mind.snikbtel.uk"]
        if isinstance(v, list):
            return [str(h).strip() for h in v if str(h).strip()]
        if not isinstance(v, str):
            raise ValueError(
                "MINIO_ALLOWED_HOSTS must be JSON list or comma-separated host list"
            )
        s = v.strip()
        if not s:
            return ["mind.snikbtel.uk"]
        # Try JSON first (the documented form).
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            cleaned = [str(h).strip() for h in parsed if str(h).strip()]
            if not cleaned:
                raise ValueError(
                    "MINIO_ALLOWED_HOSTS must be JSON list or comma-separated host list"
                )
            return cleaned
        # JSON failed (or returned a non-list). Fall back to comma-split.
        cleaned = [h.strip() for h in s.split(",") if h.strip()]
        if cleaned:
            return cleaned
        raise ValueError(
            "MINIO_ALLOWED_HOSTS must be JSON list or comma-separated host list"
        )

    # Parse tile grid size from environment variable (JSON array string)
    @field_validator('clahe_tile_grid_size', mode='before')
    @classmethod
    def parse_tile_grid_size(cls, v):
        """Parse tile grid size from JSON string or return default"""
        if v is None:
            return [8, 8]
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list) and len(parsed) == 2:
                    return [int(parsed[0]), int(parsed[1])]
            except (json.JSONDecodeError, ValueError, TypeError, IndexError):
                pass
            return [8, 8]
        if isinstance(v, list) and len(v) == 2:
            return [int(v[0]), int(v[1])]
        return [8, 8]
    
    clahe_tile_grid_size: List[int] = json.loads(os.getenv("CLAHE_TILE_GRID_SIZE", "[16, 16]"))  # INCREASED from [8,8] (larger tiles, smoother)
    gamma_value: float = float(os.getenv("GAMMA_VALUE", "1.2"))
    unsharp_strength: float = float(os.getenv("UNSHARP_STRENGTH", "1.5"))
    unsharp_radius: int = int(os.getenv("UNSHARP_RADIUS", "1"))
    noise_reduction_kernel: int = int(os.getenv("NOISE_REDUCTION_KERNEL", "3"))

    # Concurrent activity grouping settings
    # Group overlapping activities of different types into combined records with arrays
    concurrent_grouping_enabled: bool = bool(int(os.getenv("CONCURRENT_GROUPING_ENABLED", "1")))

    # Hand Gesture Coordination Session Settings
    # Session-based tracking: session starts on first raise, ends after timeout with no activity
    # Violation only if one person NEVER raised during the entire session
    hand_gesture_session_timeout: float = float(os.getenv("HAND_GESTURE_SESSION_TIMEOUT", "10.0"))

    # Hand gesture coordination temporal window (seconds)
    # Suppress coordination failure alerts if both LP and ALP raised hands within this window
    hand_gesture_coordination_window: float = float(os.getenv("HAND_GESTURE_COORDINATION_WINDOW", "5.0"))

    # Temporal suppression window (seconds)
    # Suppress hand gestures for this duration after detecting a work activity (writing, packing, cell phone)
    temporal_suppression_window: float = float(os.getenv("TEMPORAL_SUPPRESSION_WINDOW", "10.0"))

    # Packing bags verification thresholds (stricter than initial detection)
    # TUNED 2026-01-21: Stricter thresholds to reduce false positives from bags on floor near seated crew
    packing_wrist_visibility_threshold: float = float(os.getenv("PACKING_WRIST_VIS", "0.3"))  # Min wrist visibility 30%
    packing_max_distance_ratio: float = float(os.getenv("PACKING_MAX_DIST_RATIO", "0.45"))  # Wrist within 45% of bag diagonal from center
    packing_min_bag_area: int = int(os.getenv("PACKING_MIN_BAG_AREA", "15000"))  # Min bag area 15,000 sq pixels
    packing_require_wrist_truly_inside: bool = bool(int(os.getenv("PACKING_STRICT_INSIDE", "0")))  # Use margin-based check (not strict)

    # Static backpack suppression — suppress backpacks detected in the same location across many frames
    # A backpack with IoU > threshold appearing for min_frames consecutive frames is classified as a static fixture
    # IoU threshold lowered 2026-04-11 0.80→0.60: train vibration causes small bbox jitter that
    # previously reset the static counter. Observed in run_090144: a bag visible for 6+ min in
    # the same seat position was never marked static because IoU between consecutive frames
    # kept dipping below 0.80.
    packing_static_suppression_enabled: bool = bool(int(os.getenv("PACKING_STATIC_SUPPRESSION_ENABLED", "1")))
    packing_static_iou_threshold: float = float(os.getenv("PACKING_STATIC_IOU_THRESHOLD", "0.60"))
    # min_frames lowered 2026-04-11 from 10→5: multiprocessing chunks are 15s + 12s overlap
    # (~13-14 sampled frames at 0.5fps), so 10 consecutive IoU matches were borderline. 5 frames
    # (~10s) is enough proof of a fixture and fits comfortably within a chunk's tracking window.
    packing_static_min_frames: int = int(os.getenv("PACKING_STATIC_MIN_FRAMES", "5"))

    # Static cell phone suppression — filter out fixed panel instruments misidentified as phones
    # If a "cell phone" bbox stays at the same location (IoU > threshold) for N+ frames, it's a fixture
    phone_static_suppression_enabled: bool = bool(int(os.getenv("PHONE_STATIC_SUPPRESSION_ENABLED", "1")))
    phone_static_iou_threshold: float = float(os.getenv("PHONE_STATIC_IOU_THRESHOLD", "0.70"))
    phone_static_min_frames: int = int(os.getenv("PHONE_STATIC_MIN_FRAMES", "5"))  # ~10s at 0.5fps

    # Wrist motion gate — require wrist movement when detecting packing bags
    # If both wrists are stationary (velocity below threshold), suppress packing detection
    packing_wrist_motion_gate_enabled: bool = bool(int(os.getenv("PACKING_WRIST_MOTION_GATE_ENABLED", "1")))
    packing_wrist_motion_min_velocity: float = float(os.getenv("PACKING_WRIST_MOTION_MIN_VELOCITY", "0.008"))  # Normalized velocity threshold

    # Clip duration settings - Precise clip extraction matching actual activity duration
    clip_buffer_before: float = float(os.getenv("CLIP_BUFFER_BEFORE", "1.0"))  # Seconds before activity start
    clip_buffer_after: float = float(os.getenv("CLIP_BUFFER_AFTER", "1.0"))    # Seconds after activity end

    # Mind Diversion Detection Thresholds
    # Three sub-types: looking_sideways, looking_down_distracted, looking_away_combined
    # TUNED 2026-01-21 v2: Further increased thresholds - camera angle makes forward-looking appear as ~60-70° yaw
    mind_diversion_yaw_sideways: float = float(os.getenv("MIND_DIV_YAW_SIDEWAYS", "78"))  # Head turned > 78° for sideways (was 68°, orig 55°)
    mind_diversion_yaw_combined: float = float(os.getenv("MIND_DIV_YAW_COMBINED", "58"))  # Yaw threshold for combined detection (was 50°, orig 40°)
    mind_diversion_pitch_down: float = float(os.getenv("MIND_DIV_PITCH_DOWN", "45"))  # Head down > 45° for looking_down (was 40°, orig 30°)
    mind_diversion_pitch_combined: float = float(os.getenv("MIND_DIV_PITCH_COMBINED", "35"))  # Pitch threshold for combined (was 28°, orig 20°)
    mind_diversion_yaw_max_for_down: float = float(os.getenv("MIND_DIV_YAW_MAX_DOWN", "55"))  # Max yaw for pure looking_down (was 50°, orig 40°)

    # Forward-looking exemption: Camera is behind-right of crew, so:
    # - Negative yaw = looking LEFT toward track/window (LEGITIMATE WORK)
    # - Positive yaw = looking RIGHT away from track (POTENTIAL DIVERSION)
    # When enabled, only positive yaw triggers mind diversion (exempts looking at track)
    mind_diversion_exempt_forward_looking: bool = os.getenv("MIND_DIV_EXEMPT_FORWARD", "true").lower() == "true"

    # Sleep Detection - Reclined Posture Settings (overhead/behind camera angles)
    # When crew sleeps by leaning BACK against wall, torso elongates and nose moves higher in frame
    sleep_reclined_torso_height_threshold: float = float(os.getenv("SLEEP_TORSO_HEIGHT_THRESH", "175"))  # px, torso > this = reclined
    # NOTE: nose_y_norm threshold moved to sleep_nose_y_norm_threshold (tuned to 0.30) in the scoring section below
    sleep_reclined_shoulder_width_threshold: float = float(os.getenv("SLEEP_SHOULDER_WIDTH_THRESH", "60"))  # px, shoulders < this = compressed (reclined)

    # Pose-based sleep scoring thresholds (tuned for side/overhead camera angles)
    sleep_nose_above_shoulders_threshold: float = float(os.getenv("SLEEP_NOSE_ABOVE_SHOULDERS_THRESH", "0.08"))
    sleep_nose_below_px_threshold: float = float(os.getenv("SLEEP_NOSE_BELOW_PX_THRESH", "-55"))
    sleep_head_tilt_threshold: float = float(os.getenv("SLEEP_HEAD_TILT_THRESH", "-155"))
    sleep_nose_y_norm_threshold: float = float(os.getenv("SLEEP_NOSE_Y_NORM_THRESH", "0.30"))
    sleep_score_threshold: int = int(os.getenv("SLEEP_SCORE_THRESH", "5"))

    # Baseline calibration for camera-angle adaptation
    sleep_baseline_enabled: bool = os.getenv("SLEEP_BASELINE_ENABLED", "true").lower() == "true"
    sleep_baseline_calibration_window: float = float(os.getenv("SLEEP_BASELINE_WINDOW", "5.0"))  # Raised from 2.0s for more stable baselines
    sleep_baseline_min_samples: int = int(os.getenv("SLEEP_BASELINE_MIN_SAMPLES", "3"))  # Raised from 1 for more reliable baselines

    # Head drop consecutive check — require N consecutive head_drop=True frames to confirm
    # Filters out single-frame pose estimation noise that causes false microsleep triggers
    sleep_head_drop_min_consecutive: int = int(os.getenv("SLEEP_HEAD_DROP_MIN_CONSECUTIVE", "2"))

    # Delta-from-baseline thresholds
    sleep_baseline_nose_below_delta: float = float(os.getenv("SLEEP_BASELINE_NOSE_BELOW_DELTA", "40"))
    sleep_baseline_head_tilt_delta: float = float(os.getenv("SLEEP_BASELINE_HEAD_TILT_DELTA", "25"))
    sleep_baseline_torso_height_delta: float = float(os.getenv("SLEEP_BASELINE_TORSO_DELTA", "40"))
    sleep_baseline_shoulder_width_delta: float = float(os.getenv("SLEEP_BASELINE_SHOULDER_DELTA", "20"))

    # New discriminating signals for sleep detection
    sleep_sustained_stillness_threshold: float = float(os.getenv("SLEEP_SUSTAINED_STILLNESS_THRESH", "0.03"))
    sleep_sustained_stillness_frames: int = int(os.getenv("SLEEP_SUSTAINED_STILLNESS_FRAMES", "1"))
    sleep_hands_clasped_threshold: float = float(os.getenv("SLEEP_HANDS_CLASPED_THRESH", "100"))
    sleep_hands_clasped_frames: int = int(os.getenv("SLEEP_HANDS_CLASPED_FRAMES", "1"))
    sleep_sustained_low_eye_frames: int = int(os.getenv("SLEEP_SUSTAINED_LOW_EYE_FRAMES", "2"))
    sleep_hands_spread_threshold: float = float(os.getenv("SLEEP_HANDS_SPREAD_THRESHOLD", "180"))

    # Head Bob Detection (slow drift + corrective jerk)
    sleep_head_bob_drift_max_rate: float = float(os.getenv("SLEEP_HEAD_BOB_DRIFT_MAX_RATE", "15.0"))
    sleep_head_bob_jerk_min_rate: float = float(os.getenv("SLEEP_HEAD_BOB_JERK_MIN_RATE", "20.0"))
    sleep_head_bob_min_drift_frames: int = int(os.getenv("SLEEP_HEAD_BOB_MIN_DRIFT_FRAMES", "2"))
    sleep_head_bob_min_amplitude: float = float(os.getenv("SLEEP_HEAD_BOB_MIN_AMPLITUDE", "10.0"))
    sleep_head_bob_score_bonus: int = int(os.getenv("SLEEP_HEAD_BOB_SCORE_BONUS", "2"))
    sleep_head_bob_bypass_eye_gate: bool = os.getenv("SLEEP_HEAD_BOB_BYPASS_EYE_GATE", "true").lower() == "true"

    # Wrist Velocity Tracking (still vs active hands)
    sleep_wrist_velocity_still_threshold: float = float(os.getenv("SLEEP_WRIST_VEL_STILL", "0.005"))
    sleep_wrist_velocity_active_threshold: float = float(os.getenv("SLEEP_WRIST_VEL_ACTIVE", "0.03"))
    sleep_wrist_velocity_still_frames: int = int(os.getenv("SLEEP_WRIST_VEL_STILL_FRAMES", "2"))

    # Temporal State Machine
    sleep_state_machine_enabled: bool = os.getenv("SLEEP_STATE_MACHINE_ENABLED", "true").lower() == "true"
    sleep_state_hand_activity_threshold: float = float(os.getenv("SLEEP_STATE_HAND_ACTIVITY", "0.02"))
    sleep_state_drowsy_to_microsleep_sec: float = float(os.getenv("SLEEP_DROWSY_TO_MICROSLEEP_SEC", "2.0"))
    sleep_state_microsleep_to_sleep_sec: float = float(os.getenv("SLEEP_MICROSLEEP_TO_SLEEP_SEC", "4.0"))

    # Shoulder Slump Rate (progressive downward drift)
    sleep_shoulder_slump_rate_threshold: float = float(os.getenv("SLEEP_SHOULDER_SLUMP_RATE", "0.005"))
    sleep_shoulder_slump_min_frames: int = int(os.getenv("SLEEP_SHOULDER_SLUMP_MIN_FRAMES", "3"))
    # Face-gone threshold: eye_vis below this = face not visible (side/behind camera)
    sleep_face_gone_threshold: float = float(os.getenv("SLEEP_FACE_GONE_THRESHOLD", "0.25"))

    # Head drop detection thresholds (primary microsleep signal)
    sleep_nose_y_drop_threshold: float = float(os.getenv("SLEEP_NOSE_Y_DROP_THRESHOLD", "0.15"))
    sleep_head_tilt_drop_threshold: float = float(os.getenv("SLEEP_HEAD_TILT_DROP_THRESHOLD", "30.0"))

    eyes_not_in_frame_threshold: float = float(os.getenv("EYES_NOT_IN_FRAME_THRESHOLD", "0.15"))
    sleep_overhead_nose_y_threshold: float = float(os.getenv("SLEEP_OVERHEAD_NOSE_Y_THRESHOLD", "0.10"))

    # Haar Cascade Eye Closure Detection
    haar_eye_detection_enabled: bool = bool(int(os.getenv("HAAR_EYE_DETECTION_ENABLED", "1")))
    haar_eye_closed_consecutive_frames: int = int(os.getenv("HAAR_EYE_CLOSED_FRAMES", "3"))
    haar_eye_roi_padding: float = float(os.getenv("HAAR_EYE_ROI_PADDING", "0.4"))
    haar_eye_scale_factor: float = float(os.getenv("HAAR_EYE_SCALE_FACTOR", "1.1"))
    haar_eye_min_neighbors: int = int(os.getenv("HAAR_EYE_MIN_NEIGHBORS", "3"))
    haar_eye_microsleep_duration: float = float(os.getenv("HAAR_EYE_MICROSLEEP_DURATION", "5.0"))  # Spec: eyes closed > 5 sec
    haar_eye_sleep_duration: float = float(os.getenv("HAAR_EYE_SLEEP_DURATION", "4.0"))
    haar_eye_score_boost: int = int(os.getenv("HAAR_EYE_SCORE_BOOST", "5"))

    # Lower YOLO pose confidence for sleep analysis (sleeping persons have low YOLO confidence)
    yolo_pose_sleep_confidence: float = float(os.getenv("YOLO_POSE_SLEEP_CONFIDENCE", "0.30"))

    # IR/dark frame preprocessing for YOLO detection
    yolo_dark_frame_brightness_threshold: float = float(os.getenv("YOLO_DARK_BRIGHTNESS_THRESH", "0.4"))

    # No-pose sleep detection (for IR mode where YOLO pose fails)
    sleep_no_pose_enabled: bool = bool(int(os.getenv("SLEEP_NO_POSE_ENABLED", "1")))
    sleep_no_pose_min_duration: float = float(os.getenv("SLEEP_NO_POSE_MIN_DURATION", "30.0"))  # Seconds of stable no-pose person before flagging sleep
    sleep_no_pose_bbox_stability_threshold: float = float(os.getenv("SLEEP_NO_POSE_BBOX_STABILITY", "0.15"))  # IoU change threshold for bbox stability

    # IR Forward-Lean Sleep Detection (body-only keypoints for dark/IR frames)
    ir_forward_lean_enabled: bool = bool(int(os.getenv("IR_FORWARD_LEAN_ENABLED", "1")))
    ir_forward_lean_min_body_keypoints: int = int(os.getenv("IR_FORWARD_LEAN_MIN_BODY_KPS", "3"))
    ir_forward_lean_head_vis_threshold: float = float(os.getenv("IR_FORWARD_LEAN_HEAD_VIS", "0.15"))
    ir_forward_lean_body_vis_threshold: float = float(os.getenv("IR_FORWARD_LEAN_BODY_VIS", "0.2"))
    ir_forward_lean_score_threshold: int = int(os.getenv("IR_FORWARD_LEAN_SCORE_THRESH", "4"))
    ir_forward_lean_min_duration: float = float(os.getenv("IR_FORWARD_LEAN_MIN_DURATION", "5.0"))  # Microsleep threshold (seconds)
    ir_forward_lean_sleep_duration: float = float(os.getenv("IR_FORWARD_LEAN_SLEEP_DURATION", "10.0"))  # Sleep threshold (seconds)

    # IR-adjusted no-pose bbox stability duration (applies to existing no-pose sleep detector in dark frames)
    ir_sleep_no_pose_min_duration: float = float(os.getenv("IR_SLEEP_NO_POSE_MIN_DURATION", "15.0"))  # Reduced from 30s for IR dark frames

    # Eating/Drinking Detection (mind diversion sub-type)
    eating_drinking_detection_enabled: bool = bool(int(os.getenv("EATING_DRINKING_ENABLED", "1")))
    eating_drinking_cup_confidence: float = float(os.getenv("EATING_DRINKING_CUP_CONF", "0.25"))  # Lower threshold for cups in IR
    eating_drinking_hand_face_margin: int = int(os.getenv("EATING_DRINKING_HAND_FACE_MARGIN", "80"))  # Wrist within 80px of nose height
    eating_drinking_hand_object_margin: int = int(os.getenv("EATING_DRINKING_HAND_OBJ_MARGIN", "150"))  # Hand-to-cup proximity
    eating_drinking_cup_floor_confidence: float = float(os.getenv("EATING_DRINKING_CUP_FLOOR_CONF", "0.20"))  # Pre-filter floor for cup/bottle in full-frame detection

    # Mind Diversion Suppression Settings
    # Suppress false positives when LP is doing legitimate document work
    mind_diversion_suppress_with_writing: bool = os.getenv("MIND_DIV_SUPPRESS_WRITING", "true").lower() == "true"
    mind_diversion_writing_grace_seconds: float = float(os.getenv("MIND_DIV_WRITING_GRACE", "5.0"))
    mind_diversion_wrist_distance_threshold: float = float(os.getenv("MIND_DIV_WRIST_DIST", "200"))  # Max wrist distance for writing pose

    # Job Queue Settings - Async video processing queue management
    # Note: max_concurrent_videos and job_queue_max_size defined in GPU Settings section above
    job_queue_num_workers: int = int(os.getenv("JOB_QUEUE_NUM_WORKERS", "3"))  # Number of worker tasks processing jobs

    # GPU Memory Management Settings
    gpu_memory_warning_threshold: float = float(os.getenv("GPU_MEMORY_WARNING_THRESHOLD", "80.0"))  # Percentage

    # ==========================================
    # YOLO Confidence Thresholds
    # ==========================================
    # Lowered 2026-04-11 from 0.5 → 0.40. v5_probe across 4 videos showed wide variation:
    #   all_activities  p10=0.792  (bright cabin, close view)    → 0.5 threshold fine
    #   ch01            p10=0.880  (bright cabin, close view)    → 0.5 threshold fine
    #   n_5             p10=0.376  (distant/occluded persons)    → 0.5 cut 10-20% of reals
    #   TR_1            p10=0.140  (very distant/dim cabin)      → 0.5 cut 30-40% of reals
    # 0.40 preserves the high-conf videos unchanged while recovering distant persons in
    # wide/dim cabins. If phantom detections appear, set yolo_person_min_area > 0 to filter
    # by bbox size as a second-line guard.
    yolo_person_confidence: float = float(os.getenv("YOLO_PERSON_CONFIDENCE", "0.40"))
    # Minimum bbox area (pixels) for a detection to count as a real person.
    # Filters phantom person FPs (chair backs, shadows, equipment) that pass
    # confidence but are far smaller than any real adult in overhead cabin CCTV.
    # Set to 0 to disable. Recommended ~65000 for 1080p locopilot cabin footage.
    yolo_person_min_area: int = int(os.getenv("YOLO_PERSON_MIN_AREA", "0"))
    # Lowered 2026-04-11 to 0.25 after visual review across 4 videos.
    # Earlier 0.70 change (made from stats only) was WRONG: visual review of annotated
    # frames showed real backpacks in TR_1 at conf 0.11-0.15 and in ch01 at conf 0.59
    # that 0.70 would reject. There is no confidence threshold that separates real bags
    # from static fixtures cleanly across videos — the signal is *context* (stationary
    # vs. being actively packed), not confidence. Rule-layer fixes (static suppression
    # in locopilot_monitor.py:_update_static_backpack_tracking + strict AND-gate motion
    # check on primary wrist-inside path) handle FP discrimination instead. 0.25 is a
    # pragmatic noise floor — below probe-observed FP cluster mean (0.301) while above
    # the lowest-value noise.
    yolo_bag_confidence: float = float(os.getenv("YOLO_BAG_CONFIDENCE", "0.25"))
    yolo_bag_log_confidence: float = float(os.getenv("YOLO_BAG_LOG_CONFIDENCE", "0.25"))
    yolo_book_confidence: float = float(os.getenv("YOLO_BOOK_CONFIDENCE", "0.4"))
    yolo_cell_phone_confidence: float = float(os.getenv("YOLO_CELL_PHONE_CONFIDENCE", "0.3"))

    # Cell phone detection confidence threshold (activity-level, distinct from YOLO detection threshold)
    cell_phone_confidence: float = float(os.getenv("CELL_PHONE_CONFIDENCE", "0.40"))

    # Pose-based phone-to-ear fallback: fires cell_phone when a wrist is
    # sustained close to an ear keypoint, even if YOLO didn't detect the phone
    # (common when hand occludes the phone at the ear). Tight distance gate
    # (< 20% of bbox height) keeps it off face-touching / head-scratching.
    # Still routes through the per-person temporal filter (≥2 consecutive samples).
    cell_phone_pose_fallback_enabled: bool = bool(int(os.getenv("CELL_PHONE_POSE_FALLBACK_ENABLED", "1")))

    # ==========================================
    # Wrist/Elbow Detection Thresholds
    # ==========================================
    max_wrist_distance: int = int(os.getenv("MAX_WRIST_DISTANCE", "300"))
    max_elbow_distance: int = int(os.getenv("MAX_ELBOW_DISTANCE", "450"))
    max_single_wrist_distance: int = int(os.getenv("MAX_SINGLE_WRIST_DISTANCE", "250"))
    writing_wrist_distance: int = int(os.getenv("WRITING_WRIST_DISTANCE", "300"))
    relaxed_wrist_distance: int = int(os.getenv("RELAXED_WRIST_DISTANCE", "400"))
    elbow_visibility_threshold: float = float(os.getenv("ELBOW_VISIBILITY_THRESHOLD", "0.25"))
    wrist_visibility_threshold: float = float(os.getenv("WRIST_VISIBILITY_THRESHOLD", "0.3"))

    # ==========================================
    # Writing Detection Thresholds
    # ==========================================
    writing_min_duration: float = float(os.getenv("WRITING_MIN_DURATION", "1.0"))
    writing_required_consecutive: int = int(os.getenv("WRITING_REQUIRED_CONSECUTIVE", "2"))
    # Relaxed-rule slack: when one wrist is inside the book bbox, the other
    # wrist may be up to this many pixels from the nearest bbox edge and still
    # count as "writing". Calibrated 2026-05-06 from TV22 GT diagnostic logs
    # showing 4-44 px gaps on the off-hand during real writing posture.
    # Set to 0 to fall back to the original strict "BOTH wrists inside" rule.
    writing_other_wrist_max_dist: int = int(os.getenv("WRITING_OTHER_WRIST_MAX_DIST", "50"))
    # Lower bound on pose-keypoint visibility for the dual-wrist rule. The pose
    # model can return high-confidence (>0.95) right-wrist + near-zero (~0.03)
    # left-wrist on real writing frames where the writing hand occludes the
    # other one. Lower this from 0.5 to 0.3 so frames with one moderate-vis
    # wrist still enter the rule.
    writing_min_wrist_visibility: float = float(os.getenv("WRITING_MIN_WRIST_VIS", "0.3"))
    # Single-wrist fallback: when only one wrist clears 0.5 visibility, fire
    # writing if that wrist is fully inside the book bbox. Captures occluded-
    # wrist GTs (TV22.5 4:47, TV22.7 9:32) without the edge-distance slack
    # used in the dual-wrist relaxed path. Set to 0 to disable.
    writing_allow_single_wrist: bool = bool(int(os.getenv("WRITING_ALLOW_SINGLE_WRIST", "1")))
    # Log-book ROI mask: only fire writing if the book bbox centre falls inside
    # this normalised rectangle. Drops control-panel-device-misclassified-as-
    # book FPs and books detected in the upper window/door area. Format:
    # ``WRITING_BOOK_ROI=x1,y1,x2,y2`` (each in [0,1]). Empty (default)
    # disables the mask. For the TV22 overhead camera, the desk-and-lap zone
    # is roughly ``0.15,0.30,0.75,0.95``.
    writing_book_roi: str = os.getenv("WRITING_BOOK_ROI", "")
    book_posture_min_duration: float = float(os.getenv("BOOK_POSTURE_MIN_DURATION", "2.0"))
    book_posture_required_consecutive: int = int(os.getenv("BOOK_POSTURE_REQUIRED_CONSECUTIVE", "2"))

    # ==========================================
    # Head Tilt / Sleep Detection Thresholds
    # ==========================================
    head_down_threshold: float = float(os.getenv("HEAD_DOWN_THRESHOLD", "0.05"))
    sleep_strong_score: int = int(os.getenv("SLEEP_STRONG_SCORE", "6"))
    sleep_strong_duration: int = int(os.getenv("SLEEP_STRONG_DURATION", "0"))
    sleep_moderate_duration: int = int(os.getenv("SLEEP_MODERATE_DURATION", "2"))
    sleep_microsleep_duration: int = int(os.getenv("SLEEP_MICROSLEEP_DURATION", "5"))  # Spec: eyes closed > 5 sec
    minimal_movement_threshold: float = float(os.getenv("MINIMAL_MOVEMENT_THRESHOLD", "0.15"))
    stable_posture_variance: int = int(os.getenv("STABLE_POSTURE_VARIANCE", "100"))
    eyes_not_visible_threshold: float = float(os.getenv("EYES_NOT_VISIBLE_THRESHOLD", "0.4"))

    # ==========================================
    # IR Forward Lean Detection Thresholds
    # ==========================================
    ir_shoulder_relative_threshold: float = float(os.getenv("IR_SHOULDER_RELATIVE_THRESHOLD", "0.4"))
    ir_bbox_aspect_ratio_threshold: float = float(os.getenv("IR_BBOX_ASPECT_RATIO_THRESHOLD", "1.2"))
    ir_low_movement_threshold: float = float(os.getenv("IR_LOW_MOVEMENT_THRESHOLD", "0.02"))
    sub_threshold_streak_limit: int = int(os.getenv("SUB_THRESHOLD_STREAK_LIMIT", "3"))

    # ==========================================
    # Object Detection Geometry
    # ==========================================
    bag_max_aspect_ratio: float = float(os.getenv("BAG_MAX_ASPECT_RATIO", "1.5"))  # L-08: Relaxed from 1.2 to accept more legitimate bags
    bag_min_area: int = int(os.getenv("BAG_MIN_AREA", "5000"))
    bag_max_area: int = int(os.getenv("BAG_MAX_AREA", "100000"))
    book_person_margin: int = int(os.getenv("BOOK_PERSON_MARGIN", "150"))
    person_book_overlap_margin: int = int(os.getenv("PERSON_BOOK_OVERLAP_MARGIN", "250"))

    # ==========================================
    # Pose Validation
    # ==========================================
    min_pose_landmarks: int = int(os.getenv("MIN_POSE_LANDMARKS", "10"))
    min_pose_visibility: float = float(os.getenv("MIN_POSE_VISIBILITY", "0.3"))
    face_mesh_detection_confidence: float = float(os.getenv("FACE_MESH_DETECTION_CONFIDENCE", "0.5"))
    face_mesh_tracking_confidence: float = float(os.getenv("FACE_MESH_TRACKING_CONFIDENCE", "0.5"))

    # ==========================================
    # Activity Registry Defaults (margins/regions)
    # ==========================================
    activity_cell_phone_margin: int = int(os.getenv("ACTIVITY_CELL_PHONE_MARGIN", "180"))
    activity_writing_margin: int = int(os.getenv("ACTIVITY_WRITING_MARGIN", "180"))
    activity_packing_margin: int = int(os.getenv("ACTIVITY_PACKING_MARGIN", "100"))
    activity_packing_region_margin: int = int(os.getenv("ACTIVITY_PACKING_REGION_MARGIN", "150"))
    activity_packing_wrist_inside_margin: int = int(os.getenv("ACTIVITY_PACKING_WRIST_INSIDE_MARGIN", "80"))

    # ==========================================
    # Train Motion Rules Settings
    # ==========================================
    # When True (default), activities listed in
    # ``app/core/gates.py:DEFAULT_SUPPRESSED_WHEN_STOPPED`` are zeroed out
    # while the train is STOPPED, so they never reach the VLM verifier or
    # the external API. Set to False (env: TRAIN_MOTION_SUPPRESS_WHEN_STOPPED=0)
    # to forward all detections through regardless of motion state — they
    # then carry ``motionState=STOPPED`` in activities.json and the posted
    # payload, so downstream consumers can distinguish station-context
    # events from running-train violations.
    train_motion_suppress_when_stopped: bool = bool(int(os.getenv("TRAIN_MOTION_SUPPRESS_WHEN_STOPPED", "1")))
    # Comma-separated activity ``objectType`` names that bypass the downstream
    # STOPPED motion filter when posting to the external API. Per CLAUDE.md:
    # "microsleep + cell_phone are never suppressed (safety-critical even at
    # stations)". Pipeline-1's ``apply_train_stopped_suppression`` already
    # leaves these active, but the API-post motion filter (in
    # ``video_processing_service.py`` and ``video_controller.py``) used to
    # strip ALL STOPPED activities — including these — silently. This setting
    # restores the spec'd behaviour. Names are normalised (lowercase, spaces
    # to underscores), so both "cell_phone" and "cell phone" match.
    motion_filter_bypass_types: str = os.getenv(
        "MOTION_FILTER_BYPASS_TYPES", "cell_phone,microsleep"
    )
    # Comma-separated override of the activities suppressed when train is STOPPED.
    # Empty (default) means use ``DEFAULT_SUPPRESSED_WHEN_STOPPED`` from
    # ``app/core/gates.py`` (sleep, writing, packing_bags, lp_hand_gesture,
    # alp_hand_gesture, mind_diversion, eating_drinking). Set to a smaller list
    # to let some activities through even at stations — e.g.
    # ``TRAIN_MOTION_STOPPED_SUPPRESS_LIST=sleep,packing_bags,lp_hand_gesture,alp_hand_gesture,mind_diversion,eating_drinking``
    # excludes ``writing`` so log-book writing is reported regardless of motion
    # state. Names must match registry keys exactly.
    train_motion_stopped_suppress_list: str = os.getenv("TRAIN_MOTION_STOPPED_SUPPRESS_LIST", "")

    # Train Motion Detection (vibration-based)
    train_motion_detection_enabled: bool = bool(int(os.getenv("TRAIN_MOTION_DETECTION_ENABLED", "0")))
    train_motion_vibration_threshold: float = float(os.getenv("TRAIN_MOTION_VIB_THRESHOLD", "1.0"))
    train_motion_vibration_high: float = float(os.getenv("TRAIN_MOTION_VIB_HIGH", "3.0"))
    train_motion_window_roi_lp: str = os.getenv("TRAIN_MOTION_WINDOW_ROI_LP", "0.0,0.05,0.12,0.85")
    train_motion_window_roi_alp: str = os.getenv("TRAIN_MOTION_WINDOW_ROI_ALP", "0.88,0.05,1.0,0.85")
    train_motion_running_threshold: float = float(os.getenv("TRAIN_MOTION_RUNNING_THRESHOLD", "0.45"))
    train_motion_temporal_window: int = int(os.getenv("TRAIN_MOTION_TEMPORAL_WINDOW", "5"))
    train_motion_stopped_group_threshold: int = int(os.getenv("TRAIN_MOTION_STOPPED_GROUP_THRESHOLD", "5"))
    # Threshold for group_detected. Default 5 (i.e. >5 → 6+ persons required).
    # Revised 2026-04-22: spec changed from "more than 2" to "more than 5" —
    # 3-person supervisor visits are expected and should no longer trigger.
    train_motion_running_group_threshold: int = int(os.getenv("TRAIN_MOTION_RUNNING_GROUP_THRESHOLD", "5"))
    train_motion_window_flow_threshold: float = float(os.getenv("TRAIN_MOTION_WINDOW_FLOW_THRESHOLD", "2.0"))
    train_motion_weight_vibration: float = float(os.getenv("TRAIN_MOTION_WEIGHT_VIBRATION", "0.5"))
    train_motion_weight_window: float = float(os.getenv("TRAIN_MOTION_WEIGHT_WINDOW", "0.3"))
    train_motion_weight_stability: float = float(os.getenv("TRAIN_MOTION_WEIGHT_STABILITY", "0.2"))
    train_motion_person_mask_padding: float = float(os.getenv("TRAIN_MOTION_PERSON_MASK_PADDING", "0.10"))
    # Extra normalized ROIs to mask from interior (semicolon-separated list of
    # "x1,y1,x2,y2"). Used for secondary scenery regions like a cab doorway that
    # are visible from the same camera but aren't the primary window strip.
    train_motion_extra_mask_rois: str = os.getenv("TRAIN_MOTION_EXTRA_MASK_ROIS", "")
    # Percentile cut for trimmed-mean vibration. Pixels with diff above this
    # percentile are dropped before averaging (hotspot suppression). Default 90
    # drops the top 10%. Lower values → more aggressive trimming.
    train_motion_vibration_trim_percentile: float = float(os.getenv("TRAIN_MOTION_VIB_TRIM_PERCENTILE", "90.0"))
    # Rolling window (# of frames) for temporal median smoothing of vibration_mean
    # before scoring. Only useful for isolated 1-frame spikes; defaults to 1
    # (no-op) because sustained (3+ frame) person-motion bursts aren't filtered
    # by a small median and a large median lags real stop transitions. Tune if
    # you observe isolated single-frame FP spikes.
    train_motion_vibration_median_window: int = int(os.getenv("TRAIN_MOTION_VIB_MEDIAN_WINDOW", "1"))
    # Number of prior frames whose person bboxes get unioned into the interior
    # mask (in addition to the current frame). At low sample FPS a walking
    # person crosses many pixels per sample, so 1-frame prev-mask isn't enough.
    # Default 2 (union last 2 frames + current). Set to 0 to use current only.
    train_motion_person_bbox_history: int = int(os.getenv("TRAIN_MOTION_PERSON_BBOX_HISTORY", "2"))
    # Cold-start guard for the multiprocessing chunk-boundary case. Each worker
    # creates a fresh TrainMotionDetector with empty state buffers, so the
    # first few frames of every chunk lack the temporal smoothing that catches
    # 1-2 frame vibration spikes from person motion (writing/packing seated).
    # When True (default) and the rolling vib history is shorter than
    # vibration_median_window, we require the side-window optical-flow signal
    # to be elevated before committing RAW=RUNNING; otherwise we demote to
    # STOPPED. Trade-off: ~5-10s of false-STOPPED at the start of a video that
    # actually opens with the train running, in exchange for eliminating
    # station-context FPs that leak past the gate at chunk boundaries.
    train_motion_cold_start_require_window_flow: bool = bool(int(
        os.getenv("TRAIN_MOTION_COLD_START_REQUIRE_WINDOW_FLOW", "1")
    ))
    # How many frames at the start of each per-worker detector instance the
    # cold-start guard applies to. At sample_fps=0.5 each frame is 2s, so the
    # default 5 covers the first 10s — enough to pass the chunk-overlap region
    # (5s) plus a margin for the temporal smoother to acquire history.
    train_motion_cold_start_frames: int = int(os.getenv("TRAIN_MOTION_COLD_START_FRAMES", "5"))

    # Suppress no_person_detected when trip schedule is unavailable
    # (cannot distinguish station halts from running without schedule)
    suppress_no_person_without_schedule: bool = bool(int(os.getenv("SUPPRESS_NO_PERSON_WITHOUT_SCHEDULE", "1")))

    # OCR Timestamp Extraction Settings
    ocr_enabled: bool = bool(int(os.getenv("OCR_ENABLED", "0")))
    ocr_engine: str = os.getenv("OCR_ENGINE", "auto")  # 'easyocr' (recommended), 'tesseract', or 'auto'
    ocr_roi_position: str = os.getenv("OCR_ROI_POSITION", "top-left")  # top-right, top-left, bottom-right, bottom-left
    ocr_roi_x: int = int(os.getenv("OCR_ROI_X", "10"))  # X offset from edge
    ocr_roi_y: int = int(os.getenv("OCR_ROI_Y", "10"))  # Y offset from edge
    ocr_roi_width: int = int(os.getenv("OCR_ROI_WIDTH", "200"))  # ROI width
    ocr_roi_height: int = int(os.getenv("OCR_ROI_HEIGHT", "50"))  # ROI height

    # Pre-Arrival ALP Alertness Settings
    pre_arrival_window_start: int = int(os.getenv("PRE_ARRIVAL_WINDOW_START", "60"))  # 60s before arrival
    pre_arrival_window_end: int = int(os.getenv("PRE_ARRIVAL_WINDOW_END", "30"))  # 30s before arrival

    # Halt Grace Period - allow exemptions for short time after scheduled departure
    halt_grace_period: int = int(os.getenv("HALT_GRACE_PERIOD", "120"))  # 120s after departure

    # Cache TTL for delay data (in seconds, default 30 minutes)
    etrain_cache_ttl: int = int(os.getenv("ETRAIN_CACHE_TTL", "1800"))

    # ==========================================
    # VLM Verification Layer (Pipeline-2 FP filter)
    # ==========================================
    # Post-Pipeline-1 verification using a vision-language model (Qwen2.5-VL).
    # The verifier sees the activity keyframe + an activity-specific prompt and
    # returns TRUE_POSITIVE / FALSE_POSITIVE / UNCERTAIN. Activities with
    # FALSE_POSITIVE @ confidence>=VLM_DROP_THRESHOLD are filtered out before
    # S3 upload + external API push. To disable enforcement without disabling
    # the verifier, set VLM_DROP_THRESHOLD=2.0 (impossible threshold, keeps
    # vlm_review annotations on every activity). Designed to fail-open: if the
    # vLLM endpoint is unreachable, the Pipeline-1 verdict passes through.
    vlm_verification_enabled: bool = bool(int(os.getenv("VLM_VERIFICATION_ENABLED", "0")))
    vlm_base_url: str = os.getenv("VLM_BASE_URL", "http://localhost:8001/v1")
    vlm_model: str = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
    # Comma-separated activity names (matches ACTIVITY_REGISTRY keys) to verify.
    # Activities not listed are passed through unchanged.
    vlm_verify_activities: str = os.getenv("VLM_VERIFY_ACTIVITIES", "writing,eating_drinking")
    # Minimum VLM confidence required to drop a Pipeline-1 detection. Set to a
    # value > 1.0 to disable dropping while still recording verdicts.
    vlm_drop_threshold: float = float(os.getenv("VLM_DROP_THRESHOLD", "0.80"))
    # HTTP timeout per VLM call. Verifier is fail-open on timeout.
    vlm_timeout_seconds: float = float(os.getenv("VLM_TIMEOUT_SECONDS", "8.0"))
    # Max activities verified per request. 0 = no cap.
    vlm_max_activities_per_run: int = int(os.getenv("VLM_MAX_ACTIVITIES_PER_RUN", "0"))
    # When 1, the VLM's `train_appears_to_be` observation can override Pipeline-1's
    # motionState=RUNNING → STOPPED if the VLM reports a hard visual stopped cue
    # (cabin door open, platform/station visible). Downstream STOPPED-filter then
    # drops the activity from the external API post. RUNNING-direction overrides
    # are NOT applied (those would require deferring gates.apply_train_stopped_
    # suppression; tracked separately as Direction B). Default 0 (opt-in).
    vlm_motion_override_enabled: bool = bool(int(os.getenv("VLM_MOTION_OVERRIDE_ENABLED", "0")))
    # When 1, the verifier records VLM verdicts on every activity but never drops
    # any detection (observe-only / shadow-mode rollout). Default 0 = enforcement.
    vlm_shadow_mode: bool = bool(int(os.getenv("VLM_SHADOW_MODE", "0")))
    # Target number of frames per VLM verification strip. Single-burst
    # activities have ``_resolve_keyframes`` return 1 frame; the verifier
    # supplements with frames sampled from ``activityClip`` to reach this
    # target so the VLM gets temporal evidence even on short detections.
    # Cap is 5 (matches ``_stitch_keyframes`` slice).
    vlm_strip_target_frames: int = int(os.getenv("VLM_STRIP_TARGET_FRAMES", "5"))
    # Pre-VLM no-subject gate: when 1, drop activity candidates whose
    # keyframes contain no Pipeline-1 person bbox in any frame, before
    # spending a VLM call. Catches the empty-cabin hallucination archetype
    # observed on 2026-05-08 (run_20260508_182809) where the VLM confidently
    # confabulated "hand on open book, pen in hand" on frames with no person
    # at all. Skipped for ``no_person_detected`` activity type, where an
    # empty cabin is the violation. Default 1 (enforce).
    vlm_pre_gate_enabled: bool = bool(int(os.getenv("VLM_PRE_GATE_ENABLED", "1")))
    # Minimum green-bbox pixel area to count as a person in the pre-VLM
    # gate. Smaller values risk false-positive person detections (text
    # labels, skeleton lines), larger values may miss small/distant LP
    # bboxes. 1000px = ~32x32 — well above text label noise.
    vlm_pre_gate_min_person_area: int = int(os.getenv("VLM_PRE_GATE_MIN_PERSON_AREA", "1000"))
    # Post-VLM structured-field consistency check: when 1, demote a VLM
    # ``TRUE_POSITIVE`` verdict to ``UNCERTAIN`` (capped confidence 0.5)
    # if the activity-specific structured fields contradict the verdict
    # (e.g. writing TP but ``hand_actually_on_book=false``, cell_phone TP
    # but ``object_in_hand="radio_handset"``). Catches the
    # cooperatively-filled-schema-with-wrong-verdict failure mode where
    # the model fills observation fields correctly but emits the wrong
    # overall label. Default 1 (enforce).
    vlm_consistency_check_enabled: bool = bool(int(os.getenv("VLM_CONSISTENCY_CHECK_ENABLED", "1")))
    # Wave-2 calibration scaffolding. When 1, raw VLM confidences are
    # passed through a learned mapping (temperature scaling or isotonic
    # regression fit on labelled ground truth) before threshold
    # comparison. The mapping file is loaded from
    # ``vlm_calibration_path``; when missing or malformed the calibrator
    # is identity (no-op). Default 0 (off until ground truth exists).
    vlm_calibration_enabled: bool = bool(int(os.getenv("VLM_CALIBRATION_ENABLED", "0")))
    vlm_calibration_path: str = os.getenv(
        "VLM_CALIBRATION_PATH", "/opt/poc2/app/data/vlm_calibration.json"
    )
    # Wave-2 self-consistency: re-query the VLM ``vlm_self_consistency_k``
    # times when the calibrated confidence falls in the borderline band
    # [low, high] and take the majority verdict. Costs k× latency per
    # borderline activity; bounded so it only fires for cases the
    # single-shot run wasn't confident about. Default off; enable once
    # latency budget is validated.
    vlm_self_consistency_k: int = int(os.getenv("VLM_SELF_CONSISTENCY_K", "3"))
    vlm_borderline_low: float = float(os.getenv("VLM_BORDERLINE_LOW", "0.40"))
    vlm_borderline_high: float = float(os.getenv("VLM_BORDERLINE_HIGH", "0.70"))
    # Wave-2 disagreement queue. When 1, append a JSONL entry to
    # ``vlm_disagreement_log_path`` whenever Pipeline-1 and the VLM
    # produce divergent verdicts (e.g. P1 high-conf, VLM drop, or
    # vice-versa). Captures the highest-leverage data for quarterly
    # model improvement. Default 1 (cheap to log).
    vlm_disagreement_log_enabled: bool = bool(int(os.getenv("VLM_DISAGREEMENT_LOG_ENABLED", "1")))
    vlm_disagreement_log_path: str = os.getenv(
        "VLM_DISAGREEMENT_LOG_PATH",
        "/opt/poc2/locopilot_evidence/vlm_disagreements.jsonl",
    )
    # Wave-2 telemetry. When 1, append a structured JSONL line per VLM
    # invocation to ``vlm_telemetry_log_path`` for offline analysis
    # (verdict distribution, latency, gate-drop rate, drift detection).
    # Default 1.
    vlm_telemetry_log_enabled: bool = bool(int(os.getenv("VLM_TELEMETRY_LOG_ENABLED", "1")))
    vlm_telemetry_log_path: str = os.getenv(
        "VLM_TELEMETRY_LOG_PATH",
        "/opt/poc2/locopilot_evidence/vlm_telemetry.jsonl",
    )
    # When 1, ``concurrent_activity_grouping_service.group_concurrent_activities``
    # runs AFTER VLM verification rather than before. The verifier therefore
    # sees raw single-type activities and never has to un-merge a combined
    # record (the per-sub-type fanout in ``vlm/service.py`` becomes a no-op).
    # Grouping then runs on the post-VLM survivor set, so merged clips only
    # include sub-clips that survived verification. Default 0 = legacy
    # pre-VLM grouping. Production opts in via ``.env.production`` after
    # smoke tests pass on the GPU box. Phase B (separate task) deletes the
    # now-dead fanout code.
    concurrent_grouping_after_vlm: bool = bool(int(os.getenv("CONCURRENT_GROUPING_AFTER_VLM", "0")))

    @model_validator(mode='after')
    def _validate_overlap_window(self) -> "Settings":
        """Ensure ``mp_overlap_seconds`` covers the longest temporal-state window.

        ARCH-03: per-worker ``LocopilotActivityMonitor`` instances rebuild their
        temporal state from scratch at every chunk boundary.  Unless the
        overlap (warm-up) window covers both the sleep baseline calibration
        window and the hand gesture coordination window, pose-based sleep and
        gesture coordination detection are effectively suppressed at every
        chunk seam.  This validator fails fast at startup if the operator
        picks an overlap smaller than either window.
        """
        required = max(
            float(self.sleep_baseline_calibration_window),
            float(self.hand_gesture_coordination_window),
        )
        if float(self.mp_overlap_seconds) < required:
            raise ValueError(
                "mp_overlap_seconds must cover sleep baseline and gesture "
                f"coordination windows: got {self.mp_overlap_seconds}s but "
                f"need >= {required}s "
                f"(sleep_baseline_calibration_window="
                f"{self.sleep_baseline_calibration_window}s, "
                f"hand_gesture_coordination_window="
                f"{self.hand_gesture_coordination_window}s). "
                "Raise MP_OVERLAP_SECONDS or shrink the temporal windows."
            )
        return self

    # ==========================================================================
    # Cross-field / flag coherence validator
    # ==========================================================================
    # Runs after all fields are populated. Catches silent misconfigurations such
    # as enabling the train-motion rule engine without the motion detector, or
    # selecting a pose backend whose adapter/library is not installed.
    #
    # Escape hatch: set ``LOCOPILOT_SKIP_PATH_CHECKS=1`` in the environment to
    # skip model-file existence checks. Useful for fresh clones where the .pt
    # weights have not been downloaded yet, and for unit tests.
    @model_validator(mode='after')
    def _validate_flag_combinations(self) -> 'Settings':
        """
        Validate cross-field constraints and flag coherence.

        Checks performed:
          (a) Absolute paths in referenced YOLO model fields must exist on
              disk (when set). Relative paths and missing fields are skipped
              so fresh clones without downloaded weights still boot.
          (c) ``pose_model == 'rtmpose'`` requires the ``rtmlib`` package to
              be importable.

        All checks are defensive: unknown/missing fields are skipped rather
        than raising, so the validator stays compatible with future field
        additions and partial branches.
        """
        # Escape hatch for CI, fresh clones, and unit tests that construct
        # Settings with synthetic values.
        skip_path_checks = os.getenv("LOCOPILOT_SKIP_PATH_CHECKS", "0").lower() in (
            "1", "true", "yes", "on"
        )

        # (a) Referenced model files must exist when set to an absolute path.
        # The task spec references canonical *_path field names; this branch
        # uses *_weights names. Check both sets so the validator survives a
        # future rename.
        if not skip_path_checks:
            path_fields = (
                'yolo_model_path',
                'yolo_pose_model_path',
                'yolo_roi_model_path',
                'yolo_weights',
                'yolo_pose_weights',
            )
            for attr in path_fields:
                path = getattr(self, attr, None)
                if not path:
                    continue
                # Only fail on absolute paths that are missing. Relative
                # paths may resolve lazily via ultralytics' model cache.
                if os.path.isabs(path) and not os.path.exists(path):
                    raise ValueError(
                        f"{attr}={path!r} does not exist on disk. "
                        f"Download the model or update the setting. "
                        f"Set LOCOPILOT_SKIP_PATH_CHECKS=1 to bypass."
                    )

        # (c) Pose backend adapter: rtmpose requires rtmlib installed.
        pose_model = getattr(self, 'pose_model', None)
        if pose_model == 'rtmpose':
            try:
                import rtmlib  # noqa: F401
            except ImportError as e:
                raise ValueError(
                    "POSE_MODEL=rtmpose requires the 'rtmlib' package; "
                    "install with `pip install rtmlib onnxruntime` "
                    "(or onnxruntime-gpu for CUDA)."
                ) from e

        return self

    # ==========================================================================
    # Fail-closed secrets validator (task 0005 — rotate-secrets-scrub-source)
    # ==========================================================================
    # The MinIO credentials are intentionally defaulted to empty strings in
    # this file. Previously they fell back to hardcoded production literals
    # which leaked through every fresh checkout and made an accidental
    # dev->prod credential mismatch invisible. By failing
    # fast at startup whenever ENVIRONMENT=production with empty credentials,
    # we guarantee the operator either supplies real values via the env file
    # or sees a clear error instead of a silent permission failure deep in
    # the MinIO client.
    @model_validator(mode='after')
    def _validate_minio_credentials_in_production(self) -> 'Settings':
        """
        Require MinIO credentials when running in production.

        Raises:
            ValueError: When ``environment`` is ``"production"`` and either
                ``minio_access_key`` or ``minio_secret_key`` is empty/blank.
        """
        env_name = (getattr(self, 'environment', '') or '').strip().lower()
        if env_name != 'production':
            return self

        access_key = (self.minio_access_key or '').strip()
        secret_key = (self.minio_secret_key or '').strip()
        missing = []
        if not access_key:
            missing.append('MINIO_ACCESS_KEY')
        if not secret_key:
            missing.append('MINIO_SECRET_KEY')
        if missing:
            raise ValueError(
                "Production environment requires "
                f"{' and '.join(missing)} to be set to non-empty values. "
                "Refusing to start with empty MinIO credentials. "
                "Populate them in .env.production (never commit real "
                "credentials) or in the systemd EnvironmentFile."
            )
        return self


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    
    Uses LRU cache to ensure settings are loaded only once.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()
"""
Configuration management for the Locopilot Monitoring System

Uses environment variables with sensible defaults for production deployment.
"""

import os
import json
import tempfile
from typing import Optional, List
from functools import lru_cache
from pydantic import field_validator
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

    # CORS settings - explicit allowed origins (no wildcard with credentials)
    cors_allowed_origins: List[str] = json.loads(
        os.getenv("CORS_ALLOWED_ORIGINS", '["http://localhost:3000", "http://localhost:8080"]')
    )

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
    # ✅ 15s chunks ensure hand gesture coordination detection works correctly
    # Coordination window is 10s, so 15s chunks capture full coordination sequences
    # Tradeoff: fewer chunks (~118 for 30-min video) but reliable coordination detection
    mp_chunk_duration: float = 15.0  # Chunk duration in seconds (optimized for coordination detection)
    mp_max_workers: Optional[int] = None  # None = auto-detect (uses min(CPU count, max_workers_cap))
    mp_max_workers_cap: int = 12  # Maximum number of workers (11 cores + slight oversubscription)
    
    # Model settings - YOLO26 (NMS-free, 43% faster CPU inference than YOLO11)
    # Detection: nano for fast bulk frame processing (~57K frames)
    yolo_weights: str = os.getenv("YOLO_WEIGHTS_PRELOAD", "yolo26n.pt")  # YOLO26 nano for object detection
    yolo_pose_weights: str = os.getenv("YOLO_POSE_WEIGHTS", "yolo26n-pose.pt")  # YOLO26 nano-pose for multi-person pose
    yolo_pose_confidence: float = float(os.getenv("YOLO_POSE_CONFIDENCE", "0.45"))  # Pose detection confidence

    # Voting: large model for accurate verification (~160 frames)
    # When empty, voting uses the same model as detection (backward compatible)
    yolo_voting_weights: str = os.getenv("YOLO_VOTING_WEIGHTS", "yolo26l.pt")  # YOLO26 large for voting verification
    yolo_voting_pose_weights: str = os.getenv("YOLO_VOTING_POSE_WEIGHTS", "yolo26l-pose.pt")  # YOLO26 large-pose for voting

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
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "mind.snikbtel.uk:9000")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "admin")
    minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "login123")
    minio_secure: bool = bool(int(os.getenv("MINIO_SECURE", "1")))
    minio_bucket: str = os.getenv("MINIO_BUCKET", "cvss")
    
    # Image preprocessing settings (for MediaPipe detection enhancement)
    enable_image_preprocessing: bool = bool(int(os.getenv("ENABLE_IMAGE_PREPROCESSING", "1")))  # Enable by default
    use_clahe: bool = bool(int(os.getenv("USE_CLAHE", "1")))  # CLAHE is most effective
    use_gamma_correction: bool = bool(int(os.getenv("USE_GAMMA_CORRECTION", "1")))
    use_unsharp_masking: bool = bool(int(os.getenv("USE_UNSHARP_MASKING", "0")))  # Optional, can add artifacts
    use_noise_reduction: bool = bool(int(os.getenv("USE_NOISE_REDUCTION", "1")))
    adaptive_preprocessing: bool = bool(int(os.getenv("ADAPTIVE_PREPROCESSING", "1")))  # Use quality metrics
    clahe_clip_limit: float = float(os.getenv("CLAHE_CLIP_LIMIT", "1.5"))  # REDUCED from 2.0 (less aggressive CLAHE)
    
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

    # Voting verification settings
    # Two-stage detection: when activity detected, verify with multiple native frames
    voting_enabled: bool = bool(int(os.getenv("VOTING_ENABLED", "1")))
    voting_num_frames: int = int(os.getenv("VOTING_NUM_FRAMES", "10"))
    voting_frame_spread_ms: int = int(os.getenv("VOTING_FRAME_SPREAD_MS", "400"))  # 400ms window at 25fps

    # Per-activity voting thresholds (percentage of frames required for confirmation)
    # Default 50% (5/10 frames must detect the activity)
    voting_threshold_cell_phone: float = float(os.getenv("VOTING_THRESHOLD_CELL_PHONE", "0.5"))
    voting_threshold_writing: float = float(os.getenv("VOTING_THRESHOLD_WRITING", "0.5"))
    voting_threshold_packing_bags: float = float(os.getenv("VOTING_THRESHOLD_PACKING_BAGS", "0.75"))  # 75% - stricter for false positive reduction (was 60%)
    voting_threshold_lp_hand_gesture: float = float(os.getenv("VOTING_THRESHOLD_LP_GESTURE", "0.6"))
    voting_threshold_alp_hand_gesture: float = float(os.getenv("VOTING_THRESHOLD_ALP_GESTURE", "0.6"))
    voting_threshold_mind_diversion: float = float(os.getenv("VOTING_THRESHOLD_MIND_DIVERSION", "0.5"))
    voting_threshold_eating_drinking: float = float(os.getenv("VOTING_THRESHOLD_EATING_DRINKING", "0.4"))  # Lower threshold - cups harder to detect in IR
    voting_threshold_group_detected: float = float(os.getenv("VOTING_THRESHOLD_GROUP", "0.5"))

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

    # Voting debug settings - save annotated frames for troubleshooting
    voting_save_debug_frames: bool = bool(int(os.getenv("VOTING_SAVE_DEBUG_FRAMES", "0")))  # Disabled by default (enable locally for debugging)
    voting_debug_frames_dir: str = os.getenv("VOTING_DEBUG_FRAMES_DIR", "voting_debug_frames")

    # Packing bags verification thresholds (stricter than initial detection)
    # TUNED 2026-01-21: Stricter thresholds to reduce false positives from bags on floor near seated crew
    packing_wrist_visibility_threshold: float = float(os.getenv("PACKING_WRIST_VIS", "0.5"))  # Min wrist visibility 50% (was 40%)
    packing_voting_margin: int = int(os.getenv("PACKING_VOTING_MARGIN", "0"))  # No margin - wrist must be truly inside (was 30)
    packing_max_distance_ratio: float = float(os.getenv("PACKING_MAX_DIST_RATIO", "0.30"))  # Wrist must be within 30% of bag diagonal from center (was 0.45)
    packing_min_bag_area: int = int(os.getenv("PACKING_MIN_BAG_AREA", "25000"))  # Min bag area 25,000 sq pixels (was 20000)
    packing_require_wrist_truly_inside: bool = bool(int(os.getenv("PACKING_STRICT_INSIDE", "1")))  # Require wrist truly inside bbox (no margin)

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
    sleep_baseline_calibration_window: float = float(os.getenv("SLEEP_BASELINE_WINDOW", "2.0"))
    sleep_baseline_min_samples: int = int(os.getenv("SLEEP_BASELINE_MIN_SAMPLES", "1"))

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
    haar_eye_microsleep_duration: float = float(os.getenv("HAAR_EYE_MICROSLEEP_DURATION", "1.5"))
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
    yolo_person_confidence: float = float(os.getenv("YOLO_PERSON_CONFIDENCE", "0.5"))
    yolo_bag_confidence: float = float(os.getenv("YOLO_BAG_CONFIDENCE", "0.45"))
    yolo_bag_log_confidence: float = float(os.getenv("YOLO_BAG_LOG_CONFIDENCE", "0.25"))
    yolo_book_confidence: float = float(os.getenv("YOLO_BOOK_CONFIDENCE", "0.4"))
    yolo_cell_phone_confidence: float = float(os.getenv("YOLO_CELL_PHONE_CONFIDENCE", "0.3"))

    # Cell phone detection confidence threshold (activity-level, distinct from YOLO detection threshold)
    cell_phone_confidence: float = float(os.getenv("CELL_PHONE_CONFIDENCE", "0.40"))

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
    book_posture_min_duration: float = float(os.getenv("BOOK_POSTURE_MIN_DURATION", "2.0"))
    book_posture_required_consecutive: int = int(os.getenv("BOOK_POSTURE_REQUIRED_CONSECUTIVE", "2"))

    # ==========================================
    # Head Tilt / Sleep Detection Thresholds
    # ==========================================
    head_down_threshold: float = float(os.getenv("HEAD_DOWN_THRESHOLD", "0.01"))
    sleep_strong_score: int = int(os.getenv("SLEEP_STRONG_SCORE", "6"))
    sleep_strong_duration: int = int(os.getenv("SLEEP_STRONG_DURATION", "0"))
    sleep_moderate_duration: int = int(os.getenv("SLEEP_MODERATE_DURATION", "2"))
    sleep_microsleep_duration: int = int(os.getenv("SLEEP_MICROSLEEP_DURATION", "0"))
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
    # Voting Service Margins
    # ==========================================
    voting_cell_phone_margin: int = int(os.getenv("VOTING_CELL_PHONE_MARGIN", "100"))
    voting_book_hand_margin: int = int(os.getenv("VOTING_BOOK_HAND_MARGIN", "180"))
    voting_person_book_margin: int = int(os.getenv("VOTING_PERSON_BOOK_MARGIN", "250"))

    # ==========================================
    # Train Motion Rules Settings
    # ==========================================
    # Enable/disable train motion-based rule engine
    train_motion_rules_enabled: bool = bool(int(os.getenv("TRAIN_MOTION_RULES_ENABLED", "0")))

    # Suppress no_person_detected when trip schedule is unavailable
    # (cannot distinguish station halts from running without schedule)
    suppress_no_person_without_schedule: bool = bool(int(os.getenv("SUPPRESS_NO_PERSON_WITHOUT_SCHEDULE", "1")))

    # Trip API Settings (RailRadar API)
    trip_api_url: str = os.getenv("TRIP_API_URL", "https://api.railradar.in/api/v1/trains")
    trip_api_timeout: int = int(os.getenv("TRIP_API_TIMEOUT", "10"))

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

    # ==========================================
    # Optical Flow Motion Detection Settings
    # ==========================================
    # Enable/disable optical flow motion verification
    optical_flow_enabled: bool = bool(int(os.getenv("OPTICAL_FLOW_ENABLED", "0")))

    # Side window ROI configuration (ratios of frame dimensions)
    # Captures the narrow left side window/door opening for motion detection
    # IPCamera 02 cabin view: outside scenery visible through left side bars/window
    # Calibrated from "Writing 13 25.mp4" (1280x720, 25fps)
    motion_roi_x_ratio: float = float(os.getenv("MOTION_ROI_X", "0.0"))
    motion_roi_width_ratio: float = float(os.getenv("MOTION_ROI_WIDTH", "0.08"))
    motion_roi_y_ratio: float = float(os.getenv("MOTION_ROI_Y", "0.15"))
    motion_roi_height_ratio: float = float(os.getenv("MOTION_ROI_HEIGHT", "0.50"))

    # Motion classification thresholds (optical flow magnitude - 90th percentile)
    # Calibrated from video analysis (consecutive frames at 25fps):
    #   - STOPPED: p90 magnitude typically 0.13-0.14
    #   - RUNNING: p90 magnitude typically 1.0-2.4
    #   - TRANSITION: p90 magnitude 0.5-0.7
    motion_stopped_threshold: float = float(os.getenv("MOTION_STOPPED_THRESHOLD", "0.3"))
    motion_running_threshold: float = float(os.getenv("MOTION_RUNNING_THRESHOLD", "0.8"))
    motion_confidence_threshold: float = float(os.getenv("MOTION_CONFIDENCE_THRESHOLD", "0.7"))

    # ==========================================
    # etrain.info Delay Integration Settings
    # ==========================================
    # Enable/disable etrain.info delay data fetching
    etrain_enabled: bool = bool(int(os.getenv("ETRAIN_ENABLED", "0")))

    # etrain.info base URL for train live status
    etrain_base_url: str = os.getenv("ETRAIN_BASE_URL", "https://etrain.info/train")

    # Cache TTL for delay data (in seconds, default 30 minutes)
    etrain_cache_ttl: int = int(os.getenv("ETRAIN_CACHE_TTL", "1800"))


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    
    Uses LRU cache to ensure settings are loaded only once.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()
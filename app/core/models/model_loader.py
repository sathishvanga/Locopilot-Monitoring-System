"""Centralized model loading for ML inference.

This module provides a ModelLoader class that handles loading and
configuring all ML models used in the locopilot monitoring system:
- YOLO object detection model
- YOLO-Pose model for body pose estimation
- MediaPipe FaceMesh for face landmark detection
- Image preprocessing service
- Haar cascade classifiers for eye detection

Usage:
    # Load all models fresh
    loader = ModelLoader(settings)
    models = loader.load_all_models()

    # Get models dict for worker pool injection
    preloaded = loader.get_preloaded_models()
"""

from typing import Dict, Any, Optional, Tuple
import logging
import os
import cv2
import mediapipe as mp


class ModelLoader:
    """Manages loading and configuration of ML models.

    This class centralizes model loading to:
    - Avoid duplicate model loading code
    - Support preloaded models for worker pools
    - Handle cleanup and resource management

    Attributes:
        settings: Configuration settings object
        logger: Logger instance for debug output
    """

    def __init__(
        self,
        settings: Optional[Any] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the ModelLoader.

        Args:
            settings: Configuration settings object with model paths and thresholds.
                     If None, uses default values.
            logger: Optional logger instance. If None, creates a new one.
        """
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)

        # Model references (populated by load_all_models)
        self.yolo_model = None
        self.yolo_pose = None
        self.face_mesh = None
        self.mp_face_mesh = None
        self.preprocessing_service = None
        self.face_cascade = None
        self.profile_face_cascade = None
        self.eye_cascade = None

        # MediaPipe module references
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def load_yolo_object_model(self, weights_path: Optional[str] = None) -> Any:
        """Load the YOLO object detection model.

        Args:
            weights_path: Path to YOLO weights file. If None, uses settings or default.

        Returns:
            Loaded YOLO model instance
        """
        from ultralytics import YOLO

        if weights_path is None:
            weights_path = getattr(self.settings, 'yolo_weights', 'yolo26n.pt') if self.settings else 'yolo26n.pt'

        self.logger.info(f"Loading YOLO model: {weights_path}")
        model = YOLO(weights_path)

        # Fuse Conv+BatchNorm layers for faster inference (15-20% speedup)
        if hasattr(model.model, 'fuse'):
            model.fuse()
            self.logger.info("YOLO model layers fused for optimized inference")

        self.yolo_model = model
        return model

    def load_yolo_pose_model(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: Optional[float] = None
    ) -> Any:
        """Load the pose model (YOLO-Pose or RTMPose based on config).

        Args:
            weights_path: Path to YOLO-Pose weights file. If None, uses settings or default.
                         Ignored when pose_model_backend='rtmpose'.
            conf_threshold: Confidence threshold. If None, uses settings or default.

        Returns:
            YoloPoseAdapter or RTMPoseAdapter instance
        """
        if conf_threshold is None:
            conf_threshold = getattr(self.settings, 'yolo_pose_confidence', 0.45) if self.settings else 0.45

        pose_backend = getattr(self.settings, 'pose_model_backend', 'yolo') if self.settings else 'yolo'

        if pose_backend == 'rtmpose':
            from app.services.rtmpose_adapter import RTMPoseAdapter

            rtm_mode = getattr(self.settings, 'rtmpose_mode', 'balanced') if self.settings else 'balanced'
            rtm_backend = getattr(self.settings, 'rtmpose_backend', 'onnxruntime') if self.settings else 'onnxruntime'

            # Determine device: cuda if GPU enabled, else cpu
            yolo_device = getattr(self.settings, 'yolo_device', 'cpu') if self.settings else 'cpu'
            rtm_device = 'cuda' if str(yolo_device) not in ('cpu', '') else 'cpu'

            self.logger.info(f"Loading RTMPose: mode={rtm_mode}, backend={rtm_backend}, device={rtm_device}")
            adapter = RTMPoseAdapter(
                conf_threshold=conf_threshold,
                device=rtm_device,
                mode=rtm_mode,
                backend=rtm_backend,
            )
        else:
            from app.services.yolo_pose_adapter import YoloPoseAdapter

            if weights_path is None:
                weights_path = getattr(self.settings, 'yolo_pose_weights', 'yolo26n-pose.pt') if self.settings else 'yolo26n-pose.pt'

            self.logger.info(f"Loading YOLO-Pose model: {weights_path}")
            adapter = YoloPoseAdapter(model_path=weights_path, conf_threshold=conf_threshold)

        self.yolo_pose = adapter
        return adapter

    def load_face_mesh(
        self,
        max_num_faces: int = 2,
        refine_landmarks: bool = True,
        detection_confidence: Optional[float] = None,
        tracking_confidence: Optional[float] = None
    ) -> Any:
        """Load MediaPipe FaceMesh for face landmark detection.

        Args:
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Whether to refine landmarks
            detection_confidence: Detection confidence threshold
            tracking_confidence: Tracking confidence threshold

        Returns:
            MediaPipe FaceMesh instance
        """
        if detection_confidence is None:
            detection_confidence = getattr(self.settings, 'face_mesh_detection_confidence', 0.5) if self.settings else 0.5

        if tracking_confidence is None:
            tracking_confidence = getattr(self.settings, 'face_mesh_tracking_confidence', 0.5) if self.settings else 0.5

        self.logger.info("Initializing MediaPipe FaceMesh...")
        self.mp_face_mesh = mp.solutions.face_mesh

        face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        self.face_mesh = face_mesh
        return face_mesh

    def load_preprocessing_service(self) -> Optional[Any]:
        """Load the image preprocessing service.

        Returns:
            ImagePreprocessingService instance, or None if unavailable
        """
        try:
            from app.services.image_preprocessing_service import ImagePreprocessingService
            from app.utils.config import get_settings
        except ImportError:
            self.logger.warning("Image preprocessing service not available")
            return None

        if self.settings is None:
            try:
                self.settings = get_settings()
            except Exception:
                self.logger.warning("Could not get settings for preprocessing service")
                return None

        try:
            preprocessing_config = {
                'enable_image_preprocessing': self.settings.enable_image_preprocessing,
                'use_clahe': self.settings.use_clahe,
                'use_gamma_correction': self.settings.use_gamma_correction,
                'use_unsharp_masking': self.settings.use_unsharp_masking,
                'use_noise_reduction': self.settings.use_noise_reduction,
                'adaptive_preprocessing': self.settings.adaptive_preprocessing,
                'clahe_clip_limit': self.settings.clahe_clip_limit,
                'clahe_tile_grid_size': self.settings.clahe_tile_grid_size,
                'gamma_value': self.settings.gamma_value,
                'unsharp_strength': self.settings.unsharp_strength,
                'unsharp_radius': self.settings.unsharp_radius,
                'noise_reduction_kernel': self.settings.noise_reduction_kernel
            }
            service = ImagePreprocessingService(config=preprocessing_config)
            self.logger.info("Image preprocessing service initialized")
            self.preprocessing_service = service
            return service
        except Exception as e:
            self.logger.warning(f"Failed to initialize image preprocessing service: {e}")
            return None

    def load_haar_cascades(self) -> Tuple[Any, Any, Any]:
        """Load Haar cascade classifiers for face and eye detection.

        Returns:
            Tuple of (face_cascade, profile_face_cascade, eye_cascade)
        """
        haar_enabled = getattr(self.settings, 'haar_eye_detection_enabled', False) if self.settings else False

        if not haar_enabled:
            self.logger.info("Haar cascade detection disabled in settings")
            return None, None, None

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.profile_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

        self.logger.info("Haar cascade classifiers loaded for eye closure detection")
        return self.face_cascade, self.profile_face_cascade, self.eye_cascade

    def load_all_models(self) -> Dict[str, Any]:
        """Load all models and return them as a dictionary.

        This is the main entry point for loading all models at once.
        Raises RuntimeError if required models (YOLO, YOLO-Pose) fail to load.

        Returns:
            Dictionary containing all loaded models:
            {
                'yolo': YOLO model,
                'yolo_pose': YoloPoseAdapter,
                'face_mesh': MediaPipe FaceMesh,
                'mp_face_mesh': MediaPipe face_mesh module,
                'preprocessing_service': ImagePreprocessingService,
                'face_cascade': Haar cascade for face,
                'profile_face_cascade': Haar cascade for profile face,
                'eye_cascade': Haar cascade for eyes,
                'mp_pose': MediaPipe pose module,
                'mp_drawing': MediaPipe drawing utils,
                'mp_drawing_styles': MediaPipe drawing styles
            }

        Raises:
            RuntimeError: If required models (YOLO or YOLO-Pose) fail to load
        """
        self.load_yolo_object_model()
        self.load_yolo_pose_model()
        self.load_face_mesh()
        self.load_preprocessing_service()
        self.load_haar_cascades()

        # Validate required models loaded successfully
        if self.yolo_model is None:
            raise RuntimeError("Failed to load required YOLO object detection model")
        if self.yolo_pose is None:
            raise RuntimeError("Failed to load required YOLO-Pose model")

        return self.get_preloaded_models()

    def get_preloaded_models(self) -> Dict[str, Any]:
        """Get dictionary of loaded models for worker pool injection.

        Returns:
            Dictionary containing all model references
        """
        return {
            'yolo': self.yolo_model,
            'yolo_pose': self.yolo_pose,
            'face_mesh': self.face_mesh,
            'mp_face_mesh': self.mp_face_mesh,
            'preprocessing_service': self.preprocessing_service,
            'face_cascade': self.face_cascade,
            'profile_face_cascade': self.profile_face_cascade,
            'eye_cascade': self.eye_cascade,
            'mp_pose': self.mp_pose,
            'mp_drawing': self.mp_drawing,
            'mp_drawing_styles': self.mp_drawing_styles
        }

    def cleanup(self) -> None:
        """Clean up model resources.

        Call this when done with the models to free GPU memory.
        """
        import gc

        # Close FaceMesh
        if self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except Exception:
                pass
            self.face_mesh = None

        # Clear YOLO models
        self.yolo_model = None
        self.yolo_pose = None

        # Clear preprocessing service
        self.preprocessing_service = None

        # Clear cascade classifiers
        self.face_cascade = None
        self.profile_face_cascade = None
        self.eye_cascade = None

        # Force garbage collection
        gc.collect()

        # Try to clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        self.logger.info("Model resources cleaned up")

"""RTMPose Adapter Module

Drop-in replacement for YoloPoseAdapter using rtmlib's RTMPose models.
RTMPose-M achieves 75.8 AP vs YOLO26n-Pose's 57.2 AP — an 18-point
improvement in keypoint accuracy.

Uses rtmlib's lower-level YOLOX + RTMPose components directly (not Body
wrapper) for explicit bbox access and threshold control.

Feature-flagged via POSE_MODEL=rtmpose in environment/config.

Keypoint format: COCO-17 (same as YOLO-Pose):
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

Requires: pip install rtmlib onnxruntime-gpu  (or onnxruntime for CPU)
"""

import fcntl
import logging
import os
import time

import numpy as np

from app.services.yolo_pose_adapter import (
    YoloLandmark,
    YoloPoseLandmarks,
    YOLO_KEYPOINT_INDICES,
)


def _setup_module_logger():
    """Setup a file-only logger for RTMPose adapter."""
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)

    _logger = logging.getLogger("RTMPoseAdapter")
    _logger.setLevel(logging.INFO)

    if not _logger.handlers:
        file_handler = logging.FileHandler(os.path.join(log_dir, "LocopilotMonitoring.log"))
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s,%(msecs)03d [N/A] [N/A] [N/A] [N/A] [%(levelname)s] [%(name)s] [N/A N/A] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

    return _logger


logger = _setup_module_logger()


# Model configurations matching rtmlib's Body.MODE
RTMPOSE_MODES = {
    'balanced': {
        'det': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
        'det_input_size': (640, 640),
        'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',
        'pose_input_size': (192, 256),
    },
    'performance': {
        'det': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
        'det_input_size': (640, 640),
        'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip',
        'pose_input_size': (288, 384),
    },
    'lightweight': {
        'det': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip',
        'det_input_size': (416, 416),
        'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip',
        'pose_input_size': (192, 256),
    },
}


class RTMPoseKeypoints:
    """Bridge class that converts RTMPose numpy arrays to the .xy/.conf
    format expected by YoloPoseLandmarks constructor.

    YoloPoseLandmarks expects:
        keypoints.xy: tensor-like with shape (1, 17, 2) — pixel coords
        keypoints.conf: tensor-like with shape (1, 17) — confidence scores
    """

    def __init__(self, keypoints_xy, keypoints_conf):
        """
        Args:
            keypoints_xy: numpy array of shape (17, 2) — pixel coords for one person
            keypoints_conf: numpy array of shape (17,) — confidence per keypoint
        """
        self.xy = _NumpyTensorBridge(keypoints_xy.reshape(1, 17, 2))
        self.conf = _NumpyTensorBridge(keypoints_conf.reshape(1, 17))


class _NumpyTensorBridge:
    """Minimal bridge so numpy arrays respond to .cpu().numpy() calls
    that YoloPoseLandmarks expects from PyTorch tensors."""

    def __init__(self, array):
        self._array = np.asarray(array, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def __len__(self):
        return len(self._array)

    def __getitem__(self, idx):
        return _NumpyTensorBridge(self._array[idx])


class RTMPoseAdapter:
    """Adapter that provides the same interface as YoloPoseAdapter using
    rtmlib's YOLOX (person detector) + RTMPose (pose estimator).

    All downstream code sees the identical output format:
        {person_idx: {'bbox': [x1,y1,x2,y2], 'bbox_confidence': float,
                      'keypoints': YoloPoseLandmarks}}

    Attributes:
        model: None — forces callers through process()/process_batch()
        conf_threshold: Minimum mean keypoint confidence for a person detection
    """

    def __init__(self, conf_threshold=0.45, device='cpu',
                 mode='balanced', backend='onnxruntime',
                 det_score_thr=0.25):
        """Initialize RTMPose models.

        Args:
            conf_threshold: Minimum mean keypoint confidence for post-filtering
            device: 'cpu' or 'cuda' for inference device
            mode: 'balanced' (RTMPose-M), 'performance' (RTMPose-X), 'lightweight' (RTMPose-S)
            backend: 'onnxruntime' or 'opencv'
            det_score_thr: Person detector score threshold (low=0.25 to catch sleeping persons)
        """
        from rtmlib import YOLOX, RTMPose

        self.conf_threshold = conf_threshold
        self.model = None  # No raw model — forces use of process()/process_batch()
        self.keypoint_indices = YOLO_KEYPOINT_INDICES

        if mode not in RTMPOSE_MODES:
            logger.warning(f"Unknown RTMPose mode '{mode}', falling back to 'balanced'")
            mode = 'balanced'

        config = RTMPOSE_MODES[mode]

        # Use a file lock to prevent multiple workers from downloading models
        # simultaneously (race condition: concurrent zip extract → FileNotFoundError)
        cache_dir = os.path.expanduser('~/.cache/rtmlib/hub/checkpoints')
        os.makedirs(cache_dir, exist_ok=True)
        lock_path = os.path.join(cache_dir, '.rtmlib_download.lock')

        logger.info(f"Loading RTMPose detector: mode={mode}, backend={backend}, device={device}")
        with open(lock_path, 'w') as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                self.det_model = YOLOX(
                    onnx_model=config['det'],
                    model_input_size=config['det_input_size'],
                    backend=backend,
                    device=device,
                    score_thr=det_score_thr,
                )

                logger.info(f"Loading RTMPose estimator: mode={mode}, backend={backend}, device={device}")
                self.pose_model = RTMPose(
                    onnx_model=config['pose'],
                    model_input_size=config['pose_input_size'],
                    backend=backend,
                    device=device,
                )
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

        logger.info(f"RTMPoseAdapter initialized: mode={mode}, device={device}, "
                     f"conf_threshold={conf_threshold}, det_score_thr={det_score_thr}")

    def process(self, frame, conf_threshold=None):
        """Process a single frame and return all detected persons with keypoints.

        Args:
            frame: BGR image (numpy array)
            conf_threshold: Optional confidence override for post-filtering

        Returns:
            dict: {
                person_idx: {
                    'bbox': [x1, y1, x2, y2],
                    'bbox_confidence': float,
                    'keypoints': YoloPoseLandmarks (MediaPipe-compatible)
                }
            }
        """
        effective_conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        # Stage 1: Person detection via YOLOX
        bboxes = self.det_model(frame)

        if bboxes is None or len(bboxes) == 0:
            return {}

        # Stage 2: Pose estimation via RTMPose
        # Returns: keypoints (N, 17, 2) pixel coords, scores (N, 17)
        keypoints, scores = self.pose_model(frame, bboxes=bboxes)

        if keypoints is None or len(keypoints) == 0:
            return {}

        persons = {}
        person_idx = 0

        for i in range(len(keypoints)):
            kp = keypoints[i]   # (17, 2) pixel coordinates
            sc = scores[i]      # (17,) confidence per keypoint

            # Compute mean keypoint confidence as bbox_confidence proxy
            mean_conf = float(np.mean(sc))

            # Post-filter by confidence
            if mean_conf < effective_conf:
                continue

            # Get bbox for this person
            bbox = bboxes[i].tolist() if i < len(bboxes) else [0, 0, frame.shape[1], frame.shape[0]]

            # Create bridge keypoints for YoloPoseLandmarks
            rtm_kps = RTMPoseKeypoints(kp, sc)

            persons[person_idx] = {
                'bbox': bbox[:4],  # [x1, y1, x2, y2]
                'bbox_confidence': mean_conf,
                'keypoints': YoloPoseLandmarks(rtm_kps, frame.shape),
            }
            person_idx += 1

        return persons

    def process_batch(self, frames, batch_size=8, conf_threshold=None, device=None):
        """Process multiple frames sequentially (rtmlib has no native batch API).

        Args:
            frames: List of BGR frames (numpy arrays)
            batch_size: Ignored (kept for interface compatibility)
            conf_threshold: Optional confidence threshold override
            device: Ignored (kept for interface compatibility)

        Returns:
            List of pose result dicts, one per frame (same format as process())
        """
        if not frames:
            return []

        all_poses = []
        for frame in frames:
            try:
                result = self.process(frame, conf_threshold=conf_threshold)
                all_poses.append(result)
            except Exception as e:
                logger.error(f"RTMPose inference failed for frame: {e}")
                all_poses.append({})

        return all_poses

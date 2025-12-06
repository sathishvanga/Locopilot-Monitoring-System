"""
YOLOv8-Pose Adapter Module

This module provides a MediaPipe-compatible interface for YOLOv8-Pose results,
enabling seamless migration from MediaPipe Pose to YOLOv8-Pose while maintaining
backward compatibility with existing activity detection logic.

YOLOv8-Pose provides 17 keypoints (COCO format):
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
"""

from ultralytics import YOLO
import numpy as np


# YOLO Keypoint indices (COCO format)
YOLO_KEYPOINT_INDICES = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# MediaPipe to YOLO keypoint mapping (for backward compatibility)
# MediaPipe indices that don't exist in YOLO are mapped to fallbacks
MEDIAPIPE_TO_YOLO_MAP = {
    0: 0,    # NOSE -> nose
    1: 1,    # LEFT_EYE_INNER -> left_eye (approximation)
    2: 1,    # LEFT_EYE -> left_eye
    3: 3,    # LEFT_EYE_OUTER -> left_ear (approximation)
    4: 2,    # RIGHT_EYE_INNER -> right_eye (approximation)
    5: 2,    # RIGHT_EYE -> right_eye
    6: 4,    # RIGHT_EYE_OUTER -> right_ear (approximation)
    7: 3,    # LEFT_EAR -> left_ear
    8: 4,    # RIGHT_EAR -> right_ear
    11: 5,   # LEFT_SHOULDER -> left_shoulder
    12: 6,   # RIGHT_SHOULDER -> right_shoulder
    13: 7,   # LEFT_ELBOW -> left_elbow
    14: 8,   # RIGHT_ELBOW -> right_elbow
    15: 9,   # LEFT_WRIST -> left_wrist
    16: 10,  # RIGHT_WRIST -> right_wrist
    17: 9,   # LEFT_PINKY -> left_wrist (fallback)
    18: 10,  # RIGHT_PINKY -> right_wrist (fallback)
    19: 9,   # LEFT_INDEX -> left_wrist (fallback - no finger keypoints in YOLO)
    20: 10,  # RIGHT_INDEX -> right_wrist (fallback)
    21: 9,   # LEFT_THUMB -> left_wrist (fallback)
    22: 10,  # RIGHT_THUMB -> right_wrist (fallback)
    23: 11,  # LEFT_HIP -> left_hip
    24: 12,  # RIGHT_HIP -> right_hip
    25: 13,  # LEFT_KNEE -> left_knee
    26: 14,  # RIGHT_KNEE -> right_knee
    27: 15,  # LEFT_ANKLE -> left_ankle
    28: 16,  # RIGHT_ANKLE -> right_ankle
}


class YoloLandmark:
    """Single landmark point with MediaPipe-compatible attributes.

    Attributes:
        x: Normalized X coordinate (0-1)
        y: Normalized Y coordinate (0-1)
        z: Depth coordinate (always 0 for YOLOv8-Pose)
        visibility: Confidence score (0-1)
    """

    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

    def __repr__(self):
        return f"YoloLandmark(x={self.x:.3f}, y={self.y:.3f}, visibility={self.visibility:.3f})"


class YoloPoseLandmarks:
    """Wrapper class that mimics MediaPipe NormalizedLandmarkList structure.

    Provides .landmark attribute with normalized coordinates for backward
    compatibility with existing MediaPipe-based activity detection code.
    """

    def __init__(self, yolo_keypoints, frame_shape):
        """Initialize landmarks from YOLOv8-Pose keypoints.

        Args:
            yolo_keypoints: YOLO keypoints object with .xy and .conf attributes
            frame_shape: Tuple of (height, width) or (height, width, channels)
        """
        self.landmark = []
        h, w = frame_shape[:2]

        # yolo_keypoints.xy: shape (N, 17, 2) - pixel coordinates for N detections
        # yolo_keypoints.conf: shape (N, 17) - confidence scores

        # Get the first detection's keypoints
        if yolo_keypoints.xy is not None and len(yolo_keypoints.xy) > 0:
            xy = yolo_keypoints.xy[0].cpu().numpy()  # (17, 2)
            conf = yolo_keypoints.conf[0].cpu().numpy() if yolo_keypoints.conf is not None else np.ones(17)

            for i in range(17):
                # Normalize coordinates to 0-1 range
                norm_x = xy[i][0] / w if w > 0 else 0.0
                norm_y = xy[i][1] / h if h > 0 else 0.0

                # Clamp to valid range
                norm_x = max(0.0, min(1.0, norm_x))
                norm_y = max(0.0, min(1.0, norm_y))

                landmark = YoloLandmark(
                    x=float(norm_x),
                    y=float(norm_y),
                    z=0.0,  # YOLOv8-Pose doesn't provide Z depth
                    visibility=float(conf[i])
                )
                self.landmark.append(landmark)
        else:
            # Create empty landmarks if no detection
            for _ in range(17):
                self.landmark.append(YoloLandmark(0.0, 0.0, 0.0, 0.0))

    def __getitem__(self, idx):
        """Allow indexing like MediaPipe landmarks."""
        return self.landmark[idx]

    def __len__(self):
        return len(self.landmark)


class YoloPoseAdapter:
    """Adapter class to convert YOLOv8-Pose keypoints to MediaPipe-like landmark format.

    This enables backward compatibility with existing activity detection logic
    that was written for MediaPipe Pose.

    Example:
        adapter = YoloPoseAdapter(model_path='yolov8m-pose.pt')
        results = adapter.process(frame)

        for person_idx, person_data in results.items():
            landmarks = person_data['keypoints']  # YoloPoseLandmarks object
            nose = landmarks.landmark[0]  # Access like MediaPipe
            print(f"Nose at ({nose.x}, {nose.y}) with confidence {nose.visibility}")
    """

    def __init__(self, model_path: str = 'yolov8m-pose.pt', conf_threshold: float = 0.45, 
                 preloaded_model: 'YOLO' = None):
        """Initialize YOLOv8-Pose model.

        Args:
            model_path: Path to YOLOv8-Pose weights file
            conf_threshold: Minimum confidence threshold for detections
            preloaded_model: Optional pre-loaded YOLO model (for worker reuse)
        """
        if preloaded_model is not None:
            # Use pre-loaded model (avoids expensive model loading)
            self.model = preloaded_model
        else:
            print(f"Loading YOLOv8-Pose model: {model_path}")
            self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.keypoint_indices = YOLO_KEYPOINT_INDICES

    def process(self, frame):
        """Process a frame and return all detected persons with their keypoints.

        Args:
            frame: BGR image (numpy array)

        Returns:
            dict: {
                person_idx: {
                    'bbox': [x1, y1, x2, y2],
                    'bbox_confidence': float,
                    'keypoints': YoloPoseLandmarks (MediaPipe-compatible)
                }
            }
        """
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        persons = {}

        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue

            # Iterate through each detected person
            for idx in range(len(r.boxes)):
                box = r.boxes[idx]

                # Create keypoints for this specific person
                person_keypoints = PersonKeypoints(r.keypoints, idx)

                persons[idx] = {
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                    'bbox_confidence': float(box.conf[0]),
                    'keypoints': YoloPoseLandmarks(person_keypoints, frame.shape)
                }

        return persons

    def get_keypoint_name(self, idx: int) -> str:
        """Get keypoint name by index."""
        for name, index in self.keypoint_indices.items():
            if index == idx:
                return name
        return f"keypoint_{idx}"


class PersonKeypoints:
    """Helper class to extract keypoints for a specific person from batch results."""

    def __init__(self, batch_keypoints, person_idx: int):
        """Extract keypoints for a specific person.

        Args:
            batch_keypoints: YOLO keypoints object containing all detections
            person_idx: Index of the person to extract
        """
        self.xy = batch_keypoints.xy[person_idx:person_idx+1] if batch_keypoints.xy is not None else None
        self.conf = batch_keypoints.conf[person_idx:person_idx+1] if batch_keypoints.conf is not None else None


def get_keypoint_by_name(landmarks: YoloPoseLandmarks, keypoint_name: str) -> YoloLandmark:
    """Get a keypoint from landmarks by name.

    Args:
        landmarks: YoloPoseLandmarks object
        keypoint_name: String name like 'nose', 'left_wrist', etc.

    Returns:
        YoloLandmark object with x, y, z, visibility attributes

    Raises:
        ValueError: If keypoint name is unknown
    """
    name_lower = keypoint_name.lower()

    # Handle MediaPipe-style names with underscores
    name_lower = name_lower.replace('_', '_')

    # Check if it's a valid YOLO keypoint
    if name_lower in YOLO_KEYPOINT_INDICES:
        idx = YOLO_KEYPOINT_INDICES[name_lower]
        return landmarks.landmark[idx]

    # Handle MediaPipe-specific keypoints that don't exist in YOLO (fallback to wrist)
    fallback_map = {
        'left_index': 'left_wrist',
        'right_index': 'right_wrist',
        'left_pinky': 'left_wrist',
        'right_pinky': 'right_wrist',
        'left_thumb': 'left_wrist',
        'right_thumb': 'right_wrist',
    }

    if name_lower in fallback_map:
        fallback_name = fallback_map[name_lower]
        idx = YOLO_KEYPOINT_INDICES[fallback_name]
        return landmarks.landmark[idx]

    raise ValueError(f"Unknown keypoint: {keypoint_name}")

# Task 0002: Replace Activity Detection Methods with ActivityDetector Delegation

**Status: COMPLETED**
**Completed Date: 2026-02-05**

## Overview
Replace duplicate activity detection methods in `locopilot_monitor.py` with thin delegation wrappers to `self.activity_detector` (ActivityDetector class from `app.core.detectors.activity_detector`).

## Methods to Replace

### 1. `detect_writing_posture` (~80 lines)
```python
def detect_writing_posture(self, pose_landmarks: Any, frame_shape: Tuple[int, ...]) -> bool:
    """Detect writing posture - delegates to ActivityDetector."""
    return self.activity_detector.detect_writing_posture(pose_landmarks, frame_shape)
```

### 2. `detect_head_looking_down` (~40 lines)
```python
def detect_head_looking_down(self, pose_landmarks: Any) -> bool:
    """Detect head looking down - delegates to ActivityDetector."""
    return self.activity_detector.detect_head_looking_down(pose_landmarks)
```

### 3. `is_wrist_inside_backpack` (~30 lines)
```python
def is_wrist_inside_backpack(self, wrist_coords: Optional[Tuple[float, float]],
                              backpack_bbox: List[int], margin: int = 30) -> Tuple[bool, float]:
    """Check if wrist is inside backpack - delegates to ActivityDetector."""
    return self.activity_detector.is_wrist_inside_backpack(wrist_coords, backpack_bbox, margin)
```

### 4. `analyze_packing_hand_motion` (~100 lines)
```python
def analyze_packing_hand_motion(self, person_idx: int, landmarks: Any,
                                 frame_shape: Tuple[int, ...], timestamp_sec: float,
                                 backpack_bbox: List[int]) -> Dict[str, Any]:
    """Analyze packing hand motion - delegates to ActivityDetector."""
    return self.activity_detector.analyze_packing_hand_motion(
        person_idx, landmarks, frame_shape, timestamp_sec, backpack_bbox
    )
```

### 5. `detect_packing_bags` (~80 lines)
```python
def detect_packing_bags(self, landmarks: Any, bag_detections: List,
                        person_bbox: List[int], person_idx: int,
                        timestamp: float, frame_shape: Tuple[int, ...]) -> Tuple[bool, Dict]:
    """Detect packing bags activity - delegates to ActivityDetector."""
    return self.activity_detector.detect_packing_bags(
        landmarks, bag_detections, person_bbox, person_idx, timestamp, frame_shape
    )
```

## Already Completed
- `calculate_wrist_distance` - Already replaced with delegation

## Prerequisites
- `self.activity_detector` is already initialized in `__init__` method
- ActivityDetector class exists at `app.core.detectors.activity_detector`

## Verification
```bash
python3 -m py_compile locopilot_monitor.py
```

## Estimated Lines Removed
~330 lines

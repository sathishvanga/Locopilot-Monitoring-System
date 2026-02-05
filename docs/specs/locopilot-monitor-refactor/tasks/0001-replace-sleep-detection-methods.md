# Task 0001: Replace Sleep Detection Methods with SleepDetector Delegation

**Status: COMPLETED**
**Completed Date: 2026-02-05**

## Overview
Replace duplicate sleep detection methods in `locopilot_monitor.py` with thin delegation wrappers to `self.sleep_detector` (SleepDetector class from `app.core.detectors.sleep_detector`).

## Methods to Replace

### 1. `calculate_head_tilt_angle` (~30 lines)
```python
def calculate_head_tilt_angle(self, landmarks: Any) -> Optional[float]:
    """Calculate head tilt angle - delegates to SleepDetector."""
    return self.sleep_detector.calculate_head_tilt_angle(landmarks)
```

### 2. `calculate_movement_score` (~50 lines)
```python
def calculate_movement_score(self, current_landmarks: Any, previous_landmarks: Any) -> float:
    """Calculate movement score - delegates to SleepDetector."""
    return self.sleep_detector.calculate_movement_score(current_landmarks, previous_landmarks)
```

### 3. `detect_pose_based_sleep` (~500 lines)
```python
def detect_pose_based_sleep(self, landmarks: Any, timestamp_sec: float, person_idx: int,
                            frame_shape: Tuple[int, ...], haar_result: Optional[Dict] = None) -> Tuple[bool, bool, Dict]:
    """Detect pose-based sleep - delegates to SleepDetector."""
    return self.sleep_detector.detect_pose_based_sleep(
        landmarks, timestamp_sec, person_idx, frame_shape, haar_result
    )
```

### 4. `detect_ir_forward_lean_sleep` (~200 lines)
```python
def detect_ir_forward_lean_sleep(self, landmarks: Any, bbox: List[int], timestamp_sec: float,
                                  person_idx: int, frame_shape: Tuple[int, ...]) -> Tuple[bool, bool, Dict]:
    """Detect IR forward lean sleep - delegates to SleepDetector."""
    return self.sleep_detector.detect_ir_forward_lean_sleep(
        landmarks, bbox, timestamp_sec, person_idx, frame_shape
    )
```

### 5. `detect_eye_closure_haar` (~150 lines)
```python
def detect_eye_closure_haar(self, frame, pose_landmarks, person_idx, bbox, timestamp_sec) -> Dict:
    """Detect eye closure using Haar cascade - delegates to SleepDetector."""
    return self.sleep_detector.detect_eye_closure_haar(
        frame, pose_landmarks, person_idx, bbox, timestamp_sec
    )
```

## Prerequisites
- `self.sleep_detector` is already initialized in `__init__` method
- SleepDetector class exists at `app.core.detectors.sleep_detector`

## Verification
```bash
python3 -m py_compile locopilot_monitor.py
```

## Estimated Lines Removed
~930 lines

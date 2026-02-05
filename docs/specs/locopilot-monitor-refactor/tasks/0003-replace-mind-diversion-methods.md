# Task 0003: Replace Mind Diversion Methods with MindDiversionDetector Delegation

**Status: COMPLETED**
**Completed Date: 2026-02-05**

## Overview
Replace duplicate mind diversion detection methods in `locopilot_monitor.py` with thin delegation wrappers to `self.mind_diversion_detector` (MindDiversionDetector class from `app.core.detectors.mind_diversion_detector`).

## Methods to Replace

### 1. `calculate_head_pose_angles` (~240 lines)
```python
def calculate_head_pose_angles(self, pose_landmarks: Any, face_landmarks: Any,
                                frame_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """Calculate head pose angles - delegates to MindDiversionDetector."""
    return self.mind_diversion_detector.calculate_head_pose_angles(
        pose_landmarks, face_landmarks, frame_shape
    )
```

## Method Details

The `calculate_head_pose_angles` method:
- Calculates yaw (side turn) and pitch (up/down tilt) angles
- Detects three types of mind diversion:
  1. `looking_sideways` - head turned > 55°
  2. `looking_away_combined` - head turned > 40° AND down > 20°
  3. `looking_down_distracted` - head down > 30°
- Uses both pose landmarks and face mesh landmarks
- Returns dict with: yaw, pitch, detected, sub_type, method

## Prerequisites
- `self.mind_diversion_detector` is already initialized in `__init__` method
- MindDiversionDetector class exists at `app.core.detectors.mind_diversion_detector`

## Verification
```bash
python3 -m py_compile locopilot_monitor.py
```

## Estimated Lines Removed
~240 lines

# Task 0005: Add Nose Visibility Fallback for Mind Diversion [COMPLETED]

## Overview
Add fallback logic in `calculate_head_pose_angles()` when nose visibility is low, using ear asymmetry for yaw estimation.

## Problem
- `nose_low_visibility` in 13+ frames prevents angle calculation
- Camera angle causes face to be partially visible
- Mind diversion cannot be calculated when nose is not visible

## File to Modify
`/Users/satishvanga/Desktop/Practice/locopilot_monitor.py` in `calculate_head_pose_angles()` (~line 4177)

## Changes Required

When nose visibility < 0.5, use alternative landmarks:
- Use ear visibility asymmetry (if one ear visible, head turned away from that side)
- Use shoulder-to-nose offset if shoulders visible
- Fall back to ear-based yaw estimation

## Implementation

Add fallback logic in `calculate_head_pose_angles()`:

```python
# FALLBACK: When nose not visible, use ear asymmetry for yaw estimation
if nose.visibility < 0.5:
    # If only one ear visible, person is turned away from hidden ear
    left_ear_vis = left_ear.visibility if left_ear else 0
    right_ear_vis = right_ear.visibility if right_ear else 0

    if left_ear_vis > 0.5 and right_ear_vis < 0.3:
        # Right ear hidden = turned right
        yaw_angle = 60  # Estimate significant right turn
        result['method'] = 'ear_asymmetry'
    elif right_ear_vis > 0.5 and left_ear_vis < 0.3:
        # Left ear hidden = turned left
        yaw_angle = -60  # Estimate significant left turn
        result['method'] = 'ear_asymmetry'

    # Check if this exceeds sideways threshold
    if abs(yaw_angle) > settings.mind_diversion_yaw_sideways:
        result['detected'] = True
        result['sub_type'] = 'looking_sideways'
```

## Expected Outcome
- Mind diversion can be detected even when nose is not visible
- Ear asymmetry provides reliable yaw estimation
- Fewer missed detections due to camera angle

## Testing
Re-run diagnostic script after changes:
```bash
python diagnose_video.py /Users/satishvanga/Documents/poc/n_1.mp4
```

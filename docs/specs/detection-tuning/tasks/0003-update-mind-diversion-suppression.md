# Task 0003: Update Mind Diversion Suppression Threshold

## Overview
Reduce the mind diversion wrist distance suppression threshold in config.py to only suppress when truly in writing pose.

## Problem
- All mind diversion detections are being **suppressed** by `wrist_distance < 350px`
- Person's wrists are close together (writing pose)
- **Issue**: Suppression threshold too aggressive - suppresses valid mind diversion detections

## File to Modify
`/Users/satishvanga/Desktop/Practice/app/utils/config.py` (line ~202)

## Changes Required

| Parameter | Current | New | Reason |
|-----------|---------|-----|--------|
| `mind_diversion_wrist_distance_threshold` | 350px | **200px** | Only suppress when truly writing |

## Implementation

Update the configuration:

```python
mind_diversion_wrist_distance_threshold: float = float(os.getenv("MIND_DIV_WRIST_DIST", "200"))  # Was 350
```

## Expected Outcome
- Fewer false suppressions of valid mind diversion detections
- Writing pose suppression still works for actual writing (wrists very close together)

## Testing
Re-run diagnostic script after changes:
```bash
python diagnose_video.py /Users/satishvanga/Documents/poc/n_1.mp4
```

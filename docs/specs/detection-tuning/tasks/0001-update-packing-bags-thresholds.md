# Task 0001: Update Packing Bags Thresholds

## Overview
Update packing bags detection thresholds in locopilot_monitor.py to increase detection rate from 33% to 80%+.

## Problem
- **Primary issue**: `wrist_not_inside_bag_bbox` - 10 occurrences
- **Secondary**: `no_bag_detected_by_yolo` - 3 occurrences
- **Current threshold**: `wrist_inside_margin = 40px` (too strict)
- **Observed**: Wrists often 50-60px from bag bbox

## File to Modify
`/Users/satishvanga/Desktop/Practice/locopilot_monitor.py` (lines ~314-322)

## Changes Required

| Parameter | Current | New | Reason |
|-----------|---------|-----|--------|
| `wrist_inside_margin` | 40px | **80px** | Wrists often 50-60px from bag |
| `margin` (hand_proximity_margin) | 50px | **100px** | Relax proximity check |

## Implementation

Update the `packing_bags` threshold configuration:

```python
'packing_bags': {
    'min_duration': 0.0,
    'required_consecutive': 1,
    'margin': 100,                 # Was 50 - hand proximity margin
    'region_margin': 150,
    'grace_frames': 5,
    'wrist_inside_margin': 80,     # Was 40 - INCREASED
    'sustained_proximity_seconds': 4.0
}
```

## Expected Outcome
- packing_bags detection rate: 33% → 80%+

## Testing
Re-run diagnostic script after changes:
```bash
python diagnose_video.py /Users/satishvanga/Documents/poc/n_1.mp4
```

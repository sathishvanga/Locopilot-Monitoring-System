# Task 0004: Update Cell Phone Detection Confidence

## Overview
Lower cell phone detection confidence threshold in locopilot_monitor.py for better detection at marginal visibility.

## Problem
- Phone detected at confidence 0.489
- Current threshold (0.45) has minimal margin
- **Issue**: Phone may be missed at slightly lower confidence levels

## File to Modify
`/Users/satishvanga/Desktop/Practice/locopilot_monitor.py` (line ~278)

## Changes Required

| Parameter | Current | New | Reason |
|-----------|---------|-----|--------|
| `cell_phone_confidence` | 0.45 | **0.40** | Phone detected at 0.489, need margin for lower visibility |

## Implementation

Update the cell phone confidence setting:

```python
self.cell_phone_confidence = float(os.getenv("CELL_PHONE_CONFIDENCE", "0.40"))  # Was 0.45
```

## Expected Outcome
- Better cell phone detection at marginal visibility
- Phones at 0.40-0.45 confidence will now be detected

## Testing
Re-run diagnostic script after changes:
```bash
python diagnose_video.py /Users/satishvanga/Documents/poc/n_1.mp4
```

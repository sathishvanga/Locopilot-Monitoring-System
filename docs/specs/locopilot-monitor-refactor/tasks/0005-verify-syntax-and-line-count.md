# Task 0005: Verify Syntax and Check Line Count Reduction

**Status: COMPLETED**
**Completed Date: 2026-02-05**

## Overview
Final verification task to ensure all refactoring is complete and the file compiles correctly.

## Verification Steps

### 1. Syntax Check
```bash
python3 -m py_compile locopilot_monitor.py
```
Expected: No errors

### 2. Line Count Check
```bash
wc -l locopilot_monitor.py
```
Expected: ~2000-3000 lines (down from original 8200+)

### 3. Import Verification
Ensure all required imports are present at the top of the file:
```python
from app.core.detectors.sleep_detector import SleepDetector
from app.core.detectors.activity_detector import ActivityDetector
from app.core.detectors.gesture_detector import GestureDetector
from app.core.detectors.mind_diversion_detector import MindDiversionDetector
from app.core.activity_tracker import ActivityTracker, ActivityConfig, ActivityState
from app.core.evidence_manager import EvidenceManager
from app.core.utils.geometry import calculate_iou, bbox_overlap_with_margin, deduplicate_person_boxes
from app.core.models import YOLOHandler, YOLO_KEYPOINT_INDICES, YOLO_HEAD_INDICES, YOLO_BODY_INDICES
```

### 4. Quick Functional Test
```bash
python3 -c "from locopilot_monitor import LocopilotActivityMonitor; print('Import successful')"
```

## Summary of Expected Reductions

| Category | Lines Removed |
|----------|---------------|
| Sleep Detection | ~930 |
| Activity Detection | ~330 |
| Mind Diversion | ~240 |
| Evidence Methods | ~80 |
| Geometry Methods | ~120 |
| **Total** | **~1700** |

## Success Criteria
- [ ] File compiles without syntax errors
- [ ] Line count reduced to target range
- [ ] All imports resolve correctly
- [ ] Basic import test passes

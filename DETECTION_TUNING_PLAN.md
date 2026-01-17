# Activity Detection Threshold Tuning Plan

## Based on Diagnostic Results

Diagnostic script completed analysis of video `/Users/satishvanga/Documents/poc/n_1.mp4`

### Detection Rate Summary
| Activity | Expected | Detected | Rate |
|----------|----------|----------|------|
| packing_bags | 3 | 1 | **33%** |
| cell_phone | 1 | 0 | **0%** |
| mind_diversion | 7 | 7 | 100% |
| writing | 4 | 3 | 75% |
| no_person_detected | 3 | 3 | 100% |
| **TOTAL** | 18 | 14 | **78%** |

---

## Root Cause Analysis

### 1. PACKING BAGS (67% missed)
- **Primary issue**: `wrist_not_inside_bag_bbox` - 10 occurrences
- **Secondary**: `no_bag_detected_by_yolo` - 3 occurrences
- **Current threshold**: `wrist_inside_margin = 40px` (too strict)
- **Observed**: Wrists often just outside bag bbox

### 2. CELL PHONE (100% missed at expected timestamps)
- **Issue**: Phone detected at **1:04** (conf 0.489) but expected at **1:24-1:40**
- Phone IS detected by YOLO when visible
- **Root cause**: Timestamp mismatch, not detection failure

### 3. MIND DIVERSION (detected but suppressed)
- All detections are being **suppressed** by `wrist_distance < 350px`
- Person's wrists are close together (writing pose)
- **Issue**: Suppression threshold too aggressive

### 4. WRITING (25% missed)
- **Issue**: `no_book_detected` - 13 occurrences
- Wrist proximity heuristic catches most cases
- Book detection via YOLO is unreliable

### 5. POSE VISIBILITY
- `nose_low_visibility` in 13+ frames prevents angle calculation
- Camera angle causes face to be partially visible

---

## Implementation Plan

### Files to Modify

1. **`/Users/satishvanga/Desktop/Practice/locopilot_monitor.py`**
   - Packing bags thresholds (lines ~314-322)
   - Cell phone thresholds (lines ~331-336)
   - Mind diversion angle fallback logic

2. **`/Users/satishvanga/Desktop/Practice/app/utils/config.py`**
   - Mind diversion suppression threshold
   - Voting thresholds

---

## Confirmed Threshold Changes

### Change 1: Packing Bags - Increase wrist margin ✅ CONFIRMED
**File**: `locopilot_monitor.py` lines ~314-322

| Parameter | Current | New | Reason |
|-----------|---------|-----|--------|
| `wrist_inside_margin` | 40px | **80px** | Wrists often 50-60px from bag |
| `hand_proximity_margin` | 50px | **100px** | Relax proximity check |
| Bag confidence | 0.45 | **0.35** | Bag detected at 0.27-0.40 in some frames |

### Change 2: Mind Diversion - Reduce suppression ✅ CONFIRMED
**File**: `config.py` line ~202

| Parameter | Current | New | Reason |
|-----------|---------|-----|--------|
| `mind_diversion_wrist_distance_threshold` | 350px | **200px** | Only suppress when truly writing |

### Change 3: Cell Phone - Lower confidence threshold
**File**: `locopilot_monitor.py` line ~278

| Parameter | Current | New | Reason |
|-----------|---------|-----|--------|
| `cell_phone_confidence` | 0.45 | **0.40** | Phone detected at 0.489, margin for lower visibility |

### Change 4: Mind Diversion - Add nose visibility fallback ✅ CONFIRMED
**File**: `locopilot_monitor.py` in `calculate_head_pose_angles()`

When nose visibility < 0.5, use alternative landmarks:
- Use ear visibility asymmetry (if one ear visible, head turned away from that side)
- Use shoulder-to-nose offset if shoulders visible
- Fall back to ear-based yaw estimation

---

## Implementation Steps

### Step 1: Update packing_bags thresholds in locopilot_monitor.py (~line 314-322)
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

### Step 2: Update bag confidence in voting_verification_service.py
Find bag detection confidence check and lower from 0.45 to 0.35

### Step 3: Update mind_diversion_wrist_distance_threshold in config.py (~line 202)
```python
mind_diversion_wrist_distance_threshold: float = float(os.getenv("MIND_DIV_WRIST_DIST", "200"))  # Was 350
```

### Step 4: Update cell_phone_confidence in locopilot_monitor.py (~line 278)
```python
self.cell_phone_confidence = float(os.getenv("CELL_PHONE_CONFIDENCE", "0.40"))  # Was 0.45
```

### Step 5: Add nose visibility fallback in calculate_head_pose_angles() (~line 4177)
Add fallback logic when `nose.visibility < 0.5`:

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

---

## No Changes Required

### No Person Detection ✅ NO CHANGES NEEDED
- Current detection rate: **100%** (3/3)
- User confirmed no changes needed
- Note: Pose model occasionally detects "ghost" poses but doesn't affect overall detection

---

## Testing

After changes, re-run diagnostic script:
```bash
python diagnose_video.py /Users/satishvanga/Documents/poc/n_1.mp4
```

Expected improvements:
- packing_bags: 33% → 80%+
- mind_diversion: Fewer false suppressions
- cell_phone: Better detection at lower confidence
- no_person_detected: Maintain 100%

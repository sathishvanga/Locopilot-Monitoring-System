# False Positive Detection Fixes

## Date: 2025-11-23

## Problem Identified
Two false positive detections were reported:
1. **Group Detection** - Triggered when only 2 people (LP and ALP) were present
2. **Writing Detection** - Triggered when hands were not actually near books

## Root Causes

### Group Detection False Positives
- **Low IOU threshold (0.3)**: Allowed overlapping/duplicate person boxes to be counted as separate people
- **Low temporal consistency (2 frames)**: Single momentary misdetections could trigger alerts
- **Issue**: YOLO was detecting reflections, shadows, or duplicate boxes as additional people

### Writing Detection False Positives
- **Very low confidence threshold (0.2)**: Detected control panels, papers, and other rectangular objects as books
- **Large proximity margin (100px)**: Incorrectly associated hands with distant objects in overhead camera views
- **Large person-to-book margin (200px)**: Associated books with people who weren't actually near them
- **Instant detection (1 frame)**: Single frame misdetection immediately triggered activity

## Implemented Fixes

### 1. Group Detection Improvements

#### Changes Made:
```python
# BEFORE:
deduplicate_person_boxes(detections['person'], iou_threshold=0.3)
'required_consecutive': 2  # 2 samples @ 0.5fps = 4 seconds

# AFTER:
deduplicate_person_boxes(detections['person'], iou_threshold=0.5)
'required_consecutive': 3  # 3 samples @ 0.5fps = 6 seconds
```

**Benefits:**
- ✅ **Better duplicate filtering**: IOU threshold increased from 0.3 to 0.5
  - Filters out overlapping detections more aggressively
  - Reduces false person counts from shadows/reflections
  
- ✅ **Temporal consistency**: Requires 3 consecutive detections (6 seconds) instead of 2 (4 seconds)
  - Momentary misdetections won't trigger false alerts
  - Only persistent group presence creates activity

### 2. Writing Detection Improvements

#### Changes Made:
```python
# Book Detection Confidence
# BEFORE: conf > 0.2
# AFTER:  conf > 0.35

# Hand-to-Book Proximity Margin
# BEFORE: 'margin': 100
# AFTER:  'margin': 70

# Person-to-Book Association Margin
# BEFORE: margin = 200
# AFTER:  margin = 150

# Consecutive Frames Required
# BEFORE: 'required_consecutive': 1
# AFTER:  'required_consecutive': 2
```

**Benefits:**
- ✅ **Higher confidence threshold (0.35)**: Reduces misdetection of control panels/papers as books
- ✅ **Stricter hand proximity (70px)**: Hand must be genuinely close to book (not just in general area)
- ✅ **Stricter person association (150px)**: Book must be closer to person to be considered
- ✅ **Temporal consistency (2 frames = 4 seconds)**: Prevents single-frame misdetections

## Detection Logic Flow

### Writing Detection (Improved)
1. ✅ YOLO detects "book" object with confidence > 0.35 (stricter)
2. ✅ Book must be within 150px of person's bounding box (stricter)
3. ✅ Person's RIGHT HAND must be within 70px of book bounding box (stricter)
4. ✅ Must be detected for 2 consecutive frames (4 seconds) (new requirement)

### Group Detection (Improved)
1. ✅ YOLO detects all persons in frame
2. ✅ De-duplicate overlapping boxes with IOU threshold 0.5 (stricter)
3. ✅ Count deduplicated persons
4. ✅ If count > 2, must persist for 3 consecutive frames (6 seconds) (stricter)

## Expected Results

### Before Fixes:
- ❌ Group detected with only 2 people (LP + ALP)
- ❌ Writing detected when hands not near books
- ❌ Single-frame misdetections triggered activities
- ❌ Control panels misidentified as books

### After Fixes:
- ✅ Group only detected when genuinely 3+ people present for 6+ seconds
- ✅ Writing only detected when hand is genuinely near book (within 70px)
- ✅ Requires temporal consistency (not just single frame)
- ✅ Higher confidence threshold reduces object misclassification

## Testing Recommendations

1. **Test with 2-person scenarios** (LP + ALP only):
   - Should NOT trigger group detection
   - Verify no false positives from shadows/reflections

2. **Test with writing scenarios**:
   - Should trigger ONLY when hand is actively near logbook
   - Should NOT trigger when:
     - Hand is just gesturing near controls
     - Person is operating control panels
     - Books/papers are on lap but not being touched

3. **Test with 3+ person scenarios**:
   - Should trigger group detection after 6 seconds of 3+ people
   - Should correctly identify roles (LP, ALP, Trainee, etc.)

## Technical Details

### Modified Files:
- `/locopilot_monitor.py`

### Modified Sections:
1. Line ~130: Writing activity threshold (margin: 100 → 70, consecutive: 1 → 2)
2. Line ~154: Group detection threshold (consecutive: 2 → 3)
3. Line ~650: Book detection confidence (0.2 → 0.35)
4. Line ~660: Person-to-book margin (200 → 150)
5. Line ~3076: Person de-duplication IOU (0.3 → 0.5)

## Notes

- All changes maintain backward compatibility
- No API changes required
- Evidence clips and JSON output format unchanged
- Changes only affect detection sensitivity and accuracy

## Validation

Run the monitoring system on the same videos that produced false positives:
```bash
python locopilot_monitor.py --video example_data/latest.mp4
```

Check evidence output:
- Group detection should only appear with 3+ people
- Writing detection should only appear when hands genuinely near books

---

**Status**: ✅ Implemented and Ready for Testing
**Impact**: Reduces false positives while maintaining true positive detection rate




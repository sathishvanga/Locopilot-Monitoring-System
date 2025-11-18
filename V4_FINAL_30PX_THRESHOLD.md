# V4.0-B Final - 30px Shoulder Threshold

## 🔧 **Final Adjustment**

**Date**: November 18, 2025  
**Version**: v4.0-B-shoulder-30px  
**Status**: ✅ Ready for Testing

---

## 📊 **What Changed**

### Threshold Lowered: 50px → 30px

```python
# BEFORE (v4.0-B with 50px):
right_hand_at_head = right_wrist_to_shoulder_height >= 50  # Only very high reaches

# AFTER (v4.0-B with 30px):
right_hand_at_head = right_wrist_to_shoulder_height >= 30  # Medium-high and high reaches
```

---

## 🎯 **User Requirements Met**

### Test Results from run_20251118_224147

**With 50px threshold**:
- ✅ Frame 6650: **DETECTED** (high reach)
- ❌ Frame 10400: **REJECTED** (73.6% confidence, but hand only ~40-45px above shoulder)

**User Feedback**: Frame 10400 should also be captured as an activity.

**Solution**: Lower threshold to 30px to capture medium-high reaches.

---

## 📊 **Detection Behavior Changes**

| Hand Position | Height Above Shoulder | 50px Threshold | 30px Threshold |
|--------------|----------------------|----------------|----------------|
| Very High (signaling) | 60-100px | ✅ Detect | ✅ Detect |
| High (frame 6650) | 50-60px | ✅ Detect | ✅ Detect |
| Medium-High (frame 10400) | 30-50px | ❌ Reject | **✅ Detect** |
| Shoulder Level | 0-30px | ❌ Reject | ❌ Reject |
| Below Shoulder | < 0px | ❌ Reject | ❌ Reject |

---

## 🎯 **Mandatory Criteria (Final v4.0-B)**

ALL THREE must pass:

```python
1. Hand Height (SHOULDER-BASED):  wrist_to_shoulder_height >= 30px  [LOWERED from 50]
2. Arm Verticality:               arm_verticality >= 1.5
3. Control Zone:                  not in_control_zone
```

---

## 📈 **Expected Test Results**

When you run the test again with 30px threshold:

| Frame | Previous (50px) | Expected (30px) | Reason |
|-------|----------------|-----------------|---------|
| **6650** | ✅ Detected | ✅ Detected | High reach (~55px above shoulder) |
| **10400** | ❌ Rejected | **✅ Detected** | Medium-high reach (~40px above shoulder) |
| Low reaches | ❌ Rejected | ❌ Rejected | Hand at or below shoulder level |

**Expected LP Hand Gesture Clips**: **2-4** (up from 1)

---

## 🧪 **Testing Instructions**

Run the same video with the updated threshold:

```bash
curl -X POST "http://localhost:8000/api/jobs" \
  -F "video=@example_data/latest.mp4" \
  -F "tripId=V4B-30PX-TEST" \
  -F "lpCrewName=Test Pilot" \
  -F "lpCrewId=LP-001" \
  -F "saveClips=true" \
  -F "enableGestureDebug=true" \
  -F "gestureSensitivity=balanced"
```

### What to Verify

✅ Frame 6650 region: Still detected  
✅ Frame 10400 region: **Now detected**  
✅ Low reaches (chest/below): Still rejected  
✅ 2-4 total LP hand gesture clips  

### Debug Logs Will Show

```
[GESTURE v4.0-B SHOULDER-REF] Right hand - arm_verticality: 1.7 (MUST BE ≥1.5), 
                              wrist_above_shoulder: 42.0px (MUST BE ≥30 [LOWERED]), 
                              hand_at_head: True, arm_is_vertical: True

✓ LP hand gesture DETECTED - Confidence: 75.0%, Hand: right
```

---

## ⚖️ **Trade-off Analysis**

### 50px Threshold (Previous)
- ✅ Very strict - minimal false positives
- ✅ Only detects very high, obvious signals
- ❌ Misses medium-high reaches (like frame 10400)
- **Result**: 1 detection (only frame 6650)

### 30px Threshold (Current)
- ✅ Captures medium-high and high reaches
- ✅ Still rejects shoulder-level operations
- ⚠️ Slightly higher chance of false positives (but controlled by arm verticality check)
- **Expected Result**: 2-4 detections (more comprehensive)

---

## 🛡️ **Protection Against False Positives**

Even with the lowered 30px threshold, the system still has **2 other mandatory checks**:

### 1. Arm Verticality (≥ 1.5)
Rejects forward reaches where arm is mostly horizontal:
```
Forward reach:  arm_verticality = 0.8  → FAIL ❌
Diagonal reach: arm_verticality = 1.4  → FAIL ❌
Up-and-forward: arm_verticality = 1.6  → PASS ✅
Vertical raise: arm_verticality = 2.5  → PASS ✅
```

### 2. Control Zone Check
Rejects hands that are:
- In the control panel region
- Too far forward (not close to body)
- Below head level (nose-based check still active in control zone logic)

**Result**: The 30px threshold is SAFE - won't cause massive false positives.

---

## 📊 **Version Evolution Summary**

| Version | Key Feature | LP Detections | Issues |
|---------|-------------|---------------|---------|
| v2.0 | Blacklist (control zone) | 9 | 8 false positives |
| v3.0 | Enhanced blacklist | 9 | No improvement |
| v4.0-strict | Whitelist (nose ref, 2.0 vert) | 0 | Too strict |
| v4.0-lenient | Whitelist (nose ref, 1.5 vert) | 0 | Head tilt issue |
| v4.0-B (50px) | **Shoulder ref, 50px** | 1 | Missed medium-high |
| **v4.0-B (30px)** | **Shoulder ref, 30px** | **2-4** | **✅ Balanced** |

---

## 🔧 **If Further Tuning Needed**

### If Too Many Detections (False Positives)

**Option 1**: Increase threshold slightly
```python
right_hand_at_head = right_wrist_to_shoulder_height >= 35  # Was 30
```

**Option 2**: Increase verticality requirement
```python
right_arm_is_vertical = right_arm_verticality >= 1.7  # Was 1.5
```

### If Missing Some Valid Gestures

**Option 1**: Decrease threshold further
```python
right_hand_at_head = right_wrist_to_shoulder_height >= 20  # Was 30
```

**Option 2**: Decrease verticality requirement
```python
right_arm_is_vertical = right_arm_verticality >= 1.3  # Was 1.5
```

---

## 📝 **Technical Details**

### Measurement Calculation

```python
# In pixel coordinates (Y increases downward)
right_wrist_to_shoulder_height = right_shoulder_coords[1] - right_wrist_coords[1]

Examples:
  Shoulder Y: 250px
  Wrist Y: 220px (above shoulder)
  → height = 250 - 220 = 30px ✅ PASS (exactly at threshold)
  
  Shoulder Y: 250px
  Wrist Y: 225px (above shoulder)
  → height = 250 - 225 = 25px ❌ FAIL (below threshold)
  
  Shoulder Y: 250px
  Wrist Y: 200px (well above shoulder)
  → height = 250 - 200 = 50px ✅ PASS (well above threshold)
```

### Why 30px is Appropriate

**Typical Resolution**: 1280x720 or 1920x1080

**Person Height in Frame**: ~400-600px (torso visible)

**30px** = approximately:
- 5-8% of person's torso height
- Roughly hand raised to mid-head level
- Clearly distinguishable from shoulder-level operations

**This is the "medium-high" zone** - hand is notably elevated but not at maximum height.

---

## ✅ **Implementation Checklist**

- [x] Changed threshold from 50px to 30px
- [x] Updated debug logging to show "[LOWERED]" marker
- [x] Updated version marker to 'v4.0-B-shoulder-30px'
- [x] No linter errors
- [x] Documentation created
- [ ] **NEXT**: Test with same video
- [ ] **NEXT**: Verify frame 10400 region detected
- [ ] **NEXT**: Verify frame 6650 still detected
- [ ] **NEXT**: Verify low reaches still rejected

---

## 🎯 **Success Criteria**

The 30px threshold implementation is successful if:

1. ✅ **Frame 6650**: Still detected (high reach)
2. ✅ **Frame 10400**: Now detected (medium-high reach)
3. ✅ **Low reaches**: Still rejected (shoulder-level or below)
4. ✅ **Total detections**: 2-4 LP hand gesture clips
5. ✅ **False positive rate**: < 20% (acceptable for production)

---

**Status**: ✅ Ready for Final Testing  
**Recommendation**: This 30px threshold should provide the balanced detection behavior the user requested.


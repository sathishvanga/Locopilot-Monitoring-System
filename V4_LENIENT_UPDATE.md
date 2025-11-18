# V4.0-LENIENT - Adjusted for High Reach Detection

## 🔄 **Change Summary**

**User Requirement**: Frame 6650 (high reach to control panel) should be **DETECTED** as a hand gesture.

**Problem**: V4.0 with strict verticality (≥ 2.0) was rejecting high forward reaches.

**Solution**: Lowered arm verticality threshold from **2.0 → 1.5** to accept up-and-forward arm movements.

---

## 🔧 **Technical Changes**

### Updated Thresholds (line 1492-1493):

```python
# Before (v4.0 strict):
right_arm_is_vertical = right_arm_verticality >= 2.0  # Very strict
left_arm_is_vertical = left_arm_verticality >= 2.0

# After (v4.0-lenient):
right_arm_is_vertical = right_arm_verticality >= 1.5  # More lenient
left_arm_is_vertical = left_arm_verticality >= 1.5
```

### What This Means

**Arm Verticality Ratio**: `vertical_distance / horizontal_distance`

| Arm Motion | Ratio | v4.0 Strict | v4.0-Lenient |
|------------|-------|-------------|--------------|
| Straight up | 3.0+ | ✅ Detect | ✅ Detect |
| Up-and-slightly-forward | 2.0-3.0 | ✅ Detect | ✅ Detect |
| Up-and-forward (45° ish) | 1.5-2.0 | ❌ Reject | **✅ Detect** |
| Forward-and-up | 1.0-1.5 | ❌ Reject | ❌ Reject |
| Mostly forward | < 1.0 | ❌ Reject | ❌ Reject |

---

## 🎯 **Mandatory Criteria (v4.0-lenient)**

ALL THREE must pass:

```python
1. Hand Height:       wrist_to_nose_vertical >= -30  (allow 30px below nose)
2. Arm Verticality:   arm_verticality >= 1.5          (UPDATED: was 2.0)
3. Control Zone:      not in_control_zone             (must not be in control zone)
```

---

## 📊 **Expected Behavior Changes**

### Frame 6650 (High Forward Reach)

**v4.0 Strict**:
- Hand height: PASS ✅
- Arm verticality (≥ 2.0): **FAIL** ❌ → REJECTED
- Result: Not detected

**v4.0-Lenient**:
- Hand height: PASS ✅  
- Arm verticality (≥ 1.5): **PASS** ✅
- Control zone: PASS ✅
- Result: **DETECTED** ✅

### Low/Medium Forward Reaches (Main False Positives)

**Frames 10750, 11300, 13100, etc.**:
- Hand height: **FAIL** ❌ (hand is below head level)
- Result: Still **REJECTED** ✅

**Key Point**: These frames will still be rejected because they fail the **hand height check** (hand is not high enough), regardless of arm angle.

---

## ⚖️ **Trade-off Analysis**

### Benefits ✅
- Accepts high reaches (even to control panel) when hand is elevated
- More practical for real-world cockpit operations
- Still enforces hand must be HIGH (within 30px of nose)

### Potential Risks ⚠️
- May accept some diagonal reaches that aren't true signals
- Relies more heavily on hand height check (less on arm angle)

### Mitigation 🛡️
- Hand height check (-30px) remains strict
- Control zone check still active
- Only accepts reaches where hand is demonstrably HIGH

---

## 🧪 **Testing**

Run the same video again with v4.0-lenient:

```bash
curl -X POST "http://localhost:8000/api/jobs" \
  -F "video=@example_data/latest.mp4" \
  -F "tripId=V4-LENIENT-$(date +%s)" \
  -F "lpCrewName=Test Pilot" \
  -F "lpCrewId=LP-001" \
  -F "saveClips=true" \
  -F "enableGestureDebug=true" \
  -F "gestureSensitivity=balanced"
```

**Expected Results**:
- ✅ Frame 6650: **DETECTED** (high forward reach)
- ✅ Frame 20450: **DETECTED** (true vertical signal, if present)
- ❌ Frames 10750, 11300, etc.: **REJECTED** (low/medium reaches)
- **Total LP detections: 1-3** (up from 0, but still down from original 9)

---

## 📝 **Version History**

- **v2.0**: Blacklist approach with control zone filtering → 9 false positives
- **v3.0**: Enhanced geometric analysis (blacklist) → 9 false positives (no improvement)
- **v4.0 strict**: Whitelist with strict criteria (≥ 2.0 verticality) → 0 detections (too strict)
- **v4.0-lenient**: Whitelist with lenient criteria (≥ 1.5 verticality) → **CURRENT**

---

## 🔄 **Rollback Plan**

If v4.0-lenient produces too many false positives, we can:

1. **Tighten verticality back to 1.8**:
   ```python
   right_arm_is_vertical = right_arm_verticality >= 1.8
   ```

2. **Tighten hand height to exactly at nose**:
   ```python
   right_hand_at_head = right_wrist_to_nose_vertical >= 0  # No compromise
   ```

3. **Add temporal filter** (require sustained high position for 2-3 frames)

---

**Status**: ✅ Implemented  
**Version**: v4.0-lenient  
**Date**: November 18, 2025  
**Rationale**: Accept high reaches as valid detections per user requirement


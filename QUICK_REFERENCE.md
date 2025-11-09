# 🎯 Temporal Filtering - Quick Reference Card

## What Changed?

### ❌ BEFORE (False Positives)
```
Frame 1: Hand near bag → INSTANT RECORDING STARTS
Frame 2: Hand away → Recording ends (0.04s)
Result: FALSE POSITIVE saved as evidence ❌
```

### ✅ AFTER (Temporally Validated)
```
Frames 1-149: Hand near bag → Counting... (not recording yet)
Frame 150: Hand still near bag → NOW START RECORDING ✅
Frames 151-300: Continue packing → Recording...
End: Duration 5.2s → Meets threshold → SAVE EVIDENCE ✅

OR

Frames 1-50: Hand near bag → Counting...
Frame 51: Hand away → Reset counter (never started) ✅
```

---

## 📊 Activity Settings at a Glance

| Activity | Consecutive Frames Needed | Minimum Duration | Proximity |
|----------|---------------------------|------------------|-----------|
| 🎒 **Packing Bags** | **150 frames** (5 sec) | **5.0 seconds** | 30px (tight) |
| ✍️ **Writing** | **90 frames** (3 sec) | **3.0 seconds** | 50px |
| 📱 **Cell Phone** | **60 frames** (2 sec) | **2.0 seconds** | 50px |
| 😴 **Microsleep** | 30 frames (1 sec) | **30.0 seconds** | N/A |
| 💤 **Sleep** | 30 frames (1 sec) | **180.0 seconds** | N/A |

---

## 🔄 Two-Gate Filtering System

### Gate 1: Consecutive Frames (Before Recording Starts)
**Purpose:** Ensure activity is sustained before starting to record

```python
if consecutive_detections >= 150:  # For packing
    START_RECORDING()
```

### Gate 2: Minimum Duration (When Recording Ends)
**Purpose:** Ensure recorded activity meets minimum length

```python
if duration < 5.0:  # For packing
    DISCARD_EVIDENCE()
else:
    SAVE_EVIDENCE()
```

---

## 🎬 Example Scenarios

### Scenario 1: Real Packing Activity ✅
```
Time     | Action                | Consecutive | Status
---------|----------------------|-------------|------------------
0:00:00  | Hand near bag        | 1          | Counting...
0:00:01  | Hand still near      | 30         | Counting...
0:00:04  | Hand still near      | 120        | Counting...
0:00:05  | Hand still near      | 150        | ✅ START RECORDING
0:00:06  | Packing continues    | 180        | Recording...
0:00:10  | Packing continues    | 300        | Recording...
0:00:12  | Hand away            | 0          | END (12s > 5s ✅)
Result: SAVED as evidence
```

### Scenario 2: Brief Touch (False Positive) ❌
```
Time     | Action                | Consecutive | Status
---------|----------------------|-------------|------------------
0:00:00  | Hand near bag        | 1          | Counting...
0:00:01  | Hand near bag        | 30         | Counting...
0:00:02  | Hand away            | 0          | RESET (never started)
Result: Nothing recorded (correct!)
```

### Scenario 3: Started but Too Short ❌
```
Time     | Action                | Consecutive | Status
---------|----------------------|-------------|------------------
0:00:00  | Hand near bag        | 1          | Counting...
0:00:05  | Hand still near      | 150        | ✅ START RECORDING
0:00:06  | Hand still near      | 180        | Recording...
0:00:06.5| Hand away            | 0          | END (1.5s < 5s ❌)
Result: DISCARDED (too short)
```

---

## 🔧 How to Adjust Thresholds

### If Missing Real Activities (Too Strict)
```python
# In __init__ method, reduce thresholds:
'packing_bags': {
    'min_duration': 3.0,          # Was 5.0
    'required_consecutive': 90,   # Was 150
    'margin': 40                  # Was 30
}
```

### If Still Getting False Positives (Too Lenient)
```python
# In __init__ method, increase thresholds:
'packing_bags': {
    'min_duration': 7.0,          # Was 5.0
    'required_consecutive': 210,  # Was 150
    'margin': 20                  # Was 30
}
```

---

## 📝 JSON Output Changes

### New Fields Added:
```json
{
    "activity_duration_seconds": 12.5,
    "min_duration_threshold": 5.0,           ← NEW!
    "required_consecutive_frames": 150,      ← NEW!
    "video_clip": "packing_bags_0001.mp4"
}
```

These fields help you understand why an activity was saved and what thresholds it met.

---

## 🚀 Running the System

```python
# In locopilot_monitor.py (already configured)
monitor = LocopilotActivityMonitor(
    video_path="latest_1.mp4", 
    output_dir="locopilot_evidence",
    save_annotated_frames=True
)

monitor.process_video()
```

**Expected Console Output:**
```
Progress: 300/18000 frames (1.7%)
[0:00:15] Activity started: packing_bags
[0:00:27] Activity ended: packing_bags
  Activity Duration: 12.50s | Total Clip: 17.50s
  Min Duration Threshold: 5.0s | Required Consecutive: 150 frames
  Evidence saved: packing_bags_0001.mp4 (525 frames)

[0:00:45.80] Activity 'packing_bags' too short (0.80s < 5.0s) - discarded
```

---

## 🎯 Key Benefits

| Benefit | Description |
|---------|-------------|
| 🛡️ **No More False Positives** | 0.04s detections eliminated |
| 📊 **Better Evidence Quality** | Only real sustained activities |
| ⚡ **Resource Efficient** | Fewer clips to review |
| 🔧 **Configurable** | Easy to tune per activity |
| 🎯 **Accurate** | Multiple bag types, tighter margins |

---

## 🧪 Quick Test

1. **Run on your video:**
   ```bash
   python locopilot_monitor.py
   ```

2. **Check the JSON files:**
   ```bash
   cat locopilot_evidence/json/packing_bags_0000.json
   ```

3. **Look for:**
   - `activity_duration_seconds` ≥ 5.0 for packing
   - `min_duration_threshold` in output
   - Console messages about discarded activities

4. **Expected:** No more 0.04s detections! 🎉

---

## 📞 Support

If you see:
- **Too many false positives still:** Increase thresholds
- **Missing real activities:** Decrease thresholds
- **Clips too short:** Check minimum duration settings
- **Not detecting bags:** Check margin settings

All configurable in the `self.activity_thresholds` dictionary in `__init__()`.


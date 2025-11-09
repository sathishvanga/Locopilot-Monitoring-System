# Temporal Filtering Implementation - Activity Detection Enhancement

## Overview
This implementation adds **temporal filtering with consecutive frame detection** to eliminate false positives in activity detection. Activities must now be detected consistently over multiple consecutive frames before recording begins, and must meet minimum duration thresholds to be saved as evidence.

---

## 🎯 Key Problems Solved

### Before Implementation:
- ❌ **0.04 second detections** - Single frame false positives
- ❌ **Hand briefly passing near bags** triggered recording
- ❌ **YOLO misdetections** caused spurious clips
- ❌ **No temporal consistency** - instant triggers

### After Implementation:
- ✅ **Consecutive frame requirement** - Activities must persist
- ✅ **Minimum duration thresholds** - Filters out brief interactions
- ✅ **Tighter proximity margins** - More accurate hand-object detection
- ✅ **Enhanced bag detection** - Backpack, handbag, and suitcase

---

## 📊 Activity Thresholds

| Activity | Min Duration | Required Consecutive Frames | Margin | Reasoning |
|----------|--------------|----------------------------|--------|-----------|
| **Packing Bags** | 5.0s | 150 frames (@30fps) | 30px | Packing takes time - filters out brief touches |
| **Writing** | 3.0s | 90 frames (@30fps) | 50px | Writing is sustained activity |
| **Cell Phone** | 2.0s | 60 frames (@30fps) | 50px | Quick glances OK, but needs confirmation |
| **Microsleep** | 30.0s | 30 frames (@30fps) | N/A | Eyes must stay closed for 30s |
| **Sleep** | 180.0s | 30 frames (@30fps) | N/A | Extended eye closure for 3 minutes |

---

## 🔄 Detection Flow

### Example: Packing Bags Detection

```
Frame 1:    Hand near bag → consecutive_detections = 1    (not recording yet)
Frame 2:    Hand near bag → consecutive_detections = 2    (not recording yet)
Frame 3:    Hand near bag → consecutive_detections = 3    (not recording yet)
...
Frame 149:  Hand near bag → consecutive_detections = 149  (not recording yet)
Frame 150:  Hand near bag → consecutive_detections = 150  ✅ START RECORDING
Frame 151:  Hand near bag → Recording continues...
Frame 152:  Hand near bag → Recording continues...
...
Frame 300:  Hand away    → consecutive_detections = 0     → Check duration
            Duration = 5.2s → Meets threshold (≥5s) → ✅ SAVE EVIDENCE
```

### Rejected Detection Example:

```
Frame 1:    Hand near bag → consecutive_detections = 1
Frame 2:    Hand near bag → consecutive_detections = 2
Frame 3:    Hand away     → consecutive_detections = 0    ❌ RESET (never started)

OR

Frame 150:  Started recording (5s sustained)
Frame 180:  Ended → Duration = 1.0s → Below threshold (5s) → ❌ DISCARDED
```

---

## 🛠️ Implementation Details

### 1. Activity Thresholds Configuration

```python
self.activity_thresholds = {
    'packing_bags': {
        'min_duration': 5.0,          # Must last 5 seconds minimum
        'required_consecutive': 150,  # 5 seconds @ 30fps
        'margin': 30                  # Tighter proximity check (30px vs 50px)
    },
    # ... other activities
}
```

### 2. Consecutive Detection Tracking

```python
self.consecutive_detections = {
    'microsleep': 0,
    'sleep': 0,
    'cell_phone': 0,
    'writing': 0,
    'packing_bags': 0
}
```

### 3. Enhanced Bag Detection

```python
# Detect multiple bag types
all_bags = (detections['backpack'] + 
           detections['handbag'] + 
           detections['suitcase'])

# Use tighter margin for packing (30px instead of 50px)
margin = self.activity_thresholds['packing_bags']['margin']

for bag_bbox in all_bags:
    if (self.check_hand_object_interaction(right_hand_coords, bag_bbox, margin) or
        self.check_hand_object_interaction(left_hand_coords, bag_bbox, margin)):
        packing_detected = True
```

### 4. Temporal Filtering Logic

```python
for activity_name, detected in activities_map.items():
    if detected:
        # Increment consecutive detection counter
        self.consecutive_detections[activity_name] += 1
        
        # Only start recording after required consecutive frames threshold is met
        required_consecutive = self.activity_thresholds[activity_name]['required_consecutive']
        
        if self.consecutive_detections[activity_name] >= required_consecutive:
            # Start activity if not already active
            if not self.activities[activity_name]['active']:
                self.start_activity(activity_name, timestamp, fps, frame_count)
            
            # Continue recording frames
            if self.activities[activity_name]['active']:
                self.activities[activity_name]['frames'].append(frame.copy())
    else:
        # Activity not detected - reset consecutive counter and end if active
        if self.activities[activity_name]['active']:
            self.end_activity(activity_name, timestamp, fps, frame_count)
        self.consecutive_detections[activity_name] = 0
```

### 5. Duration Validation on Activity End

```python
def end_activity(self, activity_name, timestamp, fps, frame_count):
    """End tracking an activity and save evidence (only if meets minimum duration)"""
    if self.activities[activity_name]['active']:
        # Calculate actual duration
        duration = (frame_count - start_frame) / fps
        
        # Check if activity meets minimum duration threshold
        min_duration = self.activity_thresholds[activity_name]['min_duration']
        
        if duration < min_duration:
            print(f"Activity '{activity_name}' too short ({duration:.2f}s < {min_duration}s) - discarded")
            return  # Don't save evidence
        
        # Duration met - save evidence
        self.save_video_clip(...)
```

---

## 📄 Enhanced JSON Output

```json
{
    "activity_name": "packing_bags",
    "start_time": "0:00:15.000000",
    "end_time": "0:00:27.500000",
    "activity_duration_seconds": 12.5,
    "total_clip_duration_seconds": 17.5,
    "total_frames_in_clip": 525,
    "min_duration_threshold": 5.0,
    "required_consecutive_frames": 150,
    "includes_pre_buffer": true,
    "pre_buffer_seconds": 5.0,
    "video_clip": "packing_bags_0001.mp4",
    "evidence_id": 0,
    "timestamp": "2025-11-09T23:40:10.123456"
}
```

**New Fields:**
- `min_duration_threshold`: Minimum duration required for this activity type
- `required_consecutive_frames`: Number of consecutive frames needed to start recording

---

## 🎬 Processing Output Example

### Previous System (False Positives):
```
[0:00:15] Activity started: packing_bags
[0:00:15.04] Activity ended: packing_bags
  Activity Duration: 0.04s | Total Clip: 5.04s
  Evidence saved: packing_bags_0001.mp4 (152 frames)
```
❌ **This was clearly a false positive!**

### New System (Filtered):
```
Progress: 300/18000 frames (1.7%)
[0:00:15] Activity started: packing_bags  ← Only after 150 consecutive frames
[0:00:27] Activity ended: packing_bags
  Activity Duration: 12.50s | Total Clip: 17.50s
  Min Duration Threshold: 5.0s | Required Consecutive: 150 frames
  Evidence saved: packing_bags_0001.mp4 (525 frames)
```
✅ **Real packing activity detected and saved!**

### New System (Rejected):
```
Progress: 300/18000 frames (1.7%)
[0:00:15.80] Activity 'packing_bags' too short (0.80s < 5.0s) - discarded
```
✅ **False positive filtered out!**

---

## 🔍 Variable Proximity Margins

Different activities use different proximity margins for hand-object interaction:

```python
def check_hand_object_interaction(self, hand_coords, object_bbox, margin=50):
    """
    Args:
        margin: proximity margin in pixels
                - 30px for packing (tighter check - must be very close)
                - 50px for phone/writing (more lenient)
    """
    hx, hy = hand_coords
    x1, y1, x2, y2 = object_bbox
    return (x1 - margin <= hx <= x2 + margin and 
            y1 - margin <= hy <= y2 + margin)
```

**Rationale:**
- **Packing (30px)**: Must be very close to bag to count as packing - reduces false positives from hands passing nearby
- **Phone/Writing (50px)**: Can be slightly away from object while still interacting

---

## 🚀 Benefits

### 1. **Eliminates False Positives**
- No more 0.04s detections
- Only real sustained activities are recorded

### 2. **Better Evidence Quality**
- All saved clips meet minimum duration thresholds
- Captures actual activities, not momentary gestures

### 3. **Resource Efficiency**
- Fewer clips to review
- Less storage used for false positives
- Faster processing with meaningful results

### 4. **Configurable Thresholds**
- Easy to adjust per activity type
- Can fine-tune based on real-world testing
- Balances sensitivity vs. specificity

### 5. **Enhanced Detection Accuracy**
- Multiple bag types supported
- Tighter margins for specific activities
- Better hand-object interaction logic

---

## 📝 Testing Recommendations

### 1. Test with Known False Positives
- Brief hand movements near objects
- YOLO misdetections
- Quick glances vs. sustained usage

### 2. Test with Real Activities
- Actual packing for 10+ seconds
- Writing for 5+ seconds
- Phone usage for 3+ seconds

### 3. Tune Thresholds if Needed
```python
# If too sensitive (missing real activities):
'packing_bags': {
    'min_duration': 3.0,          # Reduce from 5.0s
    'required_consecutive': 90,   # Reduce from 150 frames
    'margin': 40                  # Increase from 30px
}

# If too lenient (still getting false positives):
'packing_bags': {
    'min_duration': 7.0,          # Increase from 5.0s
    'required_consecutive': 210,  # Increase from 150 frames
    'margin': 20                  # Decrease from 30px
}
```

---

## 🎯 Summary

This implementation transforms the activity detection system from **instant-trigger** to **temporally-validated**, ensuring that only genuine, sustained activities are recorded as evidence. The dual-gate approach (consecutive frames + minimum duration) provides robust filtering while maintaining high accuracy for real activities.

**Result:** No more 0.04s false positives! 🎉


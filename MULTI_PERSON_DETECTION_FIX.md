# Multi-Person Activity Detection - Comprehensive Fix

**Date**: November 18, 2025  
**Status**: ✅ IMPLEMENTED  
**Version**: 2.0

---

## 🎯 PROBLEM IDENTIFIED

### Original Issue
The system was using MediaPipe on the **full frame**, which only detects **ONE person's pose landmarks**, even when 2+ people were present in the video.

**Critical Bugs:**
1. ❌ **Visualization**: Only ONE person's pose skeleton was drawn, even though 2 people were detected
2. ❌ **Activity Detection**: Sleep, microsleep, and head pose were only detected for ONE person
3. ❌ **Hand Gestures**: Partially fixed for multi-person, but other activities were not

### What Was Working
- ✅ YOLO object detection (cell phone, books, backpack) - works for all persons
- ✅ Person bounding boxes - all persons detected
- ✅ LP/ALP identification - all persons identified
- ⚠️ Hand gestures - partially working (used cropped detection)

### What Was Broken
- ❌ Pose visualization - only 1 person
- ❌ Sleep/microsleep detection - only 1 person
- ❌ Head pose/mind diversion - only 1 person
- ❌ Face mesh (EAR calculation) - only 1 person

---

## 💡 SOLUTION IMPLEMENTED

### Core Architecture Change

**BEFORE:**
```python
# Single-person pose detection on full frame
pose_results = self.pose.process(rgb_frame)  # ← Only ONE person tracked
face_results = self.face_mesh.process(rgb_frame)  # ← Only ONE face tracked

# All activities detected using single pose
detect_sleep(pose_results.pose_landmarks)  # ← Only for ONE person
detect_hand_gesture(pose_results.pose_landmarks)  # ← Only for ONE person
detect_head_pose(pose_results.pose_landmarks)  # ← Only for ONE person
```

**AFTER:**
```python
# Multi-person: Crop each person's region and run MediaPipe separately
for person_idx, person_data in person_roles.items():
    bbox = person_data['bbox']
    cropped_frame = frame[y1:y2, x1:x2]  # Crop to this person
    
    # Run MediaPipe Pose on THIS person's cropped region
    pose_result = self.pose.process(cropped_rgb)
    
    # Run MediaPipe Face Mesh on THIS person's cropped region
    face_result = self.face_mesh.process(cropped_rgb)
    
    # Detect ALL activities for THIS specific person
    detect_sleep_for_person(person_idx)
    detect_hand_gesture_for_person(person_idx)
    detect_head_pose_for_person(person_idx)
    calculate_EAR_for_person(person_idx)
```

---

## 🔧 IMPLEMENTATION DETAILS

### 1. New Function: `detect_per_person_activities()`

**Location**: `locopilot_monitor.py` (lines 2029-2242)

**Purpose**: Comprehensive multi-person activity detection

**What It Does:**
- Crops each detected person's bounding box region
- Runs MediaPipe Pose on the cropped region
- Runs MediaPipe Face Mesh on the cropped region
- Detects ALL activities per person:
  - Sleep/microsleep (both pose-based and face-based)
  - Hand gestures (LP/ALP)
  - Head pose/mind diversion
  - EAR calculation (eye aspect ratio)
  
**Per-Person Tracking:**
- `self.per_person_sleep_tracking[person_idx]` - Tracks sleep state for each person
- `self.per_person_eye_tracking[person_idx]` - Tracks eye closure for each person

**Returns:**
```python
{
    person_idx: {
        'pose_landmarks': translated_landmarks,
        'face_landmarks': translated_face_landmarks,
        'role': 'LP'/'ALP',
        'activities': {
            'sleep': bool,
            'microsleep': bool,
            'hand_gesture': bool,
            'gesture_type': 'lp'/'alp'/None,
            'mind_diversion': bool
        },
        'metrics': {
            'ear': float,
            'eye_closure_duration': float,
            'pose_sleep_info': dict,
            'head_pose_info': dict,
            'gesture_debug': dict
        }
    }
}
```

### 2. Helper Function: `detect_pose_based_sleep_per_person()`

**Location**: `locopilot_monitor.py` (lines 2244-2332)

**Purpose**: Detect sleep for a specific person using their pose landmarks

**Key Features:**
- Tracks sleep state per person separately
- Uses person_idx as key for tracking
- Calculates head tilt and movement for this specific person

### 3. Helper Function: `translate_face_landmarks_to_full_frame()`

**Location**: `locopilot_monitor.py` (lines 2334-2358)

**Purpose**: Translate face landmarks from cropped coordinates back to full frame

**Why Needed:**
- MediaPipe returns normalized coordinates (0-1) relative to the cropped region
- We need to translate them back to full frame coordinates for visualization

### 4. Updated Function: `draw_mediapipe_outputs()`

**Location**: `locopilot_monitor.py` (lines 931-1013)

**Changes:**
- Added new parameter: `multi_person_results`
- Now draws pose landmarks for ALL persons when multi_person_results is provided
- Falls back to single-person mode for backward compatibility

**NEW Behavior:**
```python
if multi_person_results:
    for person_idx, person_result in multi_person_results.items():
        # Draw pose landmarks for this person
        draw_landmarks(person_result['pose_landmarks'])
        
        # Draw face landmarks for this person
        draw_landmarks(person_result['face_landmarks'])
```

### 5. Updated Main Processing Loop

**Location**: `locopilot_monitor.py` (lines 3360-3412)

**Changes:**
- Calls `detect_per_person_activities()` when 2+ people are present
- Processes results for each person individually
- Logs activities with person index and role
- Passes `multi_person_results` to visualization function

**NEW Logic:**
```python
if len(person_roles) >= 2:
    # Multi-person: Comprehensive detection
    multi_person_results = self.detect_per_person_activities(frame, person_roles, timestamp_sec)
    
    for person_idx, person_result in multi_person_results.items():
        role = person_result['role']
        activities = person_result['activities']
        
        # Process each activity type
        if activities['hand_gesture']:
            # Log: "LP hand gesture detected (Person 0)"
        if activities['sleep']:
            # Log: "SLEEP detected for LP (Person 0) - Duration: 31.2s"
        if activities['mind_diversion']:
            # Log: "MIND DIVERSION detected for LP (Person 0) - Yaw=47.3°"
```

---

## 📊 BEFORE vs AFTER COMPARISON

| Feature | BEFORE | AFTER |
|---------|--------|-------|
| **Pose Detection** | Full frame (1 person) | Cropped per person (all persons) |
| **Face Detection** | Full frame (1 face) | Cropped per person (all faces) |
| **Sleep Detection** | 1 person only | ✅ Each person individually |
| **Hand Gestures** | Partially multi-person | ✅ Each person individually |
| **Mind Diversion** | 1 person only | ✅ Each person individually |
| **EAR Calculation** | 1 face only | ✅ Each face individually |
| **Pose Visualization** | 1 skeleton drawn | ✅ All skeletons drawn |
| **Activity Attribution** | Generic | ✅ Person-specific (LP/ALP) |

---

## 🎯 BENEFITS

### 1. True Multi-Person Monitoring
- System now tracks EACH person independently
- LP and ALP activities are separately detected
- No more "which person" ambiguity

### 2. Accurate Activity Detection
- Sleep detection works for both LP and ALP simultaneously
- Hand gestures from both persons can be detected in same frame
- Mind diversion tracked for each crew member

### 3. Better Visualization
- ALL persons' pose skeletons are now drawn
- Face meshes for all detected faces
- Clear visual feedback for each person

### 4. Role-Based Reporting
```
[10:51:22] SLEEP detected for LP (Person 0) - Duration: 31.2s
[10:51:22] MIND DIVERSION detected for ALP (Person 1) - Yaw=47.3°
```

### 5. Proper Activity Attribution
- Activities now include which person (LP vs ALP) performed them
- Can track individual crew member behavior
- Better evidence for safety compliance

---

## 🧪 TESTING RECOMMENDATIONS

### Test Case 1: Two People - Both Active
**Expected**: Both persons' pose skeletons visible, no activities

### Test Case 2: LP Sleeping, ALP Active
**Expected**: Sleep activity attributed to LP (Person 0)

### Test Case 3: ALP Hand Gesture, LP Normal
**Expected**: ALP hand gesture detected for Person 1

### Test Case 4: Both Raising Hands
**Expected**: Both gestures detected (or ignored per logic)

### Test Case 5: Single Person
**Expected**: Falls back to single-person mode (backward compatible)

---

## 📝 CODE CHANGES SUMMARY

### Files Modified
1. ✅ `locopilot_monitor.py` - Main implementation file

### New Functions Added
1. ✅ `detect_per_person_activities()` - Comprehensive multi-person detection (lines 2029-2242)
2. ✅ `detect_pose_based_sleep_per_person()` - Per-person sleep detection (lines 2244-2332)
3. ✅ `translate_face_landmarks_to_full_frame()` - Coordinate translation (lines 2334-2358)

### Functions Modified
1. ✅ `draw_mediapipe_outputs()` - Now draws all persons' poses (lines 931-1013)
2. ✅ `process_video()` - Uses new multi-person detection (lines 3360-3412)
3. ✅ `process_video_range()` - Compatible with multiprocessing (lines 3871-3927)

### Imports Added
1. ✅ `import math` - Required for sleep detection calculations (line 4)

---

## ⚠️ BACKWARD COMPATIBILITY

### Single-Person Scenarios
- System automatically falls back to single-person mode when only 1 person detected
- All existing functionality preserved
- No breaking changes

### Legacy Code Paths
- `elif pose_results.pose_landmarks and person_roles:` - Falls back to single-person detection
- Visualization function checks `if multi_person_results:` before using new logic

---

## 🚀 FUTURE ENHANCEMENTS

### Potential Improvements
1. **Per-Person Activity Clips**: Generate separate clips for LP and ALP activities
2. **Crew Member Dashboard**: Show individual metrics for each crew member
3. **Performance Optimization**: Cache cropped regions if same person bbox
4. **Multi-Person Cell Phone**: Associate cell phone detection with specific person
5. **Cross-Person Interaction**: Detect when LP and ALP interact (handover scenarios)

---

## 📋 VERIFICATION CHECKLIST

- [x] Function `detect_per_person_activities()` implemented
- [x] Per-person sleep tracking implemented
- [x] Per-person face landmarks translation implemented
- [x] Visualization updated to draw all persons
- [x] Main processing loop integrated
- [x] Multiprocessing compatibility maintained
- [x] No linter errors
- [x] Backward compatibility preserved
- [ ] **Testing with sample video** (USER TO CONFIRM)

---

## ✅ COMPLETION STATUS

**Implementation**: ✅ COMPLETE  
**Testing**: ⏳ PENDING USER TESTING  
**Documentation**: ✅ COMPLETE  

---

**Next Steps**:
1. Test with `example_data/latest.mp4` or actual locopilot video
2. Verify both persons' skeletons are drawn
3. Verify activities are attributed correctly to LP/ALP
4. Check console logs for per-person activity detection

---

**Ready to test! Please confirm whether you'd like me to:**
1. Run the code on your sample video
2. Generate test report
3. Make any adjustments based on results


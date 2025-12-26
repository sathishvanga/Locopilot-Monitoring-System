# CVVRS False Positive Reduction - Implementation Plan

**Date:** December 27, 2024
**Version:** 1.0
**Status:** Implemented

---

## Executive Summary

This document describes the comprehensive plan to reduce false positives in the CVVRS (Cabin Video Voice Recording System) activity detection system. Based on analysis of 12 false positive cases from field testing, we identified 5 root cause categories and implemented a 3-phase fix plan.

---

## Table of Contents

1. [Problem Analysis](#1-problem-analysis)
2. [Root Cause Analysis](#2-root-cause-analysis)
3. [Implementation Plan](#3-implementation-plan)
4. [Technical Details](#4-technical-details)
5. [Configuration Reference](#5-configuration-reference)
6. [Testing & Validation](#6-testing--validation)

---

## 1. Problem Analysis

### 1.1 False Positive Cases Identified

From the CVVRS activity validation Excel file, we identified the following false positive patterns:

| Video | Total Activities | Correct | Wrong | Error Rate |
|-------|-----------------|---------|-------|------------|
| Video 1 (Night, 34min) | 14 | 7 | **7** | 50% |
| Video 2 (Day, 27min) | 50 | 45 | **5** | 10% |

### 1.2 False Positive Categories

| Category | Count | Description |
|----------|-------|-------------|
| Writing → Mind Diversion | 3 | LP writing in logbook detected as distracted |
| Writing → Packing Bags | 1 | LP writing detected as handling bags |
| Group Detection FP | 2 | System detected >2 people when only 2 present |
| Hand Gesture FP | 1 | False coordination failure alert |
| Misattribution | 5 | Group detection instead of specific gesture issue |

---

## 2. Root Cause Analysis

### 2.1 Writing Detected as "Mind Diversion"

**Root Cause:**
The mind diversion detection triggers when head is tilted down/sideways. When LP is legitimately writing in the logbook, their head naturally looks down - triggering both "writing" AND "mind diversion" simultaneously.

**Code Location:** `locopilot_monitor.py:3471-3478`

```python
# PROBLEM: Mind diversion detected before writing is confirmed
head_pose_info = self.calculate_head_pose_angles(...)
person_activities['mind_diversion'] = head_pose_info.get('detected', False)
```

**Issue:** The suppression mechanism existed but had timing gaps - mind diversion could trigger before writing detection was confirmed.

---

### 2.2 Writing Detected as "Packing Bags"

**Root Cause:**
The packing detection uses a 40px margin tolerance. When a bag is visible in the cabin and LP is writing, their wrist may be near the bag's bounding box, triggering false packing detection.

**Code Location:** `locopilot_monitor.py:3726-3809`

```python
# PROBLEM: No check if person is in writing posture
right_inside, right_dist = self.is_wrist_inside_backpack(
    right_hand_coords, backpack_bbox, margin=40
)
```

**Issue:** No cross-activity validation - system didn't check if person was in writing posture before detecting packing.

---

### 2.3 Group Detection False Positives

**Root Cause:**
Group detection triggers when `deduplicated_count > 2` after de-duplication. False positives occurred due to:
- Reflections in cabin windows/mirrors
- Low-confidence person detections
- Partial body detections counted as separate persons

**Code Location:** `locopilot_monitor.py:4803-4824`

```python
# PROBLEM: No confidence or size filtering
deduplicated_persons = self.deduplicate_person_boxes(detections['person'], iou_threshold=0.5)
if deduplicated_count > 2:
    group_detected_flag = True
```

---

### 2.4 Hand Gesture Coordination False Positives

**Root Cause:**
The coordination check flagged when one person raised hand but the other didn't respond within 5 seconds. Issues:
- Single-frame detection (`required_consecutive: 1`)
- Short coordination window (5 seconds)
- Normal hand movements falsely detected as gestures

**Code Location:** `locopilot_monitor.py:2883-2938`

---

## 3. Implementation Plan

### 3.1 Phase 1: Immediate Fixes (P0/P1)

#### Fix 1.1: Writing vs Mind Diversion - Priority Suppression

**Approach:** Check for writing posture BEFORE mind diversion detection.

```python
# PRE-CHECK: Writing posture indicators
writing_posture_detected = False
wrist_distance = self.calculate_wrist_distance(translated_landmarks, frame.shape)
head_looking_down = self.detect_head_looking_down(translated_landmarks)

if wrist_distance is not None and wrist_distance < 300 and head_looking_down:
    writing_posture_detected = True
    # Suppress mind diversion
    person_activities['mind_diversion'] = False
```

**Impact:** Eliminates false mind diversion alerts when LP is writing.

---

#### Fix 1.2: Writing vs Packing Bags - Mutual Exclusion

**Approach:** Skip packing detection if writing posture or activity detected.

```python
# MUTUAL EXCLUSION
packing_suppressed_by_writing = False
if writing_posture_detected or person_activities.get('writing', False):
    packing_suppressed_by_writing = True
    # Skip packing detection entirely
```

**Impact:** Eliminates false packing alerts when LP is writing.

---

#### Fix 1.3: Group Detection - Stricter Filtering

**Changes:**

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| Min bbox area | None | 10,000 px² | Filter partial detections |
| IOU threshold | 0.5 | 0.4 | More aggressive deduplication |
| required_consecutive | 3 | 5 | Require 10s confirmation |
| Person confidence | 0.5 | 0.5 | Already in place |

**Implementation:**

```python
# Filter by minimum bbox area
person_area = person_width * person_height
min_person_area = self.activity_thresholds['group_detected'].get('min_bbox_area', 10000)

if person_area >= min_person_area:
    detections['person'].append(xyxy)
```

---

#### Fix 1.4: Hand Gesture Coordination - Temporal Buffer

**Changes:**

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| Coordination window | 5.0s | 8.0s | More time to respond |
| LP gesture required_consecutive | 1 | 1 | Instant (voting handles smoothing) |
| ALP gesture required_consecutive | 1 | 1 | Instant (voting handles smoothing) |
| LP/ALP voting window_frames | N/A | 2 | Minimal smoothing |
| LP/ALP voting votes_required | N/A | 1 | Instant confirmation |
| Mind diversion required_consecutive | 2 | 4 | 8s confirmation |

---

### 3.2 Phase 2: Voting System (P2)

#### Fix 2.1: Activity Voting Service

**Concept:** An activity must be detected in N out of M frames to be confirmed.

**New File:** `app/services/activity_voting_service.py`

```python
class ActivityVotingService:
    def __init__(self):
        self.voting_config = {
            'writing': {'votes_required': 2, 'window_frames': 4},
            'mind_diversion': {'votes_required': 3, 'window_frames': 5},
            'packing_bags': {'votes_required': 2, 'window_frames': 4},
            'group_detected': {'votes_required': 4, 'window_frames': 6},
            'lp_hand_gesture': {'votes_required': 2, 'window_frames': 4},
            'alp_hand_gesture': {'votes_required': 2, 'window_frames': 4},
            'cell_phone': {'votes_required': 2, 'window_frames': 3},
        }

    def vote(self, activity_type: str, detected: bool, person_idx: int = None) -> bool:
        # Cast vote and return True only if votes >= votes_required
        ...
```

**Integration Point:** `locopilot_monitor.py:5064-5086`

---

### 3.3 Phase 3: Priority Matrix (P3)

#### Fix 3.2: Activity Priority Matrix

**Concept:** Define which activities suppress others when detected simultaneously.

**Configuration:** `app/utils/config.py`

```python
@property
def activity_priority_matrix(self) -> dict:
    return {
        # Writing suppresses mind_diversion and packing_bags
        'writing': ['mind_diversion', 'packing_bags'],

        # Cell phone suppresses writing
        'cell_phone': ['writing'],

        # Packing bags suppresses writing
        'packing_bags': ['writing'],

        # Sleep suppresses all active behaviors
        'sleep': ['writing', 'cell_phone', 'packing_bags', 'mind_diversion'],

        # Microsleep suppresses mind_diversion
        'microsleep': ['mind_diversion'],
    }
```

**Integration Point:** `locopilot_monitor.py:5088-5104`

---

## 4. Technical Details

### 4.1 Files Modified

| File | Type | Description |
|------|------|-------------|
| `locopilot_monitor.py` | Modified | Main detection logic with all fixes |
| `app/utils/config.py` | Modified | Added voting and priority matrix config |
| `app/services/activity_voting_service.py` | **New** | Voting system implementation |

### 4.2 Detection Flow (After Fixes)

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRAME PROCESSING                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. PRE-CHECK: Writing Posture Detection                        │
│     - Calculate wrist distance                                   │
│     - Check head looking down                                    │
│     - If writing posture → suppress mind_diversion early         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. INDIVIDUAL ACTIVITY DETECTION                                │
│     - Mind diversion (with pre-suppression)                      │
│     - Sleep/Microsleep                                           │
│     - Cell phone                                                 │
│     - Writing (book + pose-based)                                │
│     - Packing bags (with writing exclusion)                      │
│     - Hand gestures                                              │
│     - Group detection (with bbox filtering)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. VOTING FILTER (Fix 2.1)                                      │
│     - Each activity voted: N out of M frames required            │
│     - Single-frame detections filtered out                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. PRIORITY MATRIX (Fix 3.2)                                    │
│     - Apply activity conflict resolution                         │
│     - Higher-priority activities suppress lower-priority         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. CONSECUTIVE DETECTION THRESHOLD                              │
│     - Activity must pass required_consecutive frames             │
│     - Grace period for brief interruptions                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. ACTIVITY RECORDING                                           │
│     - Start/continue/end activity                                │
│     - Generate evidence clips                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Threshold Changes Summary

| Activity | Old Threshold | New Threshold | Change |
|----------|--------------|---------------|--------|
| `group_detected` | 3 consecutive | 5 consecutive | +67% stricter |
| `mind_diversion` | 2 consecutive | 4 consecutive | +100% stricter |
| `lp_hand_gesture` | 1 consecutive | 1 consecutive | No change (voting: 1/2) |
| `alp_hand_gesture` | 1 consecutive | 1 consecutive | No change (voting: 1/2) |
| Coordination window | 5.0 seconds | 8.0 seconds | +60% longer |

---

## 5. Configuration Reference

### 5.1 Environment Variables

```bash
# Voting System (Fix 2.1)
ACTIVITY_VOTING_ENABLED=1                    # Enable/disable voting (default: 1)

# Threshold Overrides (Fix 1.3, 1.4)
ACTIVITY_THRESHOLD_GROUP_REQUIRED_CONSECUTIVE=5
ACTIVITY_THRESHOLD_MIND_DIVERSION_REQUIRED_CONSECUTIVE=4
ACTIVITY_THRESHOLD_LP_HAND_GESTURE_REQUIRED_CONSECUTIVE=1
ACTIVITY_THRESHOLD_ALP_HAND_GESTURE_REQUIRED_CONSECUTIVE=1

# Hand Gesture Coordination (Fix 1.4)
HAND_GESTURE_COORDINATION_WINDOW=8.0         # Seconds
```

### 5.2 Voting Configuration

| Activity | Votes Required | Window Frames | Effective Filter |
|----------|---------------|---------------|------------------|
| `writing` | 2 | 4 | 2/4 = 50% |
| `mind_diversion` | 3 | 5 | 3/5 = 60% |
| `packing_bags` | 2 | 4 | 2/4 = 50% |
| `group_detected` | 4 | 6 | 4/6 = 67% |
| `lp_hand_gesture` | 1 | 2 | 1/2 = instant |
| `alp_hand_gesture` | 1 | 2 | 1/2 = instant |
| `cell_phone` | 2 | 3 | 2/3 = 67% |
| `microsleep` | 2 | 3 | 2/3 = 67% |
| `sleep` | 3 | 4 | 3/4 = 75% |

### 5.3 Priority Matrix

```
writing ────────► suppresses ────────► mind_diversion, packing_bags
cell_phone ─────► suppresses ────────► writing
packing_bags ───► suppresses ────────► writing
sleep ──────────► suppresses ────────► writing, cell_phone, packing_bags, mind_diversion
microsleep ─────► suppresses ────────► mind_diversion
```

---

## 6. Testing & Validation

### 6.1 Expected Improvements

Based on the root cause analysis, we expect:

| False Positive Type | Before | After | Improvement |
|---------------------|--------|-------|-------------|
| Writing → Mind Diversion | 3 cases | 0 cases | 100% |
| Writing → Packing Bags | 1 case | 0 cases | 100% |
| Group Detection FP | 2 cases | 0-1 cases | 50-100% |
| Hand Gesture FP | 1 case | 0 cases | 100% |

### 6.2 Validation Checklist

- [ ] Re-process Video 1 (night, 34min) - expect <2 false positives
- [ ] Re-process Video 2 (day, 27min) - expect <2 false positives
- [ ] Verify legitimate detections still work
- [ ] Check detection latency impact (voting may add slight delay)
- [ ] Monitor logs for suppression events

### 6.3 Logging for Debugging

New log messages added:

```
DEBUG: Person X: WRITING POSTURE detected (wrists=XXpx, head_down=True)
DEBUG: Person X: Mind diversion SUPPRESSED due to writing posture
DEBUG: Person X: PACKING SUPPRESSED - writing posture/activity detected
DEBUG: VOTING: activity_name detected but not yet confirmed (building votes)
DEBUG: PRIORITY: suppressed_activity SUPPRESSED by primary_activity
DEBUG: PERSON FILTERED: conf=X.XX area=XXXX (below min 10000)
```

---

## Appendix A: Code Diff Summary

### Key Changes in `locopilot_monitor.py`

1. **Line 312:** Hand gesture coordination window: `5.0` → `8.0`
2. **Line 314-322:** Voting service initialization
3. **Line 337-344:** Group detection thresholds updated
4. **Line 345-362:** Hand gesture and mind diversion thresholds updated
5. **Line 1401-1412:** Person bbox area filtering
6. **Line 3483-3511:** Writing posture pre-check and mind diversion suppression
7. **Line 3714-3722:** Packing bags mutual exclusion with writing
8. **Line 4815:** IOU threshold change for deduplication
9. **Line 5064-5086:** Voting filter integration
10. **Line 5088-5104:** Priority matrix integration

---

## Appendix B: Rollback Instructions

If issues arise, disable fixes via environment variables:

```bash
# Disable voting system
ACTIVITY_VOTING_ENABLED=0

# Restore original thresholds
ACTIVITY_THRESHOLD_GROUP_REQUIRED_CONSECUTIVE=3
ACTIVITY_THRESHOLD_MIND_DIVERSION_REQUIRED_CONSECUTIVE=2
ACTIVITY_THRESHOLD_LP_HAND_GESTURE_REQUIRED_CONSECUTIVE=1
ACTIVITY_THRESHOLD_ALP_HAND_GESTURE_REQUIRED_CONSECUTIVE=1
HAND_GESTURE_COORDINATION_WINDOW=5.0
```

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-12-27 | Initial implementation |
| 1.1 | 2024-12-27 | Reverted hand gesture thresholds to instant (1 consecutive, 1/2 voting) |

---

**Document End**

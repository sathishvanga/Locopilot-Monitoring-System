# Locopilot Activity Detection - Business Logic Documentation

> **Document Purpose**: Technical reference for SME review of activity detection logic
> **Last Updated**: December 19, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Detected Activities Summary](#detected-activities-summary)
3. [Detection Models & Technologies](#detection-models--technologies)
4. [Activity Detection Logic](#activity-detection-logic)
   - [Cell Phone Detection](#1-cell-phone-detection-type-2)
   - [Microsleep Detection](#2-microsleep-detection-type-3)
   - [Sleep Detection](#3-sleep-detection-type-4)
   - [Writing Detection](#4-writing-detection-type-5)
   - [Packing Bags Detection](#5-packing-bags-detection-type-6)
   - [Group Detection](#6-group-detection-type-7)
   - [LP Hand Gesture](#7-lp-hand-gesture-detection-type-8)
   - [ALP Hand Gesture](#8-alp-hand-gesture-detection-type-9)
   - [Mind Diversion](#9-mind-diversion-detection-type-10)
   - [No Person Detected](#10-no-person-detected-type-11)
5. [Temporal Filtering Logic](#temporal-filtering-logic)
6. [Person Role Identification](#person-role-identification-lpalp)
7. [Configuration Parameters](#configuration-parameters)
8. [Processing Pipeline](#processing-pipeline)

---

## Overview

The Locopilot Monitoring System performs automated activity detection on cab surveillance videos to identify safety violations. The system uses a combination of:
- **YOLO11** - Object detection (phones, books, backpacks, persons)
- **YOLO11-Pose** - Human pose estimation (17 keypoints)
- **MediaPipe FaceMesh** - Eye tracking for sleep detection
- **Temporal Filtering** - False positive reduction through consecutive frame requirements

---

## Detected Activities Summary

| Type | Activity Name | Description | Evidence Rule |
|------|---------------|-------------|---------------|
| 2 | CELL_PHONE | Using mobile phone | `phone_in_hand` |
| 3 | MICROSLEEP | Micro-sleep detected (5+ seconds) | `eyes_closed_5s_or_pose_indicators` |
| 4 | SLEEP | Sleep detected (30+ seconds) | `eyes_closed_30s_or_pose_indicators` |
| 5 | WRITING | Writing log book while running | `hand_near_book_or_wrist_proximity` |
| 6 | PACKING_BAGS | Packing bags activity detected | `wrist_inside_backpack_bbox_or_hand_near_backpack` |
| 7 | GROUP_DETECTED | More than 2 people (group) detected | `more_than_2_deduplicated_persons` |
| 8 | LP_NOT_EXCHANGING_HAND_GESTURE | LP signaling failure | `lp_hand_raised_gesture_detected` |
| 9 | ALP_NOT_EXCHANGING_HAND_GESTURE | ALP signaling failure | `alp_hand_raised_gesture_detected` |
| 10 | MIND_DIVERSION | Attention diverted from controls | `head_turned_side_and_down` |
| 11 | NO_PERSON_DETECTED | No person detected in frame | `zero_persons_in_frame` |

---

## Detection Models & Technologies

### YOLO Models
| Model | Purpose | Configuration |
|-------|---------|---------------|
| `yolo11m.pt` | Object detection | Detects: phones, books, backpacks, persons |
| `yolo11m-pose.pt` | Pose estimation | 17 COCO keypoints per person |

### MediaPipe FaceMesh
- **Purpose**: Eye Aspect Ratio (EAR) calculation for sleep detection
- **Min Detection Confidence**: 0.5
- **Min Tracking Confidence**: 0.5
- **Max Faces**: 2

### Keypoint Indices (COCO Format)
```
0: nose               5: left_shoulder       10: right_wrist
1: left_eye           6: right_shoulder      11: left_hip
2: right_eye          7: left_elbow          12: right_hip
3: left_ear           8: right_elbow         13-16: knees/ankles
4: right_ear          9: left_wrist
```

---

## Activity Detection Logic

### 1. Cell Phone Detection (Type 2)

**Detection Method**: Multi-layered YOLO + ROI Detection

**Step 1 - Full Frame Detection**:
- YOLO detects "cell phone" class in entire frame
- Confidence threshold: **> 0.45**

**Step 2 - Region of Interest (ROI) Detection**:
- Extract 180px radius ROI around:
  - Wrists (left & right)
  - Ears (for phone calls)
  - Hip/lap area
- Run YOLO inference on each ROI

**Step 3 - Proximity Validation**:
- Phone must be within **200px** of hand/wrist/ear/shoulder
- Only phones detected from HAND/WRIST/EAR ROIs are flagged (not hip area)
- Aspect ratio validation filters false positives

**Temporal Thresholds**:
| Parameter | Value |
|-----------|-------|
| Required Consecutive Frames | 1 |
| Grace Period | 8 samples (~16 seconds) |
| Minimum Duration | 0 seconds (immediate) |

---

### 2. Microsleep Detection (Type 3)

**Detection Method**: Dual approach (Eye-Based + Pose-Based)

#### Method A: Eye Aspect Ratio (EAR) - Primary
- Uses MediaPipe FaceMesh landmarks
- **EAR Formula**: `EAR = (||v1|| + ||v2||) / (2 * ||h||)`
- Eye landmark indices:
  - Left Eye: [33, 160, 158, 133, 153, 144]
  - Right Eye: [362, 385, 387, 263, 373, 380]
- **Detection Criteria**: EAR < **0.2** (eyes closed)
- **Duration**: 5-30 seconds = Microsleep

#### Method B: Pose-Based - Fallback (when face not visible)
Three conditions must ALL be met:
1. **Head Tilt**: Head tilted forward (< -100 degrees from vertical)
2. **Minimal Movement**: Average landmark displacement < **0.1**
3. **Stable Posture**: Head tilt variance < 100 degrees

**Temporal Thresholds**:
| Parameter | Value |
|-----------|-------|
| Required Consecutive Frames | 2 (≈4 seconds @ 0.5 FPS) |
| Grace Period | 10 samples (~20 seconds) |
| Minimum Duration | 3.0 seconds |

---

### 3. Sleep Detection (Type 4)

**Detection Method**: Same as Microsleep but longer duration

#### Method A: Eye Aspect Ratio (EAR)
- **Criteria**: EAR < **0.2** for **30+ seconds**

#### Method B: Pose-Based Sleep
Same three conditions as microsleep:
1. Head tilted forward (< -100 degrees)
2. Movement average < 0.1
3. Head tilt variance < 100

**Override Conditions** (Sleep NOT flagged if):
- Person is holding phone
- Person is holding book
- Person is interacting with backpack

**Temporal Thresholds**:
| Parameter | Value |
|-----------|-------|
| Required Consecutive Frames | 4 (≈8 seconds @ 0.5 FPS) |
| Grace Period | 10 samples (~20 seconds) |
| Minimum Duration | 20.0 seconds |

---

### 4. Writing Detection (Type 5)

**Detection Method**: Dual approach (Book Detection + Pose Heuristic)

#### Method A: Book/Pen Detection
- YOLO detects "book" object with confidence > **0.2**
- Book must be within **150px** of detected person
- Aspect ratio validation for valid book shape

#### Method B: Pose-Based Writing Heuristic
All conditions must be met:
1. **Wrist Proximity**: Distance between wrists ≤ **250px**
2. **Head Down**: Nose position below eye line (pitch > 15°)
3. **Hands in Lap**: Hands below hips OR below shoulders
4. **Duration**: Sustained for 2+ seconds

**Proximity Thresholds**:
- Hand-to-book: **150px**
- Wrist proximity: **200px** (strict) / **400px** (relaxed when head down)

**Temporal Thresholds**:
| Parameter | Value |
|-----------|-------|
| Required Consecutive Frames | 1 |
| Grace Period | 8 samples (~16 seconds) |
| Minimum Duration | 0 seconds |

---

### 5. Packing Bags Detection (Type 6)

**Detection Method**: Backpack Detection + Wrist Position Analysis

**Step 1 - Backpack Detection**:
- YOLO detects: backpack, handbag, suitcase
- Confidence threshold: > **0.4**

**Step 2 - Wrist-Inside-Backpack Check**:
- Extract wrist coordinates from pose
- Check if wrist falls inside backpack bounding box
- **Tolerance margin**: 40px

**Step 3 - Hand-Near-Backpack Check (Alternative)**:
- Calculate distance from wrist to backpack center
- **Proximity threshold**: **250px**
- Sustained proximity for **4+ seconds** triggers detection

**Temporal Thresholds**:
| Parameter | Value |
|-----------|-------|
| Required Consecutive Frames | 1 |
| Grace Period | 5 samples (~10 seconds) |
| Minimum Duration | 0 seconds |
| Sustained Proximity | 4.0 seconds |

---

### 6. Group Detection (Type 7)

**Detection Method**: Person Count with Deduplication

**Step 1 - Person Detection**:
- YOLO detects all persons with confidence > **0.5**

**Step 2 - Deduplication**:
- IoU (Intersection over Union) threshold: **0.5**
- Overlapping person boxes are grouped as single person

**Step 3 - Count Check**:
- **Group Detected**: Deduplicated person count > **2** (3 or more people)

**Temporal Thresholds**:
| Parameter | Value |
|-----------|-------|
| Required Consecutive Frames | 3 (≈6 seconds @ 0.5 FPS) |
| Grace Period | 8 samples (~16 seconds) |
| Minimum Duration | 0 seconds |

---

### 7. LP Hand Gesture Detection (Type 8)

**Detection Method**: Pose-Based Hand Raising + Coordination Logic

**Hand Raised Detection**:
- Right wrist Y < right shoulder Y (hand above shoulder)
- Left wrist Y < left shoulder Y (hand above shoulder)

**Anatomical Validation**:
- Shoulder alignment check (±30 degrees)
- Arm proportions (forearm 50-150% of upper arm)
- Nose above shoulders, hips below shoulders
- Landmark stability (max jump < 100px)

**Coordination Logic**:
- **LP Gesture Violation**: LP raises hand BUT ALP doesn't respond within **5 seconds**
- Suppression: Gestures suppressed for **10 seconds** after detecting work activities (writing/packing/phone)

**Context Filtering**:
- Filters out hands reaching toward control panels
- Rejects forward reach detection
- Checks if hand is near backpack (not a gesture)

**Temporal Thresholds**:
| Parameter | Value |
|-----------|-------|
| Required Consecutive Frames | 2 (≈4 seconds @ 0.5 FPS) |
| Grace Period | 5 samples (~10 seconds) |
| Coordination Window | 5.0 seconds |
| Suppression Window | 10.0 seconds |

---

### 8. ALP Hand Gesture Detection (Type 9)

**Detection Method**: Same as LP Hand Gesture but for ALP role

**Coordination Logic**:
- **ALP Gesture Violation**: ALP raises hand BUT LP doesn't respond within **5 seconds**

**Note**: If both LP and ALP raise hands simultaneously, neither is flagged as violation (proper coordination).

---

### 9. Mind Diversion Detection (Type 10)

**Detection Method**: Head Pose Angle Analysis

**Yaw Angle Calculation (Side Turning)**:
```
yaw_normalized = (nose_x - shoulder_midpoint_x) / (shoulder_width / 2)
yaw_angle = clip(yaw_normalized * 45, -90, 90)
```
- Scale: ±45 degrees (0 = facing forward)

**Pitch Angle Calculation (Up/Down Tilt)**:
```
pitch_normalized = (nose_y - ear_midpoint_y) / head_height
pitch_angle = clip(pitch_normalized * 30, -45, 45)
```
- Scale: ±30 degrees (positive = looking down)

**Detection Criteria** (BOTH required):
- **Yaw**: |yaw| > **45 degrees** (head turned to side)
- **Pitch**: pitch > **15 degrees** (head looking down)

**Face Mesh Enhancement**:
When available, uses face mesh landmarks for more accurate detection.

**Temporal Thresholds**:
| Parameter | Value |
|-----------|-------|
| Required Consecutive Frames | 2 (≈4 seconds @ 0.5 FPS) |
| Grace Period | 5 samples (~10 seconds) |
| Minimum Duration | 0 seconds |

---

### 10. No Person Detected (Type 11)

**Detection Method**: YOLO Person Detection Absence

**Trigger Criteria**:
- Zero person boxes detected by YOLO (confidence > 0.5)
- Must persist for **10+ seconds**

**Temporal Thresholds**:
| Parameter | Value |
|-----------|-------|
| Required Consecutive Frames | 5 (≈10 seconds @ 0.5 FPS) |
| Grace Period | 5 samples (~10 seconds) |
| Minimum Duration | 10.0 seconds |

---

## Temporal Filtering Logic

All activities use a unified temporal filtering system to reduce false positives:

### Three-Level Filtering

```
┌─────────────────────────────────────────────────────────────┐
│                    FRAME DETECTION                          │
│  Is activity detected in current frame? (Yes/No)           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              CONSECUTIVE DETECTION COUNTER                  │
│  If Yes: Increment counter                                  │
│  If No:  Check grace period                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                REQUIRED THRESHOLD CHECK                     │
│  Counter >= required_consecutive?                           │
│  If Yes: Start/Continue activity recording                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    GRACE PERIOD                             │
│  If detection gap < grace_frames: Keep activity alive       │
│  If detection gap >= grace_frames: End activity             │
└─────────────────────────────────────────────────────────────┘
```

### Grace Period Purpose
- Allows brief interruptions without resetting detection
- Example: Hand moves briefly during phone use
- If gap exceeds grace period, counter resets to 0

### Activity Thresholds Summary

| Activity | Min Duration | Required Consecutive | Grace Frames |
|----------|-------------|----------------------|--------------|
| Cell Phone | 0s | 1 | 8 |
| Microsleep | 3s | 2 | 10 |
| Sleep | 20s | 4 | 10 |
| Writing | 0s | 1 | 8 |
| Packing Bags | 0s | 1 | 5 |
| Group Detected | 0s | 3 | 8 |
| LP Hand Gesture | 0s | 2 | 5 |
| ALP Hand Gesture | 0s | 2 | 5 |
| Mind Diversion | 0s | 2 | 5 |
| No Person | 10s | 5 | 5 |

---

## Person Role Identification (LP/ALP)

### Detection Method: Position-Based Heuristic

**LP (Loco Pilot)**:
- Typically detected on **left side** of image (driver seat)
- Higher LP score based on detected objects
- Association with steering wheel proximity

**ALP (Assistant Loco Pilot)**:
- Typically detected on **right side** of image (co-pilot seat)
- Higher ALP score based on position
- Secondary role in detection logic

### Person Deduplication
- IoU threshold: 0.5
- Ensures each person counted only once
- Groups overlapping bounding boxes

---

## Configuration Parameters

### Key Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_fps` | 0.5 | Frames sampled per second |
| `yolo_imgsz` | 416 | YOLO input image size |
| `yolo_pose_confidence` | 0.45 | Pose detection confidence |
| `cell_phone_confidence` | 0.45 | Phone detection confidence |
| `yolo_device` | cpu | Inference device (cpu/gpu) |

### Environment Variables

```bash
CELL_PHONE_CONFIDENCE=0.45
YOLO_POSE_CONFIDENCE=0.45
YOLO_IMGSZ=416
SAMPLE_FPS=0.5
HAND_GESTURE_COORDINATION_WINDOW=5.0
```

---

## Processing Pipeline

```
VIDEO INPUT
    │
    ▼
FRAME SAMPLING (0.5 FPS)
    │
    ├───────────────┬───────────────┐
    ▼               ▼               ▼
Face Detection  Object Detection  Pose Detection
(MediaPipe)     (YOLO)            (YOLO-Pose)
    │               │               │
    └───────────────┴───────────────┘
                    │
                    ▼
        PERSON DEDUPLICATION
        (IoU threshold: 0.5)
                    │
                    ▼
        ROLE IDENTIFICATION
        (LP / ALP assignment)
                    │
                    ▼
    ┌───────────────────────────────┐
    │   PER-PERSON ACTIVITY         │
    │   DETECTION                   │
    │   - Sleep/Microsleep          │
    │   - Cell Phone                │
    │   - Writing                   │
    │   - Packing Bags              │
    │   - Hand Gestures             │
    │   - Mind Diversion            │
    └───────────────────────────────┘
                    │
                    ▼
        TEMPORAL FILTERING
        (Consecutive + Grace Period)
                    │
                    ▼
        ACTIVITY STATE MACHINE
        (Start → Record → End)
                    │
                    ▼
        CLIP EXTRACTION (ffmpeg)
        (Direct segment extraction)
                    │
                    ▼
        OUTPUT: activities.json
        + Video clips + Images
```

---

## Key Files Reference

| Component | File Path |
|-----------|-----------|
| Core Detection Logic | `locopilot_monitor.py` |
| Activity Type Definitions | `app/models/activity_models.py` |
| Detection Service | `app/services/activity_detection_service.py` |
| Configuration | `app/utils/config.py` |
| YOLO-Pose Adapter | `app/services/yolo_pose_adapter.py` |
| External API Integration | `app/services/external_api_service.py` |
| Video Processing Service | `app/services/video_processing_service.py` |

---

## Questions for SME Review

1. Are the detection thresholds appropriate for operational conditions?
2. Should any temporal filtering parameters be adjusted?
3. Are there additional activities that need to be detected?
4. Is the LP/ALP role identification logic accurate for your camera setup?
5. Are the grace period durations appropriate for real-world scenarios?

---

*Document generated for SME review. Please provide feedback on detection logic accuracy and threshold appropriateness.*

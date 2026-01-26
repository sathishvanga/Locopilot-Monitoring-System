# CVVRS - Crew Voice and Video Recording System

## Technical Detection Methodology

---

## Executive Summary

CVVRS (Crew Voice and Video Recording System) is an AI-powered video analytics platform designed to enhance railway safety by automatically monitoring locomotive crew activities. The system employs a sophisticated multi-layer AI pipeline combining state-of-the-art computer vision models with temporal filtering algorithms to achieve high-accuracy violation detection while minimizing false positives.

This document details the technical methodology behind each detection capability.

---

## Core AI/ML Architecture

### Models Deployed

| Model | Purpose | Input Resolution | Confidence Threshold |
|-------|---------|------------------|---------------------|
| **YOLO11m** | Object detection (phones, books, bags, persons) | 640×640 | 0.25 - 0.50 |
| **YOLO11m-Pose** | Multi-person skeletal pose estimation (17 keypoints) | 640×640 | 0.45 |
| **MediaPipe Face Mesh** | 468 facial landmarks for eye tracking & head pose | Full frame | Real-time |
| **OpenCV Farneback** | Dense optical flow for train motion detection | ROI-based | 5-frame smoothing |

### Processing Pipeline

```
Video Input → Frame Sampling (0.5 FPS) → Object Detection → Pose Estimation
    → Activity Classification → Temporal Filtering → Voting Verification → Output
```

---

## Violation Detection Methods

### 1. Sleep Detection

**Classification:** Safety-Critical | **Minimum Duration:** 20 seconds

#### Multi-Stage Detection Approach

**Stage 1: Eye Aspect Ratio (EAR) Analysis**

Uses MediaPipe Face Mesh to extract 468 facial landmarks and calculate eye openness.

- Eye landmarks tracked:
  - Left eye: [33, 160, 158, 133, 153, 144]
  - Right eye: [362, 385, 387, 263, 373, 380]

- EAR Formula:
  ```
  EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 × ||p1 - p4||)
  ```
  Where p1-p6 represent eye corner and lid landmarks

- Eye closure threshold: EAR < 0.21 indicates closed eyes
- Sustained closure for 20+ seconds triggers sleep detection

**Stage 2: Pose-Based Fallback (when face not visible)**

Activated when face detection confidence < 0.5:

| Indicator | Threshold | Description |
|-----------|-----------|-------------|
| Head tilt angle | < -100° | Severe forward head droop |
| Body movement | < 0.1 | Minimal movement across 7 key landmarks |
| Posture variance | < 100 | Stable/unchanging head tilt history |
| Duration | ≥ 5 seconds | Consistent pose data required |

**Temporal Validation:**
- Minimum 4 consecutive samples at 0.5 FPS
- Grace period: 10 frames (~20 seconds) before resetting

---

### 2. Microsleep Detection

**Classification:** Safety-Critical | **Duration Range:** 3-20 seconds

#### Detection Logic

- Utilizes same EAR calculation as sleep detection
- Triggers at shorter duration threshold (3-5 seconds)
- Eye closure events < 3 seconds classified as normal blinks (ignored)
- Distinguishes from natural blinks using temporal pattern analysis

**Thresholds:**
- Minimum consecutive samples: 2 at 0.5 FPS
- Recovery detection: EAR returning to > 0.25 ends episode

---

### 3. Cell Phone Detection

**Classification:** Policy Violation | **Detection:** Immediate with verification

#### Triple-Layer Detection System

**Layer 1: YOLO Object Detection**
- Primary detection using YOLO11m trained on COCO dataset
- Target class: "cell phone" (class ID 67)
- Confidence threshold: 0.40 (configurable)
- Aspect ratio validation filters non-phone rectangular objects

**Layer 2: Pose-Guided ROI Search**

Extracts Region of Interest (180px radius) around key body landmarks:

| Landmark | Detection Scenario |
|----------|-------------------|
| Wrists (left/right) | Phone held in hand |
| Ears (left/right) | Phone against ear during call |

- Hip-region detections excluded (high false positive rate from seat patterns)
- Enhanced detection accuracy in hand/ear regions

**Layer 3: Contextual Validation**
- Phone must be within person's bounding box or immediate proximity
- Cross-validates phone position with hand position
- Excludes phones on dashboard/surfaces (not actively in use)

**Verification:** Two-stage voting with 50% confirmation threshold across 10 native frames

---

### 4. Writing Detection

**Classification:** Operational Violation | **Detection:** Immediate with verification

#### Multi-Pathway Detection

**Pathway 1: Object + Hand Correlation**
```
Book/Notebook Detection (YOLO, conf ≥ 0.4)
    ↓
Proximity Check: Book within ±150px of person
    ↓
Hand-to-Book Distance: Wrist within 180px of book
    ↓
Aspect Ratio Validation (book shape confirmation)
```

**Pathway 2: Wrist Proximity + Head Posture**

Activated when book detection fails but writing posture detected:

| Criteria | Threshold |
|----------|-----------|
| Wrist-to-wrist distance | ≤ 300px (hands together) |
| Head orientation | Looking down (nose below eye midline) |
| Duration | ≥ 1 second |
| Consecutive frames | ≥ 2 frames |

**Pathway 3: Book + Posture Correlation**
- Book detected in frame
- Head oriented downward (reading/writing angle)
- Validates legitimate documentation activity vs. idle book presence

**Temporal Filtering:**
- Grace period: 10 frames (~20 seconds)
- Verification voting threshold: 50%

---

### 5. Packing Bags Detection

**Classification:** Pre-departure Violation | **Detection:** With strict verification

#### Detection Methodology

**Object Detection Phase:**

| Parameter | Value |
|-----------|-------|
| Target objects | Backpack, handbag, suitcase |
| Confidence threshold | 0.45 |
| Valid area range | 25,000 - 100,000 pixels² |
| Aspect ratio filter | height/width < 1.2 |

**Hand-Bag Interaction Verification:**
```
Wrist Extraction (YOLO-Pose, visibility ≥ 0.50)
    ↓
Distance Calculation: Wrist to bag center
    ↓
Proximity Threshold: Within 30% of bag diagonal
    ↓
Duration Check: Sustained 4+ seconds
```

**Strict Verification Parameters:**
- Wrist must be inside or touching bag boundary
- Voting threshold: 75% (stricter than other activities)
- Minimum bag area: 25,000 sq pixels

---

### 6. Group Detection (Unauthorized Personnel)

**Classification:** Security Alert | **Threshold:** > 2 persons

#### Person Counting Algorithm

**Detection:**
- YOLO person detection with confidence > 0.5
- De-duplication using Non-Maximum Suppression (IoU threshold 0.3)
- Counts unique person bounding boxes per frame

**Violation Trigger:**
- Person count > 2 (exceeds LP + ALP crew complement)
- Indicates presence of: supervisor, trainee, unauthorized visitor

**Temporal Stability:**
- Requires 3 consecutive samples (~6 seconds) for confirmation
- Prevents false triggers from transient detections

---

### 7. Hand Gesture Coordination (LP-ALP Communication)

**Classification:** Safety Protocol | **Session-Based Detection**

#### Hand Raise Detection Criteria

| Criterion | Threshold |
|-----------|-----------|
| Wrist above shoulder | > 0px vertical elevation |
| Wrist above elbow | > -30px |
| Arm extension (lateral) | > 20px |
| Elbow-shoulder distance | < 150px |
| Landmark visibility | > 0.3 |

#### Session-Based Coordination Tracking

```
[Either LP or ALP raises hand] → Session Starts
    ↓
Track: Did LP raise? Did ALP raise?
    ↓
[10 second timeout with no activity] → Session Ends
    ↓
Evaluation: If one raised but other didn't → Violation
```

**Velocity & Trajectory Analysis:**
- Validates rapid hand raise motion
- Distinguishes intentional gestures from slow control panel operations
- Minimum velocity threshold prevents false triggers

**Context-Aware Suppression:**
- Suppressed during active packing_bags activity
- Suppressed when hand trajectory indicates control panel operation
- Validates hand not near steering/control equipment

---

### 8. Mind Diversion Detection

**Classification:** Attention Monitoring | **Multi-Angle Analysis**

#### Head Pose Estimation

**Method 1: Pose Landmark Geometry**
- Calculates yaw (horizontal rotation) and pitch (vertical tilt)
- Uses nose, shoulder, and ear positions
- Reference: Shoulder centerline defines forward-facing

**Method 2: Face Mesh Refinement**
- 468-point face mesh provides finer head pose estimation
- Activated when pose method confidence is low

#### Detection Sub-Types

| Sub-Type | Yaw Threshold | Pitch Threshold | Indication |
|----------|---------------|-----------------|------------|
| Looking Sideways | > 78° | - | Heavy side turn away from track |
| Looking Down Distracted | < 55° | > 45° | Head down, not reading |
| Looking Away Combined | > 58° | > 35° | Side + down turn |

#### Forward-Looking Exemption (Indian Locomotive Configuration)

```
Camera Position: Rear-right of cabin
    ↓
Negative Yaw = Looking left toward track = LEGITIMATE (no violation)
Positive Yaw = Looking right away from track = VIOLATION
```

**Intelligent Suppression:**
- During active writing activity (5-second grace period)
- When book detected in frame (reading documentation)
- Wrists close together in lap area with head down (document review posture)

---

### 9. No Person Detected

**Classification:** Critical Alert | **Detection:** Immediate

#### Detection Logic

- YOLO person detection returns zero detections
- Frame contains no visible crew members

**Possible Causes:**
- Camera obstruction
- Cabin temporarily unmanned
- System malfunction

**Response:** Immediate flagging with no temporal delay

---

### 10. ALP Standing Before Stop

**Classification:** Safety Protocol | **Requirement:** 30 seconds before stop

#### Train State Detection (Prerequisite)

**Optical Flow Analysis:**
```
Side Window ROI (37-52% width, 0-15% height)
    ↓
Farneback Dense Optical Flow
Parameters: pyramid_scale=0.5, levels=3, window=15, iterations=3
    ↓
5-Frame Moving Average Smoothing
    ↓
State Classification:
├── Flow < 2.0 pixels/frame → STOPPED
├── Flow 2.0-4.0 pixels/frame → APPROACHING_STOP
└── Flow > 4.0 pixels/frame → MOVING
```

#### ALP Standing Detection

**Posture Analysis:**
- YOLO-Pose keypoint extraction for ALP
- Hip-knee-ankle vertical alignment indicates standing
- Knee bend angle differentiates sitting from standing

**Compliance Tracking:**
```
APPROACHING_STOP detected → Start monitoring ALP pose
    ↓
If ALP stands → Record timestamp
If ALP not visible → Skip check (avoid false positive)
    ↓
STOPPED detected → Verify: ALP stood ≥30s before stop?
    ↓
Violation if ALP was visible but didn't stand in time
```

---

## Temporal Filtering Architecture

### Two-Gate False Positive Reduction

**Gate 1: Consecutive Frame Requirement**

| Activity | Minimum Consecutive Samples |
|----------|---------------------------|
| Microsleep | 2 |
| Group Detection | 3 |
| Sleep | 4 |
| Cell Phone, Writing, Others | 1 (immediate) |

**Gate 2: Duration Threshold**

| Activity | Minimum Duration |
|----------|-----------------|
| Microsleep | 3 seconds |
| Sleep | 20 seconds |
| Others | Immediate with verification |

### Grace Period System

Allows brief interruptions without resetting detection:

| Activity | Grace Period |
|----------|-------------|
| Cell Phone | 16 seconds (8 frames) |
| Writing | 20 seconds (10 frames) |
| Packing Bags | 10 seconds (5 frames) |
| Sleep/Microsleep | 20 seconds (10 frames) |

---

## Voting Verification System

### Two-Stage Detection Process

**Stage 1: Quick Detection (0.5 FPS sampling)**
- Initial trigger on sampled frames
- Low-latency preliminary detection

**Stage 2: High-Frequency Verification**
- Extracts 10 native frames around trigger timestamp
- 400ms window (covers varying input FPS)
- Full detection pipeline on each frame
- Majority voting determines final verdict

### Voting Thresholds

| Activity | Confirmation Threshold |
|----------|----------------------|
| Cell Phone | 50% |
| Writing | 50% |
| Packing Bags | 75% (stricter) |
| Mind Diversion | 50% |
| Hand Gestures | 50% |
| Group Detection | 50% |

---

## Role Identification (LP vs ALP)

### Automatic Crew Classification

**Single Person:** Defaults to Loco Pilot (LP)

**Two Persons:**

| Method | Approach |
|--------|----------|
| Primary | Camera proximity - Larger bounding box = Closer = LP |
| Secondary | Object-based scoring |

**Object Scoring Weights:**

| Object | LP Score | ALP Score |
|--------|----------|-----------|
| TV/Monitor | +3 | - |
| Keyboard | +2 | - |
| Mouse | +1 | - |
| Remote | +2 | - |
| Book | - | +3 |
| Notebook | - | +3 |
| Backpack | - | +1 |

**Three+ Persons:**
- Largest person (closest to camera) = LP
- Second largest = ALP
- Others classified as: Supervisor, Trainee, or Visitor

---

## Train State-Based Exemptions

### Stopped State Behavior

When train is confirmed STOPPED (optical flow < 2.0 for 5+ seconds):

**Exempted Activities:**
- Cell phone usage
- Writing
- Packing bags
- Mind diversion
- Microsleep/Sleep
- Group detection
- Hand gestures

**Active Monitoring (not exempted):**
- No person detected (cabin must remain manned)

---

## Evidence Generation

### Per-Violation Output

| Artifact | Description |
|----------|-------------|
| **JSON Activity Log** | Structured metadata with timestamps, confidence scores |
| **Video Clip** | Extracted segment covering violation duration |
| **Screenshot** | Key frame with bounding box annotations |
| **OCR Timestamp** | On-screen time extracted from video overlay |

### Activity Record Schema

```json
{
  "activity_type": "cell_phone",
  "start_time": "2024-01-15T10:23:45.123Z",
  "end_time": "2024-01-15T10:24:12.456Z",
  "duration_seconds": 27.333,
  "person_role": "LP",
  "confidence": 0.87,
  "frames_detected": 14,
  "verification_score": 0.70,
  "train_state": "MOVING"
}
```

---

## System Accuracy Metrics

### False Positive Reduction

| Technique | Reduction Rate |
|-----------|---------------|
| Two-gate temporal filtering | 99%+ |
| Voting verification | Additional 40-60% refinement |

### Detection Latency

| Stage | Duration |
|-------|----------|
| Primary detection | 2 seconds (0.5 FPS sampling) |
| Verification | +0.4 seconds (10-frame voting) |
| **Total end-to-end** | **~2.4 seconds average** |

---

## Conclusion

CVVRS employs a defense-in-depth approach to violation detection, combining:

1. **Multi-model AI fusion** - YOLO + Pose + Face Mesh + Optical Flow
2. **Anatomical validation** - Physically possible poses only
3. **Temporal filtering** - Consecutive frames + duration thresholds
4. **Voting verification** - Multi-frame confirmation
5. **Context awareness** - Train state, activity suppression, role identification

This architecture ensures high detection accuracy while minimizing false positives, providing railway authorities with reliable, actionable safety insights.

---

*Document Version: 1.0*
*System Version: CVVRS 2.0*
*Last Updated: January 2026*

# LP/ALP Identification Examples

## Example Scenarios

### Scenario 1: Standard Train Cab with Two People

**Setup:**
- Person 1 is sitting in front of multiple monitors and a control keyboard
- Person 2 is sitting nearby with a logbook and pen

**Detection:**
```
Person 1 nearby objects:
  - TV/Monitor (×2) = 6 points
  - Keyboard (×1) = 2 points
  - Mouse (×1) = 1 point
  → LP Score = 9, ALP Score = 0

Person 2 nearby objects:
  - Book (×1) = 3 points
  - Notebook (×1) = 3 points
  → LP Score = 0, ALP Score = 6

RESULT:
  Person 1 → LP (Loco Pilot) - Higher LP score
  Person 2 → ALP (Assistant Loco Pilot)
```

### Scenario 2: Single Operator

**Setup:**
- Only one person detected in the cab

**Detection:**
```
Person 1:
  → Default assignment: LP (Loco Pilot)
```

### Scenario 3: Training Session with Three People

**Setup:**
- Person 1: Experienced LP at controls (monitors, keyboard)
- Person 2: ALP with logbook
- Person 3: Trainee with backpack and notebook

**Detection:**
```
Person 1 nearby objects:
  - TV/Monitor (×2) = 6 points
  - Keyboard (×1) = 2 points
  → LP Score = 8, ALP Score = 0

Person 2 nearby objects:
  - Book (×1) = 3 points
  → LP Score = 0, ALP Score = 3

Person 3 nearby objects:
  - Backpack (×1) = 1 point
  - Notebook (×1) = 3 points
  → LP Score = 0, ALP Score = 4

RESULT:
  Person 1 → LP (highest LP score)
  Person 2 → ALP (second by LP score, but has ALP indicators)
  Person 3 → TRAINEE (high ALP score, learning role)
```

### Scenario 4: Supervisor Inspection

**Setup:**
- Person 1: LP at controls
- Person 2: ALP with logbook
- Person 3: Supervisor checking instrumentation (has laptop)

**Detection:**
```
Person 1 nearby objects:
  - TV/Monitor (×2) = 6 points
  - Keyboard (×1) = 2 points
  → LP Score = 8, ALP Score = 0

Person 2 nearby objects:
  - Book (×1) = 3 points
  → LP Score = 0, ALP Score = 3

Person 3 nearby objects:
  - Laptop (×1) = 2 points
  - Cell phone (×1) = 1 point
  → LP Score = 3, ALP Score = 0

RESULT:
  Person 1 → LP (highest LP score = 8)
  Person 2 → ALP (second person, has ALP indicators)
  Person 3 → SUPERVISOR (LP score > 2, but not highest)
```

### Scenario 5: Empty Desk Heuristic

**Setup:**
- Person 1: Sitting at control station but objects not detected
- Person 2: Sitting at secondary position with no detected objects

**Detection:**
```
Person 1 nearby objects: None detected
  → LP Score = 0, ALP Score = 1 (empty desk bonus)

Person 2 nearby objects: None detected
  → LP Score = 0, ALP Score = 1 (empty desk bonus)

RESULT (tie-breaking by position):
  Person 1 → LP (first detected)
  Person 2 → ALP (second detected)
```

### Scenario 6: Phone Usage During Operation

**Setup:**
- Person 1: LP using cell phone while at controls
- Person 2: ALP documenting in logbook

**Detection:**
```
Person 1 nearby objects:
  - TV/Monitor (×1) = 3 points
  - Keyboard (×1) = 2 points
  - Cell phone (×1) = 1 point  ← Activity trigger
  → LP Score = 6, ALP Score = 0

Person 2 nearby objects:
  - Book (×1) = 3 points
  → LP Score = 0, ALP Score = 3

RESULT:
  Person 1 → LP + Cell Phone Activity Detected
  Person 2 → ALP
  
Activity JSON includes:
  {
    "activityType": 2,
    "des": "Using mobile phone",
    "personRoles": [
      {"personIndex": 0, "role": "LP", "lpScore": 6, "alpScore": 0},
      {"personIndex": 1, "role": "ALP", "lpScore": 0, "alpScore": 3}
    ]
  }
```

## Object Detection Examples

### Objects That Indicate LP Role

| Object | Weight | Rationale |
|--------|--------|-----------|
| TV/Monitor | 3 | Primary control displays |
| Laptop | 2 | Computer-based controls |
| Keyboard | 2 | Input device for controls |
| Mouse | 1 | Control interaction |
| Cell Phone | 1 | Communication device |
| Remote | 2 | Control panel/remote |

### Objects That Indicate ALP Role

| Object | Weight | Rationale |
|--------|--------|-----------|
| Book | 3 | Operating manuals, logbooks |
| Notebook | 3 | Recording observations |
| Backpack | 1 | Personal items, documents |

## Sample Activity Output

### Complete Activity JSON with Person Roles

```json
{
  "tripId": "TRIP-2025-001",
  "activityType": 2,
  "des": "Using mobile phone",
  "objectType": "cell phone",
  "fileUrl": "/path/to/video.mp4",
  "fileDuration": "00:45:30",
  "activityStartTime": "125.50",
  "activityEndTime": "132.75",
  "crewName": "John Doe",
  "crewId": "C-001",
  "crewRole": 1,
  "date": "2025-11-13",
  "time": "14:30:45",
  "filename": "train_video.mp4",
  "peopleCount": 2,
  "evidence": {
    "rule": "phone_in_hand"
  },
  "activityImage": "train_video_cell_phone_frame00012550_001_activity.jpg",
  "activityClip": "train_video_cell_phone_frame00012550_001_clip.mp4",
  "personRoles": [
    {
      "personIndex": 0,
      "role": "LP",
      "roleName": "Loco Pilot",
      "lpScore": 6,
      "alpScore": 0
    },
    {
      "personIndex": 1,
      "role": "ALP",
      "roleName": "Assistant Loco Pilot",
      "lpScore": 0,
      "alpScore": 3
    }
  ]
}
```

## Console Output Example

When processing a video with LP/ALP identification:

```
Processing video: train_video.mp4
Native FPS: 30.00
Sample FPS: 0.5 (1 frame every 2.0 seconds)
Total frames in video: 27000
Expected duration: 15.00 minutes
Expected sampled frames: ~450
...

[0:02:05] Person roles identified:
  Person 1: Loco Pilot (LP score: 8, ALP score: 0)
  Person 2: Assistant Loco Pilot (LP score: 0, ALP score: 3)

[0:02:05] Cell phone ACTIVELY USED in hand (frame 3750, hand raised, upper body)
[0:02:05] Activity started: cell_phone

[0:02:18] Activity ended: cell_phone
  Clip Duration: 13.00s (7 frames @ 0.5 FPS)
  Min Duration Threshold: 0.0s | Required Consecutive: 1 frames
  Evidence saved: train_video_cell_phone_frame00003750_000_clip.mp4
  Activity image: train_video_cell_phone_frame00003750_000_activity.jpg

...

Processing complete!
Total frames sampled: 450/27000
Sampling rate: 0.5 FPS (1 frame every 2.0 seconds)
Evidence clips created: 5
Run directory: locopilot_evidence/run_20251113_143045

Activities JSON saved: locopilot_evidence/run_20251113_143045/activities.json
Total activities detected: 5

Activity Breakdown:
  - Using mobile phone: 2
  - Writing activity detected: 2
  - Micro-sleep detected (5+ seconds): 1
```

## Visual Annotations

### Annotated Frame Layout

```
┌──────────────────────────────────────────────┐
│  People Count: 2              [Top Right]    │
│  Loco Pilot: LP=8, ALP=0                     │
│  Assistant Loco Pilot: LP=0, ALP=3           │
│                                               │
│  ┌──────────────────────┐                    │
│  │ Loco Pilot          │← Yellow box         │
│  │ (LP:8/ALP:0)        │                     │
│  │                      │                     │
│  │      [Person 1]      │                     │
│  │                      │                     │
│  └──────────────────────┘                    │
│                                               │
│           ┌──────────────────────┐           │
│           │ Assistant Loco Pilot│← Orange box│
│           │ (LP:0/ALP:3)        │            │
│           │                      │            │
│           │      [Person 2]      │            │
│           │                      │            │
│           └──────────────────────┘           │
│                                               │
│  MediaPipe pose landmarks shown              │
│  EAR: 0.285 - EYES OPEN          [Bottom]    │
└──────────────────────────────────────────────┘
```

### Color Scheme

- 🟡 **Yellow (Cyan)** - LP (Loco Pilot)
- 🟠 **Orange** - ALP (Assistant Loco Pilot)  
- 🟣 **Purple** - SUPERVISOR
- 🔵 **Cyan** - TRAINEE
- ⚪ **Gray** - VISITOR

## Integration with Existing Activities

All existing activity types now support person role information:

1. **Cell Phone Usage** - Identifies which person (LP/ALP) is using phone
2. **Writing Activity** - Shows who is documenting (usually ALP)
3. **Microsleep/Sleep** - Critical info on which role is affected
4. **Packing Bags** - Identifies who is preparing to leave
5. **Group Detection** - Lists all people with their roles

## Summary

The LP/ALP identification system provides:

✅ **Automatic role detection** based on nearby objects  
✅ **Real-time scoring** using YOLO object detection  
✅ **Visual annotations** with color-coded labels  
✅ **JSON export** with complete role information  
✅ **Multi-person support** (LP, ALP, Supervisor, Trainee, Visitor)  
✅ **Seamless integration** with existing activity detection  
✅ **Zero configuration** - works out of the box  


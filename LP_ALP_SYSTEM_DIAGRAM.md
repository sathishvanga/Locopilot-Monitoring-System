# LP/ALP Identification System - Visual Diagram

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     VIDEO INPUT (Train Cab Footage)                  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    YOLO OBJECT DETECTION                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Detect:                                                     │   │
│  │  • Persons (person boxes)                                    │   │
│  │  • Control objects (tv, laptop, keyboard, mouse, etc.)      │   │
│  │  • Documentation objects (book, notebook, backpack)         │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              PERSON DEDUPLICATION (NMS Algorithm)                    │
│  Remove overlapping person boxes → Get unique persons               │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              FOR EACH PERSON: IDENTIFY NEARBY OBJECTS                │
│                                                                       │
│  Search Region:                                                      │
│  ┌────────────────────────────────────────────────┐                │
│  │  Horizontal: ±1.5× person width                │                │
│  │  Vertical: Chest to desk area                  │                │
│  │  (person_y + 0.3×height to person_y + 1.5×height)              │
│  └────────────────────────────────────────────────┘                │
│                                                                       │
│  Count objects by category:                                          │
│  • LP objects: monitors, keyboards, laptops, etc.                   │
│  • ALP objects: books, notebooks, backpacks                         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CALCULATE SCORES                                  │
│                                                                       │
│  LP Score = monitors×3 + laptops×2 + keyboards×2 + mouse×1 +       │
│             cell_phone×1 + remotes×2                                │
│                                                                       │
│  ALP Score = books×3 + notebooks×3 + backpacks×1                   │
│                                                                       │
│  Empty Desk Bonus: If no objects detected, ALP score += 1          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ASSIGN ROLES                                      │
│                                                                       │
│  IF 1 person:                                                        │
│    → Person 1 = LP                                                   │
│                                                                       │
│  IF 2 people:                                                        │
│    → Higher LP score = LP                                            │
│    → Other person = ALP                                              │
│                                                                       │
│  IF 3+ people:                                                       │
│    → Highest LP score = LP                                           │
│    → 2nd highest LP score = ALP                                      │
│    → Others:                                                         │
│        • ALP score > 0 → TRAINEE                                     │
│        • LP score > 2 → SUPERVISOR                                   │
│        • Otherwise → VISITOR                                         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  STORE ROLE INFORMATION                              │
│                                                                       │
│  {                                                                    │
│    0: {role: 'LP', role_name: 'Loco Pilot',                         │
│        lp_score: 8, alp_score: 0},                                  │
│    1: {role: 'ALP', role_name: 'Assistant Loco Pilot',              │
│        lp_score: 0, alp_score: 3}                                   │
│  }                                                                    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│   VISUAL ANNOTATIONS         │  │   ACTIVITY JSON OUTPUT       │
│                              │  │                              │
│  • Color-coded boxes         │  │  {                           │
│    🟡 Yellow = LP            │  │    "personRoles": [          │
│    🟠 Orange = ALP           │  │      {                       │
│    🟣 Purple = SUPERVISOR    │  │        "personIndex": 0,     │
│    🔵 Cyan = TRAINEE         │  │        "role": "LP",         │
│    ⚪ Gray = VISITOR         │  │        "roleName": "Loco Pilot",│
│                              │  │        "lpScore": 8,         │
│  • Labels: "Role (LP:X/ALP:Y)"│ │        "alpScore": 0         │
│                              │  │      }                       │
│  • Role summary overlay      │  │    ]                         │
│                              │  │  }                           │
└──────────────────────────────┘  └──────────────────────────────┘
```

## Object Detection Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRAME ANALYSIS                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │  YOLO Detection Results  │
              └─────────┬────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   PERSONS    │ │ LP OBJECTS   │ │ ALP OBJECTS  │
│              │ │              │ │              │
│ • Person 1   │ │ • TV: 2      │ │ • Book: 1    │
│ • Person 2   │ │ • Keyboard: 1│ │ • Notebook: 1│
│              │ │ • Mouse: 1   │ │ • Backpack: 1│
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       │     ┌──────────┴────────┐       │
       │     │ Spatial Analysis  │       │
       └────▶│ (Near Person 1?)  │◀──────┘
             └──────────┬────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │  Person 1:           │
             │  • TV (×2) = 6 pts   │
             │  • Keyboard = 2 pts  │
             │  • Mouse = 1 pt      │
             │  ─────────────────── │
             │  LP Score: 9         │
             │  ALP Score: 0        │
             │  → ROLE: LP          │
             └──────────────────────┘
```

## Role Assignment Decision Tree

```
                    START
                      │
                      ▼
              ┌───────────────┐
              │ How many      │
              │ people?       │
              └───┬───────────┘
                  │
        ┌─────────┼─────────┐
        │         │         │
        ▼         ▼         ▼
    ┌──────┐  ┌──────┐  ┌──────┐
    │  1   │  │  2   │  │  3+  │
    └──┬───┘  └──┬───┘  └──┬───┘
       │         │         │
       ▼         ▼         │
   ┌─────────┐ ┌─────────┐│
   │Person 1 │ │ Compare ││
   │  → LP   │ │LP scores││
   └─────────┘ └────┬────┘│
                    │     │
                    ▼     │
          ┌──────────────┐│
          │Higher = LP   ││
          │Other = ALP   ││
          └──────────────┘│
                          ▼
              ┌────────────────────┐
              │ Sort by LP score   │
              │ 1st → LP           │
              │ 2nd → ALP          │
              │ 3rd+ → ?           │
              └──────┬─────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
   ┌─────────┐  ┌──────────┐ ┌─────────┐
   │ALP      │  │LP score  │ │No clear │
   │score>0? │  │ > 2?     │ │indicator│
   └────┬────┘  └────┬─────┘ └────┬────┘
        │YES         │YES         │
        ▼            ▼            ▼
   ┌─────────┐  ┌──────────┐ ┌─────────┐
   │TRAINEE  │  │SUPERVISOR│ │VISITOR  │
   └─────────┘  └──────────┘ └─────────┘
```

## Scoring Weights Visualization

```
LP OBJECTS (Control-Oriented)
═════════════════════════════════════════

  TV/Monitor     ███████████████████████████ (3 points) ← Strongest indicator
  Laptop         ██████████████████ (2 points)
  Keyboard       ██████████████████ (2 points)
  Remote         ██████████████████ (2 points)
  Mouse          ██████████ (1 point)
  Cell Phone     ██████████ (1 point)


ALP OBJECTS (Documentation-Oriented)
═════════════════════════════════════════

  Book           ███████████████████████████ (3 points) ← Strongest indicator
  Notebook       ███████████████████████████ (3 points)
  Backpack       ██████████ (1 point)
```

## Example Scenario Visualization

```
╔════════════════════════════════════════════════════════════════╗
║                    TRAIN CAB - TOP VIEW                        ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║   [MONITORS]  [MONITORS]    [DESK/CONSOLE]                    ║
║      ▓▓▓▓       ▓▓▓▓                                          ║
║                                                                ║
║   ┌──────────┐                    ┌──────────┐               ║
║   │          │                    │          │               ║
║   │ PERSON 1 │                    │ PERSON 2 │               ║
║   │   (LP)   │                    │  (ALP)   │               ║
║   │          │                    │          │               ║
║   └────┬─────┘                    └────┬─────┘               ║
║        │                               │                      ║
║   [KEYBOARD]                      [LOGBOOK]                  ║
║     ▒▒▒▒                           ░░░░░░                    ║
║   [MOUSE]                         [PEN]                      ║
║     ▒▒                             ░░                        ║
║                                                                ║
║   Detection:                     Detection:                   ║
║   • Monitors: 2 (×3 = 6)         • Book: 1 (×3 = 3)          ║
║   • Keyboard: 1 (×2 = 2)         • Pen: 0                    ║
║   • Mouse: 1 (×1 = 1)                                        ║
║                                                                ║
║   LP Score: 9                    LP Score: 0                 ║
║   ALP Score: 0                   ALP Score: 3                ║
║   → ROLE: LP ✓                   → ROLE: ALP ✓               ║
╚════════════════════════════════════════════════════════════════╝
```

## Integration with Activity Detection

```
┌─────────────────────────────────────────────────────────────┐
│               ACTIVITY DETECTION PIPELINE                    │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Detect Persons (YOLO)                                   │
│  2. Identify Person Roles (LP/ALP) ← NEW                    │
│  3. Detect Activities (phone, writing, sleep)               │
│  4. Associate Activity with Role ← ENHANCED                 │
│  5. Generate Evidence (clips, images, JSON)                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   ACTIVITY OUTPUT                            │
│                                                              │
│  Activity: "Using mobile phone"                             │
│  Who: Person 0 (Loco Pilot) ← Role information             │
│  When: 125.50s - 132.75s                                    │
│  Evidence: phone_in_hand                                     │
│  Scores: LP=8, ALP=0                                        │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

```
VIDEO FRAME
    │
    ├─→ YOLO Detection
    │      │
    │      ├─→ Person Boxes
    │      ├─→ Object Detections
    │      └─→ Confidence Scores
    │
    ├─→ Person Deduplication (NMS)
    │      │
    │      └─→ Unique Person List
    │
    ├─→ Role Identification
    │      │
    │      ├─→ Calculate LP Score
    │      ├─→ Calculate ALP Score
    │      └─→ Assign Roles
    │
    ├─→ Activity Detection
    │      │
    │      ├─→ Check Activity Rules
    │      └─→ Associate with Roles
    │
    └─→ Output Generation
           │
           ├─→ Annotated Frames (with role labels)
           ├─→ Video Clips (with role info)
           └─→ JSON (with personRoles field)
```

## Summary

The LP/ALP identification system seamlessly integrates into the existing activity detection pipeline, adding intelligent role recognition based on detected objects near each person. The system is fully automatic, requiring no configuration, and provides rich contextual information for every detected activity.


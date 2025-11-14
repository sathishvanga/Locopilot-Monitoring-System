# LP/ALP Identification System - Implementation Guide

## Overview

The Locopilot Monitoring System now includes automatic identification of **LP (Loco Pilot)** and **ALP (Assistant Loco Pilot)** based on objects detected near each person using YOLO object detection.

## How It Works

### Detection Logic

For each person detected in the video frame:

1. **Identify the person's region** using YOLO person detection
2. **Search for objects near the person** (within 1.5x person width horizontally, and chest to desk area vertically)
3. **Calculate LP Score** based on control-oriented objects
4. **Calculate ALP Score** based on documentation-oriented objects
5. **Assign roles** based on the calculated scores

### Scoring System

#### LP Score (Control-Oriented Objects)
```
lp_score = (
    monitors × 3 +        # TV/monitor displays
    laptops × 2 +          # Laptop computers
    keyboards × 2 +        # Control keyboards
    mouse × 1 +            # Computer mouse
    cell_phone × 1 +       # Mobile phones
    remotes × 2            # Control panels/remotes
)
```

**Objects that indicate LP role:**
- **TV/Monitor** (weight: 3) - Strong indicator of control station
- **Laptop** (weight: 2) - Computer for controls
- **Keyboard** (weight: 2) - Control input device
- **Mouse** (weight: 1) - Control pointing device
- **Cell Phone** (weight: 1) - Communication device
- **Remote** (weight: 2) - Control panel indicators

#### ALP Score (Documentation-Oriented Objects)
```
alp_score = (
    books × 3 +           # Books/logbooks
    notebooks × 3 +       # Writing notebooks
    backpacks × 1         # Personal bags
)
```

**Objects that indicate ALP role:**
- **Book/Logbook** (weight: 3) - Strong indicator of record-keeping
- **Notebook** (weight: 3) - Documentation and logs
- **Backpack** (weight: 1) - Personal items

#### Empty Desk Heuristic
If no objects are detected near a person:
- `alp_score += 1` (slight preference for ALP with cleaner workspace)

### Role Assignment Rules

#### Case 1: Single Person
```
Only 1 person detected → Assigned as LP (Loco Pilot)
```

#### Case 2: Two People
```
Person with higher LP score → LP (Loco Pilot)
Other person → ALP (Assistant Loco Pilot)
```

#### Case 3: Three or More People
```
Highest LP score → LP (Loco Pilot)
Second highest LP score → ALP (Assistant Loco Pilot)

For remaining people:
  - If ALP score > 0 → TRAINEE (likely learning/documenting)
  - Else if LP score > 2 → SUPERVISOR (has some control access)
  - Else → VISITOR (no clear indicators)
```

## Implementation Details

### New Method: `identify_person_roles()`

Located in `locopilot_monitor.py`, this method:

```python
def identify_person_roles(self, frame, person_boxes, detections):
    """
    Identify LP (Loco Pilot) and ALP (Assistant Loco Pilot) 
    based on objects near each person.
    
    Args:
        frame: Current video frame
        person_boxes: List of person bounding boxes
        detections: Dictionary of YOLO detections
        
    Returns:
        Dictionary mapping person index to role info
    """
```

**Returns:**
```python
{
    0: {
        'role': 'LP',
        'role_name': 'Loco Pilot',
        'lp_score': 5,
        'alp_score': 1,
        'bbox': [x1, y1, x2, y2],
        'objects': [...]  # List of nearby objects
    },
    1: {
        'role': 'ALP',
        'role_name': 'Assistant Loco Pilot',
        'lp_score': 2,
        'alp_score': 4,
        'bbox': [x1, y1, x2, y2],
        'objects': [...]
    }
}
```

### Integration with Activity Detection

Person role information is now automatically:

1. **Detected** during video frame processing
2. **Stored** in activity tracking data
3. **Visualized** in annotated frames with color-coded labels
4. **Exported** in the `activities.json` output

### Activity JSON Output Format

Each activity now includes an optional `personRoles` field:

```json
{
  "tripId": "TRIP-123",
  "activityType": 2,
  "des": "Using mobile phone",
  "peopleCount": 2,
  "personRoles": [
    {
      "personIndex": 0,
      "role": "LP",
      "roleName": "Loco Pilot",
      "lpScore": 5,
      "alpScore": 1
    },
    {
      "personIndex": 1,
      "role": "ALP",
      "roleName": "Assistant Loco Pilot",
      "lpScore": 2,
      "alpScore": 4
    }
  ],
  ...
}
```

## Visualization

### Color Coding in Annotated Frames

Person bounding boxes are color-coded by role:

- **🟡 Yellow (Cyan)** - LP (Loco Pilot)
- **🟠 Orange** - ALP (Assistant Loco Pilot)
- **🟣 Purple** - SUPERVISOR
- **🔵 Cyan** - TRAINEE
- **⚪ Gray** - VISITOR

### Labels Display

Each person box shows:
```
Role Name (LP:5/ALP:1)
```

Where:
- **Role Name**: Human-readable role (e.g., "Loco Pilot")
- **LP:5**: LP score calculated from nearby objects
- **ALP:1**: ALP score calculated from nearby objects

### Role Summary Overlay

In the top-right corner of annotated frames:
```
People Count: 2
Loco Pilot: LP=5, ALP=1
Assistant Loco Pilot: LP=2, ALP=4
```

## Usage Examples

### Example 1: Basic Video Processing

```python
from locopilot_monitor import LocopilotActivityMonitor

# Create monitor
monitor = LocopilotActivityMonitor(
    video_path="video.mp4",
    output_dir="locopilot_evidence",
    save_annotated_frames=True,  # Enable to see role annotations
    sample_fps=0.5
)

# Process video (LP/ALP identification happens automatically)
monitor.process_video()

# Check results
import json
with open(f"{monitor.run_dir}/activities.json", 'r') as f:
    activities = json.load(f)

# Find activities with person roles
for activity in activities:
    if 'personRoles' in activity:
        print(f"Activity: {activity['des']}")
        for role in activity['personRoles']:
            print(f"  {role['roleName']}: LP={role['lpScore']}, ALP={role['alpScore']}")
```

### Example 2: Using the API

```python
from app.services.activity_detection_service import ActivityDetectionService

service = ActivityDetectionService()

# Process video with real detection (includes LP/ALP identification)
activities = service.detect_activities_real(
    video_path="video.mp4",
    trip_id="TRIP-001",
    crew_name="John Doe",
    crew_id="C-001",
    crew_role=1
)

# Activities now include personRoles field
for activity in activities:
    if activity.get('personRoles'):
        print(f"Detected roles in {activity['des']}:")
        for role in activity['personRoles']:
            print(f"  - {role['roleName']}")
```

## YOLO Object Classes Used

The system uses pre-trained YOLO11 model which detects:

### Control Objects (LP indicators)
- `tv` - Television/Monitor displays
- `laptop` - Laptop computers
- `keyboard` - Computer keyboards
- `mouse` - Computer mice
- `cell phone` - Mobile phones
- `remote` - Remote controls/panels

### Documentation Objects (ALP indicators)
- `book` - Books and logbooks
- `backpack` - Backpacks and bags

### Person Detection
- `person` - Human detection

## Model Files

The system uses YOLO11 models located in the project root:
- `yolo11s.pt` - Small model (faster, default)
- `yolo11m.pt` - Medium model (more accurate)

## Configuration

### Adjusting Scoring Weights

To modify the scoring logic, edit `identify_person_roles()` in `locopilot_monitor.py`:

```python
# LP score calculation
lp_score = (
    lp_objects['tv'] * 3 +         # Adjust weight here
    lp_objects['laptop'] * 2 +
    lp_objects['keyboard'] * 2 +
    lp_objects['mouse'] * 1 +
    lp_objects['cell phone'] * 1 +
    lp_objects['remote'] * 2
)

# ALP score calculation
alp_score = (
    alp_objects['book'] * 3 +      # Adjust weight here
    alp_objects['notebook'] * 3 +
    alp_objects['backpack'] * 1
)
```

### Search Region Configuration

The search region for objects near a person is defined by:

```python
search_margin = person_width * 1.5  # Horizontal margin
search_y1 = py1 + (person_height * 0.3)  # Start from chest level
search_y2 = py2 + (person_height * 0.5)  # Extend below person (desk area)
```

Adjust these multipliers to change the search area.

## Testing

### Running Tests

```bash
# Test with example video
python3 test_lp_alp_identification.py

# Or use the main script directly
python3 locopilot_monitor.py
```

### Viewing Results

1. **Activities JSON**: Check `locopilot_evidence/run_YYYYMMDD_HHMMSS/activities.json`
2. **Annotated Frames**: Check `locopilot_evidence/run_YYYYMMDD_HHMMSS/frames/` (if enabled)
3. **Video Clips**: Check `locopilot_evidence/run_YYYYMMDD_HHMMSS/clips/`

### Validation

Verify the implementation by:

1. **Check console output** for role identification messages:
   ```
   [HH:MM:SS] Person roles identified:
     Person 1: Loco Pilot (LP score: 5, ALP score: 1)
     Person 2: Assistant Loco Pilot (LP score: 2, ALP score: 4)
   ```

2. **Inspect annotated frames** to see role labels and color coding

3. **Review activities.json** to confirm `personRoles` field is populated

## Troubleshooting

### Issue: No person roles detected

**Possible causes:**
- Only one person in video (default assigned as LP)
- YOLO confidence too low for object detection
- Objects not in the search region near persons

**Solutions:**
- Lower YOLO confidence threshold in `identify_person_roles()`: `conf=0.3` → `conf=0.2`
- Adjust search region margins
- Check that video quality is sufficient for object detection

### Issue: Incorrect role assignment

**Possible causes:**
- Object detection missed key items
- Search region not covering desk/console area
- Scoring weights not appropriate for scenario

**Solutions:**
- Adjust scoring weights for different objects
- Modify search region parameters
- Add more object classes to detect (if needed)

### Issue: Role changes during activity

**Behavior:**
Person roles are updated continuously during activity tracking. The final activity JSON contains the most recent role assignment.

**To change:** Modify `start_activity()` to lock roles at activity start instead of updating them.

## API Integration

### REST API Endpoint

The role information is automatically included in activity detection API responses:

```bash
POST /api/videos/process
{
  "video_path": "path/to/video.mp4",
  "trip_id": "TRIP-001",
  "use_multiprocessing": false
}

# Response includes personRoles in activities
{
  "status": "success",
  "activities": [
    {
      "tripId": "TRIP-001",
      "activityType": 2,
      "des": "Using mobile phone",
      "personRoles": [
        {
          "personIndex": 0,
          "role": "LP",
          "roleName": "Loco Pilot",
          "lpScore": 5,
          "alpScore": 1
        }
      ]
    }
  ]
}
```

## Performance Considerations

The LP/ALP identification:
- **Adds minimal overhead** - single YOLO inference per sampled frame
- **Runs in parallel** with existing detection pipeline
- **No significant impact** on processing speed
- **Works with multiprocessing** mode

## Future Enhancements

Potential improvements:
1. **Machine learning for role prediction** - Train a model to recognize LP/ALP from posture/position
2. **Temporal consistency** - Smooth role assignments across frames
3. **Custom object classes** - Train YOLO to detect locomotive-specific control panels
4. **Spatial analysis** - Use seating position relative to controls
5. **Activity-based refinement** - Adjust role confidence based on activity patterns

## References

- Main implementation: `locopilot_monitor.py` → `identify_person_roles()`
- Activity models: `app/models/activity_models.py` → `PersonRoleModel`
- Visualization: `locopilot_monitor.py` → `draw_bounding_boxes()`
- Test script: `test_lp_alp_identification.py`


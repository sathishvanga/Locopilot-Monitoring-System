# Implementation Plan: Majority Voting for Activity Detection

## Overview

Replace consecutive frame detection with **majority voting** using a sliding window approach. This provides more robust detection by requiring a majority of positive detections within a configurable window, rather than requiring all consecutive frames to be positive.

**User Preferences:**
- Window size: 5 frames (10 seconds @ 0.5 FPS)
- Per-activity configuration: Yes
- Mode: Hybrid (start voting on first detection, use majority to confirm, keep grace periods)

---

## Files to Modify

| File | Changes |
|------|---------|
| `locopilot_monitor.py` | Main implementation - voting classes, config, and processing loop |

---

## Implementation Steps

### Step 1: Add Voting Data Structures (lines ~60-100)

Add after imports:

```python
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

class VotingState(Enum):
    IDLE = "idle"           # No recent detections
    VOTING = "voting"       # Collecting votes after first detection
    CONFIRMED = "confirmed" # Majority reached

@dataclass
class ActivityVotingWindow:
    window_size: int
    min_positive: int
    detection_history: Deque[bool] = field(default_factory=deque)
    state: VotingState = VotingState.IDLE
    grace_counter: int = 0

    def add_detection(self, detected: bool) -> bool:
        """Add detection to window, return True if majority confirmed"""
        self.detection_history.append(detected)
        if len(self.detection_history) > self.window_size:
            self.detection_history.popleft()

        if detected:
            self.grace_counter = 0
            if self.state == VotingState.IDLE:
                self.state = VotingState.VOTING

        positive_count = sum(self.detection_history)
        if positive_count >= self.min_positive:
            self.state = VotingState.CONFIRMED
            return True
        return self.state == VotingState.CONFIRMED

    def check_grace(self, grace_frames: int) -> bool:
        """Check if still in grace period. Returns False if expired."""
        if self.state == VotingState.CONFIRMED:
            self.grace_counter += 1
            if self.grace_counter <= grace_frames:
                return True
            self.reset()
        return False

    def reset(self):
        self.detection_history.clear()
        self.state = VotingState.IDLE
        self.grace_counter = 0
```

### Step 2: Add Voting Configuration (replace lines 300-366)

Replace `activity_thresholds` with `voting_config`:

```python
self.voting_config = {
    'cell_phone': {
        'window_size': 5,    # 5 frames = 10 seconds window
        'min_positive': 3,   # Need 3/5 positive (60%)
        'grace_frames': 8,
        'min_duration': 0.1,
        'margin': 180,
    },
    'writing': {
        'window_size': 5,
        'min_positive': 3,
        'grace_frames': 10,
        'min_duration': 0.1,
        'margin': 180,
    },
    'sleep': {
        'window_size': 7,    # Longer window for sleep
        'min_positive': 4,   # Need 4/7 positive (~57%)
        'grace_frames': 10,
        'min_duration': 20.0,
    },
    'microsleep': {
        'window_size': 4,
        'min_positive': 2,   # Need 2/4 positive (50%)
        'grace_frames': 10,
        'min_duration': 3.0,
    },
    'packing_bags': {
        'window_size': 3,
        'min_positive': 2,   # Need 2/3 positive (67%)
        'grace_frames': 5,
        'min_duration': 0.0,
        'margin': 50,
    },
    'group_detected': {
        'window_size': 5,
        'min_positive': 3,
        'grace_frames': 8,
        'min_duration': 0.0,
    },
    'lp_hand_gesture': {
        'window_size': 4,
        'min_positive': 2,
        'grace_frames': 5,
        'min_duration': 0.0,
    },
    'alp_hand_gesture': {
        'window_size': 4,
        'min_positive': 2,
        'grace_frames': 5,
        'min_duration': 0.0,
    },
    'mind_diversion': {
        'window_size': 4,
        'min_positive': 2,
        'grace_frames': 5,
        'min_duration': 0.0,
    },
    'no_person_detected': {
        'window_size': 6,
        'min_positive': 4,   # Need 4/6 positive (67%)
        'grace_frames': 5,
        'min_duration': 10.0,
    },
}
```

### Step 3: Replace Detection Counters (lines 369-394)

Replace `consecutive_detections` and `grace_counters` with voting windows:

```python
# Initialize voting windows for each activity
self.voting_windows = {}
for activity_name, config in self.voting_config.items():
    self.voting_windows[activity_name] = ActivityVotingWindow(
        window_size=config['window_size'],
        min_positive=config['min_positive']
    )
```

### Step 4: Update Temporal Filtering Loop (lines 4978-5021)

Replace the detection loop in `process_video()`:

```python
for activity_name, detected in activities_map.items():
    config = self.voting_config[activity_name]
    voting_window = self.voting_windows[activity_name]

    if detected:
        # Add positive detection to voting window
        is_confirmed = voting_window.add_detection(True)

        if is_confirmed:
            # Majority confirmed - start/continue activity
            if not self.activities[activity_name]['active']:
                self.start_activity(activity_name, timestamp, fps, frame_idx, person_roles=person_roles)

            if self.activities[activity_name]['active']:
                self.activities[activity_name]['frames'].append(frame.copy())
                self.activities[activity_name]['last_frame_count'] = frame_idx
                self.activities[activity_name]['last_detected_frame'] = frame_idx
                if person_roles:
                    self.activities[activity_name]['person_roles'] = person_roles
    else:
        # Add negative detection to voting window
        voting_window.add_detection(False)

        if self.activities[activity_name]['active']:
            # Check grace period
            still_in_grace = voting_window.check_grace(config['grace_frames'])
            if not still_in_grace:
                # Grace expired - end activity
                self.end_activity(activity_name, timestamp, fps, frame_idx, people_count)
```

### Step 5: Update Debug Logging (lines 5023-5037)

Update progress logging to show voting status:

```python
if sample_idx % 50 == 0:
    progress = (frame_idx / total_frames) * 100
    self.logger.info(f"Progress: {sample_idx} samples (frame {frame_idx}/{total_frames}, {progress:.1f}%)")

    active_detections = []
    for act_name, voting_window in self.voting_windows.items():
        if voting_window.state != VotingState.IDLE:
            pos = sum(voting_window.detection_history)
            total = len(voting_window.detection_history)
            min_req = voting_window.min_positive
            status = "ACTIVE" if self.activities[act_name]['active'] else f"voting {pos}/{total} (need {min_req})"
            active_detections.append(f"{act_name}: {status}")

    if active_detections:
        self.logger.debug(f"  Voting: {', '.join(active_detections)}")
```

### Step 6: Update `process_video_range()` (multiprocessing)

Apply the same changes to the `process_video_range()` method (around line 5280) for multiprocessing support.

---

## Voting Logic Summary

```
Frame N detected → Add to sliding window → Check majority
                                              ↓
                         ┌────────────────────┴────────────────────┐
                         │                                         │
                   Majority reached                          Not yet reached
                   (e.g., 3/5 positive)                      (e.g., 2/5 positive)
                         │                                         │
                         ↓                                         ↓
                   CONFIRMED state                          Continue voting
                   Start/continue activity                  (wait for more frames)
                         │
                         ↓
              Frame not detected? → Grace period countdown
                                         │
                         ┌───────────────┴───────────────┐
                         │                               │
                   Within grace                    Grace expired
                   (keep activity alive)           (end activity, reset)
```

---

## Benefits Over Consecutive Detection

| Scenario | Consecutive (old) | Majority Voting (new) |
|----------|-------------------|----------------------|
| Brief detection gap (1 frame) | Resets counter | Tolerates (3/5 still passes) |
| Intermittent detection | May never trigger | Triggers if majority positive |
| False positive (1 frame) | May trigger | Filtered out (1/5 not enough) |

---

## Testing Checklist

- [ ] Cell phone detection confirms with 3/5 frames positive
- [ ] Sleep detection confirms with 4/7 frames positive
- [ ] Brief gaps (1-2 frames) don't reset confirmed activities
- [ ] Activities end correctly after grace period expires
- [ ] Multiprocessing mode works correctly
- [ ] Debug logging shows voting progress

# Train Motion Detection — Vibration-Based Approach

## Problem

The CCTV camera points **down into the cab interior**, not at the scenery. Traditional "scenery change through the window" optical flow fails because the outside world is only visible through a tiny strip on the far-left edge (~12% of frame width). The persons (LP/ALP) occupy 40-60% of the frame, and the remaining visible area is mostly static cab interior (green walls, instruments, chairs).

## Solution

Detect motion via **mechanical vibration** — a moving train shakes the entire cab, and this shows up as subtle frame-to-frame pixel jitter on static surfaces (walls, control panel, seat backs). Combined with optical flow on the narrow window strip and block-wise variance analysis.

## Step-by-Step Pipeline (per sampled frame)

```
Frame N-1                    Frame N
   |                            |
   v                            v
+--------------------------------------+
|  STEP 1: YOLO Person Detection       |
|  -> Find all person bounding boxes   |
+----------------+---------------------+
                 v
+--------------------------------------+
|  STEP 2: Create Interior Mask        |
|  Mask OUT (ignore):                  |
|    - Person bboxes (+ 10% padding)   |
|    - Left window strip (0-12% x)     |
|    - Top 5% (timestamp overlay)      |
|    - Bottom 8% (camera text)         |
|  Keep (analyze):                     |
|    - Green walls                     |
|    - Control panel surface           |
|    - Chair backs / seat surfaces     |
|    - Right-side wall & instruments   |
+----------------+---------------------+
                 v
+------------------------------------------------------+
|  STEP 3: Three Parallel Signals                      |
|                                                      |
|  +-----------------------------------+               |
|  | SIGNAL 1: VIBRATION (weight=50%)  |               |
|  |                                   |               |
|  | cv2.absdiff(frame_N, frame_N-1)   |               |
|  | on interior mask pixels only      |               |
|  |                                   |               |
|  | Moving train vibrates the cab ->  |               |
|  | walls/instruments shift by        |               |
|  | 1-3 pixels between frames ->      |               |
|  | higher mean absolute difference   |               |
|  |                                   |               |
|  | Stopped: vib_mean = 0.1 - 0.3    |               |
|  | Smooth:  vib_mean = 2.0 - 4.0    |               |
|  | Rough:   vib_mean = 5.0 - 28.0   |               |
|  |                                   |               |
|  | Score: 0 if <1.0                  |               |
|  |        linear 1.0 -> 3.0         |               |
|  |        1.0 if >=3.0              |               |
|  +-----------------------------------+               |
|                                                      |
|  +-----------------------------------+               |
|  | SIGNAL 2: WINDOW FLOW (wt=30%)   |               |
|  |                                   |               |
|  | Optical flow (Farneback) on the   |               |
|  | narrow left-edge window strip     |               |
|  | (0-12% of frame width)            |               |
|  |                                   |               |
|  | If scenery visible -> flow        |               |
|  | magnitude rises when moving       |               |
|  |                                   |               |
|  | Score: 0 if flow_mag < 2.0        |               |
|  |        linear 2.0 -> 4.0         |               |
|  |        1.0 if >=4.0              |               |
|  +-----------------------------------+               |
|                                                      |
|  +-----------------------------------+               |
|  | SIGNAL 3: BLOCK VAR (wt=20%)     |               |
|  |                                   |               |
|  | Divide interior into 16x16 blocks |               |
|  | Compute variance per block        |               |
|  | Compare with previous frame       |               |
|  |                                   |               |
|  | Vibration changes local texture   |               |
|  | patterns -> variance shifts       |               |
|  |                                   |               |
|  | Score: 0 if delta < 800           |               |
|  |        linear 800 -> 1200         |               |
|  |        1.0 if >=1200              |               |
|  +-----------------------------------+               |
+----------------+-------------------------------------+
                 v
+--------------------------------------+
|  STEP 4: Combined Score              |
|                                      |
|  score = 0.5 x vib_score            |
|        + 0.3 x window_score         |
|        + 0.2 x block_var_score      |
|                                      |
|  if score >= 0.45 -> raw = RUNNING  |
|  if score <  0.45 -> raw = STOPPED  |
+----------------+---------------------+
                 v
+--------------------------------------+
|  STEP 5: Temporal Smoothing          |
|                                      |
|  Sliding window of last 5 frames:   |
|                                      |
|  >=60% RUNNING -> smoothed = RUNNING|
|  >=60% STOPPED -> smoothed = STOPPED|
|  otherwise     -> UNCERTAIN          |
|                                      |
|  Absorbs 1-2 frame flickers from    |
|  person movement or H.264 keyframe  |
|  compression artifacts              |
+--------------------------------------+
```

## Why Person Masking is Critical

Without masking, person movement (standing up, reaching for controls, shifting in seat) dominates the frame difference and produces false RUNNING signals even when the train is stopped.

```
Without mask:  person moves arm -> vib_mean = 8.0 -> FALSE RUNNING
With mask:     person blacked out -> only walls jitter -> vib_mean = 0.2 -> CORRECT STOPPED
```

The 10% padding on person bboxes ensures edge pixels around the person boundary (which change as the person moves) are also excluded.

## Why This Works at 0.5 FPS

At 2-second intervals between frames, the vibration signal is actually **amplified** — the train has 2 seconds of continuous shaking, so the cumulative pixel displacement on static surfaces is larger and easier to detect. In contrast, at 30fps the per-frame jitter would be sub-pixel and much harder to threshold.

## Key Threshold Separation

```
              STOPPED              RUNNING
              <------>            <---------------------->
vib_mean:  0.1 --- 0.3    |    2.0 --- 4.0 --- 28.0
                           |
                     threshold = 1.0
                      (clear 10x gap)
```

The gap between stopped baseline (0.1-0.3) and running minimum (2.0+) is ~10x, making the classification robust.

## Validation Results

Tested across 3 videos, 2 locomotive types (IPCamera 02 & 03), day and night conditions:

| Video | Duration | Expected | Result | Accuracy |
|-------|----------|----------|--------|----------|
| `n_4.mp4` | 28 min | Running | 94.3% RUNNING | Correct |
| `actual_v2.mp4` | 6 min | Running | 86.4% RUNNING | Correct |
| `all_activities.mp4` | 38 min | Mixed (mostly stopped) | 75.2% STOPPED + RUNNING in driving segments | Correct |

## Performance

- **Speed**: ~7 samples/sec (0.14s per frame including YOLO person detection)
- **Overhead**: Adds ~2 minutes for a 28-minute video at 0.5 FPS
- **Dependencies**: OpenCV (already used), YOLO (already loaded in pipeline)
- **No additional models needed** — reuses existing YOLO person detection

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_fps` | 0.5 | Frame sampling rate (matches main pipeline) |
| `yolo_confidence` | 0.25 | YOLO person detection confidence |
| `person_mask_padding` | 0.10 | 10% bbox expansion for mask |
| `window_roi_x1/y1/x2/y2` | 0.0, 0.05, 0.12, 0.85 | Side window ROI (normalized) |
| `vibration_threshold` | 1.0 | Min mean abs diff for motion |
| `vibration_high` | 3.0 | Definite motion threshold |
| `window_flow_threshold` | 2.0 | Min optical flow magnitude in window |
| `weight_vibration` | 0.5 | Weight for vibration signal |
| `weight_window` | 0.3 | Weight for window flow signal |
| `weight_stability` | 0.2 | Weight for block variance signal |
| `running_threshold` | 0.45 | Combined score threshold for RUNNING |
| `temporal_window` | 5 | Smoothing window (frames) |
| `confidence_threshold` | 0.60 | >=60% of window must agree |

## Files

- **Test script**: `test_train_motion.py` — standalone test with CSV + annotated video output
- **Integration target**: `app/core/detectors/train_motion_detector.py` (to be created)

## Why Not VLMs?

VLMs (GPT-4V, Claude Vision, Gemini, LLaVA etc.) were researched but rejected:

- **Accuracy**: Best VLM (Qwen2VL-72B) scores only 47% on camera motion detection (MotionBench CVPR 2025)
- **Speed**: ~2 sec/frame vs 0.14 sec/frame for vibration approach (14x slower)
- **Cost**: API-based: $9-27/video. Self-hosted: ~$1/video
- **Determinism**: LLMs are stochastic — same frame can get different answers
- **Night/tunnels**: VLMs hallucinate when scenery isn't visible

## Why Not Scenery Optical Flow?

The original Option 3+4 approach (mask persons, compute optical flow on background scenery) was implemented first but failed because:

- The cab camera points **down at the interior** — outside scenery is only visible through a ~12% wide strip on the far-left edge
- After masking persons, the remaining "background" is mostly **static cab interior** (walls, instruments) not scenery
- The scenery strip is too small and often obscured by reflections/glare to produce reliable flow

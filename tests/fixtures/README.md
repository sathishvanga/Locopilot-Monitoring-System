# Test fixtures

This directory holds static, hand-curated fixtures that the test suite
replays instead of running real YOLO / pose / MediaPipe inference.  The
goal is for every unit and integration test to be **offline, deterministic,
and GPU-free**.

## Fixture strategy

The Locopilot pipeline has three expensive upstream dependencies:

1. **Ultralytics YOLO** object detector (person, cup, cell_phone, book, ...).
2. **YOLO-Pose / RTMPose** keypoint estimator (17 COCO keypoints/person).
3. **MediaPipe FaceMesh** eye-aspect-ratio and head-pose landmarks.

Tests avoid all three by injecting stand-ins:

| Input | Stand-in | Where defined |
| --- | --- | --- |
| Pose landmarks | `FakePoseLandmarks` + `FakeLandmark` dataclasses with `.x/.y/.z/.visibility` | `tests/conftest.py` |
| Detector `Settings` | `Settings(_env_file=None)` (pristine defaults) | `tests/conftest.py::minimal_settings` |
| Logger | `logging.getLogger("locopilot.tests")` | `tests/conftest.py::stub_logger` |
| Pose factory | `stub_yolo_keypoints` fixture returning a callable that builds a pose | `tests/conftest.py` |

### Planned fixture files

When the integration snapshot test lands (see
`tests/integration/test_frame_pipeline.py`) the following files will live
here alongside this README:

- `frames/` — JPEG frames sampled from the 30-second clip, keyed by frame
  index.
- `yolo_outputs.json` — canned YOLO detections per frame (bbox, class,
  confidence) keyed by frame index.
- `pose_outputs.json` — canned pose keypoints per frame, same keying.
- `expected_activities.json` — the frozen golden `all_activities` list
  produced by running the full pipeline over the canned detections.

## How to add new fixtures

1. Hand-author a minimal JSON snippet rather than dumping a whole video.
   Tests should fail loudly when fixtures drift, not silently accept any
   shape of input.
2. Keep each fixture under a few KB so the repo stays lightweight.
3. Add a short comment at the top of each fixture file describing which
   test consumes it and what scenario it captures (e.g. "microsleep onset
   at frame 12 — regresses the 2026 head-tilt wrap-around fix").

## Why not real model outputs?

Running YOLO / RTMPose against a real clip inside tests is slow,
non-deterministic (torch + cuDNN seeding quirks), and requires model
weights on every CI runner.  Committing canned detections gives us:

- Reproducibility: the same input always produces the same output.
- Speed: unit tests run in milliseconds, not seconds.
- Isolation: detector logic is tested independently of inference quality.

# Locopilot Monitoring System

## Project Overview
CCTV-based activity monitoring for Indian railway locomotive pilots (LP) and assistant loco pilots (ALP). Detects safety-critical activities (sleeping, microsleep, phone use, eating/drinking, etc.) from overhead cabin cameras using YOLO + pose estimation.

## Architecture
- **Main monitor**: `locopilot_monitor.py` (~5200 lines) — core frame processing
- **Detectors**: `app/core/detectors/` — sleep, writing, gesture, head pose, posture entropy, confounder, trajectory
- **Services**: `app/services/` — voting, OCR, S3 upload, multiprocessing, temporal filtering, trip data
- **Config**: `app/utils/config.py` (pydantic-settings) loads from `.env` file
- **API**: `app/main.py` (FastAPI + Gunicorn)

## GPU Server
- **IP**: 103.116.80.162, **Port**: 3781, **User**: admin1
- **App path**: `/opt/poc2`
- **Deploy**: `./deploy-gpu.sh` (rsync + restart systemd service)
- **Logs**: `tail -f /opt/poc2/logs/LocopilotMonitoring.log`
- **Service**: `sudo systemctl status|restart|stop locopilot`
- **Evidence**: `/opt/poc2/locopilot_evidence/run_*/`

## Deployment Flow
1. `deploy-gpu.sh` rsyncs code to server
2. Copies `.env.production` → `.env` on server (production overrides)
3. Installs deps, restarts systemd service
4. **Important**: `.env.production` is what runs on the server, not `.env`

## Environment Config
- `.env` — local development settings
- `.env.production` — GPU server production settings (copied to server `.env` during deploy)
- `.env.example` — template with all available settings documented

### Key Feature Flags
- `TRAIN_MOTION_RULES_ENABLED=1` — enables motion-based rule engine (exempts activities when train is stopped at stations)
- `TRAIN_MOTION_DETECTION_ENABLED=1` — enables vibration-based train motion detection from video frames
- `YOLO_ALWAYS_PREPROCESS=1` — bilateral filter + CLAHE for all frames
- `ZONE_SUPPRESSION_ENABLED=1` — suppress static objects (chairs, etc.)
- `GESTURE_TRAJECTORY_ENABLED=1` — raise-hold-lower state machine for hand gesture FP reduction
- `WRITING_VISUAL_DETECTION_ENABLED=0` — HSV paper segmentation (off by default)
- `POSE_MODEL=yolo|rtmpose|rtmw` — pose estimation backend

## Common Debugging
1. **Check latest run results**: `cat /opt/poc2/locopilot_evidence/run_<latest>/activities.json`
2. **Check processing logs**: grep for `Progress:`, `Activity started`, `SLEEP/MICROSLEEP`, `ERROR`
3. **Evidence frames**: downloaded from `/opt/poc2/locopilot_evidence/run_*/clips/`
4. **FP investigation**: grep for `HEAD DROP DEBUG`, `Pose-Based Sleep`, `Sleep State Machine` with person index

## Known FP Patterns & Fixes
- **Head tilt angle wrapping**: `atan2` discontinuity causes 300+ degree drops on distant persons. Fixed with `(delta + 180) % 360 - 180` normalization + `nose_y_drop >= 0` guard in `sleep_detector.py`
- **Small LP bbox**: LP is far from camera → noisy pose estimation → spurious sleep triggers. The `nose_y_drop >= 0` guard prevents head_tilt-only triggers when nose actually moved up.

## Testing Videos
- Upload via the API at `http://103.116.80.162:8000`
- Results appear in `/opt/poc2/locopilot_evidence/run_<timestamp>/activities.json`
- Evidence frames and clips saved alongside in `clips/` subdirectory

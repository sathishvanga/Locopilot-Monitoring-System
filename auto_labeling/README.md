# Auto-Labeling Pipeline for YOLO Training

Grounding DINO-based auto-labeling for locomotive cabin CCTV videos.

## Files

| File | Description |
|------|-------------|
| `label_video.sh` | One-command script to label a video and append to dataset |
| `auto_label_gpu.py` | Grounding DINO labeling script (runs on GPU server) |
| `visualize_labels.py` | Draw annotations on frames for visual review |

## Quick Start

```bash
# Label a single video
./label_video.sh /path/to/video.mp4

# Label with custom fps (default 0.5 = 1 frame every 2s)
./label_video.sh /path/to/video.mp4 1.0

# Label all videos in a folder
for v in /path/to/videos/*.mp4; do ./label_video.sh "$v"; done
```

## Classes (9)

| ID | Class | Prompt |
|----|-------|--------|
| 0 | person | person sitting in train locomotive cabin |
| 1 | cell_phone | small rectangular mobile phone held in hand |
| 2 | book | open log book or register on desk surface |
| 3 | cup | drinking cup or mug held in hand or on surface |
| 4 | bottle | steel thermos flask or water bottle on desk |
| 5 | backpack | backpack or rucksack bag on floor or seat |
| 6 | handbag | small handbag or carry bag |
| 7 | suitcase | suitcase or luggage bag on floor |
| 8 | radio_handset | handheld radio transceiver or walkie talkie held near face |

## Output

Dataset is saved to `/Users/satishvanga/Documents/training_cvvrs/` with YOLO format:
```
training_cvvrs/
  dataset.yaml
  classes.txt
  train/images/  train/labels/
  val/images/    val/labels/
```

## Train

```bash
yolo train model=yolo11n.pt data=/Users/satishvanga/Documents/training_cvvrs/dataset.yaml epochs=100 imgsz=640
```

#!/bin/bash
# =============================================================
# Auto-Label Video for YOLO Training
# Usage: ./label_video.sh <video_path> [fps]
# Example: ./label_video.sh /path/to/video.mp4 0.5
# =============================================================

set -e

# --- Config (edit these if needed) ---
DATASET_DIR="/Users/satishvanga/Documents/training_cvvrs"
FRAMES_BASE="/Users/satishvanga/Documents/frames_temp"
SERVER="admin1@103.116.80.162"
SERVER_PORT=3781
SERVER_PASS='9o\P`3#W(9}K'
SERVER_PYTHON="/opt/poc2/venv/bin/python3"
SERVER_WORKDIR="/home/admin1/auto_label"
TRAIN_RATIO=0.85

# --- Input ---
VIDEO_PATH="$1"
FPS="${2:-0.5}"  # Default: 1 frame every 2 seconds

if [ -z "$VIDEO_PATH" ]; then
    echo "Usage: ./label_video.sh <video_path> [fps]"
    echo "  fps: frames per second to extract (default: 0.5 = 1 frame every 2s)"
    echo ""
    echo "Examples:"
    echo "  ./label_video.sh /path/to/video1.mp4"
    echo "  ./label_video.sh /path/to/video2.mp4 1.0"
    exit 1
fi

if [ ! -f "$VIDEO_PATH" ]; then
    echo "ERROR: Video not found: $VIDEO_PATH"
    exit 1
fi

# --- Derive video name for unique frame prefixes ---
VIDEO_NAME=$(basename "$VIDEO_PATH" .mp4)
VIDEO_NAME=$(echo "$VIDEO_NAME" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
FRAMES_DIR="$FRAMES_BASE/${VIDEO_NAME}"

echo "============================================================"
echo "Auto-Label Pipeline"
echo "============================================================"
echo "Video:    $VIDEO_PATH"
echo "Name:     $VIDEO_NAME"
echo "FPS:      $FPS"
echo "Dataset:  $DATASET_DIR"
echo "============================================================"

# --- Step 1: Extract frames ---
echo ""
echo "[Step 1/5] Extracting frames at ${FPS} fps..."
rm -rf "$FRAMES_DIR"
mkdir -p "$FRAMES_DIR"
ffmpeg -i "$VIDEO_PATH" -vf "fps=$FPS" "$FRAMES_DIR/${VIDEO_NAME}_%04d.jpg" -loglevel warning
FRAME_COUNT=$(ls "$FRAMES_DIR"/*.jpg 2>/dev/null | wc -l | tr -d ' ')
echo "  Extracted: $FRAME_COUNT frames"

if [ "$FRAME_COUNT" -eq 0 ]; then
    echo "ERROR: No frames extracted!"
    exit 1
fi

# --- Step 2: Upload frames to GPU server ---
echo ""
echo "[Step 2/5] Uploading frames to GPU server..."
sshpass -p "$SERVER_PASS" ssh -p $SERVER_PORT -o StrictHostKeyChecking=no $SERVER \
    "rm -rf $SERVER_WORKDIR/new_frames && mkdir -p $SERVER_WORKDIR/new_frames"
sshpass -p "$SERVER_PASS" rsync -az -e "ssh -p $SERVER_PORT -o StrictHostKeyChecking=no" \
    "$FRAMES_DIR/" "$SERVER@:$SERVER_WORKDIR/new_frames/"
echo "  Uploaded: $FRAME_COUNT frames"

# --- Step 3: Run auto-labeling on GPU ---
echo ""
echo "[Step 3/5] Running Grounding DINO auto-labeling on GPU..."
echo "  (This takes ~${FRAME_COUNT} / 2.5 ≈ $((FRAME_COUNT * 10 / 25)) seconds)"
sshpass -p "$SERVER_PASS" ssh -p $SERVER_PORT -o StrictHostKeyChecking=no $SERVER \
    "rm -rf $SERVER_WORKDIR/new_output && \
     sed 's|/home/admin1/auto_label/frames|$SERVER_WORKDIR/new_frames|;s|/home/admin1/auto_label/training_cvvrs|$SERVER_WORKDIR/new_output|' \
     $SERVER_WORKDIR/auto_label_gpu.py > $SERVER_WORKDIR/auto_label_new.py && \
     $SERVER_PYTHON $SERVER_WORKDIR/auto_label_new.py"

# --- Step 4: Download labels ---
echo ""
echo "[Step 4/5] Downloading labels..."
TEMP_DOWNLOAD="/tmp/new_labels_${VIDEO_NAME}"
rm -rf "$TEMP_DOWNLOAD"
mkdir -p "$TEMP_DOWNLOAD"
sshpass -p "$SERVER_PASS" rsync -az -e "ssh -p $SERVER_PORT -o StrictHostKeyChecking=no" \
    "$SERVER:$SERVER_WORKDIR/new_output/" "$TEMP_DOWNLOAD/"

# --- Step 5: Append to existing dataset ---
echo ""
echo "[Step 5/5] Appending to existing dataset..."
mkdir -p "$DATASET_DIR/train/images" "$DATASET_DIR/train/labels"
mkdir -p "$DATASET_DIR/val/images" "$DATASET_DIR/val/labels"

# Copy train split
if [ -d "$TEMP_DOWNLOAD/train/images" ]; then
    TRAIN_NEW=$(ls "$TEMP_DOWNLOAD/train/images/"*.jpg 2>/dev/null | wc -l | tr -d ' ')
    cp "$FRAMES_DIR"/${VIDEO_NAME}_*.jpg /dev/null 2>&1 || true  # ensure frames exist
    # Copy images from frames dir (not symlinks from server)
    for img in "$TEMP_DOWNLOAD/train/labels/"*.txt; do
        base=$(basename "$img" .txt)
        # Copy actual frame image
        if [ -f "$FRAMES_DIR/${base}.jpg" ]; then
            cp "$FRAMES_DIR/${base}.jpg" "$DATASET_DIR/train/images/"
        fi
        # Copy label
        cp "$img" "$DATASET_DIR/train/labels/"
    done
    echo "  Train: +$(ls "$TEMP_DOWNLOAD/train/labels/"*.txt 2>/dev/null | wc -l | tr -d ' ') frames"
fi

# Copy val split
if [ -d "$TEMP_DOWNLOAD/val/images" ]; then
    for img in "$TEMP_DOWNLOAD/val/labels/"*.txt; do
        base=$(basename "$img" .txt)
        if [ -f "$FRAMES_DIR/${base}.jpg" ]; then
            cp "$FRAMES_DIR/${base}.jpg" "$DATASET_DIR/val/images/"
        fi
        cp "$img" "$DATASET_DIR/val/labels/"
    done
    echo "  Val:   +$(ls "$TEMP_DOWNLOAD/val/labels/"*.txt 2>/dev/null | wc -l | tr -d ' ') frames"
fi

# --- Summary ---
echo ""
echo "============================================================"
echo "DONE! Dataset updated."
echo "============================================================"
TOTAL_TRAIN=$(ls "$DATASET_DIR/train/images/"*.jpg 2>/dev/null | wc -l | tr -d ' ')
TOTAL_VAL=$(ls "$DATASET_DIR/val/images/"*.jpg 2>/dev/null | wc -l | tr -d ' ')
echo "  Total train: $TOTAL_TRAIN images"
echo "  Total val:   $TOTAL_VAL images"
echo "  Dataset:     $DATASET_DIR/dataset.yaml"
echo ""
echo "Review samples with:"
echo "  open $DATASET_DIR/train/images/"
echo ""
echo "Train with:"
echo "  yolo train model=yolo11n.pt data=$DATASET_DIR/dataset.yaml epochs=100 imgsz=640"
echo "============================================================"

# Cleanup temp
rm -rf "$TEMP_DOWNLOAD"

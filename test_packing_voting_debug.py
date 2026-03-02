#!/usr/bin/env python3
"""
Test script: Packing Bags Voting Debug
=======================================
Extracts frames at packing detection timestamps from the video,
runs both nano and large YOLO models, and compares backpack/suitcase detections.

Purpose: Diagnose why voting (large model) gets 0/10 backpack detections
while initial detection (nano model) finds backpacks.

Usage: python3 test_packing_voting_debug.py
"""

import cv2
import sys
import os

# Packing detection timestamps (seconds) from logs
PACKING_TIMESTAMPS = [1170.0, 2076.0]

# Video path
VIDEO_PATH = "/tmp/locopilot_uploads/all_activities.mp4"

# Models
NANO_MODEL = "yolo26n.pt"
LARGE_MODEL = "yolo26l.pt"

# Voting frame spread: 10 frames at 400ms intervals around the timestamp
VOTING_NUM_FRAMES = 10
VOTING_FRAME_SPREAD_MS = 400

# YOLO classes of interest
BAG_CLASSES = {"backpack", "suitcase", "handbag"}

# Output directory for debug frames
OUTPUT_DIR = "/tmp/packing_voting_debug"


def extract_voting_frames(video_path, timestamp_sec, num_frames=10, spread_ms=400):
    """Extract the same frames that voting verification would use."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = int(timestamp_sec * fps)

    print(f"\n{'='*70}")
    print(f"Timestamp: {timestamp_sec}s | Target frame: {target_frame} | FPS: {fps} | Total: {total_frames}")
    print(f"{'='*70}")

    # Calculate frame positions (same logic as voting service)
    spread_frames = int(spread_ms * fps / 1000)  # 400ms * 25fps / 1000 = 10 frames
    half = num_frames // 2
    frame_positions = []
    for i in range(num_frames):
        offset = (i - half) * spread_frames
        pos = max(0, min(target_frame + offset, total_frames - 1))
        frame_positions.append(pos)

    print(f"Frame positions: {frame_positions}")
    print(f"Time range: {frame_positions[0]/fps:.2f}s - {frame_positions[-1]/fps:.2f}s")

    # Also include the exact sampled frame (what nano model sees at 0.5fps)
    sampled_frame_num = int(round(timestamp_sec * fps / (fps / 0.5)) * (fps / 0.5))
    if sampled_frame_num not in frame_positions:
        frame_positions.insert(0, sampled_frame_num)
        print(f"Added sampled frame (0.5fps): {sampled_frame_num} ({sampled_frame_num/fps:.2f}s)")

    frames = {}
    for pos in frame_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frames[pos] = frame

    cap.release()
    return frames, fps


def run_yolo_on_frames(frames, model_path, device="0"):
    """Run YOLO model on extracted frames and report bag detections."""
    from ultralytics import YOLO

    print(f"\n--- Loading model: {model_path} ---")
    model = YOLO(model_path)

    results_summary = {}

    for frame_num, frame in sorted(frames.items()):
        results = model(frame, device=device, verbose=False, imgsz=640)

        bag_detections = []
        all_detections = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]
                conf = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy().tolist()

                all_detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": [round(x, 1) for x in bbox]
                })

                if cls_name in BAG_CLASSES:
                    bag_detections.append({
                        "class": cls_name,
                        "confidence": conf,
                        "bbox": [round(x, 1) for x in bbox],
                        "area": round((bbox[2]-bbox[0]) * (bbox[3]-bbox[1]))
                    })

        results_summary[frame_num] = {
            "bag_detections": bag_detections,
            "all_classes": [f"{d['class']}({d['confidence']:.2f})" for d in all_detections]
        }

    return results_summary


def save_debug_frames(frames, nano_results, large_results, timestamp, fps, output_dir):
    """Save annotated debug frames showing detections from both models."""
    ts_dir = os.path.join(output_dir, f"t{int(timestamp)}s")
    os.makedirs(ts_dir, exist_ok=True)

    for frame_num, frame in sorted(frames.items()):
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw nano detections (green)
        if frame_num in nano_results:
            for det in nano_results[frame_num]["bag_detections"]:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"NANO: {det['class']} {det['confidence']:.2f}"
                cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw large detections (blue)
        if frame_num in large_results:
            for det in large_results[frame_num]["bag_detections"]:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"LARGE: {det['class']} {det['confidence']:.2f}"
                cv2.putText(annotated, label, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Add frame info
        time_sec = frame_num / fps
        info = f"Frame {frame_num} ({time_sec:.2f}s)"
        cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out_path = os.path.join(ts_dir, f"frame_{frame_num}.jpg")
        cv2.imwrite(out_path, annotated)

    print(f"  Debug frames saved to: {ts_dir}/")


def main():
    print("=" * 70)
    print("  PACKING BAGS VOTING DEBUG TEST")
    print("  Comparing nano vs large YOLO model backpack detections")
    print("=" * 70)

    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video not found at {VIDEO_PATH}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ts in PACKING_TIMESTAMPS:
        frames, fps = extract_voting_frames(VIDEO_PATH, ts, VOTING_NUM_FRAMES, VOTING_FRAME_SPREAD_MS)
        print(f"Extracted {len(frames)} frames")

        # Run nano model
        print(f"\n{'*'*50}")
        print(f"  NANO MODEL ({NANO_MODEL})")
        print(f"{'*'*50}")
        nano_results = run_yolo_on_frames(frames, NANO_MODEL)

        nano_bag_count = 0
        for fn in sorted(nano_results):
            bags = nano_results[fn]["bag_detections"]
            all_cls = nano_results[fn]["all_classes"]
            bag_str = f" -> BAGS: {bags}" if bags else ""
            has_bag = "YES" if bags else "NO "
            if bags:
                nano_bag_count += 1
            print(f"  Frame {fn:>6} ({fn/fps:>7.2f}s): [{has_bag}] classes={all_cls}{bag_str}")

        # Run large model
        print(f"\n{'*'*50}")
        print(f"  LARGE MODEL ({LARGE_MODEL})")
        print(f"{'*'*50}")
        large_results = run_yolo_on_frames(frames, LARGE_MODEL)

        large_bag_count = 0
        for fn in sorted(large_results):
            bags = large_results[fn]["bag_detections"]
            all_cls = large_results[fn]["all_classes"]
            bag_str = f" -> BAGS: {bags}" if bags else ""
            has_bag = "YES" if bags else "NO "
            if bags:
                large_bag_count += 1
            print(f"  Frame {fn:>6} ({fn/fps:>7.2f}s): [{has_bag}] classes={all_cls}{bag_str}")

        # Summary for this timestamp
        print(f"\n{'='*70}")
        print(f"  SUMMARY @ {ts}s")
        print(f"  Nano  model: {nano_bag_count}/{len(frames)} frames with bag detected")
        print(f"  Large model: {large_bag_count}/{len(frames)} frames with bag detected")
        voting_threshold = 0.4  # 40%
        would_pass = large_bag_count / len(frames) >= voting_threshold
        print(f"  Voting threshold: {voting_threshold*100:.0f}% -> {'PASS' if would_pass else 'FAIL'} ({large_bag_count}/{len(frames)} = {large_bag_count/len(frames)*100:.0f}%)")
        print(f"{'='*70}")

        # Save debug frames
        save_debug_frames(frames, nano_results, large_results, ts, fps, OUTPUT_DIR)

    print(f"\nDone! Debug frames saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

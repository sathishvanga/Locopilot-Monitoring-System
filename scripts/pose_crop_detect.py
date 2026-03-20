"""
Pose-Guided Crop Detection: Two-stage pipeline test script.

Stage 1: YOLO-Pose → wrist keypoints
Stage 2: YOLO object detection on cropped hand regions

Usage:
    python scripts/pose_crop_detect.py --video <path> [--sample-fps 0.5] [--output-dir output_crops]
"""

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# COCO keypoint indices
KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10

# Target classes we care about (COCO class IDs)
TARGET_CLASSES = {
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    41: "cup",
    43: "knife",
    46: "banana",
    47: "apple",
    49: "sandwich",
    67: "cell_phone",
    73: "book",
}

# Classes expected near hands
HAND_CLASSES = {67: "cell_phone", 41: "cup", 39: "bottle", 73: "book"}

# Classes expected near feet / lower body
LOWER_BODY_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}


def crop_region(frame, cx, cy, margin, frame_h, frame_w):
    """Crop a square region around (cx, cy) with given margin."""
    x1 = max(0, int(cx - margin))
    y1 = max(0, int(cy - margin))
    x2 = min(frame_w, int(cx + margin))
    y2 = min(frame_h, int(cy + margin))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None, (x1, y1, x2, y2)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def run_detection_on_crop(det_model, crop, conf_threshold=0.15):
    """Run object detection on a cropped region."""
    if crop is None or crop.size == 0:
        return []
    results = det_model(crop, conf=conf_threshold, verbose=False)
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id in TARGET_CLASSES:
            detections.append({
                "class_id": cls_id,
                "class_name": TARGET_CLASSES[cls_id],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist(),
            })
    return detections


def process_frame(frame, frame_idx, timestamp, pose_model, det_model,
                  hand_margin=150, lower_margin=200, wrist_conf_threshold=0.3,
                  det_conf=0.15, save_crops=False, output_dir=None):
    """
    Process a single frame:
    1. Run pose to get keypoints
    2. Crop around wrists → detect hand-held objects
    3. Crop lower person bbox → detect bags
    4. Full-frame fallback for comparison
    """
    frame_h, frame_w = frame.shape[:2]
    frame_results = {
        "frame_idx": frame_idx,
        "timestamp": timestamp,
        "persons": [],
        "hand_detections": [],
        "lower_body_detections": [],
        "full_frame_detections": [],
    }

    # --- Stage 1: Pose estimation ---
    pose_results = pose_model(frame, conf=0.25, verbose=False)
    if pose_results[0].keypoints is None:
        return frame_results

    keypoints_data = pose_results[0].keypoints.data
    boxes = pose_results[0].boxes

    for person_idx, (kps, box) in enumerate(zip(keypoints_data, boxes.xyxy)):
        person_conf = float(boxes.conf[person_idx])
        px1, py1, px2, py2 = box.tolist()
        person_h = py2 - py1
        person_w = px2 - px1

        # Scale margin based on person bbox size
        scaled_hand_margin = max(80, int(person_h * 0.25))
        scaled_lower_margin = max(100, int(person_h * 0.35))

        person_info = {
            "person_idx": person_idx,
            "confidence": person_conf,
            "bbox": [px1, py1, px2, py2],
        }
        frame_results["persons"].append(person_info)

        # --- Stage 2a: Hand crop detection (phone, cup, bottle) ---
        for wrist_idx, wrist_name in [(KP_LEFT_WRIST, "left"), (KP_RIGHT_WRIST, "right")]:
            wx, wy, wconf = kps[wrist_idx].tolist()
            if wconf < wrist_conf_threshold:
                continue

            crop, (cx1, cy1, cx2, cy2) = crop_region(
                frame, wx, wy, scaled_hand_margin, frame_h, frame_w
            )
            if crop is None:
                continue

            dets = run_detection_on_crop(det_model, crop, conf_threshold=det_conf)
            for det in dets:
                # Map crop-local bbox back to frame coordinates
                bx1, by1, bx2, by2 = det["bbox"]
                det["bbox_frame"] = [bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1]
                det["source"] = f"{wrist_name}_wrist"
                det["wrist_conf"] = wconf
                det["person_idx"] = person_idx
                frame_results["hand_detections"].append(det)

            if save_crops and output_dir and len(dets) > 0:
                crop_path = os.path.join(
                    output_dir, "crops",
                    f"frame{frame_idx:05d}_p{person_idx}_{wrist_name}_wrist.jpg"
                )
                os.makedirs(os.path.dirname(crop_path), exist_ok=True)
                cv2.imwrite(crop_path, crop)

        # --- Stage 2b: Lower body crop detection (bags) ---
        lower_y_start = int(py1 + person_h * 0.6)
        lower_cx = (px1 + px2) / 2
        lower_cy = (lower_y_start + py2) / 2
        lower_crop, (lx1, ly1, lx2, ly2) = crop_region(
            frame, lower_cx, lower_cy, scaled_lower_margin, frame_h, frame_w
        )
        if lower_crop is not None:
            dets = run_detection_on_crop(det_model, lower_crop, conf_threshold=det_conf)
            for det in dets:
                bx1, by1, bx2, by2 = det["bbox"]
                det["bbox_frame"] = [bx1 + lx1, by1 + ly1, bx2 + lx1, by2 + ly1]
                det["source"] = "lower_body"
                det["person_idx"] = person_idx
                frame_results["lower_body_detections"].append(det)

    # --- Stage 3: Full-frame detection (baseline comparison) ---
    full_results = det_model(frame, conf=det_conf, verbose=False)
    for box in full_results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id in TARGET_CLASSES:
            frame_results["full_frame_detections"].append({
                "class_id": cls_id,
                "class_name": TARGET_CLASSES[cls_id],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist(),
                "source": "full_frame",
            })

    return frame_results


def draw_results(frame, results):
    """Draw detections on frame for visualization."""
    annotated = frame.copy()

    # Draw person bboxes
    for p in results["persons"]:
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 1)

    # Draw hand detections (GREEN — these are the new finds)
    for det in results["hand_detections"]:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_frame"]]
        label = f"HAND-{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw lower body detections (BLUE)
    for det in results["lower_body_detections"]:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_frame"]]
        label = f"LOWER-{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

    # Draw full-frame detections (RED — baseline)
    for det in results["full_frame_detections"]:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = f"FULL-{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(annotated, label, (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return annotated


def main():
    parser = argparse.ArgumentParser(description="Pose-guided crop detection test")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--sample-fps", type=float, default=0.5,
                        help="Sample rate in FPS (default: 0.5 = 1 frame per 2 seconds)")
    parser.add_argument("--output-dir", default="output_pose_crop",
                        help="Output directory for results")
    parser.add_argument("--det-model", default="yolo26n.pt",
                        help="YOLO detection model weights")
    parser.add_argument("--pose-model", default="yolo26n-pose.pt",
                        help="YOLO pose model weights")
    parser.add_argument("--det-conf", type=float, default=0.15,
                        help="Detection confidence threshold")
    parser.add_argument("--hand-margin", type=int, default=150,
                        help="Base crop margin around wrists (px)")
    parser.add_argument("--save-crops", action="store_true",
                        help="Save crop images with detections")
    parser.add_argument("--save-annotated", action="store_true",
                        help="Save annotated frames")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames to process (0=all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print(f"Loading pose model: {args.pose_model}")
    pose_model = YOLO(args.pose_model)
    print(f"Loading detection model: {args.det_model}")
    det_model = YOLO(args.det_model)

    # Open video
    cap = cv2.VideoCapture(args.video)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / args.sample_fps))

    print(f"Video: {args.video}")
    print(f"  FPS: {video_fps:.1f}, Frames: {total_frames}")
    print(f"  Sampling every {frame_interval} frames ({args.sample_fps} fps)")
    print(f"  Detection conf: {args.det_conf}")
    print()

    # CSV log
    csv_path = os.path.join(args.output_dir, "detections.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "frame_idx", "timestamp", "source", "person_idx",
        "class_name", "confidence", "x1", "y1", "x2", "y2"
    ])

    # Stats
    stats = {
        "frames_processed": 0,
        "hand_detections": 0,
        "lower_body_detections": 0,
        "full_frame_detections": 0,
        "hand_only": {},      # classes found ONLY by hand crop
        "full_only": {},      # classes found ONLY by full frame
        "both": {},           # classes found by both
    }

    frame_idx = 0
    processed = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / video_fps
        results = process_frame(
            frame, frame_idx, timestamp,
            pose_model, det_model,
            hand_margin=args.hand_margin,
            det_conf=args.det_conf,
            save_crops=args.save_crops,
            output_dir=args.output_dir,
        )

        # Log to CSV
        for det in results["hand_detections"]:
            bx1, by1, bx2, by2 = det["bbox_frame"]
            writer.writerow([
                frame_idx, f"{timestamp:.1f}", det["source"],
                det["person_idx"], det["class_name"],
                f"{det['confidence']:.3f}", f"{bx1:.0f}", f"{by1:.0f}",
                f"{bx2:.0f}", f"{by2:.0f}"
            ])
        for det in results["lower_body_detections"]:
            bx1, by1, bx2, by2 = det["bbox_frame"]
            writer.writerow([
                frame_idx, f"{timestamp:.1f}", det["source"],
                det["person_idx"], det["class_name"],
                f"{det['confidence']:.3f}", f"{bx1:.0f}", f"{by1:.0f}",
                f"{bx2:.0f}", f"{by2:.0f}"
            ])
        for det in results["full_frame_detections"]:
            bx1, by1, bx2, by2 = det["bbox"]
            writer.writerow([
                frame_idx, f"{timestamp:.1f}", "full_frame",
                -1, det["class_name"],
                f"{det['confidence']:.3f}", f"{bx1:.0f}", f"{by1:.0f}",
                f"{bx2:.0f}", f"{by2:.0f}"
            ])

        # Update stats
        stats["frames_processed"] += 1
        stats["hand_detections"] += len(results["hand_detections"])
        stats["lower_body_detections"] += len(results["lower_body_detections"])
        stats["full_frame_detections"] += len(results["full_frame_detections"])

        # Track which classes found by which method
        hand_classes = {d["class_name"] for d in results["hand_detections"]}
        lower_classes = {d["class_name"] for d in results["lower_body_detections"]}
        crop_classes = hand_classes | lower_classes
        full_classes = {d["class_name"] for d in results["full_frame_detections"]}

        for cls in crop_classes - full_classes:
            stats["hand_only"][cls] = stats["hand_only"].get(cls, 0) + 1
        for cls in full_classes - crop_classes:
            stats["full_only"][cls] = stats["full_only"].get(cls, 0) + 1
        for cls in crop_classes & full_classes:
            stats["both"][cls] = stats["both"].get(cls, 0) + 1

        # Save annotated frame
        if args.save_annotated and (
            results["hand_detections"] or results["lower_body_detections"]
        ):
            annotated = draw_results(frame, results)
            ann_path = os.path.join(
                args.output_dir, "annotated",
                f"frame_{frame_idx:05d}.jpg"
            )
            os.makedirs(os.path.dirname(ann_path), exist_ok=True)
            cv2.imwrite(ann_path, annotated)

        processed += 1
        if processed % 20 == 0:
            elapsed = time.time() - t_start
            print(f"  Processed {processed} frames ({timestamp:.0f}s) "
                  f"| hand={stats['hand_detections']} "
                  f"lower={stats['lower_body_detections']} "
                  f"full={stats['full_frame_detections']} "
                  f"| {elapsed:.1f}s elapsed")

        if args.max_frames > 0 and processed >= args.max_frames:
            break

        frame_idx += 1

    cap.release()
    csv_file.close()
    elapsed = time.time() - t_start

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Frames processed: {stats['frames_processed']}")
    print(f"Time elapsed: {elapsed:.1f}s ({stats['frames_processed']/elapsed:.1f} frames/s)")
    print()
    print(f"Hand crop detections:       {stats['hand_detections']}")
    print(f"Lower body crop detections: {stats['lower_body_detections']}")
    print(f"Full frame detections:      {stats['full_frame_detections']}")
    print()
    print("Classes found ONLY by hand/lower crop (new finds):")
    for cls, count in sorted(stats["hand_only"].items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count} frames")
    print()
    print("Classes found ONLY by full frame (missed by crop):")
    for cls, count in sorted(stats["full_only"].items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count} frames")
    print()
    print("Classes found by BOTH methods:")
    for cls, count in sorted(stats["both"].items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count} frames")
    print()
    print(f"CSV log: {csv_path}")
    if args.save_annotated:
        print(f"Annotated frames: {args.output_dir}/annotated/")


if __name__ == "__main__":
    main()

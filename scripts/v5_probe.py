#!/usr/bin/env python3
"""
V5 Raw Detection Probe — bypass all rule logic, see what the trained
yolo26s_locopilot_v5 model detects per class, per frame, with confidence
stats and adjacent-frame IoU stability.

Purpose:
  Calibrate confidence thresholds empirically (instead of inheriting COCO
  defaults) and diagnose FP patterns by seeing the raw detection stream
  the rule engine is working from.

Usage (on GPU server):
    cd /opt/poc2
    source venv/bin/activate

    # From MinIO (same place the API downloads from):
    python scripts/v5_probe.py --minio-key all_activities.mp4

    # From a local path (already on server):
    python scripts/v5_probe.py --video /tmp/locopilot_uploads/some.mp4

    # Filter to a single class:
    python scripts/v5_probe.py --minio-key all_activities.mp4 --class backpack

    # Custom confidence cutoff (default 0.10 — catches everything):
    python scripts/v5_probe.py --minio-key all_activities.mp4 --conf 0.05

    # Custom sample FPS (default 0.5 = production rate):
    python scripts/v5_probe.py --minio-key all_activities.mp4 --fps 1.0

Outputs (in --out-dir, default /tmp/v5_probe_<ts>/):
    detections.jsonl   per-frame per-detection records (grep/jq friendly)
    summary.json       per-class stats (count, conf percentiles, IoU stability)
    frames/*.jpg       annotated sample frames (one per detection-bearing frame)

Key columns in the stdout summary table:
    #det         total detections above --conf
    #frames      number of sampled frames that had at least one of the class
    %frames      #frames as % of all sampled frames
    mean_conf    mean confidence across all detections
    p10/p50/p90  confidence percentiles (use these to pick rule thresholds)
    min/mean iou IoU between this class's highest-conf bbox in adjacent frames.
                 ~1.0  = truly static object (should be static-suppressed)
                 ~0.6  = stationary but jittery (current static-tracker struggles)
                 <0.3  = moving, NOT static
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# Class ID → name for yolo26s_locopilot_v5.pt
V5_CLASSES = {
    0: 'person',
    1: 'cell_phone',
    2: 'book',
    3: 'cup',
    4: 'bottle',
    5: 'backpack',
    6: 'handbag',
    7: 'suitcase',
    8: 'radio_handset',
}

DEFAULT_MODEL_PATH = '/opt/poc2/yolo26s_locopilot_v5.pt'


def iou(a, b):
    """IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def pct(data, p):
    if not data:
        return None
    sd = sorted(data)
    k = (len(sd) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sd) - 1)
    return sd[f] + (sd[c] - sd[f]) * (k - f)


def download_from_minio(bucket, key, dest):
    """Download a video from MinIO using the server's .env credentials."""
    from minio import Minio

    endpoint = os.getenv('MINIO_ENDPOINT', 'gpu.mindcoinapps.com:9000')
    access_key = os.getenv('MINIO_ACCESS_KEY', 'admin')
    secret_key = os.getenv('MINIO_SECRET_KEY', 'login123')
    secure = os.getenv('MINIO_SECURE', '1') == '1'

    print(f'[probe] MinIO endpoint={endpoint} secure={secure} bucket={bucket}')
    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    client.fget_object(bucket, key, dest)
    return dest


def main():
    ap = argparse.ArgumentParser(description='Raw detection probe for yolo26s_locopilot_v5')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--video', help='Local video path')
    src.add_argument('--minio-key', help='MinIO object key (bucket via --bucket)')
    ap.add_argument('--bucket', default='cvss', help='MinIO bucket (default: cvss)')
    ap.add_argument('--model', default=DEFAULT_MODEL_PATH, help=f'Model path (default: {DEFAULT_MODEL_PATH})')
    ap.add_argument('--fps', type=float, default=0.5, help='Sample FPS (default: 0.5 — production rate)')
    ap.add_argument('--conf', type=float, default=0.10, help='YOLO confidence threshold (default: 0.10)')
    ap.add_argument('--class', dest='klass', default=None, help='Filter to single class name (e.g. backpack, book, cell_phone)')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--max-frame-dumps', type=int, default=30, help='Max annotated frames saved per class (default: 30)')
    ap.add_argument('--device', default=None, help='Device override (e.g. cuda:0, cpu). Default: auto.')
    args = ap.parse_args()

    out_dir = args.out_dir or f'/tmp/v5_probe_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    frames_dir = os.path.join(out_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    print(f'[probe] out_dir: {out_dir}')

    # Resolve video source
    if args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f'ERROR: video not found: {video_path}')
            sys.exit(1)
    else:
        video_path = os.path.join(out_dir, os.path.basename(args.minio_key))
        print(f'[probe] downloading s3://{args.bucket}/{args.minio_key} → {video_path}')
        download_from_minio(args.bucket, args.minio_key, video_path)

    # Load model
    print(f'[probe] loading model: {args.model}')
    model = YOLO(args.model)
    model_names = getattr(model, 'names', {})
    print(f'[probe] model classes: {model_names}')

    # Validate class filter
    if args.klass and args.klass not in V5_CLASSES.values():
        print(f'ERROR: --class "{args.klass}" not in known classes: {list(V5_CLASSES.values())}')
        sys.exit(1)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'ERROR: could not open video: {video_path}')
        sys.exit(1)

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / native_fps if native_fps > 0 else 0
    step = max(1, int(round(native_fps / args.fps)))
    sampled_idxs = list(range(0, total_frames, step))

    print(f'[probe] video: {total_frames} frames @ {native_fps:.1f} fps = {duration:.1f}s')
    print(f'[probe] sampling every {step} frames @ {args.fps} fps = {len(sampled_idxs)} samples')
    print(f'[probe] conf threshold: {args.conf}')
    print(f'[probe] class filter: {args.klass or "ALL"}')
    print()

    # Accumulators
    per_class_confs = {c: [] for c in V5_CLASSES.values()}
    per_class_frame_count = {c: 0 for c in V5_CLASSES.values()}
    per_class_last_bbox = {}
    per_class_ious = {c: [] for c in V5_CLASSES.values()}
    per_class_dumps = {c: 0 for c in V5_CLASSES.values()}

    jsonl_path = os.path.join(out_dir, 'detections.jsonl')
    f_out = open(jsonl_path, 'w')

    t0 = time.time()
    processed = 0
    predict_kwargs = {'conf': args.conf, 'verbose': False}
    if args.device:
        predict_kwargs['device'] = args.device

    for frame_idx in sampled_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        t_sec = frame_idx / native_fps

        try:
            results = model.predict(frame, **predict_kwargs)[0]
        except Exception as e:
            print(f'[probe] predict error at frame {frame_idx}: {e}')
            continue

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            processed += 1
            continue

        frame_best_per_class = {}  # class → (conf, bbox) — highest-conf in this frame
        frame_had_class = set()

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            xyxy = [float(x) for x in boxes.xyxy[i].cpu().numpy().tolist()]
            cls_name = V5_CLASSES.get(cls_id, f'class_{cls_id}')

            if args.klass and cls_name != args.klass:
                continue

            det = {
                'frame': frame_idx,
                't': round(t_sec, 2),
                'class': cls_name,
                'class_id': cls_id,
                'conf': round(conf, 4),
                'bbox': [round(x, 1) for x in xyxy],
            }
            per_class_confs[cls_name].append(conf)
            f_out.write(json.dumps(det) + '\n')

            best = frame_best_per_class.get(cls_name)
            if best is None or conf > best[0]:
                frame_best_per_class[cls_name] = (conf, xyxy)
            frame_had_class.add(cls_name)

        for c in frame_had_class:
            per_class_frame_count[c] += 1
            _, bbox = frame_best_per_class[c]
            if c in per_class_last_bbox:
                iou_val = iou(per_class_last_bbox[c], bbox)
                per_class_ious[c].append(iou_val)
            per_class_last_bbox[c] = bbox

        # Dump an annotated frame once per class (capped)
        if frame_had_class:
            annotated = None
            for c in list(frame_had_class):
                if per_class_dumps[c] < args.max_frame_dumps:
                    if annotated is None:
                        annotated = results.plot()
                    dump_path = os.path.join(frames_dir, f'{c}_{frame_idx:08d}_{t_sec:07.1f}s.jpg')
                    cv2.imwrite(dump_path, annotated)
                    per_class_dumps[c] += 1

        processed += 1
        if processed % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (len(sampled_idxs) / processed - 1) if processed else 0
            print(f'[probe]   {processed}/{len(sampled_idxs)} frames · elapsed {elapsed:.1f}s · ETA {eta:.0f}s')

    cap.release()
    f_out.close()
    elapsed = time.time() - t0
    print(f'[probe] done in {elapsed:.1f}s')
    print()

    # Build summary
    summary = {
        'video': video_path,
        'native_fps': round(native_fps, 2),
        'duration_sec': round(duration, 2),
        'sample_fps': args.fps,
        'sampled_frames': len(sampled_idxs),
        'conf_threshold': args.conf,
        'model': args.model,
        'class_filter': args.klass,
        'classes': {},
    }

    header = (
        f'{"class":<14} {"#det":>6} {"#frames":>8} {"%frames":>8} '
        f'{"mean_conf":>10} {"p10":>7} {"p50":>7} {"p90":>7} '
        f'{"min_iou":>8} {"mean_iou":>9} {"max_iou":>8}'
    )
    print('=' * len(header))
    print(header)
    print('-' * len(header))

    for c in V5_CLASSES.values():
        confs = per_class_confs[c]
        ious = per_class_ious[c]
        frame_pct = round(100 * per_class_frame_count[c] / max(1, len(sampled_idxs)), 1)

        summary['classes'][c] = {
            'count': len(confs),
            'frames_with_class': per_class_frame_count[c],
            'frame_percentage': frame_pct,
            'conf': {
                'mean': round(statistics.mean(confs), 4) if confs else None,
                'median': round(statistics.median(confs), 4) if confs else None,
                'min': round(min(confs), 4) if confs else None,
                'max': round(max(confs), 4) if confs else None,
                'p10': round(pct(confs, 10), 4) if confs else None,
                'p25': round(pct(confs, 25), 4) if confs else None,
                'p75': round(pct(confs, 75), 4) if confs else None,
                'p90': round(pct(confs, 90), 4) if confs else None,
            },
            'adjacent_iou': {
                'mean': round(statistics.mean(ious), 4) if ious else None,
                'median': round(statistics.median(ious), 4) if ious else None,
                'min': round(min(ious), 4) if ious else None,
                'max': round(max(ious), 4) if ious else None,
                'samples': len(ious),
            },
        }

        if confs:
            mi = summary['classes'][c]['adjacent_iou']['min']
            mni = summary['classes'][c]['adjacent_iou']['mean']
            mxi = summary['classes'][c]['adjacent_iou']['max']
            print(
                f'{c:<14} {len(confs):>6} {per_class_frame_count[c]:>8} {frame_pct:>7.1f}% '
                f'{summary["classes"][c]["conf"]["mean"]:>10.3f} '
                f'{summary["classes"][c]["conf"]["p10"]:>7.3f} '
                f'{summary["classes"][c]["conf"]["median"]:>7.3f} '
                f'{summary["classes"][c]["conf"]["p90"]:>7.3f} '
                f'{(mi if mi is not None else 0):>8.3f} '
                f'{(mni if mni is not None else 0):>9.3f} '
                f'{(mxi if mxi is not None else 0):>8.3f}'
            )

    print('=' * len(header))
    print()

    summary_path = os.path.join(out_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'[probe] summary      → {summary_path}')
    print(f'[probe] detections   → {jsonl_path}')
    print(f'[probe] frames       → {frames_dir}')
    print(f'[probe] total dumps  : {sum(per_class_dumps.values())} annotated frames')
    print()
    print('Diagnostic hints:')
    print('  - mean_iou > 0.85 → object is truly static; static-suppression should catch it')
    print('  - mean_iou 0.5-0.85 → object is jittery; current IoU threshold (0.60) may help')
    print('  - mean_iou < 0.5 → object is moving/changing; not a static fixture')
    print('  - p10 conf tells you the lowest confidence the rule engine sees for real detections')
    print('  - use p10 (or slightly higher) as a per-class confidence cutoff')


if __name__ == '__main__':
    main()

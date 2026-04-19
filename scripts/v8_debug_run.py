"""
v8 debug runner — sample video at production rate, draw all YOLO detections,
save annotated frames + per-class summary.

Usage (on GPU server):
    cd /opt/poc2 && source venv/bin/activate
    python3 scripts/v8_debug_run.py [video_path]

Default video: /opt/poc2/uploads/all_activities.mp4
Output: /opt/poc2/v8_debug_frames/run_<timestamp>/
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO_DEFAULT = "/opt/poc2/uploads/all_activities.mp4"
MODEL_PATH = "/opt/poc2/yolo26s_locopilot_v8.pt"
SAMPLE_FPS = 0.5
OUT_BASE = Path("/opt/poc2/v8_debug_frames")

# Production thresholds (match .env.production). Detections below these
# would be rejected at runtime — we still draw them, marked as REJECT,
# so we can see what the heuristic layer is filtering.
PROD_CONF = {
    "person": 0.50,
    "cell_phone": 0.40,
    "book": 0.50,
    "cup": 0.25,
    "bottle": 0.40,
    "backpack": 0.40,
    "handbag": 0.40,
    "suitcase": 0.65,
    "radio_handset": 0.40,
}
# BGR colors per class for bbox drawing
COLORS = {
    "person": (0, 255, 0),
    "cell_phone": (0, 0, 255),
    "book": (0, 255, 255),
    "cup": (255, 0, 255),
    "bottle": (255, 255, 0),
    "backpack": (255, 128, 0),
    "handbag": (128, 0, 255),
    "suitcase": (0, 128, 255),
    "radio_handset": (255, 0, 128),
}
GLOBAL_MIN_CONF = 0.15  # show everything at/above this


def draw_box(img, xyxy, color, thickness, label):
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, cv2.FILLED)
    cv2.putText(img, label, (x1 + 2, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def main() -> int:
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO_DEFAULT
    if not Path(video).exists():
        print(f"ERROR: video not found: {video}")
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_BASE / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:  {MODEL_PATH}")
    print(f"Video:  {video}")
    print(f"Output: {out_dir}")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(src_fps / SAMPLE_FPS)))
    print(f"Source fps={src_fps:.2f}, total_frames={total_frames}, "
          f"sampling every {frame_interval} frames ({SAMPLE_FPS} fps)")

    stats_all = defaultdict(list)        # class -> [conf...]
    stats_pass = defaultdict(list)       # class -> [conf...] that pass threshold
    stats_reject = defaultdict(list)     # class -> [conf...] that fail threshold
    frames_with_class = defaultdict(set) # class -> {frame_idx...}

    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            time_s = frame_idx / src_fps
            res = model.predict(frame, conf=GLOBAL_MIN_CONF, imgsz=640, verbose=False)[0]
            annotated = frame.copy()
            per_frame = Counter()

            for box in res.boxes:
                cls_id = int(box.cls)
                name = model.names[cls_id]
                conf = float(box.conf)
                xyxy = box.xyxy[0].cpu().numpy().astype(int)

                stats_all[name].append(conf)
                passes = conf >= PROD_CONF.get(name, 0.40)
                if passes:
                    stats_pass[name].append(conf)
                    per_frame[name] += 1
                    frames_with_class[name].add(frame_idx)
                    thickness, label_tag = 2, ""
                else:
                    stats_reject[name].append(conf)
                    thickness, label_tag = 1, " X"

                color = COLORS.get(name, (180, 180, 180))
                draw_box(annotated, xyxy, color, thickness,
                         f"{name} {conf:.2f}{label_tag}")

            # Header overlay
            counts_str = " ".join(f"{k}:{v}" for k, v in per_frame.most_common())
            header = f"t={time_s:7.2f}s  f={frame_idx:06d}  {counts_str}"
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 28), (0, 0, 0), cv2.FILLED)
            cv2.putText(annotated, header, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            out_path = out_dir / f"f{frame_idx:06d}_t{int(time_s):04d}s.jpg"
            cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
            saved += 1
            if saved % 100 == 0:
                print(f"  ...saved {saved} frames (frame={frame_idx}, t={time_s:.0f}s)")

        frame_idx += 1

    cap.release()

    # Summary
    summary_path = out_dir / "SUMMARY.txt"
    with open(summary_path, "w") as f:
        f.write(f"Model:        {MODEL_PATH}\n")
        f.write(f"Video:        {video}\n")
        f.write(f"Sample fps:   {SAMPLE_FPS} (every {frame_interval} source frames)\n")
        f.write(f"Frames saved: {saved}\n")
        f.write(f"Global min conf: {GLOBAL_MIN_CONF}\n\n")

        f.write("=== PER-CLASS STATS ===\n")
        f.write(f"{'class':14s} {'total':>6s} {'pass':>6s} {'reject':>6s} "
                f"{'uniq_frm':>9s} {'thr':>5s} {'min':>5s} {'max':>5s} {'mean':>5s}\n")
        all_classes = sorted(set(stats_all) | set(PROD_CONF))
        for name in all_classes:
            a = stats_all.get(name, [])
            p = stats_pass.get(name, [])
            r = stats_reject.get(name, [])
            thr = PROD_CONF.get(name, 0.40)
            uniq = len(frames_with_class.get(name, set()))
            if a:
                mn, mx, mean = min(a), max(a), sum(a) / len(a)
                f.write(f"{name:14s} {len(a):6d} {len(p):6d} {len(r):6d} "
                        f"{uniq:9d} {thr:5.2f} {mn:5.2f} {mx:5.2f} {mean:5.2f}\n")
            else:
                f.write(f"{name:14s} {0:6d} {0:6d} {0:6d} "
                        f"{0:9d} {thr:5.2f} {'-':>5s} {'-':>5s} {'-':>5s}\n")

        f.write("\n=== FRAMES W/ PERSON CONF BELOW THRESHOLD ===\n")
        low_person = [c for c in stats_reject.get("person", []) if c >= 0.30]
        f.write(f"person detections at 0.30–0.50 (rejected by prod): {len(low_person)}\n")
        if low_person:
            f.write(f"  distribution: min={min(low_person):.2f} max={max(low_person):.2f} "
                    f"mean={sum(low_person)/len(low_person):.2f}\n")

    print(f"\nDone. {saved} frames in {out_dir}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

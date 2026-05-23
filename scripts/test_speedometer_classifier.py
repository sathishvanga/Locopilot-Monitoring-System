"""Standalone validation harness for app/services/vlm/speedometer_classifier.

Groups keyframes by activity, runs the classifier per group, and prints the
per-activity verdict alongside the user-supplied ground-truth set. The
threshold defaults to the production setting (5.0) but can be overridden.

Usage:
  python scripts/test_speedometer_classifier.py \
      --clips-dir /opt/poc2/locopilot_evidence/run_20260511_191710/clips \
      --camera 1 \
      --known-stationary 793,915,969,1025,1281
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Bootstrap so we can import the production module without packaging.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from app.services.vlm.speedometer_classifier import (  # noqa: E402
    classify_motion,
    DEFAULT_STOPPED_THRESHOLD,
)


_ACT_RE = re.compile(r"(writing|cell_phone|eating_drinking|packing_bags)_frame(\d+)_\d+")


def group_keyframes(clips_dir: Path) -> Dict[Tuple[str, int], List[Path]]:
    """Group activity.jpg + supp*.jpg files by (act_type, frame_id)."""
    groups: Dict[Tuple[str, int], List[Path]] = defaultdict(list)
    for f in sorted(clips_dir.glob("*activity.jpg")):
        m = _ACT_RE.search(f.name)
        if m:
            groups[(m.group(1), int(m.group(2)))].append(f)
    for f in sorted(clips_dir.glob("*supp*.jpg")):
        m = _ACT_RE.search(f.name)
        if m:
            groups[(m.group(1), int(m.group(2)))].append(f)
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--camera", type=int, required=True, choices=[1, 2],
                    help="1 = LP (ch03), 2 = ALP (ch02)")
    ap.add_argument("--threshold", type=float,
                    default=DEFAULT_STOPPED_THRESHOLD)
    ap.add_argument("--known-stationary", default="",
                    help="comma-separated activity_start_time seconds")
    ap.add_argument("--fps", type=float, default=25.0,
                    help="video fps for frame->second mapping")
    args = ap.parse_args()

    gt = set()
    for s in args.known_stationary.split(","):
        s = s.strip()
        if s:
            try:
                gt.add(float(s))
            except ValueError:
                pass

    groups = group_keyframes(Path(args.clips_dir))
    if not groups:
        print(f"no activities matched in {args.clips_dir}", file=sys.stderr)
        return

    print(f"camera={args.camera}  threshold={args.threshold}  fps={args.fps}")
    print(f"{'frame_id':>10s}  {'t_sec':>6s}  {'score':>7s}  {'stopped':>7s}"
          f"  {'n':>3s}  GT_STATIONARY")
    tp = fp = tn = fn = 0
    for (act_type, frame_id), paths in sorted(groups.items(),
                                              key=lambda kv: kv[0][1]):
        if len(paths) < 3:
            continue
        t_sec = frame_id / args.fps
        result = classify_motion(paths, camera_angle=args.camera,
                                 threshold=args.threshold)
        if result is None:
            print(f"{frame_id:10d}  {t_sec:6.1f}  (skipped, no result)")
            continue
        is_gt_stat = any(abs(t_sec - g) < 30 for g in gt) if gt else None
        stopped_str = "YES" if result["stopped"] else "no"
        gt_str = ("STAT" if is_gt_stat else "") if gt else "?"
        print(f"{frame_id:10d}  {t_sec:6.1f}  {result['score']:7.2f}  "
              f"{stopped_str:>7s}  {result['n_frames']:3d}  {gt_str}")
        if gt:
            if is_gt_stat and result["stopped"]:
                tp += 1
            elif is_gt_stat and not result["stopped"]:
                fn += 1
            elif not is_gt_stat and result["stopped"]:
                fp += 1
            else:
                tn += 1
    if gt:
        print()
        print(f"vs known-stationary set ({sorted(gt)}):")
        print(f"  TP={tp}  FN={fn}  TN={tn}  FP={fp}")
        if tp + fn > 0:
            print(f"  recall  (true stationary caught) = {tp/(tp+fn):.2f}")
        if tp + fp > 0:
            print(f"  precision (predicted stop is correct) = {tp/(tp+fp):.2f}")


if __name__ == "__main__":
    main()

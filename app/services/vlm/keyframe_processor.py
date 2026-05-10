"""Keyframe-handling helpers extracted from ``vlm_verification_service``.

Owns: keyframe resolution (which sampled frames to send to the VLM
for an activity), supplementary-frame sampling from the activity clip,
bbox counting against Pipeline-1 overlay rendering, and the multi-
keyframe stitcher. Functions are copied verbatim from the original
monolithic file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ...utils.logger import get_logger
from .image_encoder import _crop_to_roi


logger = get_logger(__name__)


# Activity types that need FULL-FRAME context (rather than the hand+book ROI
# crop used for writing/eating/packing/cell_phone). Sleep needs body posture,
# mind_diversion needs head pose vs window, no_person checks cabin emptiness,
# group_detected counts distinct people, solo_person needs to scan the whole
# cab for an occluded ALP — all benefit from the whole scene.
_FULL_FRAME_OBJECT_TYPES: frozenset = frozenset({
    "sleep",
    "mind_diversion",
    "no_person_detected",
    "group_detected",
    "solo_person",
})


# Activity types where the pre-VLM no-subject gate must NOT fire because
# the person count in the rendered keyframe is either the violation itself
# (no_person_detected, solo_person) or operationally ambiguous
# (group_detected expects multiple persons but Pipeline-1 may render zero
# bboxes when its own count disagrees with the rendered set). For these
# types, the VLM is the right adjudicator.
_PRE_GATE_SKIP_OBJECT_TYPES: frozenset = frozenset({
    "no_person_detected",
    "group_detected",
    "solo_person",
})


# Activity types whose Pipeline-1 trigger requires a target object
# (book / cup / bag / phone). For these, an absent orange/yellow object
# bbox in EVERY original keyframe is strong evidence of a stale-state
# Pipeline-1 trigger — the rule fired without the actual object bbox
# being detected at the keyframe time. Used by the pre-VLM object gate.
_OBJECT_REQUIRED_TYPES: frozenset = frozenset({
    "writing",
    "eating_drinking",
    "packing_bags",
    "cell_phone",
})


def _resolve_keyframes(activity: Dict[str, Any]) -> List[Path]:
    """Collect per-burst keyframe jpg paths from a (possibly grouped) activity.

    A grouped activity carries ``_sourceActivities`` with each burst's clip
    path; the corresponding ``*_activity.jpg`` lives alongside the clip.
    Falls back to the single ``activityImage`` field when no source-clip
    metadata is present.

    Returned in time order.
    """
    sources = activity.get("_sourceActivities") or []
    paths: List[Path] = []
    seen: set = set()
    for entry in sources:
        clip = entry.get("clip")
        if not clip:
            continue
        jpg = Path(str(clip).replace("_clip.mp4", "_activity.jpg"))
        if jpg.is_file() and str(jpg) not in seen:
            paths.append(jpg)
            seen.add(str(jpg))
    if not paths:
        single = activity.get("activityImage")
        if single:
            p = Path(str(single))
            if p.is_file():
                paths.append(p)
    return paths


def _supplement_keyframes_from_clip(
    activity: Dict[str, Any],
    existing: List[Path],
    target_n: int,
) -> List[Path]:
    """Sample additional frames from ``activityClip`` to reach ``target_n``.

    Single-burst activities (``_sourceActivities`` absent) yield exactly
    one keyframe from :func:`_resolve_keyframes`. The setting
    ``vlm_strip_target_frames`` (default 5) promises temporal evidence for
    those cases, so when fewer keyframes are present than ``target_n``
    we open the activity's ``activityClip`` and decode a small set of
    evenly-spaced frames. Frames are written next to the existing keyframe
    with a ``_supp{idx}.jpg`` suffix.

    Returns the (possibly-extended) path list in time order. Failures
    here are non-fatal: if cv2 can't open the clip we just return
    ``existing`` unchanged so the verifier proceeds with whatever it has.
    """
    if target_n <= 1 or len(existing) >= target_n:
        return existing
    clip_path = activity.get("activityClip") or ""
    if not clip_path:
        return existing
    clip = Path(str(clip_path))
    if not clip.is_file():
        return existing

    # Anchor name for sibling frames (use first existing keyframe so the
    # supplements colocate with the burst evidence, not the clip).
    anchor = existing[0] if existing else clip.with_suffix(".jpg")
    parent = anchor.parent
    stem = anchor.stem.replace("_activity", "")

    try:
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            return existing
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count <= 0:
                return existing
            need = target_n - len(existing)
            # Evenly distribute the supplementary frames across the clip,
            # avoiding the exact endpoints (which often differ from the
            # already-saved activity keyframe by only a frame or two).
            offsets: List[int] = []
            for i in range(need):
                # i runs 0..need-1; map to (i+1)/(need+1) of the clip
                pos = int(round((i + 1) * frame_count / (need + 1)))
                pos = max(0, min(frame_count - 1, pos))
                offsets.append(pos)
            # De-dup while preserving order.
            seen_ofs: set = set()
            offsets = [o for o in offsets if not (o in seen_ofs or seen_ofs.add(o))]

            new_paths: List[Path] = []
            for idx, ofs in enumerate(offsets):
                cap.set(cv2.CAP_PROP_POS_FRAMES, ofs)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                out = parent / f"{stem}_supp{idx}.jpg"
                # If we already wrote this on a prior call, reuse it.
                if not out.is_file():
                    cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if out.is_file():
                    new_paths.append(out)
            if not new_paths:
                return existing
            # Place originals first (they have Pipeline-1 bbox overlays the
            # gate relies on); supplements after as additional context.
            return existing + new_paths
        finally:
            cap.release()
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "[vlm] clip-supplement failed for %s: %s", clip_path, e,
        )
        return existing


def _count_bboxes_in_keyframes(
    jpg_paths: List[Path],
    min_person_area: int = 1000,
    min_object_area: int = 300,
) -> Dict[str, int]:
    """Count Pipeline-1 overlay bboxes by colour across keyframes.

    Pipeline-1 renders each detected person bbox in bright GREEN (HSV hue
    ~60) and each object bbox (book / cup / bottle / bag / phone) in
    ORANGE/YELLOW (HSV hue 15-35) onto the saved ``*_activity.jpg``.

    IMPORTANT: this helper is only meaningful for paths returned by
    :func:`_resolve_keyframes` (the per-burst saved keyframes which
    carry the rendered overlay). Supplementary frames sampled from
    ``activityClip`` via :func:`_supplement_keyframes_from_clip` are
    raw decoded frames with NO bbox overlay; passing them in will
    return zero counts and defeat the gate logic that depends on
    rendering being active.

    Returns:
        ``{"with_person": int, "with_object": int, "with_any_bbox": int, "total": int}``.
        ``with_person`` and ``with_object`` are independent counts. The
        rule of thumb for callers: only enforce gate decisions when
        ``with_any_bbox > 0`` (i.e. overlay rendering is confirmed
        active), otherwise fall through to the VLM.
    """
    with_person = 0
    with_object = 0
    with_any = 0
    for p in jpg_paths:
        try:
            img = cv2.imread(str(p))
            if img is None:
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Pipeline-1 colour palette as observed on the saved keyframes:
            #   GREEN   ~ hue 60   → person bbox
            #   YELLOW/ORANGE ~ hue 15-35 → object bbox (book, cup, etc.)
            #   MAGENTA ~ hue 140-170 → bag/backpack bbox
            #   RED     ~ hue 0 / 170-180 → keypoint markers + sometimes labels
            # We use green strictly for the person count, orange for the
            # object count, and treat green|orange|magenta|red collectively
            # as the "rendering is active" signal so a frame with only a
            # bag bbox (e.g. an empty-cab clip with a backpack on the seat)
            # still confirms the overlay is on.
            green = cv2.inRange(hsv, (45, 150, 150), (75, 255, 255))
            orange = cv2.inRange(hsv, (15, 150, 150), (35, 255, 255))
            magenta = cv2.inRange(hsv, (140, 100, 100), (170, 255, 255))
            red_low = cv2.inRange(hsv, (0, 150, 100), (10, 255, 255))
            red_high = cv2.inRange(hsv, (170, 150, 100), (180, 255, 255))

            person_found = False
            for c in cv2.findContours(
                green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]:
                if cv2.contourArea(c) >= min_person_area:
                    _, _, w, h = cv2.boundingRect(c)
                    # A person bbox is at least ~40x40; reject thin
                    # skeleton-line fragments that survive area threshold.
                    if w >= 40 and h >= 40:
                        person_found = True
                        break

            object_found = False
            for c in cv2.findContours(
                orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]:
                if cv2.contourArea(c) >= min_object_area:
                    _, _, w, h = cv2.boundingRect(c)
                    if w >= 30 and h >= 30:
                        object_found = True
                        break

            # Render-active probe: any sufficiently-large saturated cluster
            # in green / orange / magenta / red. The threshold is small
            # (200 px) because we only need *some* evidence of overlay
            # rendering — a bag bbox or a few keypoint markers count.
            render_active = person_found or object_found
            if not render_active:
                misc_mask = cv2.bitwise_or(
                    magenta, cv2.bitwise_or(red_low, red_high),
                )
                for c in cv2.findContours(
                    misc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )[0]:
                    if cv2.contourArea(c) >= 200:
                        render_active = True
                        break

            if person_found:
                with_person += 1
            if object_found:
                with_object += 1
            if render_active:
                with_any += 1
        except Exception:  # noqa: BLE001
            # Any per-frame failure is non-fatal — fall back to
            # treating the frame as "unknown", which is conservatively
            # the safer choice (don't drop on shaky signal).
            continue
    return {
        "with_person": with_person,
        "with_object": with_object,
        "with_any_bbox": with_any,
        "total": len(jpg_paths),
    }


def _stitch_keyframes(
    jpg_paths: List[Path],
    max_strip_width: int = 1500,
    crop_to_roi: bool = True,
) -> Optional[bytes]:
    """Stitch up to 5 keyframes left-to-right into a single labelled JPEG.

    When ``crop_to_roi=True`` (default), each frame is cropped to the
    Pipeline-1-rendered person + object bbox union before stitching, so
    the VLM sees only the hand + book/desk region at native pixel density.
    This is the right mode for fine-grained hand-on-object verdicts
    (writing, eating_drinking, packing_bags, cell_phone).

    When ``crop_to_roi=False``, frames are stitched at full resolution.
    This is the right mode for activities where the WHOLE cabin matters:
    sleep posture, head-pose for mind_diversion, "is the cabin empty"
    checks (no_person_detected), or counting people (group_detected).

    The combined strip is capped at ``max_strip_width`` either way to stay
    within vLLM's max-model-len budget. Each frame gets a ``FRAME N``
    tag in its top-left corner so the VLM can reference it via
    ``evidence_frame``. Single-frame input returns the original bytes
    unchanged (no relabel, no crop) regardless of mode.
    """
    if not jpg_paths:
        return None
    if len(jpg_paths) == 1:
        try:
            return jpg_paths[0].read_bytes()
        except OSError:
            return None

    cap = jpg_paths[:5]
    frames: List[np.ndarray] = []
    for idx, p in enumerate(cap):
        img = cv2.imread(str(p))
        if img is None:
            logger.warning("[vlm] cv2.imread returned None for %s", p)
            continue
        if crop_to_roi:
            img = _crop_to_roi(img)
        cv2.rectangle(img, (0, 0), (110, 30), (0, 0, 0), -1)
        cv2.putText(img, f"FRAME {idx + 1}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        frames.append(img)
    if not frames:
        return None

    # Pad each frame to a common height so np.hstack works.
    target_h = max(img.shape[0] for img in frames)
    padded: List[np.ndarray] = []
    for img in frames:
        h = img.shape[0]
        if h < target_h:
            img = cv2.copyMakeBorder(img, 0, target_h - h, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded.append(img)

    strip = np.hstack(padded)

    # If the combined width exceeds the budget, downscale uniformly.
    if strip.shape[1] > max_strip_width:
        scale = max_strip_width / float(strip.shape[1])
        new_h = max(1, int(strip.shape[0] * scale))
        strip = cv2.resize(strip, (max_strip_width, new_h),
                           interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return None
    return buf.tobytes()

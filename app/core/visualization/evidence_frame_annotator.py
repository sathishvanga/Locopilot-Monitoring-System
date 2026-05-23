"""Evidence-frame annotator extracted from the monolith.

Re-runs object + pose inference on a saved evidence frame so the resulting
JPEG visually explains why an activity fired — reviewer sees the person
bbox, book/cell-phone/cup bbox, and the wrist/shoulder/nose keypoints the
detector was looking at.

Behavior is byte-identical to
``LocopilotActivityMonitor._annotate_evidence_frame``: log strings, drawing
parameters, color tables, and skeleton edges are all preserved verbatim.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import cv2


_DEFAULT_LOGGER = logging.getLogger(__name__)


def annotate_evidence_frame(
    frame: Any,
    *,
    activity_name: str,
    frame_number: int,
    object_detector: Any,
    yolo_pose: Any,
    get_keypoint: Callable[[Any, str], Any],
    logger: Optional[logging.Logger] = None,
) -> Any:
    """Overlay object bboxes + pose skeleton on an evidence frame.

    Re-runs object + pose inference on the extracted frame so the saved
    *_activity.jpg visually explains why the activity fired — the reviewer
    sees exactly which person bbox, book/cell-phone/cup bbox, and which
    wrist/shoulder keypoints the detector was looking at.

    Safe no-op if inference fails — returns the original frame unchanged.
    """
    try:
        img = frame.copy()

        # Object detections (full-frame, no pose-guided ROI — we just want
        # a display, not re-triggering the activity logic)
        det = object_detector.detect_objects(img, pose_landmarks=None, use_pose_guided=False)

        color_by_class = {
            'person': (0, 255, 0),        # green
            'book': (0, 215, 255),        # amber
            'cell_phone': (0, 0, 255),    # red
            'cup': (255, 200, 0),         # cyan
            'bottle': (255, 200, 0),
            'backpack': (255, 0, 255),    # magenta
            'handbag': (255, 0, 255),
            'suitcase': (255, 0, 255),
            'radio_handset': (128, 64, 255),
        }
        for cls_name, boxes in det.items():
            if cls_name in ('rois', 'num_detections'):
                continue
            color = color_by_class.get(cls_name, (200, 200, 200))
            for b in boxes or []:
                try:
                    x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                except Exception:
                    continue
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = cls_name
                if len(b) > 4 and isinstance(b[4], (int, float)):
                    label = f"{cls_name} {float(b[4]):.2f}"
                cv2.putText(img, label, (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Pose skeleton per person
        try:
            persons = yolo_pose.process(img) if yolo_pose else {}
        except Exception:
            persons = {}
        SKELETON = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
            ('nose', 'left_shoulder'), ('nose', 'right_shoulder'),
        ]
        h, w = img.shape[:2]
        for pidx, pdata in (persons or {}).items():
            kp = pdata.get('keypoints')
            if kp is None:
                continue
            def _pt(name):
                try:
                    k = get_keypoint(kp, name)
                    if k.visibility < 0.3:
                        return None
                    return (int(k.x * w), int(k.y * h))
                except Exception:
                    return None
            for a, b in SKELETON:
                pa, pb = _pt(a), _pt(b)
                if pa and pb:
                    cv2.line(img, pa, pb, (255, 255, 255), 2, cv2.LINE_AA)
            # highlight wrists + shoulders + nose
            for kname, color, r in (
                ('left_wrist', (0, 0, 255), 7),
                ('right_wrist', (0, 0, 255), 7),
                ('left_shoulder', (0, 255, 255), 5),
                ('right_shoulder', (0, 255, 255), 5),
                ('nose', (255, 0, 255), 5),
            ):
                pt = _pt(kname)
                if pt:
                    cv2.circle(img, pt, r, color, -1, cv2.LINE_AA)

        # Header banner so the viewer knows what this frame is
        banner = f"{activity_name.upper()}  f{frame_number}"
        cv2.rectangle(img, (0, 0), (max(420, 10 * len(banner) + 20), 28), (0, 0, 0), -1)
        cv2.putText(img, banner, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return img
    except Exception as e:
        if logger:
            logger.debug(f"[annotate_evidence_frame] failed, saving raw: {e}")
        return frame

"""YOLO pose batch inference helper extracted from ``locopilot_monitor.py`` (T6).

Lifts ``LocopilotActivityMonitor.detect_poses_batch`` (lines 1172-1234)
into a pure module-level function. ``self.yolo_pose`` and
``self.yolo_device`` were previously read off the monitor instance; they
are now passed in as ``yolo_pose_adapter`` and ``device`` keyword
arguments respectively. The body is otherwise a byte-identical lift,
including the ``[GPU BATCH]`` log strings and the lazy import of
``YoloPoseLandmarks`` / ``PersonKeypoints`` from
``app.services.yolo_pose_adapter`` to match today's import behavior.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional


def detect_poses_batch(
    yolo_pose_adapter,
    frames: List[Any],
    *,
    batch_size: int = 8,
    conf_threshold: Optional[float] = None,
    device: str = 'cpu',
    logger: Optional[logging.Logger] = None,
) -> List[Dict[int, Dict[str, Any]]]:
    """Run YOLO pose detection on multiple frames in a single batch.

    This maximizes GPU utilization by processing multiple frames at once.

    Args:
        yolo_pose_adapter: The pose adapter instance (the monitor's
            ``self.yolo_pose``). Must expose ``.model`` and
            ``.conf_threshold`` exactly like the existing adapter.
        frames: List of BGR frames (numpy arrays)
        batch_size: Maximum batch size for inference (default 8)
        conf_threshold: Optional confidence threshold override (default: use model's conf_threshold)
        device: Device string passed to the YOLO model (the monitor's
            ``self.yolo_device``).
        logger: Logger for ``[GPU BATCH]`` debug/error lines. Defaults to a
            module-level logger; callers should pass ``self.logger``.

    Returns:
        List of pose result dictionaries, one per frame.
        Format matches self.yolo_pose.process() output:
        {person_idx: {'bbox': [...], 'bbox_confidence': float, 'keypoints': YoloPoseLandmarks}}
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Import at method level (not inside loop)
    from app.services.yolo_pose_adapter import YoloPoseLandmarks, PersonKeypoints

    if not frames:
        return []

    effective_conf = conf_threshold if conf_threshold is not None else yolo_pose_adapter.conf_threshold
    logger.debug(f"[GPU BATCH] detect_poses_batch: {len(frames)} frames, batch_size={batch_size}, conf={effective_conf}")
    all_poses: List[Dict[int, Dict[str, Any]]] = []

    # Process frames in batches
    for batch_start in range(0, len(frames), batch_size):
        batch_frames = frames[batch_start:batch_start + batch_size]

        try:
            # Run batch inference on pose model with device parameter
            batch_results = yolo_pose_adapter.model(
                batch_frames,
                verbose=False,
                conf=effective_conf,
                device=device
            )
        except Exception as e:
            logger.error(f"[GPU BATCH] Pose detection failed for batch starting at {batch_start}: {e}")
            # Fallback: return empty poses for this batch
            for _ in batch_frames:
                all_poses.append({})
            continue

        # Process results for each frame in batch
        for frame_idx, (frame, results) in enumerate(zip(batch_frames, batch_results)):
            persons = {}

            if results.keypoints is not None and results.boxes is not None:
                for idx in range(len(results.boxes)):
                    box = results.boxes[idx]
                    person_keypoints = PersonKeypoints(results.keypoints, idx)

                    persons[idx] = {
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),
                        'bbox_confidence': float(box.conf[0]),
                        'keypoints': YoloPoseLandmarks(person_keypoints, frame.shape)
                    }

            all_poses.append(persons)

    logger.debug(f"[GPU BATCH] detect_poses_batch complete: {len(all_poses)} results")
    return all_poses


__all__ = ["detect_poses_batch"]

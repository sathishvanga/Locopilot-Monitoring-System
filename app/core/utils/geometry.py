"""Geometry utilities for bounding box operations.

This module provides utility functions for geometric calculations on bounding boxes,
including Intersection over Union (IoU), overlap detection, and Non-Maximum Suppression
(NMS) for deduplication.
"""
from typing import List, Tuple, Optional


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    IoU is a metric used to measure the overlap between two bounding boxes.
    It is calculated as the area of intersection divided by the area of union.

    Args:
        bbox1: First bounding box in format [x1, y1, x2, y2] where
               (x1, y1) is top-left and (x2, y2) is bottom-right.
        bbox2: Second bounding box in format [x1, y1, x2, y2] where
               (x1, y1) is top-left and (x2, y2) is bottom-right.

    Returns:
        IoU value between 0.0 and 1.0, where 0.0 means no overlap
        and 1.0 means complete overlap.

    Example:
        >>> calculate_iou([0, 0, 10, 10], [5, 5, 15, 15])
        0.14285714285714285
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - intersection_area

    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area
    return iou


def _compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.

    This is a convenience wrapper around calculate_iou for backward compatibility.
    Helper for temporal role tracking to match persons across frames.

    Args:
        box1: First bounding box in format [x1, y1, x2, y2].
        box2: Second bounding box in format [x1, y1, x2, y2].

    Returns:
        IoU value between 0.0 and 1.0.

    Note:
        This function delegates to calculate_iou to avoid code duplication.
        Consider using calculate_iou directly for new code.
    """
    return calculate_iou(box1, box2)


def bbox_overlap_with_margin(
    obj_bbox: List[int],
    person_bbox: List[int],
    margin: int
) -> bool:
    """Check if object bbox overlaps with person bbox with an expanded margin.

    This function expands the person bounding box by the specified margin
    in all directions and then checks if the object bbox overlaps with
    the expanded region.

    Args:
        obj_bbox: Object bounding box in format [x1, y1, x2, y2] where
                  (x1, y1) is top-left and (x2, y2) is bottom-right.
        person_bbox: Person bounding box in format [x1, y1, x2, y2] where
                     (x1, y1) is top-left and (x2, y2) is bottom-right.
        margin: Number of pixels to expand the person bbox in all directions.
                A larger margin makes overlap detection more lenient.

    Returns:
        True if the object bbox overlaps with the expanded person region,
        False otherwise.

    Example:
        >>> bbox_overlap_with_margin([100, 100, 120, 120], [50, 50, 90, 90], margin=20)
        True
    """
    ox1, oy1, ox2, oy2 = obj_bbox
    px1, py1, px2, py2 = person_bbox

    # Expand person bbox with margin
    px1_expanded = px1 - margin
    py1_expanded = py1 - margin
    px2_expanded = px2 + margin
    py2_expanded = py2 + margin

    # Check overlap - no overlap if boxes are separated horizontally or vertically
    if ox2 < px1_expanded or ox1 > px2_expanded:
        return False
    if oy2 < py1_expanded or oy1 > py2_expanded:
        return False

    return True


def deduplicate_person_boxes(
    person_boxes: List[List[int]],
    iou_threshold: float = 0.3
) -> List[List[int]]:
    """De-duplicate overlapping person bounding boxes using Non-Maximum Suppression.

    This function removes duplicate detections of the same person by keeping
    only boxes that don't significantly overlap with each other. It uses a
    greedy NMS algorithm that prioritizes larger boxes (typically more confident
    detections) and removes smaller overlapping boxes.

    Args:
        person_boxes: List of person bounding boxes, where each box is
                      [x1, y1, x2, y2] format with (x1, y1) as top-left
                      and (x2, y2) as bottom-right.
        iou_threshold: IoU threshold for considering boxes as duplicates.
                       Boxes with IoU >= threshold are considered duplicates.
                       Default is 0.3 (30% overlap).

    Returns:
        List of de-duplicated person boxes in the same format as input.
        The returned boxes are sorted by area (largest first).

    Example:
        >>> boxes = [[0, 0, 100, 100], [5, 5, 105, 105], [200, 200, 300, 300]]
        >>> deduplicate_person_boxes(boxes, iou_threshold=0.3)
        [[0, 0, 100, 100], [200, 200, 300, 300]]

    Note:
        The algorithm processes boxes from largest to smallest area,
        keeping each box only if it doesn't significantly overlap with
        any previously kept box.
    """
    if len(person_boxes) == 0:
        return []

    # Convert to list of lists if numpy arrays
    boxes = [list(box) if hasattr(box, 'tolist') else list(box) for box in person_boxes]

    # Calculate areas for each box
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]

    # Sort by area (larger boxes first - usually more confident detections)
    sorted_indices = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)

    keep_boxes: List[List[int]] = []

    while sorted_indices:
        # Take the first box (largest remaining)
        idx = sorted_indices[0]
        keep_boxes.append(boxes[idx])
        sorted_indices.pop(0)

        # Remove boxes that significantly overlap with this box
        remaining_indices = []
        for other_idx in sorted_indices:
            iou = calculate_iou(boxes[idx], boxes[other_idx])
            if iou < iou_threshold:
                # Keep this box (not a duplicate)
                remaining_indices.append(other_idx)
            # else: discard as duplicate

        sorted_indices = remaining_indices

    return keep_boxes

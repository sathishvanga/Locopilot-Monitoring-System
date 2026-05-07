"""Hand-gesture coordination check.

Pure helper extracted from locopilot_monitor.py
(``_check_hand_gesture_coordination``, lines 2349-2404 in T3 of the refactor
plan). Reads ``recent_person_activities`` but never mutates it.
"""

from __future__ import annotations

from typing import Dict, Tuple


def check_hand_gesture_coordination(
    *,
    lp_detected: bool,
    alp_detected: bool,
    current_time: float,
    recent_person_activities: Dict,
    hand_gesture_coordination_window: float,
) -> Tuple[bool, bool]:
    """
    Check for hand gesture coordination failures with temporal window support.

    Prevents false positives when both people raise hands within a time window
    (collaborative discussion) but not in the exact same frame.

    Args:
        lp_detected: LP hand gesture detected in current frame
        alp_detected: ALP hand gesture detected in current frame
        current_time: Current timestamp in seconds
        recent_person_activities: dict mapping person_idx -> activities dict
            (live ref from the monitor; never mutated here).
        hand_gesture_coordination_window: window in seconds within which two
            raises count as coordinated.

    Returns:
        tuple: (lp_not_coordinating, alp_not_coordinating)
            - lp_not_coordinating: True if ALP raised hand but LP failed to coordinate
            - alp_not_coordinating: True if LP raised hand but ALP failed to coordinate
    """
    # Get last hand raise times from recent activities
    lp_last_raise_time = None
    alp_last_raise_time = None

    for person_idx, activities in recent_person_activities.items():
        if 'lp_hand_raise' in activities:
            t = activities['lp_hand_raise']
            if lp_last_raise_time is None or t > lp_last_raise_time:
                lp_last_raise_time = t
        if 'alp_hand_raise' in activities:
            t = activities['alp_hand_raise']
            if alp_last_raise_time is None or t > alp_last_raise_time:
                alp_last_raise_time = t

    # Helper: Check if both raised hands within coordination window
    def both_within_window(lp_time, alp_time):
        if lp_time is None or alp_time is None:
            return False
        lp_recent = (current_time - lp_time) <= hand_gesture_coordination_window
        alp_recent = (current_time - alp_time) <= hand_gesture_coordination_window
        return lp_recent and alp_recent

    # Check coordination with temporal window logic
    lp_not_coordinating = False
    alp_not_coordinating = False

    if alp_detected and not lp_detected:
        # ALP raised hand, LP didn't in current frame
        # Check if LP raised recently (within window)
        if not both_within_window(lp_last_raise_time, alp_last_raise_time):
            lp_not_coordinating = True  # True coordination failure

    if lp_detected and not alp_detected:
        # LP raised hand, ALP didn't in current frame
        # Check if ALP raised recently (within window)
        if not both_within_window(lp_last_raise_time, alp_last_raise_time):
            alp_not_coordinating = True  # True coordination failure

    return lp_not_coordinating, alp_not_coordinating

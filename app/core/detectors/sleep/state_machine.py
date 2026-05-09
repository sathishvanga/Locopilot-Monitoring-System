"""Temporal state machine for sleep detection.

States: ALERT, LOOKING_DOWN_WORKING, DROWSY, MICROSLEEP, SLEEPING

Verbatim move of ``_update_sleep_state_machine`` from the original
``sleep_detector.py`` with ``self`` rebound to ``detector``.
"""

from typing import Any, Dict


def update_sleep_state_machine(
    detector: Any,
    tracking: Dict[str, Any],
    timestamp_sec: float,
    is_head_down: bool,
    is_sustained_low_eyes: bool,
    is_minimal_movement: bool,
    head_bob_detected: bool,
    avg_wrist_velocity: float
) -> str:
    """Temporal state machine for sleep detection.

    States: ALERT, LOOKING_DOWN_WORKING, DROWSY, MICROSLEEP, SLEEPING

    Args:
        tracking: Per-person sleep tracking dict
        timestamp_sec: Current timestamp in seconds
        is_head_down: Whether head is tilted down
        is_sustained_low_eyes: Whether eyes have been low-visibility for sustained frames
        is_minimal_movement: Whether body movement is minimal
        head_bob_detected: Whether a head bob pattern was detected
        avg_wrist_velocity: Average wrist velocity from recent history

    Returns:
        str: Current sleep state after transition
    """
    current_state = tracking.get('sleep_state', 'ALERT')
    has_hand_activity = avg_wrist_velocity > detector.SLEEP_STATE_HAND_ACTIVITY_THRESHOLD

    # Calculate time in current state
    if tracking.get('state_enter_time') is not None:
        time_in_state = timestamp_sec - tracking['state_enter_time']
    else:
        time_in_state = 0.0
        tracking['state_enter_time'] = timestamp_sec

    new_state = current_state

    if current_state == 'ALERT':
        if is_head_down and has_hand_activity:
            new_state = 'LOOKING_DOWN_WORKING'
        elif ((is_sustained_low_eyes and is_minimal_movement and not has_hand_activity)
              or head_bob_detected):
            new_state = 'DROWSY'

    elif current_state == 'LOOKING_DOWN_WORKING':
        if not is_head_down or not has_hand_activity:
            new_state = 'ALERT'

    elif current_state == 'DROWSY':
        if has_hand_activity or (not is_sustained_low_eyes and not head_bob_detected):
            new_state = 'ALERT'
        elif time_in_state >= detector.SLEEP_DROWSY_TO_MICROSLEEP_SEC:
            new_state = 'MICROSLEEP'

    elif current_state == 'MICROSLEEP':
        if has_hand_activity or (not is_sustained_low_eyes and not head_bob_detected):
            new_state = 'ALERT'
        elif time_in_state >= detector.SLEEP_MICROSLEEP_TO_SLEEP_SEC:
            new_state = 'SLEEPING'

    elif current_state == 'SLEEPING':
        if has_hand_activity:
            new_state = 'ALERT'
        elif not is_sustained_low_eyes and not head_bob_detected:
            # Duration dropped below sleep threshold but still showing
            # drowsy/microsleep signals -- transition back to MICROSLEEP
            # instead of jumping straight to ALERT.
            new_state = 'MICROSLEEP'

    # Handle state transition
    if new_state != current_state:
        tracking['state_history'].append((current_state, new_state, timestamp_sec))
        tracking['sleep_state'] = new_state
        tracking['state_enter_time'] = timestamp_sec
        detector.logger.debug(
            f"[Sleep State Machine] {current_state} -> {new_state} "
            f"(time_in_prev={time_in_state:.1f}s, hand_activity={has_hand_activity}, "
            f"head_bob={head_bob_detected})"
        )

    return tracking['sleep_state']

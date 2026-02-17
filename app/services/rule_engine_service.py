"""
Rule engine service - Applies motion-based rules to activity violations

This service implements the rule engine that dynamically changes violation
detection based on whether the train is running or stopped.
"""

import logging
import threading
from typing import Dict, Optional

from ..models.trip_models import (
    TrainMotionState,
    TrainMotionContext,
    ViolationRuleResult
)
from ..models.activity_models import ActivityTypeEnum
from ..utils.config import get_settings

logger = logging.getLogger(__name__)


class RuleEngineService:
    """
    Service for applying motion-based rules to activity violations

    Rule definitions:
    - When RUNNING: All activities are violations (normal detection)
    - When STOPPED: writing and packing_bags are ALLOWED, no_person is VIOLATION
    - Pre-arrival window: Check for ALP standing requirement

    Features:
    - Evaluates individual activities against motion state
    - Batch evaluation for all activities in a frame
    - Pre-arrival ALP alertness checking
    - Detailed logging of exemptions
    """

    # Activity type to name mapping
    ACTIVITY_NAMES = {
        ActivityTypeEnum.CELL_PHONE: 'cell_phone',
        ActivityTypeEnum.MICROSLEEP: 'microsleep',
        ActivityTypeEnum.SLEEP: 'sleep',
        ActivityTypeEnum.WRITING: 'writing',
        ActivityTypeEnum.PACKING_BAGS: 'packing_bags',
        ActivityTypeEnum.GROUP_DETECTED: 'group_detected',
        ActivityTypeEnum.LP_NOT_EXCHANGING_HAND_GESTURE: 'lp_hand_gesture',
        ActivityTypeEnum.ALP_NOT_EXCHANGING_HAND_GESTURE: 'alp_hand_gesture',
        ActivityTypeEnum.MIND_DIVERSION: 'mind_diversion',
        ActivityTypeEnum.NO_PERSON_DETECTED: 'no_person_detected',
        ActivityTypeEnum.EATING_DRINKING: 'eating_drinking',
    }

    # Activities ALLOWED when train is STOPPED at station
    ALLOWED_WHEN_STOPPED = {
        'writing',          # Writing logbook is allowed at stations
        'packing_bags',     # Packing bags is allowed at stations
        'mind_diversion',   # Brief attention diversion allowed at stations
        'cell_phone',       # Cell phone usage allowed at stations
        'lp_hand_gesture',  # Hand signaling not required at stations
        'alp_hand_gesture',  # Hand signaling not required at stations
        'eating_drinking'   # Eating/drinking allowed at stations
    }

    # Activities that are always violations regardless of motion state
    ALWAYS_VIOLATION = {
        'microsleep',       # Sleeping is never allowed
        'sleep',            # Sleeping is never allowed
        'group_detected'    # Unauthorized persons never allowed
    }

    def __init__(self):
        """Initialize the rule engine service"""
        self.settings = get_settings()
        self.enabled = self.settings.train_motion_rules_enabled

        logger.info(
            f"RuleEngineService initialized - "
            f"enabled: {self.enabled}"
        )

    def evaluate_activity(
        self,
        activity_name: str,
        detected: bool,
        motion_context: Optional[TrainMotionContext]
    ) -> ViolationRuleResult:
        """
        Evaluate whether a detected activity is a violation

        Args:
            activity_name: Name of the activity (e.g., 'writing', 'cell_phone')
            detected: Whether the activity was detected
            motion_context: Current train motion context

        Returns:
            ViolationRuleResult with violation decision and reason
        """
        # Get activity type code
        activity_type = self._get_activity_type(activity_name)

        motion_state_str = motion_context.motion_state.name if motion_context else "None"
        logger.debug(
            f"[RULE-ENGINE] evaluate_activity() - "
            f"activity: {activity_name}, detected: {detected}, motion: {motion_state_str}"
        )

        # If not detected, it's not a violation
        if not detected:
            logger.debug(f"[RULE-ENGINE] {activity_name}: NOT_DETECTED -> not a violation")
            return ViolationRuleResult(
                activity_type=activity_type,
                activity_name=activity_name,
                is_violation=False,
                is_exempted=False,
                motion_state=motion_context.motion_state if motion_context else TrainMotionState.UNKNOWN,
                rule_applied="NOT_DETECTED",
                reason="Activity not detected",
                was_detected=False
            )

        # If rule engine disabled, all detected activities are violations
        if not self.enabled or motion_context is None:
            logger.info(
                f"[RULE-ENGINE] {activity_name}: RULES_DISABLED -> VIOLATION "
                f"(enabled: {self.enabled}, context: {'present' if motion_context else 'None'})"
            )
            return ViolationRuleResult(
                activity_type=activity_type,
                activity_name=activity_name,
                is_violation=True,
                is_exempted=False,
                motion_state=TrainMotionState.UNKNOWN,
                rule_applied="RULES_DISABLED",
                reason="Rule engine disabled, treating as violation",
                was_detected=True
            )

        # Get motion state
        motion_state = motion_context.motion_state

        # Unknown state - treat as RUNNING (safe default)
        if motion_state == TrainMotionState.UNKNOWN:
            logger.info(
                f"[RULE-ENGINE] {activity_name}: UNKNOWN motion state -> VIOLATION (safe default)"
            )
            return ViolationRuleResult(
                activity_type=activity_type,
                activity_name=activity_name,
                is_violation=True,
                is_exempted=False,
                motion_state=motion_state,
                rule_applied="UNKNOWN_STATE_RUNNING_DEFAULT",
                reason="Motion state unknown, treating as running (safe default)",
                was_detected=True
            )

        # RUNNING state - all activities are violations
        if motion_state == TrainMotionState.RUNNING:
            logger.info(
                f"[RULE-ENGINE] {activity_name}: train RUNNING -> VIOLATION"
            )
            return ViolationRuleResult(
                activity_type=activity_type,
                activity_name=activity_name,
                is_violation=True,
                is_exempted=False,
                motion_state=motion_state,
                rule_applied="RUNNING_ALL_VIOLATIONS",
                reason=f"Train is running - {activity_name} is a violation",
                was_detected=True
            )

        # STOPPED state - check exemptions
        if motion_state == TrainMotionState.STOPPED:
            station_info = ""
            if motion_context.current_station:
                station_info = f" at {motion_context.current_station.station_name}"

            # Check if activity is allowed when stopped
            if activity_name in self.ALLOWED_WHEN_STOPPED:
                logger.info(
                    f"[RULE-ENGINE] ✅ {activity_name.upper()}: EXEMPTED - "
                    f"train STOPPED{station_info} (allowed activities: {self.ALLOWED_WHEN_STOPPED})"
                )

                return ViolationRuleResult(
                    activity_type=activity_type,
                    activity_name=activity_name,
                    is_violation=False,
                    is_exempted=True,
                    motion_state=motion_state,
                    rule_applied="STOPPED_EXEMPTION",
                    reason=f"{activity_name} is allowed when train is stopped{station_info}",
                    was_detected=True
                )

            # Special case: no_person_detected is ALWAYS a violation when stopped
            # (at least 1 crew required in cabin)
            if activity_name == 'no_person_detected':
                logger.info(
                    f"[RULE-ENGINE] {activity_name}: train STOPPED but VIOLATION "
                    f"(crew required in cabin)"
                )
                return ViolationRuleResult(
                    activity_type=activity_type,
                    activity_name=activity_name,
                    is_violation=True,
                    is_exempted=False,
                    motion_state=motion_state,
                    rule_applied="STOPPED_CREW_REQUIRED",
                    reason="At least 1 crew member required when train is stopped",
                    was_detected=True
                )

            # All other activities are still violations when stopped
            logger.info(
                f"[RULE-ENGINE] {activity_name}: train STOPPED but VIOLATION "
                f"(not in allowed list: {self.ALLOWED_WHEN_STOPPED})"
            )
            return ViolationRuleResult(
                activity_type=activity_type,
                activity_name=activity_name,
                is_violation=True,
                is_exempted=False,
                motion_state=motion_state,
                rule_applied="STOPPED_VIOLATION",
                reason=f"{activity_name} is still a violation when stopped",
                was_detected=True
            )

        # Fallback (should not reach here)
        logger.warning(
            f"[RULE-ENGINE] {activity_name}: FALLBACK -> VIOLATION (unexpected motion state: {motion_state})"
        )
        return ViolationRuleResult(
            activity_type=activity_type,
            activity_name=activity_name,
            is_violation=True,
            is_exempted=False,
            motion_state=motion_state,
            rule_applied="FALLBACK_VIOLATION",
            reason="Fallback rule - treating as violation",
            was_detected=True
        )

    def evaluate_all_activities(
        self,
        activities_map: Dict[str, bool],
        motion_context: Optional[TrainMotionContext]
    ) -> Dict[str, ViolationRuleResult]:
        """
        Evaluate all activities in a frame against motion rules

        Args:
            activities_map: Dictionary mapping activity names to detection status
            motion_context: Current train motion context

        Returns:
            Dictionary mapping activity names to ViolationRuleResults
        """
        motion_state_str = motion_context.motion_state.name if motion_context else "None"
        detected_activities = [k for k, v in activities_map.items() if v]

        logger.debug(
            f"[RULE-ENGINE] evaluate_all_activities() - "
            f"motion: {motion_state_str}, "
            f"detected: {detected_activities}"
        )

        results = {}
        violations = []
        exemptions = []

        for activity_name, detected in activities_map.items():
            result = self.evaluate_activity(activity_name, detected, motion_context)
            results[activity_name] = result

            if detected:
                if result.is_violation:
                    violations.append(activity_name)
                elif result.is_exempted:
                    exemptions.append(activity_name)

        if detected_activities:
            logger.info(
                f"[RULE-ENGINE] Batch evaluation complete - "
                f"motion: {motion_state_str}, "
                f"violations: {violations}, "
                f"exemptions: {exemptions}"
            )

        return results

    def should_check_alp_alertness(
        self,
        motion_context: Optional[TrainMotionContext]
    ) -> bool:
        """
        Check if ALP alertness should be checked (pre-arrival window)

        Args:
            motion_context: Current train motion context

        Returns:
            True if in pre-arrival window and should check ALP standing
        """
        if not self.enabled or motion_context is None:
            return False

        return motion_context.is_pre_arrival_window

    def get_filtered_activities_map(
        self,
        activities_map: Dict[str, bool],
        motion_context: Optional[TrainMotionContext]
    ) -> Dict[str, bool]:
        """
        Get filtered activities map with exemptions applied

        This returns a modified activities_map where exempted activities
        are set to False (not detected), allowing easy integration with
        existing activity tracking code.

        Args:
            activities_map: Original activities detection map
            motion_context: Current train motion context

        Returns:
            Filtered activities map with exemptions applied
        """
        if not self.enabled or motion_context is None:
            logger.debug(
                f"[RULE-ENGINE] get_filtered_activities_map() - "
                f"bypassing (enabled: {self.enabled}, context: {'present' if motion_context else 'None'})"
            )
            return activities_map

        results = self.evaluate_all_activities(activities_map, motion_context)

        filtered_map = {}
        exempted_list = []

        for activity_name, detected in activities_map.items():
            result = results.get(activity_name)
            if result and result.is_exempted:
                # Activity is exempted - set to not detected
                filtered_map[activity_name] = False
                if detected:
                    exempted_list.append(activity_name)
            else:
                # Keep original detection status
                filtered_map[activity_name] = detected

        if exempted_list:
            logger.info(
                f"[RULE-ENGINE] Filtered activities - "
                f"exempted (set to False): {exempted_list}"
            )

        return filtered_map

    def _get_activity_type(self, activity_name: str) -> int:
        """
        Get activity type code from name

        Args:
            activity_name: Activity name

        Returns:
            Activity type code
        """
        name_to_type = {
            'cell_phone': ActivityTypeEnum.CELL_PHONE,
            'microsleep': ActivityTypeEnum.MICROSLEEP,
            'sleep': ActivityTypeEnum.SLEEP,
            'writing': ActivityTypeEnum.WRITING,
            'packing_bags': ActivityTypeEnum.PACKING_BAGS,
            'group_detected': ActivityTypeEnum.GROUP_DETECTED,
            'lp_hand_gesture': ActivityTypeEnum.LP_NOT_EXCHANGING_HAND_GESTURE,
            'alp_hand_gesture': ActivityTypeEnum.ALP_NOT_EXCHANGING_HAND_GESTURE,
            'mind_diversion': ActivityTypeEnum.MIND_DIVERSION,
            'no_person_detected': ActivityTypeEnum.NO_PERSON_DETECTED,
            'eating_drinking': ActivityTypeEnum.EATING_DRINKING,
        }
        return name_to_type.get(activity_name, ActivityTypeEnum.UNKNOWN)


# Global service instance
_rule_engine: Optional[RuleEngineService] = None
_rule_engine_lock = threading.Lock()


def get_rule_engine_service() -> RuleEngineService:
    """
    Get the global rule engine service instance.

    M-25: Thread-safe double-checked locking pattern.

    Returns:
        RuleEngineService instance
    """
    global _rule_engine
    if _rule_engine is None:
        with _rule_engine_lock:
            if _rule_engine is None:
                _rule_engine = RuleEngineService()
    return _rule_engine

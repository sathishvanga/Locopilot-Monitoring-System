"""
Activity Aggregation Service - Merges consecutive activities of the same type/role

This service reduces noise for reviewers by aggregating multiple detections of the
same activity (e.g., "packing bags" across 500 frames) into a single violation report.
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional

from ..utils.logger import get_logger
from ..utils.config import get_settings

logger = get_logger(__name__)


class ActivityAggregationService:
    """
    Service for merging consecutive activities that are temporally close.

    Merge criteria:
    1. Same activityType (numeric code)
    2. Same performingRole (LP or ALP)
    3. Same crewRole
    4. Gap between end_time of first and start_time of next <= merge_window_seconds

    Merged activity properties:
    - activityStartTime: earliest start time
    - activityEndTime: latest end time
    - activityImage: from longest segment
    - activityClip: from longest segment
    """

    def __init__(self):
        self.settings = get_settings()
        self.merge_window = self.settings.activity_merge_window_seconds
        self.enabled = self.settings.activity_merge_enabled
        self.preserve_raw = self.settings.activity_preserve_raw
        logger.info(
            f"Activity aggregation service initialized - "
            f"Enabled: {self.enabled}, Merge window: {self.merge_window}s, "
            f"Preserve raw: {self.preserve_raw}"
        )

    def aggregate_activities(
        self,
        activities: List[Dict[str, Any]],
        run_dir: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """
        Aggregate consecutive activities of the same type and role.

        Args:
            activities: List of activity dictionaries (sorted by start time)
            run_dir: Run directory for saving debug info

        Returns:
            Tuple of (merged_activities, original_activities_if_preserve_raw)
        """
        if not self.enabled or not activities:
            return activities, None

        # Ensure activities are sorted by start time
        sorted_activities = sorted(
            activities,
            key=lambda x: float(x.get('activityStartTime', 0))
        )

        # Save raw activities if preservation is enabled
        raw_activities = sorted_activities.copy() if self.preserve_raw else None

        # Group and merge activities
        merged = []
        current_group = []

        for activity in sorted_activities:
            if not current_group:
                current_group.append(activity)
                continue

            # Check if this activity can be merged with current group
            if self._can_merge(current_group[-1], activity):
                current_group.append(activity)
            else:
                # Finalize current group and start new one
                merged_activity = self._merge_group(current_group)
                merged.append(merged_activity)
                current_group = [activity]

        # Don't forget the last group
        if current_group:
            merged_activity = self._merge_group(current_group)
            merged.append(merged_activity)

        # Log aggregation results
        original_count = len(sorted_activities)
        merged_count = len(merged)
        if merged_count < original_count:
            logger.info(
                f"Activity aggregation: {original_count} -> {merged_count} activities "
                f"({original_count - merged_count} merged, window={self.merge_window}s)"
            )

        # Save raw activities to debug file if preservation is enabled
        if self.preserve_raw and run_dir and raw_activities:
            self._save_raw_activities(raw_activities, run_dir)

        return merged, raw_activities

    def _can_merge(self, prev: Dict[str, Any], curr: Dict[str, Any]) -> bool:
        """
        Check if two activities can be merged.

        Criteria:
        1. Same activityType
        2. Same performingRole (LP/ALP)
        3. Same crewRole
        4. Gap between prev.endTime and curr.startTime <= merge_window
        """
        # Check type match
        if prev.get('activityType') != curr.get('activityType'):
            return False

        # Check role match
        if prev.get('performingRole') != curr.get('performingRole'):
            return False

        if prev.get('crewRole') != curr.get('crewRole'):
            return False

        # Check temporal proximity
        try:
            prev_end = float(prev.get('activityEndTime', 0))
            curr_start = float(curr.get('activityStartTime', 0))
            gap = curr_start - prev_end

            # Merge if:
            # - gap <= merge_window (activities are close enough)
            # - OR gap < 0 (activities overlap - always merge overlapping activities)
            return gap <= self.merge_window
        except (ValueError, TypeError):
            return False

    def _merge_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge a group of activities into a single activity.

        Strategy:
        - Start time: earliest
        - End time: latest
        - Image: from longest segment
        - Clip: from longest segment
        """
        if len(group) == 1:
            return group[0]

        # Find earliest start and latest end
        earliest_start = min(float(a.get('activityStartTime', 0)) for a in group)
        latest_end = max(float(a.get('activityEndTime', 0)) for a in group)

        # Find longest segment for representative image and clip
        longest_segment = max(
            group,
            key=lambda a: float(a.get('activityEndTime', 0)) - float(a.get('activityStartTime', 0))
        )

        # Create merged activity (copy from first in group)
        merged = group[0].copy()

        # Update timestamps
        merged['activityStartTime'] = f"{earliest_start:.2f}"
        merged['activityEndTime'] = f"{latest_end:.2f}"

        # Use longest segment's image and clip as representative
        merged['activityImage'] = longest_segment.get('activityImage')
        merged['activityClip'] = longest_segment.get('activityClip')

        # Add aggregation metadata
        merged['_aggregated'] = True
        merged['_mergedCount'] = len(group)
        merged['_originalSegments'] = [
            {
                'startTime': a.get('activityStartTime'),
                'endTime': a.get('activityEndTime'),
                'clip': a.get('activityClip'),
                'image': a.get('activityImage')
            }
            for a in group
        ]

        # Update people count to max seen
        merged['peopleCount'] = max(a.get('peopleCount', 1) for a in group)

        logger.debug(
            f"Merged {len(group)} activities of type {merged.get('activityType')} "
            f"({earliest_start:.2f}s - {latest_end:.2f}s)"
        )

        return merged

    def _save_raw_activities(
        self,
        raw_activities: List[Dict[str, Any]],
        run_dir: str
    ) -> None:
        """Save raw (unmerged) activities to debug file."""
        try:
            raw_path = os.path.join(run_dir, "activities_raw.json")
            with open(raw_path, 'w') as f:
                json.dump(raw_activities, f, indent=2)
            logger.debug(f"Saved {len(raw_activities)} raw activities to {raw_path}")
        except Exception as e:
            logger.warning(f"Failed to save raw activities: {e}")


# Singleton instance
_aggregation_service: Optional[ActivityAggregationService] = None


def get_activity_aggregation_service() -> ActivityAggregationService:
    """Get singleton instance of activity aggregation service."""
    global _aggregation_service
    if _aggregation_service is None:
        _aggregation_service = ActivityAggregationService()
    return _aggregation_service

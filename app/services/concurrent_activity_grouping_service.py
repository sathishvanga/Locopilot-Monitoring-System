"""
Concurrent Activity Grouping Service - Groups overlapping activities of different types

This service groups activities with overlapping time ranges into combined records,
regardless of activity type. For example, if a cell phone activity (5-10s) overlaps
with a hand gesture activity (5-10s), they become a single combined record with arrays.

Output format:
{
    "activityTypes": [2, 8],
    "descriptions": ["Using mobile phone", "LP not exchanging hand gesture"],
    "objectTypes": ["cell phone", "lp hand gesture"],
    "activityType": 2,      # Legacy: first element
    "des": "Using mobile phone; LP not exchanging hand gesture",
    ...
}
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict

from ..utils.logger import get_logger
from ..utils.config import get_settings
from .activity_aggregation_service import parse_time_to_seconds

logger = get_logger(__name__)


class ConcurrentActivityGroupingService:
    """
    Service for grouping concurrent (overlapping) activities of different types.

    Grouping criteria:
    1. Time ranges overlap (any overlap, not just identical times)
    2. Same performingRole (LP or ALP) - activities must be for same person
    3. Different activityType (same-type are already merged by aggregation service)

    Overlap detection uses interval overlap algorithm:
    - Activities A and B overlap if: A.start < B.end AND B.start < A.end
    """

    def __init__(self):
        self.settings = get_settings()
        self.enabled = self.settings.concurrent_grouping_enabled
        logger.info(f"Concurrent activity grouping service initialized - Enabled: {self.enabled}")

    def group_concurrent_activities(
        self,
        activities: List[Dict[str, Any]],
        run_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Group concurrent activities into combined records.

        Algorithm:
        1. Sort activities by start time
        2. Build overlap graph (activities that overlap)
        3. Find connected components (groups of mutually overlapping activities)
        4. Merge each group into a combined record

        Args:
            activities: List of activity dictionaries
            run_dir: Run directory (for future clip generation if needed)

        Returns:
            List of activities with overlapping ones combined into single records
        """
        if not self.enabled or not activities:
            return activities

        if len(activities) <= 1:
            return activities

        # Find groups of overlapping activities
        groups = self._find_overlapping_groups(activities)

        # Convert groups to combined records
        result = []
        for group in groups:
            if len(group) == 1:
                # Single activity - return as-is but ensure array format for consistency
                result.append(self._ensure_array_format(group[0]))
            else:
                # Multiple overlapping activities - combine into single record
                combined = self._merge_group_to_combined(group)
                result.append(combined)

        # Sort by start time
        result.sort(key=lambda x: parse_time_to_seconds(x.get('activityStartTime', 0)))

        # Log grouping results
        original_count = len(activities)
        final_count = len(result)
        if final_count < original_count:
            logger.info(
                f"Concurrent grouping: {original_count} -> {final_count} activities "
                f"({original_count - final_count} grouped)"
            )

        return result

    def _find_overlapping_groups(
        self,
        activities: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Find groups of overlapping activities using Union-Find algorithm.

        Uses Union-Find (Disjoint Set) for efficient grouping of overlapping intervals.
        Time complexity: O(n^2 * alpha(n)) where alpha is inverse Ackermann
        Space complexity: O(n)

        Args:
            activities: List of activity dictionaries

        Returns:
            List of groups, where each group is a list of overlapping activities
        """
        n = len(activities)
        if n <= 1:
            return [activities] if activities else []

        # Initialize Union-Find
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            """Find root with path compression"""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            """Union by rank"""
            px, py = find(x), find(y)
            if px != py:
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1

        # Parse times and roles once for efficiency
        times = []
        for a in activities:
            start = parse_time_to_seconds(a.get('activityStartTime', 0))
            end = parse_time_to_seconds(a.get('activityEndTime', 0))
            role = a.get('performingRole', 'LP')
            times.append((start, end, role))

        # Find overlapping pairs and union them
        for i in range(n):
            for j in range(i + 1, n):
                s1, e1, r1 = times[i]
                s2, e2, r2 = times[j]

                # Must be same performer role and have overlapping time ranges
                if r1 == r2 and self._times_overlap(s1, e1, s2, e2):
                    union(i, j)

        # Group activities by their root parent
        groups_dict = defaultdict(list)
        for i in range(n):
            root = find(i)
            groups_dict[root].append(activities[i])

        return list(groups_dict.values())

    def _times_overlap(
        self,
        s1: float,
        e1: float,
        s2: float,
        e2: float
    ) -> bool:
        """
        Check if two time ranges overlap.

        Overlap condition: s1 < e2 AND s2 < e1

        Args:
            s1, e1: Start and end of first interval
            s2, e2: Start and end of second interval

        Returns:
            True if intervals overlap, False otherwise
        """
        return s1 < e2 and s2 < e1

    def _ensure_array_format(self, activity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure activity has array format fields for consistency.

        Adds activityTypes, descriptions, objectTypes arrays if not present.
        """
        result = activity.copy()

        # Add array versions if not already present
        if 'activityTypes' not in result:
            result['activityTypes'] = [result.get('activityType', 0)]
        if 'descriptions' not in result:
            result['descriptions'] = [result.get('des', '')]
        if 'objectTypes' not in result:
            result['objectTypes'] = [result.get('objectType', '')]

        # Ensure evidence is in array format
        evidence = result.get('evidence', {})
        if isinstance(evidence, dict):
            result['evidence'] = [evidence]

        return result

    def _merge_group_to_combined(
        self,
        group: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge a group of overlapping activities into a single combined record.

        Merging strategy:
        - activityTypes: collect all unique types in order
        - descriptions: collect all descriptions
        - objectTypes: collect all object types
        - activityStartTime: max of all start times (intersection start)
        - activityEndTime: min of all end times (intersection end)
        - activityImage: from longest activity
        - activityClip: from longest activity
        - evidence: combine all evidence objects into array
        - Legacy fields: first element for backward compatibility

        Args:
            group: List of overlapping activities to merge

        Returns:
            Combined activity dictionary
        """
        if len(group) == 1:
            return self._ensure_array_format(group[0])

        # Sort by activity type for consistent ordering
        sorted_group = sorted(group, key=lambda x: x.get('activityType', 0))

        # Collect arrays
        activity_types = []
        descriptions = []
        object_types = []
        evidences = []

        for activity in sorted_group:
            activity_types.append(activity.get('activityType', 0))
            descriptions.append(activity.get('des', ''))
            object_types.append(activity.get('objectType', ''))

            evidence = activity.get('evidence', {})
            if isinstance(evidence, dict):
                evidences.append(evidence)
            elif isinstance(evidence, list):
                evidences.extend(evidence)

        # Calculate intersection time (overlap period)
        earliest_start = max(
            parse_time_to_seconds(a.get('activityStartTime', 0))
            for a in group
        )
        latest_end = min(
            parse_time_to_seconds(a.get('activityEndTime', 0))
            for a in group
        )

        # If no valid intersection (shouldn't happen), use union instead
        if earliest_start >= latest_end:
            earliest_start = min(
                parse_time_to_seconds(a.get('activityStartTime', 0))
                for a in group
            )
            latest_end = max(
                parse_time_to_seconds(a.get('activityEndTime', 0))
                for a in group
            )

        # Find longest activity for representative image/clip
        longest_activity = max(
            group,
            key=lambda a: (
                parse_time_to_seconds(a.get('activityEndTime', 0)) -
                parse_time_to_seconds(a.get('activityStartTime', 0))
            )
        )

        # Create combined record from first activity as base
        combined = sorted_group[0].copy()

        # Update with array fields
        combined['activityTypes'] = activity_types
        combined['descriptions'] = descriptions
        combined['objectTypes'] = object_types
        combined['evidence'] = evidences

        # Update timestamps to intersection
        combined['activityStartTime'] = f"{earliest_start:.2f}"
        combined['activityEndTime'] = f"{latest_end:.2f}"

        # Use longest activity's image and clip
        combined['activityImage'] = longest_activity.get('activityImage')
        combined['activityClip'] = longest_activity.get('activityClip')

        # Legacy single-value fields (first element for backward compatibility)
        combined['activityType'] = activity_types[0] if activity_types else 0
        combined['des'] = "; ".join(descriptions)
        combined['objectType'] = object_types[0] if object_types else ''

        # Add metadata
        combined['_isCombined'] = True
        combined['_mergedCount'] = len(group)
        combined['_sourceActivities'] = [
            {
                'activityType': a.get('activityType'),
                'startTime': a.get('activityStartTime'),
                'endTime': a.get('activityEndTime'),
                'des': a.get('des')
            }
            for a in sorted_group
        ]

        # Use max people count from all activities
        combined['peopleCount'] = max(a.get('peopleCount', 1) for a in group)

        logger.debug(
            f"Combined {len(group)} activities: types={activity_types}, "
            f"time={earliest_start:.2f}s-{latest_end:.2f}s"
        )

        return combined


# Singleton instance
_concurrent_grouping_service: Optional[ConcurrentActivityGroupingService] = None


def get_concurrent_grouping_service() -> ConcurrentActivityGroupingService:
    """Get singleton instance of concurrent activity grouping service."""
    global _concurrent_grouping_service
    if _concurrent_grouping_service is None:
        _concurrent_grouping_service = ConcurrentActivityGroupingService()
    return _concurrent_grouping_service

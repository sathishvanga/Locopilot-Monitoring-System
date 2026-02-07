"""
Minute-Based Activity Grouping Service - Groups activities within the same minute

This service groups all activities that start within the same calendar minute into
combined records with merged video clips, making it easier for reviewers to process
violations. Activities are grouped by minute bucket (0-59s = minute 0, 60-119s = minute 1, etc.)

Example:
- Activity A: cell phone at 65.5s (minute 1)
- Activity B: writing at 68.2s (minute 1)
- Activity C: cell phone at 125.0s (minute 2)

Result:
- Combined record for minute 1: [A, B] with merged clip
- Single record for minute 2: [C]

Output format:
{
    "activityTypes": [2, 5],
    "descriptions": ["Using mobile phone", "Writing on paper"],
    "objectTypes": ["cell phone", "writing"],
    "activityType": 2,      # Legacy: first element
    "des": "Using mobile phone; Writing on paper",
    "activityStartTime": "65.50",  # Earliest in minute
    "activityEndTime": "72.30",    # Latest in minute
    "activityClip": "merged_minute001.mp4",
    "_isCombined": True,
    "_mergedCount": 2,
    "_minuteBucket": 1,
    ...
}
"""

import os
import subprocess
from typing import List, Dict, Any, Optional
from collections import defaultdict

from ..utils.logger import get_logger
from ..utils.config import get_settings
from ..repositories.activity_repository import parse_time_to_seconds

logger = get_logger(__name__)


class ConcurrentActivityGroupingService:
    """
    Service for grouping activities by minute with merged video clips.

    Grouping criteria:
    1. Activities that START within the same calendar minute are grouped together
    2. Same performingRole (LP or ALP) - activities must be for same person
    3. Video clips from grouped activities are merged into a single clip

    Minute bucket calculation:
    - minute_bucket = floor(start_time_seconds / 60)
    - Example: 65.5s -> minute 1, 125.3s -> minute 2
    """

    def __init__(self):
        self.settings = get_settings()
        self.enabled = self.settings.concurrent_grouping_enabled
        logger.info(f"Minute-based activity grouping service initialized - Enabled: {self.enabled}")

    def group_concurrent_activities(
        self,
        activities: List[Dict[str, Any]],
        run_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Group activities by minute into combined records with merged clips.

        Algorithm:
        1. Group activities by minute bucket (floor(start_seconds / 60))
        2. Merge each minute's activities into a combined record
        3. Create merged video clips for each minute group

        Args:
            activities: List of activity dictionaries
            run_dir: Run directory for clip merging

        Returns:
            List of activities with same-minute ones combined into single records
        """
        if not self.enabled or not activities:
            return activities

        if len(activities) <= 1:
            return [self._ensure_array_format(a) for a in activities]

        # Find groups by minute bucket
        groups = self._find_minute_groups(activities)

        # Convert groups to combined records
        result = []
        for group in groups:
            if len(group) == 1:
                # Single activity - return as-is but ensure array format for consistency
                result.append(self._ensure_array_format(group[0]))
            else:
                # Multiple activities in same minute - combine into single record with merged clip
                combined = self._merge_group_to_combined(group, run_dir)
                result.append(combined)

        # Sort by start time
        result.sort(key=lambda x: parse_time_to_seconds(x.get('activityStartTime', 0)))

        # Log grouping results
        original_count = len(activities)
        final_count = len(result)
        if final_count < original_count:
            minute_buckets = len(groups)
            logger.info(
                f"Minute-based grouping: {original_count} -> {final_count} activities "
                f"({original_count - final_count} grouped across {minute_buckets} minute buckets)"
            )

        return result

    def _find_minute_groups(
        self,
        activities: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Group activities by minute bucket.

        All activities starting within the same calendar minute (0-59s, 60-119s, etc.)
        are grouped together, regardless of overlap.

        Args:
            activities: List of activity dictionaries

        Returns:
            List of groups, where each group contains activities from the same minute
        """
        if not activities:
            return []

        # Group by minute bucket AND performing role
        minute_role_groups = defaultdict(list)

        for activity in activities:
            start_seconds = parse_time_to_seconds(activity.get('activityStartTime', 0))
            minute_bucket = int(start_seconds // 60)
            role = activity.get('performingRole', 'LP')

            # Key: (minute_bucket, role) to keep LP and ALP activities separate
            key = (minute_bucket, role)
            minute_role_groups[key].append(activity)

        return list(minute_role_groups.values())

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

    def _merge_video_clips(
        self,
        clip_paths: List[str],
        run_dir: str,
        minute_bucket: int
    ) -> Optional[str]:
        """
        Merge multiple video clips into a single combined clip.

        Uses ffmpeg concat demuxer for seamless concatenation.

        Args:
            clip_paths: List of paths to source video clips (in chronological order)
            run_dir: Run directory for output
            minute_bucket: Minute bucket for naming the output file

        Returns:
            Path to the merged video clip, or None if merge fails
        """
        if not clip_paths:
            return None

        if len(clip_paths) == 1:
            return clip_paths[0]

        # Create output path
        clips_dir = os.path.join(run_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)

        # Generate unique filename for merged clip
        first_clip_name = os.path.basename(clip_paths[0])
        base_name = first_clip_name.rsplit('_clip.mp4', 1)[0] if '_clip.mp4' in first_clip_name else first_clip_name.rsplit('.mp4', 1)[0]
        merged_clip_filename = f"{base_name}_minute{minute_bucket:03d}_merged.mp4"
        merged_clip_path = os.path.join(clips_dir, merged_clip_filename)

        # Create concat file list for ffmpeg
        concat_list_path = os.path.join(clips_dir, f"concat_minute{minute_bucket:03d}.txt")

        try:
            # Write concat file (ffmpeg concat demuxer format)
            with open(concat_list_path, 'w') as f:
                for clip_path in clip_paths:
                    # Escape single quotes in paths
                    escaped_path = clip_path.replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")

            # Run ffmpeg concat
            ffmpeg_path = os.environ.get('FFMPEG_PATH', 'ffmpeg')
            cmd = [
                ffmpeg_path, '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_list_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-an',  # No audio
                '-movflags', '+faststart',
                merged_clip_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=120
            )

            if result.returncode == 0 and os.path.exists(merged_clip_path):
                logger.info(f"Merged {len(clip_paths)} clips into: {merged_clip_filename}")
                return merged_clip_path
            else:
                logger.warning(f"ffmpeg concat failed: {result.stderr.decode()[:200]}")
                return clip_paths[0]  # Fallback to first clip

        except subprocess.TimeoutExpired:
            logger.warning(f"Clip merge timed out for minute {minute_bucket}")
            return clip_paths[0]
        except Exception as e:
            logger.warning(f"Clip merge failed: {e}")
            return clip_paths[0]
        finally:
            # Cleanup concat file
            if os.path.exists(concat_list_path):
                try:
                    os.remove(concat_list_path)
                except OSError:
                    pass

    def _merge_group_to_combined(
        self,
        group: List[Dict[str, Any]],
        run_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Merge a group of minute-grouped activities into a single combined record.

        Merging strategy for minute-based grouping:
        - activityTypes: collect all types in chronological order
        - descriptions: collect all descriptions
        - objectTypes: collect all object types
        - activityStartTime: EARLIEST start time (display time requirement)
        - activityEndTime: LATEST end time (full coverage)
        - activityImage: from longest activity
        - activityClip: MERGED clip from all source clips
        - evidence: combine all evidence objects into array
        - Legacy fields: first element for backward compatibility

        Args:
            group: List of activities in same minute to merge
            run_dir: Run directory for clip merging

        Returns:
            Combined activity dictionary
        """
        if len(group) == 1:
            return self._ensure_array_format(group[0])

        # Sort by start time (chronological order for display)
        sorted_group = sorted(
            group,
            key=lambda x: parse_time_to_seconds(x.get('activityStartTime', 0))
        )

        # Collect UNIQUE activity types (deduplicated) while preserving order
        # Use dict to track unique types with their descriptions/object_types
        unique_activities = {}  # {activityType: {'des': str, 'objectType': str}}
        evidences = []
        source_clips = []  # Collect clip paths for merging

        for activity in sorted_group:
            act_type = activity.get('activityType', 0)

            # Only add if this activity type hasn't been seen yet
            if act_type not in unique_activities:
                unique_activities[act_type] = {
                    'des': activity.get('des', ''),
                    'objectType': activity.get('objectType', '')
                }

            evidence = activity.get('evidence', {})
            if isinstance(evidence, dict):
                evidences.append(evidence)
            elif isinstance(evidence, list):
                evidences.extend(evidence)

            # Collect clip paths for merging
            clip_path = activity.get('activityClip')
            if clip_path and os.path.exists(clip_path):
                source_clips.append(clip_path)

        # Extract unique arrays (preserving insertion order from Python 3.7+)
        activity_types = list(unique_activities.keys())
        descriptions = [v['des'] for v in unique_activities.values()]
        object_types = [v['objectType'] for v in unique_activities.values()]

        # For minute-based grouping: use union time (earliest start to latest end)
        earliest_start = min(
            parse_time_to_seconds(a.get('activityStartTime', 0))
            for a in group
        )
        latest_end = max(
            parse_time_to_seconds(a.get('activityEndTime', 0))
            for a in group
        )

        # Calculate minute bucket for metadata
        minute_bucket = int(earliest_start // 60)

        # Find longest activity for representative image
        longest_activity = max(
            group,
            key=lambda a: (
                parse_time_to_seconds(a.get('activityEndTime', 0)) -
                parse_time_to_seconds(a.get('activityStartTime', 0))
            )
        )

        # Create combined record from first (earliest) activity as base
        combined = sorted_group[0].copy()

        # Update with array fields
        combined['activityTypes'] = activity_types
        combined['descriptions'] = descriptions
        combined['objectTypes'] = object_types
        combined['evidence'] = evidences

        # Update timestamps - use EARLIEST start for display time requirement
        combined['activityStartTime'] = f"{earliest_start:.2f}"
        combined['activityEndTime'] = f"{latest_end:.2f}"

        # Use longest activity's image
        combined['activityImage'] = longest_activity.get('activityImage')

        # Merge clips if we have multiple and run_dir is available
        if len(source_clips) > 1 and run_dir:
            merged_clip_path = self._merge_video_clips(source_clips, run_dir, minute_bucket)
            combined['activityClip'] = merged_clip_path
            combined['_sourceClips'] = source_clips  # Keep reference to originals
        else:
            combined['activityClip'] = longest_activity.get('activityClip')

        # Legacy single-value fields (first element for backward compatibility)
        combined['activityType'] = activity_types[0] if activity_types else 0
        combined['des'] = "; ".join(descriptions)
        combined['objectType'] = object_types[0] if object_types else ''

        # Add metadata
        combined['_isCombined'] = True
        combined['_mergedCount'] = len(group)
        combined['_groupingMethod'] = 'minute'
        combined['_minuteBucket'] = minute_bucket
        combined['_sourceActivities'] = [
            {
                'activityType': a.get('activityType'),
                'startTime': a.get('activityStartTime'),
                'endTime': a.get('activityEndTime'),
                'des': a.get('des'),
                'clip': a.get('activityClip')
            }
            for a in sorted_group
        ]

        # Use max people count from all activities
        combined['peopleCount'] = max(a.get('peopleCount', 1) for a in group)

        logger.debug(
            f"Combined {len(group)} activities in minute {minute_bucket}: "
            f"types={activity_types}, time={earliest_start:.2f}s-{latest_end:.2f}s"
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

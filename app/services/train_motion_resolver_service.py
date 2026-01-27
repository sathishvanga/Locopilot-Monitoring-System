"""
Train motion resolver service - Maps video timestamp to train motion state

This service determines whether the train is running or stopped at a given
timestamp by comparing against the trip schedule.
"""

import logging
from typing import Optional

from ..models.trip_models import (
    TrainMotionState,
    TrainMotionContext,
    TripSchedule,
    StationHalt
)
from ..utils.config import get_settings

logger = logging.getLogger(__name__)


class TrainMotionResolverService:
    """
    Service for resolving train motion state from video timestamps

    Maps video timestamps to train motion states (RUNNING, STOPPED, UNKNOWN)
    by comparing against the trip schedule.

    Features:
    - Determines if train is at station (within halt period)
    - Identifies pre-arrival window for ALP alertness checks
    - Handles grace period after scheduled departure
    - Day rollover support for overnight trips
    """

    def __init__(self):
        """Initialize the motion resolver service"""
        self.settings = get_settings()
        self.enabled = self.settings.train_motion_rules_enabled

        # Pre-arrival window settings (seconds before arrival)
        self.pre_arrival_start = self.settings.pre_arrival_window_start  # 60s
        self.pre_arrival_end = self.settings.pre_arrival_window_end  # 30s

        # Grace period after departure (allow brief overlap)
        self.halt_grace_period = self.settings.halt_grace_period  # 120s

        logger.info(
            f"TrainMotionResolverService initialized - "
            f"enabled: {self.enabled}, "
            f"pre-arrival window: {self.pre_arrival_end}-{self.pre_arrival_start}s, "
            f"grace period: {self.halt_grace_period}s"
        )

    def resolve_motion_state(
        self,
        video_timestamp: str,
        trip_schedule: Optional[TripSchedule]
    ) -> TrainMotionContext:
        """
        Resolve train motion state for a given video timestamp

        Args:
            video_timestamp: Timestamp in HH:MM:SS format
            trip_schedule: Train schedule with station halts

        Returns:
            TrainMotionContext with motion state and relevant context
        """
        logger.debug(
            f"[MOTION-RESOLVER] resolve_motion_state() called - "
            f"timestamp: {video_timestamp}, "
            f"schedule: {'present' if trip_schedule else 'None'}"
        )

        # Default context when unable to determine
        default_context = TrainMotionContext(
            motion_state=TrainMotionState.UNKNOWN,
            timestamp=video_timestamp,
            resolution_source="fallback_default"
        )

        if not self.enabled:
            logger.debug("[MOTION-RESOLVER] ❌ Rules disabled - returning UNKNOWN")
            default_context.resolution_source = "rules_disabled"
            return default_context

        if trip_schedule is None or not trip_schedule.halts:
            logger.debug("[MOTION-RESOLVER] ❌ No schedule/halts - returning UNKNOWN")
            default_context.resolution_source = "no_schedule"
            return default_context

        try:
            # Parse video timestamp to minutes from midnight
            query_minutes = self._time_to_minutes(video_timestamp)
            logger.debug(
                f"[MOTION-RESOLVER] Parsed timestamp {video_timestamp} -> {query_minutes} minutes from midnight"
            )

            # Check if at a station (within halt period)
            current_halt, minutes_into_halt = self._find_current_halt(
                query_minutes, trip_schedule.halts
            )

            if current_halt is not None:
                # Train is STOPPED at station
                logger.info(
                    f"[MOTION-RESOLVER] 🛑 STOPPED at station - "
                    f"{current_halt.station_name} ({current_halt.station_code}), "
                    f"{minutes_into_halt} min into halt, "
                    f"timestamp: {video_timestamp}"
                )
                return TrainMotionContext(
                    motion_state=TrainMotionState.STOPPED,
                    timestamp=video_timestamp,
                    current_station=current_halt,
                    next_station=self._find_next_halt(query_minutes, trip_schedule.halts),
                    is_pre_arrival_window=False,
                    seconds_to_arrival=None,
                    seconds_since_departure=None,
                    in_grace_period=False,
                    resolution_source="at_station"
                )

            # Check if in grace period after departure
            last_halt, seconds_since = self._find_recent_departure(
                query_minutes, trip_schedule.halts
            )

            if last_halt is not None and seconds_since <= self.halt_grace_period:
                # Still in grace period after departure
                logger.info(
                    f"[MOTION-RESOLVER] 🛑 STOPPED (grace period) - "
                    f"departed {last_halt.station_name} {int(seconds_since)}s ago, "
                    f"grace period: {self.halt_grace_period}s, "
                    f"timestamp: {video_timestamp}"
                )
                return TrainMotionContext(
                    motion_state=TrainMotionState.STOPPED,
                    timestamp=video_timestamp,
                    current_station=last_halt,
                    next_station=self._find_next_halt(query_minutes, trip_schedule.halts),
                    is_pre_arrival_window=False,
                    seconds_to_arrival=None,
                    seconds_since_departure=int(seconds_since),
                    in_grace_period=True,
                    resolution_source="grace_period"
                )

            # Check if approaching station (pre-arrival window)
            next_halt = self._find_next_halt(query_minutes, trip_schedule.halts)
            if next_halt is not None:
                seconds_to_arrival = (next_halt.arrival_minutes - query_minutes) * 60

                # Account for day rollover
                if seconds_to_arrival < 0:
                    seconds_to_arrival += 24 * 60 * 60

                is_pre_arrival = self.pre_arrival_end <= seconds_to_arrival <= self.pre_arrival_start

                if is_pre_arrival:
                    logger.info(
                        f"[MOTION-RESOLVER] 🚂 RUNNING (PRE-ARRIVAL) - "
                        f"approaching {next_halt.station_name} in {int(seconds_to_arrival)}s, "
                        f"window: {self.pre_arrival_end}-{self.pre_arrival_start}s, "
                        f"timestamp: {video_timestamp}"
                    )
                else:
                    logger.debug(
                        f"[MOTION-RESOLVER] 🚂 RUNNING - "
                        f"next: {next_halt.station_name} in {int(seconds_to_arrival)}s, "
                        f"timestamp: {video_timestamp}"
                    )

                return TrainMotionContext(
                    motion_state=TrainMotionState.RUNNING,
                    timestamp=video_timestamp,
                    current_station=None,
                    next_station=next_halt,
                    is_pre_arrival_window=is_pre_arrival,
                    seconds_to_arrival=int(seconds_to_arrival) if seconds_to_arrival >= 0 else None,
                    seconds_since_departure=int(seconds_since) if last_halt else None,
                    in_grace_period=False,
                    resolution_source="running_with_next"
                )

            # Train is running (no upcoming station found)
            logger.debug(
                f"[MOTION-RESOLVER] 🚂 RUNNING (no next station) - timestamp: {video_timestamp}"
            )
            return TrainMotionContext(
                motion_state=TrainMotionState.RUNNING,
                timestamp=video_timestamp,
                current_station=None,
                next_station=None,
                is_pre_arrival_window=False,
                seconds_to_arrival=None,
                seconds_since_departure=int(seconds_since) if last_halt else None,
                in_grace_period=False,
                resolution_source="running_no_next"
            )

        except Exception as e:
            logger.error(f"[MOTION-RESOLVER] ❌ Error resolving motion state: {e}", exc_info=True)
            default_context.resolution_source = f"error: {str(e)}"
            return default_context

    def _time_to_minutes(self, time_str: str) -> int:
        """
        Convert HH:MM:SS to minutes from midnight

        Args:
            time_str: Time in HH:MM:SS format

        Returns:
            Minutes from midnight
        """
        try:
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            return hours * 60 + minutes
        except (ValueError, IndexError):
            return 0

    def _time_to_seconds(self, time_str: str) -> int:
        """
        Convert HH:MM:SS to seconds from midnight

        Args:
            time_str: Time in HH:MM:SS format

        Returns:
            Seconds from midnight
        """
        try:
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2]) if len(parts) > 2 else 0
            return hours * 3600 + minutes * 60 + seconds
        except (ValueError, IndexError):
            return 0

    def _find_current_halt(
        self,
        query_minutes: int,
        halts: list
    ) -> tuple:
        """
        Find if train is currently at a station halt

        Args:
            query_minutes: Query time in minutes from midnight
            halts: List of station halts

        Returns:
            Tuple of (StationHalt, minutes_into_halt) or (None, 0)
        """
        for halt in halts:
            arrival = halt.arrival_minutes
            departure = halt.departure_minutes

            # Handle day rollover (departure after midnight)
            if departure < arrival:
                # Halt spans midnight
                if query_minutes >= arrival or query_minutes <= departure:
                    minutes_into_halt = query_minutes - arrival
                    if minutes_into_halt < 0:
                        minutes_into_halt += 24 * 60
                    return halt, minutes_into_halt
            else:
                # Normal halt (same day)
                if arrival <= query_minutes <= departure:
                    return halt, query_minutes - arrival

        return None, 0

    def _find_next_halt(
        self,
        query_minutes: int,
        halts: list
    ) -> Optional[StationHalt]:
        """
        Find the next upcoming station halt

        Args:
            query_minutes: Query time in minutes from midnight
            halts: List of station halts

        Returns:
            Next StationHalt or None
        """
        # First pass: find halt after query time on same day
        for halt in halts:
            if halt.arrival_minutes > query_minutes:
                return halt

        # Second pass: wrap around to next day (for overnight trains)
        if halts:
            return halts[0]

        return None

    def _find_recent_departure(
        self,
        query_minutes: int,
        halts: list
    ) -> tuple:
        """
        Find the most recent station departure

        Args:
            query_minutes: Query time in minutes from midnight
            halts: List of station halts

        Returns:
            Tuple of (StationHalt, seconds_since_departure) or (None, 0)
        """
        last_departure = None
        min_elapsed = float('inf')

        for halt in halts:
            departure = halt.departure_minutes

            # Calculate time since departure
            elapsed = query_minutes - departure

            # Handle day rollover
            if elapsed < 0:
                elapsed += 24 * 60

            # Track the most recent departure
            if 0 <= elapsed < min_elapsed:
                min_elapsed = elapsed
                last_departure = halt

        if last_departure is not None:
            return last_departure, min_elapsed * 60  # Convert to seconds

        return None, 0

    def is_pre_arrival_window(self, context: TrainMotionContext) -> bool:
        """
        Check if currently in pre-arrival window

        Args:
            context: Train motion context

        Returns:
            True if in pre-arrival window
        """
        return context.is_pre_arrival_window

    def is_stopped_at_station(self, context: TrainMotionContext) -> bool:
        """
        Check if train is stopped at a station

        Args:
            context: Train motion context

        Returns:
            True if stopped (including grace period)
        """
        return context.motion_state == TrainMotionState.STOPPED


# Global service instance
_motion_resolver: Optional[TrainMotionResolverService] = None


def get_motion_resolver_service() -> TrainMotionResolverService:
    """
    Get the global motion resolver service instance

    Returns:
        TrainMotionResolverService instance
    """
    global _motion_resolver
    if _motion_resolver is None:
        _motion_resolver = TrainMotionResolverService()
    return _motion_resolver

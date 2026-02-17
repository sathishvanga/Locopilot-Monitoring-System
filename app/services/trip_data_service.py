"""
Trip data service - Fetches train schedule from RailRadar API

This service integrates with the RailRadar API to fetch train schedules
for determining train motion state (running vs stopped).

Enhanced with etrain.info delay integration for actual arrival/departure times.
"""

import logging
import requests
import threading
from typing import Optional
from datetime import datetime
from functools import lru_cache
from cachetools import TTLCache

from ..models.trip_models import TripSchedule, StationHalt
from ..utils.config import get_settings

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency
_etrain_service = None


def _get_etrain_service():
    """Get the etrain delay service (lazy import)."""
    global _etrain_service
    if _etrain_service is None:
        try:
            from .etrain_delay_service import get_etrain_delay_service
            _etrain_service = get_etrain_delay_service()
        except ImportError:
            logger.warning("[TRIP-DATA] etrain_delay_service not available")
            _etrain_service = None
    return _etrain_service

# TTL cache for trip schedules (1 hour)
_schedule_cache = TTLCache(maxsize=100, ttl=3600)


def minutes_to_time(minutes: int) -> str:
    """
    Convert minutes from midnight to HH:MM:SS format

    Args:
        minutes: Minutes from midnight (0-1439)

    Returns:
        Time string in HH:MM:SS format
    """
    # Handle day overflow (for overnight trains)
    minutes = minutes % 1440  # 24 * 60 = 1440 minutes per day
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}:00"


def time_to_minutes(time_str: str) -> int:
    """
    Convert HH:MM:SS time string to minutes from midnight

    Args:
        time_str: Time in HH:MM:SS format

    Returns:
        Minutes from midnight
    """
    try:
        parts = time_str.split(':')
        hours = int(parts[0])
        mins = int(parts[1])
        return hours * 60 + mins
    except (ValueError, IndexError):
        return 0


class TripDataService:
    """
    Service for fetching train trip schedules from RailRadar API

    Features:
    - Fetches train schedules with station arrival/departure times
    - TTL caching to reduce API calls
    - Graceful error handling with fallback to empty schedule
    """

    def __init__(self):
        """Initialize the trip data service"""
        self.settings = get_settings()
        self.api_url = self.settings.trip_api_url
        self.api_timeout = self.settings.trip_api_timeout
        self.enabled = self.settings.train_motion_rules_enabled

        logger.info(
            f"TripDataService initialized - "
            f"enabled: {self.enabled}, "
            f"API URL: {self.api_url}"
        )

    def fetch_trip_schedule(
        self,
        train_number: str,
        journey_date: str,
        division: Optional[str] = None
    ) -> Optional[TripSchedule]:
        """
        Fetch train schedule from RailRadar API

        Args:
            train_number: 5-digit train number
            journey_date: Journey date in YYYY-MM-DD format
            division: Optional division identifier (not used for RailRadar)

        Returns:
            TripSchedule if successful, None otherwise
        """
        logger.info(
            f"[TRIP-DATA] fetch_trip_schedule() called - "
            f"train: {train_number}, date: {journey_date}, division: {division}"
        )

        if not self.enabled:
            logger.warning("[TRIP-DATA] ❌ Train motion rules disabled in config, skipping schedule fetch")
            return None

        if not train_number or not journey_date:
            logger.warning(
                f"[TRIP-DATA] ❌ Missing required params - "
                f"train_number: '{train_number}', journey_date: '{journey_date}'"
            )
            return None

        # Validate train number format (Indian Railways: 1-5 digits)
        train_num_clean = str(train_number).strip()
        if not train_num_clean.isdigit() or not (1 <= len(train_num_clean) <= 5):
            logger.warning(
                f"[TRIP-DATA] ❌ Invalid train number format: '{train_number}' "
                f"(must be 1-5 digits). Skipping API call."
            )
            return None

        # Check cache first
        cache_key = f"{train_number}_{journey_date}"
        if cache_key in _schedule_cache:
            cached = _schedule_cache[cache_key]
            logger.info(
                f"[TRIP-DATA] ✅ Cache HIT for {cache_key} - "
                f"returning cached schedule with {len(cached.halts)} halts"
            )
            return cached

        logger.info(f"[TRIP-DATA] Cache MISS for {cache_key} - fetching from API")

        try:
            # Build API URL - RailRadar API v1 format
            url = f"{self.api_url}/{train_number}"
            params = {
                "dataType": "full",
                "journeyDate": journey_date  # YYYY-MM-DD format
            }
            headers = {"Accept": "application/json"}

            logger.info(
                f"[TRIP-DATA] 🌐 Making API request - "
                f"URL: {url}, params: {params}, timeout: {self.api_timeout}s"
            )

            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=self.api_timeout
            )

            logger.info(
                f"[TRIP-DATA] API response - "
                f"status: {response.status_code}, "
                f"content_length: {len(response.content)} bytes"
            )

            if response.status_code == 200:
                json_data = response.json()

                # Check for success flag in response
                if not json_data.get('success', False):
                    logger.error(f"[TRIP-DATA] ❌ API returned success=false")
                    return None

                # Extract data from response wrapper
                data = json_data.get('data', {})
                train_info = data.get('train', {})

                logger.info(
                    f"[TRIP-DATA] ✅ API returned success - "
                    f"train_name: {train_info.get('trainName', 'N/A')}, "
                    f"route_count: {len(data.get('route', []))}"
                )

                schedule = self._parse_schedule(train_number, journey_date, data)

                # Cache the result
                _schedule_cache[cache_key] = schedule
                logger.info(
                    f"[TRIP-DATA] ✅ Schedule parsed and cached - "
                    f"train: {train_number}, halts: {len(schedule.halts)}, "
                    f"origin: {schedule.origin_station}, dest: {schedule.destination_station}"
                )

                return schedule
            else:
                logger.error(
                    f"[TRIP-DATA] ❌ API error - "
                    f"status: {response.status_code}, "
                    f"response: {response.text[:500]}"
                )
                return None

        except requests.Timeout:
            logger.error(
                f"[TRIP-DATA] ❌ API timeout after {self.api_timeout}s - "
                f"train: {train_number}"
            )
            return None
        except requests.RequestException as e:
            logger.error(f"[TRIP-DATA] ❌ Request error: {e}")
            return None
        except Exception as e:
            logger.error(f"[TRIP-DATA] ❌ Unexpected error: {e}", exc_info=True)
            return None

    def _parse_schedule(
        self,
        train_number: str,
        journey_date: str,
        data: dict
    ) -> TripSchedule:
        """
        Parse RailRadar API response into TripSchedule model

        Args:
            train_number: Train number
            journey_date: Journey date
            data: API response data (contains 'train' and 'route' keys)

        Returns:
            Parsed TripSchedule
        """
        halts = []
        route = data.get("route", [])
        train_info = data.get("train", {})

        logger.debug(f"[TRIP-DATA] Parsing {len(route)} stations from route")

        for station in route:
            # Only include stations where train actually stops (isHalt=1)
            # or include first/last station regardless
            is_halt = station.get("isHalt", 0) == 1
            is_first = station.get("sequence", -1) == 0
            is_last = station == route[-1] if route else False

            if not (is_halt or is_first or is_last):
                continue  # Skip passing stations

            # Get arrival/departure times (already in minutes from midnight)
            arrival_minutes = station.get("scheduledArrival", 0) or 0
            departure_minutes = station.get("scheduledDeparture", 0) or 0

            # For first station, use departure as arrival
            if is_first and arrival_minutes == 0:
                arrival_minutes = departure_minutes

            # For last station, use arrival as departure
            if is_last and departure_minutes == 0:
                departure_minutes = arrival_minutes

            # Handle journey day adjustment (for overnight trains)
            journey_day = station.get("day", 1)
            if journey_day > 1:
                # Add 24 hours (1440 minutes) for each additional day
                arrival_minutes += (journey_day - 1) * 1440
                departure_minutes += (journey_day - 1) * 1440

            halt = StationHalt(
                station_code=station.get("stationCode", ""),
                station_name=station.get("stationName", ""),
                scheduled_arrival=minutes_to_time(arrival_minutes % 1440),
                scheduled_departure=minutes_to_time(departure_minutes % 1440),
                halt_duration_minutes=station.get("haltDurationMinutes", 0) or 0,
                distance_km=float(station.get("distanceFromSourceKm", 0) or 0),
                journey_day=journey_day,
                platform=station.get("platform"),
                arrival_minutes=arrival_minutes % 1440,  # Store normalized for same-day comparison
                departure_minutes=departure_minutes % 1440
            )
            halts.append(halt)

            logger.debug(
                f"[TRIP-DATA]   Station: {halt.station_code} ({halt.station_name}) - "
                f"Arr: {halt.scheduled_arrival}, Dep: {halt.scheduled_departure}, Day: {journey_day}"
            )

        # Extract origin and destination
        origin = halts[0].station_code if halts else None
        destination = halts[-1].station_code if halts else None
        total_distance = halts[-1].distance_km if halts else 0.0

        logger.info(
            f"[TRIP-DATA] Parsed schedule: {len(halts)} halts, "
            f"origin: {origin}, dest: {destination}, distance: {total_distance}km"
        )

        return TripSchedule(
            train_number=train_number,
            train_name=train_info.get("trainName"),
            journey_date=journey_date,
            origin_station=origin,
            destination_station=destination,
            halts=halts,
            total_distance_km=total_distance,
            fetched_at=datetime.now()
        )

    def fetch_trip_schedule_with_delays(
        self,
        train_number: str,
        journey_date: str,
        division: Optional[str] = None
    ) -> Optional[TripSchedule]:
        """
        Fetch train schedule with actual delay data merged in.

        This method:
        1. Fetches base schedule from RailRadar API
        2. Fetches delay data from etrain.info
        3. Merges delays to create adjusted schedule with actual times

        Args:
            train_number: 5-digit train number
            journey_date: Journey date in YYYY-MM-DD format
            division: Optional division identifier

        Returns:
            TripSchedule with actual_arrival_minutes and actual_departure_minutes
            populated based on delay data
        """
        logger.info(
            f"[TRIP-DATA] fetch_trip_schedule_with_delays() called - "
            f"train: {train_number}, date: {journey_date}"
        )

        # 1. Fetch base schedule from RailRadar
        base_schedule = self.fetch_trip_schedule(
            train_number, journey_date, division
        )

        if base_schedule is None:
            logger.warning("[TRIP-DATA] No base schedule fetched")
            return None

        # 2. Fetch delay data from etrain.info
        etrain_service = _get_etrain_service()
        if etrain_service is None or not etrain_service.enabled:
            logger.info("[TRIP-DATA] etrain service disabled, returning base schedule")
            return base_schedule

        delay_data = etrain_service.fetch_delay_data(
            train_number, journey_date
        )

        if delay_data.fetch_status != "success" and delay_data.fetch_status != "parsed":
            logger.info(
                f"[TRIP-DATA] Delay fetch status: {delay_data.fetch_status}, "
                f"returning base schedule without delay adjustments"
            )
            return base_schedule

        # 3. Merge delays into schedule
        adjusted_schedule = etrain_service.get_adjusted_schedule(
            base_schedule, delay_data
        )

        if adjusted_schedule:
            logger.info(
                f"[TRIP-DATA] Schedule adjusted with delay data - "
                f"avg delay: {delay_data.get_average_delay()} min"
            )
            return adjusted_schedule

        return base_schedule

    def clear_cache(self):
        """Clear the schedule cache"""
        _schedule_cache.clear()
        logger.info("Schedule cache cleared")


# Global service instance
_trip_data_service: Optional[TripDataService] = None
_trip_data_service_lock = threading.Lock()


def get_trip_data_service() -> TripDataService:
    """
    Get the global trip data service instance.

    M-25: Thread-safe double-checked locking pattern.

    Returns:
        TripDataService instance
    """
    global _trip_data_service
    if _trip_data_service is None:
        with _trip_data_service_lock:
            if _trip_data_service is None:
                _trip_data_service = TripDataService()
    return _trip_data_service

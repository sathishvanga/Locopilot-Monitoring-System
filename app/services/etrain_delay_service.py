"""
eTrain Delay Service - Fetches actual delay data from etrain.info

This service integrates with etrain.info to fetch actual train delay data
for determining real arrival/departure times (scheduled + delay).
"""

import logging
import re
import requests
from typing import Optional, Dict, List
from datetime import datetime
from functools import lru_cache
from cachetools import TTLCache

from ..utils.config import get_settings

logger = logging.getLogger(__name__)

# TTL cache for delay data (30 minutes default)
_delay_cache: Optional[TTLCache] = None


def _get_delay_cache() -> TTLCache:
    """Get or create the delay cache with configurable TTL."""
    global _delay_cache
    if _delay_cache is None:
        settings = get_settings()
        ttl = getattr(settings, 'etrain_cache_ttl', 1800)
        _delay_cache = TTLCache(maxsize=100, ttl=ttl)
    return _delay_cache


class StationDelayInfo:
    """Container for station delay information."""

    def __init__(
        self,
        station_code: str,
        station_name: str = "",
        scheduled_arrival: str = "",
        scheduled_departure: str = "",
        actual_arrival: str = "",
        actual_departure: str = "",
        delay_minutes: int = 0
    ):
        self.station_code = station_code
        self.station_name = station_name
        self.scheduled_arrival = scheduled_arrival
        self.scheduled_departure = scheduled_departure
        self.actual_arrival = actual_arrival
        self.actual_departure = actual_departure
        self.delay_minutes = delay_minutes

    def __repr__(self):
        return (
            f"StationDelayInfo({self.station_code}, "
            f"delay={self.delay_minutes}min)"
        )


class TrainDelayData:
    """Container for complete train delay data."""

    def __init__(
        self,
        train_number: str,
        journey_date: str,
        station_delays: Optional[List[StationDelayInfo]] = None,
        fetch_status: str = "unknown",
        fetch_error: Optional[str] = None
    ):
        self.train_number = train_number
        self.journey_date = journey_date
        self.station_delays = station_delays or []
        self.fetch_status = fetch_status
        self.fetch_error = fetch_error
        self.fetched_at = datetime.now()

    def get_delay_for_station(self, station_code: str) -> Optional[StationDelayInfo]:
        """Get delay info for a specific station."""
        for delay in self.station_delays:
            if delay.station_code.upper() == station_code.upper():
                return delay
        return None

    def get_average_delay(self) -> int:
        """Calculate average delay across all stations."""
        if not self.station_delays:
            return 0
        total_delay = sum(s.delay_minutes for s in self.station_delays)
        return total_delay // len(self.station_delays)

    def __repr__(self):
        return (
            f"TrainDelayData(train={self.train_number}, "
            f"stations={len(self.station_delays)}, "
            f"status={self.fetch_status})"
        )


class EtrainDelayService:
    """
    Service for fetching train delay data from etrain.info

    Features:
    - Web scraping of etrain.info live status pages
    - Parsing of delay information for each station
    - TTL caching to reduce scraping frequency
    - Graceful error handling with fallback to no delay
    """

    def __init__(self):
        """Initialize the etrain delay service."""
        self.settings = get_settings()
        self.enabled = getattr(self.settings, 'etrain_enabled', True)
        self.base_url = getattr(
            self.settings,
            'etrain_base_url',
            "https://etrain.info/train"
        )
        self.timeout = getattr(self.settings, 'trip_api_timeout', 10)

        logger.info(
            f"EtrainDelayService initialized - "
            f"enabled: {self.enabled}, "
            f"base_url: {self.base_url}"
        )

    def fetch_delay_data(
        self,
        train_number: str,
        journey_date: str
    ) -> TrainDelayData:
        """
        Fetch delay data from etrain.info for a train

        Args:
            train_number: 5-digit train number
            journey_date: Journey date in YYYY-MM-DD format

        Returns:
            TrainDelayData containing station delays
        """
        logger.info(
            f"[ETRAIN-DELAY] fetch_delay_data() called - "
            f"train: {train_number}, date: {journey_date}"
        )

        # Return empty data if disabled
        if not self.enabled:
            logger.info("[ETRAIN-DELAY] Service disabled, returning empty data")
            return TrainDelayData(
                train_number=train_number,
                journey_date=journey_date,
                fetch_status="disabled"
            )

        # Check cache first
        cache = _get_delay_cache()
        cache_key = f"{train_number}_{journey_date}"
        if cache_key in cache:
            cached = cache[cache_key]
            logger.info(
                f"[ETRAIN-DELAY] Cache HIT for {cache_key} - "
                f"returning cached data with {len(cached.station_delays)} stations"
            )
            return cached

        logger.info(f"[ETRAIN-DELAY] Cache MISS for {cache_key} - fetching from etrain.info")

        try:
            # Build URL for live running status
            # Format: https://etrain.info/train/17612/live
            url = f"{self.base_url}/{train_number}/live"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            }

            logger.info(f"[ETRAIN-DELAY] Fetching URL: {url}")

            response = requests.get(
                url,
                headers=headers,
                timeout=self.timeout
            )

            logger.info(
                f"[ETRAIN-DELAY] Response - status: {response.status_code}, "
                f"content_length: {len(response.content)} bytes"
            )

            if response.status_code == 200:
                delay_data = self._parse_delay_page(
                    response.text,
                    train_number,
                    journey_date
                )
                delay_data.fetch_status = "success"

                # Cache the result
                cache[cache_key] = delay_data
                logger.info(
                    f"[ETRAIN-DELAY] Parsed {len(delay_data.station_delays)} station delays, "
                    f"avg delay: {delay_data.get_average_delay()} min"
                )

                return delay_data
            else:
                logger.warning(
                    f"[ETRAIN-DELAY] HTTP error: {response.status_code}"
                )
                return TrainDelayData(
                    train_number=train_number,
                    journey_date=journey_date,
                    fetch_status="http_error",
                    fetch_error=f"HTTP {response.status_code}"
                )

        except requests.Timeout:
            logger.warning(
                f"[ETRAIN-DELAY] Timeout after {self.timeout}s"
            )
            return TrainDelayData(
                train_number=train_number,
                journey_date=journey_date,
                fetch_status="timeout",
                fetch_error="Request timed out"
            )
        except requests.RequestException as e:
            logger.warning(f"[ETRAIN-DELAY] Request error: {e}")
            return TrainDelayData(
                train_number=train_number,
                journey_date=journey_date,
                fetch_status="error",
                fetch_error=str(e)
            )
        except Exception as e:
            logger.error(f"[ETRAIN-DELAY] Unexpected error: {e}", exc_info=True)
            return TrainDelayData(
                train_number=train_number,
                journey_date=journey_date,
                fetch_status="error",
                fetch_error=str(e)
            )

    def _parse_delay_page(
        self,
        html_content: str,
        train_number: str,
        journey_date: str
    ) -> TrainDelayData:
        """
        Parse delay information from etrain.info HTML page

        Args:
            html_content: Raw HTML content from etrain.info
            train_number: Train number
            journey_date: Journey date

        Returns:
            TrainDelayData with parsed station delays
        """
        station_delays = []

        try:
            # Parse delay information using regex patterns
            # etrain.info shows delay in format like "12:30 (Late by 15 min)"
            # or "Arrived: 12:45 (Delay: 20 min)"

            # Pattern for station rows with delay info
            # Look for patterns like: StationCode ... Arr: HH:MM ... Delay: XX min
            station_pattern = re.compile(
                r'<td[^>]*>([A-Z]{2,5})</td>'  # Station code
                r'.*?'
                r'(?:Arr|Arrival)[:\s]*(\d{1,2}:\d{2})'  # Arrival time
                r'.*?'
                r'(?:Delay|Late)[:\s]*(\d+)\s*(?:min|mins|minutes)?',
                re.IGNORECASE | re.DOTALL
            )

            # Alternative pattern for delay shown separately
            delay_pattern = re.compile(
                r'(\d+)\s*(?:min|mins|minutes?)\s*(?:late|delay)',
                re.IGNORECASE
            )

            # Try to extract station-specific delays
            matches = station_pattern.findall(html_content)

            if matches:
                for match in matches:
                    station_code, arrival_time, delay_str = match
                    try:
                        delay_minutes = int(delay_str)
                        station_delays.append(StationDelayInfo(
                            station_code=station_code.upper(),
                            actual_arrival=arrival_time,
                            delay_minutes=delay_minutes
                        ))
                        logger.debug(
                            f"[ETRAIN-DELAY] Parsed: {station_code} - "
                            f"delay: {delay_minutes} min"
                        )
                    except ValueError:
                        continue

            # If no station-specific delays found, try to get overall delay
            if not station_delays:
                # Look for overall train delay
                overall_match = delay_pattern.search(html_content)
                if overall_match:
                    try:
                        overall_delay = int(overall_match.group(1))
                        logger.info(
                            f"[ETRAIN-DELAY] Found overall delay: {overall_delay} min"
                        )
                        # Create a placeholder with overall delay
                        station_delays.append(StationDelayInfo(
                            station_code="OVERALL",
                            delay_minutes=overall_delay
                        ))
                    except ValueError:
                        pass

            logger.info(
                f"[ETRAIN-DELAY] Parsed {len(station_delays)} delay entries"
            )

        except Exception as e:
            logger.warning(f"[ETRAIN-DELAY] Parse error: {e}")

        return TrainDelayData(
            train_number=train_number,
            journey_date=journey_date,
            station_delays=station_delays,
            fetch_status="parsed"
        )

    def get_adjusted_schedule(
        self,
        base_schedule,
        delay_data: TrainDelayData
    ):
        """
        Merge delay data into base schedule to create adjusted schedule

        Args:
            base_schedule: TripSchedule from RailRadar API
            delay_data: TrainDelayData from etrain.info

        Returns:
            TripSchedule with actual arrival/departure times adjusted for delays
        """
        if base_schedule is None:
            logger.warning("[ETRAIN-DELAY] No base schedule to adjust")
            return None

        if not delay_data.station_delays:
            logger.info("[ETRAIN-DELAY] No delay data to apply")
            return base_schedule

        logger.info(
            f"[ETRAIN-DELAY] Adjusting schedule with {len(delay_data.station_delays)} delay entries"
        )

        # Check for overall delay (applies to all stations)
        overall_delay = 0
        for delay_info in delay_data.station_delays:
            if delay_info.station_code == "OVERALL":
                overall_delay = delay_info.delay_minutes
                logger.info(f"[ETRAIN-DELAY] Applying overall delay: {overall_delay} min")
                break

        # Adjust each halt in the schedule
        for halt in base_schedule.halts:
            # Look for station-specific delay
            station_delay = delay_data.get_delay_for_station(halt.station_code)

            if station_delay:
                delay_mins = station_delay.delay_minutes
            else:
                delay_mins = overall_delay

            # Set delay fields (these will be used by motion resolver)
            if hasattr(halt, 'delay_minutes'):
                halt.delay_minutes = delay_mins
            if hasattr(halt, 'actual_arrival_minutes'):
                halt.actual_arrival_minutes = halt.arrival_minutes + delay_mins
            if hasattr(halt, 'actual_departure_minutes'):
                halt.actual_departure_minutes = halt.departure_minutes + delay_mins

            if delay_mins > 0:
                logger.debug(
                    f"[ETRAIN-DELAY] {halt.station_code}: "
                    f"scheduled_arr={halt.arrival_minutes}, "
                    f"delay={delay_mins}, "
                    f"actual_arr={halt.arrival_minutes + delay_mins}"
                )

        return base_schedule

    def clear_cache(self):
        """Clear the delay data cache."""
        cache = _get_delay_cache()
        cache.clear()
        logger.info("[ETRAIN-DELAY] Cache cleared")


# Global service instance
_etrain_delay_service: Optional[EtrainDelayService] = None


def get_etrain_delay_service() -> EtrainDelayService:
    """
    Get the global etrain delay service instance

    Returns:
        EtrainDelayService instance
    """
    global _etrain_delay_service
    if _etrain_delay_service is None:
        _etrain_delay_service = EtrainDelayService()
    return _etrain_delay_service

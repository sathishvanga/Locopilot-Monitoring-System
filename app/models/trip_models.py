"""
Trip models - Data models for train schedule tracking
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class StationHalt(BaseModel):
    """
    Model for a station halt in the train schedule

    Represents a single station stop with arrival/departure times.
    """
    station_code: str = Field(..., description="3-4 letter station identifier")
    station_name: str = Field(..., description="Full station name")
    scheduled_arrival: str = Field(..., description="Scheduled arrival time (HH:MM:SS)")
    scheduled_departure: str = Field(..., description="Scheduled departure time (HH:MM:SS)")
    halt_duration_minutes: int = Field(default=0, description="Halt duration at station in minutes")
    distance_km: float = Field(default=0.0, description="Distance from origin in kilometers")
    journey_day: int = Field(default=1, description="Day of journey (1 = start day, 2 = next day)")
    platform: Optional[str] = Field(None, description="Platform number if available")

    # Calculated fields for easy comparison
    arrival_minutes: int = Field(default=0, description="Arrival time in minutes from midnight")
    departure_minutes: int = Field(default=0, description="Departure time in minutes from midnight")

    # Delay-adjusted fields (from etrain.info)
    actual_arrival_minutes: Optional[int] = Field(None, description="Actual arrival time with delay applied")
    actual_departure_minutes: Optional[int] = Field(None, description="Actual departure time with delay applied")
    delay_minutes: int = Field(default=0, description="Delay in minutes from etrain.info")

    class Config:
        json_schema_extra = {
            "example": {
                "station_code": "SPJ",
                "station_name": "Samastipur Jn",
                "scheduled_arrival": "02:15:00",
                "scheduled_departure": "02:20:00",
                "halt_duration_minutes": 5,
                "distance_km": 58,
                "journey_day": 1,
                "platform": "2",
                "arrival_minutes": 135,
                "departure_minutes": 140
            }
        }


class TripSchedule(BaseModel):
    """
    Model for complete train trip schedule

    Contains all station halts for a train journey.
    """
    train_number: str = Field(..., description="5-digit train number")
    train_name: Optional[str] = Field(None, description="Train name")
    journey_date: str = Field(..., description="Journey date in YYYY-MM-DD format")
    origin_station: Optional[str] = Field(None, description="Origin station code")
    destination_station: Optional[str] = Field(None, description="Destination station code")
    halts: List[StationHalt] = Field(default_factory=list, description="List of station halts")
    total_distance_km: Optional[float] = Field(None, description="Total journey distance")
    fetched_at: datetime = Field(default_factory=datetime.now, description="Timestamp when schedule was fetched")

    def get_halt_at_time(self, time_str: str) -> Optional[StationHalt]:
        """
        Find the station halt that includes the given time

        Args:
            time_str: Time in HH:MM:SS format

        Returns:
            StationHalt if train is at a station at this time, None otherwise
        """
        try:
            parts = time_str.split(':')
            query_minutes = int(parts[0]) * 60 + int(parts[1])

            for halt in self.halts:
                # Check if time falls within halt period
                if halt.arrival_minutes <= query_minutes <= halt.departure_minutes:
                    return halt

            return None
        except (ValueError, IndexError):
            return None

    def get_next_halt(self, time_str: str) -> Optional[StationHalt]:
        """
        Find the next upcoming station halt after the given time

        Args:
            time_str: Time in HH:MM:SS format

        Returns:
            Next StationHalt if found, None otherwise
        """
        try:
            parts = time_str.split(':')
            query_minutes = int(parts[0]) * 60 + int(parts[1])

            for halt in self.halts:
                if halt.arrival_minutes > query_minutes:
                    return halt

            return None
        except (ValueError, IndexError):
            return None

    class Config:
        json_schema_extra = {
            "example": {
                "train_number": "12345",
                "train_name": "Express Mail",
                "journey_date": "2025-01-27",
                "origin_station": "NED",
                "destination_station": "HWH",
                "halts": [],
                "total_distance_km": 1500.0
            }
        }



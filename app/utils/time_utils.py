"""Shared time utility functions."""


def time_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS[.microseconds] to seconds.

    Args:
        time_str: Time string in HH:MM:SS or HH:MM:SS.ffffff format

    Returns:
        Total seconds as float
    """
    parts = time_str.split(':')
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

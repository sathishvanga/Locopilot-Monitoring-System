"""
External API Service - Handles posting results to external CVVR API

This service transforms internal activity data and posts it to the external
CVVR API endpoints for trip violation tracking.
"""

import os
import json
import re
import time
import threading
import random
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..utils.logger import get_logger
from ..utils.config import get_settings


logger = get_logger(__name__)
settings = get_settings()


class ExternalAPIService:
    """
    Service for posting activity results to external CVVR API

    Transforms internal activity format to external API payload format
    and handles HTTP communication with error recovery.
    """

    # Division validation pattern: alphanumeric, underscores, hyphens only
    _DIVISION_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')

    def __init__(self):
        """Initialize external API service"""
        self.settings = get_settings()
        self._dead_letter_queue: List[Dict[str, Any]] = []
        logger.info(
            f"[INFO] External API service initialized - "
            f"Enabled: {self.settings.cvvr_api_enabled}, "
            f"URL: {self.settings.cvvr_api_url}, "
            f"Timeout: {self.settings.cvvr_api_timeout}s"
        )

    def _validate_division(self, division: str) -> bool:
        """Validate division parameter contains only safe characters."""
        if not division:
            return False
        if not self._DIVISION_PATTERN.match(division):
            logger.error(
                f"[ERROR] [external_api] Invalid division parameter: {division!r}. "
                f"Only alphanumeric characters, underscores, and hyphens are allowed."
            )
            return False
        return True

    def _post_with_retry(
        self,
        url: str,
        json_payload: Any,
        headers: Dict[str, str],
        timeout: tuple,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        jitter: float = 0.5,
        context_label: str = "request"
    ) -> requests.Response:
        """
        Post with retry logic using exponential backoff and jitter.

        Retries on requests.ConnectionError and requests.Timeout.
        On final failure, stores the payload in the dead letter queue.

        Args:
            url: Target URL
            json_payload: JSON-serializable payload
            headers: HTTP headers
            timeout: Tuple of (connect_timeout, read_timeout)
            max_attempts: Maximum number of attempts
            base_delay: Base delay in seconds for exponential backoff
            max_delay: Maximum delay cap in seconds
            jitter: Maximum random jitter in seconds added to delay
            context_label: Label for log messages

        Returns:
            requests.Response on success

        Raises:
            The last exception if all retries are exhausted
        """
        last_exception = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(
                    url,
                    json=json_payload,
                    headers=headers,
                    timeout=timeout
                )
                return response
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exception = e
                if attempt < max_attempts:
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    delay += random.uniform(0, jitter)
                    logger.warning(
                        f"[WARN] [external_api] {context_label} attempt {attempt}/{max_attempts} "
                        f"failed ({type(e).__name__}), retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"[ERROR] [external_api] {context_label} failed after "
                        f"{max_attempts} attempts. Adding to dead letter queue."
                    )
                    self._dead_letter_queue.append({
                        "url": url,
                        "payload": json_payload,
                        "headers": {k: v for k, v in headers.items() if k != "Authorization"},
                        "error": str(e),
                        "timestamp": time.time(),
                        "context": context_label
                    })
                    raise

    def get_dead_letter_queue(self) -> List[Dict[str, Any]]:
        """Return the current dead letter queue contents."""
        return list(self._dead_letter_queue)

    def clear_dead_letter_queue(self) -> int:
        """Clear the dead letter queue and return the number of items removed."""
        count = len(self._dead_letter_queue)
        self._dead_letter_queue.clear()
        return count

    def post_cvvr_results(
        self,
        trip_id: str,
        events: List[Dict[str, Any]],
        job_id: Optional[str] = None,
        host_url: Optional[str] = None,
        video_s3_url: Optional[str] = None,
        division: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Post activity results to CVVR API

        Args:
            trip_id: Trip identifier
            events: List of detected activities
            job_id: Job/run identifier (optional, used for fileUrl construction)
            host_url: Host URL for building media links (optional, deprecated - use video_s3_url)
            video_s3_url: S3 URL of the uploaded video (preferred for fileUrl)
            division: Division identifier for API URL (optional, uses default if not provided)

        Returns:
            Dict with posting result (success status, response data, etc.)
        """
        # Check if external API is enabled
        if not self.settings.cvvr_api_enabled:
            logger.warning(
                f"[WARN] [external_api] CVVR API posting is disabled in configuration for trip_id={trip_id}"
            )
            return {
                "success": False,
                "message": "External API posting disabled",
                "posted": False
            }

        # Use configured host URL if not provided
        if not host_url:
            host_url = self.settings.host_url

        # Use default division if not provided
        if not division:
            division = self.settings.cvvr_api_default_division

        # Validate division parameter
        if not self._validate_division(division):
            return {
                "success": False,
                "message": f"Invalid division parameter: {division!r}",
                "posted": False
            }

        logger.info(
            f"[INFO] [external_api] Preparing to post results for trip_id={trip_id}, "
            f"events_count={len(events)}, job_id={job_id}, division={division}, "
            f"video_s3_url={'provided' if video_s3_url else 'not provided'}"
        )

        # If no events, post no-events notice
        if not events or len(events) == 0:
            return self._post_no_events(trip_id, job_id, division)

        # Transform events to violations
        violations = self._transform_events_to_violations(
            trip_id=trip_id,
            events=events,
            job_id=job_id,
            host_url=host_url,
            video_s3_url=video_s3_url
        )

        # Post violations to API
        return self._post_violations(trip_id, violations, job_id, division)

    def _post_no_events(self, trip_id: str, job_id: Optional[str] = None, division: Optional[str] = None) -> Dict[str, Any]:
        """
        Post no-events notice to CVVR API

        Args:
            trip_id: Trip identifier
            job_id: Job/run identifier (optional)
            division: Division identifier for API URL (optional)

        Returns:
            Dict with posting result
        """
        # Build URL with division
        division = division or self.settings.cvvr_api_default_division
        url_no_events = self.settings.cvvr_api_url_no_events.format(division=division)
        timeout = self.settings.cvvr_api_timeout

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
        }

        # Add authentication token if provided
        if self.settings.cvvr_api_token:
            headers["Authorization"] = f"Bearer {self.settings.cvvr_api_token}"

        # Prepare payload
        payload = {"tripId": trip_id}

        try:
            logger.info(
                f"[INFO] [external_api] Posting no-events notice to {url_no_events} "
                f"for trip_id={trip_id} (job_id={job_id})"
            )

            response = self._post_with_retry(
                url=url_no_events,
                json_payload=payload,
                headers=headers,
                timeout=(5, timeout),
                context_label=f"no-events for trip_id={trip_id}"
            )

            # Check response
            if response.status_code in [200, 201]:
                logger.info(
                    f"[OK] [external_api] No-events notice posted successfully: "
                    f"{response.status_code} - {response.text[:200]}"
                )
                # Safe JSON parsing
                try:
                    response_data = response.json() if response.text else {}
                except (json.JSONDecodeError, ValueError):
                    response_data = {"raw_text": response.text[:500]}
                return {
                    "success": True,
                    "message": "No-events notice posted successfully",
                    "posted": True,
                    "status_code": response.status_code,
                    "response": response_data
                }
            else:
                logger.warning(
                    f"[WARN] [external_api] No-events notice posting got non-2xx: "
                    f"{response.status_code} - {response.text[:200]}"
                )
                return {
                    "success": False,
                    "message": f"API returned status {response.status_code}",
                    "posted": False,
                    "status_code": response.status_code,
                    "response_text": response.text[:500]
                }

        except requests.exceptions.Timeout:
            logger.error(f"[ERROR] [external_api] No-events notice posting timed out after {timeout}s")
            return {
                "success": False,
                "message": "Request timed out",
                "posted": False,
                "error": "timeout"
            }

        except Exception as e:
            logger.error(f"[ERROR] [external_api] Failed to post no-events notice: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to post no-events notice: {str(e)}",
                "posted": False,
                "error": str(e)
            }

    def _post_violations(
        self,
        trip_id: str,
        violations: List[Dict[str, Any]],
        job_id: Optional[str] = None,
        division: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Post violations to CVVR API

        Args:
            trip_id: Trip identifier
            violations: List of transformed violation objects
            job_id: Job/run identifier (optional)
            division: Division identifier for API URL (optional)

        Returns:
            Dict with posting result
        """
        # Build URL with division
        division = division or self.settings.cvvr_api_default_division
        url = self.settings.cvvr_api_url.format(division=division)
        timeout = self.settings.cvvr_api_timeout

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
        }

        # Add authentication token if provided
        if self.settings.cvvr_api_token:
            headers["Authorization"] = f"Bearer {self.settings.cvvr_api_token}"

        # Deduplicate violations (same tripId + type + startTime)
        unique_violations = self._deduplicate_violations(violations)

        # Debug: log all fileUrls being sent to external API for this trip/job
        try:
            file_urls = [v.get("fileUrl") for v in unique_violations if v.get("fileUrl")]
            logger.info(
                f"[INFO] [external_api] fileUrls for trip_id={trip_id}, job_id={job_id}: "
                f"{file_urls if file_urls else 'NO fileUrls (clips) present'}"
            )
        except Exception as e:
            logger.warning(f"[WARN] [external_api] Failed to log fileUrls for trip_id={trip_id}: {e}")

        logger.info(
            f"[INFO] [external_api] Posting {len(unique_violations)} unique violations "
            f"(from {len(violations)} total) to {url} for trip_id={trip_id}"
        )

        try:
            # Send payload as array of violation objects with retry
            response = self._post_with_retry(
                url=url,
                json_payload=unique_violations,
                headers=headers,
                timeout=(5, timeout),
                context_label=f"violations for trip_id={trip_id}"
            )

            # Check response
            if response.status_code in [200, 201]:
                logger.info(
                    f"[OK] [external_api] Violations posted successfully: "
                    f"{response.status_code} - {response.text[:200]}"
                )
                # Safe JSON parsing
                try:
                    response_data = response.json() if response.text else {}
                except (json.JSONDecodeError, ValueError):
                    response_data = {"raw_text": response.text[:500]}
                return {
                    "success": True,
                    "message": f"Posted {len(unique_violations)} violations successfully",
                    "posted": True,
                    "violations_count": len(unique_violations),
                    "status_code": response.status_code,
                    "response": response_data
                }
            else:
                logger.warning(
                    f"[WARN] [external_api] Violations posting got non-2xx: "
                    f"{response.status_code} - {response.text[:200]}"
                )
                return {
                    "success": False,
                    "message": f"API returned status {response.status_code}",
                    "posted": False,
                    "violations_count": len(unique_violations),
                    "status_code": response.status_code,
                    "response_text": response.text[:500]
                }

        except requests.exceptions.Timeout:
            logger.error(f"[ERROR] [external_api] Violations posting timed out after {timeout}s")
            return {
                "success": False,
                "message": "Request timed out",
                "posted": False,
                "violations_count": len(unique_violations),
                "error": "timeout"
            }

        except Exception as e:
            logger.error(f"[ERROR] [external_api] Failed to post violations: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to post violations: {str(e)}",
                "posted": False,
                "violations_count": len(unique_violations),
                "error": str(e)
            }

    def _transform_events_to_violations(
        self,
        trip_id: str,
        events: List[Dict[str, Any]],
        job_id: Optional[str] = None,
        host_url: Optional[str] = None,
        video_s3_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Transform internal events to external API violation format

        Args:
            trip_id: Trip identifier
            events: List of internal activity events
            job_id: Job/run identifier (optional)
            host_url: Host URL for building fileUrl (deprecated - use video_s3_url)
            video_s3_url: S3 URL of the uploaded video (preferred for fileUrl)

        Returns:
            List of violation payloads ready for API
        """
        violations = []

        for event in events:
            try:
                violation = self._event_to_violation(
                    event=event,
                    trip_id=trip_id,
                    job_id=job_id,
                    host_url=host_url,
                    video_s3_url=video_s3_url
                )
                if violation:
                    violations.append(violation)
            except Exception as e:
                logger.warning(f"[WARN] [external_api] Failed to transform event: {e}")
                continue

        logger.info(f"[INFO] [external_api] Transformed {len(violations)} events to violations")
        return violations

    def _calculate_clip_duration(self, start_time: str, end_time: str) -> str:
        """
        Calculate clip duration from start and end time

        Args:
            start_time: Start time in HH:mm:ss format OR seconds (e.g., "6.00")
            end_time: End time in HH:mm:ss format OR seconds (e.g., "12.00")

        Returns:
            Duration in HH:mm:ss format
        """
        try:
            # Parse time strings to seconds
            def time_to_seconds(time_str: str) -> float:
                time_str = str(time_str).strip()

                # Check if it's in HH:mm:ss or mm:ss format
                if ":" in time_str:
                    parts = time_str.split(":")
                    if len(parts) == 3:
                        h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
                        return h * 3600 + m * 60 + s
                    elif len(parts) == 2:
                        m, s = float(parts[0]), float(parts[1])
                        return m * 60 + s
                    else:
                        return float(parts[0])
                else:
                    # It's already in seconds format (e.g., "6.00", "12.00")
                    return float(time_str)

            start_seconds = time_to_seconds(start_time)
            end_seconds = time_to_seconds(end_time)

            # Calculate duration
            duration_seconds = int(max(0, end_seconds - start_seconds))

            # Convert back to HH:mm:ss format
            hours = duration_seconds // 3600
            minutes = (duration_seconds % 3600) // 60
            seconds = duration_seconds % 60

            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except Exception as e:
            logger.warning(f"[WARN] [external_api] Failed to calculate clip duration: {e}")
            return "00:00:00"

    def _event_to_violation(
        self,
        event: Dict[str, Any],
        trip_id: str,
        job_id: Optional[str] = None,
        host_url: Optional[str] = None,
        video_s3_url: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a single event to violation payload

        Args:
            event: Internal activity event
            trip_id: Trip identifier
            job_id: Job/run identifier (optional)
            host_url: Host URL for building fileUrl (deprecated - use video_s3_url)
            video_s3_url: S3 URL of the uploaded video (preferred for fileUrl)

        Returns:
            Violation payload dict or None if transformation fails
        """
        try:
            # Check if this is a combined activity with multiple types
            activity_types = event.get("activityTypes", [])
            descriptions = event.get("descriptions", [])
            object_types_list = event.get("objectTypes", [])
            is_combined = len(activity_types) > 1

            # Extract event data - use arrays for combined, single values for legacy
            if is_combined:
                # Combined activity - use arrays
                types_value = activity_types
                descriptions_value = descriptions
                object_types_value = object_types_list
            else:
                # Single activity - use single values (wrapped in array for consistency)
                types_value = [event.get("activityType", 1)]
                descriptions_value = [event.get("des", "Unknown activity")]
                object_types_value = [event.get("objectType", "unknown")]

            start_time = event.get("activityStartTime", "0")
            end_time = event.get("activityEndTime", "0")
            filename = event.get("filename", "unknown.mp4")
            file_duration = event.get("fileDuration", "00:00:00")
            crew_name = event.get("crewName", "Unknown")
            activity_clip = event.get("activityClip", "")

            # Calculate clip duration from video timestamps (actual clip length)
            # Use videoStartTime/videoEndTime which represent actual video positions
            # Fallback to activityStartTime/activityEndTime if video timestamps not available
            video_start = event.get("videoStartTime", start_time)
            video_end = event.get("videoEndTime", end_time)
            clip_duration = self._calculate_clip_duration(video_start, video_end)

            # Determine fileUrl - prefer S3 URL, fallback to local backend URL
            file_url = ""

            # Priority 1: Use video S3 URL (preferred - this is the uploaded video)
            if video_s3_url:
                file_url = video_s3_url
                logger.debug(f"Using video S3 URL for fileUrl: {video_s3_url[:50]}...")
            # Priority 2: Use activityClip if it's already an S3 URL (from updated activities.json)
            elif activity_clip and (activity_clip.startswith("http://") or activity_clip.startswith("https://")):
                file_url = activity_clip
                logger.debug(f"Using activityClip S3 URL for fileUrl: {activity_clip[:50]}...")
            # Priority 3: Build local backend URL (fallback for backward compatibility)
            elif host_url and job_id and activity_clip:
                clip_name = os.path.basename(activity_clip)
                media_prefix = f"{host_url}/api/jobs/{job_id}/media"
                file_url = f"{media_prefix}/{clip_name}"
                logger.debug(f"Using local backend URL for fileUrl: {file_url}")

            # Build violation payload with arrays
            payload = {
                "tripId": trip_id,
                "types": types_value,  # Array of activity type codes
                "startTime": start_time,
                "endTime": end_time,
                "clipDuration": clip_duration,
                "remarks": "",
                "reason": "Automated detection",
                "descriptions": descriptions_value,  # Array of descriptions
                "objectTypes": object_types_value,  # Array of object types
                "fileName": filename,
                "fileDuration": file_duration,
                "crewName": crew_name,
                "fileType": 2,  # Default file type (2 = video)
                "fileUrl": file_url,
                "createdDate": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "createdBy": "system",
                "status": 1,  # Default status (1 = active/complete)
                "roleType": event.get("crewRole", 1),  # 1 = LP, 2 = ALP
            }

            return payload

        except Exception as e:
            logger.warning(f"[WARN] [external_api] Failed to convert event to violation: {e}")
            return None

    def _deduplicate_violations(self, violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate violations based on tripId, types, and startTime

        Args:
            violations: List of violation payloads

        Returns:
            List of unique violations
        """
        seen = set()
        unique = []

        for v in violations:
            # Create key for deduplication - convert types array to tuple for hashing
            types_value = v.get("types", [])
            types_tuple = tuple(types_value) if isinstance(types_value, list) else (types_value,)

            key = (
                v.get("tripId", ""),
                types_tuple,
                v.get("startTime", "")
            )

            if key not in seen:
                seen.add(key)
                unique.append(v)

        if len(unique) < len(violations):
            logger.info(
                f"[INFO] [external_api] Deduplicated {len(violations)} violations "
                f"to {len(unique)} unique violations"
            )

        return unique


# Singleton instance
_external_api_service = None
_external_api_service_lock = threading.Lock()


def get_external_api_service() -> ExternalAPIService:
    """
    Get singleton instance of external API service (thread-safe)

    Returns:
        ExternalAPIService: Singleton service instance
    """
    global _external_api_service
    if _external_api_service is None:
        with _external_api_service_lock:
            if _external_api_service is None:
                _external_api_service = ExternalAPIService()
    return _external_api_service

"""
External API Service - Handles posting results to external CVVR API

This service transforms internal activity data and posts it to the external
CVVR API endpoints for trip violation tracking.
"""

import os
import threading
import time
import requests
from typing import List, Dict, Any, Optional, Tuple
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
    
    # Retry configuration
    MAX_RETRIES = 3
    INITIAL_BACKOFF_SECONDS = 1.0
    BACKOFF_MULTIPLIER = 2.0
    RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(self):
        """Initialize external API service"""
        self.settings = get_settings()
        logger.info(
            f"🔌 External API service initialized - "
            f"Enabled: {self.settings.cvvr_api_enabled}, "
            f"URL: {self.settings.cvvr_api_url}, "
            f"Timeout: {self.settings.cvvr_api_timeout}s"
        )

    @staticmethod
    def _normalize_division(division: str) -> str:
        """Strip ai_ prefix and _api suffix if present to avoid double-wrapping.

        The URL template already wraps as ai_{division}_api, so if the caller
        passes 'ai_demo_api' it would become 'ai_ai_demo_api_api'.
        This normalizes to just 'demo' in that case.
        """
        if division.startswith("ai_"):
            division = division[3:]
        if division.endswith("_api"):
            division = division[:-4]
        return division

    def _request_with_retry(
        self,
        url: str,
        json_payload: Any,
        headers: Dict[str, str],
        timeout: int,
        context_label: str
    ) -> Tuple[Optional[requests.Response], Optional[str]]:
        """
        Execute an HTTP POST request with exponential backoff retry logic.

        Retries on retriable HTTP status codes (429, 500, 502, 503, 504),
        connection errors, and timeouts. Uses exponential backoff between
        attempts (1s, 2s, 4s by default).

        Args:
            url: The endpoint URL to POST to
            json_payload: JSON-serializable payload for the request body
            headers: HTTP headers dict
            timeout: Request timeout in seconds
            context_label: Human-readable label for log messages (e.g. "no-events notice", "violations")

        Returns:
            Tuple of (response, error_string). On success or non-retriable failure,
            response is set. On exhausted retries or exception, error_string describes
            the final failure.
        """
        last_exception = None
        last_response = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = requests.post(
                    url,
                    json=json_payload,
                    headers=headers,
                    timeout=timeout
                )
                last_response = response

                # If success or non-retriable status, return immediately
                if response.status_code not in self.RETRIABLE_STATUS_CODES:
                    return response, None

                # Retriable status code -- retry unless this was the last attempt
                if attempt < self.MAX_RETRIES:
                    backoff = self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_MULTIPLIER ** (attempt - 1))
                    logger.warning(
                        f"[external_api] {context_label} received retriable status "
                        f"{response.status_code} on attempt {attempt}/{self.MAX_RETRIES}. "
                        f"Retrying in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        f"[external_api] {context_label} received retriable status "
                        f"{response.status_code} on final attempt {attempt}/{self.MAX_RETRIES}. "
                        f"All retries exhausted."
                    )
                    return response, None

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_exception = e
                if attempt < self.MAX_RETRIES:
                    backoff = self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_MULTIPLIER ** (attempt - 1))
                    logger.warning(
                        f"[external_api] {context_label} attempt {attempt}/{self.MAX_RETRIES} "
                        f"failed with {type(e).__name__}: {e}. Retrying in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        f"[external_api] {context_label} attempt {attempt}/{self.MAX_RETRIES} "
                        f"failed with {type(e).__name__}: {e}. All retries exhausted. "
                        f"Data may need dead-letter queue recovery."
                    )

            except Exception as e:
                # Non-retriable exception (e.g. invalid JSON, programming error)
                logger.error(
                    f"[external_api] {context_label} failed with non-retriable error: {e}",
                    exc_info=True
                )
                return None, str(e)

        # All retries exhausted
        if last_exception:
            return None, str(last_exception)
        return last_response, None
    
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
                f"⚠️ [external_api] CVVR API posting is disabled in configuration for trip_id={trip_id}"
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
        # Strip ai_ prefix and _api suffix if present to avoid double-wrapping
        # (URL template already adds ai_{division}_api)
        division = self._normalize_division(division)

        logger.info(
            f"📤 [external_api] Preparing to post results for trip_id={trip_id}, "
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
        division = self._normalize_division(division or self.settings.cvvr_api_default_division)
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
        
        logger.info(
            f"📭 [external_api] Posting no-events notice to {url_no_events} "
            f"for trip_id={trip_id} (job_id={job_id})"
        )

        response, error = self._request_with_retry(
            url=url_no_events,
            json_payload=payload,
            headers=headers,
            timeout=timeout,
            context_label=f"No-events notice for trip_id={trip_id}"
        )

        # Handle complete failure (all retries exhausted with exceptions)
        if response is None:
            logger.error(
                f"❌ [external_api] Failed to post no-events notice after "
                f"{self.MAX_RETRIES} attempts: {error}. "
                f"Consider dead-letter queue recovery for trip_id={trip_id}."
            )
            return {
                "success": False,
                "message": f"Failed to post no-events notice: {error}",
                "posted": False,
                "error": error
            }

        # Check response status
        if response.status_code in [200, 201]:
            logger.info(
                f"✅ [external_api] No-events notice posted successfully: "
                f"{response.status_code} - {response.text[:200]}"
            )
            return {
                "success": True,
                "message": "No-events notice posted successfully",
                "posted": True,
                "status_code": response.status_code,
                "response": response.json() if response.text else {}
            }
        else:
            logger.warning(
                f"⚠️ [external_api] No-events notice posting got non-2xx: "
                f"{response.status_code} - {response.text[:200]}"
            )
            return {
                "success": False,
                "message": f"API returned status {response.status_code}",
                "posted": False,
                "status_code": response.status_code,
                "response_text": response.text[:500]
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
        division = self._normalize_division(division or self.settings.cvvr_api_default_division)
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
                f"📎 [external_api] fileUrls for trip_id={trip_id}, job_id={job_id}: "
                f"{file_urls if file_urls else 'NO fileUrls (clips) present'}"
            )
        except Exception as e:
            logger.warning(f"⚠️ [external_api] Failed to log fileUrls for trip_id={trip_id}: {e}")
        
        logger.info(
            f"📦 [external_api] Posting {len(unique_violations)} unique violations "
            f"(from {len(violations)} total) to {url} for trip_id={trip_id}"
        )
        
        response, error = self._request_with_retry(
            url=url,
            json_payload=unique_violations,
            headers=headers,
            timeout=timeout,
            context_label=f"Violations for trip_id={trip_id}"
        )

        # Handle complete failure (all retries exhausted with exceptions)
        if response is None:
            logger.error(
                f"❌ [external_api] Failed to post violations after "
                f"{self.MAX_RETRIES} attempts: {error}. "
                f"Consider dead-letter queue recovery for trip_id={trip_id}."
            )
            return {
                "success": False,
                "message": f"Failed to post violations: {error}",
                "posted": False,
                "violations_count": len(unique_violations),
                "error": error
            }

        # Check response status
        if response.status_code in [200, 201]:
            logger.info(
                f"✅ [external_api] Violations posted successfully: "
                f"{response.status_code} - {response.text[:200]}"
            )
            return {
                "success": True,
                "message": f"Posted {len(unique_violations)} violations successfully",
                "posted": True,
                "violations_count": len(unique_violations),
                "status_code": response.status_code,
                "response": response.json() if response.text else {}
            }
        else:
            logger.warning(
                f"⚠️ [external_api] Violations posting got non-2xx: "
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
                logger.warning(f"⚠️ [external_api] Failed to transform event: {e}")
                continue

        logger.info(f"🔄 [external_api] Transformed {len(violations)} events to violations")
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
            logger.warning(f"⚠️ [external_api] Failed to calculate clip duration: {e}")
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
                # motionState lets the customer distinguish station-context
                # activities (RUNNING|STOPPED|UNCERTAIN|UNKNOWN) from violations
                # that occurred while the train was moving. Only meaningful when
                # TRAIN_MOTION_SUPPRESS_WHEN_STOPPED=0 is set; otherwise stopped
                # events are filtered upstream and never reach this transform.
                "motionState": event.get("motionState", "UNKNOWN"),
            }

            return payload

        except Exception as e:
            logger.warning(f"[external_api] Failed to convert event to violation: {e}")
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
                f"🔍 [external_api] Deduplicated {len(violations)} violations "
                f"to {len(unique)} unique violations"
            )

        return unique


# Singleton instance
_external_api_service = None
_external_api_service_lock = threading.Lock()


def get_external_api_service() -> ExternalAPIService:
    """
    Get singleton instance of external API service.

    M-25: Thread-safe double-checked locking pattern.

    Returns:
        ExternalAPIService: Singleton service instance
    """
    global _external_api_service
    if _external_api_service is None:
        with _external_api_service_lock:
            if _external_api_service is None:
                _external_api_service = ExternalAPIService()
    return _external_api_service


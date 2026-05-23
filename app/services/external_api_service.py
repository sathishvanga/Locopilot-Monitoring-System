"""
External API Service - Handles posting results to external CVVR API

This service transforms internal activity data and posts it to the external
CVVR API endpoints for trip violation tracking.
"""

import hashlib
import json
import os
import threading
import time
import requests
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..utils.logger import get_logger
from ..utils.config import get_settings
from .dlq import write_dlq


logger = get_logger(__name__)
settings = get_settings()


def _time_to_seconds(time_str: Any) -> float:
    """Normalize a timestamp string to a float seconds value.

    Accepts both human-readable ``HH:mm:ss`` / ``mm:ss`` strings and bare
    seconds strings (``"6.00"``). The dedup key uses this so two payloads
    that describe the same instant in different notations collapse into a
    single violation rather than the customer seeing duplicates.
    """
    s = str(time_str).strip()
    try:
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 3:
                h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
                return h * 3600.0 + m * 60.0 + sec
            if len(parts) == 2:
                m, sec = float(parts[0]), float(parts[1])
                return m * 60.0 + sec
            return float(parts[0])
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _compute_idempotency_key(trip_id: str, payload: Any) -> str:
    """Stable, content-addressed key for one logical external-API call.

    Spec 0004 format: ``f"{trip_id}:{sha256(canonical_json(payload))}"``.
    The trip prefix keeps the key human-readable in logs (an operator can
    grep DLQ files for a specific trip) and the canonical-JSON digest of
    the payload makes it deterministic regardless of dict iteration order.

    Computed once per logical post so every retry attempt — including the
    post-DLQ drain replay — sends the exact same ``Idempotency-Key`` header.
    The customer API uses this header to drop a duplicate write that comes
    in after an accepted-but-timed-out response, which is the bug we're
    fixing.

    ``sort_keys=True`` makes the digest deterministic regardless of dict
    iteration order; ``default=str`` swallows numpy-float32 leakage that
    occasionally shows up in violation payloads.
    """
    canonical = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{trip_id}:{digest}"


class ExternalAPIService:
    """
    Service for posting activity results to external CVVR API
    
    Transforms internal activity format to external API payload format
    and handles HTTP communication with error recovery.
    """
    
    # Retry configuration. ``MAX_RETRIES = 4`` per spec 0004 — three
    # short-lived retries is too aggressive for a 5-minute upstream blip,
    # and the existing test scenarios labelled "3x 503 then 200" / "4x 503"
    # require four attempts to fire as documented.
    MAX_RETRIES = 4
    INITIAL_BACKOFF_SECONDS = 1.0
    BACKOFF_MULTIPLIER = 2.0
    RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(self):
        """Initialize external API service"""
        self.settings = get_settings()
        logger.info(
            f"[external_api] External API service initialized - "
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
        context_label: str,
        idempotency_key: Optional[str] = None,
    ) -> Tuple[Optional[requests.Response], Optional[str]]:
        """
        Execute an HTTP POST request with exponential backoff retry logic.

        Retries on retriable HTTP status codes (429, 500, 502, 503, 504),
        connection errors, and timeouts. Uses exponential backoff between
        attempts (1s, 2s, 4s by default).

        ``idempotency_key`` — when provided, attached as ``Idempotency-Key``
        on every attempt. Computed once per logical call by the caller so the
        same key is reused on every retry; this lets the customer API dedupe
        an accepted-but-timed-out response that we otherwise end up retrying.

        Args:
            url: The endpoint URL to POST to
            json_payload: JSON-serializable payload for the request body
            headers: HTTP headers dict
            timeout: Request timeout in seconds
            context_label: Human-readable label for log messages (e.g. "no-events notice", "violations")
            idempotency_key: Optional value for the ``Idempotency-Key`` header,
                reused identically across every retry attempt for the same call.

        Returns:
            Tuple of (response, error_string). On success or non-retriable failure,
            response is set. On exhausted retries or exception, error_string describes
            the final failure.
        """
        last_exception = None
        last_response = None

        # Build the per-call header set once. Mutating ``headers`` here would
        # bleed the Idempotency-Key into other calls if the caller reuses the
        # dict, so copy first and only then add the per-call header.
        request_headers = dict(headers or {})
        if idempotency_key:
            request_headers["Idempotency-Key"] = idempotency_key

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = requests.post(
                    url,
                    json=json_payload,
                    headers=request_headers,
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
        division: Optional[str] = None,
        run_dir: Optional[str] = None,
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
            run_dir: Per-run output directory. When provided, retries-exhausted
                payloads are persisted to ``<run_dir>/_failed_external_api/`` per
                spec 0004. Optional for backward compatibility — when absent,
                the DLQ write is skipped and the failure surfaces in the result
                dict only.

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
        # Strip ai_ prefix and _api suffix if present to avoid double-wrapping
        # (URL template already adds ai_{division}_api)
        division = self._normalize_division(division)

        logger.info(
            f"[external_api] Preparing to post results for trip_id={trip_id}, "
            f"events_count={len(events)}, job_id={job_id}, division={division}, "
            f"video_s3_url={'provided' if video_s3_url else 'not provided'}"
        )

        # If no events, post no-events notice
        if not events or len(events) == 0:
            return self._post_no_events(trip_id, job_id, division, run_dir=run_dir)

        # Transform events to violations
        violations = self._transform_events_to_violations(
            trip_id=trip_id,
            events=events,
            job_id=job_id,
            host_url=host_url,
            video_s3_url=video_s3_url
        )

        # Post violations to API
        return self._post_violations(trip_id, violations, job_id, division, run_dir=run_dir)
    
    def _post_no_events(
        self,
        trip_id: str,
        job_id: Optional[str] = None,
        division: Optional[str] = None,
        run_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Post no-events notice to CVVR API

        Args:
            trip_id: Trip identifier
            job_id: Job/run identifier (optional)
            division: Division identifier for API URL (optional)
            run_dir: Per-run output directory used for DLQ writes; optional.

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

        # Compute idempotency key once per logical call. Reused across retries
        # so the customer API can dedupe an accepted-but-timed-out response.
        idempotency_key = _compute_idempotency_key(trip_id, payload)

        logger.info(
            f"[external_api] Posting no-events notice to {url_no_events} "
            f"for trip_id={trip_id} (job_id={job_id}) idempotency_key={idempotency_key[:24]}"
        )

        response, error = self._request_with_retry(
            url=url_no_events,
            json_payload=payload,
            headers=headers,
            timeout=timeout,
            context_label=f"No-events notice for trip_id={trip_id}",
            idempotency_key=idempotency_key,
        )

        # Handle complete failure (all retries exhausted with exceptions)
        if response is None:
            logger.error(
                f"[FAIL] [external_api] Failed to post no-events notice after "
                f"{self.MAX_RETRIES} attempts: {error}. "
                f"Writing to dead-letter queue for trip_id={trip_id}."
            )
            dlq_path = self._dlq_write(
                run_dir=run_dir,
                url=url_no_events,
                payload=payload,
                headers=headers,
                idempotency_key=idempotency_key,
                context_label=f"No-events notice for trip_id={trip_id}",
                timeout=timeout,
                trip_id=trip_id,
                last_error=error,
                attempts=self.MAX_RETRIES,
            )
            return {
                "success": False,
                "message": f"Failed to post no-events notice: {error}",
                "posted": False,
                "error": error,
                "dlq_path": dlq_path,
            }

        # Check response status
        if response.status_code in [200, 201]:
            logger.info(
                f"[OK] [external_api] No-events notice posted successfully: "
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
                f"[WARN] [external_api] No-events notice posting got non-2xx: "
                f"{response.status_code} - {response.text[:200]}"
            )
            result = {
                "success": False,
                "message": f"API returned status {response.status_code}",
                "posted": False,
                "status_code": response.status_code,
                "response_text": response.text[:500]
            }
            # Retriable status that survived every retry counts as
            # "retries-exhausted" for DLQ purposes.
            if response.status_code in self.RETRIABLE_STATUS_CODES:
                result["dlq_path"] = self._dlq_write(
                    run_dir=run_dir,
                    url=url_no_events,
                    payload=payload,
                    headers=headers,
                    idempotency_key=idempotency_key,
                    context_label=f"No-events notice for trip_id={trip_id}",
                    timeout=timeout,
                    trip_id=trip_id,
                    last_error=f"status={response.status_code} body={response.text[:200]}",
                    attempts=self.MAX_RETRIES,
                )
            return result

    def _post_violations(
        self,
        trip_id: str,
        violations: List[Dict[str, Any]],
        job_id: Optional[str] = None,
        division: Optional[str] = None,
        run_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Post violations to CVVR API

        Args:
            trip_id: Trip identifier
            violations: List of transformed violation objects
            job_id: Job/run identifier (optional)
            division: Division identifier for API URL (optional)
            run_dir: Per-run output directory used for DLQ writes; optional.

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
                f"[external_api] fileUrls for trip_id={trip_id}, job_id={job_id}: "
                f"{file_urls if file_urls else 'NO fileUrls (clips) present'}"
            )
        except Exception as e:
            logger.warning(f"[WARN] [external_api] Failed to log fileUrls for trip_id={trip_id}: {e}")

        # Compute idempotency key once per logical call. Reused across retries
        # so the customer API can dedupe an accepted-but-timed-out response.
        idempotency_key = _compute_idempotency_key(trip_id, unique_violations)

        logger.info(
            f"[external_api] [PAYLOAD] Posting {len(unique_violations)} unique violations "
            f"(from {len(violations)} total) to {url} for trip_id={trip_id} "
            f"idempotency_key={idempotency_key[:24]}"
        )

        response, error = self._request_with_retry(
            url=url,
            json_payload=unique_violations,
            headers=headers,
            timeout=timeout,
            context_label=f"Violations for trip_id={trip_id}",
            idempotency_key=idempotency_key,
        )

        # Handle complete failure (all retries exhausted with exceptions)
        if response is None:
            logger.error(
                f"[FAIL] [external_api] Failed to post violations after "
                f"{self.MAX_RETRIES} attempts: {error}. "
                f"Writing to dead-letter queue for trip_id={trip_id}."
            )
            dlq_path = self._dlq_write(
                run_dir=run_dir,
                url=url,
                payload=unique_violations,
                headers=headers,
                idempotency_key=idempotency_key,
                context_label=f"Violations for trip_id={trip_id}",
                timeout=timeout,
                trip_id=trip_id,
                last_error=error,
                attempts=self.MAX_RETRIES,
            )
            return {
                "success": False,
                "message": f"Failed to post violations: {error}",
                "posted": False,
                "violations_count": len(unique_violations),
                "error": error,
                "dlq_path": dlq_path,
            }

        # Check response status
        if response.status_code in [200, 201]:
            logger.info(
                f"[OK] [external_api] Violations posted successfully: "
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
                f"[WARN] [external_api] Violations posting got non-2xx: "
                f"{response.status_code} - {response.text[:200]}"
            )
            result = {
                "success": False,
                "message": f"API returned status {response.status_code}",
                "posted": False,
                "violations_count": len(unique_violations),
                "status_code": response.status_code,
                "response_text": response.text[:500]
            }
            # Retriable status that survived every retry counts as
            # "retries-exhausted" for DLQ purposes.
            if response.status_code in self.RETRIABLE_STATUS_CODES:
                result["dlq_path"] = self._dlq_write(
                    run_dir=run_dir,
                    url=url,
                    payload=unique_violations,
                    headers=headers,
                    idempotency_key=idempotency_key,
                    context_label=f"Violations for trip_id={trip_id}",
                    timeout=timeout,
                    trip_id=trip_id,
                    last_error=f"status={response.status_code} body={response.text[:200]}",
                    attempts=self.MAX_RETRIES,
                )
            return result

    def _dlq_write(
        self,
        *,
        run_dir: Optional[str],
        url: str,
        payload: Any,
        headers: Dict[str, str],
        idempotency_key: str,
        context_label: str,
        timeout: int,
        trip_id: str,
        last_error: Optional[str],
        attempts: int,
    ) -> Optional[str]:
        """Wrapper around ``dlq.write_dlq`` that no-ops when ``run_dir`` is unset.

        Spec 0004 places DLQ files under ``<run_dir>/_failed_external_api/``,
        so a missing ``run_dir`` (older code paths, tests that don't simulate
        a run) means we cannot safely persist the payload — log and return
        ``None`` rather than fall back to a global drop dir, which the spec
        explicitly forbids.
        """
        if not run_dir:
            logger.warning(
                f"[external_api] No run_dir available for DLQ write "
                f"(trip_id={trip_id}); failure surfaced in result only"
            )
            return None
        return write_dlq(
            run_dir=run_dir,
            url=url,
            payload=payload,
            headers=headers,
            idempotency_key=idempotency_key,
            context_label=context_label,
            timeout=timeout,
            trip_id=trip_id,
            last_error=last_error,
            attempts=attempts,
        )
    
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

        logger.info(f"[external_api] Transformed {len(violations)} events to violations")
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
            # Reuse the module-level normalizer so dedup and duration math
            # parse the same notations identically.
            start_seconds = _time_to_seconds(start_time)
            end_seconds = _time_to_seconds(end_time)

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
        Deduplicate violations based on tripId, types, and startTime.

        ``startTime`` is normalized via ``_time_to_seconds`` before hashing
        so that ``"6.00"`` and ``"00:00:06"`` collapse into the same dedup
        key — otherwise a single activity that the writer emitted in two
        notations would post twice. The bucket value is rounded to two
        decimals (spec 0004) so float drift between the two representations
        does not defeat the dedup; two-decimal precision is also tolerant
        enough that a sub-frame timestamp jitter (~10ms) still hashes the
        same.

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

            # Normalize startTime so "6.00" and "00:00:06" hash identically.
            # Rounded to 2 decimals per spec 0004.
            start_seconds = round(_time_to_seconds(v.get("startTime", "")), 2)

            key = (
                v.get("tripId", ""),
                types_tuple,
                start_seconds,
            )

            if key not in seen:
                seen.add(key)
                unique.append(v)

        if len(unique) < len(violations):
            logger.info(
                f"[external_api] Deduplicated {len(violations)} violations "
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


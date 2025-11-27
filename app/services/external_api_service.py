"""
External API Service - Handles posting results to external CVVR API

This service transforms internal activity data and posts it to the external
CVVR API endpoints for trip violation tracking.
"""

import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..utils.logger import get_logger
from ..utils.config import get_settings
from ..exceptions import ExternalAPIError


logger = get_logger(__name__)
settings = get_settings()


class ExternalAPIService:
    """
    Service for posting activity results to external CVVR API
    
    Transforms internal activity format to external API payload format
    and handles HTTP communication with error recovery.
    """
    
    def __init__(self):
        """Initialize external API service"""
        self.settings = get_settings()
        logger.info(
            f"🔌 External API service initialized - "
            f"Enabled: {self.settings.cvvr_api_enabled}, "
            f"URL: {self.settings.cvvr_api_url}, "
            f"Timeout: {self.settings.cvvr_api_timeout}s"
        )
    
    def post_cvvr_results(
        self,
        trip_id: str,
        events: List[Dict[str, Any]],
        job_id: Optional[str] = None,
        host_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Post activity results to CVVR API
        
        Args:
            trip_id: Trip identifier
            events: List of detected activities
            job_id: Job/run identifier (optional, used for fileUrl construction)
            host_url: Host URL for building media links (optional)
            
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
        
        logger.info(
            f"📤 [external_api] Preparing to post results for trip_id={trip_id}, "
            f"events_count={len(events)}, job_id={job_id}, host_url={host_url}"
        )
        
        # If no events, post no-events notice
        if not events or len(events) == 0:
            return self._post_no_events(trip_id, job_id)
        
        # Transform events to violations
        violations = self._transform_events_to_violations(
            trip_id=trip_id,
            events=events,
            job_id=job_id,
            host_url=host_url
        )
        
        # Post violations to API
        return self._post_violations(trip_id, violations, job_id)
    
    def _post_no_events(self, trip_id: str, job_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Post no-events notice to CVVR API
        
        Args:
            trip_id: Trip identifier
            job_id: Job/run identifier (optional)
            
        Returns:
            Dict with posting result
        """
        url_no_events = self.settings.cvvr_api_url_no_events
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
                f"📭 [external_api] Posting no-events notice to {url_no_events} "
                f"for trip_id={trip_id} (job_id={job_id})"
            )
            
            response = requests.post(
                url_no_events,
                json=payload,
                headers=headers,
                timeout=timeout
            )
            
            # Check response
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
        
        except requests.exceptions.Timeout:
            logger.error(f"⏱️ [external_api] No-events notice posting timed out after {timeout}s")
            return {
                "success": False,
                "message": "Request timed out",
                "posted": False,
                "error": "timeout"
            }
        
        except Exception as e:
            logger.error(f"❌ [external_api] Failed to post no-events notice: {e}", exc_info=True)
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
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Post violations to CVVR API
        
        Args:
            trip_id: Trip identifier
            violations: List of transformed violation objects
            job_id: Job/run identifier (optional)
            
        Returns:
            Dict with posting result
        """
        url = self.settings.cvvr_api_url
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
        
        try:
            # Send payload as array of violation objects
            response = requests.post(
                url,
                json=unique_violations,
                headers=headers,
                timeout=timeout
            )
            
            # Check response
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
        
        except requests.exceptions.Timeout:
            logger.error(f"⏱️ [external_api] Violations posting timed out after {timeout}s")
            return {
                "success": False,
                "message": "Request timed out",
                "posted": False,
                "violations_count": len(unique_violations),
                "error": "timeout"
            }
        
        except Exception as e:
            logger.error(f"❌ [external_api] Failed to post violations: {e}", exc_info=True)
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
        host_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Transform internal events to external API violation format
        
        Args:
            trip_id: Trip identifier
            events: List of internal activity events
            job_id: Job/run identifier (optional)
            host_url: Host URL for building fileUrl
            
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
                    host_url=host_url
                )
                if violation:
                    violations.append(violation)
            except Exception as e:
                logger.warning(f"⚠️ [external_api] Failed to transform event: {e}")
                continue
        
        logger.info(f"🔄 [external_api] Transformed {len(violations)} events to violations")
        return violations
    
    def _event_to_violation(
        self,
        event: Dict[str, Any],
        trip_id: str,
        job_id: Optional[str] = None,
        host_url: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a single event to violation payload
        
        Args:
            event: Internal activity event
            trip_id: Trip identifier
            job_id: Job/run identifier (optional)
            host_url: Host URL for building fileUrl
            
        Returns:
            Violation payload dict or None if transformation fails
        """
        try:
            # Extract event data
            activity_type = event.get("activityType", 1)
            description = event.get("des", "Unknown activity")
            object_type = event.get("objectType", "unknown")
            start_time = event.get("activityStartTime", "0")
            end_time = event.get("activityEndTime", "0")
            filename = event.get("filename", "unknown.mp4")
            file_duration = event.get("fileDuration", "00:00:00")
            crew_name = event.get("crewName", "Unknown")
            activity_clip = event.get("activityClip", "")
            
            # Build fileUrl using the same pattern as the first project (POC_2):
            # {host_url}/api/jobs/{job_id}/media/{clip_filename}
            file_url = ""
            if host_url and job_id and activity_clip:
                clip_name = os.path.basename(activity_clip)
                media_prefix = f"{host_url}/api/jobs/{job_id}/media"
                file_url = f"{media_prefix}/{clip_name}"
            
            # Build violation payload
            payload = {
                "tripId": trip_id,
                "type": activity_type,
                "startTime": start_time,
                "endTime": end_time,
                "remarks": "Violation detected during trip processing",
                "reason": "Automated detection",
                "description": description,
                "objectTypes": object_type,
                "fileName": filename,
                "fileDuration": file_duration,
                "crewName": crew_name,
                "fileType": 2,  # Default file type (2 = video)
                "fileUrl": file_url,
                "createdDate": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "createdBy": "system",
                "status": 1,  # Default status (1 = active/complete)
            }
            
            return payload
        
        except Exception as e:
            logger.warning(f"[external_api] Failed to convert event to violation: {e}")
            return None
    
    def _deduplicate_violations(self, violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate violations based on tripId, type, and startTime
        
        Args:
            violations: List of violation payloads
            
        Returns:
            List of unique violations
        """
        seen = set()
        unique = []
        
        for v in violations:
            # Create key for deduplication
            key = (
                v.get("tripId", ""),
                v.get("type", 0),
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


def get_external_api_service() -> ExternalAPIService:
    """
    Get singleton instance of external API service
    
    Returns:
        ExternalAPIService: Singleton service instance
    """
    global _external_api_service
    if _external_api_service is None:
        _external_api_service = ExternalAPIService()
    return _external_api_service


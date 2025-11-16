# External API Integration - Implementation Summary

This document summarizes the external API integration for posting activity detection results to the CVVR API.

## Overview

After processing a video and detecting activities, the system now automatically posts the results to an external CVVR API endpoint. This integration is non-blocking and does not fail the processing job if the API call fails.

## Components Added/Modified

### 1. Configuration (`app/utils/config.py`)

Added the following configuration parameters:

```python
# External API settings (CVVR API)
cvvr_api_url: str = "https://api.mindcoinapps.com/ai_demo_api/cvvr/cvvrTripViolations/addUpdateBulk"
cvvr_api_url_no_events: str = "https://api.mindcoinapps.com/ai_demo_api/cvvr/cvvrTripViolations/addUpdateBulkNoEvents"
cvvr_api_token: Optional[str] = None  # Set via CVVR_API_TOKEN env var
cvvr_api_timeout: int = 30  # Request timeout in seconds
cvvr_api_enabled: bool = True  # Enable/disable external API posting
host_url: str = "https://celebxmedia.info"  # URL for building fileUrl
```

**Environment Variables:**
- `CVVR_API_URL` - Override the bulk violations endpoint
- `CVVR_API_URL_NO_EVENTS` - Override the no-events endpoint
- `CVVR_API_TOKEN` - Authentication token (Bearer token)
- `CVVR_API_TIMEOUT` - Request timeout in seconds (default: 30)
- `CVVR_API_ENABLED` - Enable/disable API posting (1=enabled, 0=disabled)
- `HOST_URL` - Base URL for constructing media file URLs

### 2. External API Service (`app/services/external_api_service.py`)

New service that handles all communication with the external CVVR API.

**Key Methods:**

#### `post_cvvr_results(trip_id, events, job_id, host_url)`
Main entry point for posting results.

**Arguments:**
- `trip_id`: Trip identifier
- `events`: List of detected activities (from activities.json)
- `job_id`: Run directory name (used for constructing fileUrl)
- `host_url`: Base URL for media files

**Behavior:**
- If no events detected → Posts to no-events endpoint
- If events detected → Transforms and posts to bulk violations endpoint
- Automatically deduplicates violations based on `(tripId, type, startTime)`

**Returns:**
```python
{
    "success": True/False,
    "message": "Success or error message",
    "posted": True/False,
    "violations_count": 5,  # Number of violations posted
    "status_code": 200,
    "response": {...}  # API response data
}
```

#### Internal Methods:

- `_post_no_events()` - Posts no-events notice
- `_post_violations()` - Posts violations array
- `_transform_events_to_violations()` - Transforms internal format to API format
- `_event_to_violation()` - Converts single event to violation payload
- `_deduplicate_violations()` - Removes duplicate violations

### 3. Video Processing Service (`app/services/video_processing_service.py`)

Modified to call the external API after saving activities.

**Changes in `process_video()` method:**

```python
# After saving activities.json
activities_json_path = self.activity_repository.save_activities(...)

# Post results to external API (non-blocking)
api_result = external_api_service.post_cvvr_results(
    trip_id=trip_id,
    events=activities,
    job_id=run_id,
    host_url=settings.host_url
)

# API result included in response
response["externalApiResult"] = api_result
```

**Error Handling:**
- Exceptions are caught and logged
- Processing job continues even if API call fails
- API result is included in the response for transparency

### 4. Dependencies (`requirements.txt`)

Added:
```
requests>=2.31.0  # For external API calls
```

## Payload Format

### Violation Object (sent to API)

```json
{
  "tripId": "TRIP-123",
  "type": 2,
  "startTime": "125.50",
  "endTime": "132.75",
  "remarks": "Violation detected during trip processing",
  "reason": "Automated detection",
  "description": "Using mobile phone",
  "objectTypes": "cell phone",
  "fileName": "latest.mp4",
  "fileDuration": "00:10:30",
  "crewName": "John Doe",
  "fileType": 2,
  "fileUrl": "https://celebxmedia.info/api/media/run_20251116_143045/clips/latest_cell_phone_frame00001250_001_clip.mp4",
  "createdDate": "2025-11-16T14:30:45",
  "createdBy": "system",
  "status": 1
}
```

### No-Events Payload

```json
{
  "tripId": "TRIP-123"
}
```

## HTTP Headers

```
Content-Type: application/json
Authorization: Bearer {CVVR_API_TOKEN}  # If token is configured
```

## API Endpoints

### 1. Bulk Violations Endpoint (when activities detected)

```
POST https://api.mindcoinapps.com/ai_demo_api/cvvr/cvvrTripViolations/addUpdateBulk
```

**Request Body:** Array of violation objects

**Expected Response:** HTTP 200/201

### 2. No-Events Endpoint (when no activities detected)

```
POST https://api.mindcoinapps.com/ai_demo_api/cvvr/cvvrTripViolations/addUpdateBulkNoEvents
```

**Request Body:** `{"tripId": "TRIP-123"}`

**Expected Response:** HTTP 200/201

## Configuration Options

### Enable/Disable External API Posting

Set environment variable:
```bash
export CVVR_API_ENABLED=0  # Disable
export CVVR_API_ENABLED=1  # Enable (default)
```

Or in code:
```python
settings.cvvr_api_enabled = False
```

### Configure Authentication

Set the bearer token:
```bash
export CVVR_API_TOKEN="your_token_here"
```

### Configure Timeout

```bash
export CVVR_API_TIMEOUT=60  # 60 seconds
```

### Override API URLs

```bash
export CVVR_API_URL="https://custom.api.com/violations"
export CVVR_API_URL_NO_EVENTS="https://custom.api.com/no-events"
```

### Configure Host URL (for fileUrl construction)

```bash
export HOST_URL="https://your-server.com"
```

## Testing

### Test with Mock Detection

```bash
curl -X POST "http://localhost:8000/api/jobs" \
  -F "video=@example_data/latest.mp4" \
  -F "tripId=TEST-001" \
  -F "lpCrewName=John Doe" \
  -F "lpCrewId=LP-001" \
  -F "useMockDetection=true"
```

Response will include:
```json
{
  "status": "success",
  "tripId": "TEST-001",
  "activitiesCount": 3,
  "externalApiResult": {
    "success": true,
    "message": "Posted 3 violations successfully",
    "posted": true,
    "violations_count": 3,
    "status_code": 200
  }
}
```

### Disable External API for Testing

```bash
export CVVR_API_ENABLED=0
# Then restart the server
```

## Logging

The external API service logs all operations:

```
[external_api] Preparing to post results for trip_id=TRIP-123, events_count=5, job_id=run_20251116_143045
[external_api] Posting 5 unique violations to https://api.mindcoinapps.com/...
[external_api] Successfully posted 5 violations to external API for trip TRIP-123
```

Error scenarios:
```
[external_api] CVVR API posting is disabled in configuration
[external_api] Violations posting timed out after 30s
[external_api] Failed to post violations: Connection refused
```

## Error Handling

### API Timeout
- Default: 30 seconds
- Configurable via `CVVR_API_TIMEOUT`
- Returns: `{"success": false, "error": "timeout"}`

### Non-200 Response
- Logs response status and body
- Returns: `{"success": false, "status_code": 400, "response_text": "..."}`

### Network Errors
- Logs exception with traceback
- Returns: `{"success": false, "error": "Connection refused"}`

### Processing Continues
- Job never fails due to external API errors
- All errors are logged for debugging
- API result is included in response for visibility

## Security Considerations

1. **Bearer Token**: Store in environment variable, never in code
2. **HTTPS**: API endpoints should use HTTPS in production
3. **Timeout**: Configure reasonable timeout to prevent hanging
4. **Non-blocking**: External API failures don't stop video processing
5. **Validation**: Input validation is performed before posting

## Future Enhancements

Potential improvements:

1. **Retry Logic**: Implement exponential backoff for failed requests
2. **Queue System**: Use message queue for reliable delivery
3. **Batch Processing**: Accumulate and batch multiple trips
4. **Webhook**: Support callback URLs for async processing
5. **Metrics**: Track success/failure rates and latency
6. **Rate Limiting**: Respect API rate limits
7. **Caching**: Cache API responses to avoid duplicate posts

## Troubleshooting

### API not posting

1. Check if enabled:
   ```bash
   echo $CVVR_API_ENABLED
   ```

2. Check logs for errors:
   ```bash
   grep "external_api" logs/locopilot.log
   ```

3. Test connectivity:
   ```bash
   curl -X POST https://api.mindcoinapps.com/ai_demo_api/cvvr/cvvrTripViolations/addUpdateBulk \
     -H "Content-Type: application/json" \
     -d '[]'
   ```

### Authentication failures

1. Verify token is set:
   ```bash
   echo $CVVR_API_TOKEN
   ```

2. Check token format in logs (redacted)

3. Test with curl:
   ```bash
   curl -X POST {URL} \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '[]'
   ```

### Timeout issues

1. Increase timeout:
   ```bash
   export CVVR_API_TIMEOUT=60
   ```

2. Check network latency to API endpoint

3. Monitor API response times

## Summary

The external API integration is:

✅ **Non-blocking** - Doesn't fail the processing job  
✅ **Configurable** - All settings via environment variables  
✅ **Secure** - Uses Bearer token authentication  
✅ **Robust** - Comprehensive error handling  
✅ **Transparent** - Results included in API response  
✅ **Production-ready** - Tested with real CVVR API endpoints  

The system continues to save results to `activities.json` locally while also posting to the external API, providing both local backup and remote integration.


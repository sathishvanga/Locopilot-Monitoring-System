# API Contract: POST `/api/video/analyze`

## Overview

Process uploaded video for locomotive pilot activity detection.

| Property | Value |
|----------|-------|
| **Base URL** | `http://103.195.244.66:8000` |
| **Endpoint** | `POST /api/video/analyze` |
| **Content-Type** | `multipart/form-data` |

---

## Request Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video` | File | Conditional | Video file to process (required if `videoUrl` not provided) |
| `videoUrl` | string | Conditional | MinIO URL to download video from (required if `video` not provided) |
| `tripId` | string | **Yes** | Unique trip identifier |
| `lpCrewName` | string | No | Loco Pilot crew member name |
| `lpCrewId` | string | No | Loco Pilot crew member ID |
| `alpCrewName` | string | No | Assistant Loco Pilot crew member name |
| `alpCrewId` | string | No | Assistant Loco Pilot crew member ID |
| `useMockDetection` | boolean | No | Use mock detection for testing (default: `false`) |
| `useMultiprocessing` | boolean | No | Enable parallel processing (default: from server config) |
| `saveClips` | boolean | No | Save annotated frames for debugging (default: `false`) |

> **Note:** Either `video` OR `videoUrl` must be provided, not both.

---

## Success Response (200)

```json
{
  "status": "success",
  "message": "Video processed successfully",
  "tripId": "TRIP-20251210-001",
  "videoFilename": "uploaded_video.mp4",
  "runDirectory": "/path/to/locopilot_evidence/run_20251210_143045",
  "activitiesJsonPath": "/path/to/locopilot_evidence/run_20251210_143045/activities.json",
  "activitiesCount": 5,
  "processingTime": 45.67,
  "activities": [
    {
      "tripId": "TRIP-20251210-001",
      "activityType": 2,
      "des": "Using mobile phone",
      "objectType": "cell phone",
      "fileUrl": "/path/to/video.mp4",
      "fileDuration": "00:10:30",
      "activityStartTime": "125.50",
      "activityEndTime": "132.75",
      "crewName": "John Doe",
      "crewId": "LP-001",
      "crewRole": 1,
      "performingRole": "LP",
      "date": "2025-12-10",
      "time": "14:30:45",
      "filename": "latest.mp4",
      "peopleCount": 1,
      "evidence": {
        "rule": "phone_in_hand"
      },
      "activityImage": "latest_cell_phone_frame00001250_001_activity.jpg",
      "activityClip": "latest_cell_phone_frame00001250_001_clip.mp4",
      "personRoles": [
        {
          "personIndex": 0,
          "role": "LP",
          "roleName": "Loco Pilot",
          "lpScore": 5,
          "alpScore": 1
        }
      ]
    }
  ]
}
```

---

## Response Fields

### Root Level

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Processing status (`success`, `error`) |
| `message` | string | Human-readable status message |
| `tripId` | string | Trip identifier from request |
| `videoFilename` | string | Uploaded video filename |
| `runDirectory` | string | Output directory path for this processing run |
| `activitiesJsonPath` | string | Path to activities.json file |
| `activitiesCount` | integer | Total number of activities detected |
| `processingTime` | float | Total processing time in seconds |
| `activities` | array | List of detected activities |

### Activity Object

| Field | Type | Description |
|-------|------|-------------|
| `tripId` | string | Unique trip identifier |
| `activityType` | integer | Activity type code (see Activity Types table) |
| `des` | string | Human-readable activity description |
| `objectType` | string | Object type involved in activity |
| `fileUrl` | string | Absolute path to source video file |
| `fileDuration` | string | Total video duration (HH:MM:SS) |
| `activityStartTime` | string | Activity start time in seconds |
| `activityEndTime` | string | Activity end time in seconds |
| `crewName` | string | Crew member name who performed the activity |
| `crewId` | string | Crew member ID who performed the activity |
| `crewRole` | integer | Crew role (1 = LP, 2 = ALP) |
| `performingRole` | string | Role of crew member (LP or ALP) |
| `date` | string | Date of activity (YYYY-MM-DD) |
| `time` | string | Time of activity (HH:MM:SS) |
| `filename` | string | Source video filename |
| `peopleCount` | integer | Number of people detected |
| `evidence` | object | Evidence details (`{ "rule": "string" }`) |
| `activityImage` | string | Activity screenshot filename |
| `activityClip` | string | Activity video clip filename |
| `personRoles` | array | List of person roles identified |

### PersonRole Object

| Field | Type | Description |
|-------|------|-------------|
| `personIndex` | integer | Index of the person (0, 1, 2, ...) |
| `role` | string | Role code (LP, ALP, SUPERVISOR, TRAINEE, VISITOR) |
| `roleName` | string | Human-readable role name |
| `lpScore` | integer | Loco Pilot score based on detected objects |
| `alpScore` | integer | Assistant Loco Pilot score based on detected objects |

---

## Activity Types

| Code | Name | Description |
|------|------|-------------|
| 1 | UNKNOWN | Unknown activity |
| 2 | CELL_PHONE | Using mobile phone |
| 3 | MICROSLEEP | Short sleep/drowsiness |
| 4 | SLEEP | Extended sleep detected |
| 5 | WRITING | Writing activity |
| 6 | PACKING_BAGS | Packing bags activity |
| 7 | GROUP_DETECTED | Group of people detected |
| 8 | LP_NOT_EXCHANGING_HAND_GESTURE | LP not exchanging hand gesture |
| 9 | ALP_NOT_EXCHANGING_HAND_GESTURE | ALP not exchanging hand gesture |
| 10 | MIND_DIVERSION | Mind diversion detected |
| 11 | NO_PERSON_DETECTED | No person detected in frame |

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "tripId is required and cannot be empty"
}
```

```json
{
  "detail": "Either 'video' file or 'videoUrl' must be provided"
}
```

```json
{
  "detail": "Failed to download video from MinIO: <error_message>"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Failed to process video: <error_message>"
}
```

---

## Example Requests

### With File Upload

```bash
curl -X POST "http://103.195.244.66:8000/api/video/analyze" \
  -F "video=@/path/to/video.mp4" \
  -F "tripId=TRIP-20251210-001" \
  -F "lpCrewName=John Doe" \
  -F "lpCrewId=LP-001" \
  -F "alpCrewName=Jane Smith" \
  -F "alpCrewId=ALP-002"
```

### With MinIO URL

```bash
curl -X POST "http://103.195.244.66:8000/api/video/analyze" \
  -F "videoUrl=https://mind.snikbtel.uk:9000/cvss/video.mp4" \
  -F "tripId=TRIP-20251210-001" \
  -F "lpCrewName=John Doe" \
  -F "lpCrewId=LP-001"
```

---

## Media Files Access

Evidence clips and images can be accessed via:

```
GET http://103.195.244.66:8000/api/jobs/{run_id}/media/{filename}
```

**Example:**

```
GET http://103.195.244.66:8000/api/jobs/run_20251210_143045/media/latest_cell_phone_frame00001250_001_clip.mp4
```

Supports HTTP Range headers for video streaming.

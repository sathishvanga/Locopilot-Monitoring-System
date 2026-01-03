# API Contract: Activity Detection Response Format v2

## Overview
Activities with overlapping time ranges are now combined into single records with array fields. This reduces redundancy when multiple violations occur simultaneously (e.g., "using mobile" + "LP not exchanging hand" at the same time).

**Version:** 2.0
**Last Updated:** 2025-01-03
**Server:** https://celebxmedia.info

---

## Response Structure

### Violation Object Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tripId` | `string` | Yes | Unique trip identifier |
| `types` | `number[]` | Yes | **Array** of activity type codes |
| `descriptions` | `string[]` | Yes | **Array** of activity descriptions |
| `objectTypes` | `string[]` | Yes | **Array** of detected object types |
| `startTime` | `string` | Yes | Start time (seconds as string, e.g., "5.00") |
| `endTime` | `string` | Yes | End time (seconds as string, e.g., "10.00") |
| `clipDuration` | `string` | Yes | Duration in HH:MM:SS format |
| `fileName` | `string` | Yes | Original video filename |
| `fileDuration` | `string` | Yes | Total video duration (HH:MM:SS) |
| `fileUrl` | `string` | Yes | URL to evidence clip (S3 or local) |
| `crewName` | `string` | Yes | Crew member name |
| `roleType` | `number` | Yes | 1 = LP, 2 = ALP |
| `fileType` | `number` | Yes | Always 2 (video) |
| `status` | `number` | Yes | Always 1 (active) |
| `createdDate` | `string` | Yes | ISO timestamp (YYYY-MM-DDTHH:MM:SS) |
| `createdBy` | `string` | Yes | Always "system" |
| `reason` | `string` | Yes | Always "Automated detection" |
| `remarks` | `string` | Yes | Empty string |

---

## Activity Type Codes

| Code | Key | Description |
|------|-----|-------------|
| 2 | `cell_phone` | Using mobile phone |
| 3 | `microsleep` | Micro-sleep detected (5+ seconds) |
| 4 | `sleep` | Sleep detected (30+ seconds) |
| 5 | `writing` | Writing log book while running |
| 6 | `packing_bags` | Packing bags activity detected |
| 7 | `group_detected` | More than 2 people detected |
| 8 | `lp_hand_gesture` | LP not exchanging hand gesture |
| 9 | `alp_hand_gesture` | ALP not exchanging hand gesture |
| 10 | `mind_diversion` | Mind diversion - attention diverted |
| 11 | `no_person_detected` | No person detected in frame |

---

## Examples

### Single Activity (one violation type)
```json
{
  "tripId": "string",
  "types": [6],
  "descriptions": ["Packing bags activity detected"],
  "objectTypes": ["packing bags"],
  "startTime": "11:02:57",
  "endTime": "11:03:05",
  "clipDuration": "00:00:08",
  "fileName": "string_1767416065.mp4",
  "fileDuration": "0:07:33",
  "fileUrl": "https://celebxmedia.info/api/jobs/run_20260103_055425/media/string_1767416065_packing_bags_frame00009250_000_clip.mp4",
  "crewName": "string",
  "roleType": 1,
  "fileType": 2,
  "status": 1,
  "createdDate": "2026-01-03T05:56:09",
  "createdBy": "system",
  "reason": "Automated detection",
  "remarks": ""
}
```

### Combined Activity (multiple violation types at same time)
```json
{
  "tripId": "string",
  "types": [2, 6, 9],
  "descriptions": [
    "Using mobile phone",
    "Packing bags activity detected",
    "ALP not exchanging hand gesture"
  ],
  "objectTypes": ["cell phone", "packing bags", "alp hand gesture"],
  "startTime": "41111.00",
  "endTime": "41115.00",
  "clipDuration": "00:00:04",
  "fileName": "string_1767416065.mp4",
  "fileDuration": "0:07:33",
  "fileUrl": "https://celebxmedia.info/api/jobs/run_20260103_055425/media/string_1767416065_alp_hand_gesture_frame00000600_000_clip.mp4",
  "crewName": "string",
  "roleType": 1,
  "fileType": 2,
  "status": 1,
  "createdDate": "2026-01-03T05:56:09",
  "createdBy": "system",
  "reason": "Automated detection",
  "remarks": ""
}
```

---

## UI Integration Notes

### 1. Always Iterate Arrays
`types`, `descriptions`, and `objectTypes` are **always arrays** (even for single activities).

```javascript
// Check array length to determine if combined
const isCombined = violation.types.length > 1;
```

### 2. Array Indices Are Aligned
`types[i]` corresponds to `descriptions[i]` and `objectTypes[i]`.

```javascript
violation.types.forEach((type, index) => {
  console.log(`Type: ${type}`);
  console.log(`Description: ${violation.descriptions[index]}`);
  console.log(`Object: ${violation.objectTypes[index]}`);
});
```

### 3. Badge/Tag Display
For combined activities, show multiple badges:
```
[Mobile Phone] [Hand Gesture]
05:00 - 10:00 | LP - John Doe
```

### 4. Filtering by Activity Type
When filtering by activity type, check if `types` array includes the filter value:

```javascript
// Filter violations that include a specific type
const filtered = violations.filter(v => v.types.includes(filterType));

// Filter violations that include ANY of selected types
const filtered = violations.filter(v =>
  selectedTypes.some(type => v.types.includes(type))
);

// Filter violations that include ALL of selected types
const filtered = violations.filter(v =>
  selectedTypes.every(type => v.types.includes(type))
);
```

### 5. Counting Violations
When counting by type, consider that one violation can have multiple types:

```javascript
// Count occurrences of each type
const typeCounts = {};
violations.forEach(v => {
  v.types.forEach(type => {
    typeCounts[type] = (typeCounts[type] || 0) + 1;
  });
});
```

---

## Migration from Old Format

| Old Field | New Field | Change |
|-----------|-----------|--------|
| `type` (number) | `types` (number[]) | Now array |
| `description` (string) | `descriptions` (string[]) | Now array |
| `objectTypes` (string) | `objectTypes` (string[]) | Now array |

### Backward Compatibility Helper
```javascript
// Helper to handle both old and new format
function getTypes(violation) {
  return violation.types || [violation.type];
}

function getDescriptions(violation) {
  return violation.descriptions || [violation.description];
}

function getObjectTypes(violation) {
  if (Array.isArray(violation.objectTypes)) {
    return violation.objectTypes;
  }
  return [violation.objectTypes];
}
```

---

## Sample API Response (Full)

```json
{
  "status": "success",
  "message": "Video processed successfully",
  "tripId": "string",
  "videoFilename": "string_1767416065.mp4",
  "runDirectory": "/opt/poc2/locopilot_evidence/run_20260103_055425",
  "activitiesJsonPath": "/opt/poc2/locopilot_evidence/run_20260103_055425/activities.json",
  "activitiesCount": 7,
  "violations": [
    {
      "tripId": "string",
      "types": [6],
      "descriptions": ["Packing bags activity detected"],
      "objectTypes": ["packing bags"],
      "startTime": "11:02:57",
      "endTime": "11:03:05",
      "clipDuration": "00:00:08",
      "fileName": "string_1767416065.mp4",
      "fileDuration": "0:07:33",
      "fileUrl": "https://celebxmedia.info/api/jobs/run_20260103_055425/media/string_1767416065_packing_bags_frame00009250_000_clip.mp4",
      "crewName": "string",
      "roleType": 1,
      "fileType": 2,
      "status": 1,
      "createdDate": "2026-01-03T05:56:09",
      "createdBy": "system",
      "reason": "Automated detection",
      "remarks": ""
    },
    {
      "tripId": "string",
      "types": [9],
      "descriptions": ["ALP not exchanging hand gesture"],
      "objectTypes": ["alp hand gesture"],
      "startTime": "11:05:11",
      "endTime": "11:05:19",
      "clipDuration": "00:00:08",
      "fileName": "string_1767416065.mp4",
      "fileDuration": "0:07:33",
      "fileUrl": "https://celebxmedia.info/api/jobs/run_20260103_055425/media/string_1767416065_alp_hand_gesture_frame00004600_000_clip.mp4",
      "crewName": "string",
      "roleType": 1,
      "fileType": 2,
      "status": 1,
      "createdDate": "2026-01-03T05:56:09",
      "createdBy": "system",
      "reason": "Automated detection",
      "remarks": ""
    },
    {
      "tripId": "string",
      "types": [10],
      "descriptions": ["Mind diversion - attention diverted from controls"],
      "objectTypes": ["mind diversion"],
      "startTime": "11:11:51",
      "endTime": "11:11:59",
      "clipDuration": "00:00:08",
      "fileName": "string_1767416065.mp4",
      "fileDuration": "0:07:33",
      "fileUrl": "https://celebxmedia.info/api/jobs/run_20260103_055425/media/string_1767416065_mind_diversion_frame00011100_000_clip.mp4",
      "crewName": "string",
      "roleType": 1,
      "fileType": 2,
      "status": 1,
      "createdDate": "2026-01-03T05:56:09",
      "createdBy": "system",
      "reason": "Automated detection",
      "remarks": ""
    },
    {
      "tripId": "string",
      "types": [6],
      "descriptions": ["Packing bags activity detected"],
      "objectTypes": ["packing bags"],
      "startTime": "11:12:50",
      "endTime": "11:12:56",
      "clipDuration": "00:00:06",
      "fileName": "string_1767416065.mp4",
      "fileDuration": "0:07:33",
      "fileUrl": "https://celebxmedia.info/api/jobs/run_20260103_055425/media/string_1767416065_packing_bags_frame00009500_000_clip.mp4",
      "crewName": "string",
      "roleType": 1,
      "fileType": 2,
      "status": 1,
      "createdDate": "2026-01-03T05:56:09",
      "createdBy": "system",
      "reason": "Automated detection",
      "remarks": ""
    },
    {
      "tripId": "string",
      "types": [2, 6, 9],
      "descriptions": [
        "Using mobile phone",
        "Packing bags activity detected",
        "ALP not exchanging hand gesture"
      ],
      "objectTypes": ["cell phone", "packing bags", "alp hand gesture"],
      "startTime": "41111.00",
      "endTime": "41115.00",
      "clipDuration": "00:00:04",
      "fileName": "string_1767416065.mp4",
      "fileDuration": "0:07:33",
      "fileUrl": "https://celebxmedia.info/api/jobs/run_20260103_055425/media/string_1767416065_alp_hand_gesture_frame00000600_000_clip.mp4",
      "crewName": "string",
      "roleType": 1,
      "fileType": 2,
      "status": 1,
      "createdDate": "2026-01-03T05:56:09",
      "createdBy": "system",
      "reason": "Automated detection",
      "remarks": ""
    },
    {
      "tripId": "string",
      "types": [10],
      "descriptions": ["Mind diversion - attention diverted from controls"],
      "objectTypes": ["mind diversion"],
      "startTime": "11:41:11",
      "endTime": "11:41:17",
      "clipDuration": "00:00:06",
      "fileName": "string_1767416065.mp4",
      "fileDuration": "0:07:33",
      "fileUrl": "https://celebxmedia.info/api/jobs/run_20260103_055425/media/string_1767416065_mind_diversion_frame00003300_000_clip.mp4",
      "crewName": "string",
      "roleType": 1,
      "fileType": 2,
      "status": 1,
      "createdDate": "2026-01-03T05:56:09",
      "createdBy": "system",
      "reason": "Automated detection",
      "remarks": ""
    },
    {
      "tripId": "string",
      "types": [6],
      "descriptions": ["Packing bags activity detected"],
      "objectTypes": ["packing bags"],
      "startTime": "42171.00",
      "endTime": "42177.00",
      "clipDuration": "00:00:06",
      "fileName": "string_1767416065.mp4",
      "fileDuration": "0:07:33",
      "fileUrl": "https://celebxmedia.info/api/jobs/run_20260103_055425/media/string_1767416065_packing_bags_merged_42171s-42177s.mp4",
      "crewName": "string",
      "roleType": 1,
      "fileType": 2,
      "status": 1,
      "createdDate": "2026-01-03T05:56:09",
      "createdBy": "system",
      "reason": "Automated detection",
      "remarks": ""
    }
  ],
  "processingTime": 103.99
}
```

---

## TypeScript Interface

```typescript
interface Violation {
  tripId: string;
  types: number[];
  descriptions: string[];
  objectTypes: string[];
  startTime: string;
  endTime: string;
  clipDuration: string;
  fileName: string;
  fileDuration: string;
  fileUrl: string;
  crewName: string;
  roleType: 1 | 2;  // 1 = LP, 2 = ALP
  fileType: number;
  status: number;
  createdDate: string;
  createdBy: string;
  reason: string;
  remarks: string;
}

interface VideoProcessingResponse {
  status: 'success' | 'error' | 'processing';
  message: string;
  tripId: string;
  videoFilename: string;
  runDirectory: string;
  activitiesJsonPath: string;
  activitiesCount: number;
  violations: Violation[];
  processingTime: number;
}
```

---

## Changelog

### v2.0 (2025-01-03)
- **BREAKING:** Changed `type` to `types` (array) to support combined activities
- **BREAKING:** Changed `description` to `descriptions` (array)
- **BREAKING:** Changed `objectTypes` from string to string array
- Activities with overlapping time ranges are now grouped into single records
- Added TypeScript interface definitions
- Added UI integration examples for filtering and counting

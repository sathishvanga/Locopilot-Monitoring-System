# API Endpoints for UI Team - 1GB Video Upload Feature

## Base URL
```
http://103.195.244.67/api
```

## API Documentation (Swagger UI)
```
http://103.195.244.67/docs
```
**Use this to test endpoints interactively!**

---

## V2 Endpoints - 1GB Upload Support

### Option 1: Streaming Upload (Recommended for Simple Cases)

**Single Request Upload** - Streams video directly to disk (memory-efficient)

```
POST /api/v2/jobs/streaming
```

**Request Format:**
- Content-Type: `multipart/form-data`
- Method: `POST`

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video` | File | Yes | Video file (max 1GB) |
| `tripId` | String | Yes | Unique trip identifier |
| `lpCrewName` | String | No | LP crew member name |
| `lpCrewId` | String | No | LP crew member ID |
| `alpCrewName` | String | No | ALP crew member name |
| `alpCrewId` | String | No | ALP crew member ID |
| `useMockDetection` | Boolean | No | Use mock detection (default: false) |
| `useMultiprocessing` | Boolean | No | Enable parallel processing |
| `saveClips` | Boolean | No | Save annotated frames (default: false) |

**Example (cURL):**
```bash
curl -X POST "http://103.195.244.67/api/v2/jobs/streaming" \
  -F "video=@/path/to/video.mp4" \
  -F "tripId=trip123" \
  -F "lpCrewName=John Doe" \
  -F "lpCrewId=LP001"
```

**Example (JavaScript/Fetch):**
```javascript
const formData = new FormData();
formData.append('video', videoFile);
formData.append('tripId', 'trip123');
formData.append('lpCrewName', 'John Doe');
formData.append('lpCrewId', 'LP001');

const response = await fetch('http://103.195.244.67/api/v2/jobs/streaming', {
  method: 'POST',
  body: formData,
  // Note: Don't set Content-Type header, browser will set it with boundary
});

const result = await response.json();
```

**Response:**
```json
{
  "status": "success",
  "trip_id": "trip123",
  "activities": [...],
  "output_path": "/opt/poc2/output/...",
  "summary": {...}
}
```

**Error Responses:**
- `413`: File exceeds 1GB limit
- `507`: Insufficient disk space
- `400`: Invalid request or file
- `500`: Server error

---

### Option 2: Chunked Upload (Recommended for Large Files & Unreliable Networks)

**3-Step Resumable Upload Process**

#### Step 1: Initiate Upload Session

```
POST /api/v2/upload/initiate
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `filename` | String | Yes | Original filename (e.g., "video.mp4") |
| `total_size` | Integer | Yes | Total file size in bytes |
| `tripId` | String | Yes | Unique trip identifier |

**Response:**
```json
{
  "upload_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunk_size_recommendation": 10485760,
  "total_chunks": 100,
  "expires_at": "2025-11-30T02:00:00"
}
```

**Example:**
```javascript
const response = await fetch('http://103.195.244.67/api/v2/upload/initiate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  body: new URLSearchParams({
    filename: 'video.mp4',
    total_size: file.size,
    tripId: 'trip123'
  })
});

const { upload_id, chunk_size_recommendation, total_chunks } = await response.json();
```

---

#### Step 2: Upload Chunks

```
POST /api/v2/upload/chunk
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `upload_id` | String | Yes | Upload session ID from Step 1 |
| `part_number` | Integer | Yes | Chunk number (1-based, starting from 1) |
| `chunk` | File | Yes | Chunk data (recommended: 10MB per chunk) |

**Response:**
```json
{
  "status": "ok",
  "part": 1,
  "message": "Chunk 1 uploaded successfully"
}
```

**Example:**
```javascript
const CHUNK_SIZE = 10 * 1024 * 1024; // 10 MB
let partNumber = 1;
let offset = 0;

while (offset < file.size) {
  const chunk = file.slice(offset, offset + CHUNK_SIZE);
  
  const formData = new FormData();
  formData.append('upload_id', upload_id);
  formData.append('part_number', partNumber);
  formData.append('chunk', chunk, `chunk_${partNumber}`);
  
  const response = await fetch('http://103.195.244.67/api/v2/upload/chunk', {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    // Retry logic here
    console.error(`Chunk ${partNumber} failed, retrying...`);
    continue;
  }
  
  offset += CHUNK_SIZE;
  partNumber++;
}
```

**Note:** Chunks can be uploaded in any order and retried if they fail.

---

#### Step 3: Complete Upload & Process Video

```
POST /api/v2/upload/complete
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `upload_id` | String | Yes | Upload session ID |
| `lpCrewName` | String | No | LP crew member name |
| `lpCrewId` | String | No | LP crew member ID |
| `alpCrewName` | String | No | ALP crew member name |
| `alpCrewId` | String | No | ALP crew member ID |
| `useMockDetection` | Boolean | No | Use mock detection (default: false) |
| `useMultiprocessing` | Boolean | No | Enable parallel processing |
| `saveClips` | Boolean | No | Save annotated frames (default: false) |

**Response:**
```json
{
  "status": "success",
  "trip_id": "trip123",
  "activities": [...],
  "output_path": "/opt/poc2/output/...",
  "summary": {...}
}
```

**Example:**
```javascript
const formData = new URLSearchParams({
  upload_id: upload_id,
  lpCrewName: 'John Doe',
  lpCrewId: 'LP001'
});

const response = await fetch('http://103.195.244.67/api/v2/upload/complete', {
  method: 'POST',
  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  body: formData
});

const result = await response.json();
```

---

### Additional Endpoints

#### Check Upload Status
```
GET /api/v2/upload/{upload_id}/status
```

**Response:**
```json
{
  "upload_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "in_progress",
  "total_chunks": 100,
  "uploaded_chunks": [1, 2, 3, 5, 7],
  "missing_chunks": [4, 6, 8, 9, 10, ...],
  "progress_percentage": 5.0
}
```

#### Cancel Upload
```
DELETE /api/v2/upload/{upload_id}
```

**Response:**
```json
{
  "status": "cancelled",
  "upload_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Upload session has been cancelled and cleaned up"
}
```

---

## Recommended Implementation

### For Small Files (< 100MB) or Reliable Networks:
**Use Streaming Upload** (`/api/v2/jobs/streaming`)
- Simpler implementation
- Single request
- Faster for small files

### For Large Files (> 100MB) or Unreliable Networks:
**Use Chunked Upload** (`/api/v2/upload/*`)
- Resumable on network failure
- Progress tracking
- Better for mobile/unstable connections

---

## File Size Limits

- **Maximum file size:** 1GB (1,073,741,824 bytes)
- **Recommended chunk size:** 10MB (10,485,760 bytes)
- **Supported formats:** `.mp4`, `.avi`, `.mov`, `.mkv`

---

## Error Handling

| Status Code | Meaning | Action |
|-------------|---------|--------|
| 200 | Success | Process response |
| 400 | Bad Request | Check request parameters |
| 413 | File Too Large | File exceeds 1GB limit |
| 507 | Insufficient Storage | Server disk space full |
| 404 | Not Found | Upload session expired or invalid |
| 500 | Server Error | Retry or contact support |

---

## Testing

1. **Interactive Testing:**
   - Visit: http://103.195.244.67/docs
   - Find endpoints under "video-v2" tag
   - Test directly from browser

2. **Health Check:**
   ```
   GET http://103.195.244.67/api/health
   ```

---

## Example: Complete Chunked Upload Flow (JavaScript)

```javascript
class VideoUploader {
  constructor(apiBaseUrl = 'http://103.195.244.67/api') {
    this.apiBaseUrl = apiBaseUrl;
  }

  async uploadVideo(file, tripId, options = {}) {
    // Step 1: Initiate
    const { upload_id, chunk_size_recommendation } = await this.initiateUpload(
      file.name,
      file.size,
      tripId
    );

    // Step 2: Upload chunks
    await this.uploadChunks(file, upload_id, chunk_size_recommendation);

    // Step 3: Complete
    return await this.completeUpload(upload_id, options);
  }

  async initiateUpload(filename, totalSize, tripId) {
    const response = await fetch(`${this.apiBaseUrl}/v2/upload/initiate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        filename,
        total_size: totalSize,
        tripId
      })
    });

    if (!response.ok) throw new Error(`Initiate failed: ${response.statusText}`);
    return await response.json();
  }

  async uploadChunks(file, uploadId, chunkSize, onProgress) {
    const totalChunks = Math.ceil(file.size / chunkSize);

    for (let partNumber = 1; partNumber <= totalChunks; partNumber++) {
      const offset = (partNumber - 1) * chunkSize;
      const chunk = file.slice(offset, offset + chunkSize);

      const formData = new FormData();
      formData.append('upload_id', uploadId);
      formData.append('part_number', partNumber);
      formData.append('chunk', chunk);

      const response = await fetch(`${this.apiBaseUrl}/v2/upload/chunk`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Chunk ${partNumber} upload failed`);
      }

      if (onProgress) {
        onProgress({
          partNumber,
          totalChunks,
          percentage: (partNumber / totalChunks) * 100
        });
      }
    }
  }

  async completeUpload(uploadId, options = {}) {
    const params = new URLSearchParams({ upload_id: uploadId });
    
    if (options.lpCrewName) params.append('lpCrewName', options.lpCrewName);
    if (options.lpCrewId) params.append('lpCrewId', options.lpCrewId);
    if (options.alpCrewName) params.append('alpCrewName', options.alpCrewName);
    if (options.alpCrewId) params.append('alpCrewId', options.alpCrewId);
    if (options.useMockDetection !== undefined) {
      params.append('useMockDetection', options.useMockDetection);
    }

    const response = await fetch(`${this.apiBaseUrl}/v2/upload/complete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: params
    });

    if (!response.ok) throw new Error(`Complete failed: ${response.statusText}`);
    return await response.json();
  }
}

// Usage
const uploader = new VideoUploader();
const fileInput = document.querySelector('input[type="file"]');

fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  
  try {
    const result = await uploader.uploadVideo(file, 'trip123', {
      lpCrewName: 'John Doe',
      lpCrewId: 'LP001'
    });
    
    console.log('Upload successful:', result);
  } catch (error) {
    console.error('Upload failed:', error);
  }
});
```

---

## Support

For questions or issues, refer to:
- API Documentation: http://103.195.244.67/docs
- Health Check: http://103.195.244.67/api/health


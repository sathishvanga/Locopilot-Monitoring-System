# Chunked Video Upload Guide

## Overview

The chunked upload system allows you to upload large video files (100+ MB) by splitting them into smaller 8 MB chunks. This solves the network connection reset issues that occur with large single-file uploads.

## Features

✅ **Reliable**: Upload large videos without connection timeouts
✅ **Resumable**: Retry individual chunks on failure
✅ **Out-of-order**: Chunks can be uploaded in any sequence
✅ **Idempotent**: Re-upload the same chunk safely
✅ **Automatic cleanup**: Expired sessions cleaned up automatically
✅ **Backward compatible**: Original `/api/jobs` endpoint still works

## Architecture

### Three-Step Upload Process

```
1. Initiate → POST /api/chunked-upload/initiate
   ↓ Returns uploadId

2. Upload Chunks → POST /api/chunked-upload/chunk (multiple times)
   ↓ Upload each 8 MB chunk

3. Finalize → POST /api/chunked-upload/finalize
   ↓ Reassemble & process video
   ↓
   ✅ Returns processing results
```

## API Endpoints

### 1. Initiate Upload

**Endpoint**: `POST /api/chunked-upload/initiate`

**Purpose**: Create a new upload session

**Request** (multipart/form-data):
```json
{
  "tripId": "TRIP-123",
  "filename": "video.mp4",
  "totalSize": 104857600,
  "lpCrewName": "John Doe",
  "lpCrewId": "LP-001",
  "alpCrewName": "Jane Smith",
  "alpCrewId": "ALP-002"
}
```

**Response**:
```json
{
  "status": "initiated",
  "uploadId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "totalChunks": 13,
  "chunkSize": 8388608,
  "expiresAt": "2025-11-30T15:30:00Z"
}
```

**Validations**:
- `tripId`: Required, non-empty
- `filename`: Must have valid extension (.mp4, .avi, .mov, .mkv)
- `totalSize`: Must be > 0 and ≤ 500 MB

---

### 2. Upload Chunk

**Endpoint**: `POST /api/chunked-upload/chunk`

**Purpose**: Upload a single chunk

**Request** (multipart/form-data):
```
uploadId: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
chunkIndex: 5
chunk: <binary data>
```

**Response**:
```json
{
  "status": "received",
  "uploadId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "chunkIndex": 5,
  "receivedChunks": 6,
  "totalChunks": 13,
  "complete": false
}
```

**Features**:
- **Idempotent**: Can re-upload same chunk
- **Out-of-order**: Chunks can arrive in any order
- **Size validation**: Each chunk ≤ 8 MB (last chunk can be smaller)

---

### 3. Finalize Upload

**Endpoint**: `POST /api/chunked-upload/finalize`

**Purpose**: Reassemble chunks and start video processing

**Request** (multipart/form-data):
```json
{
  "uploadId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "useMockDetection": false,
  "useMultiprocessing": true,
  "saveClips": false
}
```

**Response**: Standard `VideoProcessingResponse`
```json
{
  "status": "success",
  "message": "Video processed successfully",
  "tripId": "TRIP-123",
  "videoFilename": "video.mp4",
  "runDirectory": "/path/to/run_20251130_143045",
  "activitiesJsonPath": "/path/to/activities.json",
  "activitiesCount": 5,
  "processingTime": 45.67,
  "activities": [...]
}
```

**Process**:
1. Validates all chunks received
2. Reassembles chunks into complete video
3. Processes video (activity detection)
4. Cleans up chunks (background)
5. Returns processing results

---

## Error Handling

| Error Code | Meaning | Action |
|------------|---------|--------|
| 400 | Bad Request | Check request parameters |
| 404 | Session Not Found | Initiate new upload |
| 410 | Session Expired | Initiate new upload (1 hour timeout) |
| 500 | Internal Server Error | Retry or contact support |

**Common Error Messages**:
- `"Upload session expired. Please restart upload."` → Session timed out (1 hour)
- `"Cannot finalize: Missing chunks [3, 7, 12]"` → Re-upload missing chunks
- `"Chunk index 25 exceeds total chunks 20"` → Invalid chunk index
- `"File size exceeds maximum 500 MB"` → File too large

---

## Client Implementation

### Python Example

```python
import os
import requests

BASE_URL = "http://localhost:8000/api"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB

def upload_video_chunked(video_path: str, trip_id: str):
    file_size = os.path.getsize(video_path)
    filename = os.path.basename(video_path)
    total_chunks = (file_size + CHUNK_SIZE - 1) // CHUNK_SIZE

    # Step 1: Initiate
    response = requests.post(
        f"{BASE_URL}/chunked-upload/initiate",
        data={
            "tripId": trip_id,
            "filename": filename,
            "totalSize": file_size
        }
    )
    upload_id = response.json()["uploadId"]

    # Step 2: Upload chunks
    with open(video_path, 'rb') as f:
        for i in range(total_chunks):
            chunk = f.read(CHUNK_SIZE)

            # Retry logic
            max_retries = 3
            for retry in range(max_retries):
                try:
                    requests.post(
                        f"{BASE_URL}/chunked-upload/chunk",
                        data={"uploadId": upload_id, "chunkIndex": i},
                        files={"chunk": chunk}
                    )
                    break
                except Exception as e:
                    if retry == max_retries - 1:
                        raise

    # Step 3: Finalize
    response = requests.post(
        f"{BASE_URL}/chunked-upload/finalize",
        data={"uploadId": upload_id, "useMultiprocessing": "true"}
    )
    return response.json()
```

### JavaScript Example

```javascript
async function uploadVideoChunked(videoFile, tripId) {
    const CHUNK_SIZE = 8 * 1024 * 1024;
    const totalChunks = Math.ceil(videoFile.size / CHUNK_SIZE);

    // Step 1: Initiate
    const initData = new FormData();
    initData.append('tripId', tripId);
    initData.append('filename', videoFile.name);
    initData.append('totalSize', videoFile.size);

    const initRes = await fetch('/api/chunked-upload/initiate', {
        method: 'POST',
        body: initData
    });
    const { uploadId } = await initRes.json();

    // Step 2: Upload chunks
    for (let i = 0; i < totalChunks; i++) {
        const start = i * CHUNK_SIZE;
        const end = Math.min(start + CHUNK_SIZE, videoFile.size);
        const chunk = videoFile.slice(start, end);

        const chunkData = new FormData();
        chunkData.append('uploadId', uploadId);
        chunkData.append('chunkIndex', i);
        chunkData.append('chunk', chunk);

        await fetch('/api/chunked-upload/chunk', {
            method: 'POST',
            body: chunkData
        });
    }

    // Step 3: Finalize
    const finalData = new FormData();
    finalData.append('uploadId', uploadId);
    finalData.append('useMultiprocessing', 'true');

    const finalRes = await fetch('/api/chunked-upload/finalize', {
        method: 'POST',
        body: finalData
    });
    return await finalRes.json();
}
```

---

## Configuration

Settings in `app/utils/config.py`:

```python
# Chunked upload settings
chunk_size: int = 8 * 1024 * 1024  # Fixed 8 MB
upload_session_timeout: int = 3600  # 1 hour
max_upload_sessions: int = 100  # Max concurrent sessions
chunks_cleanup_interval: int = 300  # Cleanup every 5 minutes
```

**Environment Variables** (optional):
- `CHUNK_SIZE`: Chunk size in bytes (default: 8388608)
- `UPLOAD_SESSION_TIMEOUT`: Session timeout in seconds (default: 3600)
- `MAX_UPLOAD_SESSIONS`: Max concurrent sessions (default: 100)

---

## Testing

### Using the Test Client

```bash
# Basic test
python test_chunked_upload_client.py ./large_video.mp4 TRIP-123

# Test with crew info
python test_chunked_upload_client.py ./100mb_video.mp4 TRIP-456
```

### Using cURL

```bash
# Step 1: Initiate
curl -X POST http://localhost:8000/api/chunked-upload/initiate \
  -F "tripId=TRIP-123" \
  -F "filename=video.mp4" \
  -F "totalSize=104857600"

# Step 2: Upload chunk
curl -X POST http://localhost:8000/api/chunked-upload/chunk \
  -F "uploadId=<upload-id>" \
  -F "chunkIndex=0" \
  -F "chunk=@chunk_0000.bin"

# Step 3: Finalize
curl -X POST http://localhost:8000/api/chunked-upload/finalize \
  -F "uploadId=<upload-id>" \
  -F "useMultiprocessing=true"
```

---

## Storage & Cleanup

### Chunk Storage Location

```
/tmp/locopilot_uploads/chunks/
  └── {uploadId}/
      ├── chunk_0000.bin
      ├── chunk_0001.bin
      ├── chunk_0002.bin
      └── ...
```

### Automatic Cleanup

- **Session expiration**: 1 hour from creation
- **Cleanup interval**: Every 5 minutes (background task)
- **On finalize**: Chunks deleted after reassembly (background task)
- **On error**: Chunks deleted on finalize failure

### Manual Cleanup (if needed)

```bash
# Remove all chunks
rm -rf /tmp/locopilot_uploads/chunks/

# Remove specific upload session
rm -rf /tmp/locopilot_uploads/chunks/<upload-id>
```

---

## Performance

### Expected Performance

| Metric | Value |
|--------|-------|
| Chunk upload | < 2 seconds per 8 MB chunk |
| Reassembly | < 5 seconds for 100 MB video |
| Total overhead | ~10-15 seconds vs direct upload |
| Memory usage | Minimal (streaming I/O) |

### Optimization Tips

1. **Parallel chunk uploads**: Upload multiple chunks concurrently (client-side)
2. **Retry failed chunks**: Don't restart entire upload
3. **Progress tracking**: Use `receivedChunks` from chunk response
4. **Network conditions**: Adjust retry logic based on network quality

---

## Security

### Built-in Security Measures

1. **UUID-based upload IDs**: No user input in file paths
2. **Path traversal prevention**: Validated file paths
3. **Session limits**: Max 100 concurrent sessions
4. **Size validation**: Total file ≤ 500 MB, chunks ≤ 8 MB
5. **Format validation**: Only allowed video extensions
6. **Automatic cleanup**: Prevent disk space exhaustion

---

## Backward Compatibility

### Existing Endpoint Still Works

The original `/api/jobs` endpoint remains unchanged:
- **Small videos (< 50 MB)**: Use `/api/jobs` for simplicity
- **Large videos (≥ 50 MB)**: Use chunked upload for reliability

### Migration Path

No migration required! Both endpoints coexist:
- Desktop apps can update to chunked upload at their own pace
- Web clients can continue using `/api/jobs`
- API is fully backward compatible

---

## Troubleshooting

### Problem: "Upload session expired"

**Solution**: Session timeout is 1 hour. Complete upload faster or increase timeout:
```python
# In config.py
upload_session_timeout: int = 7200  # 2 hours
```

### Problem: "Missing chunks" error on finalize

**Solution**: Check which chunks are missing and re-upload them:
```python
# Error message shows: "Missing chunks [3, 7, 12]"
# Re-upload chunks 3, 7, and 12
```

### Problem: Slow chunk uploads

**Solution**:
1. Check network connection
2. Upload chunks in parallel (client-side)
3. Reduce chunk size if needed (edit config.py)

### Problem: Server running out of disk space

**Solution**:
1. Check cleanup interval: `chunks_cleanup_interval` (default: 5 minutes)
2. Reduce session timeout: `upload_session_timeout` (default: 1 hour)
3. Monitor disk usage: `df -h /tmp/locopilot_uploads/chunks/`

---

## API Documentation

### OpenAPI/Swagger Docs

Access interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Health Check

Check if chunked upload is enabled:
```bash
curl http://localhost:8000/api/health
```

Response includes chunked upload configuration.

---

## Support

### Logs

Check application logs for debugging:
```bash
tail -f logs/LocopilotMonitoring.log
```

Look for:
- `📥 Initiating chunked upload`
- `💾 Saved chunk X/Y`
- `🔧 Reassembling chunks`
- `✅ Chunked upload complete`

### Common Log Messages

- `"Upload session created"` → Initiate successful
- `"Chunk X received"` → Chunk upload successful
- `"Reassembled video"` → Reassembly successful
- `"Cleaning up expired sessions"` → Background cleanup running

---

## Summary

The chunked upload system provides:

✅ **Reliability**: No more connection resets for large videos
✅ **Resilience**: Retry individual chunks on failure
✅ **Flexibility**: Upload chunks in any order
✅ **Safety**: Automatic cleanup prevents disk issues
✅ **Compatibility**: Works alongside existing endpoint
✅ **Simplicity**: Clear 3-step API

**Ready to use!** Start testing with `test_chunked_upload_client.py` or integrate the client code into your application.

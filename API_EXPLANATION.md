# API Explanation - 1GB Video Upload Feature

## Overview

We've implemented **two upload methods** for handling videos up to 1GB:

1. **Streaming Upload** - Simple, single-request upload
2. **Chunked Upload** - Resumable, multi-step upload for large files

Both methods are memory-efficient and stream files directly to disk without loading the entire file into memory.

---

## Method 1: Streaming Upload API

### Endpoint
```
POST http://103.195.244.67/api/v2/jobs/streaming
```

### What It Does
- Accepts the entire video file in **one request**
- Streams the file to disk in **1MB chunks** (memory-efficient)
- Processes the video immediately after upload
- Returns processing results

### When to Use
✅ **Best for:**
- Files under 100MB
- Reliable network connections
- Simple implementation
- Quick uploads

❌ **Not ideal for:**
- Very large files (> 500MB)
- Unreliable networks (no resume capability)
- Need for upload progress tracking

### How It Works

```
Client                    Server
  |                         |
  |--- POST /v2/jobs/streaming --->|
  |   (video file + metadata)      |
  |                         |
  |                         | Streams to disk (1MB chunks)
  |                         | Validates size on-the-fly
  |                         | Processes video
  |                         |
  |<-- Processing Results ---|
  |   (activities, summary)  |
```

### Request Example

```javascript
const formData = new FormData();
formData.append('video', videoFile);        // File object
formData.append('tripId', 'trip123');       // Required
formData.append('lpCrewName', 'John Doe');  // Optional
formData.append('lpCrewId', 'LP001');       // Optional

const response = await fetch('http://103.195.244.67/api/v2/jobs/streaming', {
  method: 'POST',
  body: formData
});

const result = await response.json();
// {
//   "status": "success",
//   "trip_id": "trip123",
//   "activities": [...],
//   "output_path": "...",
//   "summary": {...}
// }
```

### Response Structure

```json
{
  "status": "success",
  "trip_id": "trip123",
  "activities": [
    {
      "activity": "cell_phone",
      "start_time": 10.5,
      "end_time": 15.2,
      "confidence": 0.95
    }
  ],
  "output_path": "/opt/poc2/output/...",
  "summary": {
    "total_activities": 5,
    "duration": 120.5
  }
}
```

### Error Handling

| Status | Meaning | What Happened |
|--------|---------|---------------|
| 200 | Success | Video uploaded and processed |
| 400 | Bad Request | Invalid file or missing tripId |
| 413 | Payload Too Large | File exceeds 1GB limit |
| 507 | Insufficient Storage | Server disk space full |
| 500 | Server Error | Processing failed |

---

## Method 2: Chunked Upload API (3-Step Process)

### Overview
This is a **3-step process** that allows you to:
- Upload files in smaller chunks (10MB each)
- Resume uploads if network fails
- Track upload progress
- Retry failed chunks individually

### When to Use
✅ **Best for:**
- Large files (> 500MB)
- Unreliable networks (mobile, WiFi)
- Need for progress tracking
- Need for resume capability

❌ **Not ideal for:**
- Small files (< 100MB) - Overhead not worth it
- Simple use cases - Streaming is easier

---

## Step 1: Initiate Upload Session

### Endpoint
```
POST http://103.195.244.67/api/v2/upload/initiate
```

### What It Does
- Creates an upload session on the server
- Returns an `upload_id` for tracking
- Tells you recommended chunk size
- Calculates total number of chunks needed

### Request

```javascript
const response = await fetch('http://103.195.244.67/api/v2/upload/initiate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  body: new URLSearchParams({
    filename: 'video.mp4',        // Original filename
    total_size: file.size,         // File size in bytes
    tripId: 'trip123'              // Trip identifier
  })
});

const data = await response.json();
// {
//   "upload_id": "550e8400-e29b-41d4-a716-446655440000",
//   "chunk_size_recommendation": 10485760,  // 10 MB
//   "total_chunks": 100,
//   "expires_at": "2025-11-30T02:00:00"
// }
```

### Response

```json
{
  "upload_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunk_size_recommendation": 10485760,
  "total_chunks": 100,
  "expires_at": "2025-11-30T02:00:00"
}
```

**Important:** Save the `upload_id` - you'll need it for steps 2 and 3!

---

## Step 2: Upload Chunks

### Endpoint
```
POST http://103.195.244.67/api/v2/upload/chunk
```

### What It Does
- Accepts individual chunks of the file
- Saves each chunk to temporary storage
- Can be called multiple times (once per chunk)
- Chunks can be uploaded in **any order**
- Failed chunks can be **retried**

### Request

```javascript
const CHUNK_SIZE = 10 * 1024 * 1024; // 10 MB
const uploadId = data.upload_id;     // From Step 1
let partNumber = 1;
let offset = 0;

while (offset < file.size) {
  // Get chunk from file
  const chunk = file.slice(offset, offset + CHUNK_SIZE);
  
  // Create form data
  const formData = new FormData();
  formData.append('upload_id', uploadId);
  formData.append('part_number', partNumber);
  formData.append('chunk', chunk, `chunk_${partNumber}`);
  
  // Upload chunk
  const response = await fetch('http://103.195.244.67/api/v2/upload/chunk', {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    // Retry this chunk
    console.error(`Chunk ${partNumber} failed, retrying...`);
    continue; // Don't increment, retry same chunk
  }
  
  // Success - move to next chunk
  offset += CHUNK_SIZE;
  partNumber++;
  
  // Update progress
  const progress = (offset / file.size) * 100;
  console.log(`Upload progress: ${progress.toFixed(1)}%`);
}
```

### Response

```json
{
  "status": "ok",
  "part": 1,
  "message": "Chunk 1 uploaded successfully"
}
```

### Key Features

1. **Chunks can be uploaded in any order** - Server handles ordering
2. **Retry failed chunks** - Just retry the specific chunk that failed
3. **Progress tracking** - You know which chunks are uploaded
4. **Resume capability** - If upload fails, check status and resume

### Check Upload Status (Optional)

```javascript
// Check which chunks are uploaded
const statusResponse = await fetch(
  `http://103.195.244.67/api/v2/upload/${uploadId}/status`
);

const status = await statusResponse.json();
// {
//   "upload_id": "...",
//   "status": "in_progress",
//   "total_chunks": 100,
//   "uploaded_chunks": [1, 2, 3, 5, 7],
//   "missing_chunks": [4, 6, 8, 9, 10, ...],
//   "progress_percentage": 5.0
// }

// Resume upload for missing chunks
for (const chunkNum of status.missing_chunks) {
  // Upload that specific chunk
}
```

---

## Step 3: Complete Upload & Process

### Endpoint
```
POST http://103.195.244.67/api/v2/upload/complete
```

### What It Does
- Assembles all uploaded chunks into final file
- Validates file size matches expected
- Processes the video (same as streaming endpoint)
- Cleans up temporary chunks
- Returns processing results

### Request

```javascript
const formData = new URLSearchParams({
  upload_id: uploadId,              // From Step 1
  lpCrewName: 'John Doe',           // Optional
  lpCrewId: 'LP001',                // Optional
  alpCrewName: 'Jane Smith',        // Optional
  alpCrewId: 'ALP001',              // Optional
  useMockDetection: false,          // Optional
  saveClips: false                  // Optional
});

const response = await fetch('http://103.195.244.67/api/v2/upload/complete', {
  method: 'POST',
  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  body: formData
});

const result = await response.json();
// Same structure as streaming upload response
```

### Response

```json
{
  "status": "success",
  "trip_id": "trip123",
  "activities": [...],
  "output_path": "/opt/poc2/output/...",
  "summary": {...}
}
```

### What Happens Behind the Scenes

```
Server receives complete request
  ↓
Checks all chunks are uploaded
  ↓
Assembles chunks in order:
  part_000001.chunk
  part_000002.chunk
  ...
  part_000100.chunk
  ↓
Validates final file size
  ↓
Processes video (activity detection)
  ↓
Returns results
  ↓
Cleans up temporary chunks
```

---

## Complete Example: Chunked Upload Flow

```javascript
class ChunkedVideoUploader {
  constructor(apiBaseUrl = 'http://103.195.244.67/api') {
    this.apiBaseUrl = apiBaseUrl;
  }

  async uploadVideo(file, tripId, options = {}) {
    try {
      // Step 1: Initiate
      console.log('Step 1: Initiating upload session...');
      const { upload_id, chunk_size_recommendation, total_chunks } = 
        await this.initiateUpload(file.name, file.size, tripId);
      
      console.log(`Upload ID: ${upload_id}`);
      console.log(`Total chunks: ${total_chunks}`);

      // Step 2: Upload chunks with progress
      console.log('Step 2: Uploading chunks...');
      await this.uploadChunks(
        file, 
        upload_id, 
        chunk_size_recommendation,
        (progress) => {
          console.log(`Progress: ${progress.percentage.toFixed(1)}%`);
          // Update UI progress bar
        }
      );

      // Step 3: Complete and process
      console.log('Step 3: Completing upload and processing...');
      const result = await this.completeUpload(upload_id, options);
      
      console.log('Upload successful!', result);
      return result;

    } catch (error) {
      console.error('Upload failed:', error);
      throw error;
    }
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

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`Initiate failed: ${error.message || response.statusText}`);
    }

    return await response.json();
  }

  async uploadChunks(file, uploadId, chunkSize, onProgress) {
    const totalChunks = Math.ceil(file.size / chunkSize);

    for (let partNumber = 1; partNumber <= totalChunks; partNumber++) {
      const offset = (partNumber - 1) * chunkSize;
      const chunk = file.slice(offset, offset + chunkSize);

      // Retry logic
      let retries = 3;
      let success = false;

      while (retries > 0 && !success) {
        try {
          const formData = new FormData();
          formData.append('upload_id', uploadId);
          formData.append('part_number', partNumber);
          formData.append('chunk', chunk);

          const response = await fetch(`${this.apiBaseUrl}/v2/upload/chunk`, {
            method: 'POST',
            body: formData
          });

          if (!response.ok) {
            throw new Error(`Chunk ${partNumber} failed: ${response.statusText}`);
          }

          success = true;

          // Progress callback
          if (onProgress) {
            onProgress({
              partNumber,
              totalChunks,
              percentage: (partNumber / totalChunks) * 100
            });
          }

        } catch (error) {
          retries--;
          if (retries === 0) {
            throw new Error(`Failed to upload chunk ${partNumber} after 3 retries`);
          }
          console.warn(`Chunk ${partNumber} failed, retrying... (${retries} retries left)`);
          await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1s before retry
        }
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

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`Complete failed: ${error.message || response.statusText}`);
    }

    return await response.json();
  }

  // Optional: Check upload status
  async getUploadStatus(uploadId) {
    const response = await fetch(
      `${this.apiBaseUrl}/v2/upload/${uploadId}/status`
    );

    if (!response.ok) {
      throw new Error(`Status check failed: ${response.statusText}`);
    }

    return await response.json();
  }

  // Optional: Cancel upload
  async cancelUpload(uploadId) {
    const response = await fetch(
      `${this.apiBaseUrl}/v2/upload/${uploadId}`,
      { method: 'DELETE' }
    );

    if (!response.ok) {
      throw new Error(`Cancel failed: ${response.statusText}`);
    }

    return await response.json();
  }
}

// Usage
const uploader = new ChunkedVideoUploader();

const fileInput = document.querySelector('input[type="file"]');
fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  
  try {
    const result = await uploader.uploadVideo(file, 'trip123', {
      lpCrewName: 'John Doe',
      lpCrewId: 'LP001'
    });
    
    console.log('Success!', result);
  } catch (error) {
    console.error('Upload failed:', error);
  }
});
```

---

## Comparison: Streaming vs Chunked

| Feature | Streaming Upload | Chunked Upload |
|---------|----------------|----------------|
| **Complexity** | Simple (1 request) | Complex (3 steps) |
| **Network Failure** | Must restart entire upload | Can resume from failed chunk |
| **Progress Tracking** | Limited (browser progress) | Detailed (per chunk) |
| **Best For** | Small files, reliable network | Large files, unreliable network |
| **Memory Usage** | Low (1MB chunks) | Low (10MB chunks) |
| **Implementation Time** | 5 minutes | 30-60 minutes |

---

## Error Scenarios & Solutions

### Scenario 1: Network Failure During Streaming Upload
**Problem:** Upload fails at 80%  
**Solution:** Must restart entire upload  
**Better Option:** Use chunked upload for large files

### Scenario 2: Network Failure During Chunked Upload
**Problem:** Upload fails at chunk 50/100  
**Solution:** 
1. Check status: `GET /v2/upload/{upload_id}/status`
2. Resume uploading missing chunks
3. Complete when all chunks uploaded

### Scenario 3: File Too Large
**Error:** `413 Payload Too Large`  
**Solution:** File exceeds 1GB limit - compress or split video

### Scenario 4: Disk Space Full
**Error:** `507 Insufficient Storage`  
**Solution:** Server disk is full - contact admin or wait for cleanup

---

## Best Practices

1. **Choose the right method:**
   - Files < 100MB → Use Streaming
   - Files > 100MB → Use Chunked

2. **Implement retry logic:**
   - Always retry failed requests
   - Exponential backoff for retries

3. **Show progress:**
   - For streaming: Use browser's native progress
   - For chunked: Calculate from uploaded chunks

4. **Handle errors gracefully:**
   - Show user-friendly error messages
   - Allow retry/cancel options

5. **Validate before upload:**
   - Check file size (max 1GB)
   - Check file type (.mp4, .avi, .mov, .mkv)
   - Check tripId is provided

---

## Testing

### Test Streaming Upload
```bash
curl -X POST "http://103.195.244.67/api/v2/jobs/streaming" \
  -F "video=@test_video.mp4" \
  -F "tripId=test123"
```

### Test Chunked Upload
```bash
# Step 1: Initiate
curl -X POST "http://103.195.244.67/api/v2/upload/initiate" \
  -d "filename=test.mp4&total_size=104857600&tripId=test123"

# Step 2: Upload chunk (repeat for each chunk)
curl -X POST "http://103.195.244.67/api/v2/upload/chunk" \
  -F "upload_id=YOUR_UPLOAD_ID" \
  -F "part_number=1" \
  -F "chunk=@chunk1.bin"

# Step 3: Complete
curl -X POST "http://103.195.244.67/api/v2/upload/complete" \
  -d "upload_id=YOUR_UPLOAD_ID"
```

### Interactive Testing
Visit: **http://103.195.244.67/docs**  
- Test all endpoints directly from browser
- See request/response examples
- Understand parameters

---

## Summary

**Streaming Upload:**
- ✅ Simple, one request
- ✅ Best for small files
- ❌ No resume on failure

**Chunked Upload:**
- ✅ Resumable, progress tracking
- ✅ Best for large files
- ❌ More complex implementation

Both methods are memory-efficient and support files up to 1GB!


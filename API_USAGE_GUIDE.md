# API Usage Guide - Locopilot Monitoring System

## Quick Start

### 1. Install and Start Server

```bash
# Make startup script executable (first time only)
chmod +x start_server.sh

# Start the server
./start_server.sh
```

The server will be available at: **http://localhost:8000**

### 2. Test the API

Open your browser and go to:
- **http://localhost:8000/docs** - Interactive API documentation (Swagger UI)
- **http://localhost:8000/redoc** - Alternative documentation (ReDoc)

## API Endpoints

### 1. Upload and Process Video

**Endpoint:** `POST /api/v1/video/process`

**Using cURL:**

```bash
curl -X POST "http://localhost:8000/api/v1/video/process" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@example_data/latest.mp4" \
  -F "tripId=TRIP-20251110-001" \
  -F "crewName=John Doe" \
  -F "crewId=C-001" \
  -F "crewRole=1"
```

**Using Python:**

```python
import requests

url = "http://localhost:8000/api/v1/video/process"

# Open video file
with open('example_data/latest.mp4', 'rb') as video_file:
    files = {'video': video_file}
    
    data = {
        'tripId': 'TRIP-20251110-001',
        'crewName': 'John Doe',
        'crewId': 'C-001',
        'crewRole': 1,
        'useMockDetection': False  # Set to True for testing
    }
    
    response = requests.post(url, files=files, data=data)
    result = response.json()
    
    print(f"Status: {result['status']}")
    print(f"Activities detected: {result['activitiesCount']}")
    print(f"Activities JSON: {result['activitiesJsonPath']}")
```

**Using JavaScript/Node.js:**

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('video', fs.createReadStream('example_data/latest.mp4'));
form.append('tripId', 'TRIP-20251110-001');
form.append('crewName', 'John Doe');
form.append('crewId', 'C-001');
form.append('crewRole', '1');

axios.post('http://localhost:8000/api/v1/video/process', form, {
    headers: form.getHeaders()
})
.then(response => {
    console.log('Status:', response.data.status);
    console.log('Activities detected:', response.data.activitiesCount);
    console.log('Activities JSON:', response.data.activitiesJsonPath);
})
.catch(error => {
    console.error('Error:', error.response?.data || error.message);
});
```

**Response Example:**

```json
{
  "status": "success",
  "message": "Video processed successfully",
  "tripId": "TRIP-20251110-001",
  "videoFilename": "TRIP-20251110-001_1699876543.mp4",
  "runDirectory": "/path/to/locopilot_evidence/run_20251110_143045",
  "activitiesJsonPath": "/path/to/locopilot_evidence/run_20251110_143045/activities.json",
  "activitiesCount": 3,
  "activities": [
    {
      "tripId": "TRIP-20251110-001",
      "activityType": 2,
      "des": "Using mobile phone",
      "objectType": "cell phone",
      "fileUrl": "/path/to/video.mp4",
      "fileDuration": "00:10:30",
      "activityStartTime": "45.50",
      "activityEndTime": "52.75",
      "crewName": "John Doe",
      "crewId": "C-001",
      "crewRole": 1,
      "date": "2025-11-10",
      "time": "14:30:45",
      "filename": "latest.mp4",
      "peopleCount": 1,
      "evidence": {"rule": "phone_in_hand"},
      "activityImage": "latest_cell_phone_frame00001365_000_activity.jpg",
      "activityClip": "latest_cell_phone_frame00001365_000_clip.mp4"
    }
  ],
  "processingTime": 45.67
}
```

### 2. Check Processing Status

**Endpoint:** `GET /api/v1/video/status/{run_id}`

```bash
curl "http://localhost:8000/api/v1/video/status/run_20251110_143045"
```

**Response:**

```json
{
  "status": "completed",
  "message": "Processing completed",
  "activitiesCount": 3,
  "summary": {
    "total_activities": 3,
    "activity_breakdown": {
      "Using mobile phone": 1,
      "Writing activity detected": 2
    },
    "total_duration": 45.5
  }
}
```

### 3. Health Check

**Endpoint:** `GET /api/v1/video/health`

```bash
curl "http://localhost:8000/api/v1/video/health"
```

**Response:**

```json
{
  "status": "healthy",
  "service": "video-processing",
  "version": "1.0.0",
  "config": {
    "max_upload_size_mb": 500,
    "allowed_extensions": [".mp4", ".avi", ".mov", ".mkv"],
    "sample_fps": 0.5,
    "output_dir": "locopilot_evidence"
  }
}
```

## Request Parameters

### POST /api/v1/video/process

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `video` | File | Yes | - | Video file to process |
| `tripId` | String | Yes | - | Unique trip identifier |
| `crewName` | String | No | "John Doe" | Crew member name |
| `crewId` | String | No | "C-001" | Crew member ID |
| `crewRole` | Integer | No | 1 | Crew role (1 = primary pilot) |
| `useMockDetection` | Boolean | No | false | Use mock detection for testing |

## Activity Types

| Code | Type | Description |
|------|------|-------------|
| 2 | Cell Phone | Using mobile phone |
| 3 | Microsleep | Micro-sleep detected (5+ seconds) |
| 4 | Sleep | Sleep detected (30+ seconds) |
| 5 | Writing | Writing activity detected |
| 6 | Packing Bags | Packing bags activity detected |
| 7 | Group Detected | More than 2 people detected |

## Output Files

After processing, files are saved in `locopilot_evidence/run_TIMESTAMP/`:

- **activities.json** - Structured activity data
- **clips/** - Video clips of detected activities
  - `{video}_{activity}_frame{number}_{counter}_clip.mp4`
  - `{video}_{activity}_frame{number}_{counter}_activity.jpg`

## Error Handling

The API returns appropriate HTTP status codes:

- **200** - Success
- **400** - Bad Request (invalid input)
- **404** - Not Found
- **422** - Validation Error
- **500** - Internal Server Error

**Error Response Example:**

```json
{
  "status": "error",
  "message": "Failed to process video",
  "error": "Invalid video format: file corrupted"
}
```

## Testing with Mock Detection

For quick testing without running the full ML pipeline:

```bash
curl -X POST "http://localhost:8000/api/v1/video/process" \
  -F "video=@example_data/latest.mp4" \
  -F "tripId=TEST-001" \
  -F "useMockDetection=true"
```

This generates sample activities instantly for testing the API structure.

## Configuration

Create a `.env` file to customize settings:

```env
# Server
PORT=8000
DEBUG=false

# Processing
SAMPLE_FPS=0.5
MAX_UPLOAD_SIZE=524288000

# Models
YOLO_WEIGHTS_PRELOAD=yolo11s.pt
```

## Production Deployment

For production, use Gunicorn with multiprocessing:

```bash
# Start with Gunicorn
gunicorn -c gunicorn_config.py app.main:app

# Or modify start_server.sh to use production mode
```

The Gunicorn configuration provides:
- Multiple worker processes (CPU count / 2)
- 600-second timeout for long videos
- Automatic worker recycling
- Request logging

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### "YOLO weights not found"
The system will download weights automatically on first use, or manually download `yolo11s.pt`.

### Video processing timeout
Increase timeout in `gunicorn_config.py`:
```python
timeout = 1200  # 20 minutes
```

### Large video files
Adjust max upload size in `.env`:
```env
MAX_UPLOAD_SIZE=1073741824  # 1 GB
```

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Verify all dependencies are installed
3. Ensure YOLO weights are available
4. Test with mock detection first

## Next Steps

1. **Integrate with your application** - Use the API in your frontend/backend
2. **Add authentication** - Implement API key or JWT authentication
3. **Deploy to production** - Use Docker, Kubernetes, or cloud services
4. **Monitor performance** - Add application monitoring and logging
5. **Scale horizontally** - Add more worker instances as needed


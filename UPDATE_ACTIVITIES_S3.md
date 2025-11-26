# Activities.json S3 URL Update

## 🎯 Feature: Automatic S3 URL Replacement

**What changed:** The backend now automatically replaces local file paths in `activities.json` with S3 URLs after upload.

---

## 📋 Before vs After

### **Before** (Local Paths)
```json
{
  "activityImage": "/Users/.../locopilot_evidence/run_XXX/clips/..._activity.jpg",
  "activityClip": "/Users/.../locopilot_evidence/run_XXX/clips/..._clip.mp4"
}
```

### **After** (S3 URLs)
```json
{
  "activityImage": "https://s3.ap-south-1.amazonaws.com/alphainspector/demo/cvvr/11-2025/..._activity.jpg",
  "activityClip": "https://s3.ap-south-1.amazonaws.com/alphainspector/demo/cvvr/11-2025/..._clip.mp4"
}
```

---

## 🔄 How It Works

### 1. **Upload All Evidence Files**
```
Backend uploads:
- Original video → S3
- Evidence clip 1 → S3
- Evidence image 1 → S3
- Evidence clip 2 → S3
- Evidence image 2 → S3
...
```

### 2. **Create Path Mapping**
```python
s3_file_mapping = {
    "/local/path/clip1.mp4": "https://s3.../clip1.mp4",
    "/local/path/image1.jpg": "https://s3.../image1.jpg",
    ...
}
```

### 3. **Update activities.json**
```python
# Read activities.json
activities = load_json("activities.json")

# Replace local paths with S3 URLs
for activity in activities:
    activity['activityClip'] = s3_file_mapping[activity['activityClip']]
    activity['activityImage'] = s3_file_mapping[activity['activityImage']]

# Save updated activities.json
save_json("activities.json", activities)
```

### 4. **Return Updated Activities**
```json
{
  "status": "success",
  "data": {
    "video_url": "https://s3.../original.mp4",
    "evidence_clips": ["https://s3.../clip1.mp4", ...],
    "activities": [
      {
        "activityClip": "https://s3.../clip1.mp4",
        "activityImage": "https://s3.../image1.jpg",
        ...
      }
    ]
  }
}
```

---

## 🚀 Implementation Details

### Modified File
**File**: `app/controllers/video_controller.py`

**Function**: `process_and_upload_video()`

### Changes Made

#### 1. **Upload Both Clips AND Images**
```python
# Before: Only uploaded clips
clip_files = result.get('clip_files', [])

# After: Upload clips + corresponding images
all_files_to_upload = []
for clip_file in clip_files:
    all_files_to_upload.append(clip_file)  # Add clip
    
    # Find and add corresponding image
    image_file = clip_file.replace('_clip.mp4', '_activity.jpg')
    if os.path.exists(image_file):
        all_files_to_upload.append(image_file)
```

#### 2. **Create Path Mapping**
```python
# Map local paths to S3 URLs
s3_file_mapping = {}
for local_path, s3_url in zip(all_files_to_upload, file_urls):
    s3_file_mapping[local_path] = s3_url
```

#### 3. **Update activities.json**
```python
# Read activities
with open(activities_json_path, 'r') as f:
    activities = json.load(f)

# Update URLs
for activity in activities:
    if activity['activityClip'] in s3_file_mapping:
        activity['activityClip'] = s3_file_mapping[activity['activityClip']]
    
    if activity['activityImage'] in s3_file_mapping:
        activity['activityImage'] = s3_file_mapping[activity['activityImage']]

# Save updated activities
with open(activities_json_path, 'w') as f:
    json.dump(activities, f, indent=2)
```

---

## 📊 Benefits

### ✅ **Accessibility**
- Activities can be viewed from anywhere
- No need for local file access
- Works with web-based frontends

### ✅ **Consistency**
- All URLs point to S3
- No mixed local/remote paths
- Easier to manage and display

### ✅ **Scalability**
- Backend can run on any server
- Files accessible from multiple clients
- Better for distributed systems

### ✅ **Portability**
- activities.json can be sent via API
- No local file dependencies
- Works across different machines

---

## 🧪 Testing

### Test the Updated Endpoint

```bash
# Start backend
uvicorn app.main:app --reload

# Upload and process video
curl -X POST "http://localhost:8000/api/v1/video/process-and-upload" \
  -F "video_file=@test_video.mp4" \
  -F "tripId=test-123"

# Check response - should include S3 URLs in activities
```

### Verify activities.json

```bash
# Check the updated activities.json
cat locopilot_evidence/run_*/activities.json

# Should show S3 URLs like:
# "activityClip": "https://s3.ap-south-1.amazonaws.com/..."
# "activityImage": "https://s3.ap-south-1.amazonaws.com/..."
```

---

## 📝 API Response

### Complete Response Structure

```json
{
  "status": "success",
  "message": "Video processed and uploaded successfully",
  "data": {
    "tripId": "test-123",
    "run_id": "run_20251126_223851",
    "run_dir": "/path/to/locopilot_evidence/run_20251126_223851",
    "activities_count": 1,
    "processing_time_seconds": 45.2,
    
    "video_url": "https://s3.../original_video.mp4",
    
    "evidence_clips": [
      "https://s3.../clip1.mp4",
      "https://s3.../clip2.mp4"
    ],
    
    "clips_uploaded": 2,
    "total_clips": 2,
    "upload_errors": null,
    
    "activities": [
      {
        "tripId": "test-123",
        "activityType": 6,
        "des": "Packing bags activity detected",
        "activityStartTime": "20.00",
        "activityEndTime": "26.00",
        "activityImage": "https://s3.../image1.jpg",  // ← S3 URL
        "activityClip": "https://s3.../clip1.mp4",    // ← S3 URL
        "date": "2025-11-26",
        "time": "22:39:03",
        ...
      }
    ]
  }
}
```

---

## 🔄 Workflow Summary

```
1. Desktop App uploads video
   ↓
2. Backend processes video
   ↓
3. Backend generates evidence clips + images
   ↓
4. Backend uploads to S3:
   - Original video
   - All clips
   - All images
   ↓
5. Backend updates activities.json:
   - Replace local paths with S3 URLs
   ↓
6. Backend returns response with:
   - S3 URLs
   - Updated activities array
   ↓
7. Desktop App receives S3 URLs
   ✅ Complete!
```

---

## 📚 Files Updated

- ✅ `app/controllers/video_controller.py` - Upload images + update activities.json
- ✅ `locopilot_evidence/run_*/activities.json` - Now contains S3 URLs
- ✅ This documentation file

---

## 🎯 Use Cases

### **Web Frontend**
Can now display activities directly from S3:
```html
<img src="{{ activity.activityImage }}" />
<video src="{{ activity.activityClip }}" />
```

### **Mobile App**
Can fetch and display evidence:
```swift
let imageURL = URL(string: activity.activityImage)
```

### **API Integration**
Third parties can access evidence directly:
```javascript
const response = await fetch(activity.activityClip);
```

---

## ✅ Status

**Status**: ✅ **IMPLEMENTED**

**Available in**: `/api/v1/video/process-and-upload` endpoint

**Backward Compatibility**: ✅ Yes (local files still exist)

---

## 📞 Support

Questions? Contact: info@mindcoinservices.com


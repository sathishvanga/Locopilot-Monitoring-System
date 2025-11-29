# Browser Upload Fix - "Failed to fetch" Error

## Root Cause

The error "Failed to fetch" with CORS/Network Failure occurs because:

1. **Error responses don't include CORS headers** - When validation fails (400), the browser can't read the error response
2. **Content-Type header issue** - Manually setting `Content-Type: multipart/form-data` breaks the request (missing boundary)

## Solution Applied

✅ **Added CORS headers to all exception handlers** - Now error responses include CORS headers so browsers can read them

## How to Fix in Your Code

### ❌ WRONG - Don't set Content-Type manually

```javascript
// DON'T DO THIS
const formData = new FormData();
formData.append('video', file);
formData.append('tripId', 'test123');

fetch('https://celebxmedia.info/api/v2/jobs/streaming', {
  method: 'POST',
  headers: {
    'Content-Type': 'multipart/form-data'  // ❌ WRONG - breaks the request!
  },
  body: formData
});
```

### ✅ CORRECT - Let browser set Content-Type automatically

```javascript
// DO THIS
const formData = new FormData();
formData.append('video', file);
formData.append('tripId', 'test123');
formData.append('useMultiprocessing', 'true');
formData.append('saveClips', 'false');
formData.append('useMockDetection', 'false');

fetch('https://celebxmedia.info/api/v2/jobs/streaming', {
  method: 'POST',
  // ✅ DON'T set Content-Type - browser will set it with boundary automatically
  body: formData
})
.then(response => {
  if (!response.ok) {
    return response.json().then(err => Promise.reject(err));
  }
  return response.json();
})
.then(data => {
  console.log('Success:', data);
})
.catch(error => {
  console.error('Error:', error);
});
```

### ✅ CORRECT - Using axios

```javascript
import axios from 'axios';

const formData = new FormData();
formData.append('video', file);
formData.append('tripId', 'test123');
formData.append('useMultiprocessing', 'true');
formData.append('saveClips', 'false');
formData.append('useMockDetection', 'false');

// ✅ Axios automatically sets Content-Type with boundary
axios.post('https://celebxmedia.info/api/v2/jobs/streaming', formData, {
  headers: {
    // Don't set Content-Type - axios handles it
  },
  timeout: 600000, // 10 minutes for large uploads
  onUploadProgress: (progressEvent) => {
    const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
    console.log(`Upload progress: ${percentCompleted}%`);
  }
})
.then(response => {
  console.log('Success:', response.data);
})
.catch(error => {
  console.error('Error:', error.response?.data || error.message);
});
```

### ✅ CORRECT - Using curl

```bash
# ✅ CORRECT - Don't set Content-Type, curl handles it
curl -X POST https://celebxmedia.info/api/v2/jobs/streaming \
  -F 'tripId=test123' \
  -F 'video=@actual_v7.mp4' \
  -F 'useMultiprocessing=true' \
  -F 'saveClips=false' \
  -F 'useMockDetection=false'

# ❌ WRONG - Don't manually set Content-Type
curl -X POST https://celebxmedia.info/api/v2/jobs/streaming \
  -H 'Content-Type: multipart/form-data' \  # ❌ This breaks it!
  -F 'tripId=test123' \
  -F 'video=@actual_v7.mp4'
```

## Testing

1. **Test with curl (bypasses browser):**
   ```bash
   curl -X POST https://celebxmedia.info/api/v2/jobs/streaming \
     -F 'tripId=test123' \
     -F 'video=@your_video.mp4' \
     -v
   ```

2. **Test OPTIONS preflight:**
   ```bash
   curl -X OPTIONS https://celebxmedia.info/api/v2/jobs/streaming \
     -H 'Origin: https://yourdomain.com' \
     -H 'Access-Control-Request-Method: POST' \
     -v
   ```

3. **Check browser console** - You should now see actual error messages instead of "Failed to fetch"

## Common Issues

### Issue: "Failed to fetch" in browser
- **Cause**: Error response missing CORS headers (now fixed)
- **Solution**: Deploy the updated code

### Issue: "There was an error parsing the body"
- **Cause**: Content-Type set manually without boundary
- **Solution**: Remove Content-Type header, let browser/curl set it

### Issue: Upload timeout
- **Cause**: Large file on slow connection
- **Solution**: Use chunked upload endpoint (`/api/v2/upload/*`)

### Issue: CORS preflight fails
- **Cause**: OPTIONS request not handled
- **Solution**: Already fixed in nginx and FastAPI

## Next Steps

1. Deploy the updated code with CORS headers in exception handlers
2. Update your frontend code to NOT set Content-Type manually
3. Test with a small file first, then try the 53 MB file
4. Monitor browser console for actual error messages


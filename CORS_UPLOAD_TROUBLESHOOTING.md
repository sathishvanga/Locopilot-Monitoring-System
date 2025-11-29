# CORS and Large File Upload Troubleshooting Guide

## Issue: "Failed to fetch" when uploading 53 MB video

### Root Causes

The error "Failed to fetch" with CORS/Network Failure typically indicates one of these issues:

1. **Reverse Proxy (Nginx) Timeout** - Most common for HTTPS endpoints
2. **Browser Upload Timeout** - Browser giving up during slow uploads
3. **CORS Preflight Failure** - OPTIONS request not handled properly
4. **Network/Connection Issues** - Unstable connection during upload

## Solutions

### 1. Nginx Configuration (If using reverse proxy)

If your server is behind nginx (common for HTTPS), add these settings to your nginx config:

```nginx
server {
    listen 443 ssl;
    server_name celebxmedia.info;

    # Increase client body size for large uploads (1GB)
    client_max_body_size 1G;
    
    # Increase timeouts for large file uploads
    client_body_timeout 300s;      # 5 minutes for upload
    proxy_connect_timeout 300s;
    proxy_send_timeout 300s;
    proxy_read_timeout 1800s;      # 30 minutes for processing
    
    # Buffer settings for large uploads
    client_body_buffer_size 128k;
    proxy_buffering off;            # Disable buffering for streaming
    
    # CORS headers (if not handled by FastAPI)
    add_header 'Access-Control-Allow-Origin' '*' always;
    add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
    add_header 'Access-Control-Allow-Headers' '*' always;
    
    # Handle OPTIONS preflight
    if ($request_method = 'OPTIONS') {
        add_header 'Access-Control-Allow-Origin' '*';
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS';
        add_header 'Access-Control-Allow-Headers' '*';
        add_header 'Access-Control-Max-Age' 3600;
        add_header 'Content-Type' 'text/plain; charset=utf-8';
        add_header 'Content-Length' 0;
        return 204;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Disable request buffering for streaming uploads
        proxy_request_buffering off;
    }
}
```

### 2. FastAPI CORS Configuration (Already Updated)

The application now includes:
- Explicit CORS middleware with proper headers
- OPTIONS handler for preflight requests
- Extended timeouts in gunicorn (30 minutes)

### 3. Browser/Client Side

If using fetch/axios, ensure proper timeout settings:

```javascript
// Example with fetch
const controller = new AbortController();
const timeoutId = setTimeout(() => controller.abort(), 600000); // 10 minutes

const formData = new FormData();
formData.append('video', videoFile);
formData.append('tripId', 'string');
// ... other fields

try {
    const response = await fetch('https://celebxmedia.info/api/v2/jobs/streaming', {
        method: 'POST',
        body: formData,
        signal: controller.signal,
        // Don't set Content-Type header - browser will set it with boundary
    });
    clearTimeout(timeoutId);
    // Handle response
} catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
        console.error('Upload timeout');
    } else {
        console.error('Upload failed:', error);
    }
}
```

### 4. Testing Steps

1. **Test CORS Preflight:**
   ```bash
   curl -X OPTIONS https://celebxmedia.info/api/v2/jobs/streaming \
     -H "Origin: https://yourdomain.com" \
     -H "Access-Control-Request-Method: POST" \
     -v
   ```
   Should return 200/204 with CORS headers.

2. **Test Direct Upload (bypassing browser):**
   ```bash
   curl -X POST https://celebxmedia.info/api/v2/jobs/streaming \
     -F 'tripId=test123' \
     -F 'video=@actual_v7.mp4' \
     -v
   ```
   This helps identify if it's a browser/CORS issue or server issue.

3. **Check Server Logs:**
   ```bash
   journalctl -u poc2 -f
   ```
   Look for:
   - CORS errors
   - Timeout errors
   - Request received but not completed

### 5. Alternative: Use Chunked Upload

For unreliable networks or large files, use the chunked upload endpoint:

```bash
# Step 1: Initiate upload
curl -X POST https://celebxmedia.info/api/v2/upload/initiate \
  -F 'filename=actual_v7.mp4' \
  -F 'total_size=55574528' \
  -F 'tripId=test123'

# Step 2: Upload chunks (can retry failed chunks)
curl -X POST https://celebxmedia.info/api/v2/upload/chunk \
  -F 'upload_id=<upload_id_from_step1>' \
  -F 'part_number=1' \
  -F 'chunk=@chunk1.bin'

# Step 3: Complete and process
curl -X POST https://celebxmedia.info/api/v2/upload/complete \
  -F 'upload_id=<upload_id_from_step1>' \
  -F 'tripId=test123'
```

## Quick Fix Checklist

- [ ] Check nginx configuration (if using reverse proxy)
- [ ] Verify `client_max_body_size` is at least 1G
- [ ] Increase nginx timeouts (300s+ for upload, 1800s+ for processing)
- [ ] Test CORS preflight with curl
- [ ] Check server logs for actual errors
- [ ] Try direct upload (bypassing browser) to isolate issue
- [ ] Consider using chunked upload for large files

## Current Application Settings

- **Max Upload Size:** 1 GB (configurable via `MAX_UPLOAD_SIZE`)
- **Gunicorn Timeout:** 1800 seconds (30 minutes)
- **CORS:** Enabled for all origins with proper headers
- **Streaming:** Enabled with 1 MB chunk size

## Need Help?

If issues persist:
1. Check server logs: `journalctl -u poc2 -f`
2. Test with curl to bypass browser CORS
3. Verify nginx configuration (if applicable)
4. Check network connectivity and stability


# Swagger UI Upload Guide - Fixing "Failed to fetch" Error

## Current Status

✅ **CORS headers added to all error responses** - Deployed  
✅ **Nginx configuration updated** - Timeouts increased, CORS enabled  
✅ **Better filename handling** - Now handles cases where Swagger UI doesn't send filename  

## Common Issues in Swagger UI

### Issue 1: "Failed to fetch" Error

**Cause:** Error responses missing CORS headers (now fixed, but needs deployment)

**Solution:** Deploy the updated code with CORS headers in exception handlers

### Issue 2: "Invalid file extension" Error

**Cause:** 
- File selected doesn't have proper extension
- Swagger UI not preserving filename
- File type not recognized

**Solution:** 
1. Ensure your video file has a proper extension (`.mp4`, `.avi`, `.mov`, `.mkv`)
2. The updated code now tries to infer extension from Content-Type if filename is missing

### Issue 3: "There was an error parsing the body"

**Cause:** 
- Swagger UI sometimes sends malformed multipart data
- Large file uploads timing out

**Solution:**
- Try with a smaller file first to test
- Use chunked upload for large files (>100MB)

## How to Use Swagger UI for Uploads

### Step-by-Step Instructions

1. **Open Swagger UI:**
   ```
   https://celebxmedia.info/docs
   ```

2. **Find the endpoint:**
   - Look for `POST /api/v2/jobs/streaming` under "video-v2" tag

3. **Fill in the form:**
   - **tripId** (required): Enter a unique trip identifier (e.g., "test123")
   - **video** (required): Click "Choose File" and select your `.mp4` file
   - **useMultiprocessing**: Set to `true` or `false`
   - **saveClips**: Set to `false` (unless you need clips)
   - **useMockDetection**: Set to `false` for real detection
   - Other fields are optional

4. **Click "Execute"**

5. **Check the response:**
   - If successful: You'll see the processing results
   - If error: Check the error message in the response

## Troubleshooting in Swagger UI

### If you get "Failed to fetch":

1. **Check browser console** (F12 → Console tab)
   - Look for actual error messages
   - Check network tab for request/response details

2. **Check file size:**
   - Maximum: 1GB
   - For files >100MB, consider using chunked upload

3. **Check file extension:**
   - Must be: `.mp4`, `.avi`, `.mov`, or `.mkv`
   - Ensure the file actually has this extension

4. **Check server logs:**
   ```bash
   ssh root@103.195.244.67
   tail -f /opt/poc2/logs/LocopilotMonitoring.log
   ```

### If you get "Invalid file extension":

1. **Verify file extension:**
   - The file must have `.mp4`, `.avi`, `.mov`, or `.mkv` extension
   - Check the actual filename, not just the file type

2. **Try renaming the file:**
   - Rename to `video.mp4` if it's an MP4 file
   - Ensure the extension matches the actual file format

3. **Check Content-Type:**
   - The updated code will try to infer extension from Content-Type
   - But it's better to have a proper filename

### If upload times out:

1. **Use chunked upload instead:**
   - Go to `POST /api/v2/upload/initiate`
   - Follow the 3-step chunked upload process

2. **Check nginx timeout settings:**
   - Already increased to 30 minutes for processing
   - 5 minutes for upload

## Testing with Swagger UI

### Test with Small File First

1. Use a small test video (< 10MB)
2. Set `useMockDetection=true` for faster testing
3. Set `saveClips=false` to reduce processing time

### Test with Real File

1. Use your actual 53MB file
2. Set `useMockDetection=false` for real detection
3. Set `useMultiprocessing=true` for faster processing
4. Be patient - processing can take 5-15 minutes

## Alternative: Use Chunked Upload in Swagger UI

If streaming upload fails, try chunked upload:

1. **Step 1:** `POST /api/v2/upload/initiate`
   - filename: `actual_v7.mp4`
   - total_size: `55574528` (your file size in bytes)
   - tripId: `your_trip_id`

2. **Step 2:** `POST /api/v2/upload/chunk`
   - upload_id: (from step 1)
   - part_number: `1`
   - chunk: (upload file chunk)

3. **Step 3:** `POST /api/v2/upload/complete`
   - upload_id: (from step 1)
   - Fill in other optional fields

## Quick Fix Checklist

- [ ] Deploy updated code with CORS headers in exception handlers
- [ ] Ensure file has proper extension (`.mp4`, `.avi`, `.mov`, `.mkv`)
- [ ] Check file size is under 1GB
- [ ] Try with smaller file first to test
- [ ] Check browser console for actual error messages
- [ ] Check server logs for detailed errors
- [ ] Consider using chunked upload for large files

## Expected Behavior

### Successful Upload:
```json
{
  "status": "success",
  "trip_id": "your_trip_id",
  "activities": [...],
  "output_path": "...",
  "summary": {...}
}
```

### Error Response (now with CORS headers):
```json
{
  "status": "error",
  "message": "Invalid file extension. Allowed: .mp4, .avi, .mov, .mkv",
  "error": "Invalid file extension. Allowed: .mp4, .avi, .mov, .mkv"
}
```

## Next Steps

1. **Deploy the updated code:**
   ```bash
   ./deploy_to_server.sh
   ```

2. **Restart the service:**
   ```bash
   ssh root@103.195.244.67
   systemctl restart poc2
   ```

3. **Test in Swagger UI:**
   - Go to https://celebxmedia.info/docs
   - Try uploading your 53MB file again
   - Check the response for actual error messages

The "Failed to fetch" error should now show actual error messages that help diagnose the issue!


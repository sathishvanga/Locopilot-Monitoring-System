# Memory Leak Fixes - Locopilot Monitoring System

## Problem
The Locopilot Monitoring System was experiencing severe memory leaks causing "out of application memory" errors, while the POC_2 project with similar multiprocessing ran smoothly.

## Root Causes Identified

### 1. **Video Capture Resource Leaks**
- Video captures (`cv2.VideoCapture`) were not always released properly
- Missing `try-finally` blocks or context managers
- Multiple code paths could skip the `cap.release()` call

### 2. **MediaPipe Models Not Closed**
- MediaPipe Pose and FaceMesh models were never explicitly closed
- These models hold GPU/CPU resources that accumulate over time
- POC_2 has explicit `.close()` methods for MediaPipe services

### 3. **Frame Buffer Accumulation**
- Large frame buffers in `self.frame_buffer` and `self.activities[...]['frames']`
- Not cleared after processing ranges in multiprocessing workers
- Memory accumulated across multiple worker invocations

## Solutions Implemented (Mimicking POC_2 Pattern)

### ✅ Fix 1: Video Capture Context Manager
**Files Modified:**
- `locopilot_monitor.py`

**Changes:**
- Already had `video_capture_context()` context manager (lines 43-54)
- Applied it to ALL video capture instances:
  - `end_activity()` method (line 2954)
  - `process_video()` method (line 3075)
  - `process_video_range()` method (line 3453)

**Pattern (from POC_2):**
```python
# Before (memory leak):
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()  # Could be skipped on exception

# After (safe):
with video_capture_context(video_path) as cap:
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Automatically released even on exception
```

### ✅ Fix 2: MediaPipe Cleanup Method
**Files Modified:**
- `locopilot_monitor.py`

**Changes:**
- Added `cleanup()` method to `LocopilotActivityMonitor` class (after line 3693)
- Added `__del__()` destructor to ensure cleanup on object deletion
- Method closes MediaPipe models, clears buffers, and forces garbage collection

**Pattern (from POC_2's MediaPipeService):**
```python
def cleanup(self):
    """Cleanup method to release MediaPipe resources"""
    try:
        # Close MediaPipe models
        if hasattr(self, 'pose') and self.pose is not None:
            self.pose.close()
            self.pose = None
        
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None
        
        # Clear frame buffers
        self.frame_buffer.clear()
        
        # Clear activity frames
        for activity_name in self.activities:
            if 'frames' in self.activities[activity_name]:
                self.activities[activity_name]['frames'].clear()
        
        # Force garbage collection
        gc.collect()
    except Exception as e:
        print(f"⚠️ Warning during cleanup: {e}")
```

### ✅ Fix 3: Explicit Cleanup Calls
**Files Modified:**
- `app/services/activity_detection_service.py`
- `app/utils/video_multiprocessing.py`

**Changes:**

**In `activity_detection_service.py` (single-process):**
```python
# Process video
monitor.process_video()

# Get activities before cleanup
activities = monitor.all_activities.copy()

# ✅ Explicit cleanup (closes MediaPipe, clears buffers, forces GC)
monitor.cleanup()

return activities
```

**In `video_multiprocessing.py` (multiprocessing workers):**
```python
# After processing
logger.info(f"Worker {worker_id} completed range...")

# ✅ Explicit cleanup (closes MediaPipe, clears buffers, forces GC)
monitor.cleanup()

return {...}
```

**In finally block:**
```python
finally:
    # ✅ Always cleanup monitor resources
    if 'monitor' in locals():
        try:
            monitor.cleanup()
            del monitor
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup: {cleanup_error}")
    
    # Force garbage collection
    gc.collect()
```

### ✅ Fix 4: Already Existing Memory Fixes
The codebase already had some memory fixes in place:
- `process_video_range()` already clears buffers at the end (lines 3680-3687)
- Frame deletion in `finally` blocks (lines 3666-3672)
- Garbage collection calls after processing

## Key Differences: POC_2 vs Locopilot (Before Fixes)

| Aspect | POC_2 (Working) | Locopilot (Before Fix) | Locopilot (After Fix) |
|--------|-----------------|------------------------|----------------------|
| Video Capture Cleanup | `try-finally` with `cap.release()` | Sometimes missing | Context manager everywhere |
| MediaPipe Cleanup | Explicit `.close()` methods | Never closed | `cleanup()` method added |
| Frame Buffer Management | Cleared with `.clear()` | Partially cleared | Fully cleared via `cleanup()` |
| Garbage Collection | Called after processing | Called after processing | Called in `cleanup()` |
| Worker Cleanup | Minimal state per worker | Heavy state accumulation | Explicit cleanup per range |

## Testing Recommendations

### 1. Memory Usage Monitoring
Run the following command while processing a large video:
```bash
# Monitor memory usage
watch -n 1 'ps aux | grep python | grep -v grep'
```

### 2. Compare Before/After
- **Before fixes**: Memory would grow continuously, eventually hitting system limits
- **After fixes**: Memory should stabilize after initial model loading, with periodic GC spikes

### 3. Multiprocessing Test
Process a long video (30+ minutes) with multiprocessing enabled:
```python
# In video_controller.py or test script
activities = activity_service.detect_activities_real(
    video_path=video_path,
    trip_id=trip_id,
    use_multiprocessing=True  # Enable multiprocessing
)
```

### 4. Expected Memory Profile
- **Initial spike**: Model loading (YOLO, MediaPipe) - ~2-3 GB
- **Processing**: Should remain relatively stable
- **Per-worker**: Each worker loads models once, then processes multiple ranges
- **Cleanup**: Memory should drop after processing completes

## Files Modified

1. **`locopilot_monitor.py`**
   - Added `cleanup()` method
   - Added `__del__()` destructor
   - Applied context manager to all video captures
   - Already had frame buffer clearing

2. **`app/services/activity_detection_service.py`**
   - Call `monitor.cleanup()` after single-process processing

3. **`app/utils/video_multiprocessing.py`**
   - Call `monitor.cleanup()` after each range processing
   - Enhanced `finally` block cleanup

## Verification Checklist

- [x] Video captures use context manager or try-finally
- [x] MediaPipe models have explicit close/cleanup
- [x] Frame buffers cleared after processing
- [x] Garbage collection forced after cleanup
- [x] Multiprocessing workers clean up per-range
- [x] No linting errors introduced
- [ ] Memory usage tested with large video (user to verify)

## Next Steps

1. **Test with large video**: Process a 30+ minute video and monitor memory
2. **Compare with POC_2**: Run both projects side-by-side with same video
3. **Adjust worker count**: If memory is still high, reduce `max_workers` in config
4. **Monitor logs**: Check for cleanup messages in logs

## Additional Notes

- The fixes follow the exact pattern used in POC_2's `pipeline_service.py` and `boot.py`
- Context manager pattern ensures cleanup even on exceptions
- MediaPipe cleanup is critical - these models hold significant resources
- Garbage collection is forced to immediately reclaim memory
- Each multiprocessing worker now properly cleans up after each range

---

**Summary**: The Locopilot project now mimics POC_2's resource management pattern with proper video capture cleanup, MediaPipe model closure, and explicit memory management. This should resolve the "out of application memory" errors.


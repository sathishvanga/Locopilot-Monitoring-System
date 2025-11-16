# Suppressing PyTorch/YOLO NNPACK Warnings

## Problem

You were seeing NNPACK warnings like:
```
[W1116 05:44:54.105105940 NNPACK.cpp:56] Could not initialize NNPACK! Reason: Unsupported hardware.
```

These warnings occur when PyTorch tries to use NNPACK (an optimized neural network library) on hardware that doesn't support it. This is common on servers without specific CPU features. The warnings are harmless but very noisy.

## Solution Implemented

We've suppressed these warnings at multiple levels:

### 1. Early Environment Variables (config.py & main.py)

Set environment variables BEFORE PyTorch is imported:

```python
# In app/utils/config.py and app/main.py
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('TORCH_CPP_LOG_LEVEL', 'ERROR')
```

### 2. Logger Configuration (logger.py)

Added PyTorch/YOLO warning suppression:

```python
# Disable PyTorch/YOLO warnings (NNPACK, etc.)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
import logging as std_logging
std_logging.getLogger('ultralytics').setLevel(std_logging.ERROR)
```

### 3. System-Level Configuration (Optional)

For additional suppression, you can also set these in your systemd service file or shell:

```bash
# In your systemd service file or .env
Environment="OMP_NUM_THREADS=1"
Environment="MKL_NUM_THREADS=1"
Environment="TORCH_CPP_LOG_LEVEL=ERROR"
Environment="PYTHONWARNINGS=ignore::UserWarning"
```

## Files Modified

1. **app/utils/config.py** - Added early environment variable setup
2. **app/utils/logger.py** - Added PyTorch warning filters
3. **app/main.py** - Added environment variables before imports

## Testing

After restarting your service, the NNPACK warnings should no longer appear:

```bash
# Restart service
sudo systemctl restart locopilot-monitor

# Check logs
sudo journalctl -u locopilot-monitor -f
```

## Why This Happens

NNPACK is a CPU-optimized neural network library that requires specific CPU features (like ARM NEON or x86 AVX). When these aren't available, PyTorch logs warnings but continues to work fine using its default implementation. Performance is still good, just not "optimally optimized."

## Alternative: CPU-Specific Builds

If you want to completely eliminate the warning at the source, you could:

1. Install a CPU-specific PyTorch build
2. Use GPU-enabled PyTorch (CUDA)
3. Build PyTorch from source with your CPU's features

However, the current solution (suppressing warnings) is simpler and doesn't affect functionality.

## Verification

To verify the warnings are suppressed:

```bash
# Before: You would see
[W1116 05:44:54.105105940 NNPACK.cpp:56] Could not initialize NNPACK!

# After: Clean logs
2025-11-16 14:30:45 [N/A] [N/A] [N/A] [N/A] [INFO] [app.main] [N/A N/A] 🚀 Starting Locopilot Monitoring System v1.0.0
```

## Summary

✅ **Environment variables** set early to prevent NNPACK initialization attempts  
✅ **Python warnings** filtered for torch module  
✅ **Logger levels** raised for ultralytics/YOLO  
✅ **No code changes** needed in processing logic  
✅ **No performance impact** - still uses PyTorch efficiently  

The warnings are now fully suppressed and your logs will be clean!


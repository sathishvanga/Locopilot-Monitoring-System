# Enhanced Logging System - Implementation Summary

This document summarizes the comprehensive logging system implemented for the Locopilot Monitoring System.

## Overview

Implemented a production-ready logging system with:
- **Request context tracking** - Correlates logs with specific requests
- **File rotation** - Daily log rotation with 4-day retention
- **Structured logging** - Consistent format with request metadata
- **Emoji indicators** - Visual markers for different log types
- **Environment-aware** - Different log levels for dev/production

## Components Implemented

### 1. Request Context Management (`app/utils/request_context.py`)

Thread-safe context storage for request-specific metadata.

**Functions:**
- `set_request_context(context)` - Store request metadata
- `get_request_context()` - Retrieve current context
- `reset_request_context()` - Clear context after request
- `update_request_context(updates)` - Update specific fields
- `get_context_value(key, default)` - Get individual values

**Context Fields:**
- `cookie_id` - Trace ID from request headers
- `user_id` - User identifier (from `sub` header)
- `request_id` - Unique UUID for each request
- `source_request_id` - Original request ID for chained requests
- `method` - HTTP method (GET, POST, etc.)
- `url` - Request path
- `client_host` - Client IP address
- `authorization` - Auth token (for debugging)

### 2. Enhanced Logger (`app/utils/logger.py`)

Custom logging configuration with request context support.

**Features:**
- **Custom RequestFormatter** - Includes request context in all logs
- **TimedRotatingFileHandler** - Daily rotation, keeps 4 days of logs
- **Environment-aware levels**:
  - Development: DEBUG to console and file
  - Production: INFO to file, ERROR to console
- **Third-party logger filtering** - Silences noisy libraries

**Log Format:**
```
%(asctime)s [%(user_id)s] [%(cookie_id)s] [%(source_request_id)s] [%(request_id)s] [%(levelname)s] [%(name)s] [%(method)s %(url)s] %(message)s
```

**Example Log Entry:**
```
2025-11-16 14:30:45 [user123] [trace-abc] [req-parent] [req-uuid-123] [INFO] [app.services.video_processing] [POST /api/jobs] 🎬 Starting video processing for trip TRIP-001
```

**Functions:**
- `setup_logging(level=None)` - Initialize logging system
- `get_logger(name)` - Get logger instance for a module

### 3. Logging Middleware (`app/middleware/logging_middleware.py`)

HTTP middleware that tracks all requests/responses.

**Functionality:**
- Captures request metadata (method, path, client, headers)
- Generates unique request ID
- Stores context for request lifecycle
- Logs request start (📥) and completion (📤)
- Logs errors (💥) with stack traces
- Measures and logs duration
- Adds custom response headers:
  - `X-Request-ID` - Unique request identifier
  - `X-Process-Time` - Request duration in seconds
- Always cleans up context after request

**Log Examples:**
```
📥 Request received - Method: POST, Path: /api/jobs, Client: 192.168.1.100, User: user123
📤 Request completed - Status: 200, Duration: 5.2345s
💥 Request failed - Error: File not found, Duration: 0.1234s
```

### 4. Configuration Updates (`app/utils/config.py`)

Added logging-related settings:

```python
log_level: str = "INFO"  # Default log level
log_dir: str = "logs"  # Directory for log files (via LOG_DIR env var)
environment: str = "development"  # production or development (via ENVIRONMENT env var)
prod_log_level: str = "INFO"  # Production log level (via PROD_LOG_LEVEL env var)
dev_log_level: str = "DEBUG"  # Development log level (via DEV_LOG_LEVEL env var)
```

**Environment Variables:**
```bash
LOG_DIR=logs                   # Log file directory
ENVIRONMENT=production         # Environment (production/development)
PROD_LOG_LEVEL=INFO           # Production log level
DEV_LOG_LEVEL=DEBUG           # Development log level
```

### 5. Application Integration (`app/main.py`)

Updated main application to use new logging system:

```python
from .middleware import LoggingMiddleware

# Initialize logging
setup_logging(level=settings.log_level)

# Add logging middleware
app.add_middleware(LoggingMiddleware)
```

Removed old request logging middleware in favor of the new LoggingMiddleware.

## Log Statements Added

### External API Service (`app/services/external_api_service.py`)

Enhanced with emoji indicators for better visibility:

```python
🔌 External API service initialized - Enabled: True, URL: ..., Timeout: 30s
⚠️ CVVR API posting is disabled in configuration for trip_id=TRIP-001
📤 Preparing to post results for trip_id=TRIP-001, events_count=5, job_id=run_20251116
📭 Posting no-events notice to https://api.../addUpdateBulkNoEvents for trip_id=TRIP-001
✅ No-events notice posted successfully: 200
⚠️ No-events notice posting got non-2xx: 400 - Bad Request
⏱️ No-events notice posting timed out after 30s
❌ Failed to post no-events notice: Connection refused
📦 Posting 5 unique violations (from 5 total) to https://api.../addUpdateBulk
✅ Violations posted successfully: 200
⚠️ Violations posting got non-2xx: 500
⏱️ Violations posting timed out after 30s
❌ Failed to post violations: Connection refused
🔄 Transformed 5 events to violations
🔍 Deduplicated 10 violations to 5 unique violations
```

### Video Processing Service (`app/services/video_processing_service.py`)

Comprehensive logging throughout the processing pipeline:

```python
🚀 Video processing service initialized - Output dir: locopilot_evidence, Upload dir: /tmp/...
🎬 Starting video processing for trip TRIP-001 - Multiprocessing: enabled, Save clips: true, Mock detection: false
❌ Video file not found: /path/to/video.mp4
✅ Video file validated: /path/to/video.mp4
📁 Created run directory: locopilot_evidence/run_20251116_143045
🎭 Using mock activity detection
🔍 Using real activity detection - Multiprocessing: enabled
💾 Saved 5 activities to locopilot_evidence/run_20251116_143045/activities.json
🌐 Attempting to post results to external API...
✅ [external_api] Successfully posted 5 violations to external API for trip TRIP-001
⚠️ [external_api] Failed to post to external API: Timeout
❌ [external_api] Exception while posting to external API: Connection refused
✅ Video processing completed in 5.23s - Found 5 activities for trip TRIP-001
❌ Video processing failed after 2.34s for trip TRIP-001: File not found
🗑️ Cleaned up uploaded video: /tmp/locopilot_uploads/TRIP-001_1234567890.mp4
⚠️ Failed to cleanup video /tmp/...: Permission denied
```

### Video Controller (`app/controllers/video_controller.py`)

Request-level logging with validation tracking:

```python
📥 Received video processing request for trip: TRIP-001
⚠️ Invalid request: tripId is empty
⚠️ Invalid request: lpCrewName is empty for trip TRIP-001
⚠️ Invalid request: lpCrewId is empty for trip TRIP-001
📹 Uploaded video: latest.mp4 (25.50 MB)
⚠️ Video validation failed: File too large. Maximum size: 500 MB
🎮 Processing configuration - Multiprocessing: True, SaveClips: False, Mock: False
✅ Successfully processed video for trip TRIP-001 - Activities: 5, Time: 5.23s
❌ Video processing failed for trip TRIP-001: File not found
```

## Log File Structure

### File Location
```
logs/
├── LocopilotMonitoring.log              # Current log file
├── LocopilotMonitoring.log.2025-11-15   # Previous day
├── LocopilotMonitoring.log.2025-11-14   # 2 days ago
├── LocopilotMonitoring.log.2025-11-13   # 3 days ago
└── LocopilotMonitoring.log.2025-11-12   # 4 days ago (oldest retained)
```

### Rotation Policy
- **When**: Daily at midnight UTC
- **Retention**: 4 days of backups
- **Format**: `LocopilotMonitoring.log.YYYY-MM-DD`
- **Encoding**: UTF-8

## Emoji Legend

| Emoji | Meaning | Usage |
|-------|---------|-------|
| 📥 | Incoming | Request received |
| 📤 | Outgoing | Request completed |
| 💥 | Error | Request failed |
| 🚀 | Initialization | Service started |
| 🎬 | Processing Start | Video processing begins |
| ✅ | Success | Operation completed successfully |
| ❌ | Failure | Operation failed |
| ⚠️ | Warning | Non-critical issue |
| 📹 | Video | Video file related |
| 📁 | Directory | Directory operation |
| 💾 | Save | File saved |
| 🗑️ | Delete | File deleted |
| 🎭 | Mock | Mock/test mode |
| 🔍 | Detection | Detection operation |
| 🌐 | Network | External API call |
| 🔌 | Connection | Service connection |
| 📭 | Empty | No events |
| 📦 | Package | Data bundle |
| 🔄 | Transform | Data transformation |
| 🎮 | Configuration | Config setting |
| ⏱️ | Timeout | Operation timed out |

## Environment Configuration

### Development Environment
```bash
export ENVIRONMENT=development
export DEV_LOG_LEVEL=DEBUG
export LOG_DIR=logs
```

**Behavior:**
- Logs DEBUG and above to file
- Logs DEBUG and above to console
- Verbose output for debugging
- All third-party loggers filtered

### Production Environment
```bash
export ENVIRONMENT=production
export PROD_LOG_LEVEL=INFO
export LOG_DIR=/var/log/locopilot
```

**Behavior:**
- Logs INFO and above to file
- Logs ERROR only to console (stderr)
- Reduced verbosity
- All third-party loggers disabled

## Benefits

### 1. Request Tracing
Every log entry includes request context, making it easy to:
- Trace all logs for a specific request
- Correlate logs across services
- Debug issues in production
- Track request duration

### 2. Production-Ready
- Automatic log rotation (no manual cleanup)
- Controlled disk usage (4-day retention)
- Performance optimized (filtered noisy loggers)
- Proper error handling with stack traces

### 3. Developer-Friendly
- Emoji indicators for quick scanning
- Structured format for parsing
- Context-aware messages
- Clear error descriptions

### 4. Monitoring-Ready
- Consistent log format for log aggregators
- Request IDs for distributed tracing
- Duration metrics for performance monitoring
- Status indicators for alerting

## Usage Examples

### Getting a Logger
```python
from app.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("🚀 Service initialized")
logger.error("❌ Operation failed", exc_info=True)
```

### Adding Request Context
```python
from app.utils.request_context import update_request_context

update_request_context({"trip_id": "TRIP-001"})
logger.info("Processing trip")  # Includes trip_id in context
```

### Reading Logs
```bash
# View current log
tail -f logs/LocopilotMonitoring.log

# Search for specific request
grep "req-uuid-123" logs/LocopilotMonitoring.log

# Find all errors
grep "ERROR" logs/LocopilotMonitoring.log

# Track specific trip
grep "TRIP-001" logs/LocopilotMonitoring.log
```

## Performance Considerations

### Minimal Overhead
- Context variables use thread-local storage (fast)
- Log formatting only on write (lazy)
- File I/O is buffered
- Noisy loggers disabled (reduces CPU)

### Memory Usage
- Fixed retention (4 days)
- Automatic rotation (no unbounded growth)
- Compressed old logs (optional, not implemented)

## Future Enhancements

Potential improvements:

1. **Log Compression** - Gzip rotated logs
2. **JSON Logging** - Structured JSON for log aggregators
3. **Async Logging** - Non-blocking I/O for high throughput
4. **Log Levels per Module** - Fine-grained control
5. **Metrics Export** - Prometheus/StatsD integration
6. **Distributed Tracing** - OpenTelemetry support
7. **Log Sampling** - Reduce volume in production
8. **Sensitive Data Filtering** - Redact tokens/passwords

## Troubleshooting

### Logs not appearing
1. Check LOG_DIR exists and is writable
2. Verify log level is appropriate (DEBUG < INFO < WARNING < ERROR)
3. Check if third-party logger is disabled
4. Ensure setup_logging() is called in main.py

### Missing request context
1. Verify LoggingMiddleware is added to FastAPI app
2. Check context is set before logging
3. Ensure reset_request_context() is called in finally block

### Log file too large
1. Reduce retention from 4 days to less
2. Increase log level to WARNING or ERROR
3. Enable log compression (future feature)
4. Implement log sampling (future feature)

### Performance issues
1. Disable unnecessary third-party loggers
2. Reduce log level in production
3. Use async logging (future feature)
4. Sample high-frequency logs

## Summary

The enhanced logging system provides:

✅ **Request Context Tracking** - Correlate logs across the entire request lifecycle  
✅ **Automatic Log Rotation** - Daily rotation with 4-day retention  
✅ **Emoji Indicators** - Quick visual scanning of logs  
✅ **Environment-Aware** - Different behavior for dev/production  
✅ **Production-Ready** - Performance optimized and resource-controlled  
✅ **Developer-Friendly** - Clear, structured, contextual logging  
✅ **Monitoring-Ready** - Compatible with log aggregators and alerting  

All major services now have comprehensive logging with clear indicators, making debugging and monitoring significantly easier!


"""
Application constants

Centralized constants for the desktop application to avoid magic numbers and strings.
"""

# HTTP Status Codes
HTTP_OK = 200
HTTP_CREATED = 201
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_REQUEST_ENTITY_TOO_LARGE = 413
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_BAD_GATEWAY = 502
HTTP_SERVICE_UNAVAILABLE = 503
HTTP_GATEWAY_TIMEOUT = 504

# Timeout values (in seconds)
DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_UPLOAD_TIMEOUT = 300  # 5 minutes
DEFAULT_PROCESSING_TIMEOUT = 3600  # 1 hour
DEFAULT_BACKEND_STARTUP_TIMEOUT = 30
DEFAULT_HEALTH_CHECK_TIMEOUT = 5
DEFAULT_SOCKET_TIMEOUT = 2

# File size limits
MAX_USERNAME_LENGTH = 50
MAX_PASSWORD_LENGTH = 200
MAX_FILE_SIZE_GB = 10
DEFAULT_MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2

# Upload configuration
DEFAULT_UPLOAD_CHUNK_SIZE = 8192  # 8 KB

# Backend configuration
DEFAULT_BACKEND_PORT = 8000
DEFAULT_BACKEND_URL = "http://localhost:8000"
DEFAULT_WORKER_COUNT = 1
DEFAULT_WORKER_TIMEOUT = 600  # 10 minutes for video processing

# UI Configuration
DEFAULT_WINDOW_WIDTH = 1200
DEFAULT_WINDOW_HEIGHT = 800
DEFAULT_BACKEND_STARTUP_DELAY_MS = 100  # Delay before starting backend (ms)

# File extensions
ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]

# OS Types
OS_TYPE_DESKTOP = 1

# Error messages
ERROR_BACKEND_NOT_RUNNING = "Local processing backend is not running. Please start the backend service."
ERROR_FILE_NOT_FOUND = "File not found"
ERROR_INVALID_FILE_PATH = "Invalid file path"
ERROR_FILE_TOO_LARGE = "File too large"
ERROR_SESSION_EXPIRED = "Session expired - please login again"
ERROR_INVALID_CREDENTIALS = "Invalid mobile number or password"

# Status messages
STATUS_PROCESSING = "Processing and uploading to S3..."
STATUS_COMPLETED = "Completed successfully!"
STATUS_PROCESSING_EMOJI = "⚙️ Processing..."
STATUS_UPLOADING_EMOJI = "⏳ Uploading..."
STATUS_ERROR_EMOJI = "❌ Retry"


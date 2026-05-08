"""
Enhanced logging configuration with request context support

Provides structured logging with request tracking, file rotation,
and custom formatting for production environments.

Configuration:
    ENABLE_CONSOLE_LOGS: Set to "1" to enable console output (default: disabled)
    LOG_LEVEL: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
"""

import os
import logging
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

from .request_context import get_request_context
from .config import get_settings


settings = get_settings()

# Create logs directory if it doesn't exist
log_dir = settings.log_dir
os.makedirs(log_dir, exist_ok=True)

# Check if console logging is enabled (disabled by default for clean terminal output)
ENABLE_CONSOLE_LOGS = os.getenv("ENABLE_CONSOLE_LOGS", "0") == "1"


# Sensitive field names that must never appear in logs in cleartext.
# Match is case-insensitive on the key. Values are replaced with "***".
SENSITIVE = frozenset({
    "authorization",
    "authentication",
    "auth_token",
    "authtoken",
    "bearer",
    "cookie",
    "set-cookie",
    "csrf",
    "x-api-key",
    "api_key",
    "secret_key",
    "password",
    "minio_secret_key",
})


class RedactFilter(logging.Filter):
    """
    Logging filter that strips Authorization-like fields from log records.

    Operates on three surfaces:
    1. Any attribute on the LogRecord whose key matches a SENSITIVE term
       (covers ``extra={...}`` dicts passed to logger calls).
    2. ``record.args`` when it is a dict-style mapping (logger.info("%s", {...})).
    3. ``record.msg`` itself when it contains a sensitive keyword — the
       keyword is replaced with ``<term>=***`` so the value cannot be read
       even if the formatter interpolates it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Redact extras and any attributes whose key matches a sensitive term.
        for key in list(record.__dict__.keys()):
            if key.lower() in SENSITIVE:
                setattr(record, key, "***")

        # Redact mapping-style args (logger.info("%(authorization)s", {...}))
        if isinstance(record.args, dict):
            redacted_args = {}
            for k, v in record.args.items():
                if isinstance(k, str) and k.lower() in SENSITIVE:
                    redacted_args[k] = "***"
                else:
                    redacted_args[k] = v
            record.args = redacted_args
        # Redact positional tuple args (logger.info("auth=%s", token)). We only
        # rewrite individual elements whose stringified form contains a
        # sensitive token — this keeps the filter conservative and avoids
        # mutating unrelated arguments. The format string in ``record.msg``
        # itself is independently scrubbed below.
        elif isinstance(record.args, tuple):
            new_args = []
            mutated = False
            for item in record.args:
                try:
                    item_str = str(item)
                except Exception:
                    new_args.append(item)
                    continue
                lowered = item_str.lower()
                if any(term in lowered for term in SENSITIVE):
                    new_args.append("***")
                    mutated = True
                else:
                    new_args.append(item)
            if mutated:
                record.args = tuple(new_args)

        # Best-effort scan of msg for sensitive substrings. We replace the
        # term itself with ``term=***`` so any trailing token printed alongside
        # is visually associated with the redaction marker rather than
        # printed verbatim.
        if isinstance(record.msg, str):
            lower_msg = record.msg.lower()
            for term in SENSITIVE:
                if term in lower_msg:
                    # Case-insensitive replacement: walk through and replace
                    # each occurrence with the redaction marker.
                    idx = 0
                    new_parts = []
                    msg = record.msg
                    msg_lower = msg.lower()
                    while True:
                        found = msg_lower.find(term, idx)
                        if found == -1:
                            new_parts.append(msg[idx:])
                            break
                        new_parts.append(msg[idx:found])
                        new_parts.append(term + "=***")
                        idx = found + len(term)
                    record.msg = "".join(new_parts)
                    msg_lower = record.msg.lower()
        return True


class RequestFormatter(logging.Formatter):
    """
    Custom formatter that includes request context in log messages

    Extracts context metadata (user_id, request_id, etc.) and includes
    it in formatted log output.
    """

    def format(self, record):
        """
        Format log record with request context

        Args:
            record: Log record to format

        Returns:
            Formatted log string
        """
        context = get_request_context()
        record.cookie_id = context.get("cookie_id", "N/A")
        record.user_id = context.get("user_id", "N/A")
        record.url = context.get("url", "N/A")
        record.method = context.get("method", "N/A")
        record.request_id = context.get("request_id", "N/A")
        record.source_request_id = context.get("source_request_id", "N/A")
        return super().format(record)


def setup_logging(level: Optional[str] = None) -> None:
    """
    Setup application logging with file rotation and request tracking

    Configures:
    - Root logger with appropriate level
    - File handler with daily rotation (4-day retention)
    - Optional console handler (disabled by default, enable with ENABLE_CONSOLE_LOGS=1)
    - Custom formatter with request context
    - Disables noisy third-party loggers

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
              If None, uses settings based on environment

    Environment Variables:
        ENABLE_CONSOLE_LOGS: Set to "1" to enable console output (default: disabled)
    """
    # Configure root logger
    root_logger = logging.getLogger()

    # Determine log level based on environment or override
    if level:
        log_level = getattr(logging, level.upper())
    else:
        if settings.environment == "production":
            log_level = getattr(logging, settings.prod_log_level.upper())
        else:
            log_level = getattr(logging, settings.dev_log_level.upper())

    root_logger.setLevel(log_level)

    # Disable noisy third-party loggers
    noisy_loggers = [
        "httpcore.connection",
        "httpcore.http11",
        "openai._base_client",
        "langsmith",
        "langsmith._internal._serde",
        "urllib3.connectionpool",
        "langsmith.client",
        "asyncio",
        "multipart.multipart",
        "PIL.PngImagePlugin",
        "PIL.TiffImagePlugin"
    ]

    # Disable PyTorch/YOLO warnings (NNPACK, etc.)
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch')
    import logging as std_logging
    std_logging.getLogger('ultralytics').setLevel(std_logging.ERROR)

    # Set environment variable to suppress NNPACK warnings
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).disabled = True

    # Create log file path
    log_file_path = os.path.join(log_dir, "LocopilotMonitoring.log")

    # Custom formatter with request context
    formatter = RequestFormatter(
        "%(asctime)s [%(user_id)s] [%(cookie_id)s] [%(source_request_id)s] "
        "[%(request_id)s] [%(levelname)s] [%(name)s] [%(method)s %(url)s] %(message)s"
    )

    # File handler with daily rotation (keeps 4 days of logs)
    file_handler = TimedRotatingFileHandler(
        filename=log_file_path,
        when="midnight",
        interval=1,
        backupCount=4,
        encoding="utf-8",
        utc=True,
    )
    file_handler.setFormatter(formatter)

    # Only add handler if not already present (avoid duplicates).
    # Match by the absolute path of the project log file rather than
    # handler type — a legacy plain ``FileHandler`` pointing at the same
    # path would otherwise slip through and double-write every line.
    target_path = file_handler.baseFilename
    if not any(
        getattr(h, "baseFilename", None) == target_path
        for h in root_logger.handlers
    ):
        root_logger.addHandler(file_handler)

    # Install the redaction filter on the root logger so every record
    # propagated up the hierarchy gets scrubbed once before formatting,
    # regardless of which module created the logger.
    if not any(isinstance(f, RedactFilter) for f in root_logger.filters):
        root_logger.addFilter(RedactFilter())

    # Console handler - DISABLED by default for clean terminal output
    # Set ENABLE_CONSOLE_LOGS=1 to enable console logging for debugging
    if ENABLE_CONSOLE_LOGS:
        stream_handler = logging.StreamHandler()
        if settings.environment == "production":
            stream_handler.setLevel(logging.ERROR)
        else:
            stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)

        if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
            root_logger.addHandler(stream_handler)

    # Log initialization (to file only)
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging initialized - Level: {logging.getLevelName(log_level)}, "
        f"Environment: {settings.environment}, Log file: {log_file_path}, "
        f"Console output: {'enabled' if ENABLE_CONSOLE_LOGS else 'disabled'}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)

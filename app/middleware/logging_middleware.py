"""
Logging Middleware - Tracks and logs all HTTP requests/responses

Provides comprehensive request/response logging with context tracking,
duration measurement, and structured log formatting.
"""

import uuid
from datetime import datetime
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ..utils.request_context import set_request_context, reset_request_context
from ..utils.logger import get_logger


logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses
    
    Captures request metadata, measures duration, and logs
    all incoming/outgoing traffic with structured context.
    """
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request and log details
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/handler in chain
            
        Returns:
            Response from downstream handler
        """
        request_start_time = datetime.now()

        # Extract request metadata
        trace_id = request.headers.get("traceid", "N/A")
        user_id = request.headers.get("sub", "N/A")
        unique_request_id = str(uuid.uuid4())
        http_method = request.method
        request_path = request.url.path.rstrip("/")
        # SECURITY (task 0010): NEVER store the raw Authorization header in
        # request context. The context dict is interpolated into log lines
        # via the request formatter — capturing the bearer here would write
        # it to disk on every request. We only record presence/absence.
        has_auth = bool(request.headers.get("Authorization"))
        source_request_id = request.headers.get("source_request_id", "N/A")
        client_host = request.client.host if request.client else "unknown"

        # Store request context for logging throughout request lifecycle
        set_request_context({
            "cookie_id": trace_id,
            "user_id": user_id,
            "method": http_method,
            "url": request_path,
            "request_id": unique_request_id,
            "authorization": "***" if has_auth else "None",
            "source_request_id": source_request_id,
            "client_host": client_host
        })

        # Log incoming request
        logger.info(
            f"[REQ] Request received - Method: {http_method}, Path: {request_path}, "
            f"Client: {client_host}, User: {user_id}"
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate request duration
            request_duration = (datetime.now() - request_start_time).total_seconds()
            
            # Log successful response
            logger.info(
                f"[RES] Request completed - Status: {response.status_code}, "
                f"Duration: {request_duration:.4f}s"
            )
            
            # Add custom headers
            response.headers["X-Request-ID"] = unique_request_id
            response.headers["X-Process-Time"] = f"{request_duration:.4f}"
            
            return response
        
        except Exception as e:
            # Calculate duration for failed request
            request_duration = (datetime.now() - request_start_time).total_seconds()
            
            # Log error
            logger.error(
                f"[ERR] Request failed - Error: {str(e)}, "
                f"Duration: {request_duration:.4f}s",
                exc_info=True
            )
            
            raise
        
        finally:
            # Always reset context after request
            reset_request_context()


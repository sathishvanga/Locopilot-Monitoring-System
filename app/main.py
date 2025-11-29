"""
Main FastAPI application - Locopilot Monitoring System

A production-ready API for video processing and activity detection.
"""

import os

# Suppress PyTorch/YOLO warnings BEFORE any imports
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('TORCH_CPP_LOG_LEVEL', 'ERROR')

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .controllers import video_router
from .controllers.v2_video_controller import router as v2_video_router
from .middleware import LoggingMiddleware
from .utils.logger import setup_logging, get_logger
from .utils.config import get_settings
from .utils.video_multiprocessing import shutdown_shared_pool


# Initialize settings and logging
settings = get_settings()
setup_logging(level=settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    
    Handles startup and shutdown events for resource initialization
    and cleanup.
    """
    # Startup
    logger.info("=" * 60)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {'Development' if settings.debug else 'Production'}")
    logger.info(f"Output directory: {settings.output_dir}")
    logger.info(f"Upload directory: {settings.upload_dir}")
    logger.info(f"Sample FPS: {settings.sample_fps}")
    logger.info(f"YOLO weights: {settings.yolo_weights}")
    logger.info("=" * 60)
    
    # Preload models if configured
    if settings.preload_ocr:
        logger.info("OCR preloading enabled")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")

    # Ensure shared multiprocessing pool is shut down cleanly
    try:
        shutdown_shared_pool(wait=True)
    except Exception as e:
        logger.warning(
            f"Error while shutting down shared multiprocessing pool: {e}",
            exc_info=True,
        )


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="""
    ## Locopilot Monitoring System API
    
    A production-ready FastAPI application for video processing and activity detection.
    
    ### Features
    - **Video Upload**: Upload videos for processing
    - **Activity Detection**: Detect various activities (cell phone usage, sleep, writing, etc.)
    - **Structured Output**: Generate activities.json with detailed metadata
    - **Scalable Architecture**: Clean MVC pattern with separate layers
    
    ### Architecture
    - **Models**: Pydantic schemas for validation
    - **Controllers**: API route handlers
    - **Services**: Business logic and ML processing
    - **Repositories**: Data persistence (JSON files)
    - **Utils**: Configuration and logging
    
    ### Usage
    1. Upload a video with tripId via `/api/v1/video/process`
    2. Receive processing results with detected activities
    3. Access activities.json file in the output directory
    """,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware for request/response tracking
app.add_middleware(LoggingMiddleware)


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "error": str(exc.detail)
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Validation error",
            "errors": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    error_msg = str(exc) if exc else "Unknown error"
    logger.error(f"Unexpected error: {error_msg}", exc_info=True)
    
    # Include error details in response for better debugging
    # In production, we still want to see the error for desktop app debugging
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": f"Internal server error: {error_msg}",
            "error": error_msg,
            "detail": error_msg
        }
    )


# Register routers
app.include_router(video_router)
app.include_router(v2_video_router)


# Root endpoint
@app.get(
    "/",
    tags=["root"],
    summary="API Root",
    description="Welcome endpoint with API information"
)
async def root():
    """
    API root endpoint
    
    Returns basic information about the API.
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/v1/video/health"
    }


# Health check endpoint
@app.get(
    "/health",
    tags=["health"],
    summary="Application Health Check",
    description="Check overall application health"
)
async def health():
    """
    Application health check
    
    Returns application health status.
    """
    return {
        "status": "healthy",
        "application": settings.app_name,
        "version": settings.app_version
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )


"""
Video controller - API endpoints for video upload and processing

Handles HTTP requests and responses for video processing operations.
"""

from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from ..models.video_models import VideoProcessingResponse, VideoProcessingError
from ..services.video_processing_service import VideoProcessingService
from ..utils.logger import get_logger
from ..utils.config import get_settings


logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api/v1/video", tags=["video"])

# Initialize service (singleton pattern)
video_processing_service = VideoProcessingService()


@router.post(
    "/process",
    response_model=VideoProcessingResponse,
    responses={
        200: {"description": "Video processed successfully"},
        400: {"description": "Invalid request or file"},
        500: {"description": "Internal server error"}
    },
    summary="Process uploaded video",
    description="""
    Upload and process a video file for activity detection.
    
    The endpoint accepts:
    - video: Video file (multipart/form-data)
    - tripId: Unique trip identifier (required)
    - crewName: Crew member name (optional, default: "John Doe")
    - crewId: Crew member ID (optional, default: "C-001")
    - crewRole: Crew role (optional, default: 1)
    - useMockDetection: Use mock detection for testing (optional, default: false)
    - useMultiprocessing: Enable parallel processing (optional, default: from config)
    - saveClips: Generate video clips and images (optional, default: true)
    
    Returns:
    - Processing results with detected activities
    - Path to activities.json file
    - Summary statistics
    """
)
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Video file to process"),
    tripId: str = Form(..., description="Unique trip identifier"),
    crewName: Optional[str] = Form(default="John Doe", description="Crew member name"),
    crewId: Optional[str] = Form(default="C-001", description="Crew member ID"),
    crewRole: Optional[int] = Form(default=1, description="Crew role (1 = primary pilot)"),
    useMockDetection: Optional[bool] = Form(default=False, description="Use mock detection for testing"),
    useMultiprocessing: Optional[bool] = Form(default=None, description="Enable multiprocessing (default: from config)"),
    saveClips: Optional[bool] = Form(default=True, description="Generate video clips and images (default: true)")
):
    """
    Process uploaded video and detect activities
    
    This endpoint handles video upload, validation, processing, and activity detection.
    The video is saved temporarily, processed, and then optionally cleaned up.
    """
    video_path = None
    
    try:
        logger.info(f"Received video processing request for trip: {tripId}")
        
        # Validate tripId
        if not tripId or not tripId.strip():
            raise HTTPException(
                status_code=400,
                detail="tripId is required and cannot be empty"
            )
        
        # Read video content
        video_content = await video.read()
        file_size = len(video_content)
        
        logger.info(f"Uploaded video: {video.filename} ({file_size} bytes)")
        
        # Validate video file
        is_valid, error_message = video_processing_service.validate_video_file(
            filename=video.filename,
            file_size=file_size
        )
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
        
        # Save uploaded video
        video_path = await video_processing_service.save_uploaded_video(
            file_content=video_content,
            filename=video.filename,
            trip_id=tripId
        )
        
        # Determine multiprocessing setting
        # Priority: request parameter > config setting > default (False)
        use_mp = useMultiprocessing if useMultiprocessing is not None else settings.enable_multiprocessing
        
        logger.info(f"Processing with multiprocessing: {use_mp}, save_clips: {saveClips}")
        
        # Process video (synchronous for now, can be made async)
        result = video_processing_service.process_video(
            video_path=video_path,
            trip_id=tripId,
            crew_name=crewName,
            crew_id=crewId,
            crew_role=crewRole,
            use_mock_detection=useMockDetection,
            use_multiprocessing=use_mp,
            save_clips=saveClips
        )
        
        # Schedule cleanup of uploaded video (optional)
        # Uncomment to enable automatic cleanup after processing
        # background_tasks.add_task(
        #     video_processing_service.cleanup_uploaded_video,
        #     video_path
        # )
        
        logger.info(f"Successfully processed video for trip {tripId}")
        
        return VideoProcessingResponse(**result)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}", exc_info=True)
        
        # Cleanup on error
        if video_path:
            video_processing_service.cleanup_uploaded_video(video_path)
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process video: {str(e)}"
        )


@router.get(
    "/status/{run_id}",
    summary="Get processing status",
    description="Get the processing status for a specific run ID"
)
async def get_processing_status(run_id: str):
    """
    Get processing status for a run
    
    Args:
        run_id: Run directory name (e.g., run_20251110_143045)
    """
    try:
        import os
        
        run_dir = os.path.join(settings.output_dir, run_id)
        
        if not os.path.exists(run_dir):
            raise HTTPException(
                status_code=404,
                detail=f"Run not found: {run_id}"
            )
        
        status = video_processing_service.get_processing_status(run_dir)
        
        return JSONResponse(content=status)
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Failed to get status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get status: {str(e)}"
        )


@router.get(
    "/health",
    summary="Health check",
    description="Check if the video processing service is healthy"
)
async def health_check():
    """
    Health check endpoint
    
    Returns service status and configuration.
    """
    import multiprocessing as mp
    
    return {
        "status": "healthy",
        "service": "video-processing",
        "version": settings.app_version,
        "config": {
            "max_upload_size_mb": settings.max_upload_size / (1024 * 1024),
            "allowed_extensions": settings.allowed_video_extensions,
            "sample_fps": settings.sample_fps,
            "output_dir": settings.output_dir,
            "multiprocessing": {
                "enabled": settings.enable_multiprocessing,
                "chunk_duration": settings.mp_chunk_duration,
                "max_workers": settings.mp_max_workers,
                "max_workers_cap": settings.mp_max_workers_cap,
                "cpu_count": mp.cpu_count()
            }
        }
    }


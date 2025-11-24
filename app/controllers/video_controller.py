"""
Video controller - API endpoints for video upload and processing

Handles HTTP requests and responses for video processing operations.
"""

from typing import Optional
import os

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Response
from fastapi.responses import JSONResponse

from ..models.video_models import VideoProcessingResponse, VideoProcessingError
from ..services.video_processing_service import VideoProcessingService
from ..utils.logger import get_logger
from ..utils.config import get_settings


logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api", tags=["video"])

# Initialize service (singleton pattern)
video_processing_service = VideoProcessingService()


@router.post(
    "/jobs",
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
    - lpCrewName: LP crew member name (required)
    - lpCrewId: LP crew member ID (required)
    - alpCrewName: ALP crew member name (optional)
    - alpCrewId: ALP crew member ID (optional)
    - useMockDetection: Use mock detection for testing (optional, default: false)
    - useMultiprocessing: Enable parallel processing (optional, default: from config)
    - saveClips: Save annotated frames for debugging (optional, default: false). Clips and images are always saved for UI evidence.
    
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
    lpCrewName: str = Form(..., description="Loco Pilot crew member name"),
    lpCrewId: str = Form(..., description="Loco Pilot crew member ID"),
    alpCrewName: Optional[str] = Form(default=None, description="Assistant Loco Pilot crew member name"),
    alpCrewId: Optional[str] = Form(default=None, description="Assistant Loco Pilot crew member ID"),
    useMockDetection: Optional[bool] = Form(default=False, description="Use mock detection for testing"),
    useMultiprocessing: Optional[bool] = Form(default=None, description="Enable multiprocessing (default: from config)"),
    saveClips: Optional[bool] = Form(default=False, description="Save annotated frames for debugging (default: false). Clips/images always saved.")
):
    """
    Process uploaded video and detect activities
    
    This endpoint handles video upload, validation, processing, and activity detection.
    The video is saved temporarily, processed, and then optionally cleaned up.
    """
    video_path = None
    
    try:
        logger.info(f"📥 Received video processing request for trip: {tripId}")
        
        # Validate tripId
        if not tripId or not tripId.strip():
            logger.warning(f"⚠️ Invalid request: tripId is empty")
            raise HTTPException(
                status_code=400,
                detail="tripId is required and cannot be empty"
            )
        
        # Validate LP crew (required)
        if not lpCrewName or not lpCrewName.strip():
            logger.warning(f"⚠️ Invalid request: lpCrewName is empty for trip {tripId}")
            raise HTTPException(
                status_code=400,
                detail="lpCrewName is required and cannot be empty"
            )
        if not lpCrewId or not lpCrewId.strip():
            logger.warning(f"⚠️ Invalid request: lpCrewId is empty for trip {tripId}")
            raise HTTPException(
                status_code=400,
                detail="lpCrewId is required and cannot be empty"
            )
        
        # Build crew members dictionary
        crew_members = {}
        
        # Add LP crew (required)
        crew_members['LP'] = {
            'name': lpCrewName.strip(),
            'id': lpCrewId.strip(),
            'role': 'LP'
        }
        logger.info(f"LP Crew: {lpCrewName} ({lpCrewId})")
        
        # Add ALP crew if provided
        if alpCrewName and alpCrewId:
            if alpCrewName.strip() and alpCrewId.strip():
                crew_members['ALP'] = {
                    'name': alpCrewName.strip(),
                    'id': alpCrewId.strip(),
                    'role': 'ALP'
                }
                logger.info(f"ALP Crew: {alpCrewName} ({alpCrewId})")
        
        # Read video content
        video_content = await video.read()
        file_size = len(video_content)
        
        logger.info(f"📹 Uploaded video: {video.filename} ({file_size / (1024*1024):.2f} MB)")
        
        # Validate video file
        is_valid, error_message = video_processing_service.validate_video_file(
            filename=video.filename,
            file_size=file_size
        )
        
        if not is_valid:
            logger.warning(f"⚠️ Video validation failed: {error_message}")
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
        
        logger.info(
            f"🎮 Processing configuration - "
            f"Multiprocessing: {use_mp}, SaveClips: {saveClips}, Mock: {useMockDetection}"
        )
        
        # Process video (synchronous for now, can be made async)
        result = video_processing_service.process_video(
            video_path=video_path,
            trip_id=tripId,
            crew_members=crew_members,  # Pass crew members dict
            crew_name=lpCrewName,  # Use LP as default
            crew_id=lpCrewId,  # Use LP as default
            crew_role=1,  # LP role
            use_mock_detection=useMockDetection,
            use_multiprocessing=use_mp,
            save_clips=saveClips
        )
        
        # Schedule cleanup of uploaded video after processing (production mode)
        background_tasks.add_task(
            video_processing_service.cleanup_uploaded_video,
            video_path
        )
        
        logger.info(
            f"✅ Successfully processed video for trip {tripId} - "
            f"Activities: {result.get('activitiesCount', 0)}, "
            f"Time: {result.get('processingTime', 0):.2f}s"
        )
        
        return VideoProcessingResponse(**result)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"❌ Video processing failed for trip {tripId}: {e}", exc_info=True)
        
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


@router.get(
    "/jobs/{run_id}/media/{filename:path}",
    summary="Serve generated media (clips/images) for a run",
    description="Serve video clips and images generated for a specific run ID, "
                "matching the media URL pattern used in the first project."
)
async def get_run_media(run_id: str, filename: str, request: Request) -> Response:
    """
    Serve media files (clips/images) for a given run.

    The external API will call URLs of the form:
        {host_url}/api/jobs/{run_id}/media/{filename}

    We map that to:
        {settings.output_dir}/{run_id}/clips/{filename}
    """
    # Resolve run directory and clips root
    run_dir = os.path.join(settings.output_dir, run_id)
    clips_root = os.path.join(run_dir, "clips")

    if not os.path.isdir(clips_root):
        raise HTTPException(status_code=404, detail="run_or_clips_not_found")

    # Build absolute file path and protect against path traversal
    file_path = os.path.abspath(os.path.join(clips_root, filename))
    try:
        if os.path.commonpath([clips_root, file_path]) != clips_root:
            raise HTTPException(status_code=404, detail="file_missing")
    except Exception:
        raise HTTPException(status_code=404, detail="file_missing")

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="file_missing")

    # Detect MIME type
    import mimetypes as _m
    mime, _ = _m.guess_type(file_path)
    if file_path.lower().endswith(".mp4"):
        mime = "video/mp4"

    # Support HTTP Range for video streaming
    range_header = request.headers.get("range") or request.headers.get("Range")
    file_size = os.path.getsize(file_path)
    if mime == "video/mp4" and range_header:
        try:
            start_str, end_str = range_header.replace("bytes=", "").split("-")
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
            start = max(0, start)
            end = min(file_size - 1, end)
            length = end - start + 1
            with open(file_path, "rb") as f:
                f.seek(start)
                data = f.read(length)
            resp = Response(content=data, status_code=206, media_type=mime)
            resp.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            resp.headers["Accept-Ranges"] = "bytes"
            resp.headers["Content-Length"] = str(length)
            return resp
        except Exception:
            # Fallback to full-file response below
            pass

    # Full-file response
    with open(file_path, "rb") as f:
        data = f.read()
    resp = Response(content=data, status_code=200, media_type=mime or "application/octet-stream")
    if mime == "video/mp4":
        resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Content-Length"] = str(file_size)
    return resp


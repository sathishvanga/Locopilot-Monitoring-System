"""
V2 Video Controller - Enhanced API endpoints with streaming and chunked uploads

Provides memory-efficient video upload capabilities:
1. Streaming uploads - Direct-to-disk for 1GB files
2. Chunked uploads - Resumable 3-step protocol for unreliable networks
"""

from typing import Optional
import os
import math
import time
import asyncio
import aiofiles
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse

from ..models.video_models import VideoProcessingResponse
from ..models.chunked_upload_models import (
    InitiateUploadResponse,
    UploadChunkResponse,
    UploadStatusResponse,
    CancelUploadResponse
)
from ..services.video_processing_service import VideoProcessingService
from ..services.s3_upload_service import get_s3_upload_service
from ..services.external_api_service import get_external_api_service
from ..services.chunked_upload_service import get_chunked_upload_service
from ..utils.logger import get_logger
from ..utils.config import get_settings
from ..utils.disk_utils import check_disk_space_available

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api", tags=["video-v2"])

# Initialize services
video_processing_service = VideoProcessingService()
s3_upload_service = get_s3_upload_service()
chunked_upload_service = get_chunked_upload_service()

# Thread pool executor for running blocking video processing
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="video_processing")


@router.post(
    "/v2/jobs/streaming",
    response_model=VideoProcessingResponse,
    responses={
        200: {"description": "Video processed successfully"},
        400: {"description": "Invalid request or file"},
        413: {"description": "File exceeds 1GB limit"},
        507: {"description": "Insufficient disk space"},
        500: {"description": "Internal server error"}
    },
    summary="Process uploaded video with streaming (v2)",
    description="""
    V2 streaming endpoint: Memory-efficient video upload up to 1GB.

    Streams video directly to disk without loading entire file into memory.
    Supports videos up to 1GB in size.

    The endpoint accepts:
    - video: Video file (multipart/form-data, max 1GB)
    - tripId: Unique trip identifier (required)
    - lpCrewName, lpCrewId: LP crew info (optional)
    - alpCrewName, alpCrewId: ALP crew info (optional)
    - useMockDetection: Use mock detection (optional, default: false)
    - useMultiprocessing: Enable parallel processing (optional)
    - saveClips: Save annotated frames (optional, default: false)

    Returns:
    - Processing results with detected activities
    - Path to activities.json file
    - Summary statistics
    """
)
async def process_video_streaming(
    background_tasks: BackgroundTasks,
    request: Request,
    video: UploadFile = File(..., description="Video file to process"),
    tripId: str = Form(..., description="Unique trip identifier"),
    lpCrewName: Optional[str] = Form(default=None),
    lpCrewId: Optional[str] = Form(default=None),
    alpCrewName: Optional[str] = Form(default=None),
    alpCrewId: Optional[str] = Form(default=None),
    useMockDetection: Optional[bool] = Form(default=False),
    useMultiprocessing: Optional[bool] = Form(default=None),
    saveClips: Optional[bool] = Form(default=False)
):
    """Process uploaded video with streaming to disk (memory-efficient)."""
    video_path = None

    try:
        # Log request details for debugging
        logger.info(f"📥 [V2 Streaming] Received video processing request for trip: {tripId}")
        logger.info(f"📥 [V2 Streaming] Content-Type: {request.headers.get('content-type', 'N/A')}")
        logger.info(f"📥 [V2 Streaming] Content-Length: {request.headers.get('content-length', 'N/A')}")
        logger.info(f"📥 [V2 Streaming] Origin: {request.headers.get('origin', 'N/A')}")

        # Validate tripId
        if not tripId or not tripId.strip():
            raise HTTPException(status_code=400, detail="tripId is required and cannot be empty")

        # Validate filename and extension
        filename = video.filename
        if not filename:
            # Swagger UI sometimes sends files without filename - try to detect from content-type
            content_type = video.content_type or ""
            logger.warning(f"⚠️ No filename provided, content-type: {content_type}")
            
            # Try to infer extension from content-type
            content_type_map = {
                "video/mp4": ".mp4",
                "video/x-msvideo": ".avi",
                "video/quicktime": ".mov",
                "video/x-matroska": ".mkv",
            }
            
            inferred_ext = content_type_map.get(content_type.lower())
            if inferred_ext:
                filename = f"upload{inferred_ext}"
                logger.info(f"✅ Inferred filename: {filename} from content-type: {content_type}")
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Filename is required. Content-Type: {content_type}. Please ensure the file has a proper filename."
                )

        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in settings.allowed_video_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file extension '{file_ext}'. Allowed: {', '.join(settings.allowed_video_extensions)}. Filename: {filename}"
            )

        # Pre-check disk space using Content-Length header if available
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size_estimate = int(content_length)

                # Check if file exceeds 1GB
                if size_estimate > settings.max_upload_size:
                    max_size_gb = settings.max_upload_size / (1024 ** 3)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum size of {max_size_gb:.1f} GB"
                    )

                # Check disk space availability
                has_space, err_msg = check_disk_space_available(
                    settings.upload_dir,
                    size_estimate,
                    settings.disk_reserve_gb
                )
                if not has_space:
                    raise HTTPException(status_code=507, detail=err_msg)

            except ValueError:
                logger.warning("⚠️ Invalid Content-Length header, proceeding without pre-check")

        # Create safe filename
        safe_filename = f"{tripId}_{int(time.time())}{file_ext}"
        video_path = os.path.join(settings.upload_dir, safe_filename)

        # Stream video to disk
        total_bytes = 0
        try:
            async with aiofiles.open(video_path, "wb") as out_file:
                while True:
                    chunk = await video.read(settings.stream_chunk_size)
                    if not chunk:
                        break

                    total_bytes += len(chunk)

                    # Enforce size limit on-the-fly
                    if total_bytes > settings.max_upload_size:
                        await out_file.close()
                        if os.path.exists(video_path):
                            os.remove(video_path)
                        logger.warning(f"❌ Upload exceeded 1GB limit: {video_path}")
                        raise HTTPException(
                            status_code=413,
                            detail="File exceeds maximum size of 1GB"
                        )

                    await out_file.write(chunk)

            logger.info(f"✅ Video streamed to disk: {video_path} ({total_bytes} bytes)")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Failed to stream video to disk: {e}", exc_info=True)
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
            raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

        # Validate empty file
        if total_bytes == 0:
            if os.path.exists(video_path):
                os.remove(video_path)
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Build crew members dictionary
        crew_members = {}
        if lpCrewName and lpCrewId:
            if lpCrewName.strip() and lpCrewId.strip():
                crew_members['LP'] = {
                    'name': lpCrewName.strip(),
                    'id': lpCrewId.strip(),
                    'role': 'LP'
                }
        if alpCrewName and alpCrewId:
            if alpCrewName.strip() and alpCrewId.strip():
                crew_members['ALP'] = {
                    'name': alpCrewName.strip(),
                    'id': alpCrewId.strip(),
                    'role': 'ALP'
                }

        # Process video in thread pool to avoid blocking event loop
        # This prevents EPIPE errors when client times out during long processing
        logger.info(f"🎬 Starting video processing for trip: {tripId}")
        
        try:
            # Run blocking video processing in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _executor,
                lambda: video_processing_service.process_video(
                    video_path=video_path,
                    trip_id=tripId,
                    crew_members=crew_members,
                    use_mock_detection=useMockDetection,
                    use_multiprocessing=useMultiprocessing,
                    save_clips=saveClips
                )
            )
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            # Handle connection drops gracefully (client timeout, network issues)
            error_msg = str(e)
            if "EPIPE" in error_msg or isinstance(e, BrokenPipeError):
                logger.warning(
                    f"⚠️ Connection closed by client during processing for trip {tripId}: {error_msg}. "
                    f"Video processing may have completed in background."
                )
                # Still schedule cleanup
                background_tasks.add_task(
                    video_processing_service.cleanup_uploaded_video,
                    video_path
                )
                # Return error response if connection is still open
                raise HTTPException(
                    status_code=499,  # Client Closed Request
                    detail="Connection closed by client. Video processing may still be in progress."
                )
            else:
                raise

        # Schedule cleanup of uploaded video
        background_tasks.add_task(
            video_processing_service.cleanup_uploaded_video,
            video_path
        )

        logger.info(f"✅ Video processing completed for trip: {tripId}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Video processing failed for trip {tripId}: {e}", exc_info=True)
        if video_path and os.path.exists(video_path):
            video_processing_service.cleanup_uploaded_video(video_path)
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")


@router.post(
    "/v2/upload/initiate",
    response_model=InitiateUploadResponse,
    summary="Initiate chunked upload session",
    description="""
    Step 1 of chunked upload: Initialize upload session.

    Returns upload_id and recommended chunk size for uploading parts.
    """
)
async def initiate_chunked_upload(
    filename: str = Form(..., description="Original filename"),
    total_size: int = Form(..., description="Total file size in bytes", gt=0),
    tripId: str = Form(..., description="Trip identifier")
):
    """Initiate a new chunked upload session."""
    try:
        logger.info(f"📥 Initiating chunked upload for {filename} ({total_size} bytes), trip: {tripId}")

        # Validate tripId
        if not tripId or not tripId.strip():
            raise HTTPException(status_code=400, detail="tripId is required")

        # Validate filename
        if not filename or not filename.strip():
            raise HTTPException(status_code=400, detail="filename is required")

        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in settings.allowed_video_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file extension. Allowed: {', '.join(settings.allowed_video_extensions)}"
            )

        # Validate size limit
        if total_size > settings.max_upload_size:
            max_size_gb = settings.max_upload_size / (1024 ** 3)
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size of {max_size_gb:.1f} GB"
            )

        # Check disk space
        has_space, err_msg = check_disk_space_available(
            settings.chunk_upload_dir,
            total_size,
            settings.disk_reserve_gb
        )
        if not has_space:
            raise HTTPException(status_code=507, detail=err_msg)

        # Create upload session
        upload_id, chunk_size, total_chunks = await chunked_upload_service.initiate_upload(
            filename=filename,
            total_size=total_size,
            trip_id=tripId
        )

        # Calculate expiration
        expires_at = datetime.utcnow() + timedelta(hours=settings.chunk_session_ttl_hours)

        return InitiateUploadResponse(
            upload_id=upload_id,
            chunk_size_recommendation=chunk_size,
            total_chunks=total_chunks,
            expires_at=expires_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to initiate chunked upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/v2/upload/chunk",
    response_model=UploadChunkResponse,
    summary="Upload a chunk",
    description="""
    Step 2 of chunked upload: Upload individual chunk.

    Can be called multiple times for different part numbers.
    Chunks can be uploaded in any order and retried if failed.
    """
)
async def upload_chunk(
    upload_id: str = Form(..., description="Upload session ID"),
    part_number: int = Form(..., description="1-based part number", ge=1),
    chunk: UploadFile = File(..., description="Chunk data")
):
    """Upload a single chunk for an upload session."""
    try:
        logger.info(f"📦 Receiving chunk {part_number} for upload {upload_id}")

        # Save chunk
        success = await chunked_upload_service.save_chunk(
            upload_id=upload_id,
            part_number=part_number,
            chunk=chunk
        )

        if not success:
            raise HTTPException(status_code=404, detail="Upload session not found")

        return UploadChunkResponse(
            status="ok",
            part=part_number,
            message=f"Chunk {part_number} uploaded successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to upload chunk {part_number} for {upload_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/v2/upload/complete",
    response_model=VideoProcessingResponse,
    summary="Complete chunked upload and process video",
    description="""
    Step 3 of chunked upload: Assemble chunks and process video.

    Assembles all uploaded chunks, validates size, processes video.
    """
)
async def complete_chunked_upload(
    background_tasks: BackgroundTasks,
    upload_id: str = Form(..., description="Upload session ID"),
    lpCrewName: Optional[str] = Form(default=None),
    lpCrewId: Optional[str] = Form(default=None),
    alpCrewName: Optional[str] = Form(default=None),
    alpCrewId: Optional[str] = Form(default=None),
    useMockDetection: Optional[bool] = Form(default=False),
    useMultiprocessing: Optional[bool] = Form(default=None),
    saveClips: Optional[bool] = Form(default=False)
):
    """Complete chunked upload by assembling chunks and processing video."""
    video_path = None

    try:
        logger.info(f"🔧 Completing chunked upload: {upload_id}")

        # Get metadata
        metadata = chunked_upload_service.get_metadata(upload_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Upload session not found")

        trip_id = metadata["trip_id"]

        # Assemble chunks into final file
        video_path, total_size = await chunked_upload_service.complete_upload(upload_id)

        logger.info(f"✅ Chunks assembled successfully: {video_path} ({total_size} bytes)")

        # Build crew members dictionary
        crew_members = {}
        if lpCrewName and lpCrewId:
            if lpCrewName.strip() and lpCrewId.strip():
                crew_members['LP'] = {
                    'name': lpCrewName.strip(),
                    'id': lpCrewId.strip(),
                    'role': 'LP'
                }
        if alpCrewName and alpCrewId:
            if alpCrewName.strip() and alpCrewId.strip():
                crew_members['ALP'] = {
                    'name': alpCrewName.strip(),
                    'id': alpCrewId.strip(),
                    'role': 'ALP'
                }

        # Process video in thread pool to avoid blocking event loop
        # This prevents EPIPE errors when client times out during long processing
        logger.info(f"🎬 Starting video processing for trip: {trip_id}")
        
        try:
            # Run blocking video processing in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _executor,
                lambda: video_processing_service.process_video(
                    video_path=video_path,
                    trip_id=trip_id,
                    crew_members=crew_members,
                    use_mock_detection=useMockDetection,
                    use_multiprocessing=useMultiprocessing,
                    save_clips=saveClips
                )
            )
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            # Handle connection drops gracefully (client timeout, network issues)
            error_msg = str(e)
            if "EPIPE" in error_msg or isinstance(e, BrokenPipeError):
                logger.warning(
                    f"⚠️ Connection closed by client during processing for trip {trip_id}: {error_msg}. "
                    f"Video processing may have completed in background."
                )
                # Still schedule cleanup
                background_tasks.add_task(
                    video_processing_service.cleanup_uploaded_video,
                    video_path
                )
                background_tasks.add_task(
                    chunked_upload_service.cleanup_upload_session,
                    upload_id
                )
                # Return error response if connection is still open
                raise HTTPException(
                    status_code=499,  # Client Closed Request
                    detail="Connection closed by client. Video processing may still be in progress."
                )
            else:
                raise

        # Schedule cleanup of uploaded video and upload session
        background_tasks.add_task(
            video_processing_service.cleanup_uploaded_video,
            video_path
        )
        background_tasks.add_task(
            chunked_upload_service.cleanup_upload_session,
            upload_id
        )

        logger.info(f"✅ Chunked upload processing completed for trip: {trip_id}")
        return result

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Upload session not found")
    except ValueError as e:
        logger.error(f"❌ Validation error completing upload: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Failed to complete chunked upload: {e}", exc_info=True)
        if video_path and os.path.exists(video_path):
            video_processing_service.cleanup_uploaded_video(video_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/v2/upload/{upload_id}/status",
    response_model=UploadStatusResponse,
    summary="Get upload session status",
    description="""
    Get status of a chunked upload session.

    Returns which chunks have been uploaded and session info.
    """
)
async def get_upload_status(upload_id: str):
    """Get status of an upload session."""
    try:
        status = chunked_upload_service.get_session_status(upload_id)

        if not status:
            raise HTTPException(status_code=404, detail="Upload session not found")

        return UploadStatusResponse(**status)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get upload status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/v2/upload/{upload_id}",
    response_model=CancelUploadResponse,
    summary="Cancel upload session",
    description="""
    Cancel and cleanup an upload session.

    Removes all uploaded chunks and session metadata.
    """
)
async def cancel_upload(upload_id: str):
    """Cancel and cleanup an upload session."""
    try:
        # Check if session exists
        metadata = chunked_upload_service.get_metadata(upload_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Upload session not found")

        # Cleanup session
        chunked_upload_service.cleanup_upload_session(upload_id)

        logger.info(f"🗑️ Upload session cancelled: {upload_id}")

        return CancelUploadResponse(
            status="cancelled",
            upload_id=upload_id,
            message=f"Upload session {upload_id} has been cancelled and cleaned up"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to cancel upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Optional: Cleanup endpoint for manual triggering (useful for admin)
@router.post(
    "/v2/upload/cleanup",
    summary="Cleanup expired upload sessions",
    description="""
    Manually trigger cleanup of expired upload sessions.

    Removes sessions older than configured TTL (default: 24 hours).
    Admin/maintenance endpoint.
    """
)
async def cleanup_expired_sessions():
    """Manually trigger cleanup of expired sessions."""
    try:
        cleaned_count = chunked_upload_service.cleanup_expired_sessions()

        return {
            "status": "success",
            "message": f"Cleaned up {cleaned_count} expired sessions",
            "sessions_removed": cleaned_count
        }

    except Exception as e:
        logger.error(f"❌ Failed to cleanup sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

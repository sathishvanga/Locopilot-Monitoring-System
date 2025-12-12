"""
Video controller - API endpoints for video upload and processing

Handles HTTP requests and responses for video processing operations.
"""

from typing import Optional
import os

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Response
from fastapi.responses import JSONResponse

from ..models.video_models import (
    VideoProcessingResponse,
    VideoProcessingError,
    ChunkedUploadInitiateResponse,
    ChunkedUploadChunkResponse
)
from ..services.video_processing_service import VideoProcessingService
from ..services.s3_upload_service import get_s3_upload_service
from ..services.external_api_service import get_external_api_service
from ..services.chunked_upload_service import get_chunked_upload_service
from ..services.minio_service import get_minio_service
from ..utils.logger import get_logger
from ..utils.config import get_settings


logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api", tags=["video"])

# Initialize services (singleton pattern)
video_processing_service = VideoProcessingService()
s3_upload_service = get_s3_upload_service()
chunked_upload_service = get_chunked_upload_service()
# Note: minio_service is initialized lazily via get_minio_service() when needed


@router.post(
    "/video/analyze",
    response_model=VideoProcessingResponse,
    responses={
        200: {"description": "Video processed successfully"},
        400: {"description": "Invalid request or file"},
        500: {"description": "Internal server error"}
    },
    summary="Process uploaded video",
    description="""
    Upload and process a video file for activity detection.

    The endpoint accepts EITHER a video file OR a video URL from MinIO:
    - video: Video file (multipart/form-data) - optional if videoUrl provided
    - videoUrl: MinIO URL to download video from (e.g., https://mind.snikbtel.uk:9000/cvss/video.mp4)
    - tripId: Unique trip identifier (required)
    - lpCrewName: LP crew member name (optional)
    - lpCrewId: LP crew member ID (optional)
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
    video: Optional[UploadFile] = File(default=None, description="Video file to process (optional if videoUrl provided)"),
    videoUrl: Optional[str] = Form(default=None, description="MinIO URL to download video from (e.g., https://mind.snikbtel.uk:9000/cvss/video.mp4)"),
    tripId: str = Form(..., description="Unique trip identifier"),
    lpCrewName: Optional[str] = Form(default=None, description="Loco Pilot crew member name"),
    lpCrewId: Optional[str] = Form(default=None, description="Loco Pilot crew member ID"),
    alpCrewName: Optional[str] = Form(default=None, description="Assistant Loco Pilot crew member name"),
    alpCrewId: Optional[str] = Form(default=None, description="Assistant Loco Pilot crew member ID"),
    useMockDetection: Optional[bool] = Form(default=False, description="Use mock detection for testing"),
    useMultiprocessing: Optional[bool] = Form(default=None, description="Enable multiprocessing (default: from config)"),
    saveClips: Optional[bool] = Form(default=False, description="Save annotated frames for debugging (default: false). Clips/images always saved.")
):
    """
    Process uploaded video and detect activities

    This endpoint handles video upload or MinIO download, validation, processing,
    and activity detection. The video is saved temporarily, processed, and then cleaned up.
    """
    video_path = None
    video_filename = None

    try:
        logger.info(f"📥 Received video processing request for trip: {tripId}")

        # Validate tripId
        if not tripId or not tripId.strip():
            logger.warning(f"⚠️ Invalid request: tripId is empty")
            raise HTTPException(
                status_code=400,
                detail="tripId is required and cannot be empty"
            )

        # Validate that either video OR videoUrl is provided (not both, not neither)
        has_video = video is not None and video.filename
        # Validate URL: must be non-empty and look like a valid URL (starts with http/https)
        has_url = (
            videoUrl is not None
            and videoUrl.strip()
            and videoUrl.strip().lower().startswith(('http://', 'https://'))
        )

        if not has_video and not has_url:
            logger.warning(f"⚠️ Invalid request: Neither video file nor videoUrl provided")
            raise HTTPException(
                status_code=400,
                detail="Either 'video' file or 'videoUrl' must be provided"
            )

        # Build crew members dictionary
        crew_members = {}

        # Add LP crew if provided
        if lpCrewName and lpCrewId:
            if lpCrewName.strip() and lpCrewId.strip():
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

        # Get video either from upload or MinIO URL
        if has_url:
            # Download video from MinIO
            logger.info(f"📥 Downloading video from MinIO: {videoUrl}")
            try:
                minio_svc = get_minio_service()
                video_path = minio_svc.download_video(videoUrl, tripId)
                video_filename = os.path.basename(video_path)
                file_size = os.path.getsize(video_path)
                logger.info(f"📹 Downloaded video: {video_filename} ({file_size / (1024*1024):.2f} MB)")
            except Exception as e:
                logger.error(f"❌ Failed to download video from MinIO: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download video from MinIO: {str(e)}"
                )
        else:
            # Read video content from upload
            video_content = await video.read()
            file_size = len(video_content)
            video_filename = video.filename

            logger.info(f"📹 Uploaded video: {video_filename} ({file_size / (1024*1024):.2f} MB)")

            # Validate video file
            is_valid, error_message = video_processing_service.validate_video_file(
                filename=video_filename,
                file_size=file_size
            )

            if not is_valid:
                logger.warning(f"⚠️ Video validation failed: {error_message}")
                raise HTTPException(status_code=400, detail=error_message)

            # Save uploaded video
            video_path = await video_processing_service.save_uploaded_video(
                file_content=video_content,
                filename=video_filename,
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
            crew_name=lpCrewName if lpCrewName else "Unknown",  # Default if not provided
            crew_id=lpCrewId if lpCrewId else "N/A",  # Default if not provided
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


@router.post(
    "/v1/video/process-and-upload",
    summary="Process video and upload evidence clips to S3",
    description="""
    Complete workflow for desktop application:
    1. Process video locally (activity detection with YOLO)
    2. Generate evidence clips
    3. Upload evidence clips to S3 (original video is NOT uploaded)
    4. Post results to external API with evidence clip S3 URLs
    5. Return S3 URLs and processing results
    
    Note: Original video is processed locally only and not uploaded to S3.
    Only evidence clips are uploaded to S3.
    """
)
async def process_and_upload_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(..., description="Video file to process"),
    tripId: str = Form(..., description="Unique trip identifier"),
    subFolderName: str = Form(default="cvvr", description="S3 subfolder name"),
    authToken: Optional[str] = Form(default=None, description="Authentication token for S3 upload"),
    lpCrewName: Optional[str] = Form(default=None, description="Loco Pilot crew member name"),
    lpCrewId: Optional[str] = Form(default=None, description="Loco Pilot crew member ID"),
    alpCrewName: Optional[str] = Form(default=None, description="Assistant Loco Pilot crew member name"),
    alpCrewId: Optional[str] = Form(default=None, description="Assistant Loco Pilot crew member ID"),
    useMultiprocessing: Optional[bool] = Form(default=None, description="Enable multiprocessing (default: from config)"),
    useMockDetection: Optional[bool] = Form(default=False, description="Use mock detection for testing"),
    saveClips: Optional[bool] = Form(default=False, description="Save annotated frames for debugging")
):
    """
    Process video and upload everything to S3
    
    This is the preferred endpoint for the desktop application as it handles
    the complete workflow in one request.
    """
    video_path = None
    
    try:
        logger.info(f"📥 Process and upload request for trip: {tripId}")
        
        # Validate tripId
        if not tripId or not tripId.strip():
            raise HTTPException(
                status_code=400,
                detail="tripId is required and cannot be empty"
            )
        
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
        
        # Validate video file
        filename = video_file.filename
        if not filename:
            raise HTTPException(
                status_code=400,
                detail="Invalid video file - filename is empty"
            )
        
        # Check file extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in settings.allowed_video_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file extension {file_ext}. Allowed: {', '.join(settings.allowed_video_extensions)}"
            )
        
        # Save uploaded video
        video_path = await video_processing_service.save_uploaded_video(
            video_file.file.read(),
            filename,
            tripId
        )
        
        logger.info(f"✅ Video saved: {video_path}")
        
        # Determine multiprocessing setting
        use_mp = useMultiprocessing if useMultiprocessing is not None else settings.enable_multiprocessing
        
        # Process video (activity detection)
        logger.info(
            f"🎬 Starting video processing for trip: {tripId} - "
            f"Multiprocessing: {use_mp}, Mock: {useMockDetection}, SaveClips: {saveClips}"
        )
        
        result = video_processing_service.process_video(
            video_path=video_path,
            trip_id=tripId,
            crew_members=crew_members,
            crew_name=list(crew_members.values())[0]['name'] if crew_members else "Unknown",
            crew_id=list(crew_members.values())[0]['id'] if crew_members else "N/A",
            crew_role=1,  # LP role
            use_mock_detection=useMockDetection,
            use_multiprocessing=use_mp,  # ✅ Now using multiprocessing!
            save_clips=saveClips,
            skip_external_api=True  # Skip here - will call after S3 uploads with correct S3 URLs
        )
        
        logger.info(
            f"✅ Processing complete - "
            f"Run: {result.get('runId', result.get('run_id', 'N/A'))}, "
            f"Activities: {result.get('activitiesCount', result.get('activities_count', 0))}, "
            f"Clips: {len(result.get('clipFiles', result.get('clip_files', [])))}"
        )
        
        # Step 3: Upload evidence files (clips + images) to S3
        # Note: Original video is NOT uploaded - only evidence clips are uploaded
        video_s3_url = None  # No original video URL since we don't upload it
        clip_files = result.get('clip_files', [])
        evidence_urls = []
        upload_errors = []
        s3_file_mapping = {}  # Map local paths to S3 URLs
        
        if clip_files:
            logger.info(f"☁️ Uploading {len(clip_files)} evidence files (clips + images) to S3")
            
            # Collect all files to upload (clips + their corresponding images)
            all_files_to_upload = []
            for clip_file in clip_files:
                all_files_to_upload.append(clip_file)
                
                # Find corresponding image file
                image_file = clip_file.replace('_clip.mp4', '_activity.jpg')
                if os.path.exists(image_file):
                    all_files_to_upload.append(image_file)
            
            logger.info(f"☁️ Total files to upload: {len(all_files_to_upload)} (clips + images)")
            
            # Upload all files
            clips_success, file_urls, clip_errors = s3_upload_service.upload_multiple_files(
                file_paths=all_files_to_upload,
                subfolder=subFolderName,
                auth_token=authToken
            )
            
            # Create mapping from local path to S3 URL
            for local_path, s3_url in zip(all_files_to_upload, file_urls):
                s3_file_mapping[local_path] = s3_url
            
            evidence_urls = [url for url in file_urls if '_clip.mp4' in url]
            upload_errors = clip_errors
            
            if not clips_success:
                logger.warning(f"Some files failed to upload: {clip_errors}")
            
            logger.info(f"✅ Uploaded {len(file_urls)}/{len(all_files_to_upload)} files to S3")
        
        # Step 3 (continued): Update activities.json with S3 URLs
        logger.info("📝 Updating activities.json with S3 URLs")
        
        # Extract values from result dictionary
        run_dir = result.get('run_dir', result.get('runDir', ''))
        run_id = result.get('run_id', result.get('runId', ''))
        activities_count = result.get('activities_count', result.get('activitiesCount', 0))
        
        # Initialize activities list - try to get from result first, then from file
        activities = result.get('activities', [])
        
        if run_dir and os.path.exists(run_dir):
            activities_json_path = os.path.join(run_dir, 'activities.json')
            
            if os.path.exists(activities_json_path):
                import json
                
                try:
                    # Read existing activities
                    with open(activities_json_path, 'r', encoding='utf-8') as f:
                        activities = json.load(f)
                    
                    logger.info(f"📖 Loaded {len(activities)} activities from {activities_json_path}")
                    
                    # Update each activity with S3 URLs
                    updated_activities = []
                    for activity in activities:
                        # Update activityClip with S3 URL
                        if 'activityClip' in activity and activity['activityClip']:
                            local_clip_path = activity['activityClip']
                            if local_clip_path in s3_file_mapping:
                                activity['activityClip'] = s3_file_mapping[local_clip_path]
                                logger.debug(f"Updated clip URL: {local_clip_path} -> {s3_file_mapping[local_clip_path]}")
                        
                        # Update activityImage with S3 URL
                        if 'activityImage' in activity and activity['activityImage']:
                            local_image_path = activity['activityImage']
                            if local_image_path in s3_file_mapping:
                                activity['activityImage'] = s3_file_mapping[local_image_path]
                                logger.debug(f"Updated image URL: {local_image_path} -> {s3_file_mapping[local_image_path]}")
                        
                        updated_activities.append(activity)
                    
                    # Save updated activities.json with S3 URLs
                    with open(activities_json_path, 'w', encoding='utf-8') as f:
                        json.dump(updated_activities, f, indent=2, ensure_ascii=False)
                    
                    # Update activities for response
                    activities = updated_activities
                    
                    logger.info(f"✅ Updated {len(updated_activities)} activities with S3 URLs")
                except Exception as e:
                    logger.error(f"❌ Failed to update activities.json: {e}", exc_info=True)
            else:
                logger.warning(f"⚠️ Activities JSON file not found: {activities_json_path}")
        else:
            logger.warning(f"⚠️ Run directory not found: {run_dir}")
        
        # Step 4: Post results to external API with evidence clip S3 URLs
        # Note: We don't upload original video, so video_s3_url is None
        # The external API will use evidence clip URLs from activities
        external_api_result = None
        logger.info(f"🔍 Checking conditions for external API call: activities={len(activities) if activities else 0}")
        
        if activities:
            try:
                logger.info(f"🌐 Posting results to external API with S3 URLs for trip: {tripId}")
                external_api_service = get_external_api_service()
                
                external_api_result = external_api_service.post_cvvr_results(
                    trip_id=tripId,
                    events=activities,  # Use updated activities with S3 URLs (evidence clips)
                    job_id=run_id,
                    video_s3_url=None  # No original video URL - we don't upload original video
                )
                
                if external_api_result.get("success"):
                    logger.info(
                        f"✅ [external_api] Posted {external_api_result.get('violations_count', 0)} "
                        f"violations with S3 URLs to external API for trip {tripId}"
                    )
                else:
                    logger.warning(
                        f"⚠️ [external_api] Failed to post to external API: {external_api_result.get('message')}"
                    )
            except Exception as e:
                logger.error(f"❌ [external_api] Exception while posting to external API: {e}", exc_info=True)
                external_api_result = {
                    "success": False,
                    "message": f"Exception: {str(e)}",
                    "posted": False
                }
        
        # Prepare response
        response_data = {
            "status": "success",
            "message": "Video processed and uploaded successfully",
            "data": {
                "tripId": tripId,
                "run_id": run_id,
                "run_dir": run_dir,
                "activities_count": activities_count,
                "processing_time_seconds": result.get('processingTime', result.get('processing_time', 0)),
                "video_url": None,  # Original video is not uploaded to S3
                "evidence_clips": evidence_urls,
                "clips_uploaded": len(evidence_urls),
                "total_clips": len(clip_files),
                "upload_errors": upload_errors if upload_errors else None,
                "activities": activities,  # ← Include updated activities with S3 URLs
                "external_api_result": external_api_result  # Include external API posting result
            }
        }
        
        logger.info(f"✅ Complete workflow finished for trip: {tripId}")
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
        
    except Exception as e:
        # Include more context in error message
        error_detail = f"Processing failed for trip {tripId}"
        if video_path:
            error_detail += f" (video: {os.path.basename(video_path)})"
        error_detail += f": {str(e)}"
        
        logger.error(f"Process and upload failed: {error_detail}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=error_detail
        )
    
    finally:
        # Optional: Clean up uploaded video file
        # Uncomment if you want to delete the original after upload
        # if video_path and os.path.exists(video_path):
        #     try:
        #         os.remove(video_path)
        #         logger.info(f"Cleaned up uploaded video: {video_path}")
        #     except Exception as e:
        #         logger.warning(f"Failed to clean up video: {e}")
        pass


# ============================================================================
# CHUNKED UPLOAD ENDPOINTS
# ============================================================================


@router.post(
    "/chunked-upload/initiate",
    response_model=ChunkedUploadInitiateResponse,
    responses={
        200: {"description": "Upload session created successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error"}
    },
    summary="Initiate chunked video upload",
    description="""
    Initiate a chunked upload session for large video files.

    The client should:
    1. Call this endpoint with video metadata
    2. Split the video into 8 MB chunks
    3. Upload each chunk via /chunked-upload/chunk
    4. Finalize the upload via /chunked-upload/finalize

    Returns an uploadId that must be used for subsequent chunk uploads.
    The session expires after 1 hour if not completed.
    """
)
async def initiate_chunked_upload(
    tripId: str = Form(..., description="Unique trip identifier"),
    filename: str = Form(..., description="Original video filename"),
    totalSize: int = Form(..., description="Total video file size in bytes"),
    lpCrewName: Optional[str] = Form(default=None, description="Loco Pilot crew member name"),
    lpCrewId: Optional[str] = Form(default=None, description="Loco Pilot crew member ID"),
    alpCrewName: Optional[str] = Form(default=None, description="Assistant Loco Pilot crew member name"),
    alpCrewId: Optional[str] = Form(default=None, description="Assistant Loco Pilot crew member ID")
):
    """
    Initiate a new chunked upload session

    Creates an upload session and returns a unique uploadId for chunk uploads.
    """
    try:
        logger.info(f"📥 Initiating chunked upload - Trip: {tripId}, File: {filename}, Size: {totalSize} bytes")

        # Validate tripId
        if not tripId or not tripId.strip():
            raise HTTPException(
                status_code=400,
                detail="tripId is required and cannot be empty"
            )

        # Validate filename and extension
        if not filename or not filename.strip():
            raise HTTPException(
                status_code=400,
                detail="filename is required and cannot be empty"
            )

        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in settings.allowed_video_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file extension {file_ext}. Allowed: {', '.join(settings.allowed_video_extensions)}"
            )

        # Validate total size
        if totalSize <= 0:
            raise HTTPException(
                status_code=400,
                detail="totalSize must be greater than 0"
            )

        if totalSize > settings.max_upload_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size {totalSize} bytes exceeds maximum {settings.max_upload_size} bytes ({settings.max_upload_size // (1024*1024)} MB)"
            )

        # Build crew members metadata
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

        # Create upload session
        metadata = {
            'crew_members': crew_members,
            'lpCrewName': lpCrewName,
            'lpCrewId': lpCrewId,
            'alpCrewName': alpCrewName,
            'alpCrewId': alpCrewId
        }

        session = chunked_upload_service.initiate_upload(
            trip_id=tripId,
            filename=filename,
            total_size=totalSize,
            metadata=metadata
        )

        logger.info(
            f"✅ Upload session created - ID: {session.upload_id}, "
            f"Chunks: {session.total_chunks}, Expires: {session.expires_at.isoformat()}"
        )

        return ChunkedUploadInitiateResponse(
            status="initiated",
            uploadId=session.upload_id,
            totalChunks=session.total_chunks,
            chunkSize=session.chunk_size,
            expiresAt=session.expires_at.isoformat()
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"❌ Failed to initiate chunked upload: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate upload: {str(e)}"
        )


@router.post(
    "/chunked-upload/chunk",
    response_model=ChunkedUploadChunkResponse,
    responses={
        200: {"description": "Chunk received successfully"},
        400: {"description": "Invalid chunk or session"},
        404: {"description": "Upload session not found"},
        410: {"description": "Upload session expired"},
        500: {"description": "Internal server error"}
    },
    summary="Upload a video chunk",
    description="""
    Upload a single chunk of the video file.

    Chunks can be uploaded in any order and the same chunk can be re-uploaded
    if needed (idempotent operation).

    Each chunk should be approximately 8 MB, except the last chunk which can
    be smaller.
    """
)
async def upload_chunk(
    uploadId: str = Form(..., description="Upload session ID from initiate response"),
    chunkIndex: int = Form(..., description="Zero-based index of this chunk"),
    chunk: UploadFile = File(..., description="Chunk file data")
):
    """
    Upload a single chunk

    Saves the chunk to disk and tracks progress.
    """
    try:
        logger.debug(f"📦 Receiving chunk {chunkIndex} for upload {uploadId}")

        # Read chunk data
        chunk_data = await chunk.read()

        # Save chunk via service
        success, message = await chunked_upload_service.save_chunk(
            upload_id=uploadId,
            chunk_index=chunkIndex,
            chunk_data=chunk_data
        )

        # Get session status
        session = chunked_upload_service.get_upload_status(uploadId)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Upload session not found"
            )

        received_count = len(session.received_chunks)
        is_complete = received_count == session.total_chunks

        logger.info(
            f"✅ Chunk {chunkIndex} received - Progress: {received_count}/{session.total_chunks}"
        )

        return ChunkedUploadChunkResponse(
            status="received",
            uploadId=uploadId,
            chunkIndex=chunkIndex,
            receivedChunks=received_count,
            totalChunks=session.total_chunks,
            complete=is_complete
        )

    except ValueError as e:
        # Handle service-level validation errors
        error_msg = str(e)

        if "expired" in error_msg.lower():
            raise HTTPException(status_code=410, detail=error_msg)
        elif "not found" in error_msg.lower():
            raise HTTPException(status_code=404, detail=error_msg)
        else:
            raise HTTPException(status_code=400, detail=error_msg)

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"❌ Failed to upload chunk {chunkIndex}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload chunk: {str(e)}"
        )


@router.post(
    "/chunked-upload/finalize",
    response_model=VideoProcessingResponse,
    responses={
        200: {"description": "Video processed successfully"},
        400: {"description": "Missing chunks or invalid request"},
        404: {"description": "Upload session not found"},
        410: {"description": "Upload session expired"},
        500: {"description": "Internal server error"}
    },
    summary="Finalize chunked upload and process video",
    description="""
    Finalize the chunked upload by reassembling chunks and processing the video.

    This endpoint:
    1. Validates all chunks are present
    2. Reassembles chunks into complete video file
    3. Processes the video for activity detection
    4. Cleans up chunks (background task)
    5. Returns processing results

    Once finalized, the upload session is consumed and cannot be reused.
    """
)
async def finalize_chunked_upload(
    background_tasks: BackgroundTasks,
    uploadId: str = Form(..., description="Upload session ID"),
    useMockDetection: Optional[bool] = Form(default=False, description="Use mock detection for testing"),
    useMultiprocessing: Optional[bool] = Form(default=None, description="Enable multiprocessing (default: from config)"),
    saveClips: Optional[bool] = Form(default=False, description="Save annotated frames for debugging")
):
    """
    Finalize upload and start video processing

    Reassembles chunks and processes the video using the existing pipeline.
    """
    video_path = None

    try:
        logger.info(f"🏁 Finalizing chunked upload: {uploadId}")

        # Get session
        session = chunked_upload_service.get_upload_status(uploadId)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Upload session not found"
            )

        # Check if expired
        from datetime import datetime
        if datetime.now() > session.expires_at:
            chunked_upload_service.cleanup_upload(uploadId)
            raise HTTPException(
                status_code=410,
                detail="Upload session expired"
            )

        # Validate all chunks present
        is_valid, missing = chunked_upload_service.validate_chunks(uploadId)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot finalize: Missing chunks {missing[:10]}"
            )

        logger.info(f"🔧 Reassembling {session.total_chunks} chunks...")

        # Reassemble video
        video_path = await chunked_upload_service.reassemble_video(uploadId)

        logger.info(f"✅ Video reassembled: {video_path}")

        # Determine multiprocessing setting
        use_mp = useMultiprocessing if useMultiprocessing is not None else settings.enable_multiprocessing

        # Extract crew info from session metadata
        crew_members = session.metadata.get('crew_members', {})
        lpCrewName = session.metadata.get('lpCrewName', 'Unknown')
        lpCrewId = session.metadata.get('lpCrewId', 'N/A')

        logger.info(
            f"🎬 Starting video processing - "
            f"Multiprocessing: {use_mp}, Mock: {useMockDetection}, SaveClips: {saveClips}"
        )

        # Process video using existing service
        result = video_processing_service.process_video(
            video_path=video_path,
            trip_id=session.trip_id,
            crew_members=crew_members,
            crew_name=lpCrewName,
            crew_id=lpCrewId,
            crew_role=1,  # LP role
            use_mock_detection=useMockDetection,
            use_multiprocessing=use_mp,
            save_clips=saveClips
        )

        # Schedule cleanup of chunks (background task)
        background_tasks.add_task(
            chunked_upload_service.cleanup_upload,
            uploadId
        )

        # Schedule cleanup of reassembled video (background task)
        background_tasks.add_task(
            video_processing_service.cleanup_uploaded_video,
            video_path
        )

        logger.info(
            f"✅ Chunked upload complete - Trip: {session.trip_id}, "
            f"Activities: {result.get('activitiesCount', 0)}, "
            f"Time: {result.get('processingTime', 0):.2f}s"
        )

        return VideoProcessingResponse(**result)

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"❌ Failed to finalize chunked upload: {e}", exc_info=True)

        # Cleanup on error
        if uploadId:
            chunked_upload_service.cleanup_upload(uploadId)
        if video_path and os.path.exists(video_path):
            video_processing_service.cleanup_uploaded_video(video_path)

        raise HTTPException(
            status_code=500,
            detail=f"Failed to finalize upload: {str(e)}"
        )


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
from ..services.s3_upload_service import get_s3_upload_service
from ..utils.logger import get_logger
from ..utils.config import get_settings


logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api", tags=["video"])

# Initialize services (singleton pattern)
video_processing_service = VideoProcessingService()
s3_upload_service = get_s3_upload_service()


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
    video: UploadFile = File(..., description="Video file to process"),
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
    summary="Process video and upload to S3",
    description="""
    Complete workflow for desktop application:
    1. Process video (activity detection with YOLO)
    2. Generate evidence clips
    3. Upload original video to S3
    4. Upload all evidence clips to S3
    5. Return S3 URLs and processing results
    
    This endpoint combines processing and uploading for better efficiency.
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
            save_clips=saveClips
        )
        
        logger.info(
            f"✅ Processing complete - "
            f"Run: {result.get('runId', result.get('run_id', 'N/A'))}, "
            f"Activities: {result.get('activitiesCount', result.get('activities_count', 0))}, "
            f"Clips: {len(result.get('clipFiles', result.get('clip_files', [])))}"
        )
        
        # Step 3: Upload original video to S3
        logger.info(f"☁️ Uploading original video to S3 (subfolder: {subFolderName})")
        
        video_upload_success, video_s3_url, video_error = s3_upload_service.upload_file(
            file_path=video_path,
            subfolder=subFolderName,
            auth_token=authToken
        )
        
        if not video_upload_success:
            logger.error(f"Failed to upload video to S3: {video_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Video processing succeeded but S3 upload failed: {video_error}"
            )
        
        logger.info(f"✅ Video uploaded to S3: {video_s3_url}")
        
        # Step 4: Upload evidence files (clips + images) to S3
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
        
        # Step 5: Update activities.json with S3 URLs
        logger.info("📝 Updating activities.json with S3 URLs")
        
        activities_json_path = os.path.join(result['run_dir'], 'activities.json')
        updated_activities = []
        
        if os.path.exists(activities_json_path):
            import json
            
            # Read existing activities
            with open(activities_json_path, 'r', encoding='utf-8') as f:
                activities = json.load(f)
            
            # Update each activity with S3 URLs
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
                "video_url": video_s3_url,
                "evidence_clips": evidence_urls,
                "clips_uploaded": len(evidence_urls),
                "total_clips": len(clip_files),
                "upload_errors": upload_errors if upload_errors else None,
                "activities": activities  # ← Include updated activities with S3 URLs
            }
        }
        
        logger.info(f"✅ Complete workflow finished for trip: {tripId}")
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Process and upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
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


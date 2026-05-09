"""
Video controller - API endpoints for video upload and processing

Handles HTTP requests and responses for video processing operations.
Includes both synchronous and async job-based endpoints.
"""

import asyncio
import functools
import hmac
import re
from typing import Optional, Dict, Any
import os

from fastapi import APIRouter, Depends, UploadFile, File, Form, Header, HTTPException, BackgroundTasks, Request, Response


def sanitize_identifier(value: Optional[str], field_name: str, required: bool = True, max_length: int = 128) -> Optional[str]:
    """
    Sanitize identifier to prevent injection attacks.

    Args:
        value: The input value to sanitize
        field_name: Name of the field (for error messages)
        required: Whether the field is required
        max_length: Maximum allowed length

    Returns:
        Sanitized string or None if not required and empty

    Raises:
        HTTPException: If validation fails
    """
    if value is None or not value.strip():
        if required:
            raise HTTPException(status_code=400, detail=f"{field_name} is required and cannot be empty")
        return None

    # Strip whitespace
    sanitized = value.strip()

    # Allow alphanumeric, underscores, hyphens, and dots (for filenames)
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', sanitized):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} contains invalid characters. Only alphanumeric, underscore, hyphen, and dot are allowed."
        )

    if len(sanitized) > max_length:
        raise HTTPException(status_code=400, detail=f"{field_name} exceeds maximum length of {max_length} characters")

    return sanitized
from fastapi.responses import JSONResponse

from ..models.video_models import (
    VideoProcessingResponse,
    VideoProcessingError
)
from ..models.job_models import (
    Job,
    JobStatus,
    JobSubmitRequest,
    JobSubmitResponse,
    JobStatusResponse,
    JobResultResponse,
    QueueStatusResponse,
    JobCancelResponse
)
from ..services.video_processing_service import VideoProcessingService
from ..services.job_manager import job_manager
from ..services.s3_upload_service import get_s3_upload_service
from ..services.external_api_service import get_external_api_service
from ..services.minio_service import get_minio_service
from ..services.vlm_verification_service import get_vlm_verification_service
from ..services.gpu_resource_manager import get_gpu_resource_manager
from ..utils.logger import get_logger
from ..utils.config import get_settings
from ..utils.url_safety import validate_external_url, URLNotAllowed


logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api", tags=["video"])

# Initialize services (singleton pattern)
video_processing_service = VideoProcessingService()
s3_upload_service = get_s3_upload_service()
# Note: minio_service is initialized lazily via get_minio_service() when needed


# C-9: run_id format must be exactly ``run_YYYYMMDD_HHMMSS``. Anything else
# (``..``, absolute paths, shell metacharacters) is rejected at the handler
# boundary before any filesystem access. Gives defense-in-depth on the
# media route and closes the raw-join traversal gap on the status route.
_RUN_ID_RE = re.compile(r"^run_\d{8}_\d{6}$")


# C-9: one-shot per-process warning when MEDIA_API_KEY is unset. The
# dependency still allows the request through in that case (rollout mode);
# we just don't want to spam the log on every request.
_media_api_key_missing_warned = False


async def enforce_max_upload_size(request: Request) -> None:
    """Task 0008: reject oversize uploads BEFORE any Form/File body parsing.

    FastAPI resolves ``File(...)`` / ``Form(...)`` parameters by reading the
    multipart body — by the time the handler body runs, bytes have already
    been buffered. This dependency runs FIRST (declared on the route via
    ``dependencies=[...]``), reads only the ``Content-Length`` header, and
    short-circuits with HTTP 413 when it advertises a body larger than
    ``settings.max_upload_size``. Lying / chunked clients that omit the
    header still get caught by the post-stream ``validate_video_file`` check
    in the handler.
    """
    raw = request.headers.get("content-length")
    if not raw:
        return
    try:
        advertised_len = int(raw)
    except (TypeError, ValueError):
        return
    if advertised_len > settings.max_upload_size:
        max_size_mb = settings.max_upload_size / (1024 * 1024)
        logger.warning(
            f"[SECURITY] Upload rejected: Content-Length {advertised_len} "
            f"exceeds max_upload_size {settings.max_upload_size}"
        )
        raise HTTPException(
            status_code=413,
            detail=f"upload exceeds max size {max_size_mb:.0f} MB",
        )


async def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """FastAPI dependency that enforces the ``X-API-Key`` header for PII routes.

    Behavior:
      * If ``settings.media_api_key`` is set, the request must present a
        matching ``X-API-Key`` header (compared with ``hmac.compare_digest``
        to avoid timing leaks). Missing or mismatched keys → 401.
      * If ``settings.media_api_key`` is unset or empty, the dependency logs
        a one-shot warning per process and allows the request through. This
        is the backward-compatible rollout mode — flip to required by
        setting ``MEDIA_API_KEY`` in ``.env.production``.
    """
    global _media_api_key_missing_warned

    expected = settings.media_api_key
    if not expected:
        if not _media_api_key_missing_warned:
            logger.warning(
                "media_api_key not set — /api/status and /api/jobs/{run_id}/media "
                "are unauthenticated (rollout mode). Set MEDIA_API_KEY in the "
                "environment to enforce auth."
            )
            _media_api_key_missing_warned = True
        return

    if x_api_key is None or not hmac.compare_digest(str(x_api_key), str(expected)):
        raise HTTPException(status_code=401, detail="invalid_or_missing_api_key")


async def _run_video_pipeline(
    *,
    request: Optional[Request],
    background_tasks: Optional[BackgroundTasks],
    video: Optional[UploadFile],
    videoUrl: Optional[str],
    tripId: Optional[str],
    division: Optional[str],
    lpCrewName: Optional[str],
    lpCrewId: Optional[str],
    alpCrewName: Optional[str],
    alpCrewId: Optional[str],
    trainNumber: Optional[str],
    tripDate: Optional[str],
    videoStartTime: Optional[str],
    useMockDetection: Optional[bool],
    useMultiprocessing: Optional[bool],
    saveClips: Optional[bool],
    cameraAngle: Optional[int],
    upload_to_external_api: bool,
    subFolderName: Optional[str] = None,
    authToken: Optional[str] = None,
):
    """Shared video-processing pipeline (Task 0003).

    Owns input validation, GPU admission, executor dispatch, and response
    build for both ``/api/video/analyze`` (``upload_to_external_api=False``)
    and ``/api/v1/video/process-and-upload`` (``upload_to_external_api=True``).

    The two endpoints diverge in:
      * Input source: analyze accepts JSON-or-form + optional MinIO
        ``videoUrl``; process-and-upload is form-only with a chunked
        upload-size guard and a required file.
      * Post-processing: analyze transforms activities into the violations
        response; process-and-upload runs VLM verification, S3-uploads
        evidence clips, rewrites activities.json with S3 URLs, and POSTs to
        the external CVVR API.

    These differences are gated on ``upload_to_external_api``. The route
    handlers stay as thin wrappers so FastAPI's path/decorator/response_model
    contract is preserved.
    """
    video_path = None
    video_filename = None
    # Initialize admission-gate flags *outside* the try so the finally
    # block can rely on them existing in every code path (including
    # exceptions raised before ``try_enqueue`` runs).
    admitted = False
    slot_acquired = False
    gpu_resource_manager = get_gpu_resource_manager()

    try:
        # --- Input parsing --------------------------------------------------
        if upload_to_external_api:
            logger.info(f"[OK] Process and upload request for trip: {tripId}, train: {trainNumber}, date: {tripDate}")
        else:
            # If JSON body was sent on the analyze route, override form fields.
            if "application/json" in request.headers.get("content-type", ""):
                try:
                    body = await request.json()
                    tripId = body.get("tripId", tripId)
                    videoUrl = body.get("videoUrl", videoUrl)
                    division = body.get("division", division)
                    lpCrewName = body.get("lpCrewName", lpCrewName)
                    lpCrewId = body.get("lpCrewId", lpCrewId)
                    alpCrewName = body.get("alpCrewName", alpCrewName)
                    alpCrewId = body.get("alpCrewId", alpCrewId)
                    trainNumber = body.get("trainNumber", trainNumber)
                    tripDate = body.get("tripDate", tripDate)
                    videoStartTime = body.get("videoStartTime", videoStartTime)
                    useMockDetection = body.get("useMockDetection", useMockDetection)
                    useMultiprocessing = body.get("useMultiprocessing", useMultiprocessing)
                    saveClips = body.get("saveClips", saveClips)
                    cameraAngle = body.get("cameraAngle", cameraAngle)
                    logger.info(f"[OK] Received JSON request body for trip: {tripId}")
                except Exception as e:
                    logger.warning(f"[WARN] Failed to parse JSON body: {e}")
                    raise HTTPException(status_code=400, detail=f"Invalid JSON body: {str(e)}")

        # --- Sanitize tripId + crew identifiers (shared) --------------------
        tripId = sanitize_identifier(tripId, "tripId", required=True)
        if lpCrewId:
            lpCrewId = sanitize_identifier(lpCrewId, "lpCrewId", required=False, max_length=64)
        if alpCrewId:
            alpCrewId = sanitize_identifier(alpCrewId, "alpCrewId", required=False, max_length=64)

        # --- Video presence + extension checks ------------------------------
        has_url = False
        if not upload_to_external_api:
            logger.info(f"[OK] Received video processing request for trip: {tripId}, division: {division}, train: {trainNumber}, date: {tripDate}")

            has_video = video is not None and video.filename
            has_url = (
                videoUrl is not None
                and videoUrl.strip()
                and videoUrl.strip().lower().startswith(('http://', 'https://'))
            )

            if not has_video and not has_url:
                logger.warning(f"[WARN] Invalid request: Neither video file nor videoUrl provided")
                raise HTTPException(
                    status_code=400,
                    detail="Either 'video' file or 'videoUrl' must be provided"
                )

            # Task 0008: SSRF defense. Reject any videoUrl that targets a host
            # outside the configured allowlist or resolves to a private/loopback
            # range (cloud metadata IPs, RFC1918, localhost). Done BEFORE the
            # admission gate so we don't burn a queue slot on a hostile URL.
            if has_url:
                try:
                    validate_external_url(
                        videoUrl.strip(),
                        settings.minio_allowed_hosts,
                        allow_private_ips=settings.minio_allow_private_ips,
                    )
                except URLNotAllowed as e:
                    logger.warning(
                        f"[SECURITY] videoUrl rejected for trip {tripId}: {e}"
                    )
                    raise HTTPException(
                        status_code=400,
                        detail=f"videoUrl rejected: {e}",
                    )
        else:
            # process-and-upload: file is required, no videoUrl branch.
            filename = video.filename if video is not None else None
            if not filename:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid video file - filename is empty"
                )
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in settings.allowed_video_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file extension {file_ext}. Allowed: {', '.join(settings.allowed_video_extensions)}"
                )

        # --- Concurrency admission gate (shared) ----------------------------
        # GPU is saturated at 1 concurrent video on this hardware; additional
        # jobs must queue, not run in parallel (else they fight for VRAM/CUDA
        # and total throughput stays flat while risking OOM). Admit up to
        # ``max_concurrent_videos + job_queue_max_size`` requests — active
        # plus waiting. Reject past that point with 503 so clients retry
        # later instead of piling up unbounded work.
        admitted, queue_position = gpu_resource_manager.try_enqueue()
        if not admitted:
            tag = "(process-and-upload) " if upload_to_external_api else ""
            logger.warning(
                f"[QUEUE FULL] Rejecting trip {tripId} {tag}"
                f"system full (max {settings.max_concurrent_videos} active + "
                f"{settings.job_queue_max_size} queued)"
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "Server busy: maximum concurrent + queued jobs reached. "
                    "Retry after the current jobs finish."
                ),
            )
        if queue_position > settings.max_concurrent_videos:
            tag = " (process-and-upload)" if upload_to_external_api else ""
            logger.info(
                f"[QUEUE] Trip {tripId} admitted{tag}; waiting at position {queue_position}"
            )
        # ``slot_acquired`` flips to True inside ``acquire_gpu_slot()`` once
        # we are past the pending → active migration; the finally block
        # uses it to decide whether the pending counter still needs rollback.

        # --- Build crew_members dict (shared) -------------------------------
        crew_members: Dict[str, Dict[str, str]] = {}
        if lpCrewName and lpCrewId and lpCrewName.strip() and lpCrewId.strip():
            crew_members['LP'] = {'name': lpCrewName.strip(), 'id': lpCrewId.strip(), 'role': 'LP'}
            if not upload_to_external_api:
                logger.info(f"LP Crew: {lpCrewName} ({lpCrewId})")
        if alpCrewName and alpCrewId and alpCrewName.strip() and alpCrewId.strip():
            crew_members['ALP'] = {'name': alpCrewName.strip(), 'id': alpCrewId.strip(), 'role': 'ALP'}
            if not upload_to_external_api:
                logger.info(f"ALP Crew: {alpCrewName} ({alpCrewId})")

        # --- Acquire video bytes / save to disk -----------------------------
        if not upload_to_external_api and has_url:
            # Download video from MinIO
            logger.info(f"[OK] Downloading video from MinIO: {videoUrl}")
            try:
                minio_svc = get_minio_service()
                if not minio_svc.check_object_exists(videoUrl):
                    logger.error(f"[ERROR] Video not found in MinIO: {videoUrl}")
                    raise HTTPException(
                        status_code=404,
                        detail=f"Video not found in MinIO storage. Please verify the URL: {videoUrl}"
                    )
                video_path = minio_svc.download_video(videoUrl, tripId)
                video_filename = os.path.basename(video_path)
                file_size = os.path.getsize(video_path)
                logger.info(f"[OK] Downloaded video: {video_filename} ({file_size / (1024*1024):.2f} MB)")
            except Exception as e:
                logger.error(f"[ERROR] Failed to download video from MinIO: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download video from MinIO: {str(e)}"
                )
        elif not upload_to_external_api:
            # Read video content from upload (one-shot read)
            video_content = await video.read()
            file_size = len(video_content)
            video_filename = video.filename
            logger.info(f"[OK] Uploaded video: {video_filename} ({file_size / (1024*1024):.2f} MB)")

            is_valid, error_message = video_processing_service.validate_video_file(
                filename=video_filename, file_size=file_size,
            )
            if not is_valid:
                logger.warning(f"[WARN] Video validation failed: {error_message}")
                raise HTTPException(status_code=400, detail=error_message)

            video_path = await video_processing_service.save_uploaded_video(
                file_content=video_content, filename=video_filename, trip_id=tripId,
            )
        else:
            # process-and-upload: chunked-read with mid-stream size cap.
            #
            # Task 0008 / reviewer H3 — chunked-encoding bypass.
            # ``enforce_max_upload_size`` only inspects ``Content-Length``, so a
            # client sending ``Transfer-Encoding: chunked`` (or omitting the
            # length header entirely) skips the pre-stream cap. To close that
            # gap we must read the body in chunks and abort once we cross
            # ``max_upload_size`` rather than calling ``.read()`` unconditionally
            # (which would buffer an unbounded body before we noticed).
            max_size = settings.max_upload_size
            chunk_size = 1024 * 1024  # 1 MiB
            chunks: list[bytes] = []
            file_size = 0
            video_filename = filename
            while True:
                chunk = await video.read(chunk_size)
                if not chunk:
                    break
                file_size += len(chunk)
                if max_size and file_size > max_size:
                    max_size_mb = max_size / (1024 * 1024)
                    logger.warning(
                        f"[SECURITY] Upload rejected mid-stream for trip {tripId}: "
                        f"body exceeded max_upload_size {max_size} (chunked / "
                        f"missing Content-Length bypass)"
                    )
                    raise HTTPException(
                        status_code=413,
                        detail=f"upload exceeds max size {max_size_mb:.0f} MB",
                    )
                chunks.append(chunk)
            video_bytes = b"".join(chunks)

            # Task 0008: full size + extension validation post-stream.
            is_valid, error_message = video_processing_service.validate_video_file(
                filename=filename, file_size=file_size,
            )
            if not is_valid:
                # ``validate_video_file`` returns the same "File too large"
                # message for any size > max_upload_size; map that to 413 to
                # match the pre-stream check, while extension / empty-file
                # failures stay 400.
                status_code = 413 if file_size > settings.max_upload_size else 400
                logger.warning(f"[WARN] Video validation failed for trip {tripId}: {error_message}")
                raise HTTPException(status_code=status_code, detail=error_message)

            video_path = await video_processing_service.save_uploaded_video(
                video_bytes, filename, tripId,
            )
            logger.info(f"[OK] Video saved: {video_path}")

        # --- Determine multiprocessing setting + log ------------------------
        # Priority: request parameter > config setting > default (False)
        use_mp = useMultiprocessing if useMultiprocessing is not None else settings.enable_multiprocessing
        if upload_to_external_api:
            logger.info(
                f"[START] Starting video processing for trip: {tripId} - "
                f"Multiprocessing: {use_mp}, Mock: {useMockDetection}, SaveClips: {saveClips}"
            )
        else:
            logger.info(
                f"[CONFIG] Processing configuration - "
                f"Multiprocessing: {use_mp}, SaveClips: {saveClips}, Mock: {useMockDetection}"
            )

        # cameraAngle: only used by analyze; process-and-upload path uses
        # the default LP role inside process_video.
        camera_angle = cameraAngle if cameraAngle in (1, 2) else 1

        # --- Heavy work: GPU slot + executor dispatch (shared shape) --------
        loop = asyncio.get_running_loop()
        # Hold a GPU slot for the duration of heavy processing. Waiters block
        # here (respecting the semaphore) until a slot frees. ``_pending_count``
        # flips to ``_active_count`` inside the context manager.
        async with gpu_resource_manager.acquire_gpu_slot():
            gpu_resource_manager.mark_enqueued_started()
            slot_acquired = True

            if upload_to_external_api:
                # Controller runs its own VLM hook + posts to external API
                # after S3 uploads, so skip both inside the service.
                process_kwargs = dict(
                    crew_members=crew_members,
                    crew_name=list(crew_members.values())[0]['name'] if crew_members else "Unknown",
                    crew_id=list(crew_members.values())[0]['id'] if crew_members else "N/A",
                    skip_external_api=True,
                    skip_vlm_verification=True,
                )
            else:
                process_kwargs = dict(
                    crew_members=crew_members,
                    crew_name=lpCrewName if lpCrewName else "Unknown",
                    crew_id=lpCrewId if lpCrewId else "N/A",
                    division=division,
                    camera_angle=camera_angle,
                )

            result = await loop.run_in_executor(
                None,
                functools.partial(
                    video_processing_service.process_video,
                    video_path=video_path,
                    trip_id=tripId,
                    crew_role=1,  # LP role
                    use_mock_detection=useMockDetection,
                    use_multiprocessing=use_mp,
                    save_clips=saveClips,
                    train_number=trainNumber,
                    trip_date=tripDate,
                    video_start_time=videoStartTime,
                    **process_kwargs,
                ),
            )

        # ------------------------------------------------------------------
        # Post-processing: differs between the two endpoints.
        # ------------------------------------------------------------------
        if not upload_to_external_api:
            # /api/video/analyze: schedule cleanup, transform activities to
            # the response-format violations, return VideoProcessingResponse.
            if background_tasks is not None:
                background_tasks.add_task(
                    video_processing_service.cleanup_uploaded_video, video_path,
                )

            external_api_service = get_external_api_service()
            activities = result.get('activities', [])
            run_directory = result.get('runDirectory', '')
            run_id = os.path.basename(run_directory) if run_directory else ''
            host_url = settings.host_url

            # Ensure activityClip has full path for URL building.
            clips_dir = os.path.join(run_directory, 'clips') if run_directory else ''
            for activity in activities:
                clip_name = activity.get('activityClip', '')
                if clip_name and not os.path.isabs(clip_name) and clips_dir:
                    activity['activityClip'] = os.path.join(clips_dir, clip_name)

            violations = external_api_service._transform_events_to_violations(
                trip_id=tripId, events=activities, job_id=run_id, host_url=host_url,
            )

            result['violations'] = violations
            result.pop('activities', None)

            logger.info(
                f"[OK] Successfully processed video for trip {tripId} - "
                f"Violations: {result.get('activitiesCount', 0)}, "
                f"Time: {result.get('processingTime', 0):.2f}s"
            )

            return VideoProcessingResponse(**result)

        # /api/v1/video/process-and-upload: VLM verification, S3 upload of
        # evidence clips, activities.json rewrite, external API POST.
        logger.info(
            f"[OK] Processing complete - "
            f"Run: {result.get('runId', result.get('run_id', 'N/A'))}, "
            f"Activities: {result.get('activitiesCount', result.get('activities_count', 0))}, "
            f"Clips: {len(result.get('clipFiles', result.get('clip_files', [])))}"
        )

        # Step 2b: VLM verification (Pipeline-2 false-positive filter).
        # Runs while local clip/image paths in activities.json are still valid
        # (before S3 swap below). Attaches vlm_review to each verified activity
        # and drops FALSE_POSITIVE @ confidence>=VLM_DROP_THRESHOLD before S3
        # upload + API push. Fail-open: VLM endpoint down → activities pass
        # through unchanged.
        vlm_service = get_vlm_verification_service()
        if vlm_service.is_enabled():
            run_dir_for_vlm = (
                result.get('runDirectory') or result.get('run_dir') or result.get('runDir') or ''
            )
            activities_json_for_vlm = result.get('activitiesJsonPath') or (
                os.path.join(run_dir_for_vlm, 'activities.json') if run_dir_for_vlm else ''
            )
            if activities_json_for_vlm and os.path.exists(activities_json_for_vlm):
                try:
                    import json as _json
                    with open(activities_json_for_vlm, 'r', encoding='utf-8') as f:
                        pre_vlm_activities = _json.load(f)
                    pre_count = len(pre_vlm_activities)
                    post_vlm_activities, vlm_stats = vlm_service.verify_activities(pre_vlm_activities)
                    # Route the rewrite through ActivityRepository so it uses
                    # the canonical atomic + locked + numpy-aware writer.
                    from app.repositories.activity_repository import (
                        ActivityRepository as _ActivityRepository,
                    )
                    _ActivityRepository().save_activities(
                        post_vlm_activities, os.path.dirname(activities_json_for_vlm),
                    )
                    # Post-VLM concurrent grouping (Phase A of post-verifier
                    # merge refactor — gated by CONCURRENT_GROUPING_AFTER_VLM).
                    # When enabled, detection-side grouping is skipped (Task
                    # 0002) and grouping runs here on the post-VLM survivor
                    # set. After grouping, activityClip points at the merged
                    # minute-NNN clip rather than per-source clips, so the
                    # clip_files filter below operates on the post-grouped set.
                    if get_settings().concurrent_grouping_after_vlm:
                        from app.services.concurrent_activity_grouping_service import (
                            get_concurrent_grouping_service,
                        )
                        pre_group_count = len(post_vlm_activities)
                        post_vlm_activities = get_concurrent_grouping_service().group_concurrent_activities(
                            post_vlm_activities, run_dir_for_vlm,
                        )
                        _ActivityRepository().save_activities(
                            post_vlm_activities, os.path.dirname(activities_json_for_vlm),
                        )
                        logger.info(
                            f"[GROUP] post-VLM grouping: {pre_group_count} -> "
                            f"{len(post_vlm_activities)} activities (run after "
                            f"verifier under CONCURRENT_GROUPING_AFTER_VLM=1)"
                        )
                    if 'activities' in result:
                        result['activities'] = post_vlm_activities
                    if 'activities_count' in result:
                        result['activities_count'] = len(post_vlm_activities)
                    if 'activitiesCount' in result:
                        result['activitiesCount'] = len(post_vlm_activities)
                    # Filter clip_files to drop those tied to dropped activities,
                    # avoiding wasted S3 uploads in enforcement mode. Under
                    # CONCURRENT_GROUPING_AFTER_VLM=1, post_vlm_activities now
                    # references merged minute-NNN clips; the set-membership
                    # filter still works — it just operates on the post-grouped
                    # paths.
                    if vlm_stats['dropped'] > 0 or get_settings().concurrent_grouping_after_vlm:
                        kept_clip_paths = {
                            a.get('activityClip') for a in post_vlm_activities
                            if a.get('activityClip')
                        }
                        for clip_key in ('clip_files', 'clipFiles'):
                            if clip_key in result and isinstance(result[clip_key], list):
                                result[clip_key] = [
                                    p for p in result[clip_key] if p in kept_clip_paths
                                ]
                    logger.info(
                        f"[VLM] verified pre={pre_count} post={len(post_vlm_activities)} "
                        f"dropped={vlm_stats['dropped']} uncertain={vlm_stats['uncertain']} "
                        f"skipped_unavail={vlm_stats['skipped_unavailable']}"
                    )
                except Exception as vlm_exc:  # pragma: no cover — fail-open at top level
                    logger.error(
                        f"[VLM] verifier failed unexpectedly, passing through "
                        f"Pipeline-1 results: {vlm_exc}",
                        exc_info=True,
                    )
            else:
                logger.warning(
                    f"[VLM] enabled but activities.json missing at "
                    f"{activities_json_for_vlm!r}; skipping verifier"
                )

        # Step 3: Upload evidence files (clips + images) to S3
        # Note: Original video is NOT uploaded - only evidence clips are uploaded.
        clip_files = result.get('clip_files', [])
        evidence_urls = []
        upload_errors = []
        s3_file_mapping: Dict[str, str] = {}

        if clip_files:
            logger.info(f"[UPLOAD] Uploading {len(clip_files)} evidence files (clips + images) to S3")
            all_files_to_upload = []
            for clip_file in clip_files:
                all_files_to_upload.append(clip_file)
                image_file = clip_file.replace('_clip.mp4', '_activity.jpg')
                if os.path.exists(image_file):
                    all_files_to_upload.append(image_file)
            logger.info(f"[UPLOAD] Total files to upload: {len(all_files_to_upload)} (clips + images)")

            clips_success, file_urls, clip_errors = s3_upload_service.upload_multiple_files(
                file_paths=all_files_to_upload, subfolder=subFolderName, auth_token=authToken,
            )

            for local_path, s3_url in zip(all_files_to_upload, file_urls):
                s3_file_mapping[local_path] = s3_url

            evidence_urls = [url for url in file_urls if '_clip.mp4' in url]
            upload_errors = clip_errors

            if not clips_success:
                logger.warning(f"Some files failed to upload: {clip_errors}")

            logger.info(f"[OK] Uploaded {len(file_urls)}/{len(all_files_to_upload)} files to S3")

        # Step 3 (continued): Update activities.json with S3 URLs.
        logger.info("[UPDATE] Updating activities.json with S3 URLs")
        run_dir = result.get('run_dir', result.get('runDir', ''))
        run_id = result.get('run_id', result.get('runId', ''))
        activities_count = result.get('activities_count', result.get('activitiesCount', 0))
        activities = result.get('activities', [])

        if run_dir and os.path.exists(run_dir):
            activities_json_path = os.path.join(run_dir, 'activities.json')
            if os.path.exists(activities_json_path):
                import json
                try:
                    with open(activities_json_path, 'r', encoding='utf-8') as f:
                        activities = json.load(f)
                    logger.info(f"[OK] Loaded {len(activities)} activities from {activities_json_path}")

                    updated_activities = []
                    for activity in activities:
                        if 'activityClip' in activity and activity['activityClip']:
                            local_clip_path = activity['activityClip']
                            if local_clip_path in s3_file_mapping:
                                activity['activityClip'] = s3_file_mapping[local_clip_path]
                                logger.debug(f"Updated clip URL: {local_clip_path} -> {s3_file_mapping[local_clip_path]}")
                        if 'activityImage' in activity and activity['activityImage']:
                            local_image_path = activity['activityImage']
                            if local_image_path in s3_file_mapping:
                                activity['activityImage'] = s3_file_mapping[local_image_path]
                                logger.debug(f"Updated image URL: {local_image_path} -> {s3_file_mapping[local_image_path]}")
                        updated_activities.append(activity)

                    # Save through the canonical atomic + locked writer.
                    from app.repositories.activity_repository import (
                        ActivityRepository as _ActivityRepository,
                    )
                    _ActivityRepository().save_activities(
                        updated_activities, os.path.dirname(activities_json_path),
                    )
                    activities = updated_activities
                    logger.info(f"[OK] Updated {len(updated_activities)} activities with S3 URLs")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to update activities.json: {e}", exc_info=True)
            else:
                logger.warning(f"[WARN] Activities JSON file not found: {activities_json_path}")
        else:
            logger.warning(f"[WARN] Run directory not found: {run_dir}")

        # Step 4: Post results to external API with evidence clip S3 URLs.
        external_api_result = None
        logger.info(f"[CHECK] Checking conditions for external API call: activities={len(activities) if activities else 0}")

        if activities:
            try:
                logger.info(f"[API] Posting results to external API with S3 URLs for trip: {tripId}")
                external_api_service = get_external_api_service()

                # Filter out STOPPED activities (only post RUNNING and UNCERTAIN),
                # EXCEPT safety-critical types whitelisted via
                # ``MOTION_FILTER_BYPASS_TYPES`` (defaults to cell_phone +
                # microsleep). Per spec, those violations matter regardless of
                # train motion state and must reach the external API.
                _bypass_raw = getattr(settings, 'motion_filter_bypass_types', '') or ''
                _bypass = {
                    s.strip().lower().replace(' ', '_')
                    for s in _bypass_raw.split(',') if s.strip()
                }
                def _postable(a):
                    if (a.get('motionState') or 'UNKNOWN').upper() != 'STOPPED':
                        return True
                    ot = (a.get('objectType') or '').strip().lower().replace(' ', '_')
                    return ot in _bypass
                postable_activities = [a for a in activities if _postable(a)]
                _bypassed = sum(
                    1 for a in activities
                    if (a.get('motionState') or '').upper() == 'STOPPED' and _postable(a)
                )
                _excluded = len(activities) - len(postable_activities)
                logger.info(
                    f"[API] Motion filter: {len(postable_activities)}/{len(activities)} "
                    f"activities to post (excluded {_excluded} STOPPED, "
                    f"bypassed {_bypassed} safety-critical STOPPED)"
                )

                external_api_result = external_api_service.post_cvvr_results(
                    trip_id=tripId,
                    events=postable_activities,
                    job_id=run_id,
                    video_s3_url=None,  # No original video URL - we don't upload original video
                    division=division,
                    run_dir=run_dir,  # Per-run DLQ scoping per spec 0004
                )

                if external_api_result.get("success"):
                    logger.info(
                        f"[OK] [external_api] Posted {external_api_result.get('violations_count', 0)} "
                        f"violations with S3 URLs to external API for trip {tripId}"
                    )
                else:
                    logger.warning(
                        f"[WARN] [external_api] Failed to post to external API: {external_api_result.get('message')}"
                    )
            except Exception as e:
                logger.error(f"[ERROR] [external_api] Exception while posting to external API: {e}", exc_info=True)
                external_api_result = {
                    "success": False,
                    "message": f"Exception: {str(e)}",
                    "posted": False
                }

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
                "activities": activities,
                "external_api_result": external_api_result,
            }
        }

        logger.info(f"[OK] Complete workflow finished for trip: {tripId}")
        return JSONResponse(content=response_data)

    except HTTPException:
        raise

    except Exception:
        # task 0010: keep the full traceback + tripId on the server side via
        # logger.exception, but NEVER let str(e) escape into the response
        # body — exception messages frequently embed file paths, query
        # fragments, and (worst case) request-derived secrets.
        if upload_to_external_api:
            video_basename = os.path.basename(video_path) if video_path else None
            logger.exception(
                "Process-and-upload failed for trip=%s video=%s",
                tripId, video_basename,
            )
        else:
            logger.exception("[ERROR] Video processing failed for trip %s", tripId)
            # Cleanup on error (analyze path only — process-and-upload doesn't
            # cleanup either, to match pre-refactor behaviour).
            if video_path:
                video_processing_service.cleanup_uploaded_video(video_path)
        raise HTTPException(status_code=500, detail="internal_error")

    finally:
        # Release the pending counter if we admitted the request but never
        # made it inside ``acquire_gpu_slot()`` (e.g. MinIO download failed
        # or the upload save raised). When the slot was acquired,
        # ``mark_enqueued_started()`` already decremented pending — nothing
        # to roll back.
        if admitted and not slot_acquired:
            try:
                gpu_resource_manager.release_enqueue_on_error()
            except Exception as _release_err:
                logger.warning(f"Failed to release pending slot on error: {_release_err}")


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
    - division: Division identifier (optional)
    - lpCrewName: LP crew member name (optional)
    - lpCrewId: LP crew member ID (optional)
    - alpCrewName: ALP crew member name (optional)
    - alpCrewId: ALP crew member ID (optional)
    - trainNumber: Train number for motion-based rules (optional, e.g., "12345")
    - tripDate: Trip date in YYYY-MM-DD format for motion-based rules (optional)
    - videoStartTime: Video recording start time in HH:MM:SS format (optional, for motion rules when OCR unavailable)
    - useMockDetection: Use mock detection for testing (optional, default: false)
    - useMultiprocessing: Enable parallel processing (optional, default: from config)
    - saveClips: Save annotated frames for debugging (optional, default: false). Clips and images are always saved for UI evidence.
    - cameraAngle: Camera angle for LP/ALP role assignment (1 = LP Side, 2 = ALP Side, default: 1)

    Returns:
    - Processing results with detected activities
    - Path to activities.json file
    - Summary statistics
    """
)
async def process_video(
    request: Request,
    background_tasks: BackgroundTasks,
    video: Optional[UploadFile] = File(default=None, description="Video file to process (optional if videoUrl provided)"),
    videoUrl: Optional[str] = Form(default=None, description="MinIO URL to download video from (e.g., https://mind.snikbtel.uk:9000/cvss/video.mp4)"),
    tripId: Optional[str] = Form(default=None, description="Unique trip identifier"),
    division: Optional[str] = Form(default=None, description="Division identifier"),
    lpCrewName: Optional[str] = Form(default=None, description="Loco Pilot crew member name"),
    lpCrewId: Optional[str] = Form(default=None, description="Loco Pilot crew member ID"),
    alpCrewName: Optional[str] = Form(default=None, description="Assistant Loco Pilot crew member name"),
    alpCrewId: Optional[str] = Form(default=None, description="Assistant Loco Pilot crew member ID"),
    trainNumber: Optional[str] = Form(default=None, description="Train number for motion-based rules (e.g., '12345')"),
    tripDate: Optional[str] = Form(default=None, description="Trip date in YYYY-MM-DD format for motion-based rules"),
    videoStartTime: Optional[str] = Form(default=None, description="Video recording start time in HH:MM:SS format (for motion rules when OCR unavailable)"),
    useMockDetection: Optional[bool] = Form(default=False, description="Use mock detection for testing"),
    useMultiprocessing: Optional[bool] = Form(default=None, description="Enable multiprocessing (default: from config)"),
    saveClips: Optional[bool] = Form(default=False, description="Save annotated frames for debugging (default: false). Clips/images always saved."),
    cameraAngle: Optional[int] = Form(default=1, description="Camera angle: 1 = LP Side (default), 2 = ALP Side")
):
    """
    Process uploaded video and detect activities

    This endpoint handles video upload or MinIO download, validation, processing,
    and activity detection. The video is saved temporarily, processed, and then cleaned up.

    Accepts both multipart/form-data and application/json content types.
    Use JSON when sending only a videoUrl (no file upload needed).
    """
    return await _run_video_pipeline(
        request=request,
        background_tasks=background_tasks,
        video=video,
        videoUrl=videoUrl,
        tripId=tripId,
        division=division,
        lpCrewName=lpCrewName,
        lpCrewId=lpCrewId,
        alpCrewName=alpCrewName,
        alpCrewId=alpCrewId,
        trainNumber=trainNumber,
        tripDate=tripDate,
        videoStartTime=videoStartTime,
        useMockDetection=useMockDetection,
        useMultiprocessing=useMultiprocessing,
        saveClips=saveClips,
        cameraAngle=cameraAngle,
        upload_to_external_api=False,
    )


@router.get(
    "/status/{run_id}",
    summary="Get processing status",
    description="Get the processing status for a specific run ID",
    dependencies=[Depends(require_api_key)],
)
async def get_processing_status(run_id: str):
    """
    Get processing status for a run

    Args:
        run_id: Run directory name (e.g., run_20251110_143045)
    """
    # C-9: reject malformed / traversal-style run_id before touching the
    # filesystem. Must be exactly ``run_YYYYMMDD_HHMMSS``.
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail="invalid_run_id")

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

    except Exception:
        # task 0010: log full traceback for diagnosis; opaque body for the caller.
        logger.exception("Failed to get status for run_id=%s", run_id)
        raise HTTPException(status_code=500, detail="internal_error")


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
                "matching the media URL pattern used in the first project.",
    dependencies=[Depends(require_api_key)],
)
async def get_run_media(run_id: str, filename: str, request: Request) -> Response:
    """
    Serve media files (clips/images) for a given run.

    The external API will call URLs of the form:
        {host_url}/api/jobs/{run_id}/media/{filename}

    We map that to:
        {settings.output_dir}/{run_id}/clips/{filename}
    """
    # C-9: reject malformed / traversal-style run_id before touching the
    # filesystem. Must be exactly ``run_YYYYMMDD_HHMMSS``. The ``commonpath``
    # check below still runs as defense-in-depth for the filename segment.
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail="invalid_run_id")

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
    """,
    # Task 0008: enforce ``Content-Length <= max_upload_size`` BEFORE
    # FastAPI parses the multipart body. ``Depends`` resolves the size
    # check before the ``File(...)``/``Form(...)`` body reads kick in.
    dependencies=[Depends(enforce_max_upload_size)],
)
async def process_and_upload_video(
    video_file: UploadFile = File(..., description="Video file to process"),
    tripId: str = Form(..., description="Unique trip identifier"),
    division: Optional[str] = Form(default=None, description="Division identifier"),
    subFolderName: str = Form(default="cvvr", description="S3 subfolder name"),
    authToken: Optional[str] = Form(default=None, description="Authentication token for S3 upload"),
    lpCrewName: Optional[str] = Form(default=None, description="Loco Pilot crew member name"),
    lpCrewId: Optional[str] = Form(default=None, description="Loco Pilot crew member ID"),
    alpCrewName: Optional[str] = Form(default=None, description="Assistant Loco Pilot crew member name"),
    alpCrewId: Optional[str] = Form(default=None, description="Assistant Loco Pilot crew member ID"),
    trainNumber: Optional[str] = Form(default=None, description="Train number for motion-based rules (e.g., '12345')"),
    tripDate: Optional[str] = Form(default=None, description="Trip date in YYYY-MM-DD format for motion-based rules"),
    videoStartTime: Optional[str] = Form(default=None, description="Video recording start time in HH:MM:SS format (for motion rules when OCR unavailable)"),
    useMultiprocessing: Optional[bool] = Form(default=None, description="Enable multiprocessing (default: from config)"),
    useMockDetection: Optional[bool] = Form(default=False, description="Use mock detection for testing"),
    saveClips: Optional[bool] = Form(default=False, description="Save annotated frames for debugging")
):
    """
    Process video and upload everything to S3

    This is the preferred endpoint for the desktop application as it handles
    the complete workflow in one request.
    """
    return await _run_video_pipeline(
        request=None,
        background_tasks=None,
        video=video_file,
        videoUrl=None,
        tripId=tripId,
        division=division,
        lpCrewName=lpCrewName,
        lpCrewId=lpCrewId,
        alpCrewName=alpCrewName,
        alpCrewId=alpCrewId,
        trainNumber=trainNumber,
        tripDate=tripDate,
        videoStartTime=videoStartTime,
        useMockDetection=useMockDetection,
        useMultiprocessing=useMultiprocessing,
        saveClips=saveClips,
        cameraAngle=None,
        upload_to_external_api=True,
        subFolderName=subFolderName,
        authToken=authToken,
    )


# =============================================================================
# ASYNC JOB-BASED ENDPOINTS
# =============================================================================

@router.post(
    "/video/jobs",
    response_model=JobSubmitResponse,
    responses={
        200: {"description": "Job submitted successfully"},
        400: {"description": "Invalid request"},
        503: {"description": "Queue is full - try again later"}
    },
    summary="Submit async video processing job",
    description="""
    Submit a video for asynchronous processing. Returns immediately with a job_id
    that can be used to track progress and retrieve results.

    This is the async alternative to /video/analyze. Use this for non-blocking
    video processing where you poll for status/results.

    The job queue has a bounded capacity. When the queue is full, a 503 error
    is returned (backpressure) - the client should retry after a delay.
    """
)
async def submit_video_job(request: JobSubmitRequest) -> JobSubmitResponse:
    """
    Submit a video processing job to the async queue.

    Returns job_id immediately for status polling.
    """
    logger.info(f"Received job submission request - video_path={request.video_path}")

    # Validate video path exists
    if not os.path.exists(request.video_path):
        logger.warning(f"Video file not found: {request.video_path}")
        raise HTTPException(
            status_code=400,
            detail=f"Video file not found: {request.video_path}"
        )

    # Check if queue is full before attempting submission
    if job_manager.is_queue_full:
        logger.warning("Job queue is full - rejecting submission")
        raise HTTPException(
            status_code=503,
            detail="Job queue is full. Please try again later."
        )

    try:
        # Submit job to queue
        job_id = await job_manager.submit_job(
            video_path=request.video_path,
            config=request.config
        )

        queue_status = job_manager.get_queue_status()

        logger.info(f"Job submitted successfully - job_id={job_id}")

        return JobSubmitResponse(
            success=True,
            job_id=job_id,
            message="Job submitted successfully",
            queue_position=queue_status["queue_depth"]
        )

    except asyncio.QueueFull:
        logger.warning("Queue full during submission attempt")
        raise HTTPException(
            status_code=503,
            detail="Job queue is full. Please try again later."
        )


@router.get(
    "/video/jobs/{job_id}",
    response_model=JobStatusResponse,
    responses={
        200: {"description": "Job status retrieved successfully"},
        404: {"description": "Job not found"}
    },
    summary="Get job status",
    description="""
    Get the current status and progress of a video processing job.

    Poll this endpoint to track job progress. The progress field shows
    percentage completion (0-100).
    """
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Get status and progress of a video processing job.
    """
    logger.info(f"Getting job status - job_id={job_id}")

    job = await job_manager.get_job(job_id)

    if not job:
        logger.warning(f"Job not found: {job_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )

    # Calculate processing time if started
    processing_time = None
    if job.started_at:
        end_time = job.completed_at or job.started_at
        processing_time = (end_time - job.started_at).total_seconds()

    return JobStatusResponse(
        success=True,
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        video_path=job.video_path,
        config=job.config,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        processing_time_seconds=processing_time
    )


@router.get(
    "/video/jobs/{job_id}/result",
    response_model=JobResultResponse,
    responses={
        200: {"description": "Job result retrieved successfully"},
        400: {"description": "Job not completed yet"},
        404: {"description": "Job not found"}
    },
    summary="Get job result",
    description="""
    Get the processing results for a completed job.

    This endpoint only returns results for jobs with COMPLETED status.
    For jobs in other states, use /video/jobs/{job_id} to check status.

    Returns the full activities JSON and processing metadata.
    """
)
async def get_job_result(job_id: str) -> JobResultResponse:
    """
    Get results for a completed video processing job.
    """
    logger.info(f"Getting job result - job_id={job_id}")

    job = await job_manager.get_job(job_id)

    if not job:
        logger.warning(f"Job not found: {job_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )

    # Check if job is completed
    if job.status != JobStatus.COMPLETED:
        status_message = {
            JobStatus.PENDING: "Job is pending in queue",
            JobStatus.QUEUED: "Job is waiting in queue",
            JobStatus.PROCESSING: f"Job is still processing ({job.progress}% complete)",
            JobStatus.FAILED: f"Job failed: {job.error}",
            JobStatus.CANCELLED: "Job was cancelled"
        }
        raise HTTPException(
            status_code=400,
            detail=status_message.get(job.status, f"Job status is {job.status.value}")
        )

    # Calculate processing time
    processing_time = None
    if job.started_at and job.completed_at:
        processing_time = (job.completed_at - job.started_at).total_seconds()

    return JobResultResponse(
        success=True,
        job_id=job.id,
        status=job.status,
        result=job.result or {},
        processing_time_seconds=processing_time
    )


@router.post(
    "/video/jobs/{job_id}/cancel",
    response_model=JobCancelResponse,
    responses={
        200: {"description": "Job cancellation result"},
        404: {"description": "Job not found"}
    },
    summary="Cancel a job",
    description="""
    Cancel a pending or running job.

    Jobs can only be cancelled if they are in PENDING, QUEUED, or PROCESSING
    status. Already completed, failed, or cancelled jobs cannot be cancelled.

    Note: Cancellation of a PROCESSING job is best-effort - the job may
    complete before the cancellation takes effect.
    """
)
async def cancel_job(job_id: str) -> JobCancelResponse:
    """
    Cancel a video processing job.
    """
    logger.info(f"Cancelling job - job_id={job_id}")

    job = await job_manager.get_job(job_id)

    if not job:
        logger.warning(f"Job not found: {job_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )

    previous_status = job.status
    cancelled = await job_manager.cancel_job(job_id)

    if cancelled:
        logger.info(f"Job cancelled successfully - job_id={job_id}")
        return JobCancelResponse(
            success=True,
            job_id=job_id,
            message="Job cancelled successfully",
            previous_status=previous_status
        )
    else:
        message = f"Cannot cancel job with status: {previous_status.value}"
        logger.info(f"Job cancellation failed - job_id={job_id}, reason={message}")
        return JobCancelResponse(
            success=False,
            job_id=job_id,
            message=message,
            previous_status=previous_status
        )


@router.get(
    "/video/queue/status",
    response_model=QueueStatusResponse,
    responses={
        200: {"description": "Queue status retrieved successfully"}
    },
    summary="Get queue status",
    description="""
    Get the current status of the job queue.

    Returns information about:
    - Queue depth (jobs waiting)
    - Active jobs (currently processing)
    - Pending jobs (not yet started)
    - Completed/failed/cancelled job counts
    - Queue capacity information
    """
)
async def get_queue_status() -> QueueStatusResponse:
    """
    Get current job queue statistics.
    """
    logger.info("Getting queue status")

    status = job_manager.get_queue_status()

    return QueueStatusResponse(
        success=True,
        queue_depth=status["queue_depth"],
        active_jobs=status["active_jobs"],
        pending_jobs=status["pending_jobs"],
        completed_jobs=status["completed_jobs"],
        failed_jobs=status["failed_jobs"],
        cancelled_jobs=status["cancelled_jobs"],
        total_jobs=status["total_jobs"],
        max_queue_size=status["max_queue_size"],
        num_workers=status["num_workers"],
        queue_full=status["queue_full"]
    )


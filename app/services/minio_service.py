"""
MinIO service for downloading videos from MinIO/S3 storage

Handles video downloads from MinIO URLs for processing.
"""

import os
import tempfile
import threading
import urllib3
from typing import Optional, Tuple
from urllib.parse import urlparse, unquote
from functools import lru_cache

from minio import Minio
from minio.error import S3Error

from ..utils.logger import get_logger
from ..utils.config import get_settings
from ..utils.url_safety import URLNotAllowed, validate_external_url


logger = get_logger(__name__)
settings = get_settings()

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class MinioService:
    """
    Service for downloading videos from MinIO storage
    """

    def __init__(self):
        """Initialize MinIO client"""
        self.client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
            cert_check=False  # Skip certificate verification for self-signed certs
        )
        self.default_bucket = settings.minio_bucket
        logger.info(f"MinIO client initialized - endpoint: {settings.minio_endpoint}, bucket: {self.default_bucket}")

    def parse_minio_url(self, url: str) -> Tuple[str, str]:
        """
        Parse a MinIO URL to extract bucket and object key

        Supports formats:
        - https://endpoint:port/bucket/path/to/object.mp4
        - http://endpoint:port/bucket/path/to/object.mp4

        Args:
            url: Full MinIO URL

        Returns:
            Tuple of (bucket_name, object_key)

        Raises:
            ValueError: If URL cannot be parsed
        """
        try:
            parsed = urlparse(url)
            path = parsed.path.lstrip('/')

            if not path:
                raise ValueError(f"Invalid MinIO URL - no path: {url}")

            # Split path into bucket and object key
            parts = path.split('/', 1)

            if len(parts) < 2:
                raise ValueError(f"Invalid MinIO URL - missing object key: {url}")

            bucket = parts[0]
            object_key = unquote(parts[1])  # Decode URL-encoded characters (e.g., %20 -> space)

            if not bucket or not object_key:
                raise ValueError(f"Invalid MinIO URL - empty bucket or key: {url}")

            logger.debug(f"Parsed MinIO URL - bucket: {bucket}, key: {object_key}")
            return bucket, object_key

        except Exception as e:
            logger.error(f"Failed to parse MinIO URL: {url} - {e}")
            raise ValueError(f"Invalid MinIO URL: {url}") from e

    def download_video(self, video_url: str, trip_id: str) -> str:
        """
        Download a video from MinIO to local storage

        Args:
            video_url: Full MinIO URL (e.g., https://mind.snikbtel.uk:9000/cvss/video.mp4)
            trip_id: Trip ID for naming the downloaded file

        Returns:
            Local file path where video was downloaded

        Raises:
            ValueError: If URL is invalid
            S3Error: If download fails
        """
        try:
            # Task 0008 / reviewer M1 — defense in depth. The controller
            # already calls ``validate_external_url`` before invoking us,
            # but a future caller (worker, internal job, retry helper)
            # might forget. Re-validate here so any bypass at the
            # controller layer still fails closed.
            validate_external_url(video_url, settings.minio_allowed_hosts)

            # Parse URL to get bucket and object key
            bucket, object_key = self.parse_minio_url(video_url)

            # Get file extension from object key
            file_ext = os.path.splitext(object_key)[1] or '.mp4'

            # Create download directory if needed
            download_dir = settings.upload_dir
            os.makedirs(download_dir, exist_ok=True)

            # Generate local filename
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            local_filename = f"{trip_id}_{timestamp}_minio{file_ext}"
            local_path = os.path.join(download_dir, local_filename)

            logger.info(f"Downloading video from MinIO - bucket: {bucket}, key: {object_key}")

            # Task 0008: stream the object and enforce a hard byte cap. A
            # hostile or misconfigured remote could otherwise return a
            # multi-hundred-GB stream and exhaust local disk. We use
            # ``get_object`` (streaming response) instead of
            # ``fget_object`` (atomic write) so we can abort mid-stream.
            max_bytes = settings.max_external_download_bytes
            total = 0
            chunk_size = 8 * 1024 * 1024  # 8 MiB
            response = None
            try:
                response = self.client.get_object(bucket, object_key)
                with open(local_path, "wb") as out_f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        total += len(chunk)
                        if max_bytes and total > max_bytes:
                            # Truncate the partial download before raising
                            # so we don't leave a half-written file
                            # masquerading as a valid video.
                            out_f.close()
                            try:
                                os.unlink(local_path)
                            except OSError:
                                pass
                            raise URLNotAllowed(
                                f"download exceeded max bytes "
                                f"({total} > {max_bytes}) for "
                                f"{bucket}/{object_key}"
                            )
                        out_f.write(chunk)
            finally:
                if response is not None:
                    try:
                        response.close()
                        response.release_conn()
                    except Exception:  # pragma: no cover - defensive
                        pass

            # Verify download
            if not os.path.exists(local_path):
                raise RuntimeError(f"Download completed but file not found: {local_path}")

            file_size = os.path.getsize(local_path)
            logger.info(f"Downloaded video: {local_path} ({file_size / (1024*1024):.2f} MB)")

            return local_path

        except S3Error as e:
            logger.error(f"MinIO download failed - {e.code}: {e.message}")
            raise
        except Exception as e:
            logger.error(f"Failed to download video from MinIO: {e}", exc_info=True)
            raise

    def check_object_exists(self, video_url: str) -> bool:
        """
        Check if an object exists in MinIO

        Args:
            video_url: Full MinIO URL

        Returns:
            True if object exists, False otherwise
        """
        try:
            bucket, object_key = self.parse_minio_url(video_url)
            self.client.stat_object(bucket, object_key)
            return True
        except S3Error as e:
            if e.code == 'NoSuchKey':
                return False
            raise
        except Exception:
            return False


# Singleton instance
_minio_service: Optional[MinioService] = None
_minio_service_lock = threading.Lock()


@lru_cache()
def get_minio_service() -> MinioService:
    """
    Get or create the MinIO service singleton.

    M-25: Thread-safe double-checked locking pattern.

    Returns:
        MinioService instance
    """
    global _minio_service
    if _minio_service is None:
        with _minio_service_lock:
            if _minio_service is None:
                _minio_service = MinioService()
    return _minio_service

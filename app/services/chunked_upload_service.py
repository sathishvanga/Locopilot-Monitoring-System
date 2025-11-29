"""
Chunked upload service for managing resumable file uploads.

This service handles the 3-step chunked upload workflow:
1. Initiate upload session → Returns upload_id
2. Upload chunks → Saves individual parts
3. Complete upload → Assembles chunks into final file

Sessions are stored in temporary directories with metadata and automatically
cleaned up after the configured TTL (default: 24 hours).
"""

import os
import json
import uuid
import shutil
import math
from typing import Tuple, Optional, List, Dict
from datetime import datetime, timedelta
from fastapi import UploadFile
import aiofiles

from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.utils.disk_utils import ensure_directory_exists

settings = get_settings()
logger = get_logger(__name__)


class ChunkedUploadService:
    """Service for managing chunked/resumable video uploads."""

    def __init__(self):
        """Initialize the chunked upload service."""
        self.chunk_dir = settings.chunk_upload_dir
        self.final_dir = settings.upload_dir
        self.chunk_size = settings.chunk_size_default
        self.session_ttl_hours = settings.chunk_session_ttl_hours

        # Ensure directories exist
        ensure_directory_exists(self.chunk_dir)
        ensure_directory_exists(self.final_dir)

    def _get_session_dir(self, upload_id: str) -> str:
        """Get the directory path for an upload session."""
        return os.path.join(self.chunk_dir, upload_id)

    def _get_meta_file(self, upload_id: str) -> str:
        """Get the metadata file path for an upload session."""
        return os.path.join(self._get_session_dir(upload_id), "meta.json")

    def _get_chunk_file(self, upload_id: str, part_number: int) -> str:
        """Get the file path for a specific chunk."""
        return os.path.join(
            self._get_session_dir(upload_id),
            f"part_{part_number:06d}.chunk"
        )

    async def initiate_upload(
        self,
        filename: str,
        total_size: int,
        trip_id: str
    ) -> Tuple[str, int, int]:
        """
        Initiate a new chunked upload session.

        Args:
            filename: Original filename of the video
            total_size: Total file size in bytes
            trip_id: Trip identifier

        Returns:
            Tuple of (upload_id, chunk_size_recommendation, total_chunks_expected)

        Raises:
            OSError: If unable to create session directory or metadata file
        """
        # Generate unique upload ID
        upload_id = str(uuid.uuid4())
        session_dir = self._get_session_dir(upload_id)

        try:
            # Create session directory
            os.makedirs(session_dir, exist_ok=True)

            # Calculate total chunks expected
            total_chunks = math.ceil(total_size / self.chunk_size)

            # Create metadata
            created_at = datetime.utcnow()
            expires_at = created_at + timedelta(hours=self.session_ttl_hours)

            metadata = {
                "filename": filename,
                "total_size": total_size,
                "trip_id": trip_id,
                "created_at": created_at.isoformat(),
                "expires_at": expires_at.isoformat(),
                "chunk_size_recommendation": self.chunk_size,
                "total_chunks_expected": total_chunks
            }

            # Save metadata
            meta_file = self._get_meta_file(upload_id)
            with open(meta_file, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                f"📥 Initiated upload session {upload_id} for {filename} "
                f"({total_size} bytes, {total_chunks} chunks expected)"
            )

            return upload_id, self.chunk_size, total_chunks

        except Exception as e:
            # Cleanup on error
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir, ignore_errors=True)
            logger.error(f"❌ Failed to initiate upload session: {e}", exc_info=True)
            raise

    async def save_chunk(
        self,
        upload_id: str,
        part_number: int,
        chunk: UploadFile
    ) -> bool:
        """
        Save a chunk for an upload session.

        Args:
            upload_id: Upload session ID
            part_number: 1-based part number
            chunk: File chunk to save

        Returns:
            True if chunk saved successfully, False if session not found

        Raises:
            OSError: If unable to write chunk file
        """
        session_dir = self._get_session_dir(upload_id)

        # Check if session exists
        if not os.path.exists(session_dir):
            logger.warning(f"⚠️ Upload session not found: {upload_id}")
            return False

        chunk_file = self._get_chunk_file(upload_id, part_number)

        try:
            # Save chunk to disk
            async with aiofiles.open(chunk_file, "wb") as f:
                while True:
                    data = await chunk.read(8 * 1024 * 1024)  # Read in 8MB blocks
                    if not data:
                        break
                    await f.write(data)

            chunk_size = os.path.getsize(chunk_file)
            logger.info(
                f"📦 Chunk {part_number} received for {upload_id} "
                f"({chunk_size} bytes)"
            )

            return True

        except Exception as e:
            # Cleanup partial chunk on error
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
            logger.error(
                f"❌ Failed to save chunk {part_number} for {upload_id}: {e}",
                exc_info=True
            )
            raise

    async def complete_upload(self, upload_id: str) -> Tuple[str, int]:
        """
        Complete upload by assembling all chunks into final file.

        Args:
            upload_id: Upload session ID

        Returns:
            Tuple of (final_file_path, total_bytes_written)

        Raises:
            FileNotFoundError: If session or metadata not found
            ValueError: If assembled size doesn't match expected size
            OSError: If unable to assemble or write final file
        """
        session_dir = self._get_session_dir(upload_id)
        meta_file = self._get_meta_file(upload_id)

        # Check if session exists
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Upload session not found: {upload_id}")

        try:
            # Load metadata
            with open(meta_file, "r") as f:
                metadata = json.load(f)

            filename = metadata["filename"]
            total_size_expected = metadata["total_size"]
            trip_id = metadata["trip_id"]

            # List all chunk files in order
            chunk_files = sorted([
                f for f in os.listdir(session_dir)
                if f.startswith("part_") and f.endswith(".chunk")
            ])

            if not chunk_files:
                raise ValueError("No chunks found for assembly")

            logger.info(
                f"🔧 Assembling {len(chunk_files)} chunks for {upload_id}"
            )

            # Create final filename (safe, unique)
            import time
            file_ext = os.path.splitext(filename)[1]
            safe_filename = f"{trip_id}_{int(time.time())}{file_ext}"
            final_path = os.path.join(self.final_dir, safe_filename)

            # Assemble chunks into final file
            bytes_written = 0
            with open(final_path, "wb") as dest:
                for chunk_file in chunk_files:
                    chunk_path = os.path.join(session_dir, chunk_file)
                    with open(chunk_path, "rb") as chunk_src:
                        while True:
                            data = chunk_src.read(8 * 1024 * 1024)  # 8MB blocks
                            if not data:
                                break
                            dest.write(data)
                            bytes_written += len(data)

            # Verify assembled size matches expected
            if bytes_written != total_size_expected:
                # Cleanup partial file
                if os.path.exists(final_path):
                    os.remove(final_path)
                raise ValueError(
                    f"Assembled size mismatch: {bytes_written} bytes written, "
                    f"{total_size_expected} bytes expected"
                )

            logger.info(
                f"✅ Upload complete: {final_path} ({bytes_written} bytes assembled)"
            )

            return final_path, bytes_written

        except Exception as e:
            logger.error(
                f"❌ Failed to complete upload {upload_id}: {e}",
                exc_info=True
            )
            raise

    def cleanup_upload_session(self, upload_id: str) -> None:
        """
        Clean up upload session directory and all chunks.

        Args:
            upload_id: Upload session ID to cleanup

        Note:
            This is safe to call even if session doesn't exist.
        """
        session_dir = self._get_session_dir(upload_id)

        try:
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir)
                logger.info(f"🗑️ Cleaned up upload session: {upload_id}")
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to cleanup upload session {upload_id}: {e}"
            )

    def get_metadata(self, upload_id: str) -> Optional[Dict]:
        """
        Get metadata for an upload session.

        Args:
            upload_id: Upload session ID

        Returns:
            Metadata dict or None if session not found
        """
        meta_file = self._get_meta_file(upload_id)

        try:
            if os.path.exists(meta_file):
                with open(meta_file, "r") as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"❌ Failed to read metadata for {upload_id}: {e}")
            return None

    def get_uploaded_chunks(self, upload_id: str) -> List[int]:
        """
        Get list of successfully uploaded chunk numbers.

        Args:
            upload_id: Upload session ID

        Returns:
            Sorted list of part numbers (1-based)
        """
        session_dir = self._get_session_dir(upload_id)

        if not os.path.exists(session_dir):
            return []

        try:
            chunk_files = [
                f for f in os.listdir(session_dir)
                if f.startswith("part_") and f.endswith(".chunk")
            ]

            # Extract part numbers from filenames (part_000001.chunk -> 1)
            part_numbers = []
            for chunk_file in chunk_files:
                part_str = chunk_file.replace("part_", "").replace(".chunk", "")
                part_numbers.append(int(part_str))

            return sorted(part_numbers)

        except Exception as e:
            logger.error(
                f"❌ Failed to get uploaded chunks for {upload_id}: {e}"
            )
            return []

    def get_session_status(self, upload_id: str) -> Optional[Dict]:
        """
        Get complete status of an upload session.

        Args:
            upload_id: Upload session ID

        Returns:
            Status dict with all session info, or None if not found
        """
        metadata = self.get_metadata(upload_id)
        if not metadata:
            return None

        chunks_uploaded = self.get_uploaded_chunks(upload_id)
        total_chunks_expected = metadata.get("total_chunks_expected", 0)

        # Calculate bytes uploaded so far
        session_dir = self._get_session_dir(upload_id)
        bytes_uploaded = 0
        for part_num in chunks_uploaded:
            chunk_file = self._get_chunk_file(upload_id, part_num)
            if os.path.exists(chunk_file):
                bytes_uploaded += os.path.getsize(chunk_file)

        return {
            "upload_id": upload_id,
            "filename": metadata.get("filename"),
            "total_size": metadata.get("total_size"),
            "trip_id": metadata.get("trip_id"),
            "created_at": metadata.get("created_at"),
            "expires_at": metadata.get("expires_at"),
            "chunks_uploaded": chunks_uploaded,
            "total_chunks_expected": total_chunks_expected,
            "is_complete": len(chunks_uploaded) == total_chunks_expected,
            "bytes_uploaded": bytes_uploaded
        }

    def cleanup_expired_sessions(self, max_age_hours: Optional[int] = None) -> int:
        """
        Clean up upload sessions older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours (default: use config value)

        Returns:
            Number of sessions cleaned up
        """
        if max_age_hours is None:
            max_age_hours = self.session_ttl_hours

        if not os.path.exists(self.chunk_dir):
            return 0

        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        cleaned_count = 0

        try:
            for session_id in os.listdir(self.chunk_dir):
                session_dir = os.path.join(self.chunk_dir, session_id)

                if not os.path.isdir(session_dir):
                    continue

                # Check metadata for expiration
                meta_file = os.path.join(session_dir, "meta.json")
                if os.path.exists(meta_file):
                    try:
                        with open(meta_file, "r") as f:
                            metadata = json.load(f)
                        created_at_str = metadata.get("created_at")
                        if created_at_str:
                            created_at = datetime.fromisoformat(created_at_str)
                            if created_at < cutoff_time:
                                shutil.rmtree(session_dir)
                                cleaned_count += 1
                                logger.info(
                                    f"🗑️ Cleaned up expired session: {session_id}"
                                )
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Failed to process session {session_id}: {e}"
                        )
                else:
                    # No metadata, check directory modification time
                    dir_mtime = datetime.fromtimestamp(os.path.getmtime(session_dir))
                    if dir_mtime < cutoff_time:
                        shutil.rmtree(session_dir)
                        cleaned_count += 1
                        logger.info(
                            f"🗑️ Cleaned up orphaned session: {session_id}"
                        )

            if cleaned_count > 0:
                logger.info(
                    f"✅ Cleanup complete: {cleaned_count} expired sessions removed"
                )

            return cleaned_count

        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}", exc_info=True)
            return cleaned_count


# Singleton instance
_chunked_upload_service: Optional[ChunkedUploadService] = None


def get_chunked_upload_service() -> ChunkedUploadService:
    """
    Get singleton instance of ChunkedUploadService.

    Returns:
        ChunkedUploadService instance
    """
    global _chunked_upload_service
    if _chunked_upload_service is None:
        _chunked_upload_service = ChunkedUploadService()
    return _chunked_upload_service

"""
Chunked Upload Service - Manages chunked video upload sessions

Handles large video uploads by splitting them into smaller chunks,
storing them temporarily, and reassembling them into complete files.
"""

import os
import time
import uuid
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Set, List, Tuple, Optional
from functools import lru_cache

from ..utils.logger import get_logger
from ..utils.config import get_settings


logger = get_logger(__name__)
settings = get_settings()


@dataclass
class UploadSession:
    """
    Represents an active chunked upload session
    """
    upload_id: str
    trip_id: str
    filename: str
    total_chunks: int
    total_size: int
    chunk_size: int = 8388608  # Fixed 8 MB
    received_chunks: Set[int] = field(default_factory=set)
    chunks_dir: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=1))
    metadata: dict = field(default_factory=dict)
    status: str = "uploading"  # "uploading", "complete", "failed"


class ChunkedUploadService:
    """
    Service for managing chunked video uploads

    Provides functionality to:
    - Create and manage upload sessions
    - Store chunks to disk
    - Validate chunk completeness
    - Reassemble chunks into final video
    - Cleanup expired sessions
    """

    def __init__(self):
        """Initialize the chunked upload service"""
        self.sessions: Dict[str, UploadSession] = {}
        self.chunks_base_dir = os.path.join(settings.upload_dir, "chunks")
        self.session_timeout = settings.upload_session_timeout

        # Ensure chunks directory exists
        os.makedirs(self.chunks_base_dir, exist_ok=True)

        logger.info(f"ChunkedUploadService initialized - Base dir: {self.chunks_base_dir}")

    def initiate_upload(
        self,
        trip_id: str,
        filename: str,
        total_size: int,
        metadata: Optional[dict] = None
    ) -> UploadSession:
        """
        Create a new upload session

        Args:
            trip_id: Trip identifier
            filename: Original filename
            total_size: Total size of the video in bytes
            metadata: Optional metadata (crew info, etc.)

        Returns:
            UploadSession: Created upload session

        Raises:
            ValueError: If max sessions limit reached
        """
        # Check session limit
        if len(self.sessions) >= settings.max_upload_sessions:
            raise ValueError(
                f"Maximum upload sessions ({settings.max_upload_sessions}) reached. "
                "Please try again later."
            )

        # Generate unique upload ID
        upload_id = str(uuid.uuid4())

        # Calculate total chunks
        total_chunks = (total_size + settings.chunk_size - 1) // settings.chunk_size

        # Create chunks directory for this upload
        chunks_dir = os.path.join(self.chunks_base_dir, upload_id)
        os.makedirs(chunks_dir, exist_ok=True)

        # Create session
        session = UploadSession(
            upload_id=upload_id,
            trip_id=trip_id,
            filename=filename,
            total_chunks=total_chunks,
            total_size=total_size,
            chunk_size=settings.chunk_size,
            chunks_dir=chunks_dir,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=self.session_timeout),
            metadata=metadata or {},
            status="uploading"
        )

        # Store session
        self.sessions[upload_id] = session

        logger.info(
            f"📝 Upload session created - ID: {upload_id}, Trip: {trip_id}, "
            f"Chunks: {total_chunks}, Size: {total_size} bytes"
        )

        return session

    async def save_chunk(
        self,
        upload_id: str,
        chunk_index: int,
        chunk_data: bytes
    ) -> Tuple[bool, str]:
        """
        Save a chunk to disk

        Args:
            upload_id: Upload session ID
            chunk_index: Index of this chunk (0-based)
            chunk_data: Chunk file data

        Returns:
            Tuple[bool, str]: (success, message)

        Raises:
            ValueError: If session not found, expired, or invalid chunk
        """
        # Get session
        session = self.sessions.get(upload_id)
        if not session:
            raise ValueError(f"Upload session not found: {upload_id}")

        # Check if session expired
        if datetime.now() > session.expires_at:
            self.cleanup_upload(upload_id)
            raise ValueError(f"Upload session expired: {upload_id}")

        # Validate chunk index
        if chunk_index < 0 or chunk_index >= session.total_chunks:
            raise ValueError(
                f"Chunk index {chunk_index} out of range (0-{session.total_chunks-1})"
            )

        # Validate chunk size
        chunk_size = len(chunk_data)
        expected_size = settings.chunk_size

        # Last chunk can be smaller
        if chunk_index == session.total_chunks - 1:
            expected_size = session.total_size - (chunk_index * settings.chunk_size)

        if chunk_size > settings.chunk_size:
            raise ValueError(
                f"Chunk {chunk_index} size {chunk_size} exceeds maximum {settings.chunk_size}"
            )

        # Save chunk to disk
        chunk_path = os.path.join(session.chunks_dir, f"chunk_{chunk_index:04d}.bin")

        try:
            with open(chunk_path, 'wb') as f:
                f.write(chunk_data)

            # Update session
            session.received_chunks.add(chunk_index)

            logger.debug(
                f"💾 Saved chunk {chunk_index}/{session.total_chunks-1} "
                f"({chunk_size} bytes) for upload {upload_id}"
            )

            return (
                True,
                f"Chunk {chunk_index} received ({len(session.received_chunks)}/{session.total_chunks})"
            )

        except Exception as e:
            logger.error(f"Failed to save chunk {chunk_index}: {e}", exc_info=True)
            raise ValueError(f"Failed to save chunk: {str(e)}")

    def validate_chunks(self, upload_id: str) -> Tuple[bool, List[int]]:
        """
        Validate all chunks are present

        Args:
            upload_id: Upload session ID

        Returns:
            Tuple[bool, List[int]]: (all_present, missing_indices)
        """
        session = self.sessions.get(upload_id)
        if not session:
            return (False, [])

        # Check for missing chunks
        missing_chunks = []
        for i in range(session.total_chunks):
            if i not in session.received_chunks:
                missing_chunks.append(i)

        is_valid = len(missing_chunks) == 0

        if not is_valid:
            logger.warning(
                f"⚠️ Upload {upload_id} missing {len(missing_chunks)} chunks: {missing_chunks[:10]}"
            )

        return (is_valid, missing_chunks)

    async def reassemble_video(self, upload_id: str) -> str:
        """
        Reassemble chunks into final video file

        Args:
            upload_id: Upload session ID

        Returns:
            str: Path to reassembled video file

        Raises:
            ValueError: If chunks are missing or reassembly fails
        """
        session = self.sessions.get(upload_id)
        if not session:
            raise ValueError(f"Upload session not found: {upload_id}")

        # Validate all chunks present
        is_valid, missing = self.validate_chunks(upload_id)
        if not is_valid:
            raise ValueError(f"Cannot reassemble: Missing chunks {missing}")

        # Create final video path
        file_ext = os.path.splitext(session.filename)[1].lower()
        final_path = os.path.join(
            settings.upload_dir,
            f"{session.trip_id}_{int(time.time())}{file_ext}"
        )

        logger.info(f"🔧 Reassembling {session.total_chunks} chunks into {final_path}")

        try:
            # Reassemble chunks in order
            total_written = 0
            with open(final_path, 'wb') as outfile:
                for i in range(session.total_chunks):
                    chunk_path = os.path.join(
                        session.chunks_dir,
                        f"chunk_{i:04d}.bin"
                    )

                    if not os.path.exists(chunk_path):
                        raise ValueError(f"Chunk file missing: chunk_{i:04d}.bin")

                    # Read and write in 64KB blocks for efficiency
                    with open(chunk_path, 'rb') as chunk_file:
                        while True:
                            data = chunk_file.read(65536)  # 64KB blocks
                            if not data:
                                break
                            outfile.write(data)
                            total_written += len(data)

            # Verify size matches expected
            if total_written != session.total_size:
                os.remove(final_path)
                raise ValueError(
                    f"Size mismatch: expected {session.total_size} bytes, "
                    f"got {total_written} bytes"
                )

            # Update session status
            session.status = "complete"

            logger.info(
                f"✅ Reassembled video: {final_path} "
                f"({total_written} bytes, {session.total_chunks} chunks)"
            )

            return final_path

        except Exception as e:
            logger.error(f"❌ Reassembly failed: {e}", exc_info=True)
            if os.path.exists(final_path):
                os.remove(final_path)
            session.status = "failed"
            raise ValueError(f"Failed to reassemble video: {str(e)}")

    def get_upload_status(self, upload_id: str) -> Optional[UploadSession]:
        """
        Get upload session status

        Args:
            upload_id: Upload session ID

        Returns:
            UploadSession or None if not found
        """
        return self.sessions.get(upload_id)

    def cleanup_upload(self, upload_id: str) -> None:
        """
        Clean up upload session and chunks

        Args:
            upload_id: Upload session ID
        """
        session = self.sessions.get(upload_id)
        if not session:
            return

        try:
            # Remove chunks directory
            if os.path.exists(session.chunks_dir):
                shutil.rmtree(session.chunks_dir)
                logger.debug(f"🗑️  Cleaned up chunks for upload {upload_id}")

        except Exception as e:
            logger.warning(f"Failed to cleanup chunks for {upload_id}: {e}")

        finally:
            # Remove session from memory
            if upload_id in self.sessions:
                del self.sessions[upload_id]

    def cleanup_expired_sessions(self) -> None:
        """
        Clean up expired upload sessions

        This method should be called periodically as a background task.
        """
        now = datetime.now()
        expired_ids = []

        for upload_id, session in self.sessions.items():
            if now > session.expires_at:
                expired_ids.append(upload_id)

        if expired_ids:
            logger.info(f"🧹 Cleaning up {len(expired_ids)} expired upload sessions")

            for upload_id in expired_ids:
                self.cleanup_upload(upload_id)

        else:
            logger.debug("No expired sessions to clean up")


# Singleton pattern
@lru_cache()
def get_chunked_upload_service() -> ChunkedUploadService:
    """
    Get cached ChunkedUploadService instance

    Returns:
        ChunkedUploadService: Singleton service instance
    """
    return ChunkedUploadService()

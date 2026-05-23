"""Evidence management - clips, images, and reports."""
from typing import Dict, List, Any, Optional
import os
import json
import subprocess
import logging
from datetime import datetime

import cv2
import numpy as np

from app.utils.json_utils import atomic_write_json
# NOTE: ``ActivityRepository`` is imported lazily inside ``save_activities_json``
# below to avoid a circular import: this module is loaded as part of
# ``app.core`` package init, and ``activity_repository`` -> ``activity_models``
# -> ``app.core.activity_registry`` would re-enter ``app.core.__init__`` while
# this module is still being loaded.


class EvidenceManager:
    """Manages evidence artifacts: clips, images, JSON reports.

    This class handles all evidence-related operations including:
    - Video clip extraction using ffmpeg
    - H.264 re-encoding for browser compatibility
    - Activity image capture and saving
    - Annotated frame saving
    - JSON report generation
    """

    def __init__(
        self,
        output_dir: str,
        run_dir: str = None,
        save_annotated_frames: bool = False,
        frame_save_interval: int = 1,
        logger: logging.Logger = None
    ):
        """Initialize EvidenceManager.

        Args:
            output_dir: Base output directory for evidence
            run_dir: Specific run directory (if None, creates timestamped directory)
            save_annotated_frames: Whether to save annotated frames
            frame_save_interval: Save 1 frame every N sampled frames (1 = save all)
            logger: Logger instance (creates default if None)
        """
        self.output_dir = output_dir
        self.save_annotated_frames = save_annotated_frames
        self.frame_save_interval = frame_save_interval
        self.logger = logger or logging.getLogger(__name__)
        self.evidence_counter = 0

        # Create or use existing run directory
        if run_dir is not None:
            self.run_dir = run_dir
            self.run_timestamp = os.path.basename(run_dir).replace("run_", "")
        else:
            self.run_dir = self._create_run_directory()
            self.run_timestamp = os.path.basename(self.run_dir).replace("run_", "")

        # Set up subdirectories
        self.clips_dir = os.path.join(self.run_dir, 'clips')
        self.frames_dir = os.path.join(self.run_dir, 'frames')

        # Create directories
        os.makedirs(self.clips_dir, exist_ok=True)
        if self.save_annotated_frames:
            os.makedirs(self.frames_dir, exist_ok=True)

    def _create_run_directory(self) -> str:
        """Create a timestamped run directory.

        Returns:
            Path to the created run directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def extract_video_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_filename: str
    ) -> Optional[str]:
        """Extract video segment for an activity.

        Extracts the original video segment using ffmpeg instead of reconstructing
        from sampled frames, resulting in smooth playback at the original frame rate.

        Args:
            video_path: Path to the source video file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_filename: Name of the output file (without directory)

        Returns:
            Path to extracted clip or None on failure
        """
        output_path = os.path.join(self.clips_dir, output_filename)

        try:
            duration = end_time - start_time
            if duration <= 0:
                self.logger.warning(f"Invalid duration for clip extraction: {duration}")
                return None

            # Use ffmpeg to extract the segment directly from source video
            # -ss before -i for fast seeking, -t for duration
            # -c:v libx264 for H.264 encoding (browser compatible)
            # -movflags +faststart for web streaming optimization
            ffmpeg_path = os.environ.get('FFMPEG_PATH', 'ffmpeg')
            cmd = [
                ffmpeg_path, '-y',
                '-ss', str(start_time),
                '-i', video_path,
                '-t', str(duration),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-an',  # No audio needed for evidence clips
                '-movflags', '+faststart',
                output_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=120
            )

            if result.returncode == 0 and os.path.exists(output_path):
                self.logger.debug(
                    f"Extracted video segment: {start_time:.2f}s - {end_time:.2f}s -> {output_path}"
                )
                return output_path
            else:
                stderr_msg = result.stderr.decode()[:200] if result.stderr else "Unknown error"
                self.logger.warning(f"ffmpeg extraction failed: {stderr_msg}")
                return None

        except subprocess.TimeoutExpired:
            self.logger.warning(f"Video segment extraction timed out for: {output_path}")
            return None
        except FileNotFoundError:
            self.logger.warning("ffmpeg not found - cannot extract video segments")
            return None
        except Exception as e:
            self.logger.warning(f"Video segment extraction failed: {e}")
            return None

    def reencode_to_h264(self, input_path: str) -> bool:
        """Re-encode clip to H.264 for browser compatibility.

        OpenCV's mp4v codec (MPEG-4 Part 2) doesn't play in browsers.
        This re-encodes to H.264 which has universal browser support.

        Args:
            input_path: Path to the video file to re-encode

        Returns:
            True if re-encoding succeeded, False otherwise
        """
        temp_path = input_path + ".temp.mp4"
        try:
            result = subprocess.run([
                '/usr/bin/ffmpeg', '-y', '-i', input_path,
                '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '23', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-loglevel', 'error',
                temp_path
            ], capture_output=True, timeout=120)

            if result.returncode == 0 and os.path.exists(temp_path):
                os.replace(temp_path, input_path)
                self.logger.debug(f"Re-encoded to H.264: {input_path}")
                return True
            else:
                stderr = result.stderr.decode() if result.stderr else ""
                self.logger.warning(
                    f"H.264 re-encoding failed (code {result.returncode}): {stderr}"
                )
        except FileNotFoundError:
            self.logger.warning(
                "ffmpeg not found - videos will use mp4v codec (may not play in browsers)"
            )
        except subprocess.TimeoutExpired:
            self.logger.warning(f"H.264 re-encoding timed out for: {input_path}")
        except Exception as e:
            self.logger.warning(f"H.264 re-encoding failed: {e}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
        return False

    def save_video_clip(
        self,
        frames: List[np.ndarray],
        output_filename: str,
        fps: float
    ) -> Optional[str]:
        """Save frames as video clip at sample FPS for full-duration playback.

        Args:
            frames: List of frames to save
            output_filename: Name of the output file (without directory)
            fps: FPS to use for video (should be sample_fps for real-time duration)

        Returns:
            Path to saved clip or None if no frames provided
        """
        if len(frames) == 0:
            return None

        output_path = os.path.join(self.clips_dir, output_filename)
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Use the provided FPS (sample_fps) to create full-duration clips
        # Example: 13 frames @ 0.5 FPS = 26 seconds (real-time)
        # instead of: 13 frames @ 30 FPS = 0.43 seconds (fast-motion)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()

        # Re-encode to H.264 for browser compatibility
        # (mp4v codec from OpenCV doesn't play in browsers)
        self.reencode_to_h264(output_path)

        return output_path

    def save_activity_image(
        self,
        frame: np.ndarray,
        activity_name: str,
        frame_idx: int,
        output_filename: str = None,
        annotations: Dict = None
    ) -> str:
        """Save annotated frame as activity evidence image.

        Args:
            frame: The frame to save (numpy array)
            activity_name: Name of the activity
            frame_idx: Frame index for filename generation
            output_filename: Custom filename (optional, auto-generated if None)
            annotations: Optional annotations to draw on frame

        Returns:
            Path to saved image
        """
        if output_filename is None:
            output_filename = f"{activity_name}_frame{frame_idx:08d}_{self.evidence_counter:03d}_activity.jpg"

        image_path = os.path.join(self.clips_dir, output_filename)

        # Apply annotations if provided
        if annotations:
            frame = self._apply_annotations(frame.copy(), annotations)

        cv2.imwrite(image_path, frame)
        self.logger.debug(f"Saved activity image: {image_path}")

        return image_path

    def save_activity_image_from_video(
        self,
        video_path: str,
        frame_number: int,
        output_filename: str
    ) -> Optional[str]:
        """Extract and save a specific frame from video as activity image.

        Args:
            video_path: Path to the video file
            frame_number: Frame number to extract
            output_filename: Name for the output image file

        Returns:
            Path to saved image or None on failure
        """
        image_path = os.path.join(self.clips_dir, output_filename)

        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None:
                cv2.imwrite(image_path, frame)
                self.logger.debug(f"Saved activity image from video: {image_path}")
                return image_path
            else:
                self.logger.warning(
                    f"Failed to extract frame {frame_number} from {video_path}"
                )
                return None
        except Exception as e:
            self.logger.warning(f"Error extracting activity image: {e}")
            return None

    def save_annotated_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        sample_idx: int = None,
        quality: int = 95
    ) -> Optional[str]:
        """Save annotated frame to frames directory.

        Args:
            frame: The annotated frame to save
            frame_idx: Original frame index for filename
            sample_idx: Sample index for interval checking (optional)
            quality: JPEG quality (0-100)

        Returns:
            Path to saved frame or None if saving is disabled/skipped
        """
        if not self.save_annotated_frames or self.frames_dir is None:
            return None

        # Check frame save interval if sample_idx provided
        if sample_idx is not None and sample_idx % self.frame_save_interval != 0:
            return None

        try:
            frame_filename = f"frame_{frame_idx:08d}.jpg"
            frame_path = os.path.join(self.frames_dir, frame_filename)

            # Ensure directory exists (for multiprocessing safety)
            os.makedirs(self.frames_dir, exist_ok=True)

            # Save with specified quality
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

            return frame_path
        except Exception as e:
            self.logger.error(f"Error saving frame {frame_idx}: {e}")
            return None

    def _apply_annotations(self, frame: np.ndarray, annotations: Dict) -> np.ndarray:
        """Apply annotations to a frame.

        Args:
            frame: Frame to annotate
            annotations: Dictionary containing annotation data

        Returns:
            Annotated frame
        """
        # Draw bounding boxes if present
        if 'bboxes' in annotations:
            for bbox in annotations['bboxes']:
                x1, y1, x2, y2 = bbox.get('coords', [0, 0, 0, 0])
                color = bbox.get('color', (0, 255, 0))
                thickness = bbox.get('thickness', 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

                if 'label' in bbox:
                    cv2.putText(
                        frame, bbox['label'],
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

        # Draw text annotations if present
        if 'text' in annotations:
            for text_item in annotations['text']:
                position = text_item.get('position', (10, 30))
                text = text_item.get('text', '')
                color = text_item.get('color', (255, 255, 255))
                scale = text_item.get('scale', 0.7)
                thickness = text_item.get('thickness', 2)
                cv2.putText(
                    frame, text, position,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness
                )

        return frame

    def save_activities_json(
        self,
        activities: List[Dict],
        filename: str = "activities.json"
    ) -> str:
        """Save activities to JSON file.

        Routes through the canonical ``atomic_write_json`` helper so that
        every writer of ``activities.json`` shares the same encoder and
        the same crash-safe + cross-process-locked write protocol. Prior
        to Task 0002 this method had its own ad-hoc encoder (missing
        ``np.bool_``) and a non-atomic ``open(..., 'w')`` write that
        could leave a half-truncated file on a crash.

        For the default filename (``activities.json``) we go through
        :class:`ActivityRepository` to keep its logging behaviour. For
        other filenames we still want the same atomic+locked write but
        the repository is not the right fit, so we call the underlying
        helper directly.

        Args:
            activities: List of activity dictionaries
            filename: Output filename (default: activities.json)

        Returns:
            Path to saved JSON file
        """
        json_path = os.path.join(self.run_dir, filename)

        if filename == "activities.json":
            # Lazy import to break circular dep (see module-level NOTE).
            from app.repositories.activity_repository import ActivityRepository
            ActivityRepository().save_activities(activities, self.run_dir)
        else:
            atomic_write_json(json_path, activities, indent=2)

        self.logger.info(f"Activities JSON saved: {json_path}")
        return json_path

    def generate_summary_report(
        self,
        activities: List[Dict],
        save_json: bool = True
    ) -> Dict[str, int]:
        """Generate and log activity summary.

        Args:
            activities: List of activity dictionaries
            save_json: Whether to also save activities.json

        Returns:
            Dictionary mapping activity types to counts
        """
        if save_json:
            self.save_activities_json(activities)

        self.logger.info(f"Total activities detected: {len(activities)}")

        # Count activities by type
        activities_by_type: Dict[str, int] = {}
        for activity in activities:
            activity_type = activity.get('des', 'Unknown')
            if activity_type not in activities_by_type:
                activities_by_type[activity_type] = 0
            activities_by_type[activity_type] += 1

        # Log activity breakdown
        if activities_by_type:
            self.logger.info("Activity Breakdown:")
            for activity_type, count in activities_by_type.items():
                self.logger.info(f"  - {activity_type}: {count}")

        return activities_by_type

    def generate_evidence_filename(
        self,
        video_name: str,
        activity_name: str,
        start_frame: int,
        file_type: str = "clip"
    ) -> str:
        """Generate standardized evidence filename.

        Args:
            video_name: Name of the source video (with or without extension)
            activity_name: Name of the activity
            start_frame: Starting frame number
            file_type: Type of evidence ("clip" or "activity")

        Returns:
            Generated filename
        """
        # Remove extension if present
        video_name_without_ext = os.path.splitext(video_name)[0]

        if file_type == "clip":
            ext = ".mp4"
        else:
            ext = ".jpg"

        filename = (
            f"{video_name_without_ext}_{activity_name}_"
            f"frame{start_frame:08d}_{self.evidence_counter:03d}_{file_type}{ext}"
        )

        return filename

    def increment_evidence_counter(self) -> int:
        """Increment and return the evidence counter.

        Returns:
            The new evidence counter value
        """
        self.evidence_counter += 1
        return self.evidence_counter

    def get_clips_directory(self) -> str:
        """Get the clips directory path.

        Returns:
            Path to clips directory
        """
        return self.clips_dir

    def get_frames_directory(self) -> Optional[str]:
        """Get the frames directory path.

        Returns:
            Path to frames directory or None if frame saving is disabled
        """
        return self.frames_dir if self.save_annotated_frames else None

    def get_run_directory(self) -> str:
        """Get the run directory path.

        Returns:
            Path to run directory
        """
        return self.run_dir

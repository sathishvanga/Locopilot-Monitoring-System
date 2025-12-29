"""
Frame Timestamp Extraction Service - Extracts embedded timestamps from video frames using OCR

This service extracts the actual timestamp burned into video frames (e.g., from IP cameras)
instead of using video playback time. This provides accurate real-world timestamps for activities.

Timestamp format supported: MM-DD-YYYY Day HH:MM:SS (e.g., "11-22-2025 Sat 08:07:16")
Location: Top-left corner of the frame
"""

import re
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
from functools import lru_cache

from ..utils.logger import get_logger
from ..utils.config import get_settings

logger = get_logger(__name__)


class FrameTimestampService:
    """
    Service for extracting embedded timestamps from video frames using OCR.

    The timestamp is expected to be in the top-left corner of the frame
    in the format: MM-DD-YYYY Day HH:MM:SS (e.g., "11-22-2025 Sat 08:07:16")
    """

    def __init__(self):
        self.settings = get_settings()
        self._reader = None
        self._initialized = False
        self.enabled = self.settings.ocr_timestamp_enabled

        # ROI configuration from settings (percentages of frame dimensions)
        self.roi_x_start = self.settings.ocr_timestamp_roi_x_start
        self.roi_x_end = self.settings.ocr_timestamp_roi_x_end
        self.roi_y_start = self.settings.ocr_timestamp_roi_y_start
        self.roi_y_end = self.settings.ocr_timestamp_roi_y_end

        # Timestamp regex pattern: MM-DD-YYYY Day HH:MM:SS
        self.timestamp_pattern = re.compile(
            r'(\d{1,2})-(\d{1,2})-(\d{4})\s+([A-Za-z]{3})\s+(\d{1,2}):(\d{2}):(\d{2})'
        )

        # Cache for performance (avoid OCR on every frame)
        self._cache: Dict[int, str] = {}
        self._cache_max_size = 1000

        # Sampling: Only run OCR every N frames for performance
        self.ocr_sample_interval = self.settings.ocr_timestamp_sample_interval
        self._last_extracted_timestamp: Optional[str] = None
        self._frame_counter = 0

        logger.info(
            f"Frame timestamp service initialized - Enabled: {self.enabled}, "
            f"ROI: ({self.roi_x_start:.0%}-{self.roi_x_end:.0%}, {self.roi_y_start:.0%}-{self.roi_y_end:.0%}), "
            f"Sample interval: {self.ocr_sample_interval}"
        )

    def _initialize_reader(self):
        """Lazy initialization of EasyOCR reader"""
        if self._initialized:
            return

        try:
            import easyocr
            # Use GPU if available, English language only
            self._reader = easyocr.Reader(
                ['en'],
                gpu=True,  # Will fallback to CPU if no GPU
                verbose=False
            )
            self._initialized = True
            logger.info("EasyOCR reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self._initialized = False

    def _crop_timestamp_region(self, frame: np.ndarray) -> np.ndarray:
        """
        Crop the timestamp region from the frame (top-left corner).

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            Cropped region containing the timestamp
        """
        h, w = frame.shape[:2]

        x_start = int(w * self.roi_x_start)
        x_end = int(w * self.roi_x_end)
        y_start = int(h * self.roi_y_start)
        y_end = int(h * self.roi_y_end)

        return frame[y_start:y_end, x_start:x_end]

    def _preprocess_for_ocr(self, roi: np.ndarray) -> np.ndarray:
        """
        Preprocess the ROI for better OCR accuracy.

        For white text on variable background, minimal preprocessing works best.
        EasyOCR handles the image processing internally.

        Args:
            roi: Cropped timestamp region

        Returns:
            Preprocessed image optimized for OCR
        """
        # Minimal preprocessing - just resize for better accuracy
        # EasyOCR works well with the original color image
        h, w = roi.shape[:2]

        # Only resize if the ROI is very small
        if h < 50:
            scale = 2
            resized = cv2.resize(roi, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            return resized

        return roi

    def extract_timestamp(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        force: bool = False
    ) -> Optional[str]:
        """
        Extract the embedded timestamp from a video frame.

        Args:
            frame: Video frame (BGR format)
            frame_idx: Frame index for caching
            force: Force OCR extraction (bypass sampling interval)

        Returns:
            Extracted timestamp string in HH:MM:SS format, or None if extraction fails
        """
        # Check if OCR timestamp extraction is enabled
        if not self.enabled:
            return None

        # Check cache first
        cache_key = frame_idx
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Sampling: Only run OCR periodically for performance
        self._frame_counter += 1
        if not force and self._frame_counter % self.ocr_sample_interval != 0:
            # Return last known timestamp for intermediate frames
            return self._last_extracted_timestamp

        # Initialize reader if needed
        if not self._initialized:
            self._initialize_reader()

        if not self._reader:
            return None

        try:
            # Crop timestamp region
            roi = self._crop_timestamp_region(frame)

            # Preprocess for OCR
            processed = self._preprocess_for_ocr(roi)

            # Run OCR
            results = self._reader.readtext(processed, detail=0)

            if not results:
                return self._last_extracted_timestamp

            # Join all detected text
            text = ' '.join(results)

            # Parse timestamp from text
            timestamp = self._parse_timestamp(text)

            if timestamp:
                self._last_extracted_timestamp = timestamp

                # Cache result (with size limit)
                if len(self._cache) >= self._cache_max_size:
                    # Remove oldest entries
                    oldest_keys = sorted(self._cache.keys())[:100]
                    for k in oldest_keys:
                        del self._cache[k]

                self._cache[cache_key] = timestamp

            return timestamp or self._last_extracted_timestamp

        except Exception as e:
            logger.warning(f"Failed to extract timestamp from frame {frame_idx}: {e}")
            return self._last_extracted_timestamp

    def _parse_timestamp(self, text: str) -> Optional[str]:
        """
        Parse the timestamp from OCR text.

        Expected format: MM-DD-YYYY Day HH:MM:SS
        Example: "11-22-2025 Sat 08:07:16"

        Args:
            text: Raw OCR text

        Returns:
            Parsed timestamp in HH:MM:SS format, or None if parsing fails
        """
        # Try full timestamp pattern first
        match = self.timestamp_pattern.search(text)

        if match:
            month, day, year, weekday, hour, minute, second = match.groups()
            return f"{int(hour):02d}:{int(minute):02d}:{int(second):02d}"

        # Fallback: Try to extract just the time portion (HH:MM:SS)
        # OCR sometimes splits the text, so look for time pattern separately
        time_pattern = re.compile(r'(\d{1,2}):(\d{2}):(\d{2})')
        time_match = time_pattern.search(text)

        if time_match:
            hour, minute, second = time_match.groups()
            return f"{int(hour):02d}:{int(minute):02d}:{int(second):02d}"

        return None

    def extract_full_timestamp(
        self,
        frame: np.ndarray,
        frame_idx: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Extract full timestamp information including date.

        Args:
            frame: Video frame (BGR format)
            frame_idx: Frame index

        Returns:
            Dictionary with full timestamp info, or None if extraction fails
        """
        if not self._initialized:
            self._initialize_reader()

        if not self._reader:
            return None

        try:
            roi = self._crop_timestamp_region(frame)
            processed = self._preprocess_for_ocr(roi)
            results = self._reader.readtext(processed, detail=0)

            if not results:
                return None

            text = ' '.join(results)
            match = self.timestamp_pattern.search(text)

            if match:
                month, day, year, weekday, hour, minute, second = match.groups()

                return {
                    'date': f"{year}-{int(month):02d}-{int(day):02d}",
                    'time': f"{int(hour):02d}:{int(minute):02d}:{int(second):02d}",
                    'weekday': weekday,
                    'datetime': f"{year}-{int(month):02d}-{int(day):02d}T{int(hour):02d}:{int(minute):02d}:{int(second):02d}",
                    'raw_text': text
                }

            return None

        except Exception as e:
            logger.warning(f"Failed to extract full timestamp: {e}")
            return None

    def reset(self):
        """Reset the service state (call when processing a new video)"""
        self._cache.clear()
        self._last_extracted_timestamp = None
        self._frame_counter = 0
        logger.debug("Frame timestamp service state reset")


# Singleton instance
_frame_timestamp_service: Optional[FrameTimestampService] = None


def get_frame_timestamp_service() -> FrameTimestampService:
    """Get singleton instance of frame timestamp service."""
    global _frame_timestamp_service
    if _frame_timestamp_service is None:
        _frame_timestamp_service = FrameTimestampService()
    return _frame_timestamp_service

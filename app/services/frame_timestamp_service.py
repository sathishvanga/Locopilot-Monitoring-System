"""
Frame Timestamp Extraction Service - Extracts embedded timestamps from video frames using OCR

This service extracts the actual timestamp burned into video frames (e.g., from IP cameras)
instead of using video playback time. This provides accurate real-world timestamps for activities.

Timestamp format supported: DD-MM-YYYY Day HH:MM:SS (e.g., "11-11-2025 Tue 10:03:47")
Location: Top-left corner of the frame

OCR is called on-demand when activity is confirmed (start/end), not periodically.
"""

import re
import cv2
import numpy as np
from typing import Optional, Dict, Any, List

from ..utils.logger import get_logger
from ..utils.config import get_settings

logger = get_logger(__name__)


class FrameTimestampService:
    """
    Service for extracting embedded timestamps from video frames using OCR.

    The timestamp is expected to be in the top-left corner of the frame
    in the format: DD-MM-YYYY Day HH:MM:SS (e.g., "11-11-2025 Tue 10:03:47")

    OCR is performed on-demand when activity is confirmed, not periodically.
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

        # CLAHE settings
        self.clahe_clip_limit = getattr(self.settings, 'ocr_clahe_clip_limit', 3.0)
        self.clahe_tile_size = getattr(self.settings, 'ocr_clahe_tile_size', 8)

        # Debug mode
        self.debug = getattr(self.settings, 'ocr_timestamp_debug', False)

        # Timestamp regex patterns
        # Primary pattern: DD-MM-YYYY Day HH:MM:SS
        self.timestamp_pattern = re.compile(
            r'(\d{1,2})-(\d{1,2})-(\d{4})\s+([A-Za-z]{3})\s+(\d{1,2}):(\d{2}):(\d{2})'
        )
        # Time-only pattern: HH:MM:SS
        self.time_pattern = re.compile(r'(\d{1,2}):(\d{2}):(\d{2})')

        # Last extracted timestamp (for fallback)
        self._last_extracted_timestamp: Optional[str] = None

        logger.info(
            f"Frame timestamp service initialized - Enabled: {self.enabled}, "
            f"ROI: ({self.roi_x_start:.0%}-{self.roi_x_end:.0%}, {self.roi_y_start:.0%}-{self.roi_y_end:.0%}), "
            f"CLAHE: clip={self.clahe_clip_limit}, tile={self.clahe_tile_size}, Debug: {self.debug}"
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

        roi = frame[y_start:y_end, x_start:x_end]

        if self.debug:
            logger.debug(f"Cropped ROI: {roi.shape} from frame {frame.shape}")

        return roi

    def _preprocess_for_ocr(self, roi: np.ndarray) -> List[np.ndarray]:
        """
        Multi-method preprocessing for OCR that handles both dark and white text.

        Returns multiple preprocessed versions to try:
        1. Otsu thresholding (for dark text on light background)
        2. Otsu inverted (for white text on dark background)
        3. Adaptive thresholding (handles variable backgrounds better)
        4. Adaptive inverted

        Args:
            roi: Cropped timestamp region

        Returns:
            List of preprocessed images to try for OCR
        """
        h, w = roi.shape[:2]

        # Step 1: Upscale if ROI is small (target minimum 100px height)
        if h < 100:
            scale_factor = max(2, 100 // max(h, 1))
            roi = cv2.resize(
                roi,
                (w * scale_factor, h * scale_factor),
                interpolation=cv2.INTER_CUBIC
            )
            if self.debug:
                logger.debug(f"Upscaled ROI by {scale_factor}x to {roi.shape}")

        # Step 2: Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        # Step 3: Apply CLAHE for contrast enhancement (reduced clip limit for cleaner output)
        clahe = cv2.createCLAHE(
            clipLimit=min(self.clahe_clip_limit, 2.0),  # Cap at 2.0 for OCR
            tileGridSize=(self.clahe_tile_size, self.clahe_tile_size)
        )
        enhanced = clahe.apply(gray)

        # Step 4: Light Gaussian blur to reduce compression artifacts
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

        results = []

        # Method 1: Otsu thresholding (dark text on light background)
        _, binary_otsu = cv2.threshold(
            denoised, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        results.append(binary_otsu)

        # Method 2: Otsu inverted (white text on dark background)
        results.append(cv2.bitwise_not(binary_otsu))

        # Method 3: Adaptive thresholding (handles variable backgrounds)
        binary_adaptive = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        results.append(binary_adaptive)

        # Method 4: Adaptive inverted
        results.append(cv2.bitwise_not(binary_adaptive))

        if self.debug:
            logger.debug(f"Preprocessed ROI into {len(results)} variants, shape={binary_otsu.shape}")

        return results

    def extract_timestamp(self, frame: np.ndarray) -> Optional[str]:
        """
        Extract the embedded timestamp from a video frame (on-demand).

        This method tries multiple preprocessing methods to handle both
        dark text on light backgrounds and white text on dark backgrounds.

        Args:
            frame: Video frame (BGR format)

        Returns:
            Extracted timestamp string in HH:MM:SS format, or None if extraction fails
        """
        # Check if OCR timestamp extraction is enabled
        if not self.enabled:
            return None

        # Initialize reader if needed
        if not self._initialized:
            self._initialize_reader()

        if not self._reader:
            logger.warning("EasyOCR reader not available")
            return None

        try:
            # Crop timestamp region (top-left corner)
            roi = self._crop_timestamp_region(frame)
            print(f"[OCR-DIAG] ROI shape: {roi.shape}, size: {roi.size}", flush=True)

            if roi.size == 0:
                logger.warning("Empty ROI cropped from frame")
                return self._last_extracted_timestamp

            # Get multiple preprocessed versions
            preprocessed_versions = self._preprocess_for_ocr(roi)
            method_names = ["Otsu", "Otsu-inverted", "Adaptive", "Adaptive-inverted"]
            print(f"[OCR-DIAG] Got {len(preprocessed_versions)} preprocessed versions", flush=True)

            # Try each preprocessing method until we get a valid timestamp
            all_detected_text = []
            all_raw_texts = []
            for i, processed in enumerate(preprocessed_versions):
                results = self._reader.readtext(processed, detail=0)
                print(f"[OCR-DIAG] Method {i+1} ({method_names[i]}): results={results}", flush=True)

                if results:
                    raw_text = ' '.join(results)
                    all_detected_text.append(f"{method_names[i]}: {raw_text}")
                    all_raw_texts.append(raw_text)
                    timestamp = self._parse_timestamp(raw_text)

                    if timestamp:
                        self._last_extracted_timestamp = timestamp
                        logger.info(f"[OCR] Extracted '{timestamp}' using method {i+1} ({method_names[i]}), text: '{raw_text}'")
                        print(f"[OCR-DIAG] SUCCESS! timestamp={timestamp}", flush=True)
                        return timestamp

                    if self.debug:
                        logger.debug(f"[OCR] Method {i+1} ({method_names[i]}) detected text but no valid timestamp: '{raw_text}'")

            # Try combining all detected text as a last resort
            if all_raw_texts:
                combined_text = ' '.join(all_raw_texts)
                print(f"[OCR-DIAG] Trying combined text: {combined_text}", flush=True)
                timestamp = self._parse_timestamp(combined_text)
                if timestamp:
                    self._last_extracted_timestamp = timestamp
                    logger.info(f"[OCR] Extracted '{timestamp}' from combined text")
                    print(f"[OCR-DIAG] SUCCESS from combined! timestamp={timestamp}", flush=True)
                    return timestamp

            # None of the methods produced a valid timestamp
            print(f"[OCR-DIAG] All methods failed. Detected text: {all_detected_text}", flush=True)
            logger.info(f"[OCR] All {len(preprocessed_versions)} methods failed, using cached: {self._last_extracted_timestamp}")
            return self._last_extracted_timestamp

        except Exception as e:
            logger.warning(f"Failed to extract timestamp: {e}")
            return self._last_extracted_timestamp

    def _parse_timestamp(self, text: str) -> Optional[str]:
        """
        Parse the timestamp from OCR text with enhanced error tolerance.

        Expected format: DD-MM-YYYY Day HH:MM:SS
        Example: "11-11-2025 Tue 10:03:47"

        Handles common OCR errors:
        - 'O' misread as '0' and vice versa
        - 'l' misread as '1'
        - 'S' misread as '5'
        - Missing colons or spaces

        Args:
            text: Raw OCR text

        Returns:
            Parsed timestamp in HH:MM:SS format, or None if parsing fails
        """
        if not text:
            return None

        # Try full timestamp pattern first (most reliable)
        match = self.timestamp_pattern.search(text)
        if match:
            try:
                month, day, year, weekday, hour, minute, second = match.groups()
                h, m, s = int(hour), int(minute), int(second)
                if 0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59:
                    return f"{h:02d}:{m:02d}:{s:02d}"
            except ValueError:
                pass

        # Fallback 1: Try time pattern only (HH:MM:SS)
        time_match = self.time_pattern.search(text)
        if time_match:
            try:
                hour, minute, second = time_match.groups()
                h, m, s = int(hour), int(minute), int(second)
                if 0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59:
                    return f"{h:02d}:{m:02d}:{s:02d}"
            except ValueError:
                pass

        # Fallback 2: Clean OCR errors and try again
        cleaned = self._clean_ocr_text(text)
        if cleaned != text:
            # Try patterns again on cleaned text
            match = self.timestamp_pattern.search(cleaned)
            if match:
                try:
                    month, day, year, weekday, hour, minute, second = match.groups()
                    h, m, s = int(hour), int(minute), int(second)
                    if 0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59:
                        return f"{h:02d}:{m:02d}:{s:02d}"
                except ValueError:
                    pass

            time_match = self.time_pattern.search(cleaned)
            if time_match:
                try:
                    hour, minute, second = time_match.groups()
                    h, m, s = int(hour), int(minute), int(second)
                    if 0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59:
                        return f"{h:02d}:{m:02d}:{s:02d}"
                except ValueError:
                    pass

        # Fallback 3: Extract any 6 consecutive digits as HHMMSS
        digits_only = re.sub(r'\D', '', text)
        if len(digits_only) >= 6:
            # Look for valid time in digit sequences
            for i in range(len(digits_only) - 5):
                potential_time = digits_only[i:i+6]
                try:
                    h = int(potential_time[0:2])
                    m = int(potential_time[2:4])
                    s = int(potential_time[4:6])
                    if 0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59:
                        return f"{h:02d}:{m:02d}:{s:02d}"
                except ValueError:
                    continue

        return None

    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean common OCR misreadings from text.

        Args:
            text: Raw OCR text

        Returns:
            Cleaned text with common OCR errors corrected
        """
        cleaned = text

        # Common OCR substitutions for digits
        # Be careful: only apply in contexts that look like timestamps
        replacements = [
            ('O', '0'),  # Letter O -> digit 0
            ('o', '0'),
            ('l', '1'),  # lowercase L -> digit 1
            ('I', '1'),  # uppercase I -> digit 1
            ('|', '1'),  # pipe -> digit 1
            ('S', '5'),  # S -> 5 (in time context)
            ('s', '5'),
            ('B', '8'),  # B -> 8
            ('G', '6'),  # G -> 6
            ('Z', '2'),  # Z -> 2
            ('T', '7'),  # T -> 7 (sometimes)
        ]

        # Apply replacements only to segments that look like they could be numbers
        for old, new in replacements:
            cleaned = cleaned.replace(old, new)

        return cleaned

    def extract_full_timestamp(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Extract full timestamp information including date.

        Args:
            frame: Video frame (BGR format)

        Returns:
            Dictionary with full timestamp info, or None if extraction fails
        """
        if not self._initialized:
            self._initialize_reader()

        if not self._reader:
            return None

        try:
            roi = self._crop_timestamp_region(frame)
            preprocessed_versions = self._preprocess_for_ocr(roi)

            # Try each preprocessing method
            for processed in preprocessed_versions:
                results = self._reader.readtext(processed, detail=0)

                if not results:
                    continue

                text = ' '.join(results)

                if self.debug:
                    logger.debug(f"Full timestamp OCR text: '{text}'")

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
        self._last_extracted_timestamp = None
        logger.debug("Frame timestamp service state reset")


# Singleton instance
_frame_timestamp_service: Optional[FrameTimestampService] = None


def get_frame_timestamp_service() -> FrameTimestampService:
    """Get singleton instance of frame timestamp service."""
    global _frame_timestamp_service
    if _frame_timestamp_service is None:
        _frame_timestamp_service = FrameTimestampService()
    return _frame_timestamp_service

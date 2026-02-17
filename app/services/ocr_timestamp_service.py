"""
OCR timestamp service - Extracts timestamps from video frame overlays

This service uses EasyOCR (primary) or Tesseract (fallback) to extract embedded
timestamps from video frames, typically displayed as overlay text in railway
surveillance videos.

EasyOCR is preferred for semi-transparent overlays common in IP cameras.
"""

import logging
import re
import threading
import time
import cv2
import numpy as np
from typing import Optional, Tuple, List
from threading import Lock

from ..models.trip_models import OCRTimestampResult
from ..utils.config import get_settings

logger = logging.getLogger(__name__)

# Check available OCR engines
EASYOCR_AVAILABLE = False
TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    logger.info("easyocr not installed. Install with: pip install easyocr")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    logger.info("pytesseract not installed. Install with: pip install pytesseract")


class OCRTimestampService:
    """
    Service for extracting timestamps from video frame overlays using OCR

    Features:
    - EasyOCR as primary engine (better for semi-transparent text)
    - Tesseract as fallback (faster for clean text)
    - Lazy initialization of OCR readers
    - Time-based result caching to reduce OCR overhead
    - Full-frame scanning with automatic timestamp detection
    - Thread-safe singleton reader instance
    """

    # Regex patterns for timestamp extraction
    TIME_PATTERN = re.compile(r'(\d{1,2}):(\d{2}):(\d{2})')
    DATE_TIME_PATTERN = re.compile(
        r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})\s+\w+\s+(\d{1,2}):(\d{2}):(\d{2})'
    )

    # Singleton EasyOCR reader (expensive to initialize)
    _easyocr_reader = None
    _reader_lock = Lock()

    def __init__(self):
        """Initialize the OCR timestamp service"""
        self.settings = get_settings()

        # Determine which engine to use
        self.ocr_engine = self.settings.ocr_engine  # 'easyocr', 'tesseract', or 'auto'
        self._select_engine()

        # Cache settings
        self._last_timestamp: Optional[str] = None
        self._last_extraction_time: float = 0
        self._cache_duration_sec: float = 0.5  # Cache result for 0.5 seconds

        # Performance tracking
        self._extraction_count = 0
        self._cache_hit_count = 0

        logger.info(
            f"OCRTimestampService initialized - "
            f"engine: {self.ocr_engine}, "
            f"enabled: {self.enabled}"
        )

    def _select_engine(self):
        """Select the best available OCR engine"""
        if self.ocr_engine == 'auto':
            if EASYOCR_AVAILABLE:
                self.ocr_engine = 'easyocr'
            elif TESSERACT_AVAILABLE:
                self.ocr_engine = 'tesseract'
            else:
                self.ocr_engine = 'none'

        if self.ocr_engine == 'easyocr' and not EASYOCR_AVAILABLE:
            logger.warning("EasyOCR requested but not available, falling back to Tesseract")
            self.ocr_engine = 'tesseract' if TESSERACT_AVAILABLE else 'none'

        if self.ocr_engine == 'tesseract' and not TESSERACT_AVAILABLE:
            logger.warning("Tesseract requested but not available")
            self.ocr_engine = 'easyocr' if EASYOCR_AVAILABLE else 'none'

        self.enabled = self.settings.ocr_enabled and self.ocr_engine != 'none'

    @classmethod
    def _get_easyocr_reader(cls):
        """Get or create the singleton EasyOCR reader (thread-safe, lazy init)"""
        if cls._easyocr_reader is None:
            with cls._reader_lock:
                if cls._easyocr_reader is None:
                    logger.info("Initializing EasyOCR reader (this may take a moment)...")
                    cls._easyocr_reader = easyocr.Reader(
                        ['en'],
                        gpu=False,  # CPU mode for compatibility
                        verbose=False
                    )
                    logger.info("EasyOCR reader initialized")
        return cls._easyocr_reader

    def extract_timestamp(self, frame: np.ndarray) -> OCRTimestampResult:
        """
        Extract timestamp from video frame

        Args:
            frame: Video frame as numpy array (BGR format)

        Returns:
            OCRTimestampResult with extraction details
        """
        start_time = time.time()

        if not self.enabled:
            logger.debug(f"[OCR] ❌ OCR disabled (engine: {self.ocr_engine})")
            return OCRTimestampResult(
                success=False,
                error_message=f"OCR disabled (engine: {self.ocr_engine})"
            )

        # Check time-based cache
        if self._should_use_cache():
            self._cache_hit_count += 1
            logger.debug(
                f"[OCR] Cache HIT - returning cached timestamp: {self._last_timestamp} "
                f"(cache hits: {self._cache_hit_count})"
            )
            return OCRTimestampResult(
                success=True,
                timestamp=self._last_timestamp,
                confidence=1.0,
                extraction_time_ms=0,
                roi_position="cached"
            )

        self._extraction_count += 1
        frame_shape = frame.shape if frame is not None else "None"
        logger.debug(
            f"[OCR] Extracting timestamp - "
            f"engine: {self.ocr_engine}, frame_shape: {frame_shape}, "
            f"extraction #{self._extraction_count}"
        )

        try:
            if self.ocr_engine == 'easyocr':
                result = self._extract_with_easyocr(frame, start_time)
            else:
                result = self._extract_with_tesseract(frame, start_time)

            # Update cache on success
            if result.success:
                self._last_timestamp = result.timestamp
                self._last_extraction_time = time.time()
                logger.info(
                    f"[OCR] ✅ Extracted timestamp: {result.timestamp} "
                    f"(confidence: {result.confidence:.2f}, time: {result.extraction_time_ms:.1f}ms)"
                )
            else:
                logger.debug(
                    f"[OCR] ❌ Failed to extract timestamp - "
                    f"raw_text: '{result.raw_text}', error: {result.error_message}"
                )

            return result

        except Exception as e:
            extraction_time_ms = (time.time() - start_time) * 1000
            logger.error(f"[OCR] ❌ Exception during extraction: {e}", exc_info=True)
            return OCRTimestampResult(
                success=False,
                error_message=str(e),
                extraction_time_ms=extraction_time_ms
            )

    def _extract_with_easyocr(
        self,
        frame: np.ndarray,
        start_time: float
    ) -> OCRTimestampResult:
        """
        Extract timestamp using EasyOCR

        EasyOCR scans the full frame and finds text automatically,
        making it ideal for varying timestamp positions.
        """
        reader = self._get_easyocr_reader()

        # EasyOCR works best on the full frame - it auto-detects text regions
        # For efficiency, we can crop to likely timestamp areas
        h, w = frame.shape[:2]

        # Scan top portion where timestamps usually appear (top 15% of frame)
        top_region = frame[0:int(h * 0.15), :]
        results = reader.readtext(top_region)

        timestamp = None
        confidence = 0.0
        raw_text_parts = []

        for (bbox, text, conf) in results:
            raw_text_parts.append(text)

            # Look for time pattern in detected text
            time_match = self.TIME_PATTERN.search(text)
            if time_match and conf > confidence:
                hours = int(time_match.group(1))
                minutes = int(time_match.group(2))
                seconds = int(time_match.group(3))

                if 0 <= hours <= 23 and 0 <= minutes <= 59 and 0 <= seconds <= 59:
                    timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    confidence = conf

        extraction_time_ms = (time.time() - start_time) * 1000
        raw_text = " | ".join(raw_text_parts) if raw_text_parts else ""

        if timestamp:
            return OCRTimestampResult(
                success=True,
                timestamp=timestamp,
                raw_text=raw_text,
                confidence=confidence,
                roi_position="easyocr_auto",
                extraction_time_ms=extraction_time_ms
            )
        else:
            # Try bottom region as fallback (some cameras put timestamp at bottom)
            bottom_region = frame[int(h * 0.85):h, :]
            results = reader.readtext(bottom_region)

            for (bbox, text, conf) in results:
                time_match = self.TIME_PATTERN.search(text)
                if time_match and conf > 0.3:
                    hours = int(time_match.group(1))
                    minutes = int(time_match.group(2))
                    seconds = int(time_match.group(3))

                    if 0 <= hours <= 23 and 0 <= minutes <= 59 and 0 <= seconds <= 59:
                        timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        extraction_time_ms = (time.time() - start_time) * 1000
                        return OCRTimestampResult(
                            success=True,
                            timestamp=timestamp,
                            raw_text=text,
                            confidence=conf,
                            roi_position="easyocr_bottom",
                            extraction_time_ms=extraction_time_ms
                        )

            return OCRTimestampResult(
                success=False,
                raw_text=raw_text,
                confidence=0.0,
                roi_position="easyocr_auto",
                extraction_time_ms=extraction_time_ms,
                error_message="No valid timestamp found in frame"
            )

    def _extract_with_tesseract(
        self,
        frame: np.ndarray,
        start_time: float
    ) -> OCRTimestampResult:
        """
        Extract timestamp using Tesseract OCR

        Uses configurable ROI for faster processing.
        """
        h, w = frame.shape[:2]

        # Use configured ROI settings
        roi_x = self.settings.ocr_roi_x
        roi_y = self.settings.ocr_roi_y
        roi_w = self.settings.ocr_roi_width
        roi_h = self.settings.ocr_roi_height

        # Determine ROI based on position setting
        position = self.settings.ocr_roi_position
        if position == "top-left":
            x, y = roi_x, roi_y
        elif position == "top-right":
            x, y = w - roi_w - roi_x, roi_y
        elif position == "bottom-left":
            x, y = roi_x, h - roi_h - roi_y
        elif position == "bottom-right":
            x, y = w - roi_w - roi_x, h - roi_h - roi_y
        else:
            x, y = roi_x, roi_y

        # Validate bounds
        x = max(0, min(x, w - roi_w))
        y = max(0, min(y, h - roi_h))

        roi = frame[y:y+roi_h, x:x+roi_w]
        roi_coords = (x, y, roi_w, roi_h)

        # Preprocess for Tesseract
        processed = self._preprocess_for_tesseract(roi)

        # Run Tesseract
        config = '--oem 3 --psm 7'
        raw_text = pytesseract.image_to_string(processed, config=config).strip()

        # Parse timestamp
        timestamp = self._parse_timestamp(raw_text)

        extraction_time_ms = (time.time() - start_time) * 1000

        if timestamp:
            return OCRTimestampResult(
                success=True,
                timestamp=timestamp,
                raw_text=raw_text,
                confidence=0.8,
                roi_position=position,
                roi_coords=roi_coords,
                extraction_time_ms=extraction_time_ms
            )
        else:
            return OCRTimestampResult(
                success=False,
                raw_text=raw_text,
                confidence=0.0,
                roi_position=position,
                roi_coords=roi_coords,
                extraction_time_ms=extraction_time_ms,
                error_message=f"Could not parse timestamp from: '{raw_text}'"
            )

    def _preprocess_for_tesseract(self, roi: np.ndarray) -> np.ndarray:
        """Preprocess ROI for Tesseract OCR"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Scale up for better recognition
        scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(scaled)

        # Otsu threshold
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

    def _parse_timestamp(self, raw_text: str) -> Optional[str]:
        """Parse timestamp from OCR text"""
        if not raw_text:
            return None

        # Clean common OCR errors
        cleaned = raw_text.replace('O', '0').replace('o', '0')
        cleaned = cleaned.replace('l', '1').replace('I', '1')
        cleaned = cleaned.replace('.', ':')

        # Try to match time pattern
        match = self.TIME_PATTERN.search(cleaned)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            seconds = int(match.group(3))

            if 0 <= hours <= 23 and 0 <= minutes <= 59 and 0 <= seconds <= 59:
                return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        return None

    def _should_use_cache(self) -> bool:
        """Check if cached result should be used (time-based)"""
        if self._last_timestamp is None:
            return False

        elapsed = time.time() - self._last_extraction_time
        return elapsed < self._cache_duration_sec

    def clear_cache(self):
        """Clear the timestamp cache"""
        self._last_timestamp = None
        self._last_extraction_time = 0

    def get_stats(self) -> dict:
        """Get performance statistics"""
        total = self._extraction_count + self._cache_hit_count
        cache_rate = (self._cache_hit_count / total * 100) if total > 0 else 0

        return {
            "engine": self.ocr_engine,
            "enabled": self.enabled,
            "total_calls": total,
            "extractions": self._extraction_count,
            "cache_hits": self._cache_hit_count,
            "cache_hit_rate": f"{cache_rate:.1f}%"
        }


# Global service instance
_ocr_service: Optional[OCRTimestampService] = None
_ocr_service_lock = threading.Lock()


def get_ocr_timestamp_service() -> OCRTimestampService:
    """
    Get the global OCR timestamp service instance.

    M-25: Thread-safe double-checked locking pattern.

    Returns:
        OCRTimestampService instance
    """
    global _ocr_service
    if _ocr_service is None:
        with _ocr_service_lock:
            if _ocr_service is None:
                _ocr_service = OCRTimestampService()
    return _ocr_service

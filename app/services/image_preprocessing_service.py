"""
Image preprocessing service for enhancing MediaPipe detection accuracy

This service applies various image enhancement techniques to improve
MediaPipe Pose detection, especially for hand/wrist landmarks in challenging
lighting conditions (low light, poor contrast).
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ImagePreprocessingService:
    """
    Service for preprocessing images before MediaPipe processing
    
    Applies techniques proven to improve MediaPipe detection accuracy:
    - CLAHE (Contrast-Limited Adaptive Histogram Equalization)
    - Gamma correction for brightness adjustment
    - Unsharp masking for edge enhancement
    - Light noise reduction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessing service
        
        Args:
            config: Configuration dictionary with preprocessing settings
                   If None, uses default settings
        """
        self.config = config or {}
        
        # Default configuration
        self.enable_preprocessing = self.config.get('enable_image_preprocessing', True)
        self.use_clahe = self.config.get('use_clahe', True)
        self.use_gamma_correction = self.config.get('use_gamma_correction', True)
        self.use_unsharp_masking = self.config.get('use_unsharp_masking', False)
        self.use_noise_reduction = self.config.get('use_noise_reduction', True)
        self.adaptive_preprocessing = self.config.get('adaptive_preprocessing', True)
        
        # Technique parameters
        self.clahe_clip_limit = self.config.get('clahe_clip_limit', 2.0)
        self.clahe_tile_grid_size = tuple(self.config.get('clahe_tile_grid_size', [8, 8]))
        self.gamma_value = self.config.get('gamma_value', 1.2)
        self.unsharp_strength = self.config.get('unsharp_strength', 1.5)
        self.unsharp_radius = self.config.get('unsharp_radius', 1)
        self.noise_reduction_kernel = self.config.get('noise_reduction_kernel', 3)
        
        # Initialize CLAHE (reusable instance)
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_grid_size
            )
        else:
            self.clahe = None

        # Cache for gamma lookup tables keyed by gamma value, so the LUT is
        # not recomputed on every frame for the same gamma.
        self._gamma_lut_cache: Dict[float, np.ndarray] = {}
        
        logger.info(
            f"ImagePreprocessingService initialized - "
            f"Enabled: {self.enable_preprocessing}, "
            f"CLAHE: {self.use_clahe}, "
            f"Gamma: {self.use_gamma_correction}, "
            f"Unsharp: {self.use_unsharp_masking}, "
            f"Noise Reduction: {self.use_noise_reduction}"
        )
    
    def preprocess_for_mediapipe(self, frame: np.ndarray) -> np.ndarray:
        """
        Main preprocessing method that applies selected techniques
        
        Pipeline order:
        1. Light noise reduction (if needed)
        2. CLAHE (on LAB L channel)
        3. Gamma correction
        4. Unsharp masking (optional)
        
        Args:
            frame: Input RGB image (numpy array, uint8)
            
        Returns:
            Enhanced RGB image ready for MediaPipe
        """
        if not self.enable_preprocessing:
            return frame
        
        if frame is None or frame.size == 0:
            logger.warning("Empty frame provided to preprocessing")
            return frame
        
        # Validate input
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            logger.warning(f"Invalid frame shape: {frame.shape}, expected (H, W, 3)")
            return frame
        
        try:
            processed_frame = frame.copy()
            
            # Adaptive preprocessing: analyze image quality first
            quality_metrics = None
            if self.adaptive_preprocessing:
                quality_metrics = self.detect_image_quality(processed_frame)
                
                # NEVER skip preprocessing for very dark images (brightness < 0.2)
                # These need aggressive enhancement regardless of other metrics
                is_very_dark = quality_metrics['brightness'] < 0.2
                
                # Skip preprocessing only if image quality is good AND not very dark
                if quality_metrics['is_good_quality'] and not is_very_dark:
                    logger.debug("Image quality is good, skipping preprocessing")
                    return processed_frame
                
                # Adjust techniques based on quality metrics
                needs_noise_reduction = quality_metrics['noise_level'] > 0.3
                needs_brightness_boost = quality_metrics['brightness'] < 0.4  # More aggressive threshold
                
                # For very dark images, use more aggressive preprocessing
                if is_very_dark:
                    needs_brightness_boost = True
                    needs_noise_reduction = True  # Dark images often have more visible noise
            else:
                needs_noise_reduction = self.use_noise_reduction
                needs_brightness_boost = True
            
            # Step 1: Light noise reduction (if enabled and needed)
            if needs_noise_reduction and self.use_noise_reduction:
                processed_frame = self.apply_light_noise_reduction(
                    processed_frame, 
                    kernel_size=self.noise_reduction_kernel
                )
            
            # Step 2: CLAHE (most critical for low-light/low-contrast)
            if self.use_clahe:
                # Use more aggressive CLAHE for very dark images
                if self.adaptive_preprocessing and needs_brightness_boost:
                    # Increase clip limit for darker images (more contrast enhancement)
                    clahe_clip_limit = min(2.5, self.clahe_clip_limit * 1.3)  # REDUCED max from 4.0 to 2.5
                    processed_frame = self.apply_clahe(processed_frame, clip_limit=clahe_clip_limit)
                else:
                    processed_frame = self.apply_clahe(processed_frame)
            
            # Step 3: Gamma correction (brightness adjustment)
            if self.use_gamma_correction:
                # Adjust gamma based on brightness if adaptive
                if self.adaptive_preprocessing and needs_brightness_boost and quality_metrics:
                    # Use higher gamma for darker images (more aggressive brightening)
                    # For very dark images (brightness < 0.2), use even higher gamma
                    if quality_metrics['brightness'] < 0.2:
                        gamma = min(2.0, self.gamma_value * 1.5)  # Very aggressive for very dark
                    else:
                        gamma = min(1.8, self.gamma_value * 1.3)  # Moderate for dark
                else:
                    gamma = self.gamma_value
                processed_frame = self.apply_gamma_correction(processed_frame, gamma=gamma)
            
            # Step 4: Unsharp masking (optional, can add artifacts)
            if self.use_unsharp_masking:
                processed_frame = self.apply_unsharp_masking(
                    processed_frame,
                    strength=self.unsharp_strength,
                    radius=self.unsharp_radius
                )
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}", exc_info=True)
            # Return original frame on error
            return frame
    
    def apply_clahe(self, frame: np.ndarray, clip_limit: Optional[float] = None, 
                    tile_grid_size: Optional[tuple] = None) -> np.ndarray:
        """
        Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE)
        
        Operates on LAB color space (L channel only) to preserve color.
        This is the most effective technique for low-light/low-contrast scenarios.
        
        Args:
            frame: Input RGB image
            clip_limit: CLAHE clip limit (default: from config)
            tile_grid_size: CLAHE tile grid size (default: from config)
            
        Returns:
            Enhanced RGB image
        """
        if clip_limit is None:
            clip_limit = self.clahe_clip_limit
        if tile_grid_size is None:
            tile_grid_size = self.clahe_tile_grid_size
        
        try:
            # Convert RGB to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Apply CLAHE to L channel only
            if self.clahe is not None and self.clahe_clip_limit == clip_limit:
                # Reuse existing CLAHE instance if parameters match
                l_channel_enhanced = self.clahe.apply(l_channel)
            else:
                # Create new CLAHE instance with custom parameters
                clahe = cv2.createCLAHE(
                    clipLimit=clip_limit,
                    tileGridSize=tile_grid_size
                )
                l_channel_enhanced = clahe.apply(l_channel)
            
            # Merge channels back
            lab_enhanced = cv2.merge([l_channel_enhanced, a_channel, b_channel])
            
            # Convert back to RGB
            enhanced_frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            
            return enhanced_frame
            
        except Exception as e:
            logger.error(f"Error applying CLAHE: {e}", exc_info=True)
            return frame
    
    def apply_gamma_correction(self, frame: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
        """
        Apply gamma correction for brightness adjustment
        
        Helps in both overexposed and underexposed images.
        Gamma < 1.0 brightens, gamma > 1.0 darkens.
        
        Args:
            frame: Input RGB image
            gamma: Gamma value (default: from config, typically 1.2)
            
        Returns:
            Brightness-adjusted RGB image
        """
        if gamma is None:
            gamma = self.gamma_value
        
        try:
            # Use cached lookup table for this gamma value to avoid
            # recomputing the LUT on every frame.
            gamma_key = round(gamma, 4)
            if gamma_key not in self._gamma_lut_cache:
                inv_gamma = 1.0 / gamma
                table = np.array([
                    ((i / 255.0) ** inv_gamma) * 255
                    for i in np.arange(0, 256)
                ]).astype("uint8")
                self._gamma_lut_cache[gamma_key] = table
            else:
                table = self._gamma_lut_cache[gamma_key]

            # Apply lookup table
            corrected_frame = cv2.LUT(frame, table)
            
            return corrected_frame
            
        except Exception as e:
            logger.error(f"Error applying gamma correction: {e}", exc_info=True)
            return frame
    
    def apply_unsharp_masking(self, frame: np.ndarray, strength: Optional[float] = None,
                             radius: Optional[int] = None) -> np.ndarray:
        """
        Apply unsharp masking for edge enhancement
        
        Makes hand contours more visible by enhancing edges.
        Uses Gaussian blur + high-pass filter.
        
        Args:
            frame: Input RGB image
            strength: Unsharp masking strength (default: from config)
            radius: Gaussian blur radius (default: from config)
            
        Returns:
            Edge-enhanced RGB image
        """
        if strength is None:
            strength = self.unsharp_strength
        if radius is None:
            radius = self.unsharp_radius
        
        try:
            # Create blurred version
            blurred = cv2.GaussianBlur(frame, (0, 0), radius)
            
            # High-pass filter: original - blurred
            sharpened = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)
            
            # Clip values to valid range
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            return sharpened
            
        except Exception as e:
            logger.error(f"Error applying unsharp masking: {e}", exc_info=True)
            return frame
    
    def apply_light_noise_reduction(self, frame: np.ndarray, kernel_size: Optional[int] = None) -> np.ndarray:
        """
        Apply light Gaussian blur to reduce noise without losing detail
        
        Only applied if image quality metrics indicate high noise.
        Uses small kernel to preserve important features.
        
        Args:
            frame: Input RGB image
            kernel_size: Gaussian blur kernel size (must be odd, default: from config)
            
        Returns:
            Noise-reduced RGB image
        """
        if kernel_size is None:
            kernel_size = self.noise_reduction_kernel
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        try:
            # Light Gaussian blur
            denoised = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
            
            return denoised
            
        except Exception as e:
            logger.error(f"Error applying noise reduction: {e}", exc_info=True)
            return frame
    
    def detect_image_quality(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image quality metrics to guide preprocessing selection
        
        Calculates:
        - Brightness: Average pixel intensity (0-1)
        - Contrast: Standard deviation of pixel intensities
        - Noise level: Estimated from high-frequency content
        
        Args:
            frame: Input RGB image
            
        Returns:
            Dictionary with quality metrics:
            {
                'brightness': float,  # 0-1, higher = brighter
                'contrast': float,    # Higher = more contrast
                'noise_level': float, # 0-1, higher = more noise
                'is_good_quality': bool  # True if quality is already good
            }
        """
        try:
            # Convert to grayscale for analysis
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame.copy()
            
            # Calculate brightness (mean intensity)
            brightness = np.mean(gray) / 255.0
            
            # Calculate contrast (standard deviation)
            contrast = np.std(gray) / 255.0
            
            # Estimate noise level using Laplacian variance
            # Higher variance = more edges/details, lower = smoother/noisier
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_level = 1.0 - min(1.0, np.var(laplacian) / 10000.0)
            
            # Determine if quality is already good
            # Good quality: reasonable brightness (0.2-0.8), decent contrast (>0.1), low noise (<0.3)
            is_good_quality = (
                0.2 <= brightness <= 0.8 and
                contrast > 0.1 and
                noise_level < 0.3
            )
            
            return {
                'brightness': float(brightness),
                'contrast': float(contrast),
                'noise_level': float(noise_level),
                'is_good_quality': bool(is_good_quality)
            }
            
        except Exception as e:
            logger.error(f"Error detecting image quality: {e}", exc_info=True)
            # Return default metrics (assume poor quality)
            return {
                'brightness': 0.5,
                'contrast': 0.1,
                'noise_level': 0.5,
                'is_good_quality': False
            }


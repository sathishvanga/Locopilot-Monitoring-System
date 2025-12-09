"""
Inference Backend Abstraction Layer

This module provides a unified interface for YOLO inference with multiple backends:
- PyTorch: Standard Ultralytics YOLO with PyTorch backend
- OpenVINO: Optimized CPU inference with 2-3x speedup

The module automatically handles fallback to PyTorch if OpenVINO is unavailable.
"""

from ultralytics import YOLO
import os
from typing import Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)


def create_inference_backend(model_path: str, backend: str = 'auto') -> YOLO:
    """
    Factory function to create inference backend with automatic fallback

    This function creates a YOLO model instance using the specified backend.
    Ultralytics YOLO automatically detects the model format and uses the
    appropriate backend (PyTorch for .pt files, OpenVINO for .xml files).

    Args:
        model_path: Path to model file (.pt for PyTorch, .xml for OpenVINO)
        backend: Backend type - 'auto' (detect from path), 'pytorch', or 'openvino'

    Returns:
        YOLO model instance configured for the selected backend

    Examples:
        >>> # Auto-detect backend from file extension
        >>> model = create_inference_backend('yolo11s.pt')
        >>>
        >>> # Explicitly use OpenVINO
        >>> model = create_inference_backend('yolo11s_openvino_model/yolo11s.xml', 'openvino')
        >>>
        >>> # Explicitly use PyTorch
        >>> model = create_inference_backend('yolo11s.pt', 'pytorch')
    """

    # Auto-detect backend based on model path
    if backend == 'auto':
        if '_openvino_model' in model_path or model_path.endswith('.xml'):
            backend = 'openvino'
            logger.debug(f"Auto-detected OpenVINO backend from path: {model_path}")
        else:
            backend = 'pytorch'
            logger.debug(f"Auto-detected PyTorch backend from path: {model_path}")

    # Try to load model with OpenVINO backend
    if backend == 'openvino':
        try:
            # Verify OpenVINO is available
            import openvino as ov
            openvino_version = ov.__version__
            logger.info(f"Using OpenVINO backend (version {openvino_version})")

            # Check if OpenVINO model file exists
            if not os.path.exists(model_path):
                logger.error(f"OpenVINO model not found: {model_path}")
                logger.error(f"Current working directory: {os.getcwd()}")
                logger.error(f"Absolute path: {os.path.abspath(model_path)}")
                raise FileNotFoundError(f"OpenVINO model not found: {model_path}")

            # Load model - Ultralytics automatically handles OpenVINO format
            logger.info(f"Loading OpenVINO model: {model_path}")
            model = YOLO(model_path)

            logger.info("✓ OpenVINO model loaded successfully")
            return model

        except ImportError as e:
            logger.warning(f"OpenVINO not available: {e}")
            logger.info("Falling back to PyTorch backend")

            # Try to find equivalent PyTorch model
            pytorch_path = _convert_to_pytorch_path(model_path)
            if os.path.exists(pytorch_path):
                logger.info(f"Using PyTorch model: {pytorch_path}")
                return YOLO(pytorch_path)
            else:
                logger.error(f"PyTorch fallback model not found: {pytorch_path}")
                raise

        except Exception as e:
            logger.warning(f"OpenVINO model loading failed: {e}")
            logger.info("Falling back to PyTorch backend")

            # Try to find equivalent PyTorch model
            pytorch_path = _convert_to_pytorch_path(model_path)
            if os.path.exists(pytorch_path):
                logger.info(f"Using PyTorch model: {pytorch_path}")
                return YOLO(pytorch_path)
            else:
                logger.error(f"PyTorch fallback model not found: {pytorch_path}")
                raise

    # Default: Load PyTorch model
    logger.info(f"Using PyTorch backend")
    logger.info(f"Loading PyTorch model: {model_path}")

    if not os.path.exists(model_path):
        logger.warning(f"PyTorch model not found: {model_path}, YOLO will attempt auto-download")

    model = YOLO(model_path)
    logger.info("✓ PyTorch model loaded successfully")

    return model


def _convert_to_pytorch_path(openvino_path: str) -> str:
    """
    Convert OpenVINO model path to equivalent PyTorch path

    Args:
        openvino_path: Path to OpenVINO model (.xml file)

    Returns:
        Equivalent PyTorch model path (.pt file)

    Examples:
        >>> _convert_to_pytorch_path('yolo11s_openvino_model/yolo11s.xml')
        'yolo11s.pt'
        >>> _convert_to_pytorch_path('yolo11s-pose_openvino_model/yolo11s-pose.xml')
        'yolo11s-pose.pt'
    """
    # Handle paths like: yolo11s_openvino_model/yolo11s.xml → yolo11s.pt
    if '_openvino_model' in openvino_path:
        # Extract model name from path
        parts = openvino_path.split('/')
        if len(parts) >= 2:
            xml_filename = parts[-1]  # e.g., "yolo11s.xml"
            model_name = xml_filename.replace('.xml', '.pt')  # e.g., "yolo11s.pt"
            return model_name

    # Handle direct .xml paths: model.xml → model.pt
    if openvino_path.endswith('.xml'):
        return openvino_path.replace('.xml', '.pt')

    # Default: assume it's already a .pt path
    return openvino_path


def get_model_backend_info(model_path: str) -> dict:
    """
    Get information about the model and its backend

    Args:
        model_path: Path to model file

    Returns:
        Dictionary with backend info:
        - backend: 'pytorch' or 'openvino'
        - path: Actual model path
        - exists: Whether file exists
        - size_mb: File size in MB (if exists)
    """
    info = {
        'backend': 'unknown',
        'path': model_path,
        'exists': False,
        'size_mb': None
    }

    # Detect backend
    if '_openvino_model' in model_path or model_path.endswith('.xml'):
        info['backend'] = 'openvino'
    elif model_path.endswith('.pt'):
        info['backend'] = 'pytorch'

    # Check file existence and size
    if os.path.exists(model_path):
        info['exists'] = True
        info['size_mb'] = os.path.getsize(model_path) / (1024 * 1024)

    return info


def test_backend_availability() -> dict:
    """
    Test availability of different inference backends

    Returns:
        Dictionary with backend availability:
        - pytorch: bool (always True - required dependency)
        - openvino: bool (True if OpenVINO is installed)
        - openvino_version: str or None
    """
    results = {
        'pytorch': True,  # Always available (required dependency)
        'openvino': False,
        'openvino_version': None
    }

    # Test OpenVINO availability
    try:
        import openvino as ov
        results['openvino'] = True
        results['openvino_version'] = ov.__version__
    except ImportError:
        pass

    return results


# Export public API
__all__ = [
    'create_inference_backend',
    'get_model_backend_info',
    'test_backend_availability',
]

# src/utils.py
"""
Optimized utility functions for image handling and data structures.
Provides core utilities for image loading, rotation, visualization, and barcode handling.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Union, Optional
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# DATA CLASSES


@dataclass
class BarcodeInfo:
    """
    Holds decoded barcode metadata from pyzbar.

    Attributes:
        value (str): Decoded string from the barcode
        rect (Tuple[int, int, int, int]): (x, y, w, h) in image coordinates
        symbology (str): Barcode type, e.g. 'CODE128', 'QRCODE', 'CODE39'
    """
    value: str
    rect: Tuple[int, int, int, int]
    symbology: str

    def __post_init__(self) -> None:
        """Validate barcode data after initialization."""
        if not isinstance(self.value, str):
            raise ValueError(f"value must be string, got {type(self.value)}")

        if not isinstance(self.rect, tuple) or len(self.rect) != 4:
            raise ValueError(f"rect must be 4-tuple, got {self.rect}")

        if not all(isinstance(x, int) for x in self.rect):
            raise ValueError(f"rect values must be integers, got {self.rect}")

        if not isinstance(self.symbology, str):
            raise ValueError(f"symbology must be string, got {type(self.symbology)}")



# IMAGE LOADING & VALIDATION


def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Safely load an image from raw bytes.
    Automatically detects format and returns BGR numpy array.

    Args:
        image_bytes: Raw image bytes (JPEG, PNG, etc.)

    Returns:
        BGR image as numpy array, or None if loading fails

    Raises:
        ValueError: If input is None or empty

    Note:
        Always returns a copy to prevent accidental upstream modifications.
    """
    if image_bytes is None:
        raise ValueError("No image bytes provided")

    if not isinstance(image_bytes, bytes):
        raise ValueError(f"Expected bytes, got {type(image_bytes)}")

    try:
        # Convert bytes to numpy array
        arr = np.frombuffer(image_bytes, np.uint8)

        if arr.size == 0:
            raise ValueError("Empty image buffer")

        # Decode image from array
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image data: could not be decoded by OpenCV")

        if img.size == 0:
            raise ValueError("Decoded image is empty")

        logger.debug(f"Loaded image: shape={img.shape}, dtype={img.dtype}")

        # Always return a copy to avoid accidental in-place modifications
        return img.copy()

    except ValueError:
        # Re-raise our custom ValueError messages
        raise
    except Exception as e:
        logger.error(f"Error loading image from bytes: {e}", exc_info=True)
        raise ValueError(f"Failed to load image: {str(e)}")


def validate_image(img: np.ndarray, operation: str = "processing") -> bool:
    """
    Validate image array for operations.

    Args:
        img: Image to validate
        operation: Name of operation (for logging)

    Returns:
        True if valid, False otherwise
    """
    if img is None:
        logger.warning(f"Image is None for {operation}")
        return False

    if not isinstance(img, np.ndarray):
        logger.warning(f"Image is not ndarray for {operation}")
        return False

    if img.size == 0:
        logger.warning(f"Image is empty for {operation}")
        return False

    if img.shape[0] <= 0 or img.shape[1] <= 0:
        logger.warning(f"Image has invalid dimensions for {operation}: {img.shape}")
        return False

    return True


def get_image_stats(img: np.ndarray) -> dict:
    """
    Get statistics about an image for debugging and validation.

    Args:
        img: Image to analyze

    Returns:
        Dictionary with image statistics
    """
    if img is None or img.size == 0:
        return {
            "shape": None,
            "dtype": None,
            "channels": None,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    try:
        channels = img.shape[2] if len(img.shape) == 3 else 1

        return {
            "shape": img.shape,
            "dtype": str(img.dtype),
            "channels": channels,
            "mean": float(np.mean(img)),
            "std": float(np.std(img)),
            "min": float(np.min(img)),
            "max": float(np.max(img)),
        }
    except Exception as e:
        logger.error(f"Error calculating image stats: {e}")
        return {}



# IMAGE ROTATION


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotate image by multiples of 90 degrees efficiently.

    Optimization: Only supports 90° increments (0, 90, 180, 270).
    For 0°, returns copy. For unsupported angles, returns original copy.

    Args:
        img: Input image
        angle: Rotation angle in degrees (0, 90, 180, 270 recommended)

    Returns:
        Rotated image copy

    Raises:
        ValueError: If image is None
    """
    if img is None:
        raise ValueError("rotate_image: input image is None")

    if not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("rotate_image: invalid image array")

    try:
        # Normalize angle to 0-360 range
        angle = angle % 360

        if angle == 0:
            logger.debug("Rotation 0°: returning copy")
            return img.copy()

        elif angle == 90:
            logger.debug("Rotation 90° clockwise")
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        elif angle == 180:
            logger.debug("Rotation 180°")
            return cv2.rotate(img, cv2.ROTATE_180)

        elif angle == 270:
            logger.debug("Rotation 270° clockwise (90° counter-clockwise)")
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        else:
            logger.warning(f"Unsupported rotation angle: {angle}°. Returning copy.")
            return img.copy()

    except Exception as e:
        logger.error(f"Error rotating image by {angle}°: {e}", exc_info=True)
        raise



# VISUALIZATION


def draw_barcode_boxes(
    img: np.ndarray,
    barcodes: List[Union[BarcodeInfo, dict]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw rectangles around detected barcodes on image.

    Flexible input handling:
      - Accepts List[BarcodeInfo] from OCR pipeline
      - Accepts List[dict] from output_to_dict() JSON serialization
      - Skips invalid entries gracefully

    Args:
        img: Input image (BGR)
        barcodes: List of barcode info (BarcodeInfo or dict)
        color: RGB color for rectangles (default: green)
        thickness: Rectangle line thickness in pixels

    Returns:
        Copy of image with barcode boxes drawn

    Raises:
        ValueError: If image is None or invalid
    """
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("draw_barcode_boxes: input image is None or invalid")

    if img.size == 0:
        raise ValueError("draw_barcode_boxes: input image is empty")

    if not barcodes:
        logger.debug("No barcodes to draw")
        return img.copy()

    try:
        out = img.copy()
        drawn_count = 0

        for i, bc in enumerate(barcodes):
            rect = None
            barcode_id = f"barcode_{i}"

            # Handle BarcodeInfo dataclass
            if isinstance(bc, BarcodeInfo):
                rect = bc.rect
                barcode_id = f"{bc.symbology}_{bc.value[:10]}"

            # Handle dict from output_to_dict()
            elif isinstance(bc, dict):
                rect = bc.get("rect")
                barcode_id = bc.get("symbology", "unknown")

            else:
                logger.warning(f"Skipping invalid barcode entry {i}: {type(bc)}")
                continue

            # Validate and draw rectangle
            if rect is None:
                logger.warning(f"Barcode {barcode_id} has no rect")
                continue

            try:
                if not (isinstance(rect, (tuple, list)) and len(rect) == 4):
                    logger.warning(f"Barcode {barcode_id} has invalid rect: {rect}")
                    continue

                x, y, w, h = rect

                # Validate coordinates
                if not all(isinstance(v, (int, float)) and v >= 0 for v in [x, y, w, h]):
                    logger.warning(f"Barcode {barcode_id} has negative/invalid coordinates")
                    continue

                # Draw rectangle
                cv2.rectangle(
                    out,
                    (int(x), int(y)),
                    (int(x + w), int(y + h)),
                    color,
                    thickness
                )
                drawn_count += 1
                logger.debug(f"Drew rectangle for {barcode_id} at ({x}, {y}, {w}, {h})")

            except Exception as e:
                logger.warning(f"Error drawing barcode {barcode_id}: {e}")
                continue

        logger.info(f"Drew {drawn_count} barcode boxes")
        return out

    except Exception as e:
        logger.error(f"Error in draw_barcode_boxes: {e}", exc_info=True)
        raise


def draw_bounding_box(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a single labeled bounding box on image.

    Args:
        img: Input image
        x, y: Top-left corner coordinates
        w, h: Width and height
        label: Optional text label
        color: RGB color tuple
        thickness: Line thickness

    Returns:
        Image copy with bounding box drawn
    """
    if img is None or img.size == 0:
        raise ValueError("Invalid image")

    try:
        out = img.copy()

        # Draw rectangle
        cv2.rectangle(
            out,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            color,
            thickness
        )

        # Draw label if provided
        if label:
            cv2.putText(
                out,
                label,
                (int(x), int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        return out

    except Exception as e:
        logger.error(f"Error drawing bounding box: {e}", exc_info=True)
        raise



# IMAGE COMPARISON & ANALYSIS


def images_are_equal(img1: np.ndarray, img2: np.ndarray) -> bool:
    """
    Check if two images are pixel-identical.
    Useful for testing and validation.

    Args:
        img1, img2: Images to compare

    Returns:
        True if images are identical, False otherwise
    """
    if img1 is None or img2 is None:
        return img1 is img2

    try:
        if img1.shape != img2.shape or img1.dtype != img2.dtype:
            return False

        return np.array_equal(img1, img2)

    except Exception as e:
        logger.error(f"Error comparing images: {e}")
        return False


def get_image_difference(img1: np.ndarray, img2: np.ndarray) -> Optional[float]:
    """
    Calculate mean absolute difference between two images.
    Useful for comparing preprocessing results.

    Args:
        img1, img2: Images to compare (must be same shape)

    Returns:
        Mean absolute difference value, or None on error
    """
    if img1 is None or img2 is None or img1.size == 0 or img2.size == 0:
        return None

    try:
        if img1.shape != img2.shape:
            logger.warning(f"Images have different shapes: {img1.shape} vs {img2.shape}")
            return None

        diff = cv2.absdiff(img1.astype(np.float32), img2.astype(np.float32))
        return float(np.mean(diff))

    except Exception as e:
        logger.error(f"Error calculating image difference: {e}")
        return None

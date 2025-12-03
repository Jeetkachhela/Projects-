# src/preprocessing.py
"""
Optimized image preprocessing pipeline for OCR text extraction.
Provides specialized preprocessing for:
  - Barcode detection (crisp edges)
  - OCR blocks (large text regions)
  - ID lines (single horizontal text)
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# DEBUG CONFIGURATION


# Toggle to save intermediate images for inspection during debugging
DEBUG_SAVE = False
DEBUG_DIR = "intermediate"


def _ensure_debug_dir() -> None:
    """Create debug directory if it doesn't exist."""
    if DEBUG_SAVE and not os.path.exists(DEBUG_DIR):
        try:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            logger.debug(f"Created debug directory: {DEBUG_DIR}")
        except OSError as e:
            logger.error(f"Failed to create debug directory: {e}")


def _save_debug_image(img: np.ndarray, name: str) -> None:
    """
    Save intermediate images for inspection.
    Files go into ./intermediate/<timestamp>_<name>.png

    Args:
        img: Image array to save
        name: Descriptive name for the image
    """
    if not DEBUG_SAVE or img is None:
        return

    _ensure_debug_dir()

    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(DEBUG_DIR, f"{ts}_{name}.png")

        if cv2.imwrite(path, img):
            logger.debug(f"Saved debug image: {path}")
        else:
            logger.warning(f"Failed to write debug image: {path}")
    except Exception as e:
        logger.error(f"Error saving debug image '{name}': {e}")


# UTILITY FUNCTIONS


def _to_gray(img: np.ndarray) -> np.ndarray:
    """
    Convert image to single-channel uint8 grayscale.
    Handles color, grayscale, and edge cases.

    Args:
        img: Input image (BGR or grayscale)

    Returns:
        Grayscale uint8 image

    Raises:
        ValueError: If image is None or empty
    """
    if img is None or img.size == 0:
        raise ValueError("Input image is None or empty")

    try:
        # Check if already grayscale
        if len(img.shape) == 2:
            gray = img
        elif len(img.shape) == 3:
            # Convert from BGR to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Invalid image shape: {img.shape}")

        # Ensure uint8 dtype
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            gray = gray.astype(np.uint8)

        _save_debug_image(gray, "00_grayscale")
        return gray

    except Exception as e:
        logger.error(f"Error converting to grayscale: {e}", exc_info=True)
        raise


def _validate_image(img: np.ndarray, operation: str = "preprocessing") -> bool:
    """
    Validate image before processing.

    Args:
        img: Image to validate
        operation: Name of operation (for logging)

    Returns:
        True if valid, False otherwise
    """
    if img is None:
        logger.warning(f"Image is None for {operation}")
        return False

    if img.size == 0:
        logger.warning(f"Image is empty for {operation}")
        return False

    return True



# BARCODE PREPROCESSING


def preprocess_for_barcode(img: np.ndarray) -> np.ndarray:
    """
    Preprocessing for 1D barcode detection.

    Strategy:
      - Grayscale conversion
      - Bilateral filtering (edge-preserving denoising)
      - CLAHE contrast enhancement
      - Unsharp mask for bar edge sharpening
      - Normalization to full range

    Result: Crisp, high-contrast image with clear bar edges.

    Args:
        img: Input image

    Returns:
        Preprocessed barcode image
    """
    if not _validate_image(img, "barcode preprocessing"):
        return np.zeros((1, 1), dtype=np.uint8)

    try:
        gray = _to_gray(img)

        # Mild denoise while preserving bar edges
        # Bilateral filter: strong for edges, smooth for flat regions
        denoised = cv2.bilateralFilter(
            gray,
            d=7,
            sigmaColor=40,
            sigmaSpace=40
        )
        _save_debug_image(denoised, "01_barcode_bilateral")

        # Contrast Limited Adaptive Histogram Equalization
        # Improves local contrast without over-amplifying noise
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        _save_debug_image(enhanced, "02_barcode_clahe")

        # Unsharp mask: sharpen bar edges without introducing artifacts
        # Formula: sharpened = original + (original - blur)
        blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
        sharp = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
        _save_debug_image(sharp, "03_barcode_sharp")

        # Normalize to full 0-255 range
        out = cv2.normalize(sharp, None, 0, 255, cv2.NORM_MINMAX)
        out = out.astype(np.uint8)

        _save_debug_image(out, "04_barcode_final")
        return out

    except Exception as e:
        logger.error(f"Error in barcode preprocessing: {e}", exc_info=True)
        return np.zeros((1, 1), dtype=np.uint8)



# OCR BLOCK PREPROCESSING


def preprocess_for_ocr_block(img: np.ndarray) -> np.ndarray:
    """
    Preprocessing for large text regions (bottom half / full image).

    Strategy:
      - Grayscale conversion
      - Strong denoising (fastNlMeansDenoising)
      - CLAHE contrast enhancement
      - OTSU global thresholding to binary
      - Morphological opening (remove noise)
      - Morphological closing (solidify characters)

    Result: High-contrast, low-noise binary image optimized for OCR.

    Args:
        img: Input image

    Returns:
        Preprocessed OCR image (binary)
    """
    if not _validate_image(img, "OCR block preprocessing"):
        return np.zeros((1, 1), dtype=np.uint8)

    try:
        gray = _to_gray(img)

        # Strong denoising while preserving text edges
        # fastNlMeansDenoising is more effective than bilateral for structured text
        denoised = cv2.fastNlMeansDenoising(
            gray,
            None,
            h=15,               # Filter strength (higher = more blur)
            templateWindowSize=7,
            searchWindowSize=21
        )
        _save_debug_image(denoised, "01_block_denoised")

        # Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        _save_debug_image(enhanced, "02_block_clahe")

        # Global OTSU threshold for robust binarization
        # OTSU automatically finds optimal threshold
        _, th = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        _save_debug_image(th, "03_block_threshold")

        # Morphological operations to clean up binary image
        # Opening: removes small specks and noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_open, iterations=1)
        _save_debug_image(opened, "04_block_opened")

        # Closing: connects broken characters, solidifies text
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        _save_debug_image(closed, "05_block_closed")

        out = closed.astype(np.uint8)
        _save_debug_image(out, "06_block_final")
        return out

    except Exception as e:
        logger.error(f"Error in OCR block preprocessing: {e}", exc_info=True)
        return np.zeros((1, 1), dtype=np.uint8)



# ID-LINE PREPROCESSING

def preprocess_for_id_line(img: np.ndarray) -> np.ndarray:
    """
    Preprocessing for horizontal ID lines (under barcodes).
    Optimized for single-line text with _1_ pattern.

    Strategy:
      - Grayscale conversion
      - Lighter denoising (preserve thin characters)
      - CLAHE contrast enhancement
      - OTSU binarization
      - Horizontal-biased morphological operations
      - Opening to remove noise

    Result: Binary image with continuous, well-separated characters.
    Prioritizes horizontal continuity using rectangular kernels.

    Args:
        img: Input ROI containing ID line

    Returns:
        Preprocessed ID line image (binary)
    """
    if not _validate_image(img, "ID line preprocessing"):
        return np.zeros((1, 1), dtype=np.uint8)

    try:
        gray = _to_gray(img)

        # Lighter denoising to preserve thin characters in ID line
        denoised = cv2.fastNlMeansDenoising(
            gray,
            None,
            h=10,                # Lower than block preprocessing
            templateWindowSize=7,
            searchWindowSize=21
        )
        _save_debug_image(denoised, "01_id_denoised")

        # Boost local contrast for ID line visibility
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        _save_debug_image(enhanced, "02_id_clahe")

        # OTSU binarization
        _, th = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        _save_debug_image(th, "03_id_threshold")

        # Morphological operations optimized for horizontal lines
        # Horizontal closing kernel: connects characters in line
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        _save_debug_image(closed, "04_id_closed")

        # Vertical closing kernel: solidify character height
        kernel_close_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        closed_v = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_close_v, iterations=1)
        _save_debug_image(closed_v, "05_id_closed_v")

        # Opening: remove tiny isolated noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(closed_v, cv2.MORPH_OPEN, kernel_open, iterations=1)
        _save_debug_image(cleaned, "06_id_opened")

        out = cleaned.astype(np.uint8)
        _save_debug_image(out, "07_id_final")
        return out

    except Exception as e:
        logger.error(f"Error in ID line preprocessing: {e}", exc_info=True)
        return np.zeros((1, 1), dtype=np.uint8)



# UTILITY PREPROCESSING FUNCTIONS


def get_preprocessing_stats(img: np.ndarray) -> dict:
    """
    Get statistics about an image for debugging/monitoring.

    Args:
        img: Input image

    Returns:
        Dictionary with image statistics
    """
    if img is None or img.size == 0:
        return {
            "shape": None,
            "dtype": None,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    try:
        return {
            "shape": img.shape,
            "dtype": str(img.dtype),
            "mean": float(np.mean(img)),
            "std": float(np.std(img)),
            "min": float(np.min(img)),
            "max": float(np.max(img)),
        }
    except Exception as e:
        logger.error(f"Error calculating preprocessing stats: {e}")
        return {}

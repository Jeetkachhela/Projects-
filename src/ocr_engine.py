# src/ocr_engine.py - FIXED VERSION
"""
Fixed OCR Engine for shipping label/waybill text extraction.
Key fixes:
  - Check ALL barcodes, not just first
  - Better fallback to full image OCR
  - Removed premature returns
  - Prioritize full-image OCR for images without barcodes
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

import cv2
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol
from paddleocr import PaddleOCR

from .utils import load_image_from_bytes, rotate_image, BarcodeInfo
from .preprocessing import (
    preprocess_for_barcode,
    preprocess_for_id_line,
    preprocess_for_ocr_block,
)
from .text_extraction import (
    find_target_in_text,
    highlight_match_in_text,
    ExtractionResult,
)

logger = logging.getLogger(__name__)

# GLOBALS


_ocr_instance: Optional[PaddleOCR] = None

def get_ocr_instance() -> PaddleOCR:
    """Lazy-load OCR instance."""
    global _ocr_instance
    if _ocr_instance is None:
        logger.info("Initializing PaddleOCR instance...")
        _ocr_instance = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=False,
            show_log=False
        )
    return _ocr_instance

# Only use 1D symbologies (exclude 2D to reduce noise)
BARCODE_TYPES = [
    ZBarSymbol.CODE128,
    ZBarSymbol.CODE39,
    ZBarSymbol.CODE93,
    ZBarSymbol.EAN13,
    ZBarSymbol.EAN8,
    ZBarSymbol.UPCA,
    ZBarSymbol.UPCE,
]

# Confidence thresholds (tuned for reliability)
BARCODE_CONF = 0.99
UNDER_BARCODE_CONF = 0.94
BOTTOM_CONF = 0.86
FULL_CONF = 0.85  # INCREASED - prioritize full image OCR
HEAVY_FULL_CONF = 0.92

MIN_OCR_CONFIDENCE = 0.30  # LOWERED - accept more OCR results
ROTATION_ANGLES = [0, 90, 180, 270]


@dataclass
class OCRPipelineOutput:
    """Complete pipeline output."""
    matched_text: Optional[str]
    highlighted_text: str
    confidence: float
    source: Optional[str]
    barcodes: List[BarcodeInfo]
    raw_text_ocr: str
    errors: List[str] = field(default_factory=list)

    def is_success(self) -> bool:
        return self.matched_text is not None


# LOW-LEVEL HELPERS


def _run_paddle_ocr(img: np.ndarray) -> Tuple[str, float]:
    """Run PaddleOCR on preprocessed image."""
    if img is None or img.size == 0:
        return "", 0.0

    try:
        ocr = get_ocr_instance()
        result = ocr.ocr(img, cls=False)

        if not result or not result[0]:
            logger.debug("No OCR results")
            return "", 0.0

        texts: List[str] = []
        confs: List[float] = []

        for line in result[0]:
            text, conf = line[1]
            conf = float(conf)

            # LOWERED threshold - accept more text
            if conf >= MIN_OCR_CONFIDENCE:
                texts.append(text.strip())
                confs.append(conf)

        if not texts:
            logger.debug("All OCR lines filtered by confidence")
            return "", 0.0

        joined = "\n".join(texts)
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        logger.debug(f"OCR extracted {len(texts)} lines, avg confidence: {avg_conf:.3f}")
        return joined, avg_conf

    except Exception as e:
        logger.error(f"PaddleOCR error: {e}", exc_info=True)
        return "", 0.0


def _decode_barcodes(img: np.ndarray) -> List[BarcodeInfo]:
    """Decode 1D barcodes using pyzbar."""
    try:
        processed = preprocess_for_barcode(img)
        decoded = decode(processed, symbols=BARCODE_TYPES)

        out: List[BarcodeInfo] = []
        for b in decoded:
            try:
                value = b.data.decode("utf-8", errors="ignore").strip()
                if value:
                    out.append(
                        BarcodeInfo(
                            value=value,
                            rect=(b.rect.left, b.rect.top, b.rect.width, b.rect.height),
                            symbology=b.type,
                        )
                    )
            except (AttributeError, UnicodeDecodeError) as e:
                logger.warning(f"Failed to decode barcode: {e}")
                continue

        logger.debug(f"Decoded {len(out)} barcodes")
        return out

    except Exception as e:
        logger.error(f"Barcode decoding error: {e}", exc_info=True)
        return []


def _dedupe_barcodes(barcodes: List[BarcodeInfo]) -> List[BarcodeInfo]:
    """Remove duplicate barcodes."""
    seen = set()
    unique: List[BarcodeInfo] = []

    for b in barcodes:
        key = (b.value, b.symbology)
        if key not in seen:
            seen.add(key)
            unique.append(b)

    return unique


def _ocr_strip_under_last_1d_barcode(
    img: np.ndarray, barcodes: List[BarcodeInfo]
) -> Tuple[str, float]:
    """OCR strip under lowest 1D barcode."""
    if not barcodes:
        return "", 0.0

    one_d = [
        b for b in barcodes
        if b.symbology not in ("QRCODE", "DATAMATRIX")
    ]

    if not one_d:
        return "", 0.0

    last = max(one_d, key=lambda b: b.rect[1] + b.rect[3])
    x, y, w, h = last.rect
    H, W = img.shape[:2]

    y1 = y + h
    y2 = min(H, y + h * 4)
    x1 = max(0, x - 50)
    x2 = min(W, x + w + 80)

    if y1 >= y2 or x1 >= x2:
        logger.debug(f"Invalid ROI dimensions")
        return "", 0.0

    roi = img[y1:y2, x1:x2]

    if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 30:
        logger.debug(f"ROI too small")
        return "", 0.0

    try:
        proc = preprocess_for_id_line(roi)
        return _run_paddle_ocr(proc)
    except Exception as e:
        logger.error(f"Error in strip OCR: {e}", exc_info=True)
        return "", 0.0


def _ocr_bottom_region(img: np.ndarray) -> Tuple[str, float]:
    """OCR bottom half."""
    try:
        H, W = img.shape[:2]
        bottom = img[int(H * 0.5):, :]
        proc = preprocess_for_ocr_block(bottom)
        return _run_paddle_ocr(proc)
    except Exception as e:
        logger.error(f"Error in bottom region OCR: {e}", exc_info=True)
        return "", 0.0


def _ocr_full_image(img: np.ndarray) -> Tuple[str, float]:
    """OCR full image."""
    try:
        proc = preprocess_for_ocr_block(img)
        return _run_paddle_ocr(proc)
    except Exception as e:
        logger.error(f"Error in full-image OCR: {e}", exc_info=True)
        return "", 0.0


def _full_image_ocr_heavy(img: np.ndarray) -> Tuple[str, float]:
    """Very aggressive OCR preprocessing."""
    if img is None or img.size == 0:
        return "", 0.0

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        th = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35,
            10,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        th = cv2.dilate(th, kernel, iterations=1)
        th = cv2.medianBlur(th, 3)

        return _run_paddle_ocr(th)

    except Exception as e:
        logger.error(f"Heavy OCR preprocessing error: {e}", exc_info=True)
        return "", 0.0


# MAIN PIPELINE - FIXED TO NOT STOP PREMATURELY


def process_image_bytes(image_bytes: bytes) -> OCRPipelineOutput:
    """
    FIXED: Robust multi-stage OCR pipeline.
    Key fix: Continues through ALL strategies before returning,
    accumulates best match instead of early returns.
    """
    errors: List[str] = []

    try:
        base_img = load_image_from_bytes(image_bytes)
        if base_img is None:
            return OCRPipelineOutput(
                matched_text=None,
                highlighted_text="",
                confidence=0.0,
                source=None,
                barcodes=[],
                raw_text_ocr="",
                errors=["Image loading failed: returned None"],
            )
    except Exception as e:
        logger.error(f"Image load error: {e}", exc_info=True)
        return OCRPipelineOutput(
            matched_text=None,
            highlighted_text="",
            confidence=0.0,
            source=None,
            barcodes=[],
            raw_text_ocr="",
            errors=[f"Image load error: {str(e)}"],
        )

    all_barcodes: List[BarcodeInfo] = []
    best_ocr_text: str = ""
    best_ocr_conf: float = 0.0
    best_ocr_source: str = ""

    # Track best match found so far
    best_match: Optional[ExtractionResult] = None
    best_match_source: str = ""

    def _update_best_ocr(text: str, conf: float, source: str) -> None:
        nonlocal best_ocr_text, best_ocr_conf, best_ocr_source
        if text and conf > best_ocr_conf:
            best_ocr_text = text
            best_ocr_conf = conf
            best_ocr_source = source

    def _update_best_match(res: ExtractionResult, text_used: str) -> None:
        nonlocal best_match, best_match_source
        if res and (best_match is None or res.confidence > best_match.confidence):
            best_match = res
            best_match_source = text_used

    def _make_output(
        res: ExtractionResult,
        text_used: str,
    ) -> OCRPipelineOutput:
        highlighted = highlight_match_in_text(text_used, res)
        unique_barcodes = _dedupe_barcodes(all_barcodes)
        return OCRPipelineOutput(
            matched_text=res.matched_text,
            highlighted_text=highlighted,
            confidence=res.confidence,
            source=res.source,
            barcodes=unique_barcodes,
            raw_text_ocr=text_used,
            errors=errors,
        )

    # MULTI-ANGLE PROCESSING - CHECK ALL STRATEGIES
    for angle in ROTATION_ANGLES:
        try:
            rotated = rotate_image(base_img, angle)
            logger.debug(f"Processing angle: {angle}°")
        except Exception as e:
            logger.error(f"Image rotation error (angle={angle}): {e}")
            errors.append(f"Rotation error (angle={angle}): {str(e)}")
            continue

        # 1) CHECK ALL BARCODES (not just first!)

        try:
            barcodes = _decode_barcodes(rotated)
            if barcodes:
                all_barcodes.extend(barcodes)
                logger.debug(f"Decoded {len(barcodes)} barcode(s) at {angle}°")

                # CHECK EACH BARCODE
                for i, bc in enumerate(barcodes):
                    logger.debug(f"Checking barcode {i}: {bc.value}")
                    res = find_target_in_text(
                        bc.value,
                        source="barcode",
                        base_confidence=BARCODE_CONF,
                    )
                    if res:
                        logger.info(f"Found target in barcode {i} at {angle}°: {res.matched_text}")
                        _update_best_match(res, bc.value)

        except Exception as e:
            logger.error(f"Barcode decode error (angle={angle}): {e}", exc_info=True)
            errors.append(f"Barcode decode error (angle={angle}): {str(e)}")


        # 2) OCR STRIP UNDER LAST 1D BARCODE

        try:
            under_text, under_conf = _ocr_strip_under_last_1d_barcode(rotated, barcodes)
            if under_text:
                _update_best_ocr(under_text, under_conf, "ocr_under_barcode")

                res = find_target_in_text(
                    under_text,
                    source="ocr_under_barcode",
                    base_confidence=UNDER_BARCODE_CONF,
                )
                if res:
                    logger.info(f"Found target in under-barcode strip at {angle}°: {res.matched_text}")
                    _update_best_match(res, under_text)

        except Exception as e:
            logger.error(f"OCR under-barcode error (angle={angle}): {e}", exc_info=True)
            errors.append(f"OCR under-barcode error (angle={angle}): {str(e)}")


        # 3) OCR BOTTOM HALF

        try:
            bottom_text, bottom_conf = _ocr_bottom_region(rotated)
            if bottom_text:
                _update_best_ocr(bottom_text, bottom_conf, "ocr_bottom")

                res = find_target_in_text(
                    bottom_text,
                    source="ocr_bottom",
                    base_confidence=BOTTOM_CONF,
                )
                if res:
                    logger.info(f"Found target in bottom region at {angle}°: {res.matched_text}")
                    _update_best_match(res, bottom_text)

        except Exception as e:
            logger.error(f"OCR bottom-region error (angle={angle}): {e}", exc_info=True)
            errors.append(f"OCR bottom-region error (angle={angle}): {str(e)}")


        # 4) OCR FULL IMAGE - PRIORITIZED

        try:
            full_text, full_conf = _ocr_full_image(rotated)
            if full_text:
                _update_best_ocr(full_text, full_conf, "ocr_full")

                res = find_target_in_text(
                    full_text,
                    source="ocr_full",
                    base_confidence=FULL_CONF,
                )
                if res:
                    logger.info(f"Found target in full-image OCR at {angle}°: {res.matched_text}")
                    _update_best_match(res, full_text)

        except Exception as e:
            logger.error(f"OCR full-image error (angle={angle}): {e}", exc_info=True)
            errors.append(f"OCR full-image error (angle={angle}): {str(e)}")

    # HEAVY FULL-IMAGE OCR FALLBACK (ORIGINAL ORIENTATION)

    try:
        logger.debug("Attempting heavy preprocessing fallback...")
        heavy_text, heavy_conf = _full_image_ocr_heavy(base_img)
        if heavy_text:
            _update_best_ocr(heavy_text, heavy_conf, "ocr_heavy")

            res = find_target_in_text(
                heavy_text,
                source="ocr_heavy",
                base_confidence=HEAVY_FULL_CONF,
            )
            if res:
                logger.info(f"Found target in heavy preprocessing: {res.matched_text}")
                _update_best_match(res, heavy_text)

    except Exception as e:
        logger.error(f"OCR heavy full-image error: {e}", exc_info=True)
        errors.append(f"OCR heavy full-image error: {str(e)}")


    # RETURN BEST RESULT FOUND

    if best_match:
        logger.info(f"FINAL RESULT: {best_match.matched_text} from {best_match.source}")
        return _make_output(best_match, best_match_source)

    # No match found - return best OCR text
    unique_barcodes = _dedupe_barcodes(all_barcodes)
    logger.warning("No target pattern found. Returning best OCR text.")

    return OCRPipelineOutput(
        matched_text=None,
        highlighted_text="",
        confidence=0.0,
        source=None,
        barcodes=unique_barcodes,
        raw_text_ocr=best_ocr_text,
        errors=errors,
    )


def output_to_dict(out: OCRPipelineOutput) -> Dict:
    """Convert OCRPipelineOutput to dictionary."""
    return {
        "matched_text": out.matched_text,
        "highlighted_text": out.highlighted_text,
        "confidence": round(out.confidence, 4),
        "source": out.source,
        "is_success": out.is_success(),
        "barcodes": [
            {
                "value": b.value,
                "rect": b.rect,
                "symbology": b.symbology,
            }
            for b in out.barcodes
        ],
        "raw_text_ocr": out.raw_text_ocr,
        "errors": out.errors,
    }

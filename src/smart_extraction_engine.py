# src/smart_extraction_engine.py
"""
Strategic extraction orchestrator for multi-source OCR pipelines.

Primary goals:
    - Harvest ALL potential candidates from barcode + OCR paths
    - Normalize, validate, and score them consistently
    - Detect + reject hallucinations and low-quality garbage
    - Select a SINGLE best result, with auditability

Design philosophy:
    Collect widely, filter aggressively, rank intelligently.
    Reliability is more important than optimism.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

from .extraction_candidate import ExtractionCandidate, ExtractionSource
from .text_extraction import (
    find_target_in_text,
    validate_extracted_id,
)
from .utils import rotate_image

logger = logging.getLogger(__name__)



# MAIN ENGINE


class SmartExtractionEngine:
    """
    Multi-source candidate miner and scorer.

    Input:
        - Image
        - Barcodes (optional)
        - OCR results from multiple regions and rotations

    Output:
        - Ranked list of ExtractionCandidate
        - Best candidate selection
        - Errors accumulated but not disruptive
    """

    # Rotation penalty:
    # Higher penalty for rotated images to reflect less-reliable ROIs.
    _ROTATION_PENALTY = {
        0: 1.00,
        90: 0.96,
        180: 0.94,
        270: 0.90,
    }

    def __init__(self):
        self.candidates: List[ExtractionCandidate] = []
        self.errors: List[str] = []
        self._seen_text: set[str] = set()  # For dedupe safety

    # TOP-LEVEL EXTRACTOR


    def extract_all_candidates(
        self,
        base_img: np.ndarray,
        barcodes_decoded: list,
        ocr_results: Dict[str, Tuple[str, float]],
        rotation_angles: List[int] = (0, 90, 180, 270),
    ) -> List[ExtractionCandidate]:
        """
        Collect candidates from all sources:
            - barcode values
            - region OCR outputs
            - across all rotation angles
        """

        self.candidates = []
        self.errors = []
        self._seen_text = set()

        logger.info("\n" + "="*72)
        logger.info("SMART EXTRACTION ENGINE STARTED")
        logger.info("="*72)

        # Process barcodes without rotation
        self._extract_from_barcodes(barcodes_decoded)

        # Process OCR output from OCR engine (multi-angle info already encoded)
        self._extract_from_ocr_results(ocr_results, rotation_angles)

        # Final filtering and sorting
        self._dedupe_and_rank()

        logger.info(f"\n=== TOTAL VALID CANDIDATES: {len(self.candidates)} ===")
        for i, cand in enumerate(self.candidates[:10], 1):
            logger.info(f"{i:02d}. {cand}")

        return self.candidates


    # BEST CANDIDATE ACCESSOR

    def get_best_candidate(self) -> Tuple[Optional[ExtractionCandidate], float]:
        """Return best candidate with score."""
        if not self.candidates:
            logger.warning("No valid candidates in engine.")
            return None, 0.0

        best = self.candidates[0]
        logger.info(
            f"\n=== BEST CANDIDATE ===\n"
            f"ID: {best.matched_text}\n"
            f"Score: {best.composite_score:.3f}\n"
            f"Source: {best.source.value}\n"
            f"Angle: {best.angle}°\n"
        )
        return best, best.composite_score


    # BARCODE EXTRACTION


    def _extract_from_barcodes(self, barcodes: list) -> None:
        """Barcode values are the most reliable source, when available."""
        try:
            if not barcodes:
                return

            for idx, b in enumerate(barcodes):
                text = str(b.value).upper()

                result = find_target_in_text(
                    text=text,
                    source="barcode",
                    base_confidence=0.99,
                )
                if not result:
                    continue

                candidate = ExtractionCandidate(
                    matched_text=result.matched_text,
                    confidence=result.confidence,
                    source=self._barcode_index_to_source(idx),
                    angle=0,
                    raw_text=text,
                    validity_score=1.0,  # Barcode pattern match is reliable
                )
                self.candidates.append(candidate)

        except Exception as e:
            logger.error(f"Barcode extraction error: {e}", exc_info=True)
            self.errors.append(str(e))

    @staticmethod
    def _barcode_index_to_source(idx: int) -> ExtractionSource:
        """Semantic mapping of index to label importance."""
        if idx == 0:
            return ExtractionSource.BARCODE_0
        elif idx == 1:
            return ExtractionSource.BARCODE_1
        elif idx == 2:
            return ExtractionSource.BARCODE_2
        return ExtractionSource.BARCODE_N


    # OCR EXTRACTION


    def _extract_from_ocr_results(
        self,
        ocr_results: Dict[str, Tuple[str, float]],
        rotation_angles: List[int],
    ) -> None:
        """Pull candidates from OCR output of various strategies."""
        if not ocr_results:
            return

        for source_key, (text, conf) in ocr_results.items():
            if not text or conf <= 0:
                continue

            src = self._key_to_source(source_key)

            for angle in rotation_angles:
                self._process_ocr_text(
                    text=text,
                    base_conf=conf,
                    source=src,
                    angle=angle,
                )

    def _process_ocr_text(
        self,
        text: str,
        base_conf: float,
        source: ExtractionSource,
        angle: int,
    ) -> None:
        """
        Pass OCR text through robust extraction pipeline
        to repair OCR errors and get normalized ID.
        """

        if not text.strip():
            return

        result = find_target_in_text(
            text=text,
            source=source.value,
            base_confidence=base_conf,
        )

        if not result:
            return

        if not validate_extracted_id(result.matched_text):
            logger.debug(f"Rejected invalid candidate: {result.matched_text}")
            return

        # Compute angle penalty
        penalty = self._ROTATION_PENALTY.get(angle, 0.90)

        adjusted_conf = min(0.999, result.confidence * penalty)

        candidate = ExtractionCandidate(
            matched_text=result.matched_text,
            confidence=adjusted_conf,
            source=source,
            angle=angle,
            raw_text=text,
            validity_score=1.0,
        )
        self._add_candidate(candidate)


    # CANDIDATE MANAGEMENT

    def _add_candidate(self, candidate: ExtractionCandidate) -> None:
        """Avoid duplicates and hallucinations."""
        key = (candidate.matched_text, candidate.source.value, candidate.angle)

        if candidate.matched_text in self._seen_text:
            return

        self._seen_text.add(candidate.matched_text)
        self.candidates.append(candidate)

    def _dedupe_and_rank(self) -> None:
        """Remove duplicates, filter garbage, and sort by composite score."""

        if not self.candidates:
            return

        # Drop junk: extremely short or malformed
        filtered = []
        for c in self.candidates:
            if len(c.matched_text) < 14:
                continue
            filtered.append(c)
        self.candidates = filtered

        # Sort descending by score
        self.candidates.sort(reverse=True)

    # UTILITIES


    @staticmethod
    def _key_to_source(key: str) -> ExtractionSource:
        mapping = {
            "ocr_under_barcode": ExtractionSource.OCR_UNDER_BARCODE,
            "ocr_bottom": ExtractionSource.OCR_BOTTOM,
            "ocr_full": ExtractionSource.OCR_FULL,
            "ocr_heavy": ExtractionSource.OCR_HEAVY,
        }
        return mapping.get(key, ExtractionSource.OCR_FULL)

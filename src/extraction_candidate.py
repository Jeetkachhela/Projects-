# src/extraction_candidate.py
"""
Lightweight, structured representation of OCR extraction hypotheses.

Why this exists:
    - Each OCR/Barcode strategy produces a "guess"
    - Guesses are noisy, incomplete, and sometimes wrong
    - We capture each guess with metadata and score it
    - Final decision uses a weighted ranking, not first-match wins

This is intentionally simple, transparent, and tunable.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# EXTRACTION SOURCE ENUMS


class ExtractionSource(Enum):
    """
    Semantic label for where a candidate came from.

    This expresses both "how" and "where" the signal was extracted.
    Useful for:
        - weighting reliability
        - traceability/debugging
        - reporting in UI/API
    """

    BARCODE_0 = "barcode_0"       # primary barcode match
    BARCODE_1 = "barcode_1"       # secondary barcode match
    BARCODE_2 = "barcode_2"       # tertiary barcode match
    BARCODE_N = "barcode_n"       # Nth barcode match (fallback)

    OCR_UNDER_BARCODE = "ocr_under"   # OCR under lowest 1D barcode
    OCR_BOTTOM = "ocr_bottom"         # bottom half
    OCR_FULL = "ocr_full"             # full image (standard)
    OCR_HEAVY = "ocr_heavy"           # full image (aggressive)


class SourceReliability(Enum):
    """
    Empirical reliability of each source type, based on real-world behavior.

    These values are NOT confidences from OCR:
        - they encode prior knowledge
        - used to break ties
        - allow pipeline tuning without code changes
    """

    MAXIMUM = 1.00       # Barcodes are direct and authoritative
    VERY_HIGH = 0.97     # Under-barcode region often yields ID text
    HIGH = 0.92          # Bottom region is common but not consistent
    MEDIUM = 0.85        # Full-image OCR is weaker but valuable
    LOWER = 0.70         # Heavy preprocessing may distort text


# CANDIDATE REPRESENTATION

@dataclass
class ExtractionCandidate:
    """
    Encapsulates one extracted candidate.

    Attributes:
        matched_text:   Final parsed text (string that matched regex)
        confidence:     OCR model-reported confidence (0–1)
        source:         Which module produced this hypothesis
        angle:          Image rotation angle when extracted
        raw_text:       Raw OCR text before pattern filtering
        validity_score: Score for regex-level "goodness" (0–1)

    Design idea:
        We treat OCR outputs as noisy signals and filter them
        using heuristics, priors, and statistical scoring.
    """

    matched_text: str
    confidence: float
    source: ExtractionSource
    angle: int
    raw_text: str
    validity_score: float


    # RELIABILITY LOOK

    @property
    def source_reliability(self) -> float:
        """
        Map extraction source → reliability prior.

        Why not put this in the enum?
            - keeps weights editable without changing enum definitions
            - allows continuous tuning (e.g., A/B testing)
        """
        mapping = {
            ExtractionSource.BARCODE_0: SourceReliability.MAXIMUM.value,
            ExtractionSource.BARCODE_1: SourceReliability.MAXIMUM.value,
            ExtractionSource.BARCODE_2: SourceReliability.MAXIMUM.value,
            ExtractionSource.BARCODE_N: SourceReliability.MAXIMUM.value,

            ExtractionSource.OCR_UNDER_BARCODE: SourceReliability.VERY_HIGH.value,
            ExtractionSource.OCR_BOTTOM: SourceReliability.HIGH.value,
            ExtractionSource.OCR_FULL: SourceReliability.MEDIUM.value,
            ExtractionSource.OCR_HEAVY: SourceReliability.LOWER.value,
        }

        # Default to lower reliability if unknown source type appears
        return mapping.get(self.source, SourceReliability.LOWER.value)


    # COMPOSITE SCORING

    @property
    def composite_score(self) -> float:
        """
        Compute weighted score representing "how good this candidate is".

        We do NOT trust OCR confidence alone:
            - conf can be high on wrong text
            - conf can be low on correct text

        So we combine three signals:

            1. OCR confidence          (can it read pixels?)       40%
            2. Source reliability      (is this region reliable?)  40%
            3. Validity score          (does text look valid?)     20%

        This keeps logic interpretable while capturing useful dynamics.
        """

        # Safety clamps to avoid invalid floats
        conf = max(0.0, min(1.0, self.confidence))
        val = max(0.0, min(1.0, self.validity_score))
        rel = max(0.0, min(1.0, self.source_reliability))

        return (
            conf * 0.40 +
            rel * 0.40 +
            val * 0.20
        )


    # ORDERING SUPPORT

    def __lt__(self, other: "ExtractionCandidate") -> bool:
        """
        Sort candidates by composite score.

        Result:
            sorted(candidates) gives lowest→highest score
            sorted(candidates, reverse=True) gives best→worst
        """
        if not isinstance(other, ExtractionCandidate):
            return NotImplemented
        return self.composite_score < other.composite_score

    # DEBUGGING / LOGGING

    def __repr__(self) -> str:
        """
        Human-readable compact debug string for console/logs.
        """

        return (
            f"Candidate("
            f"text='{self.matched_text}', "
            f"score={self.composite_score:.3f}, "
            f"conf={self.confidence:.2f}, "
            f"val={self.validity_score:.2f}, "
            f"src={self.source.value}, "
            f"angle={self.angle}°"
            f")"
        )

    def pretty(self) -> str:
        """
        Verbose multi-line debug representation.

        Useful for:
            - troubleshooting
            - ML model introspection
            - dev-mode logging
        """
        return (
            f"Matched Text      : {self.matched_text}\n"
            f"Composite Score    : {self.composite_score:.3f}\n"
            f"OCR Confidence     : {self.confidence:.3f}\n"
            f"Validity Score     : {self.validity_score:.3f}\n"
            f"Source             : {self.source.value}\n"
            f"Source Reliability : {self.source_reliability:.3f}\n"
            f"Rotation Angle     : {self.angle}°\n"
            f"Raw OCR Text       : {self.raw_text[:200]}...\n"
        )

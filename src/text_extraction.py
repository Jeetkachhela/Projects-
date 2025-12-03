# src/text_extraction.py
"""
Optimized text extraction and pattern matching for ID extraction.
Identifies shipping label IDs with pattern: <14-30 digits>_1[_suffix]
"""

import os
import re
import logging
from dataclasses import dataclass
from typing import Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

# CONFIGURATION


LOG_DIR = "intermediate"
LOG_FILE = "extracted_ids.txt"

# Pattern: 14-30 digits, "_1", optional "_<suffix>"
# Examples:
#   161820476409495744_1
#   161827124813349056_1_bmd
TARGET_REGEX = re.compile(
    r"\d{14,30}_1(?:_[A-Za-z0-9]{1,10})?",
    re.IGNORECASE,
)

# Common OCR confusions (digit-lookalike letters)
OCR_SUBSTITUTIONS: Dict[str, str] = {
    "O": "0",  # Letter O → Zero
    "o": "0",  # Lowercase o → Zero
    "I": "1",  # Letter I → One
    "l": "1",  # Lowercase L → One
    "|": "1",  # Pipe → One
    "S": "5",  # Letter S → Five
    "B": "8",  # Letter B → Eight
    "Z": "2",  # Letter Z → Two
    "G": "6",  # Letter G → Six
}

# DATA CLASSES


@dataclass
class ExtractionResult:
    """Result of pattern extraction from text."""
    matched_text: str
    source: str
    confidence: float


# LOGGING HELPERS


def _ensure_log_dir() -> None:
    """Create logging directory if it doesn't exist."""
    if not os.path.exists(LOG_DIR):
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            logger.debug(f"Created log directory: {LOG_DIR}")
        except OSError as e:
            logger.error(f"Failed to create log directory: {e}")


def _log_extraction(result: ExtractionResult, raw_text: str) -> None:
    """
    Log successful extraction to file for audit trail.

    Args:
        result: Extraction result with matched text and confidence
        raw_text: Original OCR text for reference
    """
    _ensure_log_dir()
    path = os.path.join(LOG_DIR, LOG_FILE)

    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Truncate raw text to first 200 chars and normalize whitespace
        raw_snippet = " ".join(raw_text.split())[:200]

        line = (
            f"{ts} | {result.source} | "
            f"{result.confidence:.4f} | {result.matched_text} | "
            f"{raw_snippet}\n"
        )

        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

        logger.debug(f"Logged extraction: {result.matched_text}")

    except OSError as e:
        # Log but don't crash - logging should never break OCR
        logger.warning(f"Failed to log extraction: {e}")
    except Exception as e:
        # Catch all other exceptions
        logger.error(f"Unexpected error during logging: {e}", exc_info=True)



# TEXT CLEANING HELPERS


def _clean_basic(text: str) -> str:
    """
    Light, safe text cleaning.
    Preserves word characters and underscores (critical for pattern).

    Args:
        text: Raw OCR text

    Returns:
        Cleaned text with normalized whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    try:
        # Normalize line endings to spaces
        text = text.replace("\n", " ").replace("\r", " ")

        # Keep word chars (letters/digits/underscore) and spaces
        # Remove other punctuation that might interfere with digit detection
        text = re.sub(r"[^\w\s]", " ", text)

        # Normalize multiple spaces to single space
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    except Exception as e:
        logger.error(f"Error in _clean_basic: {e}", exc_info=True)
        return ""


def _clean_with_ocr_fixes(text: str) -> str:
    """
    Second-pass cleaning with common OCR character confusions.
    Applies substitution rules: O→0, I→1, S→5, etc.
    Preserves underscores (critical for pattern matching).

    Args:
        text: Text to clean

    Returns:
        Cleaned text with OCR corrections applied
    """
    if not text or not isinstance(text, str):
        return ""

    try:
        # First pass: basic cleaning
        text = _clean_basic(text)

        # Second pass: OCR character substitutions
        for letter, digit in OCR_SUBSTITUTIONS.items():
            text = text.replace(letter, digit)

        return text

    except Exception as e:
        logger.error(f"Error in _clean_with_ocr_fixes: {e}", exc_info=True)
        return ""


def _get_text_stats(text: str) -> Dict[str, int]:
    """
    Get statistics about text for debugging.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with text statistics
    """
    if not text:
        return {
            "length": 0,
            "digits": 0,
            "letters": 0,
            "underscores": 0,
            "spaces": 0,
        }

    return {
        "length": len(text),
        "digits": sum(1 for c in text if c.isdigit()),
        "letters": sum(1 for c in text if c.isalpha()),
        "underscores": sum(1 for c in text if c == "_"),
        "spaces": sum(1 for c in text if c.isspace()),
    }


# PATTERN MATCHING


def find_target_in_text(
    text: str,
    source: str,
    base_confidence: float,
) -> Optional[ExtractionResult]:
    """
    Extract ID pattern from arbitrary OCR text using multi-pass strategy.

    Algorithm:
      1. Basic cleaning (normalize spaces, remove special chars)
      2. Search for pattern in cleaned text
      3. Search in squashed text (no spaces - handles digit breaks)
      4. Repeat with OCR-fix cleaning (O→0, I→1, etc.)

    Confidence is reduced for each pass (lower priority).

    Args:
        text: Raw OCR text containing potential ID
        source: Source identifier (e.g., "barcode", "ocr_full")
        base_confidence: Initial confidence score

    Returns:
        ExtractionResult if found, None otherwise
    """
    if not text or not isinstance(text, str):
        return None

    try:
        logger.debug(f"Searching for target pattern in {source}")
        stats = _get_text_stats(text)
        logger.debug(f"Text stats: {stats}")


        # PASS 1: BASIC CLEANING


        cleaned1 = _clean_basic(text)
        squashed1 = cleaned1.replace(" ", "")

        # 1a) Search in normally cleaned text
        m = TARGET_REGEX.search(cleaned1)
        if m:
            matched = m.group(0).upper()
            logger.info(f"Found target in {source} (basic clean): {matched}")
            result = ExtractionResult(
                matched_text=matched,
                source=source,
                confidence=base_confidence,
            )
            _log_extraction(result, raw_text=text)
            return result

        # 1b) Search in squashed text (handles spaces in digits)
        m = TARGET_REGEX.search(squashed1)
        if m:
            matched = m.group(0).upper()
            logger.info(f"Found target in {source} (squashed): {matched}")
            result = ExtractionResult(
                matched_text=matched,
                source=f"{source}_squash",
                confidence=base_confidence * 0.97,  # Slightly lower confidence
            )
            _log_extraction(result, raw_text=text)
            return result


        # PASS 2: OCR-FIX CLEANING


        cleaned2 = _clean_with_ocr_fixes(text)
        squashed2 = cleaned2.replace(" ", "")

        # 2a) Search in OCR-fixed text
        m = TARGET_REGEX.search(cleaned2)
        if m:
            matched = m.group(0).upper()
            logger.info(f"Found target in {source} (OCR fix): {matched}")
            result = ExtractionResult(
                matched_text=matched,
                source=f"{source}_ocrfix",
                confidence=base_confidence * 0.95,  # Lower confidence
            )
            _log_extraction(result, raw_text=text)
            return result

        # 2b) Search in squashed OCR-fixed text
        m = TARGET_REGEX.search(squashed2)
        if m:
            matched = m.group(0).upper()
            logger.info(f"Found target in {source} (OCR fix + squash): {matched}")
            result = ExtractionResult(
                matched_text=matched,
                source=f"{source}_ocrfix_squash",
                confidence=base_confidence * 0.93,  # Lowest confidence
            )
            _log_extraction(result, raw_text=text)
            return result

        # No match found
        logger.debug(f"No target pattern found in {source}")
        return None

    except Exception as e:
        logger.error(f"Error in find_target_in_text: {e}", exc_info=True)
        return None


# TEXT HIGHLIGHTING


def highlight_match_in_text(
    text: str,
    result: Optional[ExtractionResult]
) -> str:
    """
    Highlight matched ID in original OCR text for visualization.

    Args:
        text: Original OCR text
        result: Extraction result with matched text

    Returns:
        Text with matched ID highlighted using **bold** markdown
    """
    if not result or not text:
        return text

    try:
        matched = result.matched_text

        # Try case-insensitive match and highlight
        pattern = re.escape(matched)
        highlighted, count = re.subn(
            pattern,
            f"**{matched}**",
            text,
            count=1,
            flags=re.IGNORECASE,
        )

        if count > 0:
            logger.debug(f"Highlighted match in text: {matched}")
            return highlighted

        # Fallback: append ID if not found in text (can happen with OCR fixes)
        logger.debug(f"Match not found in original text, using fallback")
        return f"{text}\n\n[ID: **{matched}**]"

    except Exception as e:
        logger.error(f"Error in highlight_match_in_text: {e}", exc_info=True)
        return text



# VALIDATION & DEBUGGING


def validate_extracted_id(matched_text: str) -> bool:
    """
    Validate extracted ID against expected pattern.

    Args:
        matched_text: Extracted text to validate

    Returns:
        True if valid, False otherwise
    """
    if not matched_text or not isinstance(matched_text, str):
        return False

    try:
        match = TARGET_REGEX.fullmatch(matched_text.upper())
        is_valid = match is not None

        if is_valid:
            logger.debug(f"ID validation passed: {matched_text}")
        else:
            logger.warning(f"ID validation failed: {matched_text}")

        return is_valid

    except Exception as e:
        logger.error(f"Error validating ID: {e}", exc_info=True)
        return False


def get_extraction_log_path() -> str:
    """
    Get path to extraction log file.

    Returns:
        Full path to log file
    """
    return os.path.join(LOG_DIR, LOG_FILE)


def clear_extraction_log() -> bool:
    """
    Clear extraction log file (useful for testing).

    Returns:
        True if successful, False otherwise
    """
    try:
        path = get_extraction_log_path()
        if os.path.exists(path):
            os.remove(path)
            logger.info("Cleared extraction log")
            return True
        return True
    except OSError as e:
        logger.error(f"Failed to clear extraction log: {e}")
        return False

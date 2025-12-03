# app.py
"""
Optimized Streamlit web application for shipping label OCR text extraction.
Provides interactive UI for uploading images and extracting IDs with _1_ pattern.
"""

import streamlit as st
import logging
import numpy as np
from typing import Optional, Dict, Any

from src.ocr_engine import process_image_bytes, output_to_dict
from src.utils import load_image_from_bytes, draw_barcode_boxes, get_image_stats
from src.preprocessing import (
    preprocess_for_barcode,
    preprocess_for_id_line,
    get_preprocessing_stats,
)
from src.text_extraction import validate_extracted_id


# LOGGING CONFIGURATION


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# STREAMLIT PAGE CONFIGURATION

st.set_page_config(
    page_title="Shipping Label OCR – _1_ Extractor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# CUSTOM CSS STYLING


st.markdown("""
    <style>
        /* Main title styling */
        .title {
            font-size: 42px;
            font-weight: bold;
            color: #1E90FF;
            margin-bottom: 10px;
        }

        /* Success box for positive results */
        .success-box {
            padding: 1.5rem;
            border-radius: 10px;
            background: #d4edda;
            border: 2px solid #28a745;
            box-shadow: 0 2px 4px rgba(40, 167, 69, 0.2);
        }

        /* Warning box for no match */
        .warning-box {
            padding: 1.5rem;
            border-radius: 10px;
            background: #fff3cd;
            border: 2px solid #ffc107;
            box-shadow: 0 2px 4px rgba(255, 193, 7, 0.2);
        }

        /* Error box for errors */
        .error-box {
            padding: 1.5rem;
            border-radius: 10px;
            background: #f8d7da;
            border: 2px solid #dc3545;
            box-shadow: 0 2px 4px rgba(220, 53, 69, 0.2);
        }

        /* Info box styling */
        .info-box {
            padding: 1rem;
            border-radius: 8px;
            background: #e7f3ff;
            border-left: 4px solid #1E90FF;
        }

        /* Footer styling */
        .footer {
            text-align: center;
            color: #666;
            padding: 20px;
            font-size: 13px;
            border-top: 1px solid #ddd;
            margin-top: 40px;
        }

        /* Monospace for IDs */
        .id-text {
            font-family: 'Courier New', monospace;
            font-size: 24px;
            font-weight: bold;
            letter-spacing: 1px;
        }

        /* Stats table styling */
        .stats-table {
            font-size: 13px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# HELPER FUNCTIONS


@st.cache_data
def get_app_info() -> Dict[str, str]:
    """Get application metadata."""
    return {
        "title": "Shipping Label OCR – _1_ Extractor",
        "description": "High-accuracy open-source solution using PaddleOCR + Smart Region Extraction",
        "version": "1.0.0",
        "model": "PaddleOCR",
    }


def display_success_result(out: Dict[str, Any]) -> None:
    """Display successful extraction result."""
    confidence_pct = out["confidence"] * 100

    st.markdown(f"""
    <div class="success-box">
        <h2>✅ Extracted Successfully!</h2>
        <p class="id-text" style="color: #28a745;">
            {out["matched_text"]}
        </p>
        <p>
            <strong>📍 Source:</strong> {out["source"].replace('_', ' ').title()}
            &nbsp; | &nbsp;
            <strong>🎯 Confidence:</strong> {confidence_pct:.1f}%
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_no_match_result() -> None:
    """Display no match found result."""
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ No matching ID found</h3>
        <p>No text containing <code>_1_</code> pattern was detected in the image.</p>
        <p style="font-size: 12px; color: #666;">
            Expected format: <code>163233702292313922_1_lWV</code>
            (14-30 digits + _1 + optional suffix)
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_raw_ocr_text(raw_text: str) -> None:
    """Display raw OCR text in collapsible section."""
    st.subheader("📝 Raw OCR Text")

    if raw_text.strip():
        # Display with line numbers
        lines = raw_text.strip().split("\n")
        with st.container():
            st.code(raw_text, language="text")
            st.caption(f"📊 {len(lines)} lines detected")
    else:
        st.info("No OCR text extracted (pattern matched directly from barcode)")


def display_highlighted_match(highlighted_text: str, matched_text: Optional[str]) -> None:
    """Display highlighted match in original text."""
    st.subheader("🎨 Match Highlighted")

    if highlighted_text:
        # Convert markdown bold to HTML for better visibility
        formatted = highlighted_text.replace("\n", "  \n")
        st.markdown(formatted)
    else:
        if matched_text:
            st.info(f"Pattern `{matched_text}` detected but not found in preview text")
        else:
            st.info("No highlighted text available")


def display_barcodes(img: np.ndarray, barcodes: list) -> None:
    """Display detected barcodes with visualization."""
    st.subheader("📊 Detected Barcodes")

    if not barcodes:
        st.info("No 1D barcodes detected in the image")
        return

    try:
        # Draw barcodes on image
        boxed = draw_barcode_boxes(img, barcodes)
        st.image(boxed, caption=f"{len(barcodes)} barcode(s) located", use_container_width=True)

        # List barcode details
        st.write(f"**Found {len(barcodes)} barcode(s):**")
        for i, b in enumerate(barcodes, 1):
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.code(b['value'], language="text")
            with col2:
                st.caption(f"Type: **{b['symbology']}**")
            with col3:
                st.caption(f"Pos: {b['rect']}")

    except Exception as e:
        logger.error(f"Error displaying barcodes: {e}", exc_info=True)
        st.error(f"Error displaying barcodes: {str(e)}")


def display_errors(errors: list) -> None:
    """Display processing errors."""
    if not errors:
        return

    st.subheader("⚠️ Processing Errors & Warnings")

    for i, error in enumerate(errors, 1):
        with st.container():
            st.warning(f"**{i}.** {error}")


def display_image_stats(img: np.ndarray) -> None:
    """Display image statistics for debugging."""
    stats = get_image_stats(img)

    if not stats:
        st.warning("Could not calculate image statistics")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Shape", str(stats.get("shape")))

    with col2:
        st.metric("Mean Brightness", f"{stats.get('mean', 0):.1f}")

    with col3:
        st.metric("Std Dev (Contrast)", f"{stats.get('std', 0):.1f}")

    with col4:
        st.metric("Channels", stats.get("channels", "?"))


def display_preprocessing_previews(img: np.ndarray) -> None:
    """Display preprocessing visualization."""
    try:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🔍 Barcode-Optimized Preprocessing**")
            img_barcode = preprocess_for_barcode(img)
            st.image(img_barcode, caption="Barcode detection prep", use_container_width=True)

            barcode_stats = get_preprocessing_stats(img_barcode)
            if barcode_stats:
                st.caption(
                    f"Mean: {barcode_stats.get('mean', 0):.1f} | "
                    f"Std: {barcode_stats.get('std', 0):.1f}"
                )

        with col2:
            st.markdown("**📝 ID-Line-Optimized Preprocessing**")
            img_line = preprocess_for_id_line(img)
            st.image(img_line, caption="ID line detection prep", use_container_width=True)

            line_stats = get_preprocessing_stats(img_line)
            if line_stats:
                st.caption(
                    f"Mean: {line_stats.get('mean', 0):.1f} | "
                    f"Std: {line_stats.get('std', 0):.1f}"
                )

    except Exception as e:
        logger.error(f"Error displaying preprocessing previews: {e}", exc_info=True)
        st.error(f"Error in preprocessing preview: {str(e)}")



# MAIN APPLICATION


def main():
    """Main Streamlit application."""

    # Header
    st.markdown("<h1 class='title'>📦 Shipping Label OCR</h1>", unsafe_allow_html=True)
    st.markdown(
        "### Extract IDs containing `_1_` pattern (e.g. `163233702292313922_1_lWV`)"
    )

    st.write(
        "High-accuracy open-source solution using "
        "**PaddleOCR** + Smart Region Extraction"
    )

    st.write("---")

    # File Upload Section
    st.subheader("📤 Upload Shipping Label")

    uploaded_file = st.file_uploader(
        "Upload a shipping label image",
        type=["png", "jpg", "jpeg", "webp"],
        help="Supported formats: PNG, JPG, JPEG, WEBP (max 200MB)",
        label_visibility="collapsed"
    )

    # No File Uploaded
    if uploaded_file is None:
        st.info("👆 Please upload a shipping label image to begin extraction")

        st.markdown("""
        **📋 Expected ID Format:**
        - Minimum: 14 digits + `_1` (e.g., `12345678901234_1`)
        - Maximum: 30 digits + `_1` + suffix (e.g., `123456789012345678901234_1_ABC`)

        **✨ Features:**
        - Multi-angle rotation processing (0°, 90°, 180°, 270°)
        - Barcode value extraction
        - OCR with smart region detection
        - OCR character confusion correction (O→0, I→1, etc.)
        - Confidence scoring
        """)

        st.write("---")
        st.markdown("""
        <div class="footer">
            Built with <strong>PaddleOCR</strong> • Open-source • No external APIs • v1.0.0
        </div>
        """, unsafe_allow_html=True)

        return

    # Load and Display Image
    try:
        image_bytes = uploaded_file.getvalue()
        img_np = load_image_from_bytes(image_bytes)

        logger.info(f"Loaded image: {img_np.shape}")

    except Exception as e:
        logger.error(f"Error loading image: {e}", exc_info=True)
        st.error(f"❌ Failed to load image: {str(e)}")
        return

    # Display uploaded image
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(img_np, caption="Uploaded Shipping Label", use_container_width=True)

    with col2:
        st.subheader("📊 Image Info")
        display_image_stats(img_np)

    st.write("---")

    # Preprocessing Preview (Expandable)
    with st.expander("🔬 Show Preprocessing Previews", expanded=False):
        display_preprocessing_previews(img_np)

    st.write("---")

    # Extraction Button
    if st.button("🚀 Run Extraction", type="primary", use_container_width=True):

        with st.spinner("⏳ Processing image... (this may take 10-30 seconds)"):
            try:
                logger.info("Starting OCR extraction")
                result = process_image_bytes(image_bytes)
                out = output_to_dict(result)
                logger.info(f"Extraction complete: {out['matched_text']}")

            except Exception as e:
                logger.error(f"Error during extraction: {e}", exc_info=True)
                st.error(f"❌ Error during extraction: {str(e)}")
                return

        st.write("---")

        # Results Display
        if out["matched_text"]:
            display_success_result(out)

            # Validate extracted ID
            if not validate_extracted_id(out["matched_text"]):
                st.warning("⚠️ Extracted text format validation failed")

        else:
            display_no_match_result()

        st.write("---")

        # Advanced Debug Section
        with st.expander("🔧 Advanced Debug & Details", expanded=False):

            tab1, tab2, tab3 = st.tabs(["OCR Text", "Barcodes", "Errors"])

            with tab1:
                display_raw_ocr_text(out["raw_text_ocr"])
                st.write("---")
                display_highlighted_match(out["highlighted_text"], out["matched_text"])

            with tab2:
                display_barcodes(img_np, out["barcodes"])

            with tab3:
                if out["errors"]:
                    display_errors(out["errors"])
                else:
                    st.success("✅ No errors during processing")

        st.write("---")

        # Export Results
        if out["matched_text"]:
            st.subheader("💾 Export Results")

            col1, col2 = st.columns(2)

            with col1:
                # Copy button (text)
                st.code(out["matched_text"], language="text")

            with col2:
                # JSON download
                import json
                json_str = json.dumps(out, indent=2)
                st.download_button(
                    "📥 Download JSON",
                    json_str,
                    file_name="ocr_result.json",
                    mime="application/json"
                )

    # Footer
    st.write("---")
    st.markdown("""
    <div class="footer">
        <strong>🎯 Accuracy Improvements:</strong>
        Multi-angle OCR • Barcode integration • Character confusion correction
        <br>
        Built with <strong>PaddleOCR</strong> • Open-source • No external APIs • v1.0.0
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

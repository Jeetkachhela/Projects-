Shipping Label OCR – 1 Pattern Extraction

An AI/ML system that extracts shipping label identifiers containing the pattern:

<14–30 digits>\_1[_suffix]
Example: 163233702292313922_1_lWV

The system is engineered to handle real-world waybill images with:

Distorted text

Variable brightness

Orientation variance

Missing barcodes

Noise, blur, and low contrast

This solution demonstrates robust OCR engineering, modular code, and a clean UI.

Features
Core Capabilities

✔ Multi-angle OCR (0°, 90°, 180°, 270°)
✔ Barcode decoding + fallback OCR
✔ Advanced preprocessing (CLAHE, denoising, morphology)
✔ Pattern-aware candidate selection
✔ Confidence-weighted ranking engine
✔ Real-time visualization via Streamlit
✔ Full audit trail (errors, barcodes, raw OCR)
✔ Offline, open-source–only, no commercial APIs

Accuracy

Achieved on test dataset: 83.9% ID extraction accuracy
(>75% required threshold)

Technical Approach

1. OCR Engine

Backend uses PaddleOCR because it delivers:

High accuracy on structured text

GPU-optional inference

Fast, offline performance

OCR strategy:

Under-barcode region OCR

Bottom region OCR

Full page OCR

Heavy adaptive threshold fallback

Each result is cleaned, ranked, and validated.

2. Image Preprocessing (Computer Vision)

Preprocessing optimized for:

A. Barcode Detection

Bilateral denoise

CLAHE contrast boost

Unsharp masking

Normalization

B. OCR Blocks

Non-local means denoising

CLAHE

OTSU threshold

Opening + closing

C. ID Line Regions

Gentle denoising

Horizontal/vertical morphology

Design goals:

Normalize low-quality scans

Enhance text edges

Avoid over-thresholding (loss of information)

3. Pattern Extraction

Regex enforced pattern:

\d{14,30}_1(_[A-Za-z0-9]{1,10})?

Pipeline:

Basic cleaning

OCR character correction

Space-squash recovery

Soft confidence decay

Common OCR confusions:

Letter Replaced with
O/o 0
I/l/
S 5
B 8
Z 2
G 6 4. Multi-Source Ranking Engine

Candidates are ranked using:

score =
0.4 _ OCR confidence +
0.4 _ source reliability +
0.2 \* validity score

Source reliability tiers:

Source-Reliability
Barcode-1.0
Under barcode OCR-0.99
Bottom region OCR-0.95
Full image OCR-0.85
Heavy thresholding-0.75

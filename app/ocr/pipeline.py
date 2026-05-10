"""
Shared OCR pipeline function.

Used by:
  - OCRService (FastAPI context) — model injected from ModelRegistry
  - ocr_document_task (RQ worker context) — model injected from module-level lazy singleton

Centralising the pipeline here eliminates duplication between the two execution contexts.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.ocr.extractor import detect_document_type, extract_fields
from app.ocr.preprocessing import QualityReport, decode_image, preprocess, validate_quality
from app.schemas.ocr import (
    DocumentOCRResult,
    DocumentType,
    ExtractedFields,
    ImageQualityReport,
    TextBlock,
)


def run_pipeline(
    image_bytes: bytes,
    model: Any,                             # PaddleOCR instance
    doc_type_hint: DocumentType | None = None,
    skip_preprocessing: bool = False,
) -> DocumentOCRResult:
    """
    Full OCR pipeline:
      decode → validate_quality → [preprocess] → OCR → detect doc type → extract fields

    Quality failures are reported (not raised) so callers can decide whether to
    reject or proceed — important for the research evaluation.

    Args:
        image_bytes: Raw bytes of the uploaded image.
        model: A loaded PaddleOCR instance.
        doc_type_hint: Optional override for document type detection.
        skip_preprocessing: Set True for already-enhanced images.
    """
    # 1. Decode
    img = decode_image(image_bytes)

    # 2. Quality validation (before preprocessing so we measure original quality)
    quality = validate_quality(img)

    # 3. Preprocessing
    if not skip_preprocessing:
        img = preprocess(img)

    # 4. PaddleOCR inference
    raw_result = model.ocr(img, cls=True)
    blocks = _parse_blocks(raw_result)

    # 5. Document type detection
    blocks_dicts = [{"text": b.text, "confidence": b.confidence} for b in blocks]
    doc_type = doc_type_hint or detect_document_type(blocks_dicts)

    # 6. Field extraction
    fields = extract_fields(blocks_dicts, doc_type)

    # 7. Aggregates
    full_text = " ".join(b.text for b in blocks)
    avg_conf = round(sum(b.confidence for b in blocks) / len(blocks), 4) if blocks else None

    return DocumentOCRResult(
        quality=_quality_to_schema(quality),
        document_type=doc_type,
        fields=fields,
        raw_blocks=blocks,
        full_text=full_text,
        average_confidence=avg_conf,
        preprocessed=not skip_preprocessing,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_blocks(raw: Any) -> list[TextBlock]:
    if not raw or not raw[0]:
        return []
    return [
        TextBlock(text=line[1][0], confidence=round(float(line[1][1]), 4), box=line[0])
        for line in raw[0]
        if line[1][1] > 0.30  # discard very low-confidence detections
    ]


def _quality_to_schema(q: QualityReport) -> ImageQualityReport:
    return ImageQualityReport(
        passed=q.passed,
        resolution_ok=q.resolution_ok,
        sharpness=q.sharpness,
        brightness=q.brightness,
        contrast=q.contrast,
        issues=q.issues,
    )

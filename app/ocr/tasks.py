"""
RQ task for background OCR execution.

This module runs inside the RQ worker process — there is no FastAPI app,
no lifespan, and no ModelRegistry. PaddleOCR is loaded lazily into a
module-level singleton on first job execution and reused for all subsequent
jobs handled by that worker process.

The task returns a plain dict (result of DocumentOCRResult.model_dump()) so
that RQ can serialise it with pickle and the FastAPI polling endpoint can
reconstruct the Pydantic model.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_ocr_model: Any = None


def _get_model() -> Any:
    global _ocr_model
    if _ocr_model is None:
        from paddleocr import PaddleOCR

        from app.core.config import settings

        logger.info("Worker: loading PaddleOCR (first job in this process)")
        _ocr_model = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=settings.use_gpu,
            show_log=False,
        )
        logger.info("Worker: PaddleOCR ready")
    return _ocr_model


def ocr_document_task(
    image_bytes: bytes,
    doc_type_hint: str | None = None,
) -> dict:
    """
    Synchronous OCR pipeline for RQ execution.

    Args:
        image_bytes: Raw bytes of the uploaded document image.
        doc_type_hint: Optional 'passport' | 'nic' override. If None, auto-detected.

    Returns:
        dict representation of DocumentOCRResult (use model_validate to reconstruct).
    """
    from app.ocr.pipeline import run_pipeline
    from app.schemas.ocr import DocumentType

    hint: DocumentType | None = None
    if doc_type_hint:
        try:
            hint = DocumentType(doc_type_hint)
        except ValueError:
            logger.warning("Worker: unknown doc_type_hint '%s', ignoring", doc_type_hint)

    result = run_pipeline(image_bytes, _get_model(), doc_type_hint=hint)
    return result.model_dump()

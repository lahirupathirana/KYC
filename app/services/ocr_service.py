"""
OCRService — async wrapper around the shared OCR pipeline.

Responsibilities:
  - Run the pipeline (preprocessing + PaddleOCR + extraction) in a ThreadPoolExecutor
    so it does not block the FastAPI event loop.
  - Enqueue background jobs via Redis/RQ for callers that want async processing.
  - Quick quality pre-check before enqueueing (fails fast on decode errors).
"""

from __future__ import annotations

import asyncio

import structlog

from app.models.model_registry import ModelRegistry
from app.ocr.pipeline import run_pipeline
from app.ocr.preprocessing import decode_image, validate_quality
from app.schemas.ocr import (
    DocumentOCRResult,
    DocumentType,
    ImageQualityReport,
    OCRJobStatus,
    OCRJobSubmitted,
)
from app.workers.redis_worker import enqueue_task, get_job_status

logger = structlog.get_logger()


class OCRService:
    def __init__(self, registry: ModelRegistry) -> None:
        self._model = registry.get("ocr")

    # ── Synchronous (blocking inference, async-safe via executor) ─────────────

    async def extract_document(
        self,
        image_bytes: bytes,
        doc_type_hint: DocumentType | None = None,
    ) -> DocumentOCRResult:
        """
        Full pipeline: decode → validate → preprocess → OCR → parse fields.
        Returns structured result including quality report and extracted fields.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, run_pipeline, image_bytes, self._model, doc_type_hint
        )
        logger.info(
            "OCR complete",
            doc_type=result.document_type.value,
            quality_ok=result.quality.passed,
            fields_confidence=result.fields.extraction_confidence,
            blocks=len(result.raw_blocks),
        )
        return result

    # ── Asynchronous (enqueue to RQ, return job ID immediately) ──────────────

    async def enqueue_extraction(
        self,
        image_bytes: bytes,
        doc_type_hint: DocumentType | None = None,
        request_base_url: str = "",
    ) -> OCRJobSubmitted:
        """
        Fast quality pre-check, then enqueue the full pipeline to a Redis worker.
        Returns a job ID the client can poll.

        Pre-check runs synchronously in the executor (fast — no ML inference):
          - Decode to verify the file is a valid image
          - Validate resolution / brightness / sharpness

        If the image can't even be decoded, raises ValueError immediately (422)
        rather than wasting a queue slot.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._quick_quality_check, image_bytes)

        from app.ocr.tasks import ocr_document_task

        job_id = enqueue_task(
            ocr_document_task,
            image_bytes,
            doc_type_hint.value if doc_type_hint else None,
            timeout=120,
        )
        poll_url = f"{request_base_url}/api/v1/ocr/jobs/{job_id}"
        logger.info("OCR job enqueued", job_id=job_id)
        return OCRJobSubmitted(job_id=job_id, poll_url=poll_url)

    async def get_job_result(self, job_id: str) -> OCRJobStatus:
        """Poll Redis for job status and deserialise the result if finished."""
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, get_job_status, job_id)

        result: DocumentOCRResult | None = None
        if raw["status"] == "finished" and raw["result"] is not None:
            result = DocumentOCRResult.model_validate(raw["result"])

        return OCRJobStatus(
            job_id=raw["id"],
            status=raw["status"],
            result=result,
            error=raw.get("error"),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _quick_quality_check(self, image_bytes: bytes) -> None:
        """Decode + validate — raises ValueError on fatal problems only."""
        img = decode_image(image_bytes)  # raises ValueError if corrupt
        report = validate_quality(img)
        if not report.resolution_ok:
            raise ValueError(
                f"Image resolution too low: {report.issues[0]}. "
                "Please upload a clearer photo of the document."
            )

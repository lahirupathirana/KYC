from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.dependencies import get_ocr_service
from app.ocr.evaluation import (
    EvaluationReport,
    InferenceTimer,
    aggregate_results,
    evaluate_sample,
)
from app.schemas.ocr import (
    DocumentOCRResult,
    DocumentType,
    OCREvaluationReport,
    OCRJobStatus,
    OCRJobSubmitted,
)
from app.services.ocr_service import OCRService

router = APIRouter()

_DOC_TYPE_DESCRIPTION = (
    "Optional hint to skip auto-detection: 'passport', 'nic', or 'driving_license'"
)


@router.post(
    "/extract",
    response_model=DocumentOCRResult,
    summary="Synchronous OCR — extract structured fields from an ID document",
    description=(
        "Runs the full pipeline synchronously: decode → quality check → "
        "preprocess → PaddleOCR → detect document type → extract name/DOB/number. "
        "Supports NIC, Passport, and Driving License. "
        "Use `/extract/async` for large images or when you need immediate HTTP response."
    ),
)
async def extract_document(
    file: UploadFile = File(..., description="ID document image (JPEG, PNG, WebP, BMP)"),
    doc_type: DocumentType | None = Query(
        default=None,
        description=_DOC_TYPE_DESCRIPTION,
    ),
    service: OCRService = Depends(get_ocr_service),
) -> DocumentOCRResult:
    image_bytes = await file.read()
    return await service.extract_document(image_bytes, doc_type_hint=doc_type)


@router.post(
    "/extract/async",
    response_model=OCRJobSubmitted,
    status_code=202,
    summary="Asynchronous OCR — enqueue job and return immediately",
    description=(
        "Performs a fast quality pre-check (decode + resolution), then enqueues the "
        "full OCR pipeline to a Redis/RQ worker. Returns a job ID to poll. "
        "Preferred for WebRTC frame batches or when the HTTP timeout is a concern."
    ),
)
async def extract_document_async(
    request: Request,
    file: UploadFile = File(..., description="ID document image"),
    doc_type: DocumentType | None = Query(default=None, description=_DOC_TYPE_DESCRIPTION),
    service: OCRService = Depends(get_ocr_service),
) -> OCRJobSubmitted:
    image_bytes = await file.read()
    base_url = str(request.base_url).rstrip("/")
    return await service.enqueue_extraction(image_bytes, doc_type_hint=doc_type, request_base_url=base_url)


@router.get(
    "/jobs/{job_id}",
    response_model=OCRJobStatus,
    summary="Poll an async OCR job",
    description=(
        "Returns current status: queued | started | finished | failed. "
        "Poll until status == 'finished', then read `result` for the DocumentOCRResult."
    ),
)
async def get_job_status(
    job_id: str,
    service: OCRService = Depends(get_ocr_service),
) -> OCRJobStatus:
    return await service.get_job_result(job_id)


@router.post(
    "/evaluate",
    response_model=OCREvaluationReport,
    summary="Research: evaluate OCR output against ground truth",
    description=(
        "Compare a DocumentOCRResult (the system's prediction) against a ground-truth "
        "record and return CER, WER, and field-level precision/recall/F1 metrics. "
        "Intended for research evaluation pipelines, not production use. "
        "Pass `predicted_result` as JSON body alongside `ground_truth` fields."
    ),
)
async def evaluate_ocr(
    predicted_doc_type: str = Query(..., description="Predicted document type string"),
    ground_truth_doc_type: str = Query(..., description="Ground truth document type string"),
    predicted_number: str | None = Query(default=None),
    predicted_name: str | None = Query(default=None),
    predicted_dob: str | None = Query(default=None),
    predicted_expiry: str | None = Query(default=None),
    gt_number: str | None = Query(default=None),
    gt_name: str | None = Query(default=None),
    gt_dob: str | None = Query(default=None),
    gt_expiry: str | None = Query(default=None),
) -> OCREvaluationReport:
    predicted = {
        "document_number": predicted_number,
        "full_name": predicted_name,
        "dob": predicted_dob,
        "expiry_date": predicted_expiry,
    }
    ground_truth = {
        "document_number": gt_number,
        "full_name": gt_name,
        "dob": gt_dob,
        "expiry_date": gt_expiry,
    }

    sample = evaluate_sample(predicted, ground_truth, predicted_doc_type, ground_truth_doc_type)
    report = aggregate_results([sample], inference_times_ms=[0.0])
    summary = report.summary()

    return OCREvaluationReport(
        total_samples=summary["total_samples"],
        document_type_accuracy=summary["document_type_accuracy"],
        avg_cer=summary["avg_cer"],
        avg_wer=summary["avg_wer"],
        avg_inference_ms=summary["avg_inference_ms"],
        std_inference_ms=summary["std_inference_ms"],
        fields=summary["fields"],
    )

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    PASSPORT = "passport"
    NIC = "nic"
    DRIVING_LICENSE = "driving_license"
    UNKNOWN = "unknown"


class ImageQualityReport(BaseModel):
    passed: bool
    resolution_ok: bool
    sharpness: float = Field(description="Laplacian variance — higher = sharper")
    brightness: float = Field(description="Mean gray intensity (0–255)")
    contrast: float = Field(description="Gray std-dev — lower = flat/washed-out")
    issues: list[str] = Field(default_factory=list)


class TextBlock(BaseModel):
    text: str
    confidence: float
    box: list[list[float]]


class ExtractedFields(BaseModel):
    document_type: DocumentType
    document_number: str | None = Field(default=None, description="Passport, NIC, or driving license number")
    full_name: str | None = None
    dob: str | None = Field(default=None, description="Date of birth — ISO 8601 (YYYY-MM-DD)")
    sex: str | None = Field(default=None, description="'M' or 'F'")
    nationality: str | None = Field(default=None, description="3-letter ISO country code (passports only)")
    expiry_date: str | None = Field(default=None, description="Expiry date — ISO 8601")
    issue_date: str | None = Field(default=None, description="Issue date — ISO 8601 (driving licenses)")
    address: str | None = Field(default=None, description="Residential address (driving licenses)")
    vehicle_categories: list[str] = Field(
        default_factory=list,
        description="Authorised vehicle categories (driving licenses only, e.g. ['A', 'B', 'C'])",
    )
    extraction_confidence: float = Field(
        default=0.0,
        description="0–1 score: fraction of core fields (number, name, dob) successfully extracted",
    )
    mrz_parsed: bool = Field(
        default=False,
        description="True if fields were parsed from MRZ lines (reliable); False if from heuristics",
    )


class OCREvaluationResult(BaseModel):
    """Single-sample evaluation output for research reporting."""
    type_correct: bool
    avg_cer: float
    avg_wer: float
    fields: dict[str, Any] = Field(default_factory=dict)


class OCREvaluationReport(BaseModel):
    """Aggregate evaluation report over a dataset split."""
    total_samples: int
    document_type_accuracy: float
    avg_cer: float
    avg_wer: float
    avg_inference_ms: float
    std_inference_ms: float
    fields: dict[str, Any] = Field(default_factory=dict)


class DocumentOCRResult(BaseModel):
    quality: ImageQualityReport
    document_type: DocumentType
    fields: ExtractedFields
    raw_blocks: list[TextBlock] = Field(description="All OCR text blocks above confidence threshold")
    full_text: str = Field(description="All detected text concatenated")
    average_confidence: float | None = None
    preprocessed: bool = True


class OCRJobSubmitted(BaseModel):
    job_id: str
    status: str = "queued"
    poll_url: str = Field(description="Relative URL to poll for the result")


class OCRJobStatus(BaseModel):
    job_id: str
    status: str = Field(description="queued | started | finished | failed")
    result: DocumentOCRResult | None = None
    error: str | None = None

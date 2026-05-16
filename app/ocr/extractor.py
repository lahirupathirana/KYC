"""
Document-type detection and field-extraction orchestrator.

Accepts raw OCR text blocks (list of dicts with 'text' and 'confidence'),
detects whether the document is a passport, NIC, or driving license, routes
to the appropriate parser, and returns a unified ExtractedFields Pydantic model.
"""

from __future__ import annotations

import re

from app.ocr.driving_license_parser import DrivingLicenseFields
from app.ocr.driving_license_parser import parse as parse_dl
from app.ocr.nic_parser import NICFields
from app.ocr.nic_parser import parse as parse_nic
from app.ocr.passport_parser import PassportFields
from app.ocr.passport_parser import parse as parse_passport
from app.schemas.ocr import DocumentType, ExtractedFields

# ── Keyword sets for explicit type detection ──────────────────────────────────

_PASSPORT_KW = {"PASSPORT", "REPUBLIC OF", "NATIONALITY", "TRAVEL DOCUMENT", "SURNAME"}
_NIC_KW = {
    "NATIONAL IDENTITY", "IDENTITY CARD", "NIC", "NICS",
    "SRI LANKA", "DEMOCRATIC SOCIALIST",
}
_DL_KW = {
    "DRIVING LICENCE", "DRIVING LICENSE", "MOTOR TRAFFIC",
    "DEPARTMENT OF MOTOR", "DMT", "VEHICLE CATEGORY",
    "LICENCE HOLDER", "LICENSE HOLDER",
}
_MRZ_LINE_RE = re.compile(r"^[A-Z0-9<]{30,}$")
_OLD_NIC_RE = re.compile(r"\b\d{9}[VvXx]\b")
_NEW_NIC_RE = re.compile(r"\b\d{12}\b")


# ── Public API ────────────────────────────────────────────────────────────────

def detect_document_type(text_blocks: list[dict]) -> DocumentType:
    all_text = " ".join(b["text"] for b in text_blocks).upper()

    # 1. Explicit keyword match (most reliable — order matters: DL before NIC
    #    because some DLs also contain "SRI LANKA")
    if any(kw in all_text for kw in _DL_KW):
        return DocumentType.DRIVING_LICENSE
    if any(kw in all_text for kw in _PASSPORT_KW):
        return DocumentType.PASSPORT
    if any(kw in all_text for kw in _NIC_KW):
        return DocumentType.NIC

    # 2. Structural: two MRZ-format lines → passport
    mrz_candidates = [
        b["text"].upper().replace(" ", "<")
        for b in text_blocks
        if _MRZ_LINE_RE.match(b["text"].upper().replace(" ", "<")) and "<" in b["text"]
    ]
    if len(mrz_candidates) >= 2:
        return DocumentType.PASSPORT

    # 3. NIC number pattern
    if _OLD_NIC_RE.search(all_text) or _NEW_NIC_RE.search(all_text):
        return DocumentType.NIC

    return DocumentType.UNKNOWN


def extract_fields(text_blocks: list[dict], document_type: DocumentType) -> ExtractedFields:
    """
    Route to the appropriate parser and map the result to the unified schema.
    Always returns an ExtractedFields object — extraction_confidence=0.0 on failure.
    """
    if document_type == DocumentType.PASSPORT:
        result: PassportFields | None = parse_passport(text_blocks)
        if result:
            return ExtractedFields(
                document_type=document_type,
                document_number=result.document_number,
                full_name=result.full_name,
                dob=result.dob,
                sex=result.sex,
                nationality=result.nationality,
                expiry_date=result.expiry_date,
                extraction_confidence=result.confidence,
                mrz_parsed=result.mrz_parsed,
            )

    elif document_type == DocumentType.NIC:
        nic_result: NICFields | None = parse_nic(text_blocks)
        if nic_result:
            return ExtractedFields(
                document_type=document_type,
                document_number=nic_result.document_number,
                full_name=nic_result.full_name,
                dob=nic_result.dob,
                sex=nic_result.sex,
                extraction_confidence=nic_result.confidence,
                mrz_parsed=False,
            )

    elif document_type == DocumentType.DRIVING_LICENSE:
        dl_result: DrivingLicenseFields | None = parse_dl(text_blocks)
        if dl_result:
            return ExtractedFields(
                document_type=document_type,
                document_number=dl_result.document_number,
                full_name=dl_result.full_name,
                dob=dl_result.dob,
                expiry_date=dl_result.expiry_date,
                issue_date=dl_result.issue_date,
                address=dl_result.address,
                vehicle_categories=dl_result.vehicle_categories,
                extraction_confidence=dl_result.confidence,
                mrz_parsed=False,
            )

    return ExtractedFields(document_type=document_type, extraction_confidence=0.0)

"""
Integration tests for the OCR API endpoints.

PaddleOCR is mocked — no GPU required.
The mock returns a realistic raw PaddleOCR output structure so the full
parsing and schema path (pipeline → extractor → nic_parser) is exercised.
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.main import create_app


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_png_bytes(h: int = 600, w: int = 900) -> bytes:
    """Generate a valid PNG image for upload."""
    import cv2

    rng = np.random.default_rng(0)
    img = rng.integers(80, 180, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _make_mock_ocr_output(lines: list[str]) -> list:
    """Fake PaddleOCR raw output: list of [[box, (text, conf)], ...]"""
    result = []
    for i, text in enumerate(lines):
        box = [[0, i * 30], [200, i * 30], [200, (i + 1) * 30], [0, (i + 1) * 30]]
        result.append([box, (text, 0.95)])
    return [result]


@asynccontextmanager
async def _mock_lifespan(app: FastAPI):
    mock_registry = MagicMock()
    mock_ocr = MagicMock()
    mock_registry.get.return_value = mock_ocr
    app.state.model_registry = mock_registry
    yield


@pytest.fixture
async def client():
    app = create_app(lifespan_handler=_mock_lifespan)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
def nic_ocr_output():
    return _make_mock_ocr_output([
        "DEMOCRATIC SOCIALIST REPUBLIC OF SRI LANKA",
        "NATIONAL IDENTITY CARD",
        "NAME",
        "KAMAL BANDARA PERERA",
        "890123456V",
        "Date of Birth  12/01/1989",
    ])


@pytest.fixture
def passport_ocr_output():
    line1 = "P<LKAPERERA<<KAMAL<BANDARA<<<<<<<<<<<<<<<<<<"[:44].ljust(44, "<")
    line2 = "A1234567<8LKA8901124M3012317<<<<<<<<<<<<<<<<".ljust(44, "<")
    return _make_mock_ocr_output([
        "REPUBLIC OF SRI LANKA",
        "PASSPORT",
        line1,
        line2,
    ])


# ── /ocr/extract (sync) ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_extract_returns_200(client, nic_ocr_output):
    with patch("app.ocr.pipeline.run_pipeline") as mock_pipeline:
        from app.schemas.ocr import (
            DocumentOCRResult,
            DocumentType,
            ExtractedFields,
            ImageQualityReport,
        )

        mock_pipeline.return_value = DocumentOCRResult(
            quality=ImageQualityReport(
                passed=True, resolution_ok=True, sharpness=120.0, brightness=130.0, contrast=45.0
            ),
            document_type=DocumentType.NIC,
            fields=ExtractedFields(
                document_type=DocumentType.NIC,
                document_number="890123456V",
                full_name="Kamal Bandara Perera",
                dob="1989-01-12",
                sex="M",
                extraction_confidence=1.0,
            ),
            raw_blocks=[],
            full_text="KAMAL BANDARA PERERA 890123456V",
            average_confidence=0.95,
        )

        png = _make_png_bytes()
        response = await client.post(
            "/api/v1/ocr/extract",
            files={"file": ("test.png", io.BytesIO(png), "image/png")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["document_type"] == "nic"
    assert body["fields"]["document_number"] == "890123456V"
    assert body["fields"]["dob"] == "1989-01-12"
    assert body["quality"]["passed"] is True


@pytest.mark.asyncio
async def test_extract_with_doc_type_hint(client):
    with patch("app.ocr.pipeline.run_pipeline") as mock_pipeline:
        from app.schemas.ocr import (
            DocumentOCRResult,
            DocumentType,
            ExtractedFields,
            ImageQualityReport,
        )

        mock_pipeline.return_value = DocumentOCRResult(
            quality=ImageQualityReport(
                passed=True, resolution_ok=True, sharpness=100.0, brightness=128.0, contrast=40.0
            ),
            document_type=DocumentType.PASSPORT,
            fields=ExtractedFields(
                document_type=DocumentType.PASSPORT,
                extraction_confidence=0.8,
                mrz_parsed=True,
            ),
            raw_blocks=[],
            full_text="",
        )

        png = _make_png_bytes()
        response = await client.post(
            "/api/v1/ocr/extract?doc_type=passport",
            files={"file": ("passport.png", io.BytesIO(png), "image/png")},
        )

    assert response.status_code == 200
    assert response.json()["document_type"] == "passport"


@pytest.mark.asyncio
async def test_extract_invalid_file_returns_422(client):
    response = await client.post(
        "/api/v1/ocr/extract",
        files={"file": ("bad.png", io.BytesIO(b"not-an-image"), "image/png")},
    )
    assert response.status_code == 422


# ── /ocr/extract/async ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_extract_async_returns_202_with_job_id(client):
    with patch("app.services.ocr_service.enqueue_task", return_value="job-abc-123"), \
         patch("app.services.ocr_service.OCRService._quick_quality_check"):
        png = _make_png_bytes()
        response = await client.post(
            "/api/v1/ocr/extract/async",
            files={"file": ("test.png", io.BytesIO(png), "image/png")},
        )

    assert response.status_code == 202
    body = response.json()
    assert body["job_id"] == "job-abc-123"
    assert "poll_url" in body
    assert "job-abc-123" in body["poll_url"]


# ── /ocr/jobs/{job_id} ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_poll_job_queued(client):
    with patch("app.services.ocr_service.get_job_status", return_value={
        "id": "job-abc-123",
        "status": "queued",
        "result": None,
        "error": None,
    }):
        response = await client.get("/api/v1/ocr/jobs/job-abc-123")

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert response.json()["result"] is None


@pytest.mark.asyncio
async def test_poll_job_finished(client):
    from app.schemas.ocr import (
        DocumentOCRResult,
        DocumentType,
        ExtractedFields,
        ImageQualityReport,
    )

    finished_result = DocumentOCRResult(
        quality=ImageQualityReport(
            passed=True, resolution_ok=True, sharpness=110.0, brightness=130.0, contrast=42.0
        ),
        document_type=DocumentType.NIC,
        fields=ExtractedFields(
            document_type=DocumentType.NIC,
            document_number="199012345678",
            dob="1990-01-12",
            extraction_confidence=0.67,
        ),
        raw_blocks=[],
        full_text="199012345678",
    ).model_dump()

    with patch("app.services.ocr_service.get_job_status", return_value={
        "id": "job-xyz-789",
        "status": "finished",
        "result": finished_result,
        "error": None,
    }):
        response = await client.get("/api/v1/ocr/jobs/job-xyz-789")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "finished"
    assert body["result"]["document_type"] == "nic"
    assert body["result"]["fields"]["document_number"] == "199012345678"

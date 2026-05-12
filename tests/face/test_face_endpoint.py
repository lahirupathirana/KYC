"""
Integration tests for the face API endpoints.

InsightFace model is mocked via the lifespan fixture from the OCR test pattern.
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.schemas.face import (
    DetectedFace,
    FaceAnalysisResult,
    FaceMatchResult,
    FaceQualityReport,
    MatchVerdict,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _png_bytes(h: int = 200, w: int = 200) -> bytes:
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _good_quality() -> FaceQualityReport:
    return FaceQualityReport(
        passed=True,
        face_detected=True,
        face_count=1,
        detection_score=0.97,
        face_size_px=140,
        sharpness=120.0,
        brightness=128.0,
        pose_yaw=5.0,
        pose_pitch=3.0,
        pose_roll=1.0,
        pose_acceptable=True,
        issues=[],
    )


def _good_face() -> DetectedFace:
    rng = np.random.default_rng(0)
    emb = rng.standard_normal(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return DetectedFace(
        bbox=[10.0, 10.0, 150.0, 150.0],
        detection_score=0.97,
        embedding=emb.tolist(),
    )


@asynccontextmanager
async def _mock_lifespan(app: FastAPI):
    mock_registry = MagicMock()
    mock_face = MagicMock()
    mock_registry.get.return_value = mock_face
    app.state.model_registry = mock_registry
    yield


@pytest.fixture
async def client():
    app = create_app(lifespan_handler=_mock_lifespan)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ── /face/analyze ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_returns_200(client):
    good_result = FaceAnalysisResult(quality=_good_quality(), face=_good_face())

    with patch("app.services.face_service.analyse_single", return_value=good_result):
        response = await client.post(
            "/api/v1/face/analyze",
            files={"file": ("face.png", io.BytesIO(_png_bytes()), "image/png")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["quality"]["passed"] is True
    assert body["quality"]["face_detected"] is True
    assert body["face"] is not None
    assert len(body["face"]["embedding"]) == 512


@pytest.mark.asyncio
async def test_analyze_no_face_returns_200_with_failed_quality(client):
    no_face_result = FaceAnalysisResult(
        quality=FaceQualityReport(
            passed=False,
            face_detected=False,
            face_count=0,
            issues=["No face detected in image"],
        ),
        face=None,
    )

    with patch("app.services.face_service.analyse_single", return_value=no_face_result):
        response = await client.post(
            "/api/v1/face/analyze",
            files={"file": ("empty.png", io.BytesIO(_png_bytes()), "image/png")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["quality"]["passed"] is False
    assert body["face"] is None


# ── /face/match ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_match_returns_match_verdict(client):
    match_result = FaceMatchResult(
        verdict=MatchVerdict.MATCH,
        is_match=True,
        similarity_score=0.85,
        threshold_used=0.40,
        review_threshold=0.20,
        id_quality=_good_quality(),
        selfie_quality=_good_quality(),
        id_face=_good_face(),
        selfie_face=_good_face(),
        explanation="Similarity 0.8500 ≥ match threshold 0.40",
        match_duration_ms=42.0,
    )

    with patch("app.services.face_service.match_faces", return_value=match_result):
        response = await client.post(
            "/api/v1/face/match",
            files={
                "id_document": ("id.png", io.BytesIO(_png_bytes()), "image/png"),
                "selfie": ("selfie.png", io.BytesIO(_png_bytes()), "image/png"),
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "match"
    assert body["is_match"] is True
    assert body["similarity_score"] == pytest.approx(0.85)
    assert body["id_quality"]["passed"] is True
    assert body["selfie_quality"]["passed"] is True


@pytest.mark.asyncio
async def test_match_returns_no_match_verdict(client):
    no_match_result = FaceMatchResult(
        verdict=MatchVerdict.NO_MATCH,
        is_match=False,
        similarity_score=0.05,
        threshold_used=0.40,
        review_threshold=0.20,
        id_quality=_good_quality(),
        selfie_quality=_good_quality(),
        id_face=_good_face(),
        selfie_face=_good_face(),
        explanation="Similarity 0.0500 < review threshold 0.20",
        match_duration_ms=38.0,
    )

    with patch("app.services.face_service.match_faces", return_value=no_match_result):
        response = await client.post(
            "/api/v1/face/match",
            files={
                "id_document": ("id.png", io.BytesIO(_png_bytes()), "image/png"),
                "selfie": ("selfie.png", io.BytesIO(_png_bytes()), "image/png"),
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "no_match"
    assert body["is_match"] is False


# ── /face/embedding ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_embedding_returns_512_floats(client):
    good_result = FaceAnalysisResult(quality=_good_quality(), face=_good_face())

    with patch("app.services.face_service.analyse_single", return_value=good_result):
        response = await client.post(
            "/api/v1/face/embedding",
            files={"file": ("face.png", io.BytesIO(_png_bytes()), "image/png")},
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["embedding"]) == 512
    assert body["detection_score"] == pytest.approx(0.97)


@pytest.mark.asyncio
async def test_embedding_no_face_returns_422(client):
    no_face_result = FaceAnalysisResult(
        quality=FaceQualityReport(
            passed=False, face_detected=False, face_count=0,
            issues=["No face detected in image"],
        ),
        face=None,
    )

    with patch("app.services.face_service.analyse_single", return_value=no_face_result):
        response = await client.post(
            "/api/v1/face/embedding",
            files={"file": ("empty.png", io.BytesIO(_png_bytes()), "image/png")},
        )

    assert response.status_code == 422

"""
Unit tests for app.face.pipeline — InsightFace model is mocked.
numpy is available; cv2 is available.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.core.config import Settings
from app.face.pipeline import analyse_single, match_faces
from app.schemas.face import FaceAnalysisResult, FaceMatchResult, MatchVerdict


# ── Helpers ───────────────────────────────────────────────────────────────────

def _png_bytes(h: int = 200, w: int = 200) -> bytes:
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _make_embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(512).astype(np.float32)
    return emb / np.linalg.norm(emb)  # L2-normalised


def _make_face_obj(det_score: float = 0.95, emb: np.ndarray | None = None, bbox=None):
    face = SimpleNamespace()
    face.det_score = det_score
    face.bbox = np.array(bbox or [10.0, 10.0, 110.0, 110.0])
    face.embedding = emb if emb is not None else _make_embedding()
    face.pose = np.array([5.0, 3.0, 1.0])
    return face


def _settings(**overrides) -> Settings:
    defaults = dict(
        face_match_threshold=0.40,
        face_review_threshold=0.20,
        face_min_detection_score=0.70,
        face_min_size_px=80,
        face_max_pose_yaw=35.0,
        face_max_pose_pitch=30.0,
    )
    defaults.update(overrides)
    s = MagicMock(spec=Settings)
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


# ── analyse_single ─────────────────────────────────────────────────────────────

def test_analyse_single_good_face():
    mock_model = MagicMock()
    mock_model.get.return_value = [_make_face_obj(bbox=[10, 10, 150, 150])]
    result = analyse_single(_png_bytes(), mock_model, _settings())
    assert isinstance(result, FaceAnalysisResult)
    assert result.quality.passed is True
    assert result.face is not None
    assert len(result.face.embedding) == 512


def test_analyse_single_no_face():
    mock_model = MagicMock()
    mock_model.get.return_value = []
    result = analyse_single(_png_bytes(), mock_model, _settings())
    assert result.quality.passed is False
    assert result.quality.face_detected is False
    assert result.face is None


def test_analyse_single_invalid_image():
    mock_model = MagicMock()
    with pytest.raises(ValueError, match="decode"):
        analyse_single(b"not-an-image", mock_model, _settings())


# ── match_faces ────────────────────────────────────────────────────────────────

def test_match_faces_identical_embeddings_is_match():
    emb = _make_embedding(seed=1)
    face_obj = _make_face_obj(emb=emb, bbox=[10, 10, 150, 150])
    mock_model = MagicMock()
    mock_model.get.return_value = [face_obj]

    result = match_faces(_png_bytes(), _png_bytes(), mock_model, _settings())
    assert isinstance(result, FaceMatchResult)
    assert result.verdict == MatchVerdict.MATCH
    assert result.is_match is True
    assert result.similarity_score == pytest.approx(1.0, abs=1e-4)


def test_match_faces_orthogonal_embeddings_is_no_match():
    emb1 = np.zeros(512, dtype=np.float32)
    emb1[0] = 1.0
    emb2 = np.zeros(512, dtype=np.float32)
    emb2[1] = 1.0

    mock_model = MagicMock()
    mock_model.get.side_effect = [
        [_make_face_obj(emb=emb1, bbox=[10, 10, 150, 150])],
        [_make_face_obj(emb=emb2, bbox=[10, 10, 150, 150])],
    ]

    result = match_faces(_png_bytes(), _png_bytes(), mock_model, _settings())
    assert result.verdict == MatchVerdict.NO_MATCH
    assert result.similarity_score == pytest.approx(0.0, abs=1e-4)


def test_match_faces_review_band():
    emb1 = _make_embedding(seed=10)
    emb2 = _make_embedding(seed=11)
    # Force similarity into review band by scaling emb2 toward emb1
    target = 0.30  # between review_threshold 0.20 and match_threshold 0.40
    emb2 = target * emb1 + (1 - target) * emb2
    emb2 /= np.linalg.norm(emb2)

    mock_model = MagicMock()
    mock_model.get.side_effect = [
        [_make_face_obj(emb=emb1, bbox=[10, 10, 150, 150])],
        [_make_face_obj(emb=emb2, bbox=[10, 10, 150, 150])],
    ]

    result = match_faces(_png_bytes(), _png_bytes(), mock_model, _settings())
    assert result.verdict == MatchVerdict.REVIEW


def test_match_faces_no_face_in_id_returns_no_match():
    mock_model = MagicMock()
    mock_model.get.side_effect = [[], [_make_face_obj()]]  # no face in ID
    result = match_faces(_png_bytes(), _png_bytes(), mock_model, _settings())
    assert result.verdict == MatchVerdict.NO_MATCH
    assert result.similarity_score is None
    assert "ID" in result.explanation


def test_match_faces_no_face_in_selfie_returns_no_match():
    mock_model = MagicMock()
    mock_model.get.side_effect = [[_make_face_obj(bbox=[10, 10, 150, 150])], []]
    result = match_faces(_png_bytes(), _png_bytes(), mock_model, _settings())
    assert result.verdict == MatchVerdict.NO_MATCH
    assert result.similarity_score is None
    assert "selfie" in result.explanation.lower() or "Selfie" in result.explanation


def test_match_faces_duration_is_populated():
    emb = _make_embedding(seed=2)
    mock_model = MagicMock()
    mock_model.get.return_value = [_make_face_obj(emb=emb, bbox=[10, 10, 150, 150])]
    result = match_faces(_png_bytes(), _png_bytes(), mock_model, _settings())
    assert result.match_duration_ms is not None
    assert result.match_duration_ms >= 0

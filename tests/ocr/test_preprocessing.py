"""Tests for image preprocessing and quality validation."""

import numpy as np
import pytest

from app.ocr.preprocessing import (
    QualityReport,
    decode_image,
    preprocess,
    validate_quality,
)


def _make_bgr(h: int, w: int, brightness: int = 128) -> np.ndarray:
    return np.full((h, w, 3), brightness, dtype=np.uint8)


def _make_noisy(h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(50, 200, (h, w, 3), dtype=np.uint8)


# ── decode_image ──────────────────────────────────────────────────────────────

def test_decode_image_raises_on_corrupt_bytes():
    with pytest.raises(ValueError, match="Could not decode"):
        decode_image(b"not-an-image")


def test_decode_image_valid_png():
    import cv2

    img = _make_bgr(100, 100)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    result = decode_image(buf.tobytes())
    assert result.shape == (100, 100, 3)


# ── validate_quality ──────────────────────────────────────────────────────────

def test_quality_fails_on_tiny_image():
    img = _make_bgr(100, 100)
    report = validate_quality(img)
    assert not report.resolution_ok
    assert any("esolution" in issue for issue in report.issues)


def test_quality_fails_on_black_image():
    img = _make_bgr(600, 900, brightness=0)
    report = validate_quality(img)
    assert any("dark" in issue.lower() for issue in report.issues)


def test_quality_fails_on_overexposed_image():
    img = _make_bgr(600, 900, brightness=255)
    report = validate_quality(img)
    assert any("overexposed" in issue.lower() or "bright" in issue.lower() for issue in report.issues)


def test_quality_passes_on_good_image():
    img = _make_noisy(600, 900)
    report = validate_quality(img)
    assert report.resolution_ok
    # Noisy image has natural contrast and brightness variation
    assert isinstance(report.sharpness, float)
    assert isinstance(report.brightness, float)


def test_quality_report_fields():
    img = _make_noisy(700, 1000)
    report = validate_quality(img)
    assert hasattr(report, "passed")
    assert hasattr(report, "sharpness")
    assert hasattr(report, "issues")
    assert isinstance(report.issues, list)


# ── preprocess ────────────────────────────────────────────────────────────────

def test_preprocess_returns_ndarray():
    img = _make_noisy(600, 800)
    result = preprocess(img)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3


def test_preprocess_upscales_small_image():
    img = _make_noisy(300, 400)
    result = preprocess(img)
    # Should be upscaled to at least 1400px wide
    assert result.shape[1] >= 1400


def test_preprocess_preserves_large_image_width():
    img = _make_noisy(800, 1600)
    result = preprocess(img)
    assert result.shape[1] >= 1400

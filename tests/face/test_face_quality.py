"""
Unit tests for app.face.quality — no InsightFace required.
Fakes are constructed directly from the public interface.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.face.quality import analyse_face_quality
from app.schemas.face import FaceQualityReport


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_face(det_score: float = 0.95, bbox: list[float] | None = None, pose=None):
    face = SimpleNamespace()
    face.det_score = det_score
    face.bbox = np.array(bbox or [10.0, 10.0, 110.0, 110.0])  # 100×100 face
    face.pose = np.array(pose) if pose else None
    return face


def _default_kwargs():
    return dict(
        min_detection_score=0.70,
        min_size_px=80,
        max_pose_yaw=35.0,
        max_pose_pitch=30.0,
    )


def _blank_img(h=200, w=200):
    return np.full((h, w, 3), 128, dtype=np.uint8)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_no_faces_returns_failed_report():
    report = analyse_face_quality([], _blank_img(), **_default_kwargs())
    assert isinstance(report, FaceQualityReport)
    assert report.passed is False
    assert report.face_detected is False
    assert report.face_count == 0
    assert any("No face" in i for i in report.issues)


def test_good_face_passes():
    faces = [_make_face(det_score=0.95, bbox=[10, 10, 150, 150])]
    img = _blank_img(300, 300)
    report = analyse_face_quality(faces, img, **_default_kwargs())
    assert report.passed is True
    assert report.face_detected is True
    assert report.face_count == 1
    assert report.issues == []


def test_low_detection_score_fails():
    faces = [_make_face(det_score=0.50, bbox=[10, 10, 150, 150])]
    report = analyse_face_quality(faces, _blank_img(300, 300), **_default_kwargs())
    assert report.passed is False
    assert any("confidence" in i for i in report.issues)


def test_small_face_fails():
    # bbox 10→40 = 30px face, below min 80
    faces = [_make_face(det_score=0.95, bbox=[10, 10, 40, 40])]
    report = analyse_face_quality(faces, _blank_img(200, 200), **_default_kwargs())
    assert report.passed is False
    assert any("small" in i.lower() for i in report.issues)


def test_multiple_faces_adds_warning_but_does_not_fail_alone():
    faces = [
        _make_face(det_score=0.95, bbox=[10, 10, 150, 150]),
        _make_face(det_score=0.80, bbox=[160, 10, 300, 150]),
    ]
    report = analyse_face_quality(faces, _blank_img(300, 400), **_default_kwargs())
    assert report.face_count == 2
    assert any("Multiple" in i for i in report.issues)
    # Multiple-face warning should not cause passed=False on its own
    non_multi = [i for i in report.issues if "Multiple" not in i]
    # If no other issues the report can still pass
    assert report.passed == (len(non_multi) == 0)


def test_excessive_yaw_fails():
    faces = [_make_face(det_score=0.95, bbox=[10, 10, 150, 150], pose=[45.0, 5.0, 0.0])]
    report = analyse_face_quality(faces, _blank_img(300, 300), **_default_kwargs())
    assert report.passed is False
    assert report.pose_acceptable is False
    assert any("yaw" in i.lower() for i in report.issues)


def test_acceptable_pose_passes():
    faces = [_make_face(det_score=0.95, bbox=[10, 10, 150, 150], pose=[10.0, 5.0, 2.0])]
    report = analyse_face_quality(faces, _blank_img(300, 300), **_default_kwargs())
    assert report.pose_acceptable is True
    assert report.pose_yaw == pytest.approx(10.0)

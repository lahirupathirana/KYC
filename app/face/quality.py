"""
Per-face quality analysis after InsightFace detection.

Runs synchronously — callers must wrap in run_in_executor if async.
"""

from __future__ import annotations

import math

import numpy as np

from app.schemas.face import FaceQualityReport


def analyse_face_quality(
    faces: list,           # list of insightface Face objects
    img: np.ndarray,       # BGR image the faces were detected from
    *,
    min_detection_score: float,
    min_size_px: int,
    min_sharpness: float,
    max_pose_yaw: float,
    max_pose_pitch: float,
) -> FaceQualityReport:
    """
    Evaluate the quality of the best detected face in an image.

    Returns a FaceQualityReport regardless of whether a face was found.
    `passed=True` only when all hard constraints are satisfied.
    """
    issues: list[str] = []

    if not faces:
        return FaceQualityReport(
            passed=False,
            face_detected=False,
            face_count=0,
            issues=["No face detected in image"],
        )

    if len(faces) > 1:
        issues.append(f"Multiple faces detected ({len(faces)}); using highest-confidence face")

    face = max(faces, key=lambda f: f.det_score)

    det_score = float(face.det_score)
    if det_score < min_detection_score:
        issues.append(
            f"Detection confidence {det_score:.2f} below minimum {min_detection_score:.2f}"
        )

    bbox = face.bbox  # [x1, y1, x2, y2]
    face_w = int(bbox[2] - bbox[0])
    face_h = int(bbox[3] - bbox[1])
    face_size = min(face_w, face_h)
    if face_size < min_size_px:
        issues.append(f"Face too small ({face_size}px); minimum is {min_size_px}px")

    # Sharpness of the face crop
    x1, y1, x2, y2 = (max(0, int(v)) for v in bbox)
    crop = img[y1:y2, x1:x2]
    sharpness = _laplacian_variance(crop)
    if sharpness < min_sharpness:
        issues.append(f"Face region is blurry (sharpness={sharpness:.1f})")

    # Brightness of crop
    brightness = float(crop.mean()) if crop.size > 0 else 0.0

    # Pose estimation
    pose_yaw: float | None = None
    pose_pitch: float | None = None
    pose_roll: float | None = None
    pose_acceptable = True

    if hasattr(face, "pose") and face.pose is not None:
        pose = face.pose  # [yaw, pitch, roll] in degrees
        pose_yaw = float(pose[0])
        pose_pitch = float(pose[1])
        pose_roll = float(pose[2])
        if abs(pose_yaw) > max_pose_yaw:
            issues.append(f"Excessive yaw ({pose_yaw:.1f}°); max is {max_pose_yaw}°")
            pose_acceptable = False
        if abs(pose_pitch) > max_pose_pitch:
            issues.append(f"Excessive pitch ({pose_pitch:.1f}°); max is {max_pose_pitch}°")
            pose_acceptable = False

    passed = len([i for i in issues if not i.startswith("Multiple faces")]) == 0

    return FaceQualityReport(
        passed=passed,
        face_detected=True,
        face_count=len(faces),
        detection_score=det_score,
        face_size_px=face_size,
        sharpness=round(sharpness, 2),
        brightness=round(brightness, 2),
        pose_yaw=pose_yaw,
        pose_pitch=pose_pitch,
        pose_roll=pose_roll,
        pose_acceptable=pose_acceptable,
        issues=issues,
    )


def _laplacian_variance(crop: np.ndarray) -> float:
    """Variance of Laplacian as a blur proxy. Returns 0 for empty crops."""
    if crop.size == 0:
        return 0.0
    import cv2

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

"""
Shared face analysis pipeline.

analyse_single  — decode → quality check → extract embedding
match_faces     — run analyse_single on both images → cosine similarity → verdict

Both functions are synchronous and CPU-bound; callers wrap them in
run_in_executor via functools.partial.
"""

from __future__ import annotations

import time

import numpy as np

from app.core.config import Settings
from app.face.preprocessing import decode_image, maybe_correct_gamma
from app.face.quality import analyse_face_quality
from app.schemas.face import (
    DetectedFace,
    FaceAnalysisResult,
    FaceMatchResult,
    FaceQualityReport,
    MatchVerdict,
)


def analyse_single(
    image_bytes: bytes,
    model,           # insightface FaceAnalysis instance
    settings: Settings,
) -> FaceAnalysisResult:
    """
    Full single-image analysis: decode → correct gamma → detect → quality check → embed.
    Raises ValueError when image cannot be decoded.
    """
    img = decode_image(image_bytes)
    img = maybe_correct_gamma(img)

    faces = model.get(img)

    quality = analyse_face_quality(
        faces,
        img,
        min_detection_score=settings.face_min_detection_score,
        min_size_px=settings.face_min_size_px,
        min_sharpness=settings.face_min_sharpness,
        max_pose_yaw=settings.face_max_pose_yaw,
        max_pose_pitch=settings.face_max_pose_pitch,
    )

    detected_face: DetectedFace | None = None

    if quality.face_detected:
        best = max(faces, key=lambda f: f.det_score)
        bbox = best.bbox.tolist()

        # Explicitly L2-normalise — buffalo_l may return unnormalised embeddings
        # depending on InsightFace version. Normalising here guarantees that
        # dot product == cosine similarity in match_faces().
        raw = best.embedding.astype(np.float32)
        norm = np.linalg.norm(raw)
        emb_normalised = (raw / norm if norm > 0 else raw).tolist()

        detected_face = DetectedFace(
            bbox=bbox,
            detection_score=float(best.det_score),
            embedding=emb_normalised,
        )

    return FaceAnalysisResult(
        quality=quality,
        face=detected_face,
    )


def match_faces(
    id_image_bytes: bytes,
    selfie_bytes: bytes,
    model,
    settings: Settings,
) -> FaceMatchResult:
    """
    Compare an ID document face with a live selfie.

    Returns a FaceMatchResult with a three-tier verdict:
      MATCH     similarity ≥ face_match_threshold
      REVIEW    face_review_threshold ≤ similarity < face_match_threshold
      NO_MATCH  similarity < face_review_threshold  (or quality failure)
    """
    t0 = time.monotonic()

    id_result = analyse_single(id_image_bytes, model, settings)
    selfie_result = analyse_single(selfie_bytes, model, settings)

    duration_ms = round((time.monotonic() - t0) * 1000, 1)

    # Hard fail paths — no similarity computed
    if not id_result.quality.passed:
        return _quality_failure(
            "ID image quality check failed",
            id_result.quality,
            selfie_result.quality,
            settings,
            duration_ms,
        )
    if not selfie_result.quality.passed:
        return _quality_failure(
            "Selfie quality check failed",
            id_result.quality,
            selfie_result.quality,
            settings,
            duration_ms,
        )

    id_emb = np.array(id_result.face.embedding, dtype=np.float32)
    selfie_emb = np.array(selfie_result.face.embedding, dtype=np.float32)

    # buffalo_l embeddings are L2-normalised — dot product equals cosine similarity
    similarity = float(np.dot(id_emb, selfie_emb))
    similarity = round(similarity, 4)

    if similarity >= settings.face_match_threshold:
        verdict = MatchVerdict.MATCH
        is_match = True
        explanation = (
            f"Similarity {similarity:.4f} ≥ match threshold {settings.face_match_threshold}"
        )
    elif similarity >= settings.face_review_threshold:
        verdict = MatchVerdict.REVIEW
        is_match = False
        explanation = (
            f"Similarity {similarity:.4f} is in review band "
            f"[{settings.face_review_threshold}, {settings.face_match_threshold})"
        )
    else:
        verdict = MatchVerdict.NO_MATCH
        is_match = False
        explanation = (
            f"Similarity {similarity:.4f} < review threshold {settings.face_review_threshold}"
        )

    return FaceMatchResult(
        verdict=verdict,
        is_match=is_match,
        similarity_score=similarity,
        threshold_used=settings.face_match_threshold,
        review_threshold=settings.face_review_threshold,
        id_quality=id_result.quality,
        selfie_quality=selfie_result.quality,
        id_face=id_result.face,
        selfie_face=selfie_result.face,
        explanation=explanation,
        match_duration_ms=duration_ms,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _quality_failure(
    reason: str,
    id_quality: FaceQualityReport,
    selfie_quality: FaceQualityReport,
    settings: Settings,
    duration_ms: float,
) -> FaceMatchResult:
    return FaceMatchResult(
        verdict=MatchVerdict.NO_MATCH,
        is_match=False,
        similarity_score=None,
        threshold_used=settings.face_match_threshold,
        review_threshold=settings.face_review_threshold,
        id_quality=id_quality,
        selfie_quality=selfie_quality,
        id_face=None,
        selfie_face=None,
        explanation=reason,
        match_duration_ms=duration_ms,
    )

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class FaceQualityReport(BaseModel):
    passed: bool
    face_detected: bool
    face_count: int = 0
    detection_score: float = 0.0
    face_size_px: int = 0
    sharpness: float = 0.0
    brightness: float = 0.0
    pose_yaw: float | None = None
    pose_pitch: float | None = None
    pose_roll: float | None = None
    pose_acceptable: bool = True
    issues: list[str] = []


class DetectedFace(BaseModel):
    bbox: list[float]           # [x1, y1, x2, y2]
    detection_score: float
    embedding: list[float]      # 512-dim L2-normalised ArcFace embedding


class FaceAnalysisResult(BaseModel):
    quality: FaceQualityReport
    face: DetectedFace | None = None


class FaceEmbedding(BaseModel):
    embedding: list[float]
    detection_score: float


class MatchVerdict(str, Enum):
    MATCH = "match"
    REVIEW = "review"
    NO_MATCH = "no_match"


class FaceMatchResult(BaseModel):
    verdict: MatchVerdict
    is_match: bool
    similarity_score: float | None = None
    threshold_used: float
    review_threshold: float
    id_quality: FaceQualityReport
    selfie_quality: FaceQualityReport
    id_face: DetectedFace | None = None
    selfie_face: DetectedFace | None = None
    explanation: str
    match_duration_ms: float | None = None

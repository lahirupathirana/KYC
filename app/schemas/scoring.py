from enum import Enum

from pydantic import BaseModel


class VerificationDecision(str, Enum):
    APPROVED = "approved"
    REVIEW = "review"       # borderline — route to human agent
    REJECTED = "rejected"


class ComponentScore(BaseModel):
    component: str
    raw_score: float
    weight: float           # normalized weight (sums to 1.0 across present components)
    contribution: float     # raw_score × weight
    explanation: str


class ScoringInput(BaseModel):
    session_id: str
    ocr_confidence: float | None = None
    face_similarity: float | None = None
    liveness_confidence: float | None = None
    voice_confidence: float | None = None


class ScoringResult(BaseModel):
    session_id: str
    final_score: float
    decision: VerificationDecision
    components: list[ComponentScore]
    threshold: float
    explanation: str

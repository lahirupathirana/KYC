from pydantic import BaseModel


class FaceEmbedding(BaseModel):
    embedding: list[float]
    detection_score: float


class FaceMatchResult(BaseModel):
    is_match: bool
    similarity_score: float
    threshold: float
    explanation: str

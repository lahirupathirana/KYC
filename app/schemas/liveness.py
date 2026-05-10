from pydantic import BaseModel


class LivenessResult(BaseModel):
    is_live: bool
    confidence: float
    explanation: str

from pydantic import BaseModel


class TranscriptionResult(BaseModel):
    text: str
    language: str
    segments: list[dict] = []

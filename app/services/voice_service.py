import asyncio
import os
import tempfile

import structlog

from app.models.model_registry import ModelRegistry
from app.schemas.voice import TranscriptionResult

logger = structlog.get_logger()


class VoiceService:
    def __init__(self, registry: ModelRegistry) -> None:
        self._model = registry.get("whisper")

    async def transcribe(self, audio_bytes: bytes, language: str = "en") -> TranscriptionResult:
        # Whisper is CPU/GPU bound; writing to temp file is also blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_whisper, audio_bytes, language)

    def _run_whisper(self, audio_bytes: bytes, language: str) -> TranscriptionResult:
        # Whisper requires a file path — write bytes to a temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        try:
            result = self._model.transcribe(tmp_path, language=language, fp16=False)
            logger.debug("Transcription complete", language=result.get("language"), chars=len(result["text"]))
            return TranscriptionResult(
                text=result["text"].strip(),
                language=result.get("language", language),
                segments=[
                    {"start": s["start"], "end": s["end"], "text": s["text"]}
                    for s in result.get("segments", [])
                ],
            )
        finally:
            os.unlink(tmp_path)

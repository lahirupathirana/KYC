from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.core.dependencies import get_voice_service
from app.schemas.voice import TranscriptionResult
from app.services.voice_service import VoiceService

router = APIRouter()


@router.post("/transcribe", response_model=TranscriptionResult, summary="Transcribe audio response")
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file (WAV/MP3/WebM)"),
    language: str = Form(default="en", description="BCP-47 language code"),
    service: VoiceService = Depends(get_voice_service),
) -> TranscriptionResult:
    audio_bytes = await audio.read()
    return await service.transcribe(audio_bytes, language=language)

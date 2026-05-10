from typing import AsyncGenerator

import redis.asyncio as aioredis
import structlog
from fastapi import Depends, Request

from app.core.config import Settings, settings
from app.models.model_registry import ModelRegistry
from app.services.face_service import FaceService
from app.services.liveness_service import LivenessService
from app.services.ocr_service import OCRService
from app.services.scoring_service import ScoringService
from app.services.voice_service import VoiceService

logger = structlog.get_logger()


# ── Singletons ────────────────────────────────────────────────────────────────

def get_settings() -> Settings:
    return settings


def get_model_registry(request: Request) -> ModelRegistry:
    return request.app.state.model_registry


# ── Redis connection (yielded per-request from shared pool) ───────────────────

async def get_redis(
    cfg: Settings = Depends(get_settings),
) -> AsyncGenerator[aioredis.Redis, None]:
    client = aioredis.from_url(cfg.redis_url, max_connections=cfg.redis_max_connections, decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


# ── Service factories — models loaded once on startup, injected here ──────────

def get_ocr_service(registry: ModelRegistry = Depends(get_model_registry)) -> OCRService:
    return OCRService(registry)


def get_face_service(registry: ModelRegistry = Depends(get_model_registry)) -> FaceService:
    return FaceService(registry)


def get_liveness_service(registry: ModelRegistry = Depends(get_model_registry)) -> LivenessService:
    return LivenessService(registry)


def get_voice_service(registry: ModelRegistry = Depends(get_model_registry)) -> VoiceService:
    return VoiceService(registry)


def get_scoring_service() -> ScoringService:
    # No model dependency — pure scoring logic
    return ScoringService()

"""
FaceService — async wrapper around the synchronous face pipeline.

All CPU-bound InsightFace calls are dispatched to the default
ThreadPoolExecutor via run_in_executor so they never block the event loop.
functools.partial passes settings into the sync pipeline functions.
"""

from __future__ import annotations

import asyncio
from functools import partial

import structlog

from app.core.config import Settings
from app.face.pipeline import analyse_single, match_faces
from app.models.model_registry import ModelRegistry
from app.schemas.face import FaceAnalysisResult, FaceEmbedding, FaceMatchResult

logger = structlog.get_logger()


class FaceService:
    def __init__(self, registry: ModelRegistry, settings: Settings) -> None:
        self._model = registry.get("face")
        self._settings = settings

    async def analyse(self, image_bytes: bytes) -> FaceAnalysisResult:
        """Decode, quality-check, and embed a single image."""
        loop = asyncio.get_event_loop()
        fn = partial(analyse_single, image_bytes, self._model, self._settings)
        result: FaceAnalysisResult = await loop.run_in_executor(None, fn)
        logger.info(
            "Face analyse complete",
            face_detected=result.quality.face_detected,
            quality_passed=result.quality.passed,
            issues=result.quality.issues,
        )
        return result

    async def get_embedding(self, image_bytes: bytes) -> FaceEmbedding:
        """Extract a 512-dim embedding from the best detected face."""
        result = await self.analyse(image_bytes)
        if not result.quality.face_detected or result.face is None:
            raise ValueError("No face detected in image")
        return FaceEmbedding(
            embedding=result.face.embedding,
            detection_score=result.face.detection_score,
        )

    async def match(self, id_image_bytes: bytes, selfie_bytes: bytes) -> FaceMatchResult:
        """Compare an ID document face with a live selfie."""
        loop = asyncio.get_event_loop()
        fn = partial(match_faces, id_image_bytes, selfie_bytes, self._model, self._settings)
        result: FaceMatchResult = await loop.run_in_executor(None, fn)
        logger.info(
            "Face match complete",
            verdict=result.verdict,
            similarity=result.similarity_score,
            duration_ms=result.match_duration_ms,
        )
        return result

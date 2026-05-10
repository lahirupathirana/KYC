import asyncio

import numpy as np
import structlog

from app.models.model_registry import ModelRegistry
from app.schemas.liveness import LivenessResult

logger = structlog.get_logger()


class LivenessService:
    """
    Passive liveness detection using InsightFace detection confidence as a proxy.

    This is a placeholder for Phase 5 (active liveness via WebRTC
    challenge-response: blink / head-turn / random digit recitation).
    Replace _analyze_frame with a dedicated anti-spoofing model (e.g.
    Silent-Face or MiniFASNet) when that phase begins.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._face_model = registry.get("face")

    async def check_liveness(self, frame_bytes: bytes) -> LivenessResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._analyze_frame, frame_bytes)

    def _analyze_frame(self, frame_bytes: bytes) -> LivenessResult:
        import cv2

        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode liveness frame")

        faces = self._face_model.get(img)
        if not faces:
            return LivenessResult(is_live=False, confidence=0.0, explanation="No face detected in frame")

        face = max(faces, key=lambda f: f.det_score)
        confidence = round(float(face.det_score), 4)
        is_live = confidence > 0.70

        return LivenessResult(
            is_live=is_live,
            confidence=confidence,
            explanation=(
                f"Passive detection confidence {confidence} "
                f"(active liveness challenge pending — Phase 5)"
            ),
        )

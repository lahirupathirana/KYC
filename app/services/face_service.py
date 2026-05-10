import asyncio

import numpy as np
import structlog

from app.models.model_registry import ModelRegistry
from app.schemas.face import FaceEmbedding, FaceMatchResult

logger = structlog.get_logger()

# InsightFace cosine similarity threshold validated for buffalo_l on LFW
SIMILARITY_THRESHOLD = 0.40


class FaceService:
    def __init__(self, registry: ModelRegistry) -> None:
        self._model = registry.get("face")

    async def get_embedding(self, image_bytes: bytes) -> FaceEmbedding:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._extract_embedding, image_bytes)

    async def match_faces(self, id_image_bytes: bytes, selfie_bytes: bytes) -> FaceMatchResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._compare, id_image_bytes, selfie_bytes)

    # ── Sync helpers (run inside executor) ────────────────────────────────────

    def _extract_embedding(self, image_bytes: bytes) -> FaceEmbedding:
        img = self._decode(image_bytes)
        faces = self._model.get(img)
        if not faces:
            raise ValueError("No face detected in image")
        # Use the highest-confidence detection when multiple faces present
        face = max(faces, key=lambda f: f.det_score)
        return FaceEmbedding(embedding=face.embedding.tolist(), detection_score=float(face.det_score))

    def _compare(self, id_bytes: bytes, selfie_bytes: bytes) -> FaceMatchResult:
        id_faces = self._model.get(self._decode(id_bytes))
        selfie_faces = self._model.get(self._decode(selfie_bytes))

        if not id_faces:
            raise ValueError("No face detected in ID document image")
        if not selfie_faces:
            raise ValueError("No face detected in selfie image")

        id_emb = id_faces[0].embedding
        selfie_emb = selfie_faces[0].embedding

        similarity = float(
            np.dot(id_emb, selfie_emb)
            / (np.linalg.norm(id_emb) * np.linalg.norm(selfie_emb))
        )
        is_match = similarity >= SIMILARITY_THRESHOLD

        logger.info("Face match result", similarity=round(similarity, 4), is_match=is_match)
        return FaceMatchResult(
            is_match=is_match,
            similarity_score=round(similarity, 4),
            threshold=SIMILARITY_THRESHOLD,
            explanation=(
                f"Cosine similarity {similarity:.4f} "
                f"{'≥' if is_match else '<'} threshold {SIMILARITY_THRESHOLD}"
            ),
        )

    def _decode(self, image_bytes: bytes) -> np.ndarray:
        import cv2

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img

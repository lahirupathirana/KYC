import structlog

from app.core.config import Settings

logger = structlog.get_logger()


class ModelRegistry:
    """
    Central store for all AI model instances.

    Startup behaviour is controlled by settings.enabled_models (e.g. ["ocr"]).
    Any model that fails to load logs a warning and is skipped — the server
    starts regardless so that other endpoints still work.  Endpoints that call
    registry.get() for a skipped model will receive a RuntimeError (→ HTTP 500)
    at request time.

    Heavy library imports are deferred inside each _load_* method so the module
    can be imported without triggering CUDA / PaddlePaddle initialisation.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._models: dict = {}

    async def load_all(self) -> None:
        enabled = {m.lower() for m in self._settings.enabled_models}
        logger.info("Loading models", enabled=sorted(enabled))

        if "ocr" in enabled:
            await self._load_ocr()
        else:
            logger.info("OCR skipped (not in enabled_models)")

        if "face" in enabled:
            await self._load_face()
        else:
            logger.info("InsightFace skipped (not in enabled_models)")

        if "whisper" in enabled:
            await self._load_whisper()
        else:
            logger.info("Whisper skipped (not in enabled_models)")

    # ── Individual loaders ────────────────────────────────────────────────────

    async def _load_ocr(self) -> None:
        logger.info("Loading PaddleOCR")
        try:
            from paddleocr import PaddleOCR  # deferred — heavy import

            self._models["ocr"] = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=self._settings.use_gpu,
                show_log=False,
                det_limit_side_len=self._settings.ocr_det_limit_side_len,
                cpu_threads=self._settings.ocr_cpu_threads,
            )
            logger.info("PaddleOCR ready")
        except Exception as exc:
            logger.warning(
                "PaddleOCR failed to load — OCR endpoints will return 500",
                error=str(exc),
            )

    async def _load_face(self) -> None:
        logger.info("Loading InsightFace", model=self._settings.insightface_model)
        try:
            import insightface  # deferred

            root = self._settings.model_dir or None
            face_app = insightface.app.FaceAnalysis(
                name=self._settings.insightface_model,
                root=root,
            )
            ctx_id = self._settings.gpu_device_id if self._settings.use_gpu else -1
            face_app.prepare(ctx_id=ctx_id)
            self._models["face"] = face_app
            logger.info("InsightFace ready", ctx_id=ctx_id)
        except Exception as exc:
            logger.warning(
                "InsightFace failed to load — face endpoints will return 500",
                error=str(exc),
            )

    async def _load_whisper(self) -> None:
        logger.info("Loading Whisper", model=self._settings.whisper_model)
        try:
            import whisper  # deferred

            download_root = self._settings.model_dir or None
            self._models["whisper"] = whisper.load_model(
                self._settings.whisper_model,
                download_root=download_root,
            )
            logger.info("Whisper ready")
        except Exception as exc:
            logger.warning(
                "Whisper failed to load — voice endpoints will return 500",
                error=str(exc),
            )

    # ── Registry access ───────────────────────────────────────────────────────

    def get(self, name: str):
        if name not in self._models:
            raise RuntimeError(
                f"Model '{name}' is not loaded. "
                f"Add '{name}' to ENABLED_MODELS in .env and restart the service."
            )
        return self._models[name]

    async def unload_all(self) -> None:
        self._models.clear()

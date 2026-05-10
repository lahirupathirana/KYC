from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.core.config import settings
from app.models.model_registry import ModelRegistry

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting KYC AI Service", version=settings.app_version, gpu=settings.use_gpu)

    registry = ModelRegistry(settings)
    await registry.load_all()
    app.state.model_registry = registry

    logger.info("All models loaded — service ready")
    yield

    logger.info("Shutting down KYC AI Service")
    await registry.unload_all()

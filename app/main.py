from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Callable

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import v1_router
from app.core.config import settings
from app.core.lifespan import lifespan
from app.core.logging import configure_logging

logger = structlog.get_logger()


def create_app(lifespan_handler: Callable | None = None) -> FastAPI:
    """
    App factory — accepts an optional lifespan override for testing
    (avoids loading GPU models in unit tests).
    """
    configure_logging(debug=settings.debug)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan_handler or lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(v1_router, prefix="/api/v1")

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        logger.warning("Validation error", path=request.url.path, error=str(exc))
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
        logger.error("Runtime error", path=request.url.path, error=str(exc))
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return app


app = create_app()

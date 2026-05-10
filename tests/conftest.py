from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@asynccontextmanager
async def _mock_lifespan(app: FastAPI):
    """Bypass GPU model loading for unit tests."""
    app.state.model_registry = MagicMock()
    yield


@pytest.fixture
async def client() -> AsyncClient:
    app = create_app(lifespan_handler=_mock_lifespan)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

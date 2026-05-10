from fastapi import APIRouter, Depends

from app.core.config import Settings, settings
from app.core.dependencies import get_settings

router = APIRouter()


@router.get("/health", tags=["health"])
async def health(cfg: Settings = Depends(get_settings)) -> dict:
    return {"status": "ok", "service": cfg.app_name, "version": cfg.app_version}

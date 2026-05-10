from fastapi import APIRouter, Depends, File, UploadFile

from app.core.dependencies import get_liveness_service
from app.schemas.liveness import LivenessResult
from app.services.liveness_service import LivenessService

router = APIRouter()


@router.post("/check", response_model=LivenessResult, summary="Check liveness in a video frame")
async def check_liveness(
    frame: UploadFile = File(..., description="Single video frame (JPEG/PNG)"),
    service: LivenessService = Depends(get_liveness_service),
) -> LivenessResult:
    frame_bytes = await frame.read()
    return await service.check_liveness(frame_bytes)

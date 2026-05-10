from fastapi import APIRouter, Depends

from app.core.dependencies import get_scoring_service
from app.schemas.scoring import ScoringInput, ScoringResult
from app.services.scoring_service import ScoringService

router = APIRouter()


@router.post("/compute", response_model=ScoringResult, summary="Compute multi-modal KYC verification score")
async def compute_score(
    payload: ScoringInput,
    service: ScoringService = Depends(get_scoring_service),
) -> ScoringResult:
    return await service.compute_score(payload)

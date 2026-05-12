from fastapi import APIRouter, Depends, File, UploadFile

from app.core.dependencies import get_face_service
from app.schemas.face import FaceAnalysisResult, FaceEmbedding, FaceMatchResult
from app.services.face_service import FaceService

router = APIRouter()


@router.post(
    "/match",
    response_model=FaceMatchResult,
    summary="Compare face in ID document against a live selfie",
)
async def match_faces(
    id_document: UploadFile = File(..., description="ID document image containing a face photo"),
    selfie: UploadFile = File(..., description="Live selfie image"),
    service: FaceService = Depends(get_face_service),
) -> FaceMatchResult:
    id_bytes = await id_document.read()
    selfie_bytes = await selfie.read()
    return await service.match(id_bytes, selfie_bytes)


@router.post(
    "/analyze",
    response_model=FaceAnalysisResult,
    summary="Detect and analyse a face in a single image (quality + embedding)",
)
async def analyze_face(
    file: UploadFile = File(..., description="Image containing exactly one face"),
    service: FaceService = Depends(get_face_service),
) -> FaceAnalysisResult:
    return await service.analyse(await file.read())


@router.post(
    "/embedding",
    response_model=FaceEmbedding,
    summary="Extract a 512-dim ArcFace embedding from a single image",
)
async def get_embedding(
    file: UploadFile = File(...),
    service: FaceService = Depends(get_face_service),
) -> FaceEmbedding:
    return await service.get_embedding(await file.read())

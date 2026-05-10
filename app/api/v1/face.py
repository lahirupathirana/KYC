from fastapi import APIRouter, Depends, File, UploadFile

from app.core.dependencies import get_face_service
from app.schemas.face import FaceEmbedding, FaceMatchResult
from app.services.face_service import FaceService

router = APIRouter()


@router.post("/match", response_model=FaceMatchResult, summary="Match face in selfie against ID document")
async def match_faces(
    id_document: UploadFile = File(..., description="ID document image containing a face"),
    selfie: UploadFile = File(..., description="Live selfie image"),
    service: FaceService = Depends(get_face_service),
) -> FaceMatchResult:
    id_bytes = await id_document.read()
    selfie_bytes = await selfie.read()
    return await service.match_faces(id_bytes, selfie_bytes)


@router.post("/embedding", response_model=FaceEmbedding, summary="Extract face embedding from a single image")
async def get_embedding(
    file: UploadFile = File(...),
    service: FaceService = Depends(get_face_service),
) -> FaceEmbedding:
    return await service.get_embedding(await file.read())

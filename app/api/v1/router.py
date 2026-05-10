from fastapi import APIRouter

from app.api.v1 import face, health, liveness, ocr, scoring, voice

v1_router = APIRouter()

v1_router.include_router(health.router)
v1_router.include_router(ocr.router,      prefix="/ocr",      tags=["OCR"])
v1_router.include_router(face.router,     prefix="/face",     tags=["Face Recognition"])
v1_router.include_router(liveness.router, prefix="/liveness", tags=["Liveness Detection"])
v1_router.include_router(voice.router,    prefix="/voice",    tags=["Voice / ASR"])
v1_router.include_router(scoring.router,  prefix="/scoring",  tags=["Decision Scoring"])

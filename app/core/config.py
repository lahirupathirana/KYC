from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_name: str = "KYC AI Service"
    app_version: str = "0.1.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 20

    # GPU
    use_gpu: bool = Field(default=False, description="Enable GPU inference across all models")
    gpu_device_id: int = 0

    # Model paths — empty string means "use each library's default cache directory"
    # (e.g. ~/.paddleocr, ~/.insightface, ~/.cache/whisper).
    # Set to /models in Docker so weights persist in the named volume.
    model_dir: str = ""
    insightface_model: str = "buffalo_l"
    whisper_model: str = "base"

    # Which models to load on startup.
    # Use ["ocr"] during development to skip InsightFace and Whisper.
    # In .env: ENABLED_MODELS=["ocr"]
    enabled_models: list[str] = Field(default=["ocr", "face", "whisper"])

    # Security
    cors_origins: list[str] = ["*"]

    # PaddleOCR memory tuning
    # det_limit_side_len: max pixel side fed to detector (default 960).
    # Lower values reduce RAM. 640 saves ~30%; 480 saves ~45% but hurts accuracy on small text.
    ocr_det_limit_side_len: int = 960
    ocr_cpu_threads: int = 4   # set to your vCPU count

    # Face recognition thresholds (buffalo_l / ArcFace R100, cosine similarity)
    # LFW benchmark: TAR@FAR=0.1% ≈ 0.363; conservative defaults below.
    face_match_threshold: float = 0.40   # ≥ this → MATCH
    face_review_threshold: float = 0.20  # ≥ this → REVIEW; < this → NO_MATCH
    face_min_detection_score: float = 0.70
    face_min_size_px: int = 80           # minimum face side in pixels
    face_min_sharpness: float = 10.0     # ID card photos score 15-25; live selfies 40+
    face_max_pose_yaw: float = 35.0      # degrees
    face_max_pose_pitch: float = 30.0    # degrees

    # Inference
    max_image_size_mb: int = 10
    inference_timeout_seconds: int = 30


settings = Settings()

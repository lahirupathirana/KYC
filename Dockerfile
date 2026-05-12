# ════════════════════════════════════════════════════════════════════════════
#  Builder stage
# ════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv — copied wholesale into the runtime stage
RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"

WORKDIR /install
COPY requirements.txt .

# Pin setuptools to 69.5.1 — newer versions (75+) changed how pkg_resources
# is exposed inside exec() which breaks openai-whisper's legacy setup.py.
RUN pip install --upgrade pip "setuptools==69.5.1" wheel

# CPU-only ML frameworks — installed before requirements.txt so pip skips them.
RUN pip install paddlepaddle==2.6.2

RUN pip install onnxruntime==1.19.2

RUN pip install \
        torch==2.4.1 \
        torchvision==0.19.1 \
        torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cpu

# openai-whisper's setup.py imports pkg_resources at the top level.
# --no-build-isolation reuses the current venv (setuptools 69.5.1 with
# pkg_resources) instead of creating a fresh isolated build environment.
RUN pip install --no-build-isolation openai-whisper==20231117

# Remaining application requirements (whisper already satisfied above)
RUN pip install -r requirements.txt

# ════════════════════════════════════════════════════════════════════════════
#  Runtime stage — slim image, no compiler toolchain
# ════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 \
        ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built venv from builder — no pip install in runtime
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app
COPY --chown=appuser:appuser . .

# Model cache volume (mounted at runtime via docker-compose)
RUN mkdir -p /models && chown appuser:appuser /models

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "info"]

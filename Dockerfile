# ════════════════════════════════════════════════════════════════════════════
#  Builder stage — full CUDA devel image needed to compile native extensions
# ════════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Python 3.11 (Ubuntu 22.04 ships 3.10; deadsnakes PPA provides 3.11)
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
        build-essential \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
        ffmpeg \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv — copied wholesale into the runtime stage
RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"

WORKDIR /install
COPY requirements.txt .

# 1. GPU ML frameworks (must come first — version strings differ from CPU wheels)
RUN pip install \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121

RUN pip install onnxruntime-gpu==1.19.2

# PaddlePaddle GPU — adjust post-fix if CUDA version differs
RUN pip install paddlepaddle-gpu==2.6.1.post120 \
        -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html || \
    pip install paddlepaddle==2.6.1  # CPU fallback if GPU wheel unavailable

# 2. Application requirements (torch/onnxruntime already satisfied; pip skips re-install)
RUN pip install -r requirements.txt

# ════════════════════════════════════════════════════════════════════════════
#  Runtime stage — slimmer CUDA runtime image (no compiler toolchain)
# ════════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built venv from builder — no pip install in runtime
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app
COPY --chown=appuser:appuser . .

# Model cache volume (mounted at runtime)
RUN mkdir -p /models && chown appuser:appuser /models

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]

# KYC AI Service

FastAPI inference service for the **AI-Based Video KYC** MSc research prototype.
Handles OCR, face recognition, liveness detection, voice transcription, and decision scoring.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start — OCR Testing (Local)](#2-quick-start--ocr-testing-local)
3. [Environment Variables](#3-environment-variables)
4. [Running the Tests](#4-running-the-tests)
5. [API Reference](#5-api-reference)
   - [Health](#51-health)
   - [OCR — Synchronous](#52-ocr--synchronous-extract)
   - [OCR — Asynchronous](#53-ocr--asynchronous-extract)
   - [OCR — Job Poll](#54-ocr--poll-async-job)
   - [Face — Match](#55-face--match-faces)
   - [Face — Embedding](#56-face--get-embedding)
   - [Liveness — Check](#57-liveness--check)
   - [Voice — Transcribe](#58-voice--transcribe)
   - [Scoring — Compute](#59-scoring--compute-decision-score)
6. [Full Docker Stack](#6-full-docker-stack)
7. [Swagger UI](#7-swagger-ui)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

| Requirement | Minimum version | Notes |
|---|---|---|
| Python | 3.11 | Must be exactly 3.11 |
| Docker Desktop | any recent | Used for Redis only during local dev |
| pip | 23+ | `python -m pip install --upgrade pip` |

> **Windows note:** All commands below are PowerShell. If you use Git Bash, replace `$env:VAR="value"` with `export VAR="value"`.

---

## 2. Quick Start — OCR Testing (Local)

This gets the OCR pipeline running in ~10 minutes without needing a GPU,
without InsightFace, and without Whisper.

### Step 1 — Create a virtual environment

```powershell
# Run from inside ai_service/
cd D:\my\my\MSC\POC\ai_service

python -m venv .venv
.venv\Scripts\Activate.ps1
```

If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 2 — Install PaddlePaddle (CPU)

PaddlePaddle must be installed before the rest of the requirements.

```powershell
pip install paddlepaddle
```

> **GPU version:** See [requirements-gpu.txt](requirements-gpu.txt).
> For the CPU version, inference is slower but fully functional for testing.

### Step 3 — Install dev requirements

```powershell
pip install -r requirements-dev.txt
```

This installs FastAPI, PaddleOCR, OpenCV, Redis client, and test tools.
It does **not** install InsightFace or Whisper (not needed for OCR testing).

### Step 4 — Start Redis

The async OCR endpoint and RQ worker need Redis. The easiest way on Windows is Docker:

```powershell
docker run -d --name kyc-redis -p 6379:6379 redis:7.4-alpine
```

Verify it is running:
```powershell
docker exec kyc-redis redis-cli ping
# Expected: PONG
```

### Step 5 — Create the `.env` file

```powershell
Copy-Item .env.example .env
```

Then open `.env` and set these values for OCR-only local dev:

```env
DEBUG=true
USE_GPU=false
ENABLED_MODELS=["ocr"]
REDIS_URL=redis://localhost:6379/0
MODEL_DIR=
```

The key settings:
- `DEBUG=true` — enables Swagger UI at `/docs`
- `ENABLED_MODELS=["ocr"]` — only PaddleOCR loads at startup (fast, no InsightFace/Whisper needed)
- `MODEL_DIR=` — empty means PaddleOCR downloads models to its default `~/.paddleocr/` cache

### Step 6 — Start the server

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Expected startup output:
```
INFO     Loading models  enabled=['ocr']
INFO     Loading PaddleOCR
[2024/...] ppocr INFO: ...
INFO     PaddleOCR ready
INFO     InsightFace skipped (not in enabled_models)
INFO     Whisper skipped (not in enabled_models)
INFO     All models loaded — service ready
INFO     Application startup complete.
INFO     Uvicorn running on http://0.0.0.0:8000
```

> **First run only:** PaddleOCR downloads ~100 MB of model weights to `~/.paddleocr/`.
> This happens once and is cached for all future runs.

### Step 7 — Verify

```powershell
Invoke-WebRequest http://localhost:8000/api/v1/health | Select-Object -ExpandProperty Content
```

Expected:
```json
{"status":"ok","service":"KYC AI Service","version":"0.1.0"}
```

Open **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 3. Environment Variables

Full reference for `.env`:

```env
# ── Application ───────────────────────────────────────────────────────────────
APP_NAME=KYC AI Service
APP_VERSION=0.1.0
DEBUG=true                    # enables /docs Swagger UI
HOST=0.0.0.0
PORT=8000
WORKERS=1

# ── Model selection ───────────────────────────────────────────────────────────
# List of models to load on startup. Options: ocr, face, whisper
# OCR only:        ENABLED_MODELS=["ocr"]
# OCR + face:      ENABLED_MODELS=["ocr","face"]
# All:             ENABLED_MODELS=["ocr","face","whisper"]
ENABLED_MODELS=["ocr"]

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=20

# ── GPU ───────────────────────────────────────────────────────────────────────
USE_GPU=false                 # set true when GPU drivers + CUDA are available
GPU_DEVICE_ID=0

# ── Model cache ───────────────────────────────────────────────────────────────
# Empty = use each library's default (~/.paddleocr, ~/.insightface, ~/.cache/whisper)
# Docker = /models  (mounted named volume)
MODEL_DIR=

# ── Model names ───────────────────────────────────────────────────────────────
INSIGHTFACE_MODEL=buffalo_l
WHISPER_MODEL=base            # tiny | base | small | medium | large

# ── Security ──────────────────────────────────────────────────────────────────
CORS_ORIGINS=["http://localhost:3000"]

# ── Inference limits ──────────────────────────────────────────────────────────
MAX_IMAGE_SIZE_MB=10
INFERENCE_TIMEOUT_SECONDS=30
```

---

## 4. Running the Tests

The test suite does **not** require PaddleOCR, a GPU, or a running Redis instance.
All ML models and external services are mocked.

```powershell
# From ai_service/ with venv activated

# All tests
pytest -v

# OCR unit tests only (parser + preprocessing — pure Python, very fast)
pytest tests/ocr/test_preprocessing.py tests/ocr/test_nic_parser.py tests/ocr/test_passport_parser.py -v

# OCR API endpoint tests (mocked PaddleOCR)
pytest tests/ocr/test_ocr_endpoint.py -v

# Single test by name
pytest tests/ocr/test_nic_parser.py::test_old_nic_male_dob -v

# With stdout output (useful for debugging)
pytest -v -s
```

Expected output:
```
tests/ocr/test_nic_parser.py::test_old_nic_male_dob      PASSED
tests/ocr/test_nic_parser.py::test_old_nic_female_dob    PASSED
tests/ocr/test_nic_parser.py::test_new_nic_male_dob      PASSED
...
tests/ocr/test_preprocessing.py::test_decode_image_valid_png  PASSED
...
```

---

## 5. API Reference

Base URL: `http://localhost:8000/api/v1`

All file upload endpoints accept **JPEG, PNG, WebP, BMP**.
All responses are JSON.

---

### 5.1 Health

**`GET /health`**

```powershell
Invoke-WebRequest http://localhost:8000/api/v1/health
```

```bash
curl http://localhost:8000/api/v1/health
```

**Response `200`**
```json
{
  "status": "ok",
  "service": "KYC AI Service",
  "version": "0.1.0"
}
```

---

### 5.2 OCR — Synchronous Extract

**`POST /ocr/extract`**

Runs the full pipeline and returns structured fields in a single HTTP call.
Use this for individual document photos. Response time: 2–8 s on CPU.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file` | form-data file | ✅ | ID document image |
| `doc_type` | query string | ❌ | `passport` or `nic` — skips auto-detection |

```bash
# Auto-detect document type
curl -X POST http://localhost:8000/api/v1/ocr/extract \
  -F "file=@/path/to/nic.jpg"

# With explicit type hint (faster, more accurate)
curl -X POST "http://localhost:8000/api/v1/ocr/extract?doc_type=nic" \
  -F "file=@/path/to/nic.jpg"
```

```powershell
# PowerShell
$form = @{ file = Get-Item "C:\images\nic.jpg" }
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ocr/extract?doc_type=nic" `
  -Method Post -Form $form
```

**Response `200` — NIC example**
```json
{
  "quality": {
    "passed": true,
    "resolution_ok": true,
    "sharpness": 143.2,
    "brightness": 128.4,
    "contrast": 47.1,
    "issues": []
  },
  "document_type": "nic",
  "fields": {
    "document_type": "nic",
    "document_number": "890123456V",
    "full_name": "Kamal Bandara Perera",
    "dob": "1989-01-12",
    "sex": "M",
    "nationality": null,
    "expiry_date": null,
    "extraction_confidence": 1.0,
    "mrz_parsed": false
  },
  "raw_blocks": [
    {
      "text": "NATIONAL IDENTITY CARD",
      "confidence": 0.9876,
      "box": [[0,0],[220,0],[220,30],[0,30]]
    },
    {
      "text": "KAMAL BANDARA PERERA",
      "confidence": 0.9512,
      "box": [[0,60],[280,60],[280,90],[0,90]]
    },
    {
      "text": "890123456V",
      "confidence": 0.9923,
      "box": [[0,120],[180,120],[180,150],[0,150]]
    }
  ],
  "full_text": "NATIONAL IDENTITY CARD KAMAL BANDARA PERERA 890123456V ...",
  "average_confidence": 0.9437,
  "preprocessed": true
}
```

**Response `200` — Passport example**
```json
{
  "quality": {
    "passed": true,
    "resolution_ok": true,
    "sharpness": 187.5,
    "brightness": 135.2,
    "contrast": 52.8,
    "issues": []
  },
  "document_type": "passport",
  "fields": {
    "document_type": "passport",
    "document_number": "N1234567",
    "full_name": "Niroshan Kamal Mendis",
    "dob": "1988-09-01",
    "sex": "M",
    "nationality": "LKA",
    "expiry_date": "2025-12-31",
    "extraction_confidence": 1.0,
    "mrz_parsed": true
  },
  "raw_blocks": [...],
  "full_text": "REPUBLIC OF SRI LANKA PASSPORT N1234567 ...",
  "average_confidence": 0.9612,
  "preprocessed": true
}
```

**Response `422` — Quality failure example**
```json
{
  "detail": "Image resolution too low: Resolution 200×150 below minimum 500×320. Please upload a clearer photo of the document."
}
```

**How `extraction_confidence` is calculated:**
It is the fraction of the three core fields (document number, full name, date of birth) that were successfully extracted. `1.0` = all three found. Fields extracted from MRZ lines are more reliable than heuristic regex extraction (`mrz_parsed: true` vs `false`).

---

### 5.3 OCR — Asynchronous Extract

**`POST /ocr/extract/async`**

Returns immediately with a job ID. The full OCR pipeline runs in the Redis/RQ worker.
Use this for WebRTC frame uploads where you cannot wait for 5+ seconds.

```bash
curl -X POST "http://localhost:8000/api/v1/ocr/extract/async?doc_type=passport" \
  -F "file=@/path/to/passport.jpg"
```

**Response `202`**
```json
{
  "job_id": "a3f2c1d8-7e4b-4a9f-b123-56789abcdef0",
  "status": "queued",
  "poll_url": "http://localhost:8000/api/v1/ocr/jobs/a3f2c1d8-7e4b-4a9f-b123-56789abcdef0"
}
```

> **Note:** The async endpoint requires the Redis worker to be running.
> Start it with: `python -m app.workers.redis_worker`

---

### 5.4 OCR — Poll Async Job

**`GET /ocr/jobs/{job_id}`**

Poll until `status == "finished"`, then read the `result` field.

```bash
curl http://localhost:8000/api/v1/ocr/jobs/a3f2c1d8-7e4b-4a9f-b123-56789abcdef0
```

**Response — while processing**
```json
{
  "job_id": "a3f2c1d8-7e4b-4a9f-b123-56789abcdef0",
  "status": "started",
  "result": null,
  "error": null
}
```

**Response — when finished**
```json
{
  "job_id": "a3f2c1d8-7e4b-4a9f-b123-56789abcdef0",
  "status": "finished",
  "result": {
    "quality": { "passed": true, "resolution_ok": true, "sharpness": 143.2, "brightness": 128.4, "contrast": 47.1, "issues": [] },
    "document_type": "nic",
    "fields": {
      "document_number": "890123456V",
      "full_name": "Kamal Bandara Perera",
      "dob": "1989-01-12",
      "sex": "M",
      "extraction_confidence": 1.0,
      "mrz_parsed": false
    },
    "raw_blocks": [...],
    "full_text": "...",
    "average_confidence": 0.9437,
    "preprocessed": true
  },
  "error": null
}
```

**Response — on failure**
```json
{
  "job_id": "a3f2c1d8-7e4b-4a9f-b123-56789abcdef0",
  "status": "failed",
  "result": null,
  "error": "Could not decode image — unsupported format or corrupt data"
}
```

**Recommended polling interval:** 1–2 seconds. Typical OCR job takes 3–8 s on CPU.

---

### 5.5 Face — Match Faces

**`POST /face/match`**

> Requires `ENABLED_MODELS=["ocr","face"]` and InsightFace installed.

Compares a face in the ID document photo against a live selfie.

```bash
curl -X POST http://localhost:8000/api/v1/face/match \
  -F "id_document=@/path/to/nic.jpg" \
  -F "selfie=@/path/to/selfie.jpg"
```

**Response `200`**
```json
{
  "is_match": true,
  "similarity_score": 0.6823,
  "threshold": 0.4,
  "explanation": "Cosine similarity 0.6823 ≥ threshold 0.4"
}
```

**Response — no match**
```json
{
  "is_match": false,
  "similarity_score": 0.2134,
  "threshold": 0.4,
  "explanation": "Cosine similarity 0.2134 < threshold 0.4"
}
```

**Response `422` — no face detected**
```json
{
  "detail": "No face detected in ID document image"
}
```

---

### 5.6 Face — Get Embedding

**`POST /face/embedding`**

Returns the raw 512-dimensional face embedding vector. Useful for storing in a database and comparing later without re-running inference.

```bash
curl -X POST http://localhost:8000/api/v1/face/embedding \
  -F "file=@/path/to/photo.jpg"
```

**Response `200`**
```json
{
  "embedding": [-0.0234, 0.1823, -0.3421, 0.0912, ...],
  "detection_score": 0.9876
}
```

---

### 5.7 Liveness — Check

**`POST /liveness/check`**

> Requires `ENABLED_MODELS=["ocr","face"]`. Active challenge-response is Phase 5; this is currently a passive detection placeholder using InsightFace detection confidence.

```bash
curl -X POST http://localhost:8000/api/v1/liveness/check \
  -F "frame=@/path/to/frame.jpg"
```

**Response `200`**
```json
{
  "is_live": true,
  "confidence": 0.8921,
  "explanation": "Passive detection confidence 0.8921 (active liveness challenge pending — Phase 5)"
}
```

---

### 5.8 Voice — Transcribe

**`POST /voice/transcribe`**

> Requires `ENABLED_MODELS=["ocr","whisper"]` and Whisper installed.

Transcribes an audio response using OpenAI Whisper.

```bash
curl -X POST http://localhost:8000/api/v1/voice/transcribe \
  -F "audio=@/path/to/response.wav" \
  -F "language=en"
```

**Request fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `audio` | file | required | WAV, MP3, WebM, OGG |
| `language` | form text | `en` | BCP-47 language code (`en`, `si`, `ta`) |

**Response `200`**
```json
{
  "text": "My name is Kamal Perera and I was born on the twelfth of January nineteen eighty nine.",
  "language": "en",
  "segments": [
    { "start": 0.0, "end": 3.4, "text": "My name is Kamal Perera" },
    { "start": 3.4, "end": 7.1, "text": "and I was born on the twelfth of January nineteen eighty nine." }
  ]
}
```

---

### 5.9 Scoring — Compute Decision Score

**`POST /scoring/compute`**

The core research contribution. Aggregates scores from all verification pipelines into a single weighted score and returns an explainable decision.

This endpoint does **not** call any AI model — it is pure scoring logic and works immediately with any `ENABLED_MODELS` configuration.

**Request body (JSON):**

```bash
curl -X POST http://localhost:8000/api/v1/scoring/compute \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session-kyc-001",
    "ocr_confidence": 0.95,
    "face_similarity": 0.68,
    "liveness_confidence": 0.85,
    "voice_confidence": 0.72
  }'
```

**Request schema:**

| Field | Type | Required | Description |
|---|---|---|---|
| `session_id` | string | ✅ | Unique session identifier |
| `ocr_confidence` | float 0–1 | ❌ | `average_confidence` from OCR result |
| `face_similarity` | float 0–1 | ❌ | `similarity_score` from face match |
| `liveness_confidence` | float 0–1 | ❌ | `confidence` from liveness check |
| `voice_confidence` | float 0–1 | ❌ | transcript confidence from voice |

At least one field must be provided. Missing fields are excluded and the weights of present fields are re-normalised to sum to 1.0.

**Default weights:**

| Component | Weight |
|---|---|
| `ocr` | 25% |
| `face_match` | 35% |
| `liveness` | 30% |
| `voice` | 10% |

**Response `200` — APPROVED**
```json
{
  "session_id": "session-kyc-001",
  "final_score": 0.8234,
  "decision": "approved",
  "components": [
    {
      "component": "ocr",
      "raw_score": 0.95,
      "weight": 0.25,
      "contribution": 0.2375,
      "explanation": "Document text extracted with 95% average confidence"
    },
    {
      "component": "face_match",
      "raw_score": 0.68,
      "weight": 0.35,
      "contribution": 0.238,
      "explanation": "Face-to-ID cosine similarity: 0.6800"
    },
    {
      "component": "liveness",
      "raw_score": 0.85,
      "weight": 0.3,
      "contribution": 0.255,
      "explanation": "Liveness detection confidence: 85%"
    },
    {
      "component": "voice",
      "raw_score": 0.72,
      "weight": 0.1,
      "contribution": 0.072,
      "explanation": "Voice response transcription confidence: 72%"
    }
  ],
  "threshold": 0.75,
  "explanation": "Final score 0.8234 → APPROVED. Strongest signal: liveness (contribution 0.255). Weakest signal: voice (contribution 0.072)."
}
```

**Response `200` — OCR only (face/liveness/voice not yet done)**

Weights of absent components are redistributed. OCR gets 100% weight:
```json
{
  "session_id": "session-ocr-only",
  "final_score": 0.95,
  "decision": "approved",
  "components": [
    {
      "component": "ocr",
      "raw_score": 0.95,
      "weight": 1.0,
      "contribution": 0.95,
      "explanation": "Document text extracted with 95% average confidence"
    }
  ],
  "threshold": 0.75,
  "explanation": "Final score 0.95 → APPROVED. Strongest signal: ocr (contribution 0.95)."
}
```

**Decision thresholds:**

| Score range | Decision | Meaning |
|---|---|---|
| ≥ 0.75 | `approved` | Auto-approve |
| 0.55 – 0.74 | `review` | Route to human agent |
| < 0.55 | `rejected` | Auto-reject |

---

## 6. Full Docker Stack

Use this when testing the full pipeline (OCR + face + Redis worker) together.

### Step 1 — Copy and configure `.env`

```powershell
Copy-Item .env.example .env
```

Edit `.env` for full-stack Docker:
```env
DEBUG=true
USE_GPU=false
ENABLED_MODELS=["ocr","face","whisper"]
REDIS_URL=redis://redis:6379/0
MODEL_DIR=/models
```

### Step 2 — Remove GPU requirement (CPU-only)

Open `docker-compose.yml` and remove the `deploy:` block from both `ai_service` and `redis_worker` if you don't have an NVIDIA GPU:

```yaml
# DELETE these lines from ai_service and redis_worker:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Step 3 — Build and start

```powershell
docker-compose up --build
```

First build downloads all model weights (~500 MB). Subsequent starts use the `models_data` volume.

### Step 4 — Verify all services

```powershell
# API
curl http://localhost:8000/api/v1/health

# Redis
docker exec kyc_redis redis-cli ping
```

### Useful Docker commands

```powershell
# View logs
docker-compose logs -f ai_service

# Stop everything
docker-compose down

# Stop and remove volumes (clears model cache — triggers re-download)
docker-compose down -v

# Rebuild after code changes
docker-compose up --build ai_service
```

---

## 7. Swagger UI

When `DEBUG=true`, interactive API documentation is available at:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

You can test every endpoint directly from the browser — file uploads are supported.

---

## 8. Troubleshooting

### "paddlepaddle is not installed"
```
ModuleNotFoundError: No module named 'paddle'
```
**Fix:**
```powershell
pip install paddlepaddle
```

### Server starts but crashes loading models
```
RuntimeError: ... InsightFace failed ...
```
**Fix:** Set `ENABLED_MODELS=["ocr"]` in `.env`. InsightFace and Whisper require additional installation and are not needed for OCR testing.

### "Cannot connect to Redis"
```
redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379
```
**Fix:** Start Redis:
```powershell
docker run -d --name kyc-redis -p 6379:6379 redis:7.4-alpine
```

### PaddleOCR downloads models on every run
**Cause:** `MODEL_DIR` is set to a path that doesn't persist.
**Fix:** Leave `MODEL_DIR=` empty in `.env`. Models cache to `~/.paddleocr/` by default.

### PowerShell execution policy error
```
.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled
```
**Fix:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### OCR returns empty `text_blocks`
- Check `quality.passed` in the response — blurry or low-resolution images fail silently.
- Check `quality.issues` for specific failure reasons.
- Try a higher-resolution scan (minimum 500×320 pixels).
- Use `doc_type=nic` or `doc_type=passport` query parameter to skip auto-detection.

### `extraction_confidence` is 0.0
The document was detected but no fields were extracted. Causes:
1. Text is too small or blurry after preprocessing.
2. The document layout doesn't match expected patterns (non-standard NIC/passport).
3. OCR detected text but the field parsers found no matching patterns — check `full_text` in the response to see what was actually read.

### Docker GPU error on CPU machine
```
Error response from daemon: could not select device driver "nvidia"
```
**Fix:** Remove the `deploy:` block from `docker-compose.yml` as described in Step 2 of [Full Docker Stack](#6-full-docker-stack).

---

## Endpoint Summary

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/api/v1/health` | None | Service health check |
| POST | `/api/v1/ocr/extract` | None | Synchronous document OCR |
| POST | `/api/v1/ocr/extract/async` | None | Async OCR — returns job ID |
| GET | `/api/v1/ocr/jobs/{job_id}` | None | Poll async OCR job |
| POST | `/api/v1/face/match` | None | Match ID face vs selfie |
| POST | `/api/v1/face/embedding` | None | Extract face embedding vector |
| POST | `/api/v1/liveness/check` | None | Passive liveness detection |
| POST | `/api/v1/voice/transcribe` | None | Whisper audio transcription |
| POST | `/api/v1/scoring/compute` | None | Multi-modal KYC decision score |

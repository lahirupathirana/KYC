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
   - [Face — Analyze](#56-face--analyze-single-image)
   - [Face — Embedding](#57-face--get-embedding)
   - [Liveness — Check](#58-liveness--check)
   - [Voice — Transcribe](#59-voice--transcribe)
   - [Scoring — Compute](#510-scoring--compute-decision-score)
6. [Server Deployment — Contabo VPS](#6-server-deployment--contabo-vps)
   - [Server specs](#61-server-specs)
   - [Step 1 — SSH into the server](#62-step-1--ssh-into-the-server)
   - [Step 2 — Install Docker](#63-step-2--install-docker)
   - [Step 3 — Clone the repository](#64-step-3--clone-the-repository)
   - [Step 4 — Configure `.env`](#65-step-4--configure-env)
   - [Step 5 — Remove GPU from docker-compose.yml](#66-step-5--remove-gpu-from-docker-composeyml)
   - [Step 6 — Open the firewall](#67-step-6--open-the-firewall)
   - [Step 7 — Build and start](#68-step-7--build-and-start)
   - [Step 8 — Verify all models loaded](#69-step-8--verify-all-models-loaded)
   - [Step 9 — Test every endpoint](#610-step-9--test-every-endpoint)
   - [Step 10 — Redeploy after code changes](#611-step-10--redeploy-after-code-changes)
   - [Server management commands](#612-server-management-commands)
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

# ── Face recognition thresholds ──────────────────────────────────────────────
# Cosine similarity on L2-normalised 512-dim ArcFace (buffalo_l) embeddings.
FACE_MATCH_THRESHOLD=0.40      # >= this → MATCH verdict
FACE_REVIEW_THRESHOLD=0.20     # >= this (and < match) → REVIEW verdict
FACE_MIN_DETECTION_SCORE=0.70  # discard detections below this confidence
FACE_MIN_SIZE_PX=80            # reject faces smaller than 80×80 pixels
FACE_MAX_POSE_YAW=35.0         # maximum head turn (degrees)
FACE_MAX_POSE_PITCH=30.0       # maximum head tilt up/down (degrees)

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

# Face unit tests (no InsightFace required — model is mocked)
pytest tests/face/ -v

# Single face test by name
pytest tests/face/test_face_pipeline.py::test_match_faces_identical_embeddings_is_match -v

# Single test by name
pytest tests/ocr/test_nic_parser.py::test_old_nic_male_dob -v

# With stdout output (useful for debugging)
pytest -v -s
```

Expected output:
```
tests/ocr/test_nic_parser.py::test_old_nic_male_dob              PASSED
tests/ocr/test_nic_parser.py::test_old_nic_female_dob            PASSED
tests/ocr/test_nic_parser.py::test_new_nic_male_dob              PASSED
...
tests/ocr/test_preprocessing.py::test_decode_image_valid_png     PASSED
...
tests/face/test_face_quality.py::test_good_face_passes           PASSED
tests/face/test_face_quality.py::test_excessive_yaw_fails        PASSED
tests/face/test_face_pipeline.py::test_match_faces_identical_embeddings_is_match  PASSED
tests/face/test_face_pipeline.py::test_match_faces_review_band   PASSED
...
tests/face/test_face_endpoint.py::test_match_returns_match_verdict  PASSED
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

> Requires `ENABLED_MODELS=["ocr","face"]` and InsightFace installed (`pip install insightface onnxruntime`).

Compares the face in the ID document photo against a live selfie. Runs quality checks on both images before computing similarity and returns a three-tier verdict.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `id_document` | form-data file | ✅ | ID card or passport photo |
| `selfie` | form-data file | ✅ | Live selfie or video frame |

```bash
curl -X POST http://localhost:8000/api/v1/face/match \
  -F "id_document=@/path/to/nic.jpg" \
  -F "selfie=@/path/to/selfie.jpg"
```

```powershell
$form = @{
  id_document = Get-Item "C:\images\nic.jpg"
  selfie      = Get-Item "C:\images\selfie.jpg"
}
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/face/match" -Method Post -Form $form
```

**Response `200` — MATCH**
```json
{
  "verdict": "match",
  "is_match": true,
  "similarity_score": 0.6823,
  "threshold_used": 0.4,
  "review_threshold": 0.2,
  "id_quality": {
    "passed": true,
    "face_detected": true,
    "face_count": 1,
    "detection_score": 0.9876,
    "face_size_px": 142,
    "sharpness": 134.2,
    "brightness": 126.5,
    "pose_yaw": 4.1,
    "pose_pitch": 2.3,
    "pose_roll": -0.8,
    "pose_acceptable": true,
    "issues": []
  },
  "selfie_quality": {
    "passed": true,
    "face_detected": true,
    "face_count": 1,
    "detection_score": 0.9934,
    "face_size_px": 198,
    "sharpness": 187.6,
    "brightness": 131.0,
    "pose_yaw": -6.2,
    "pose_pitch": 1.7,
    "pose_roll": 0.4,
    "pose_acceptable": true,
    "issues": []
  },
  "id_face": {
    "bbox": [42.0, 18.0, 184.0, 160.0],
    "detection_score": 0.9876,
    "embedding": [-0.0234, 0.1823, -0.3421, 0.0912, "... 512 values total ..."]
  },
  "selfie_face": {
    "bbox": [30.0, 25.0, 228.0, 223.0],
    "detection_score": 0.9934,
    "embedding": [-0.0198, 0.1741, -0.3389, 0.0887, "... 512 values total ..."]
  },
  "explanation": "Similarity 0.6823 ≥ match threshold 0.4",
  "match_duration_ms": 312.4
}
```

**Response `200` — REVIEW** (similarity in the 0.20–0.40 band — route to human agent)
```json
{
  "verdict": "review",
  "is_match": false,
  "similarity_score": 0.3102,
  "threshold_used": 0.4,
  "review_threshold": 0.2,
  "explanation": "Similarity 0.3102 is in review band [0.2, 0.4)",
  "match_duration_ms": 298.7,
  "id_quality": { "passed": true, "..." : "..." },
  "selfie_quality": { "passed": true, "..." : "..." }
}
```

**Response `200` — NO_MATCH**
```json
{
  "verdict": "no_match",
  "is_match": false,
  "similarity_score": 0.0821,
  "threshold_used": 0.4,
  "review_threshold": 0.2,
  "explanation": "Similarity 0.0821 < review threshold 0.2",
  "match_duration_ms": 305.1,
  "id_quality": { "passed": true, "..." : "..." },
  "selfie_quality": { "passed": true, "..." : "..." }
}
```

**Response `200` — Quality failure** (no similarity computed)
```json
{
  "verdict": "no_match",
  "is_match": false,
  "similarity_score": null,
  "threshold_used": 0.4,
  "review_threshold": 0.2,
  "id_quality": {
    "passed": false,
    "face_detected": false,
    "face_count": 0,
    "issues": ["No face detected in image"]
  },
  "selfie_quality": { "passed": true, "..." : "..." },
  "explanation": "ID image quality check failed",
  "match_duration_ms": 41.2
}
```

**Verdict thresholds** (configurable in `.env`):

| Verdict | Condition | Action |
|---|---|---|
| `match` | similarity ≥ `FACE_MATCH_THRESHOLD` (0.40) | Auto-approve face check |
| `review` | `FACE_REVIEW_THRESHOLD` ≤ similarity < `FACE_MATCH_THRESHOLD` | Route to human agent |
| `no_match` | similarity < `FACE_REVIEW_THRESHOLD` (0.20) or quality failure | Fail face check |

> **Similarity metric:** Cosine similarity on L2-normalised 512-dim ArcFace embeddings from `buffalo_l`. Range: −1 to 1. Same person same-session photos typically score 0.55–0.90.

---

### 5.6 Face — Analyze Single Image

**`POST /face/analyze`**

> Requires `ENABLED_MODELS=["ocr","face"]` and InsightFace installed.

Detects and analyses a face in a single image: quality report, bounding box, detection score, and 512-dim embedding. Use this to pre-check image quality before a match, or to extract and store embeddings for later comparison.

```bash
curl -X POST http://localhost:8000/api/v1/face/analyze \
  -F "file=@/path/to/photo.jpg"
```

**Response `200` — face found**
```json
{
  "quality": {
    "passed": true,
    "face_detected": true,
    "face_count": 1,
    "detection_score": 0.9934,
    "face_size_px": 198,
    "sharpness": 187.6,
    "brightness": 131.0,
    "pose_yaw": -6.2,
    "pose_pitch": 1.7,
    "pose_roll": 0.4,
    "pose_acceptable": true,
    "issues": []
  },
  "face": {
    "bbox": [30.0, 25.0, 228.0, 223.0],
    "detection_score": 0.9934,
    "embedding": [-0.0198, 0.1741, -0.3389, 0.0887, "... 512 values total ..."]
  }
}
```

**Response `200` — quality failure** (face found but fails checks)
```json
{
  "quality": {
    "passed": false,
    "face_detected": true,
    "face_count": 1,
    "detection_score": 0.9102,
    "face_size_px": 52,
    "sharpness": 12.4,
    "brightness": 118.0,
    "pose_yaw": 41.0,
    "pose_pitch": 5.2,
    "pose_roll": 2.1,
    "pose_acceptable": false,
    "issues": [
      "Face too small (52px); minimum is 80px",
      "Face region is blurry (sharpness=12.4)",
      "Excessive yaw (41.0°); max is 35.0°"
    ]
  },
  "face": {
    "bbox": [90.0, 80.0, 142.0, 132.0],
    "detection_score": 0.9102,
    "embedding": ["... 512 values ..."]
  }
}
```

**Response `200` — no face detected**
```json
{
  "quality": {
    "passed": false,
    "face_detected": false,
    "face_count": 0,
    "issues": ["No face detected in image"]
  },
  "face": null
}
```

**Quality checks performed:**

| Check | Threshold (default) | Config key |
|---|---|---|
| Detection confidence | ≥ 0.70 | `FACE_MIN_DETECTION_SCORE` |
| Face size | ≥ 80px | `FACE_MIN_SIZE_PX` |
| Face sharpness (Laplacian) | ≥ 30 | — |
| Head yaw (left/right turn) | ≤ 35° | `FACE_MAX_POSE_YAW` |
| Head pitch (up/down tilt) | ≤ 30° | `FACE_MAX_POSE_PITCH` |

Multiple-face images are accepted — the highest-confidence face is used and a warning is added to `issues`.

---

### 5.7 Face — Get Embedding

**`POST /face/embedding`**

Returns the raw 512-dimensional ArcFace embedding vector for the best detected face. Returns HTTP 422 if no face is detected. Useful for storing an embedding in a database and running comparisons later without re-running inference.

```bash
curl -X POST http://localhost:8000/api/v1/face/embedding \
  -F "file=@/path/to/photo.jpg"
```

**Response `200`**
```json
{
  "embedding": [-0.0234, 0.1823, -0.3421, 0.0912, "... 512 values total ..."],
  "detection_score": 0.9876
}
```

**Response `422` — no face detected**
```json
{
  "detail": "No face detected in image"
}
```

---

### 5.8 Liveness — Check

**`POST /liveness/check`**

> Requires `ENABLED_MODELS=["ocr","face"]`. Active challenge-response is Phase 5 of the roadmap; this is currently a passive detection placeholder using InsightFace detection confidence.

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

### 5.9 Voice — Transcribe

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

### 5.10 Scoring — Compute Decision Score

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

## 6. Server Deployment — Contabo VPS

Full deployment guide for the **Contabo Cloud VPS 30 NVMe** used in this project.
All commands run on the server over SSH unless marked **[local]**.

---

### 6.1 Server Specs

| Resource | Value |
|---|---|
| CPU | 8 vCores |
| RAM | 24 GB |
| Disk | 200 GB NVMe SSD |
| OS | Ubuntu 22.04 LTS |
| GPU | None (CPU inference) |

**Memory budget at full stack:**

| Service | RAM |
|---|---|
| FastAPI + PaddleOCR (4 workers) | ~3.2 GB |
| InsightFace buffalo_l | ~1.2 GB |
| Whisper base | ~0.5 GB |
| Redis | ~50 MB |
| OS + Docker overhead | ~1.0 GB |
| **Total used** | **~6.0 GB** |
| **Free headroom** | **~18 GB** |

---

### 6.2 Step 1 — SSH into the server

```bash
# [local] — replace with your VPS IP from the Contabo control panel
ssh root@YOUR_VPS_IP
```

Once logged in, update the system:

```bash
apt-get update && apt-get upgrade -y
```

---

### 6.3 Step 2 — Install Docker

```bash
# Remove any old Docker packages
apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

# Install prerequisites
apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

# Add Docker's apt repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine and Compose plugin
apt-get update && apt-get install -y \
  docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Verify
docker --version
docker compose version
```

Expected output:
```
Docker version 27.x.x, build ...
Docker Compose version v2.x.x
```

---

### 6.4 Step 3 — Clone the repository

```bash
cd /opt

# Public repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git kyc

# Private repo — use a personal access token
git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/YOUR_REPO.git kyc

cd kyc/ai_service
```

---

### 6.5 Step 4 — Configure `.env`

```bash
cp .env.example .env
nano .env
```

Paste the full config below — tuned for 8 vCPU / 24 GB RAM:

```env
# ── Core ──────────────────────────────────────────────────────────────────────
DEBUG=true
HOST=0.0.0.0
PORT=8000
WORKERS=4

# ── Models ────────────────────────────────────────────────────────────────────
ENABLED_MODELS=["ocr","face","whisper"]

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_URL=redis://redis:6379/0
REDIS_MAX_CONNECTIONS=20

# ── GPU ───────────────────────────────────────────────────────────────────────
USE_GPU=false
GPU_DEVICE_ID=0

# ── Model cache (Docker named volume — persists across restarts) ───────────────
MODEL_DIR=/models
INSIGHTFACE_MODEL=buffalo_l
WHISPER_MODEL=base

# ── OCR — full quality, 8 threads ─────────────────────────────────────────────
OCR_DET_LIMIT_SIDE_LEN=960
OCR_CPU_THREADS=8

# ── Face thresholds ───────────────────────────────────────────────────────────
FACE_MATCH_THRESHOLD=0.40
FACE_REVIEW_THRESHOLD=0.20
FACE_MIN_DETECTION_SCORE=0.70
FACE_MIN_SIZE_PX=80
FACE_MAX_POSE_YAW=35.0
FACE_MAX_POSE_PITCH=30.0

# ── Security ──────────────────────────────────────────────────────────────────
CORS_ORIGINS=["*"]
MAX_IMAGE_SIZE_MB=10
INFERENCE_TIMEOUT_SECONDS=30
```

Save and exit: `Ctrl+O` → `Enter` → `Ctrl+X`

---

### 6.6 Step 5 — Remove GPU from `docker-compose.yml`

The VPS has no NVIDIA GPU. Open the compose file and delete the `deploy:` block from **both** `ai_service` and `redis_worker`:

```bash
nano docker-compose.yml
```

Find and delete these lines wherever they appear:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Also update the uvicorn command in the `ai_service` service to use 4 workers:

```yaml
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Save and exit.

---

### 6.7 Step 6 — Open the firewall

```bash
# Enable UFW if not already active
ufw allow OpenSSH
ufw enable

# Open the API port
ufw allow 8000/tcp
ufw status
```

> Also check the **Contabo control panel** → Your VPS → Firewall.
> If a cloud-level firewall is configured there, add an inbound rule for TCP port 8000.

---

### 6.8 Step 7 — Build and start

```bash
cd /opt/kyc/ai_service

docker compose up --build -d
```

**First build takes 10–20 minutes.** It installs all Python packages and downloads model weights (~580 MB total). Every subsequent start uses the cached `models_data` volume and takes ~30 seconds.

Watch the build progress:

```bash
docker compose logs -f
```

---

### 6.9 Step 8 — Verify all models loaded

```bash
docker compose logs ai_service | grep -E "ready|skipped|failed|error"
```

Expected — all three models confirmed ready:

```
INFO  Loading models  enabled=['ocr', 'face', 'whisper']
INFO  Loading PaddleOCR
INFO  PaddleOCR ready
INFO  Loading InsightFace  model=buffalo_l
INFO  InsightFace ready    ctx_id=-1
INFO  Loading Whisper      model=base
INFO  Whisper ready
INFO  All models loaded — service ready
INFO  Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Check all containers are running:

```bash
docker compose ps
```

```
NAME              STATUS          PORTS
ai_service        Up              0.0.0.0:8000->8000/tcp
redis             Up              6379/tcp
redis_worker      Up
```

---

### 6.10 Step 9 — Test every endpoint

Run all commands below from your **local machine** — replace `YOUR_VPS_IP` with your server IP.

#### Health check

```bash
curl http://YOUR_VPS_IP:8000/api/v1/health
```

```json
{"status": "ok", "service": "KYC AI Service", "version": "0.1.0"}
```

#### OCR — synchronous (NIC)

```bash
curl -X POST "http://YOUR_VPS_IP:8000/api/v1/ocr/extract?doc_type=nic" \
  -F "file=@/path/to/nic.jpg"
```

Expected: `document_type: "nic"`, `fields.document_number`, `fields.dob`, `quality.passed: true`

#### OCR — synchronous (passport)

```bash
curl -X POST "http://YOUR_VPS_IP:8000/api/v1/ocr/extract?doc_type=passport" \
  -F "file=@/path/to/passport.jpg"
```

Expected: `mrz_parsed: true`, `fields.nationality`, `fields.expiry_date`

#### OCR — async job

```bash
# Submit
JOB=$(curl -s -X POST "http://YOUR_VPS_IP:8000/api/v1/ocr/extract/async?doc_type=nic" \
  -F "file=@/path/to/nic.jpg" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

echo "Job ID: $JOB"

# Poll until finished
curl "http://YOUR_VPS_IP:8000/api/v1/ocr/jobs/$JOB"
```

Expected: `status: "queued"` → `"started"` → `"finished"` with full result.

> **Prerequisite:** the `redis_worker` container must be running (it is, if `docker compose up` included it).

#### Face — analyze single image

```bash
curl -X POST http://YOUR_VPS_IP:8000/api/v1/face/analyze \
  -F "file=@/path/to/selfie.jpg"
```

Expected: `quality.passed: true`, `quality.face_detected: true`, `face.embedding` (512 floats)

#### Face — match ID vs selfie

```bash
curl -X POST http://YOUR_VPS_IP:8000/api/v1/face/match \
  -F "id_document=@/path/to/nic.jpg" \
  -F "selfie=@/path/to/selfie.jpg"
```

Expected: `verdict: "match"` or `"review"` or `"no_match"`, `similarity_score`, `id_quality`, `selfie_quality`

#### Face — extract embedding

```bash
curl -X POST http://YOUR_VPS_IP:8000/api/v1/face/embedding \
  -F "file=@/path/to/photo.jpg"
```

Expected: `embedding` (512 floats), `detection_score`

#### Voice — transcribe

```bash
curl -X POST http://YOUR_VPS_IP:8000/api/v1/voice/transcribe \
  -F "audio=@/path/to/speech.wav" \
  -F "language=en"
```

Expected: `text`, `language`, `segments`

#### Scoring — compute KYC decision

```bash
curl -X POST http://YOUR_VPS_IP:8000/api/v1/scoring/compute \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session-001",
    "ocr_confidence": 0.95,
    "face_similarity": 0.68,
    "liveness_confidence": 0.85,
    "voice_confidence": 0.72
  }'
```

Expected: `decision: "approved"`, `final_score`, `components` with per-stage contributions

#### Swagger UI (browser)

Open in your browser — all endpoints are testable with file upload directly from the UI:

```
http://YOUR_VPS_IP:8000/docs
```

---

### 6.11 Step 10 — Redeploy after code changes

Each time you push new code to git, run this on the server:

```bash
cd /opt/kyc/ai_service
git pull
docker compose up --build -d
```

The `models_data` volume is preserved — model weights are never re-downloaded unless you explicitly remove the volume.

To deploy a specific branch:

```bash
git fetch origin
git checkout your-branch-name
docker compose up --build -d
```

---

### 6.12 Server management commands

```bash
# View live logs from all services
docker compose logs -f

# View logs from one service only
docker compose logs -f ai_service
docker compose logs -f redis_worker

# Show running containers + ports
docker compose ps

# Check memory usage
free -h
docker stats --no-stream

# Check disk usage
df -h
docker system df

# Restart a single service without rebuilding
docker compose restart ai_service

# Stop everything (preserves volumes)
docker compose down

# Stop and wipe model cache (forces re-download on next start)
docker compose down -v

# Access Redis CLI
docker compose exec redis redis-cli ping
docker compose exec redis redis-cli info memory

# Run a one-off command inside the container
docker compose exec ai_service python -c "from app.core.config import settings; print(settings.enabled_models)"
```

---

## 7. Swagger UI

When `DEBUG=true`, interactive API documentation is available at:

| Environment | Swagger UI | ReDoc |
|---|---|---|
| Local dev | http://localhost:8000/docs | http://localhost:8000/redoc |
| Contabo VPS | http://YOUR_VPS_IP:8000/docs | http://YOUR_VPS_IP:8000/redoc |

You can test every endpoint directly from the browser — file uploads are supported.

> Set `DEBUG=false` in `.env` to disable Swagger UI in production.

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
**Fix:** Remove the `deploy:` block from `docker-compose.yml` as described in [Step 5](#66-step-5--remove-gpu-from-docker-composeyml).

### Port 8000 not reachable from local machine
```
curl: (7) Failed to connect to YOUR_VPS_IP port 8000
```
**Fix — two places to check:**
1. OS firewall on the server: `ufw allow 8000/tcp && ufw reload`
2. Contabo cloud firewall: log in to the Contabo control panel → your VPS → Firewall → add inbound TCP rule for port 8000.

### `docker compose` command not found
```
docker: 'compose' is not a docker command
```
**Fix:** You have the old `docker-compose` (v1). Either install the Compose plugin (`apt-get install docker-compose-plugin`) or replace `docker compose` with `docker-compose` in all commands.

### InsightFace fails to download buffalo_l
```
insightface.utils.storage: Unable to download buffalo_l
```
**Fix:** The VPS needs outbound internet access on port 443. Test with `curl -I https://github.com`. If blocked, download the model zip manually and place it in the `models_data` volume path.

### Build fails with "no space left on device"
**Fix:** Clean unused Docker layers: `docker system prune -f`. The full build uses ~4 GB of layer cache; your 200 GB NVMe has plenty of room once old layers are pruned.

### Container exits immediately after start
```
docker compose ps  →  ai_service  Exited (1)
```
**Fix:** Check the logs for the actual error: `docker compose logs ai_service`. Common causes: syntax error in `.env`, missing required env var, or Python import error.

### Models re-download on every restart
**Cause:** `MODEL_DIR` is not set to `/models` so weights save inside the container layer (lost on restart).
**Fix:** Confirm `.env` has `MODEL_DIR=/models` and the `docker-compose.yml` mounts the `models_data` volume to `/models` in the container.

---

## Endpoint Summary

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/api/v1/health` | None | Service health check |
| POST | `/api/v1/ocr/extract` | None | Synchronous document OCR |
| POST | `/api/v1/ocr/extract/async` | None | Async OCR — returns job ID |
| GET | `/api/v1/ocr/jobs/{job_id}` | None | Poll async OCR job |
| POST | `/api/v1/face/match` | None | Match ID face vs selfie — returns MATCH / REVIEW / NO_MATCH verdict |
| POST | `/api/v1/face/analyze` | None | Detect face + quality report + 512-dim embedding |
| POST | `/api/v1/face/embedding` | None | Extract 512-dim ArcFace embedding vector |
| POST | `/api/v1/liveness/check` | None | Passive liveness detection |
| POST | `/api/v1/voice/transcribe` | None | Whisper audio transcription |
| POST | `/api/v1/scoring/compute` | None | Multi-modal KYC decision score |

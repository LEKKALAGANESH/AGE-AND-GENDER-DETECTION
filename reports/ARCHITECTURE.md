# Lite-Vision System Architecture

> Real-time age, gender, and emotion detection powered by SCRFD, InsightFace, FairFace, and FER+ ONNX models.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Frontend](#frontend)
4. [Backend](#backend)
5. [Data Flow](#data-flow)
6. [Inference Pipeline](#inference-pipeline)
7. [Concurrency and Caching](#concurrency-and-caching)
8. [Models](#models)
9. [Configuration](#configuration)
10. [Deployment](#deployment)

---

## Overview

Lite-Vision is a two-tier web application for real-time facial analysis. A **Next.js 15** frontend captures webcam frames and sends them to a **FastAPI** backend, which runs a multi-model inference pipeline and returns structured predictions. The frontend renders bounding boxes, labels, and confidence indicators as a canvas overlay on the live video feed.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           BROWSER                                   │
│                                                                     │
│  ┌──────────────┐    base64 frame     ┌──────────────────────────┐  │
│  │  getUserMedia │ ──────────────────► │  Camera.tsx              │  │
│  │  (webcam)     │                     │  ├─ useCamera.ts         │  │
│  └──────────────┘                      │  ├─ useAnalyze.ts        │  │
│                                        │  ├─ useTemporalSmoothing │  │
│        ┌───────────────────────────────│  └─ useFileUpload.ts     │  │
│        │  canvas overlay               └──────────┬───────────────┘  │
│        ▼                                          │                  │
│  ┌──────────────┐                                 │ POST /api/       │
│  │  canvas.ts   │   bounding boxes,               │ analyze          │
│  │  (draw)      │   labels, bars                  │                  │
│  └──────────────┘                                 │                  │
│                                                   │                  │
│  Next.js 15 · TypeScript · Tailwind CSS           │                  │
└───────────────────────────────────────────────────┼──────────────────┘
                                                    │
                                          HTTPS / JSON
                                                    │
┌───────────────────────────────────────────────────┼──────────────────┐
│                        BACKEND (FastAPI)           │                  │
│                                                   ▼                  │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                     Middleware Stack                             │ │
│  │  CORS · Correlation ID (X-Request-ID) · Rate Limiting (slowapi) │ │
│  └──────────────────────────────┬──────────────────────────────────┘ │
│                                 │                                    │
│                                 ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                   /api/analyze  ·  /api/upload                  │ │
│  │                   /api/health                                   │ │
│  └──────────────────────────────┬──────────────────────────────────┘ │
│                                 │                                    │
│                                 ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                  Inference Pipeline                              │ │
│  │                                                                 │ │
│  │  base64 decode ► magic byte validation ► SHA-256 cache check    │ │
│  │  ► image decode ► resize ► SCRFD detection ► multi-crop ensemble│ │
│  │  ► FairFace fusion ► emotion detection ► JSON response          │ │
│  └──────────────────────────────┬──────────────────────────────────┘ │
│                                 │                                    │
│              ┌──────────────────┼──────────────────┐                 │
│              ▼                  ▼                   ▼                 │
│  ┌────────────────┐ ┌──────────────────┐ ┌──────────────────┐       │
│  │  app.state      │ │  LRU Cache       │ │  Semaphore       │       │
│  │                 │ │  (OrderedDict)   │ │  (4 concurrent)  │       │
│  │  face_detector  │ │  100 entries     │ │                  │       │
│  │  genderage_net  │ │  60s TTL         │ │  asyncio.to_     │       │
│  │  emotion_net    │ │                  │ │  thread for CPU  │       │
│  │  fairface_net   │ │                  │ │  work            │       │
│  └────────────────┘ └──────────────────┘ └──────────────────┘       │
│                                                                      │
│  Python 3.14 · OpenCV DNN · ONNX · pydantic-settings                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Frontend

**Stack:** Next.js 15, TypeScript, Tailwind CSS

### Component Architecture

| File | Role |
|------|------|
| `Camera.tsx` | Main component. Manages webcam lifecycle, orchestrates analysis loop, renders video and canvas overlay. |
| `useCamera.ts` | Hook encapsulating `getUserMedia` access and video element binding. |
| `useAnalyze.ts` | Hook that sends base64 frames to the backend and manages request/response state. |
| `useTemporalSmoothing.ts` | Temporal voting buffer for stable predictions. Tracks faces across frames via IoU matching, applies EMA smoothing (alpha=0.3) for age, and majority voting over a sliding window for gender. Removes stale tracks after 10 missed frames. |
| `useFileUpload.ts` | Hook for static image upload via `/api/upload`. |
| `canvas.ts` | Utility that draws bounding boxes, age/gender labels, and confidence bars onto a canvas overlay aligned with the video feed. |
| `types.ts` | TypeScript interfaces: `FaceResult`, `AnalyzeResponse`, plus helper functions `getAgeRange` and `getGenderConfidence`. |
| `constants.ts` | Shared constants (API base URL, default thresholds). |
| `api.ts` | HTTP client wrapper for backend communication. |
| `error.tsx` | Error boundary page. |
| `loading.tsx` | Loading skeleton page. |
| `not-found.tsx` | 404 page. |

### Key Data Types

```typescript
interface FaceResult {
  age: number;
  age_min?: number;
  age_max?: number;
  gender: string;
  gender_confidence?: number;
  confidence: number;
  region: [number, number, number, number];  // normalized [x, y, w, h]
  emotion?: string;
  emotion_confidence?: number;
}

interface AnalyzeResponse {
  results: FaceResult[];
  face_count: number;
  processing_time_ms: number;
}
```

---

## Backend

**Stack:** FastAPI, Python 3.14, OpenCV DNN, ONNX Runtime (via cv2.dnn)

### File Structure

The backend is implemented as a single-file module (`main.py`, ~992 lines) containing all inference logic, model management, middleware, and API routes.

### API Endpoints

| Method | Path | Rate Limit | Description |
|--------|------|------------|-------------|
| `GET` | `/api/health` | None | Returns model load status and model metadata. |
| `POST` | `/api/analyze` | 300/min | Accepts a base64-encoded image, returns face analysis results. |
| `POST` | `/api/upload` | 30/min | Accepts a multipart image file upload, returns face analysis results. |

### Application Lifecycle

The backend uses FastAPI's lifespan context manager:

1. **Startup:** Downloads all four ONNX model files (if not already present) and loads them into `app.state`.
2. **Shutdown:** Releases model references by setting `app.state` attributes to `None`.

### Middleware Stack

1. **CORS Middleware** -- Configurable allowed origins (default: `*`).
2. **Correlation ID Middleware** -- Attaches a unique `X-Request-ID` header to every request/response for traceability.
3. **Rate Limiting** -- Powered by `slowapi`, applied per-endpoint with per-IP tracking.
4. **Structured JSON Logging** -- All log output is formatted as JSON with timestamp, level, message, and optional request ID.

---

## Data Flow

```
  Browser                          Network                        Backend
  ───────                          ───────                        ───────

  1. Webcam captures frame
     │
     ▼
  2. Canvas → base64 encode
     │
     ▼
  3. POST /api/analyze ──────────── HTTPS ──────────────► 4. Receive request
     { "image": "data:image/               │                  │
       jpeg;base64,..." }                  │                  ▼
                                           │              5. base64 decode
                                           │              6. Magic byte validation
                                           │              7. SHA-256 hash → cache lookup
                                           │                  │
                                           │           cache hit?
                                           │           ┌──yes──┴──no──┐
                                           │           │              ▼
                                           │           │          8. Decode image
                                           │           │          9. Resize if needed
                                           │           │         10. SCRFD face detect
                                           │           │         11. Per-face pipeline:
                                           │           │             - ArcFace alignment
                                           │           │             - Multi-crop ensemble
                                           │           │             - FairFace fusion
                                           │           │             - Emotion detection
                                           │           │         12. Cache result
                                           │           │              │
                                           │           └──────┬───────┘
                                           │                  ▼
  14. Draw canvas overlay ◄──── HTTPS ◄── 13. Return JSON response
      - Bounding boxes                     { results: [...],
      - Age / Gender labels                  face_count: N,
      - Confidence bars                      processing_time_ms: T }
      - Emotion labels

  15. useTemporalSmoothing
      - IoU face tracking
      - EMA age smoothing
      - Gender majority vote
```

---

## Inference Pipeline

The inference pipeline runs inside `asyncio.to_thread` to avoid blocking the event loop. Each request passes through these stages:

### Stage 1: Input Validation
- Base64 decode the image payload.
- Validate magic bytes (JPEG `FF D8 FF` or PNG `89 50 4E 47`).
- Compute SHA-256 hash for cache lookup.

### Stage 2: Image Preprocessing
- Decode bytes to a NumPy array via `cv2.imdecode`.
- Resize if any dimension exceeds `max_image_dimension` (default: 4096px).

### Stage 3: Face Detection (SCRFD)
- The `SCRFDDetector` class wraps an SCRFD 10G KPS model loaded via `cv2.dnn`.
- Applies CLAHE preprocessing for improved detection in low-light conditions.
- Outputs per-face bounding boxes `[x1, y1, x2, y2, score]` and 5-point facial landmarks.

### Stage 4: Per-Face Analysis

For each detected face:

1. **ArcFace Alignment** -- Uses the 5-point landmarks to compute a similarity transform, aligning the face crop to the standard 96x96 ArcFace template.

2. **Multi-Crop Ensemble (InsightFace genderage)** -- Runs age/gender inference on 3 crop variants (aligned, center-crop, padded) for robust prediction. Age is regression-based (continuous). Gender uses softmax confidence.

3. **FairFace Fusion** -- A secondary gender prediction from the FairFace ResNet34 model, trained on a racially balanced dataset. The final gender decision fuses InsightFace and FairFace outputs using a weighted scheme, improving accuracy across demographic groups. FairFace also provides an independent age estimate used for cross-validation.

4. **Emotion Detection (FER+)** -- The `emotion-ferplus-8.onnx` model classifies facial expressions into one of 8 categories (neutral, happiness, surprise, sadness, anger, disgust, fear, contempt).

### Stage 5: Response Assembly
- Bounding box coordinates are normalized to `[0, 1]` relative to the image dimensions.
- Results are assembled into the `AnalyzeResponse` schema and cached.

---

## Concurrency and Caching

### Concurrency Control

| Mechanism | Detail |
|-----------|--------|
| `asyncio.Semaphore` | Limits concurrent inference to 4 (configurable via `LITEVISION_MAX_CONCURRENT_INFERENCES`). Prevents CPU/memory exhaustion under load. |
| `asyncio.to_thread` | Offloads CPU-bound OpenCV and NumPy operations to a thread pool, keeping the async event loop responsive. |

### LRU Cache

| Property | Value |
|----------|-------|
| Data Structure | `collections.OrderedDict` |
| Max Entries | 100 (configurable via `LITEVISION_MAX_CACHE_SIZE`) |
| TTL | 60 seconds (configurable via `LITEVISION_CACHE_TTL_SECONDS`) |
| Key | SHA-256 hash of raw image bytes |
| Eviction | Oldest entry removed when capacity is exceeded; expired entries pruned on access. |

The cache is module-level and protected by the GIL, so no explicit lock is required for dictionary operations.

---

## Models

| Model | File | Purpose | Source |
|-------|------|---------|--------|
| SCRFD 10G KPS | `scrfd_10g_kps.onnx` | Face detection with 5-point landmarks. WIDERFace Hard AP 82.8%. | GitHub (HeyGem release) |
| InsightFace GenderAge | `genderage.onnx` | Regression-based continuous age + softmax gender. | HuggingFace (InsightFace buffalo_l) |
| FairFace ResNet34 | `fairface.onnx` | Racially balanced gender/age/race classification. | GitHub (fairface-onnx release) |
| FER+ | `emotion-ferplus-8.onnx` | Facial expression recognition (8 emotions). | ONNX Model Zoo |

All models are ONNX format and loaded via `cv2.dnn.readNetFromONNX`. Total size is approximately 102 MB. Models are auto-downloaded on first startup to the configured `model_dir`.

---

## Configuration

All settings are managed via `pydantic-settings` with the environment variable prefix `LITEVISION_`.

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `LITEVISION_CONFIDENCE_THRESHOLD` | `float` | `0.7` | Minimum face detection confidence. |
| `LITEVISION_MAX_CACHE_SIZE` | `int` | `100` | Maximum number of cached inference results. |
| `LITEVISION_CORS_ORIGINS` | `list[str]` | `["*"]` | Allowed CORS origins. |
| `LITEVISION_MODEL_DIR` | `str` | `./models` | Directory for downloaded ONNX models. |
| `LITEVISION_CACHE_TTL_SECONDS` | `int` | `60` | Cache entry time-to-live in seconds. |
| `LITEVISION_MAX_IMAGE_DIMENSION` | `int` | `4096` | Maximum allowed image dimension (width or height). |
| `LITEVISION_MAX_CONCURRENT_INFERENCES` | `int` | `4` | Maximum parallel inference operations. |

---

## Deployment

### Production Architecture

```
                    ┌──────────────────────┐
                    │      Vercel CDN      │
                    │  (Edge Network)      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Next.js 15 Frontend │
                    │  (Vercel Hosting)    │
                    │                      │
                    │  Static assets +     │
                    │  SSR pages           │
                    └──────────┬───────────┘
                               │
                         HTTPS / JSON
                               │
                    ┌──────────▼───────────┐
                    │  FastAPI Backend      │
                    │  (Railway / Docker)   │
                    │                      │
                    │  ONNX inference       │
                    │  4 models in memory   │
                    │  ~102 MB models       │
                    └──────────────────────┘
```

### Frontend Deployment (Vercel)
- Deployed as a standard Next.js application on Vercel.
- Static assets served from Vercel's edge CDN.
- Environment variable points the frontend API client to the backend URL.

### Backend Deployment (Railway or Docker)
- Runs as a containerized FastAPI application.
- A `Dockerfile` is provided for container builds.
- Models are downloaded automatically on first startup and persisted in the configured model directory.
- Requires sufficient memory for 4 ONNX models loaded simultaneously (~200 MB RSS minimum recommended).

### Startup Sequence

1. FastAPI lifespan handler initializes.
2. Each model file is checked against the local `model_dir`.
3. Missing models are downloaded from their respective URLs.
4. All four models are loaded into `app.state` via `cv2.dnn`.
5. The application begins accepting requests.

---

*Document generated for the Lite-Vision project.*

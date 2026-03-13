"""Lite-Vision — Real-Time Age & Gender Detection API

Models:
  - Face detection: YuNet (ONNX) via cv2.FaceDetectorYN
  - Age + Gender:   InsightFace genderage.onnx via cv2.dnn — regression-based continuous age
"""

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import os
import time
import urllib.request
import uuid
import zipfile
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Simple JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        # Include request_id if present on the record
        if hasattr(record, "request_id"):
            log_entry["request_id"] = getattr(record, "request_id")
        return json.dumps(log_entry)


logger = logging.getLogger("lite-vision")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(JSONFormatter())
logger.addHandler(_handler)
logger.propagate = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """App settings sourced from environment variables with sensible defaults."""

    confidence_threshold: float = 0.7
    max_cache_size: int = 100
    cors_origins: list[str] = ["*"]
    model_dir: str = os.path.join(os.path.dirname(__file__), "models")
    cache_ttl_seconds: int = 60
    max_image_dimension: int = 4096
    max_concurrent_inferences: int = 4

    model_config = {"env_prefix": "LITEVISION_"}


settings = Settings()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS = {
    "face_detection_yunet_2023mar.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    },
    "genderage.onnx": {
        "url": "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/genderage.onnx",
    },
    "emotion-ferplus-8.onnx": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
    },
}

# Standard ArcFace alignment template scaled from 112x112 to 96x96
ARCFACE_DST_96 = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # left mouth
    [70.7299, 92.2041],  # right mouth
], dtype=np.float32) * (96.0 / 112.0)

# Image magic bytes for validation
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG"

# Module-level cache (shared across requests; protected by the GIL)
# Each entry stores {"data": <response dict>, "timestamp": <float>}
cache: OrderedDict[str, dict] = OrderedDict()

# Concurrency semaphore for inference
_inference_semaphore: asyncio.Semaphore | None = None


def _get_inference_semaphore() -> asyncio.Semaphore:
    """Lazily create the semaphore (must be created inside a running event loop)."""
    global _inference_semaphore
    if _inference_semaphore is None:
        _inference_semaphore = asyncio.Semaphore(settings.max_concurrent_inferences)
    return _inference_semaphore


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------


def download_models() -> None:
    """Download model files that are missing or empty."""
    os.makedirs(settings.model_dir, exist_ok=True)

    for filename, info in MODELS.items():
        path = os.path.join(settings.model_dir, filename)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue

        if "extract_from_zip" in info:
            # Download zip, extract the single file we need, delete zip
            zip_path = path + ".zip"
            logger.info("Downloading %s (from zip) ...", filename)
            urllib.request.urlretrieve(info["url"], str(zip_path))
            zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
            logger.info("  -> downloaded zip (%.1f MB), extracting %s ...", zip_size_mb, info["extract_from_zip"])

            with zipfile.ZipFile(zip_path, "r") as zf:
                member = info["extract_from_zip"]
                with zf.open(member) as src, open(path, "wb") as dst:
                    dst.write(src.read())

            os.remove(zip_path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            logger.info("  -> saved %s (%.1f MB)", filename, size_mb)
        else:
            # Direct download
            logger.info("Downloading %s ...", filename)
            urllib.request.urlretrieve(info["url"], str(path))
            size_mb = os.path.getsize(path) / (1024 * 1024)
            logger.info("  -> saved %s (%.1f MB)", filename, size_mb)


def load_models(app: FastAPI) -> None:
    """Download (if needed) and load YuNet face detector + InsightFace genderage model."""
    download_models()

    # Face detector — OpenCV's built-in YuNet API
    app.state.face_detector = cv2.FaceDetectorYN.create(
        os.path.join(settings.model_dir, "face_detection_yunet_2023mar.onnx"),
        "",
        (0, 0),  # input size set per-image before detection
        score_threshold=settings.confidence_threshold,
        nms_threshold=0.3,
        top_k=5000,
    )

    # Age + Gender — single ONNX model (regression-based continuous age)
    app.state.genderage_net = cv2.dnn.readNetFromONNX(
        os.path.join(settings.model_dir, "genderage.onnx")
    )

    # Emotion detection — FER+ ONNX model (expression-aware gender correction)
    app.state.emotion_net = cv2.dnn.readNetFromONNX(
        os.path.join(settings.model_dir, "emotion-ferplus-8.onnx")
    )

    logger.info("All models loaded (YuNet + InsightFace genderage + FER+ emotion)")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup — graceful fallback if anything fails."""
    app.state.face_detector = None
    app.state.genderage_net = None
    app.state.emotion_net = None
    try:
        load_models(app)
    except Exception as exc:
        logger.error("Model loading failed: %s", exc)
        # Don't crash — the app will return 503 on /api/analyze
    yield


app = FastAPI(
    title="Lite-Vision",
    description="Real-time age and gender detection API powered by YuNet + InsightFace ONNX models.",
    version="3.0.0",
    lifespan=lifespan,
)

# Attach rate limiter
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Try again later."},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request correlation ID middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Attach a correlation ID to every request/response cycle."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    # Store on request state so handlers can access it
    request.state.request_id = request_id

    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    image: str = Field(..., max_length=10_000_000)


class FaceResult(BaseModel):
    age: int                    # continuous regression-based predicted age
    age_min: int                # lower bound estimate (age - 3)
    age_max: int                # upper bound estimate (age + 3)
    gender: str                 # "Male" or "Female"
    gender_confidence: float    # softmax probability (0.0-1.0)
    confidence: float           # face detection confidence (YuNet)
    region: list[float]         # normalized [x, y, w, h]
    emotion: str | None = None           # detected expression (if emotion model loaded)
    emotion_confidence: float | None = None  # expression detection confidence


class AnalyzeResponse(BaseModel):
    results: list[FaceResult]
    face_count: int
    processing_time_ms: float


class ErrorResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _validate_magic_bytes(img_bytes: bytes) -> None:
    """Check that image bytes start with a known magic signature."""
    if img_bytes[:3] == JPEG_MAGIC:
        return
    if img_bytes[:4] == PNG_MAGIC:
        return
    raise HTTPException(
        status_code=422,
        detail="Unsupported image format. Only JPEG and PNG are accepted.",
    )


def _resize_if_needed(img: np.ndarray) -> np.ndarray:
    """Resize the image proportionally if either dimension exceeds the limit."""
    h, w = img.shape[:2]
    max_dim = settings.max_image_dimension
    if h <= max_dim and w <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    logger.info(
        "Resizing image from %dx%d to %dx%d (exceeded %dpx limit)",
        w, h, new_w, new_h, max_dim,
    )
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _check_cache(digest: str) -> dict | None:
    """Return cached response if present and not expired, else None."""
    if digest not in cache:
        return None
    entry = cache[digest]
    age = time.monotonic() - entry["timestamp"]
    if age > settings.cache_ttl_seconds:
        # Expired — remove and miss
        del cache[digest]
        return None
    cache.move_to_end(digest)
    return entry["data"]


def _store_cache(digest: str, data: dict) -> None:
    """Store a response in the LRU cache with a timestamp."""
    cache[digest] = {"data": data, "timestamp": time.monotonic()}
    if len(cache) > settings.max_cache_size:
        cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Face alignment
# ---------------------------------------------------------------------------


def _align_face(img: np.ndarray, landmarks_5: list[tuple[float, float]]) -> np.ndarray | None:
    """Align face using 5 landmarks with similarity transform to 96x96."""
    src = np.array(landmarks_5, dtype=np.float32)
    dst = ARCFACE_DST_96
    M = cv2.estimateAffinePartial2D(src, dst)[0]
    if M is None:
        return None
    aligned = cv2.warpAffine(img, M, (96, 96))
    return aligned



# ---------------------------------------------------------------------------
# Softmax utility
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax over a 1-D array."""
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


# ---------------------------------------------------------------------------
# Emotion detection helper
# ---------------------------------------------------------------------------


def _detect_emotion(face_roi: np.ndarray, emotion_net: cv2.dnn.Net) -> tuple[str, float]:
    """Detect emotion from a face crop using FER+ ONNX model.

    Returns (emotion_label, confidence).
    FER+ input: 64x64 grayscale, output: 8-class probabilities.
    Classes: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt
    """
    EMOTION_LABELS = ["neutral", "happiness", "surprise", "sadness",
                      "anger", "disgust", "fear", "contempt"]

    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    # Raw pixels — emotion-ferplus-8.onnx has normalization baked into the graph.
    blob = cv2.dnn.blobFromImage(resized, 1.0, (64, 64), 0, swapRB=False)
    emotion_net.setInput(blob)
    output = emotion_net.forward()
    probs = _softmax(output[0])
    idx = int(np.argmax(probs))
    return EMOTION_LABELS[idx], float(probs[idx])


# ---------------------------------------------------------------------------
# Multi-crop ensemble for robust age + gender
# ---------------------------------------------------------------------------


def _multi_crop_ensemble(
    img: np.ndarray,
    fx: int, fy: int, fw: int, fh: int,
    landmarks: list[tuple[float, float]],
    genderage_net: cv2.dnn.Net,
) -> tuple[np.ndarray, float]:
    """Run genderage inference on 3 crop variants for robust gender detection.

    Variant 1: landmark-aligned face (best quality — used for age).
    Variant 2: 15%-padded raw crop resized to 96x96 (different framing).
    Variant 3: 25%-padded raw crop resized to 96x96 (wider context).

    Age uses ONLY the aligned face prediction because the InsightFace model
    was trained on ArcFace-aligned inputs — non-aligned crops give inaccurate ages.
    Gender uses the averaged softmax across all 3 variants for expression robustness.

    Returns (averaged_gender_probs, age_raw).
    Gender logit convention: [0]=Male, [1]=Female.
    """
    h, w = img.shape[:2]
    all_gender_probs = []
    aligned_age: float | None = None

    def _infer(face_96: np.ndarray) -> tuple[np.ndarray, float]:
        # Raw pixels — genderage.onnx has normalization baked into the ONNX graph
        # (internal Sub/Mul nodes). External normalization would double-normalize,
        # compressing all inputs to a tiny range and producing identical outputs.
        blob = cv2.dnn.blobFromImage(
            face_96, 1.0, (96, 96), (0, 0, 0), swapRB=True,
        )
        genderage_net.setInput(blob)
        output = genderage_net.forward()
        gender_logits = output[0][0:2]
        gender_probs = _softmax(gender_logits)
        age_raw = float(output[0][2])
        return gender_probs, age_raw

    # Variant 1: landmark-aligned face (primary — best for age)
    aligned = _align_face(img, landmarks)
    if aligned is not None:
        gender_probs, aligned_age = _infer(aligned)
        all_gender_probs.append(gender_probs)

    # Variants 2 & 3: padded raw crops (gender ensemble only, age ignored)
    for pad_frac in [0.15, 0.25]:
        pad_w = int(fw * pad_frac)
        pad_h = int(fh * pad_frac)
        x1 = max(0, fx - pad_w)
        y1 = max(0, fy - pad_h)
        x2 = min(w, fx + fw + pad_w)
        y2 = min(h, fy + fh + pad_h)
        face_roi = img[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue
        gender_probs, crop_age = _infer(cv2.resize(face_roi, (96, 96)))
        all_gender_probs.append(gender_probs)
        # Use padded crop age only as fallback if alignment failed
        if aligned_age is None:
            aligned_age = crop_age

    if not all_gender_probs:
        return np.array([0.5, 0.5]), 0.25

    avg_gender_probs = np.mean(all_gender_probs, axis=0)
    # Age: aligned face prediction (or first available fallback)
    final_age = aligned_age if aligned_age is not None else 0.25

    return avg_gender_probs, final_age


# ---------------------------------------------------------------------------
# Inference (CPU-bound, run in a thread)
# ---------------------------------------------------------------------------


def _run_inference(
    img: np.ndarray,
    face_detector: cv2.FaceDetectorYN,
    genderage_net: cv2.dnn.Net,
    max_faces: int,
    emotion_net: cv2.dnn.Net | None = None,
) -> list[FaceResult]:
    """Detect faces with YuNet, align, and predict age/gender with InsightFace genderage."""
    h, w = img.shape[:2]

    # YuNet requires input size to be set before detection
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(img)

    if faces is None:
        return []

    results: list[FaceResult] = []
    for i in range(faces.shape[0]):
        if len(results) >= max_faces:
            break

        # Parse YuNet output row: [x, y, w, h, ...landmarks..., score]
        row = faces[i].astype(float)
        fx, fy, fw, fh = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        detection_confidence = float(row[14])

        # Extract 5 landmarks from YuNet
        landmarks = [
            (row[4], row[5]),     # right eye
            (row[6], row[7]),     # left eye
            (row[8], row[9]),     # nose tip
            (row[10], row[11]),   # right mouth corner
            (row[12], row[13]),   # left mouth corner
        ]

        # --- Multi-crop ensemble for robust age + gender ---
        avg_gender_probs, median_age = _multi_crop_ensemble(
            img, fx, fy, fw, fh, landmarks, genderage_net,
        )

        gender_idx = int(np.argmax(avg_gender_probs))
        # genderage.onnx convention: index 0 = Male, index 1 = Female
        gender = "Male" if gender_idx == 0 else "Female"
        gender_conf = float(avg_gender_probs[gender_idx])

        age = int(round(float(median_age) * 100))
        age = max(0, age)

        # --- Emotion detection (expression-aware gender adjustment) ---
        emotion_label = None
        emotion_conf = None
        if emotion_net is not None:
            try:
                # Use the 20% padded crop for emotion
                pad_w = int(fw * 0.2)
                pad_h = int(fh * 0.2)
                x1 = max(0, fx - pad_w)
                y1 = max(0, fy - pad_h)
                x2 = min(w, fx + fw + pad_w)
                y2 = min(h, fy + fh + pad_h)
                emotion_crop = img[y1:y2, x1:x2]
                if emotion_crop.size > 0:
                    emotion_label, emotion_conf = _detect_emotion(emotion_crop, emotion_net)

                    # Expression-aware gender confidence adjustment:
                    # Expressions like happiness/surprise can bias gender logits.
                    # When such expressions are detected with high confidence,
                    # reduce gender confidence to signal uncertainty.
                    EXPRESSIVE_EMOTIONS = {"happiness", "surprise", "contempt"}
                    if emotion_label in EXPRESSIVE_EMOTIONS and emotion_conf > 0.5:
                        # Scale down gender confidence proportionally to expression strength
                        adjustment = 1.0 - (emotion_conf * 0.15)  # max 15% reduction
                        gender_conf *= adjustment
            except Exception:
                pass  # Emotion detection is best-effort

        # Tighter age range: ±3 years (InsightFace regression is precise)
        age_min = max(0, age - 3)
        age_max = age + 3

        # Normalized bounding box (0.0 - 1.0) using original (unpadded) bbox
        x_norm = round(fx / w, 6)
        y_norm = round(fy / h, 6)
        w_norm = round(fw / w, 6)
        h_norm = round(fh / h, 6)

        results.append(FaceResult(
            age=age,
            age_min=age_min,
            age_max=age_max,
            gender=gender,
            gender_confidence=round(gender_conf, 4),
            confidence=round(detection_confidence, 4),
            region=[x_norm, y_norm, w_norm, h_norm],
            emotion=emotion_label,
            emotion_confidence=round(emotion_conf, 4) if emotion_conf is not None else None,
        ))

    return results


# ---------------------------------------------------------------------------
# Shared inference pipeline
# ---------------------------------------------------------------------------


async def _inference_pipeline(
    img_bytes: bytes,
    max_faces: int,
    request: Request,
) -> AnalyzeResponse:
    """Common pipeline: validate, cache check, infer, cache store, return."""
    request_id = getattr(request.state, "request_id", "unknown")

    # Ensure models are available
    if app.state.face_detector is None or app.state.genderage_net is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    t0 = time.perf_counter()

    # Magic-byte validation
    _validate_magic_bytes(img_bytes)

    # Cache check (SHA-256)
    digest = hashlib.sha256(img_bytes).hexdigest()
    cached = _check_cache(digest)
    if cached is not None:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Cache hit — request_id=%s cache_hit=true inference_ms=%.1f face_count=%d cache_size=%d",
            request_id, elapsed, cached["face_count"], len(cache),
        )
        return AnalyzeResponse(**cached)

    # Decode image
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Invalid or corrupt image data")

    # Resize if exceeds max dimension
    img = _resize_if_needed(img)

    # Run CPU-bound inference with concurrency limit
    sem = _get_inference_semaphore()
    async with sem:
        results = await asyncio.to_thread(
            _run_inference,
            img,
            app.state.face_detector,
            app.state.genderage_net,
            max_faces,
            app.state.emotion_net,
        )

    elapsed = (time.perf_counter() - t0) * 1000
    response = AnalyzeResponse(
        results=results,
        face_count=len(results),
        processing_time_ms=round(elapsed, 1),
    )

    # Update LRU cache
    _store_cache(digest, response.model_dump())

    # Metrics log
    logger.info(
        "Inference complete — request_id=%s cache_hit=false inference_ms=%.1f face_count=%d cache_size=%d",
        request_id, elapsed, len(results), len(cache),
    )

    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get(
    "/api/health",
    tags=["health"],
    summary="Health check",
)
async def health():
    models_loaded = all(
        x is not None
        for x in (app.state.face_detector, app.state.genderage_net, app.state.emotion_net)
    )
    return {
        "status": "ok",
        "models_loaded": models_loaded,
        "models": {
            "face_detector": "YuNet (face_detection_yunet_2023mar.onnx)",
            "age_gender": "InsightFace genderage.onnx (regression age + softmax gender)",
            "emotion": "FER+ emotion-ferplus-8.onnx (expression-aware gender correction)",
        },
    }


@app.post(
    "/api/analyze",
    response_model=AnalyzeResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["inference"],
    summary="Analyze a base64-encoded image for age and gender",
)
@limiter.limit("300/minute")
async def analyze(
    request: Request,
    req: AnalyzeRequest,
    max_faces: int = Query(default=20, ge=1, le=100, description="Maximum number of faces to process"),
):
    # Strip data-URL prefix
    raw = req.image
    if not raw or not raw.strip():
        raise HTTPException(status_code=422, detail="Image payload is empty")

    if "," in raw:
        raw = raw.split(",", 1)[1]

    # Decode base64
    try:
        img_bytes = base64.b64decode(raw)
    except binascii.Error:
        raise HTTPException(status_code=422, detail="Invalid base64 encoding")

    if not img_bytes:
        raise HTTPException(status_code=422, detail="Decoded image data is empty")

    return await _inference_pipeline(img_bytes, max_faces, request)


@app.post(
    "/api/upload",
    response_model=AnalyzeResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["inference"],
    summary="Analyze an uploaded image file for age and gender",
)
@limiter.limit("30/minute")
async def upload(
    request: Request,
    image: UploadFile = File(..., description="Image file (JPEG or PNG)"),
    max_faces: int = Query(default=20, ge=1, le=100, description="Maximum number of faces to process"),
):
    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    return await _inference_pipeline(img_bytes, max_faces, request)

"""Lite-Vision — Serverless Age & Gender Detection API (Vercel)"""

import base64
import hashlib
import logging
import os
import time
from collections import OrderedDict
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger("lite-vision")
logging.basicConfig(level=logging.INFO)

# Resolve from serverless/api/ up to serverless/models/
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

MODELS = {
    "face": {
        "prototxt": "deploy.prototxt",
        "caffemodel": "res10_300x300_ssd_iter_140000.caffemodel",
    },
    "age": {
        "prototxt": "age_deploy.prototxt",
        "caffemodel": "age_net.caffemodel",
    },
    "gender": {
        "prototxt": "gender_deploy.prototxt",
        "caffemodel": "gender_net.caffemodel",
    },
}

AGE_MIDPOINTS = [1, 5, 10, 17, 28, 40, 50, 80]
GENDER_LIST = ["Male", "Female"]
MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
CONFIDENCE_THRESHOLD = 0.7

face_net = None
age_net = None
gender_net = None

cache: OrderedDict[str, dict] = OrderedDict()
MAX_CACHE = 50


def _model_path(filename: str) -> str:
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}. "
            "Run the download_models.py script before deploying."
        )
    return path


def load_models() -> None:
    """Load all three Caffe models into global state."""
    global face_net, age_net, gender_net

    logger.info("Loading models from %s ...", MODEL_DIR)

    face_net = cv2.dnn.readNetFromCaffe(
        _model_path(MODELS["face"]["prototxt"]),
        _model_path(MODELS["face"]["caffemodel"]),
    )
    age_net = cv2.dnn.readNetFromCaffe(
        _model_path(MODELS["age"]["prototxt"]),
        _model_path(MODELS["age"]["caffemodel"]),
    )
    gender_net = cv2.dnn.readNetFromCaffe(
        _model_path(MODELS["gender"]["prototxt"]),
        _model_path(MODELS["gender"]["caffemodel"]),
    )

    logger.info("All models loaded successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on cold start."""
    try:
        load_models()
    except FileNotFoundError as exc:
        logger.error("Model loading failed: %s", exc)
        raise
    yield


app = FastAPI(title="Lite-Vision", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    image: str


class FaceResult(BaseModel):
    age: int
    gender: str
    confidence: float
    region: list[float]


class AnalyzeResponse(BaseModel):
    results: list[FaceResult]
    face_count: int
    processing_time_ms: float


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    models_loaded = all(net is not None for net in (face_net, age_net, gender_net))
    return {"status": "ok", "models_loaded": models_loaded}


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if face_net is None or age_net is None or gender_net is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    t0 = time.perf_counter()

    # Strip optional data-URI prefix
    raw = req.image
    if "," in raw:
        raw = raw.split(",", 1)[1]

    # Check LRU cache
    md5 = hashlib.md5(raw.encode()).hexdigest()
    if md5 in cache:
        cache.move_to_end(md5)
        return cache[md5]

    # Decode image
    img_bytes = base64.b64decode(raw)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return AnalyzeResponse(results=[], face_count=0, processing_time_ms=0)

    h, w = img.shape[:2]

    # Detect faces
    blob = cv2.dnn.blobFromImage(
        img, 1.0, (300, 300), MEAN_VALUES, swapRB=False, crop=False,
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    results: list[FaceResult] = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        x1 = max(0, int(detections[0, 0, i, 3] * w))
        y1 = max(0, int(detections[0, 0, i, 4] * h))
        x2 = min(w, int(detections[0, 0, i, 5] * w))
        y2 = min(h, int(detections[0, 0, i, 6] * h))

        face_roi = img[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        face_blob = cv2.dnn.blobFromImage(
            face_roi, 1.0, (227, 227), MEAN_VALUES, swapRB=False,
        )

        # Gender prediction
        gender_net.setInput(face_blob)
        gender_preds = gender_net.forward()
        gender_idx = int(gender_preds[0].argmax())
        gender = GENDER_LIST[gender_idx]

        # Age prediction
        age_net.setInput(face_blob)
        age_preds = age_net.forward()
        age_idx = int(age_preds[0].argmax())
        age = AGE_MIDPOINTS[age_idx]

        # Normalized bounding box (0.0 – 1.0)
        x_norm = round(x1 / w, 6)
        y_norm = round(y1 / h, 6)
        w_norm = round((x2 - x1) / w, 6)
        h_norm = round((y2 - y1) / h, 6)

        results.append(
            FaceResult(
                age=age,
                gender=gender,
                confidence=round(confidence, 4),
                region=[x_norm, y_norm, w_norm, h_norm],
            )
        )

    elapsed = (time.perf_counter() - t0) * 1000
    response = AnalyzeResponse(
        results=results,
        face_count=len(results),
        processing_time_ms=round(elapsed, 1),
    )

    # Update LRU cache
    cache[md5] = response.model_dump()
    if len(cache) > MAX_CACHE:
        cache.popitem(last=False)

    return response

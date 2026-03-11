"""Lite-Vision — Real-Time Age & Gender Detection API"""

import base64
import hashlib
import logging
import os
import time
import urllib.request
from collections import OrderedDict
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger("lite-vision")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

MODELS = {
    "deploy.prototxt": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    },
    "res10_300x300_ssd_iter_140000.caffemodel": {
        "url": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    },
    "age_deploy.prototxt": {
        "url": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
    },
    "age_net.caffemodel": {
        "url": "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_net.caffemodel",
    },
    "gender_deploy.prototxt": {
        "url": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
    },
    "gender_net.caffemodel": {
        "url": "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/gender_net.caffemodel",
    },
}

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
AGE_MIDPOINTS = [1, 5, 10, 17, 28, 40, 50, 80]
GENDER_LIST = ["Male", "Female"]
MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
CONFIDENCE_THRESHOLD = 0.7

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
face_net = None
age_net = None
gender_net = None
cache: OrderedDict[str, dict] = OrderedDict()
MAX_CACHE = 100

# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for filename, info in MODELS.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        logger.info(f"Downloading {filename} ...")
        urllib.request.urlretrieve(info["url"], str(path))
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"  -> saved {filename} ({size_mb:.1f} MB)")


def load_models():
    global face_net, age_net, gender_net
    download_models()
    face_net = cv2.dnn.readNetFromCaffe(
        os.path.join(MODEL_DIR, "deploy.prototxt"),
        os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel"),
    )
    age_net = cv2.dnn.readNetFromCaffe(
        os.path.join(MODEL_DIR, "age_deploy.prototxt"),
        os.path.join(MODEL_DIR, "age_net.caffemodel"),
    )
    gender_net = cv2.dnn.readNetFromCaffe(
        os.path.join(MODEL_DIR, "gender_deploy.prototxt"),
        os.path.join(MODEL_DIR, "gender_net.caffemodel"),
    )
    logger.info("All models loaded.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Lite-Vision...")
    load_models()
    yield

app = FastAPI(title="Lite-Vision", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    image: str

class FaceResult(BaseModel):
    age: int
    gender: str
    confidence: float
    region: list[int]

class AnalyzeResponse(BaseModel):
    results: list[FaceResult]
    face_count: int
    processing_time_ms: float

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    t0 = time.perf_counter()

    # Strip data-URL prefix
    raw = req.image
    if "," in raw:
        raw = raw.split(",", 1)[1]

    # Cache check
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
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), MEAN_VALUES, swapRB=False, crop=False)
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

        face_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), MEAN_VALUES, swapRB=False)

        # Gender
        gender_net.setInput(face_blob)
        gender_preds = gender_net.forward()
        gender_idx = int(gender_preds[0].argmax())
        gender = GENDER_LIST[gender_idx]

        # Age
        age_net.setInput(face_blob)
        age_preds = age_net.forward()
        age_idx = int(age_preds[0].argmax())
        age = AGE_MIDPOINTS[age_idx]

        results.append(FaceResult(
            age=age,
            gender=gender,
            confidence=round(confidence, 4),
            region=[x1, y1, x2 - x1, y2 - y1],
        ))

    elapsed = (time.perf_counter() - t0) * 1000
    response = AnalyzeResponse(results=results, face_count=len(results), processing_time_ms=round(elapsed, 1))

    # Update cache
    cache[md5] = response.model_dump()
    if len(cache) > MAX_CACHE:
        cache.popitem(last=False)

    return response

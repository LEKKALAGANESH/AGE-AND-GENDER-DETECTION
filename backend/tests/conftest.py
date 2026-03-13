"""Shared fixtures for the Lite-Vision test suite.

All tests run without real model files.  The ML networks on app.state
are replaced with lightweight mocks that return deterministic results,
so no model files are required.

The YuNet face detector is mocked to return a single detection.
The InsightFace genderage_net is mocked to return deterministic age/gender.
"""

import base64
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


# ── helper: build a tiny but valid JPEG ─────────────────────────────────────

def _make_test_image_b64(width: int = 10, height: int = 10) -> str:
    """Return a base64-encoded JPEG of a solid-colour image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (100, 150, 200)  # arbitrary BGR colour
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode()


# ── mock builders ────────────────────────────────────────────────────────────

def _build_yunet_mock():
    """Mock YuNet face detector that returns a single detection with landmarks."""
    detector = MagicMock()

    def _detect(img):
        h, w = img.shape[:2] if hasattr(img, 'shape') else (10, 10)
        # YuNet output row: [x, y, w, h, lm0_x, lm0_y, ..., lm4_x, lm4_y, score]
        # 15 values total: 4 bbox + 10 landmarks + 1 score
        face = np.zeros((1, 15), dtype=np.float32)
        face[0, 0] = w * 0.1    # x
        face[0, 1] = h * 0.1    # y
        face[0, 2] = w * 0.8    # width
        face[0, 3] = h * 0.8    # height
        # 5 landmarks (x,y pairs)
        face[0, 4] = w * 0.35   # right eye x
        face[0, 5] = h * 0.35   # right eye y
        face[0, 6] = w * 0.65   # left eye x
        face[0, 7] = h * 0.35   # left eye y
        face[0, 8] = w * 0.5    # nose x
        face[0, 9] = h * 0.5    # nose y
        face[0, 10] = w * 0.35  # right mouth x
        face[0, 11] = h * 0.65  # right mouth y
        face[0, 12] = w * 0.65  # left mouth x
        face[0, 13] = h * 0.65  # left mouth y
        face[0, 14] = 0.95      # confidence score
        return 1, face

    detector.detect.side_effect = _detect
    detector.setInputSize = MagicMock()
    return detector


def _build_genderage_mock():
    """Mock InsightFace genderage_net that returns deterministic age/gender.

    Output shape: [1, 3] — [0:2] = gender logits (0=Male, 1=Female), [2] = age factor.
    age_factor * 100 = predicted age.
    """
    net = MagicMock()

    def _forward():
        # gender logits: Male=2.0, Female=-1.0 → softmax ~= [0.95, 0.05]
        # age factor: 0.28 → age = 28
        return np.array([[[2.0, -1.0, 0.28]]], dtype=np.float32)[0]

    net.forward.side_effect = _forward
    net.setInput = MagicMock()
    return net


def _build_emotion_mock():
    """Mock FER+ emotion_net that returns 'neutral' with high confidence."""
    net = MagicMock()

    def _forward():
        # 8-class output: [neutral, happiness, surprise, sadness, anger, disgust, fear, contempt]
        # Return high neutral score
        output = np.zeros((1, 8), dtype=np.float32)
        output[0, 0] = 5.0  # strong neutral logit
        return output

    net.forward.side_effect = _forward
    net.setInput = MagicMock()
    return net


def _build_happy_emotion_mock():
    """Mock FER+ emotion_net that returns 'happiness' with high confidence."""
    net = MagicMock()

    def _forward():
        output = np.zeros((1, 8), dtype=np.float32)
        output[0, 1] = 5.0  # strong happiness logit
        return output

    net.forward.side_effect = _forward
    net.setInput = MagicMock()
    return net


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def _patch_models(monkeypatch):
    """Prevent the lifespan from downloading / loading real models and
    instead inject mocks onto app.state after the TestClient starts."""
    import main as mod

    # Stub out load_models so the lifespan does nothing
    monkeypatch.setattr(mod, "load_models", lambda app: None)
    # Clear the module-level cache between tests
    mod.cache.clear()


@pytest.fixture()
def client(_patch_models):
    """A FastAPI TestClient with mocked YuNet + genderage_net + emotion_net."""
    from main import app

    with TestClient(app) as c:
        # Inject mock detectors
        app.state.face_detector = _build_yunet_mock()
        app.state.genderage_net = _build_genderage_mock()
        app.state.emotion_net = _build_emotion_mock()
        yield c


@pytest.fixture()
def client_no_models(_patch_models):
    """A TestClient where models remain None (simulating load failure)."""
    from main import app

    with TestClient(app) as c:
        # lifespan already set them to None; leave them that way
        yield c


@pytest.fixture()
def client_happy_expression(_patch_models):
    """A TestClient where emotion model returns 'happiness' (tests expression-aware adjustment)."""
    from main import app

    with TestClient(app) as c:
        app.state.face_detector = _build_yunet_mock()
        app.state.genderage_net = _build_genderage_mock()
        app.state.emotion_net = _build_happy_emotion_mock()
        yield c


@pytest.fixture()
def valid_image_b64() -> str:
    """Base64-encoded 10x10 JPEG — a small but decodable image."""
    return _make_test_image_b64()


@pytest.fixture()
def invalid_image_b64() -> str:
    """A string that *is* valid base64 but does NOT decode to a valid image."""
    return base64.b64encode(b"this is definitely not a jpeg").decode()


@pytest.fixture()
def corrupt_base64() -> str:
    """A string that is not valid base64 at all."""
    return "%%%NOT-BASE64###"


# ── multi-face mock builders ─────────────────────────────────────────────────

def _build_multi_face_yunet_mock():
    """Mock YuNet that returns 3 face detections at different positions."""
    detector = MagicMock()

    def _detect(img):
        h, w = img.shape[:2] if hasattr(img, 'shape') else (100, 100)
        faces = np.zeros((3, 15), dtype=np.float32)
        # Face 1: left side
        faces[0, 0] = w * 0.05; faces[0, 1] = h * 0.1
        faces[0, 2] = w * 0.25; faces[0, 3] = h * 0.8
        faces[0, 4] = w * 0.12; faces[0, 5] = h * 0.35
        faces[0, 6] = w * 0.22; faces[0, 7] = h * 0.35
        faces[0, 8] = w * 0.17; faces[0, 9] = h * 0.5
        faces[0, 10] = w * 0.12; faces[0, 11] = h * 0.65
        faces[0, 12] = w * 0.22; faces[0, 13] = h * 0.65
        faces[0, 14] = 0.95
        # Face 2: center
        faces[1, 0] = w * 0.35; faces[1, 1] = h * 0.1
        faces[1, 2] = w * 0.25; faces[1, 3] = h * 0.8
        faces[1, 4] = w * 0.42; faces[1, 5] = h * 0.35
        faces[1, 6] = w * 0.52; faces[1, 7] = h * 0.35
        faces[1, 8] = w * 0.47; faces[1, 9] = h * 0.5
        faces[1, 10] = w * 0.42; faces[1, 11] = h * 0.65
        faces[1, 12] = w * 0.52; faces[1, 13] = h * 0.65
        faces[1, 14] = 0.92
        # Face 3: right side
        faces[2, 0] = w * 0.65; faces[2, 1] = h * 0.1
        faces[2, 2] = w * 0.25; faces[2, 3] = h * 0.8
        faces[2, 4] = w * 0.72; faces[2, 5] = h * 0.35
        faces[2, 6] = w * 0.82; faces[2, 7] = h * 0.35
        faces[2, 8] = w * 0.77; faces[2, 9] = h * 0.5
        faces[2, 10] = w * 0.72; faces[2, 11] = h * 0.65
        faces[2, 12] = w * 0.82; faces[2, 13] = h * 0.65
        faces[2, 14] = 0.88
        return 3, faces

    detector.detect.side_effect = _detect
    detector.setInputSize = MagicMock()
    return detector


def _build_varied_genderage_mock():
    """Mock genderage_net that returns DIFFERENT results based on input blob.

    Uses a call counter to produce varied outputs. Each face calls forward()
    3 times (multi-crop ensemble: aligned + 2 padded crops), so the counter
    cycles every 3 calls to keep predictions consistent within a single face
    but different between faces.
    """
    net = MagicMock()
    _call_count = [0]

    def _forward():
        # Each face calls forward() 3 times (multi-crop ensemble).
        # Cycle through predictions every 3 calls so each face is consistent.
        face_idx = _call_count[0] // 3
        _call_count[0] += 1
        idx = face_idx % 3
        if idx == 0:
            # Young male: Male=2.0, Female=-1.0, age=0.21
            return np.array([[[2.0, -1.0, 0.21]]], dtype=np.float32)[0]
        elif idx == 1:
            # Middle-aged female: Male=-1.5, Female=2.5, age=0.40
            return np.array([[[-1.5, 2.5, 0.40]]], dtype=np.float32)[0]
        else:
            # Older male: Male=1.0, Female=-0.5, age=0.56
            return np.array([[[1.0, -0.5, 0.56]]], dtype=np.float32)[0]

    net.forward.side_effect = _forward
    net.setInput = MagicMock()
    return net


# ── multi-face fixtures ──────────────────────────────────────────────────────

@pytest.fixture()
def client_multi_face(_patch_models):
    """A TestClient with 3-face YuNet mock and varied genderage mock."""
    from main import app

    with TestClient(app) as c:
        app.state.face_detector = _build_multi_face_yunet_mock()
        app.state.genderage_net = _build_varied_genderage_mock()
        app.state.emotion_net = _build_emotion_mock()
        yield c


@pytest.fixture()
def multi_face_image_b64() -> str:
    """Base64-encoded 300x100 JPEG — wide image to hold 3 faces."""
    return _make_test_image_b64(width=300, height=100)

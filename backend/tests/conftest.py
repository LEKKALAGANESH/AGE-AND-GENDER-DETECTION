"""Shared fixtures for the Lite-Vision test suite.

All tests run without real model files.  The ML networks on app.state
are replaced with lightweight mocks that return deterministic results,
so no model files are required.

The SCRFD face detector is mocked to return detections in SCRFD format.
The InsightFace genderage_net is mocked to return deterministic age/gender.
The FairFace fairface_net is mocked to return deterministic gender/age.
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

def _build_scrfd_mock():
    """Mock SCRFD face detector that returns a single detection with landmarks.

    Returns (det[N,5], kpss[N,5,2]) in SCRFD format:
      det: [x1, y1, x2, y2, score]
      kpss: 5 landmarks [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    detector = MagicMock()

    def _detect(img, conf_threshold=None, max_num=0):
        h, w = img.shape[:2] if hasattr(img, 'shape') else (10, 10)
        # Single face detection
        det = np.zeros((1, 5), dtype=np.float32)
        det[0, 0] = w * 0.1    # x1
        det[0, 1] = h * 0.1    # y1
        det[0, 2] = w * 0.9    # x2
        det[0, 3] = h * 0.9    # y2
        det[0, 4] = 0.95       # confidence score

        # 5 landmarks
        kpss = np.zeros((1, 5, 2), dtype=np.float32)
        kpss[0, 0] = [w * 0.35, h * 0.35]  # left eye
        kpss[0, 1] = [w * 0.65, h * 0.35]  # right eye
        kpss[0, 2] = [w * 0.5, h * 0.5]    # nose
        kpss[0, 3] = [w * 0.35, h * 0.65]  # left mouth
        kpss[0, 4] = [w * 0.65, h * 0.65]  # right mouth

        if max_num > 0:
            det = det[:max_num]
            kpss = kpss[:max_num]
        return det, kpss

    detector.detect = _detect
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


def _build_fairface_mock():
    """Mock FairFace ResNet34 that returns Male with high confidence + age ~28.

    FairFace has 3 separate outputs: age_output(9), gender_output(2), race_output(7).
    """
    net = MagicMock()

    def _forward(output_names=None):
        # age_output: 9 bins — strong 20-29 bin
        age_out = np.zeros((1, 9), dtype=np.float32)
        age_out[0, 3] = 5.0  # 20-29 bin (midpoint 25)

        # gender_output: [Male, Female] — strong Male
        gender_out = np.zeros((1, 2), dtype=np.float32)
        gender_out[0, 0] = 3.0  # Male logit

        # race_output: 7 classes
        race_out = np.zeros((1, 7), dtype=np.float32)

        return [age_out, gender_out, race_out]

    net.forward = _forward
    net.setInput = MagicMock()
    return net


def _build_fairface_female_mock():
    """Mock FairFace that returns Female with high confidence."""
    net = MagicMock()

    def _forward(output_names=None):
        age_out = np.zeros((1, 9), dtype=np.float32)
        age_out[0, 3] = 5.0
        gender_out = np.zeros((1, 2), dtype=np.float32)
        gender_out[0, 1] = 3.0  # Female logit
        race_out = np.zeros((1, 7), dtype=np.float32)
        return [age_out, gender_out, race_out]

    net.forward = _forward
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
    """A FastAPI TestClient with mocked SCRFD + genderage + emotion + FairFace."""
    from main import app

    with TestClient(app) as c:
        app.state.face_detector = _build_scrfd_mock()
        app.state.genderage_net = _build_genderage_mock()
        app.state.emotion_net = _build_emotion_mock()
        app.state.fairface_net = _build_fairface_mock()
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
    """A TestClient where emotion model returns 'happiness'."""
    from main import app

    with TestClient(app) as c:
        app.state.face_detector = _build_scrfd_mock()
        app.state.genderage_net = _build_genderage_mock()
        app.state.emotion_net = _build_happy_emotion_mock()
        app.state.fairface_net = _build_fairface_mock()
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

def _build_multi_face_scrfd_mock():
    """Mock SCRFD that returns 3 face detections at different positions."""
    detector = MagicMock()

    def _detect(img, conf_threshold=None, max_num=0):
        h, w = img.shape[:2] if hasattr(img, 'shape') else (100, 100)
        det = np.zeros((3, 5), dtype=np.float32)
        kpss = np.zeros((3, 5, 2), dtype=np.float32)

        # Face 1: left side
        det[0] = [w * 0.05, h * 0.1, w * 0.30, h * 0.9, 0.95]
        kpss[0, 0] = [w * 0.12, h * 0.35]
        kpss[0, 1] = [w * 0.22, h * 0.35]
        kpss[0, 2] = [w * 0.17, h * 0.5]
        kpss[0, 3] = [w * 0.12, h * 0.65]
        kpss[0, 4] = [w * 0.22, h * 0.65]

        # Face 2: center
        det[1] = [w * 0.35, h * 0.1, w * 0.60, h * 0.9, 0.92]
        kpss[1, 0] = [w * 0.42, h * 0.35]
        kpss[1, 1] = [w * 0.52, h * 0.35]
        kpss[1, 2] = [w * 0.47, h * 0.5]
        kpss[1, 3] = [w * 0.42, h * 0.65]
        kpss[1, 4] = [w * 0.52, h * 0.65]

        # Face 3: right side
        det[2] = [w * 0.65, h * 0.1, w * 0.90, h * 0.9, 0.88]
        kpss[2, 0] = [w * 0.72, h * 0.35]
        kpss[2, 1] = [w * 0.82, h * 0.35]
        kpss[2, 2] = [w * 0.77, h * 0.5]
        kpss[2, 3] = [w * 0.72, h * 0.65]
        kpss[2, 4] = [w * 0.82, h * 0.65]

        if max_num > 0:
            det = det[:max_num]
            kpss = kpss[:max_num]
        return det, kpss

    detector.detect = _detect
    return detector


def _build_varied_genderage_mock():
    """Mock genderage_net that returns DIFFERENT results based on call count.

    Each face calls forward() 3 times (multi-crop ensemble), so the counter
    cycles every 3 calls to keep predictions consistent within a single face.
    """
    net = MagicMock()
    _call_count = [0]

    def _forward():
        face_idx = _call_count[0] // 3
        _call_count[0] += 1
        idx = face_idx % 3
        if idx == 0:
            return np.array([[[2.0, -1.0, 0.21]]], dtype=np.float32)[0]
        elif idx == 1:
            return np.array([[[-1.5, 2.5, 0.40]]], dtype=np.float32)[0]
        else:
            return np.array([[[1.0, -0.5, 0.56]]], dtype=np.float32)[0]

    net.forward.side_effect = _forward
    net.setInput = MagicMock()
    return net


def _build_varied_fairface_mock():
    """Mock FairFace that returns different gender/age per call, matching the varied genderage mock."""
    net = MagicMock()
    _call_count = [0]

    def _forward(output_names=None):
        # Each face triggers 2 FairFace calls (full face + upper half)
        face_idx = _call_count[0] // 2
        _call_count[0] += 1
        idx = face_idx % 3

        age_out = np.zeros((1, 9), dtype=np.float32)
        gender_out = np.zeros((1, 2), dtype=np.float32)
        race_out = np.zeros((1, 7), dtype=np.float32)

        if idx == 0:
            age_out[0, 3] = 5.0  # 20-29
            gender_out[0, 0] = 3.0  # Male
        elif idx == 1:
            age_out[0, 4] = 5.0  # 30-39
            gender_out[0, 1] = 3.0  # Female
        else:
            age_out[0, 6] = 5.0  # 50-59
            gender_out[0, 0] = 3.0  # Male
        return [age_out, gender_out, race_out]

    net.forward = _forward
    net.setInput = MagicMock()
    return net


# ── multi-face fixtures ──────────────────────────────────────────────────────

@pytest.fixture()
def client_multi_face(_patch_models):
    """A TestClient with 3-face SCRFD mock and varied genderage/fairface mocks."""
    from main import app

    with TestClient(app) as c:
        app.state.face_detector = _build_multi_face_scrfd_mock()
        app.state.genderage_net = _build_varied_genderage_mock()
        app.state.emotion_net = _build_emotion_mock()
        app.state.fairface_net = _build_varied_fairface_mock()
        yield c


@pytest.fixture()
def multi_face_image_b64() -> str:
    """Base64-encoded 300x100 JPEG — wide image to hold 3 faces."""
    return _make_test_image_b64(width=300, height=100)

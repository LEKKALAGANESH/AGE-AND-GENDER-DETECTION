# Lite-Vision v4.0.0 -- Executive Summary

**Real-time age, gender, and emotion detection powered by lightweight ONNX inference**

---

## Overview

Lite-Vision is a full-stack computer-vision application that performs real-time age,
gender, and emotion detection from webcam streams and uploaded images. The system is
built on a FastAPI backend and a Next.js 15 frontend, with all machine-learning
inference executed through ONNX models via OpenCV's DNN module -- eliminating the need
for PyTorch or TensorFlow at runtime. This architecture yields a small deployment
footprint, fast cold-start times, and predictable resource consumption suitable for
cloud and containerized environments.

---

## Key Features

### Multi-Model Inference Pipeline

| Model                     | Role                                                        |
| ------------------------- | ----------------------------------------------------------- |
| **SCRFD 10G KPS**         | High-accuracy face detection with keypoint localization     |
| **InsightFace GenderAge** | Primary age and gender estimation                           |
| **FER+**                  | Seven-class facial emotion recognition                      |
| **FairFace ResNet34**     | Racially-balanced gender classification for bias correction |

### Robustness Techniques

- **Multi-crop ensemble** -- Multiple aligned crops per face are scored independently
  and aggregated, producing stable gender predictions across varying expressions and
  head poses.
- **Mask-aware age fusion** -- When a face mask is detected, the system shifts to
  upper-face analysis (forehead, eye region) to estimate age through occluded areas.
- **CLAHE preprocessing** -- Contrast-Limited Adaptive Histogram Equalization is
  applied to input frames, improving detection accuracy under dark or low-contrast
  lighting conditions.
- **FairFace gender fusion** -- Predictions from the FairFace model are fused with the
  InsightFace result to reduce racial and expression-related bias in gender
  classification.
- **Temporal gender smoothing** -- The frontend maintains a rolling window of recent
  predictions per tracked face, suppressing single-frame classification flicker in
  live webcam mode.

### Frontend Capabilities

- Real-time webcam capture with per-frame analysis
- Canvas overlay rendering bounding boxes, labels, and confidence scores
- Image upload mode for single-frame analysis
- Responsive layout built with Next.js 15 and Tailwind CSS

---

## Technology Stack

| Layer          | Technology                                              |
| -------------- | ------------------------------------------------------- |
| **Backend**    | Python 3.14, FastAPI, OpenCV (`cv2.dnn`), ONNX Runtime  |
| **Frontend**   | Next.js 15, React, TypeScript, Tailwind CSS             |
| **ML Runtime** | ONNX models only -- no PyTorch or TensorFlow dependency |
| **Deployment** | Vercel (frontend), Railway / Docker (backend)           |
| **Testing**    | pytest (backend), Vitest (frontend)                     |

### API Endpoints

| Method | Path           | Description                                                   |
| ------ | -------------- | ------------------------------------------------------------- |
| `POST` | `/api/analyze` | Accepts a base64-encoded image and returns detection results  |
| `POST` | `/api/upload`  | Accepts a multipart file upload and returns detection results |
| `GET`  | `/api/health`  | Returns service health status and model readiness             |

---

## Performance Metrics

| Metric                          | Result      |
| ------------------------------- | ----------- |
| Integration test images passing | **7 / 7**   |
| Unit tests passing              | **73 / 73** |

All tests validate end-to-end correctness across diverse demographics, lighting
conditions, occlusion scenarios, and expression variations.

---

## Quick Start

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend development server starts on `http://localhost:3000` and proxies API
requests to the backend at `http://localhost:8000`.

### Docker (Backend)

```bash
cd backend
docker build -t lite-vision-backend .
docker run -p 8000:8000 lite-vision-backend
```

---

## License

This project is released under the **MIT License**.

Copyright (c) 2026 Lekkala Ganesh. See [LICENSE](../LICENSE) for full terms.

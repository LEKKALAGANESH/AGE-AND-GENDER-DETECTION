# Lite-Vision — Real-Time Age & Gender Detection

A high-performance, zero-database age and gender detection system. FastAPI backend with multi-model ONNX inference, Next.js 15 frontend with live webcam streaming.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.10+, FastAPI, OpenCV DNN, Uvicorn |
| **Frontend** | Next.js 15, React 19, TypeScript, Tailwind CSS |
| **Models** | SCRFD, InsightFace, FER+, FairFace (ONNX) |
| **Inference** | OpenCV DNN module — no TensorFlow/PyTorch required |

## Project Structure

```
age_gender_detection/
├── backend/
│   ├── main.py              # FastAPI server (single file)
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile           # Container image
│   ├── .env.example         # Configuration template
│   ├── tests/               # Pytest test suite
│   └── models/              # Auto-downloaded on first run (~102 MB)
│       ├── scrfd_10g_kps.onnx
│       ├── genderage.onnx
│       ├── emotion-ferplus-8.onnx
│       └── fairface.onnx
├── frontend/
│   ├── package.json
│   ├── next.config.ts
│   ├── src/
│   │   ├── app/             # Next.js pages and layouts
│   │   ├── components/      # Camera, Controls, DropZone, ResultsPanel
│   │   ├── hooks/           # useCamera, useAnalyze, useFileUpload, useTemporalSmoothing
│   │   └── lib/             # API client, canvas drawing, types, constants
│   └── vitest.config.ts
├── reports/                 # Project documentation
│   ├── PROJECT_OVERVIEW.md
│   ├── ARCHITECTURE.md
│   ├── MODELS.md
│   ├── API_REFERENCE.md
│   ├── INFERENCE_PIPELINE.md
│   ├── FRONTEND_ARCHITECTURE.md
│   ├── TESTING_STRATEGY.md
│   ├── DEPLOYMENT_GUIDE.md
│   ├── PERFORMANCE_REPORT.md
│   └── CHANGELOG.md
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+

### 1. Start the Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Models (~102 MB) auto-download on first startup. The API runs at `http://localhost:8000`.

### 2. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

The dashboard opens at `http://localhost:3000`.

---

## API

### `POST /api/analyze`

Accepts a base64-encoded image and returns detected faces with age, gender, and emotion predictions.

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Response:**
```json
{
  "results": [
    {
      "age": 28,
      "age_min": 25,
      "age_max": 31,
      "gender": "Male",
      "gender_confidence": 0.95,
      "confidence": 0.87,
      "region": [0.12, 0.05, 0.18, 0.20],
      "emotion": "neutral",
      "emotion_confidence": 0.72
    }
  ],
  "face_count": 1,
  "processing_time_ms": 42.5
}
```

### `POST /api/upload`

Upload and analyze an image file (multipart form data).

### `GET /api/health`

Returns model status and health information.

```json
{
  "status": "ok",
  "models_loaded": true,
  "models": {
    "face_detector": "SCRFD 10G KPS (scrfd_10g_kps.onnx)",
    "age_gender": "InsightFace genderage.onnx",
    "emotion": "FER+ emotion-ferplus-8.onnx",
    "gender_fairface": "FairFace ResNet34"
  }
}
```

### API Docs

Interactive Swagger UI available at `http://localhost:8000/docs`.

## Features

- **Live Webcam Stream** — real-time detection with bounding box overlays
- **Single-Shot Capture** — manual frame capture and analysis
- **Image Upload** — drag-and-drop or file picker with instant preview
- **Emotion Detection** — 8-class expression classification (neutral, happiness, surprise, etc.)
- **Multi-Model Ensemble** — gender prediction fused across InsightFace + FairFace for bias correction
- **Frame Caching** — SHA-256 cache skips re-analysis of identical frames
- **Temporal Smoothing** — reduces jitter across consecutive frames during live streaming
- **Canvas Overlay** — gender-colored bounding boxes with age/emotion labels
- **Zero Database** — stateless API, no external services needed

## Model Details

| Model | File | Purpose | Details |
|-------|------|---------|---------|
| **SCRFD 10G KPS** | `scrfd_10g_kps.onnx` | Face detection | 82.8% WIDERFace Hard AP, outputs 5 facial landmarks |
| **InsightFace** | `genderage.onnx` | Age + gender | Continuous age regression (0-100), binary gender classification |
| **FER+** | `emotion-ferplus-8.onnx` | Emotion classification | 8 classes: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt |
| **FairFace** | `fairface.onnx` | Gender bias correction | Racially-balanced gender classifier (95.7% accuracy), fused with InsightFace |

All models run on CPU via OpenCV's DNN module (ONNX format). No GPU required.

### Inference Pipeline

1. **Preprocessing** — CLAHE contrast enhancement for low-light handling
2. **Face Detection** — SCRFD locates faces and 5 facial landmarks
3. **Alignment** — ArcFace-aligned 96x96 face crops using landmark keypoints
4. **Age Prediction** — InsightFace regression on aligned face, fused with FairFace age bins
5. **Gender Prediction** — Multi-crop ensemble (aligned + padded variants) fused with FairFace
6. **Emotion** — FER+ classification on 64x64 grayscale face crop
7. **Refinement** — Expression-aware gender confidence adjustment, mask-aware age fusion

## Performance

- Face detection + age/gender/emotion: ~30-80ms per frame (CPU)
- Supports multiple simultaneous face detections
- Concurrency-limited inference (default: 4 concurrent requests)
- Frame cache eliminates redundant computation
- 7/7 edge-case test images passing (masked, dark, glasses, expressions, multi-race)
- 73/73 unit tests passing

## Configuration

Environment variables (prefix `LITEVISION_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LITEVISION_CONFIDENCE_THRESHOLD` | `0.7` | Face detection confidence threshold |
| `LITEVISION_MAX_CACHE_SIZE` | `100` | Maximum cached frames |
| `LITEVISION_CORS_ORIGINS` | `["*"]` | CORS allowed origins |
| `LITEVISION_MAX_IMAGE_DIMENSION` | `4096` | Max input image dimension (px) |
| `LITEVISION_MAX_CONCURRENT_INFERENCES` | `4` | Concurrent inference limit |

## Deployment

### Docker

```bash
cd backend
docker build -t lite-vision-backend .
docker run -p 8000:8000 lite-vision-backend
```

### Vercel (Frontend) + Railway (Backend)

1. Deploy frontend to Vercel — set root directory to `frontend`, configure `NEXT_PUBLIC_API_URL`
2. Deploy backend to Railway — set root directory to `backend`, start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

See [reports/DEPLOYMENT_GUIDE.md](reports/DEPLOYMENT_GUIDE.md) for detailed instructions.

## Testing

### Backend

```bash
cd backend
pip install pytest httpx
pytest
```

### Frontend

```bash
cd frontend
npm test
```

## Documentation

Comprehensive project documentation is available in the [`reports/`](reports/) folder:

| Document | Description |
|----------|-------------|
| [Project Overview](reports/PROJECT_OVERVIEW.md) | Executive summary |
| [Architecture](reports/ARCHITECTURE.md) | System architecture and diagrams |
| [Models](reports/MODELS.md) | ML model specifications and fusion logic |
| [API Reference](reports/API_REFERENCE.md) | Complete endpoint documentation |
| [Inference Pipeline](reports/INFERENCE_PIPELINE.md) | Step-by-step pipeline with flowcharts |
| [Frontend Architecture](reports/FRONTEND_ARCHITECTURE.md) | Component and data flow documentation |
| [Testing Strategy](reports/TESTING_STRATEGY.md) | Test framework, mocks, and coverage |
| [Deployment Guide](reports/DEPLOYMENT_GUIDE.md) | Local, Docker, and cloud deployment |
| [Performance Report](reports/PERFORMANCE_REPORT.md) | Accuracy benchmarks and metrics |
| [Changelog](reports/CHANGELOG.md) | Version history |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `uvicorn` not found | Run with `python -m uvicorn main:app --reload` |
| Model download fails | Check internet connection; models download from GitHub/HuggingFace |
| Webcam not working | Allow camera permissions in your browser |
| CORS error | Backend must be running on `localhost:8000` |
| TensorFlow/PyTorch errors | Not needed — this project uses OpenCV DNN only |

## License

MIT License. See [LICENSE](LICENSE) for details.

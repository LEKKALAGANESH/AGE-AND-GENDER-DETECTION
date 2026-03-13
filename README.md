# Lite-Vision — Real-Time Age & Gender Detection

A high-performance, zero-database age and gender detection system. FastAPI backend with multi-model ONNX inference, Next.js 15 frontend with live webcam streaming.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.10+, FastAPI, OpenCV DNN, Uvicorn |
| **Serverless** | Vercel Python Functions, FastAPI, opencv-python-headless |
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
│   └── models/              # Auto-downloaded on first run (~135 MB)
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
├── serverless/
│   ├── api/
│   │   └── index.py         # FastAPI entry point (Vercel)
│   ├── models/
│   │   └── download_models.py
│   ├── requirements.txt
│   └── vercel.json
├── vercel.json               # Root Vercel deployment config
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

Models (~135 MB) auto-download on first startup. The API runs at `http://localhost:8000`.

### 2. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

The dashboard opens at `http://localhost:3000`.

---

### Serverless Mode (local)

Run the Vercel-ready serverless backend locally — useful for testing before deploying.

#### 1. Download Models

```bash
cd serverless/models
python download_models.py
```

#### 2. Start the Serverless API

```bash
cd serverless
pip install -r requirements.txt
pip install uvicorn
python -m uvicorn api.index:app --reload --port 8000
```

The API runs at `http://localhost:8000` with endpoints at `/api/analyze` and `/api/health`.

#### 3. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

> **Note:** The frontend proxies `/api/*` to the backend via `next.config.ts`. To use the serverless API (which serves at `/api/analyze`), update the `BACKEND_URL` environment variable or the rewrite config.

#### Key Differences

| | `backend/` | `serverless/` |
|---|---|---|
| Bounding box format | Pixel values `[x, y, w, h]` | Normalized 0.0–1.0 `[x, y, w, h]` |
| Model download | Auto-downloads on startup | Run `download_models.py` first |
| CORS | `localhost:3000` only | All origins (`*`) |
| Deployment | Local / Docker | Vercel-compatible |

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
      "region": [120, 45, 180, 200],
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

Returns `{"status": "ok", "models_loaded": true}`.

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
| **InsightFace** | `genderage.onnx` | Age + gender | Continuous age regression (0–100), binary gender classification |
| **FER+** | `emotion-ferplus-8.onnx` | Emotion classification | 8 classes: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt |
| **FairFace** | `fairface.onnx` | Gender bias correction | Racially-balanced gender classifier, fused with InsightFace predictions |

All models run on CPU via OpenCV's DNN module (ONNX format). No GPU required.

### Inference Pipeline

1. **Preprocessing** — CLAHE contrast enhancement for low-light handling
2. **Face Detection** — SCRFD locates faces and 5 facial landmarks
3. **Alignment** — ArcFace-aligned 96x96 face crops using landmark keypoints
4. **Age Prediction** — InsightFace regression on aligned face
5. **Gender Prediction** — Multi-crop ensemble (aligned + padded variants) fused with FairFace
6. **Emotion** — FER+ classification on 64x64 grayscale face crop
7. **Refinement** — Expression-aware gender confidence adjustment, mask detection

## Performance

- Face detection + age/gender/emotion: ~30–80ms per frame (CPU)
- Supports multiple simultaneous face detections
- Concurrency-limited inference (default: 4 concurrent requests)
- Frame cache eliminates redundant computation

## Configuration

Environment variables (prefix `LITEVISION_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LITEVISION_CONFIDENCE_THRESHOLD` | `0.7` | Face detection confidence threshold |
| `LITEVISION_MAX_CACHE_SIZE` | `100` | Maximum cached frames |
| `LITEVISION_CORS_ORIGINS` | `["*"]` | CORS allowed origins |
| `LITEVISION_MAX_IMAGE_DIMENSION` | `4096` | Max input image dimension (px) |
| `LITEVISION_MAX_CONCURRENT_INFERENCES` | `4` | Concurrent inference limit |

## Deploying to Vercel

```bash
# 1. Download models into serverless/models/
cd serverless/models && python download_models.py && cd ../..

# 2. Deploy
vercel
```

The `vercel.json` routes `/api/*` to the Python function and everything else to the Next.js frontend. The Python function runs with 1024 MB memory and a 10-second timeout.

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

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `uvicorn` not found | Run with `python -m uvicorn main:app --reload` |
| Model download fails | Check internet connection; models download from GitHub/HuggingFace |
| Webcam not working | Allow camera permissions in your browser |
| CORS error | Backend must be running on `localhost:8000` |
| TensorFlow/PyTorch errors | Not needed — this project uses OpenCV DNN only |
| Serverless models missing | Run `cd serverless/models && python download_models.py` |
| Serverless cold start slow | Models are ~135 MB; first request may take a few seconds |

## License

MIT License. See [LICENSE](LICENSE) for details.

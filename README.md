# Lite-Vision — Real-Time Age & Gender Detection

A high-performance, zero-database age and gender detection system. FastAPI backend with OpenCV DNN inference, Next.js 15 frontend with live webcam streaming at 10fps.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python, FastAPI, OpenCV DNN, Uvicorn |
| **Frontend** | Next.js 15, React 19, TypeScript, Tailwind CSS |
| **Models** | Pre-trained Caffe models (face detection, age, gender) |
| **Inference** | OpenCV DNN module — no TensorFlow required |

## Project Structure

```
age_gender_detection/
├── backend/
│   ├── main.py             # FastAPI server (single file)
│   ├── requirements.txt    # Python dependencies
│   └── models/             # Auto-downloaded on first run
│       ├── deploy.prototxt
│       ├── res10_300x300_ssd_iter_140000.caffemodel
│       ├── age_deploy.prototxt
│       ├── age_net.caffemodel
│       ├── gender_deploy.prototxt
│       └── gender_net.caffemodel
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx
│   │   │   ├── layout.tsx
│   │   │   └── globals.css
│   │   └── components/
│   │       └── Camera.tsx  # Webcam + upload + overlay
│   └── [config files]
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

Models (~100 MB) auto-download on first startup. The API runs at `http://localhost:8000`.

### 2. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

The dashboard opens at `http://localhost:3000`.

## API

### `POST /analyze`

Accepts a base64-encoded image and returns detected faces with age and gender predictions.

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
      "gender": "Male",
      "confidence": 0.9542,
      "region": [120, 45, 180, 200]
    }
  ],
  "face_count": 1,
  "processing_time_ms": 42.5
}
```

### `GET /health`

Returns `{"status": "ok"}`.

### API Docs

Interactive Swagger UI available at `http://localhost:8000/docs`.

## Features

- **Live Webcam Stream** — 10fps real-time detection with bounding box overlays
- **Single-Shot Capture** — Manual frame capture and analysis
- **Image Upload** — Drag-and-drop or file picker with instant preview
- **Frame Caching** — MD5-based cache skips re-analysis of identical frames
- **Canvas Overlay** — Gender-colored bounding boxes with age labels drawn on canvas
- **Zero Database** — Stateless API, no external services needed

## How It Works

1. Frontend captures a webcam frame or receives an uploaded image
2. Image is sent as base64 to `POST /analyze`
3. OpenCV DNN face detector locates faces in the image
4. Each face ROI is passed through age and gender classification networks
5. Results with bounding boxes are returned as JSON
6. Frontend draws colored overlays on a canvas layer

## Model Details

| Model | Purpose | Architecture |
|-------|---------|-------------|
| `res10_300x300_ssd` | Face detection | ResNet-10 SSD |
| `age_net` | Age classification | 8 buckets: 0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60-100 |
| `gender_net` | Gender classification | Binary: Male / Female |

All models use the Caffe framework and run on CPU via OpenCV's DNN module.

## Performance

- Face detection + age/gender prediction: ~30-80ms per frame (CPU)
- Supports multiple simultaneous face detections
- Frame cache eliminates redundant computation
- No GPU required

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `uvicorn` not found | Run with `python -m uvicorn main:app --reload` |
| Model download fails | Check internet connection; models download from GitHub |
| Webcam not working | Allow camera permissions in your browser |
| CORS error | Backend must be running on `localhost:8000` |
| TensorFlow errors | Not needed — this project uses OpenCV DNN only |

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

Built as a Computer Vision project using OpenCV Deep Neural Networks and modern web technologies.

# Lite-Vision Deployment Guide

> Complete guide for deploying the Lite-Vision age, gender, and emotion detection platform in local development, Docker, and production environments.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Local Development](#local-development)
   - [Backend Setup](#backend-setup)
   - [Frontend Setup](#frontend-setup)
4. [Environment Variables](#environment-variables)
5. [Docker Deployment](#docker-deployment)
6. [Production Deployment](#production-deployment)
   - [Frontend on Vercel](#frontend-on-vercel)
   - [Backend on Railway](#backend-on-railway)
   - [Backend on Docker (Any Platform)](#backend-on-docker-any-platform)
7. [Architecture Overview](#architecture-overview)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Ensure the following tools are installed before proceeding:

| Tool       | Minimum Version | Purpose                        |
|------------|-----------------|--------------------------------|
| Python     | 3.14+           | Backend runtime                |
| Node.js    | 18+             | Frontend runtime               |
| Git        | Latest          | Version control and deployment |

Optional (for containerized deployment):

| Tool       | Minimum Version | Purpose              |
|------------|-----------------|----------------------|
| Docker     | 20+             | Container builds     |

---

## Project Structure

```
age_gender_detection/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile           # Docker configuration
│   ├── .dockerignore        # Docker ignore rules
│   ├── .env.example         # Example environment file
│   ├── pyproject.toml       # Python project config
│   ├── models/              # Auto-downloaded ONNX models
│   └── tests/               # pytest test suite
├── frontend/
│   ├── src/                 # Next.js source code
│   ├── package.json         # Node dependencies
│   ├── next.config.ts       # Next.js config
│   ├── .env.example         # Example environment file
│   └── vitest.config.ts     # Test config
└── reports/                 # Documentation (you are here)
```

---

## Local Development

### Backend Setup

1. **Navigate to the backend directory:**

   ```bash
   cd backend
   ```

2. **Create and activate a Python virtual environment:**

   ```bash
   python -m venv .venv
   ```

   - **Linux / macOS:**
     ```bash
     source .venv/bin/activate
     ```
   - **Windows:**
     ```bash
     .venv\Scripts\activate
     ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Start the development server:**

   ```bash
   uvicorn main:app --reload --port 8000
   ```

5. **Model auto-download:** On the first startup, the application automatically downloads the required ONNX models (~102 MB total). No manual action is needed.

   | Model                     | Size     |
   |---------------------------|----------|
   | `scrfd_10g_kps.onnx`      | 16.9 MB  |
   | `genderage.onnx`          | included |
   | `emotion-ferplus-8.onnx`  | included |
   | `fairface.onnx`           | 85.2 MB  |

   Models are stored in the `models/` directory (configurable via the `LITEVISION_MODEL_DIR` variable).

6. **Verify the backend is running:** Open `http://localhost:8000/docs` in your browser to access the interactive API documentation (Swagger UI).

### Frontend Setup

1. **Navigate to the frontend directory:**

   ```bash
   cd frontend
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

3. **Start the development server:**

   ```bash
   npm run dev
   ```

4. **Access the application:** Open `http://localhost:3000` in your browser.

> **Note:** The frontend expects the backend to be running at `http://localhost:8000` by default. If your backend is running on a different host or port, set the `NEXT_PUBLIC_API_URL` environment variable accordingly (see [Environment Variables](#environment-variables)).

---

## Environment Variables

### Backend

Backend environment variables can be set in a `.env` file in the `backend/` directory or as system environment variables with the `LITEVISION_` prefix. Copy `.env.example` as a starting point:

```bash
cp .env.example .env
```

| Variable                              | Default                | Description                                      |
|---------------------------------------|------------------------|--------------------------------------------------|
| `LITEVISION_CONFIDENCE_THRESHOLD`     | `0.7`                  | Minimum confidence score for face detection       |
| `LITEVISION_MAX_CACHE_SIZE`           | `100`                  | Maximum number of cached inference results        |
| `LITEVISION_CORS_ORIGINS`            | `["*"]`                | Allowed CORS origins (JSON array)                 |
| `LITEVISION_MODEL_DIR`               | `./models`             | Directory for downloaded ONNX model files         |
| `LITEVISION_CACHE_TTL_SECONDS`       | `60`                   | Cache time-to-live in seconds                     |
| `LITEVISION_MAX_IMAGE_DIMENSION`     | `4096`                 | Maximum allowed image width or height in pixels   |
| `LITEVISION_MAX_CONCURRENT_INFERENCES` | `4`                  | Maximum number of concurrent inference operations |

> **Security note:** In production, always restrict `LITEVISION_CORS_ORIGINS` to your frontend domain instead of using the wildcard `["*"]`.

### Frontend

Frontend environment variables are set in a `.env.local` file in the `frontend/` directory:

```bash
cp .env.example .env.local
```

| Variable              | Default                   | Description             |
|-----------------------|---------------------------|-------------------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000`   | Backend API base URL    |

---

## Docker Deployment

A `Dockerfile` and `.dockerignore` are pre-configured in the `backend/` directory.

### Build the Image

```bash
docker build -t lite-vision-backend ./backend
```

### Run the Container

```bash
docker run -p 8000:8000 lite-vision-backend
```

The backend will be accessible at `http://localhost:8000`. Models auto-download inside the container on the first startup.

### Persisting Models Across Rebuilds

To avoid re-downloading models on every container rebuild, mount a local volume:

```bash
docker run -p 8000:8000 -v $(pwd)/models:/app/models lite-vision-backend
```

---

## Production Deployment

Lite-Vision uses a **two-deployment architecture**:

- **Frontend** hosted on Vercel (free tier, global CDN, automatic HTTPS).
- **Backend** hosted on Railway, Docker, or any cloud provider (requires ~512 MB RAM for models and inference).

The frontend communicates with the backend API via the `NEXT_PUBLIC_API_URL` environment variable.

### Frontend on Vercel

1. **Connect your GitHub repository** to [Vercel](https://vercel.com).

2. **Configure the project settings:**
   - Set **Root Directory** to `frontend`.
   - Framework preset will auto-detect as **Next.js**.

3. **Set environment variables:**
   - `NEXT_PUBLIC_API_URL` = your production backend URL (e.g., `https://lite-vision-api.up.railway.app`).

4. **Deploy.** Vercel handles builds, HTTPS certificates, and CDN distribution automatically.

### Backend on Railway

1. **Connect your GitHub repository** to [Railway](https://railway.app).

2. **Configure the service:**
   - Set **Root Directory** to `backend`.
   - Railway auto-detects Python and installs dependencies from `requirements.txt`.

3. **Set the start command:**

   ```
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

4. **Set environment variables** as needed (see [Environment Variables](#environment-variables)). At minimum, restrict CORS origins:

   ```
   LITEVISION_CORS_ORIGINS=["https://your-frontend.vercel.app"]
   ```

5. **Deploy.** Models auto-download on the first startup. The service needs approximately 512 MB RAM for the models and inference workload.

### Backend on Docker (Any Platform)

This approach works on AWS ECS, GCP Cloud Run, Azure Container Instances, or any Docker-compatible hosting.

1. **Build the Docker image:**

   ```bash
   docker build -t lite-vision-backend ./backend
   ```

2. **Tag and push to your container registry:**

   ```bash
   # Example for Docker Hub
   docker tag lite-vision-backend your-registry/lite-vision-backend:latest
   docker push your-registry/lite-vision-backend:latest
   ```

3. **Deploy to your platform of choice.** Ensure the following:
   - The container port (default `8000`) is exposed.
   - At least **512 MB RAM** is allocated.
   - The `LITEVISION_CORS_ORIGINS` variable is set to your frontend domain.

---

## Architecture Overview

```
┌─────────────────────────┐         HTTPS          ┌──────────────────────────┐
│                         │  ───────────────────>   │                          │
│   Frontend (Vercel)     │                         │   Backend (Railway /     │
│   Next.js 15            │  <───────────────────   │   Docker)                │
│   Port 3000 (dev)       │    JSON responses       │   FastAPI + ONNX Runtime │
│                         │                         │   Port 8000 (dev)        │
└─────────────────────────┘                         └──────────────────────────┘
        │                                                     │
        │  Global CDN                                         │  ONNX Models
        │  Automatic HTTPS                                    │  ~512 MB RAM
        │  Free tier                                          │  CPU inference
```

- The frontend captures webcam frames and sends them to the backend API.
- The backend runs face detection, age/gender classification, and emotion recognition using ONNX models.
- Results (bounding boxes, labels, confidence scores) are returned as normalized JSON and rendered on the frontend canvas.

---

## Troubleshooting

### Backend fails to start

| Symptom | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError` | Dependencies not installed | Run `pip install -r requirements.txt` inside an activated virtual environment |
| Port already in use | Another process on port 8000 | Stop the conflicting process or use `--port 8001` |
| Model download failure | Network restriction or timeout | Check internet connectivity; set `LITEVISION_MODEL_DIR` to a writable path |

### Frontend fails to connect to backend

| Symptom | Cause | Solution |
|---------|-------|----------|
| `Failed to fetch` | Backend not running or wrong URL | Verify the backend is running; check `NEXT_PUBLIC_API_URL` |
| CORS error in browser console | Frontend origin not allowed | Add your frontend URL to `LITEVISION_CORS_ORIGINS` |
| `net::ERR_CONNECTION_REFUSED` | Backend unreachable | Confirm the backend host and port are correct and accessible |

### Docker issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| Build fails at `pip install` | Network issue during build | Ensure Docker has internet access; check proxy settings |
| Container exits immediately | Missing start command or crash | Run `docker logs <container_id>` to inspect the error |
| Models re-download on every restart | No persistent volume | Mount a volume for the `models/` directory (see [Docker Deployment](#docker-deployment)) |

### Production issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| Slow first request | Models loading into memory | This is expected; subsequent requests will be fast |
| Out of memory on Railway | Plan limit too low | Ensure at least 512 MB RAM is allocated to the service |
| Mixed content warnings | Frontend on HTTPS, backend on HTTP | Deploy the backend behind HTTPS (Railway provides this by default) |

---

*This guide is part of the Lite-Vision project documentation. For API details, see [API_REFERENCE.md](./API_REFERENCE.md). For a project overview, see [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md).*

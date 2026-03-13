# Backend Improvements Audit

> Reviewed: `backend/main.py`, `serverless/api/index.py`, deployment configs, and frontend integration.

> **Scope note:** The `serverless/` folder is a lightweight, stripped-down variant of `backend/` built for Vercel deployment. It is currently non-functional and not in active use. All improvements in this document target `backend/` only — the `serverless/` folder should be left untouched unless changes are specifically needed to make it lighter, smoother, or more efficient.

---

## 1. Architecture & Code Organization

| #   | Issue                                                    | Severity | Details                                                                                                                                                                                                                                                                                   |
| --- | -------------------------------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.1 | **Code duplication across `backend/` and `serverless/`** | High     | ~80% of inference logic is copy-pasted. A single shared core module (e.g., `core/inference.py`) should be imported by both entry points. Any bug fix currently requires changes in two places.                                                                                            |
| 1.2 | **Global mutable state for models**                      | High     | `face_net`, `age_net`, `gender_net` are module-level globals mutated at runtime. This is not thread-safe with multiple Uvicorn workers, untestable in isolation, and tightly couples loading with serving. Wrap in a singleton class or use FastAPI's dependency injection (`app.state`). |
| 1.3 | **No API versioning**                                    | Medium   | Routes are `/analyze` (backend) and `/api/analyze` (serverless) — no version prefix. Adding `/v1/` now prevents breaking clients when the response schema changes.                                                                                                                        |
| 1.4 | **Inconsistent route prefixes**                          | Medium   | Backend serves at `/analyze`, serverless at `/api/analyze`. The frontend needs a rewrite to bridge this gap. A single canonical prefix eliminates proxy hacks.                                                                                                                            |
| 1.5 | **Inconsistent bounding box format**                     | High     | Backend returns pixel coordinates `[x, y, w, h]`, serverless returns normalized `[0.0-1.0]`. Frontend has to auto-detect. Pick one format (normalized is the better choice) and standardize.                                                                                              |

---

## 2. Security

| #   | Issue                                      | Severity | Details                                                                                                                                                                                                               |
| --- | ------------------------------------------ | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.1 | **CORS wide open (`allow_origins=["*"]`)** | High     | Allows any domain to call the API. In production, restrict to the actual frontend origin(s). Keep `*` only for local dev via env-based config.                                                                        |
| 2.2 | **No rate limiting**                       | High     | A single client can flood `/analyze` with expensive inference requests. Add middleware-level rate limiting (e.g., `slowapi`) or upstream limits (Vercel/Cloudflare).                                                  |
| 2.3 | **No payload size validation**             | Medium   | The Pydantic model accepts any string length for `image`. A malicious 500MB base64 string would be decoded into memory before any check. Add a `max_length` constraint on the field or validate size before decoding. |
| 2.4 | **No authentication**                      | Medium   | The API is publicly accessible. Even a simple API key header (`X-API-Key`) would prevent unauthorized usage and enable per-client rate limiting.                                                                      |
| 2.5 | **MD5 used for cache keys**                | Low      | MD5 is cryptographically broken. While not a security risk for caching, using `hashlib.sha256` or `xxhash` is a better practice and avoids audit flags.                                                               |
| 2.6 | **Model download over HTTP redirects**     | Medium   | `urllib.request.urlretrieve` follows redirects silently with no checksum verification. A compromised CDN could serve a tampered model. Add SHA-256 checksums for each model file and verify after download.           |

---

## 3. Performance & Scalability

| #   | Issue                                           | Severity | Details                                                                                                                                                                                                                     |
| --- | ----------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3.1 | **Synchronous inference blocks the event loop** | Critical | `analyze()` is a sync `def` in FastAPI, which runs on the main thread pool. OpenCV DNN inference is CPU-bound and blocks other requests. Use `run_in_executor` or move to a background thread pool with a dedicated worker. |
| 3.2 | **In-memory cache has no TTL**                  | Medium   | Cache entries persist indefinitely until evicted by size. Stale entries waste memory. Add a TTL (e.g., 60s) or use an LRU with time-based expiry.                                                                           |
| 3.3 | **Cache not shared across workers/instances**   | Medium   | Each Uvicorn worker or serverless cold start gets its own empty cache. For multi-worker setups, consider a shared in-memory cache layer.                                                                                    |
| 3.4 | **Full base64 string hashed for cache key**     | Low      | `hashlib.md5(raw.encode())` hashes the entire base64 payload (potentially megabytes) on every request, even cache hits. Hash the raw bytes after decode, or use a faster hash (xxhash).                                     |
| 3.5 | **No request concurrency control**              | Medium   | No limit on concurrent inference requests. Under load, all workers could be doing CPU-bound inference simultaneously, starving health checks. Add a semaphore or queue.                                                     |
| 3.6 | **No image preprocessing optimization**         | Low      | Every frame goes through full `blobFromImage` even at 10fps streaming. Consider resizing to a smaller dimension before blob creation for real-time use cases.                                                               |

---

## 4. Error Handling & Resilience

| #   | Issue                                      | Severity | Details                                                                                                                                                                                    |
| --- | ------------------------------------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 4.1 | **Backend crashes if models fail to load** | High     | `backend/main.py` lifespan calls `load_models()` without try/except — any download failure kills the server. The serverless version handles this gracefully; backend should too.           |
| 4.2 | **Silent empty results instead of errors** | Medium   | When `cv2.imdecode` returns `None` (corrupt image), the API returns `{"results": [], "face_count": 0}` — indistinguishable from "no faces found". Return a 422 with a clear error message. |
| 4.3 | **No structured error response schema**    | Medium   | Errors are bare `{"detail": "..."}` strings (FastAPI default). Define an `ErrorResponse` model for consistency and client-side parsing.                                                    |
| 4.4 | **Model download has no retry logic**      | Medium   | `urllib.request.urlretrieve` fails on transient network errors with no retry. Add exponential backoff (2-3 attempts).                                                                      |
| 4.5 | **No request timeout for inference**       | Medium   | A pathologically large image could make inference take 30+ seconds. Add a timeout wrapper around the DNN forward passes.                                                                   |

---

## 5. Configuration Management

| #   | Issue                                      | Severity | Details                                                                                                                                                                                                                              |
| --- | ------------------------------------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 5.1 | **All config is hardcoded**                | High     | `CONFIDENCE_THRESHOLD`, `MAX_CACHE`, image dimensions, mean values — all hardcoded constants. Use environment variables or a config file (e.g., Pydantic `BaseSettings`) so these can be tuned per environment without code changes. |
| 5.2 | **No `.env` support**                      | Medium   | No `python-dotenv` or Pydantic settings loader. Developers must export env vars manually. Add a `.env.example` with documented defaults.                                                                                             |
| 5.3 | **Model URLs hardcoded in multiple files** | Medium   | Model download URLs are duplicated in `backend/main.py` and `serverless/models/download_models.py`. Single source of truth needed.                                                                                                   |

---

## 6. Testing

| #   | Issue                                   | Severity | Details                                                                                                                                                              |
| --- | --------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6.1 | **Zero test coverage**                  | Critical | No `tests/` directory, no test files, no pytest config. Backend code is entirely untested.                                                                           |
| 6.2 | **Code not structured for testability** | High     | Global state, no dependency injection, and tight coupling to OpenCV make unit testing difficult. Abstracting the inference behind an interface would enable mocking. |
| 6.3 | **No integration/contract tests**       | High     | No tests verify the API contract (request/response schemas). A schema change could silently break the frontend. Add tests using FastAPI's `TestClient`.              |
| 6.4 | **No load/stress testing**              | Medium   | No benchmarks for concurrent request handling, memory under load, or cold-start latency.                                                                             |

---

## 7. Observability & Monitoring

| #   | Issue                                    | Severity | Details                                                                                                                                                                                                                         |
| --- | ---------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 7.1 | **No structured logging**                | Medium   | Uses Python's basic `logging` with string formatting. Switch to JSON-structured logs for production log aggregation (ELK, Datadog, etc.).                                                                                       |
| 7.2 | **No metrics collection**                | Medium   | No counters for requests, inference latency, cache hit rate, error rate, or model load time. Add Prometheus metrics or at minimum log these values.                                                                             |
| 7.3 | **No request tracing / correlation IDs** | Low      | No way to trace a request across frontend → backend. Add a middleware that generates/propagates a request ID.                                                                                                                   |
| 7.4 | **Health check is shallow**              | Medium   | `/health` returns `{"status": "ok"}` even in the backend variant where models might have failed to load. The serverless version is better (`models_loaded` flag). Align both, and add checks for memory usage, cache size, etc. |

---

## 8. Deployment & DevOps

| #   | Issue                               | Severity | Details                                                                                                                                                                  |
| --- | ----------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 8.1 | **No Dockerfile**                   | Medium   | No containerized deployment option. A Dockerfile enables consistent local dev, CI testing, and deployment to any container platform (not just Vercel).                   |
| 8.2 | **Dependencies not pinned**         | High     | `requirements.txt` has bare package names (`fastapi`, `numpy`). A new release of any dependency could break the build. Pin exact versions (e.g., `fastapi==0.115.0`).    |
| 8.3 | **No CI/CD pipeline**               | Medium   | No GitHub Actions, no automated tests, no lint checks on PR. Bugs ship uncaught.                                                                                         |
| 8.4 | **No linting or formatting**        | Low      | No `ruff`, `black`, `mypy`, or `pyproject.toml` config. Inconsistent code style across files.                                                                            |
| 8.5 | **`__pycache__` committed to repo** | Low      | Both `backend/__pycache__/` and `serverless/api/__pycache__/` are tracked. Add `__pycache__/` to `.gitignore` (it may already be there but files were committed before). |

---

## 9. API Design

| #   | Issue                                      | Severity | Details                                                                                                                                                                                                  |
| --- | ------------------------------------------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 9.1 | **No OpenAPI documentation customization** | Low      | FastAPI auto-generates docs, but there's no metadata (description, tags, examples). Adding these makes the `/docs` page useful for frontend developers.                                                  |
| 9.2 | **No request ID in responses**             | Low      | Responses don't include a unique identifier for debugging or support.                                                                                                                                    |
| 9.3 | **Binary image sent as base64 JSON**       | Medium   | Base64 encoding inflates payload size by ~33%. For production, accept `multipart/form-data` file uploads as an alternative endpoint — more efficient and avoids the Vercel payload limit issue entirely. |
| 9.4 | **No max faces limit**                     | Low      | If an image has 50 faces, all 50 are processed. Add an optional `max_faces` query param to let clients control cost.                                                                                     |

---

## 10. Data Validation

| #    | Issue                                         | Severity | Details                                                                                                                                                                         |
| ---- | --------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 10.1 | **No image format validation**                | Medium   | The API accepts any base64 string. Invalid or non-image data only fails at `cv2.imdecode` with a silent `None`. Validate the data-URI MIME type or magic bytes before decoding. |
| 10.2 | **No max image dimension check**              | Medium   | A 10000x10000 image would allocate ~300MB of RAM for the numpy array alone. Cap input dimensions server-side.                                                                   |
| 10.3 | **Pydantic `image` field has no constraints** | Medium   | `image: str` accepts any string. Add `Field(min_length=100, max_length=10_000_000)` to reject obviously invalid or oversized payloads early.                                    |

---

## Priority Roadmap

### Phase 1 — Critical (Ship Blockers)

- [x] Pin dependency versions in `requirements.txt`
- [x] Fix sync inference blocking (use thread executor)
- [x] Add payload size validation on `image` field
- [x] Restrict CORS origins for production
- [x] Add basic test suite with `pytest` + `TestClient`
- [x] Graceful error handling in backend `lifespan`

### Phase 2 — High (Stability & Maintainability)

- [ ] ~~Extract shared inference module (`core/`)~~ — Skipped (`serverless/` is inactive and out of scope)
- [x] Standardize bounding box format (normalized) across both backends
- [x] Standardize route prefixes (`/api/`)
- [x] Add rate limiting middleware (`slowapi`, 30 req/min per IP)
- [x] Add model checksum verification on download (SHA-256)
- [x] Environment-based configuration (`BaseSettings` with `LITEVISION_` prefix)

### Phase 3 — Medium (Production Readiness)

- [x] Add Dockerfile for containerized deployment
- [x] CI/CD pipeline (GitHub Actions: lint, test, build — backend + frontend)
- [x] Structured JSON logging (custom `JSONFormatter`)
- [x] Metrics collection (cache hit/miss, inference latency, face count, cache size)
- [x] Add `multipart/form-data` upload endpoint (`POST /api/upload`)
- [x] Cache TTL support (configurable, default 60s)

### Phase 4 — Polish

- [x] OpenAPI docs customization (tags, summaries, version, description)
- [x] Request correlation IDs (`X-Request-ID` middleware)
- [x] Load testing benchmarks (`tests/load_test.py`)
- [x] `max_faces` query parameter (default 20, range 1-100)
- [x] Image format magic-byte validation (JPEG/PNG signatures)

### Bonus — Additional improvements implemented

- [x] Concurrency semaphore (limits parallel inference, default 4)
- [x] Max image dimension check (auto-resize above 4096px)
- [x] Corrupt/empty base64 graceful error handling (422)
- [x] Models stored on `app.state` instead of globals
- [x] `ErrorResponse` schema for structured error responses
- [x] Health check returns `models_loaded` boolean
- [x] 503 response when models not loaded
- [x] `.env.example`, `pyproject.toml`, `.dockerignore`
- [x] Ruff + mypy configuration

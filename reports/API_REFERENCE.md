# Lite-Vision API Reference

> **Base URL:** `https://<your-deployment>/api`
>
> **Content-Type:** `application/json` (unless otherwise noted)
>
> **Version:** 1.0

---

## Table of Contents

1. [Authentication & Headers](#authentication--headers)
2. [Rate Limits](#rate-limits)
3. [Endpoints](#endpoints)
   - [POST /api/analyze](#post-apianalyze)
   - [POST /api/upload](#post-apiupload)
   - [GET /api/health](#get-apihealth)
4. [Schemas](#schemas)
   - [FaceResult](#faceresult)
   - [AnalyzeResponse](#analyzeresponse)
   - [ErrorResponse](#errorresponse)
5. [Configuration](#configuration)
6. [Error Handling](#error-handling)

---

## Authentication & Headers

Lite-Vision does not require authentication by default. All requests and responses carry the following header:

| Header         | Direction        | Description                                                                 |
| -------------- | ---------------- | --------------------------------------------------------------------------- |
| `X-Request-ID` | Request/Response | Correlation ID for tracing. Auto-generated UUID if not supplied by caller.  |

Pass your own `X-Request-ID` to correlate logs across services:

```
X-Request-ID: 550e8400-e29b-41d4-a716-446655440000
```

---

## Rate Limits

| Endpoint           | Limit          |
| ------------------ | -------------- |
| `POST /api/analyze`| 300 req/min    |
| `POST /api/upload` | 30 req/min     |
| `GET /api/health`  | Unlimited      |

When a rate limit is exceeded the server responds with **429 Too Many Requests**.

---

## Endpoints

### POST /api/analyze

Analyze a base64-encoded image for age and gender detection.

#### Query Parameters

| Parameter   | Type  | Default | Constraints | Description                          |
| ----------- | ----- | ------- | ----------- | ------------------------------------ |
| `max_faces` | `int` | `20`    | 1 -- 100    | Maximum number of faces to detect.   |

#### Request Body

| Field   | Type     | Required | Description                                                                                  |
| ------- | -------- | -------- | -------------------------------------------------------------------------------------------- |
| `image` | `string` | Yes      | Base64-encoded JPEG or PNG image (max 10 MB). Accepts optional `data:image/...;base64,` prefix. |

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

#### Response `200 OK`

```json
{
  "results": [
    {
      "age": 28,
      "age_min": 25,
      "age_max": 31,
      "gender": "Male",
      "gender_confidence": 0.9523,
      "confidence": 0.95,
      "region": [0.1, 0.1, 0.8, 0.8],
      "emotion": "neutral",
      "emotion_confidence": 0.8234
    }
  ],
  "face_count": 1,
  "processing_time_ms": 45.2
}
```

#### Error Responses

| Status | Condition                                                  |
| ------ | ---------------------------------------------------------- |
| `422`  | Invalid base64, corrupt image, unsupported format, or empty payload. |
| `429`  | Rate limit exceeded.                                       |
| `503`  | Models not yet loaded (server is still initializing).      |

#### cURL Example

```bash
curl -X POST "https://<your-deployment>/api/analyze?max_faces=5" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: my-trace-id-001" \
  -d '{
    "image": "'"$(base64 -w 0 photo.jpg)"'"
  }'
```

---

### POST /api/upload

Analyze an uploaded image file for age and gender detection. Accepts standard multipart file uploads.

#### Query Parameters

| Parameter   | Type  | Default | Constraints | Description                          |
| ----------- | ----- | ------- | ----------- | ------------------------------------ |
| `max_faces` | `int` | `20`    | 1 -- 100    | Maximum number of faces to detect.   |

#### Request Body

`multipart/form-data`

| Field   | Type   | Required | Description                  |
| ------- | ------ | -------- | ---------------------------- |
| `image` | `file` | Yes      | JPEG or PNG image file.      |

#### Response `200 OK`

Returns the same [AnalyzeResponse](#analyzeresponse) schema as `/api/analyze`.

#### Error Responses

| Status | Condition                              |
| ------ | -------------------------------------- |
| `422`  | Empty file or corrupt image.           |
| `429`  | Rate limit exceeded.                   |
| `503`  | Models not yet loaded.                 |

#### cURL Example

```bash
curl -X POST "https://<your-deployment>/api/upload?max_faces=10" \
  -H "X-Request-ID: my-trace-id-002" \
  -F "image=@photo.jpg"
```

---

### GET /api/health

Health check endpoint. Returns the current server status and loaded model information.

#### Response `200 OK`

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

#### cURL Example

```bash
curl "https://<your-deployment>/api/health"
```

---

## Schemas

### FaceResult

Describes a single detected face and its predicted attributes.

| Field                | Type            | Description                                                                 |
| -------------------- | --------------- | --------------------------------------------------------------------------- |
| `age`                | `int`           | Predicted age.                                                              |
| `age_min`            | `int`           | Lower bound of the predicted age range.                                     |
| `age_max`            | `int`           | Upper bound of the predicted age range.                                     |
| `gender`             | `string`        | Predicted gender. One of `"Male"` or `"Female"`.                            |
| `gender_confidence`  | `float`         | Confidence score for the gender prediction (0.0 -- 1.0).                    |
| `confidence`         | `float`         | Overall face detection confidence (0.0 -- 1.0).                             |
| `region`             | `list[float]`   | Normalized bounding box as `[x, y, w, h]` relative to image dimensions.    |
| `emotion`            | `string \| null` | Predicted primary emotion (e.g., `"neutral"`, `"happy"`), or `null`.       |
| `emotion_confidence` | `float \| null`  | Confidence score for the emotion prediction (0.0 -- 1.0), or `null`.      |

### AnalyzeResponse

Top-level response returned by both `/api/analyze` and `/api/upload`.

| Field                | Type              | Description                                      |
| -------------------- | ----------------- | ------------------------------------------------ |
| `results`            | `list[FaceResult]` | Array of detected faces and their attributes.   |
| `face_count`         | `int`             | Total number of faces detected.                  |
| `processing_time_ms` | `float`           | Server-side processing time in milliseconds.     |

### ErrorResponse

Returned for all non-2xx responses.

| Field    | Type     | Description                        |
| -------- | -------- | ---------------------------------- |
| `detail` | `string` | Human-readable error description.  |

Example:

```json
{
  "detail": "Invalid base64 encoding in image field."
}
```

---

## Configuration

All configuration is managed through environment variables prefixed with `LITEVISION_`.

| Variable                              | Type         | Default     | Description                                                     |
| ------------------------------------- | ------------ | ----------- | --------------------------------------------------------------- |
| `LITEVISION_CONFIDENCE_THRESHOLD`     | `float`      | `0.7`       | Minimum face detection confidence to include in results.        |
| `LITEVISION_MAX_CACHE_SIZE`           | `int`        | `100`       | Maximum number of entries in the inference result cache.         |
| `LITEVISION_CORS_ORIGINS`             | `list[str]`  | `["*"]`     | Allowed CORS origins.                                           |
| `LITEVISION_MODEL_DIR`                | `str`        | `./models`  | Directory path where model files are stored.                    |
| `LITEVISION_CACHE_TTL_SECONDS`        | `int`        | `60`        | Time-to-live in seconds for cached inference results.           |
| `LITEVISION_MAX_IMAGE_DIMENSION`      | `int`        | `4096`      | Maximum allowed width or height (in pixels) for input images.   |
| `LITEVISION_MAX_CONCURRENT_INFERENCES`| `int`        | `4`         | Maximum number of inference tasks that may run concurrently.    |

---

## Error Handling

All errors follow a consistent structure using the [ErrorResponse](#errorresponse) schema.

| Status Code             | Meaning                                                                                  |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| `422 Unprocessable Entity` | The request was well-formed but the image payload is invalid (bad base64, corrupt file, unsupported format, or empty). |
| `429 Too Many Requests`    | The client has exceeded the rate limit for the endpoint.                                |
| `503 Service Unavailable`  | The server is still loading models and cannot process inference requests yet.           |

Clients should inspect the `detail` field for a human-readable explanation of the failure and use the `X-Request-ID` header to correlate errors with server-side logs.

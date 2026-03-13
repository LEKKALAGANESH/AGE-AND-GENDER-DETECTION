# Lite-Vision Frontend Architecture

> **Application:** Lite-Vision -- Real-time Age & Gender Detection
> **Stack:** Next.js 15 (App Router) | TypeScript | Tailwind CSS | Vercel
> **Last updated:** 2026-03-13

---

## Table of Contents

1. [Technology Stack](#1-technology-stack)
2. [Project Structure](#2-project-structure)
3. [Component Architecture](#3-component-architecture)
4. [Custom Hooks](#4-custom-hooks)
5. [Library Modules](#5-library-modules)
6. [Data Flow](#6-data-flow)
7. [Page Structure & Routing](#7-page-structure--routing)
8. [Security & Middleware](#8-security--middleware)
9. [Configuration](#9-configuration)
10. [Testing](#10-testing)
11. [Deployment](#11-deployment)

---

## 1. Technology Stack

| Layer          | Technology              | Version   |
|----------------|-------------------------|-----------|
| Framework      | Next.js (App Router)    | 15.1+     |
| Language       | TypeScript              | 5.7+      |
| UI Runtime     | React                   | 19.0+     |
| Styling        | Tailwind CSS            | 3.4+      |
| Icons          | Lucide React            | 0.468+    |
| Utilities      | clsx                    | 2.1+      |
| Test Runner    | Vitest                  | latest    |
| Test Libraries | Testing Library (React, Jest-DOM, User Event) | latest |
| Bundler (Dev)  | Turbopack               | built-in  |
| Hosting        | Vercel                  | --        |

---

## 2. Project Structure

```
frontend/
├── next.config.ts              # Next.js config (rewrites, standalone output)
├── tailwind.config.ts          # Tailwind CSS theme and content paths
├── postcss.config.mjs          # PostCSS plugin chain
├── vitest.config.ts            # Vitest test runner configuration
├── .eslintrc.json              # ESLint rule overrides
├── .prettierrc                 # Prettier formatting rules
├── .env.example                # Environment variable template
├── package.json                # Dependencies and scripts
│
└── src/
    ├── middleware.ts            # Security headers middleware
    │
    ├── app/                    # Next.js App Router pages
    │   ├── layout.tsx          # Root layout (Inter font, metadata, viewport)
    │   ├── page.tsx            # Home page (renders Camera in Suspense boundary)
    │   ├── globals.css         # Global styles (Tailwind directives)
    │   ├── error.tsx           # Error boundary component
    │   ├── loading.tsx         # Loading skeleton (streaming SSR fallback)
    │   └── not-found.tsx       # Custom 404 page
    │
    ├── components/
    │   ├── Camera.tsx           # Main orchestrator component (246 lines)
    │   ├── camera/
    │   │   ├── Controls.tsx     # Camera control buttons
    │   │   ├── DropZone.tsx     # Drag-and-drop image upload zone
    │   │   ├── ResultsPanel.tsx # Detection results display
    │   │   └── DebugPanel.tsx   # Developer diagnostics panel
    │   └── __tests__/           # Component test suites
    │
    ├── hooks/
    │   ├── useCamera.ts         # Webcam stream lifecycle
    │   ├── useAnalyze.ts        # API request management
    │   ├── useFileUpload.ts     # File upload with compression
    │   └── useTemporalSmoothing.ts  # Prediction stabilization (214 lines)
    │
    ├── lib/
    │   ├── api.ts               # HTTP client (fetch + compression)
    │   ├── canvas.ts            # Canvas overlay renderer (393 lines)
    │   ├── constants.ts         # Application-wide constants
    │   └── types.ts             # TypeScript interfaces and helpers
    │
    └── test/                    # Test utilities and setup
```

---

## 3. Component Architecture

### 3.1 Component Hierarchy

```
page.tsx
└── <Suspense fallback={<CameraFallback />}>
    └── <Camera>                           # Main orchestrator
        ├── <DropZone>                     # Drag-and-drop container
        │   ├── <video>                    # Live webcam feed
        │   ├── <img>                      # Upload preview
        │   ├── <canvas> (overlay)         # Bounding box overlay
        │   ├── <canvas> (capture)         # Hidden frame capture
        │   └── Placeholder / Loading      # State-dependent UI
        ├── <Controls>                     # Action buttons
        ├── <DebugPanel>                   # Dev-only diagnostics
        ├── Error alert                    # Inline error display
        └── <ResultsPanel>                 # Detection result cards
            ├── <ProcessingTimeBadge>      # Latency indicator
            ├── <GenderConfidenceBars>     # Dual-bar gender chart
            └── <AgeRangeBar>             # Age range visualization
```

### 3.2 Camera.tsx -- Main Orchestrator

**Path:** `src/components/Camera.tsx` (246 lines)

The Camera component is the central orchestrator of the application. It coordinates webcam access, frame capture, API communication, temporal smoothing, and overlay rendering. It does not contain rendering logic for sub-elements directly; instead, it delegates to modular sub-components.

**Responsibilities:**

- Initializes and manages webcam stream via `useCamera`
- Captures video frames to an offscreen `<canvas>` at configurable intervals
- Converts frames to base64 JPEG (quality 0.8) for API transmission
- Orchestrates the streaming loop with backpressure-safe recursive `setTimeout`
- Applies temporal smoothing during live streaming mode
- Passes smoothed results to `drawOverlay()` for canvas rendering
- Manages two operational modes: single capture and continuous streaming

**Key State Management:**

| Ref / State       | Purpose                                     |
|--------------------|---------------------------------------------|
| `canvasRef`        | Offscreen canvas for frame capture           |
| `overlayRef`       | Visible canvas for bounding box rendering    |
| `busyRef`          | Prevents overlapping frame captures          |
| `streamingRef`     | Synchronizes streaming state with setTimeout |
| `timeoutRef`       | Tracks the active streaming timeout          |
| `streaming`        | React state controlling live detection mode  |
| `debug`            | Toggles the developer debug panel            |

**Streaming Architecture:**

The streaming loop uses recursive `setTimeout` (not `setInterval`) to provide natural backpressure. Each frame waits for the previous analysis to complete before scheduling the next, with a minimum gap of 200ms (~5 fps maximum). This prevents request pile-up during high server latency.

```
┌──────────────────────────────────────────────────┐
│              Streaming Loop                       │
│                                                   │
│  scheduleNextFrame()                              │
│    └── setTimeout(200ms)                          │
│          └── captureFrame()                       │
│                ├── Draw video to offscreen canvas  │
│                ├── Convert to base64 JPEG          │
│                ├── await analyze(base64)            │
│                ├── smoothResults(data.results)      │
│                ├── drawOverlay(overlay, data, ...)  │
│                └── scheduleNextFrame()  ←── recurse │
└──────────────────────────────────────────────────┘
```

### 3.3 Sub-Components (camera/ directory)

#### Controls.tsx

Renders the action button bar. Adapts its layout based on camera state:

- **Idle state:** Start Camera + Upload Image buttons
- **Active state:** Capture (single frame) + Live Detect (toggle streaming) + Stop buttons
- Disables the Capture button during active streaming
- Visual feedback: streaming toggle uses red (stop) / green (start) color coding

#### DropZone.tsx

An accessible drag-and-drop container wrapping the video/upload area:

- Keyboard accessible (`role="button"`, `tabIndex={0}`, Enter/Space triggers file select)
- Visual feedback during drag hover (blue border highlight, "Drop image here" overlay)
- Aspect ratio locked to 16:9 (`aspect-video`)

#### ResultsPanel.tsx

Displays structured detection results below the video area:

- **ProcessingTimeBadge:** Color-coded latency indicator (green < 200ms, yellow < 500ms, red >= 500ms). Shows a pulsing "LIVE" badge during streaming.
- **GenderConfidenceBars:** Dual-bar horizontal chart showing Male/Female probability split with percentage labels.
- **AgeRangeBar:** Visual range bar with a dot marker at the predicted age position. Displays min/max bounds and deviation.
- **Emotion indicator:** Shows detected facial expression with confidence percentage.
- **Accuracy warning:** Alerts when expressive faces (happiness, surprise, contempt) may affect gender prediction accuracy.

#### DebugPanel.tsx

Developer-only panel (hidden in production builds) for API diagnostics:

- **Ping:** Sends a GET request to `/api/health` and displays the response with latency.
- **Test POST:** Creates a 10x10 gray canvas, sends it as a JPEG to `/api/analyze`, and reports the round-trip result.
- Color-coded output: green for success, red for failure.

---

## 4. Custom Hooks

### 4.1 useCamera

**Path:** `src/hooks/useCamera.ts`

Manages the webcam `MediaStream` lifecycle using the `getUserMedia` API.

| Export        | Type                | Description                              |
|---------------|---------------------|------------------------------------------|
| `videoRef`    | `Ref<HTMLVideoElement>` | Ref to attach to the `<video>` element |
| `status`      | `CameraStatus`      | Current state: `idle`, `starting`, `active`, `streaming`, `error` |
| `setStatus`   | Setter              | Allows parent to update status (e.g., to `streaming`) |
| `startCamera` | `() => Promise<void>` | Requests camera access with 640x480 preferred, front-facing |
| `stopCamera`  | `() => void`        | Stops all tracks and releases the stream |

Performs cleanup on unmount to prevent zombie media streams.

### 4.2 useAnalyze

**Path:** `src/hooks/useAnalyze.ts`

Manages API request lifecycle with automatic cancellation of superseded requests.

| Export       | Type                  | Description                                 |
|--------------|-----------------------|---------------------------------------------|
| `results`   | `AnalyzeResponse | null` | Latest successful response                |
| `setResults` | Setter               | Allows external updates (e.g., after smoothing) |
| `error`      | `string | null`      | Error message from the latest request       |
| `isLoading`  | `boolean`            | True while a request is in flight           |
| `analyze`   | `(base64) => Promise` | Sends image, cancels any previous in-flight request |
| `clearError` | `() => void`         | Resets the error state                      |

**Request cancellation:** Uses `AbortController` to cancel stale requests. When a new `analyze()` call is made before the previous one completes, the previous request is aborted. Aborted requests do not update component state.

### 4.3 useFileUpload

**Path:** `src/hooks/useFileUpload.ts`

Handles image file upload via drag-and-drop or file picker with automatic compression.

| Export            | Type          | Description                              |
|-------------------|---------------|------------------------------------------|
| `uploadPreview`   | `string | null` | Data URL for preview display           |
| `dragOver`        | `boolean`     | True when a file is dragged over the zone |
| `isProcessing`    | `boolean`     | True during file read + compression      |
| `clearPreview`    | `() => void`  | Clears the upload preview                |
| `dropProps`       | Object        | Spread onto the drop zone element        |
| `inputProps`      | Object        | Spread onto the hidden file `<input>`    |
| `triggerFileSelect` | `() => void` | Programmatically opens the file picker  |

Uploads are always compressed before analysis. Phone camera images (5-15 MB) are resized to a maximum width of 1024px and re-encoded as JPEG at 70% quality.

### 4.4 useTemporalSmoothing

**Path:** `src/hooks/useTemporalSmoothing.ts` (214 lines)

The most algorithmically significant hook. It stabilizes face detection predictions across streaming frames to prevent visual jitter.

**Algorithm Overview:**

```
Frame N arrives with raw detections
        │
        ▼
┌───────────────────────────────┐
│  Step 1: IoU Matching         │
│  Match each new detection to  │
│  existing tracked faces using │
│  greedy IoU assignment        │
│  (threshold: 0.3)            │
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│  Step 2: EMA Smoothing        │
│  For matched faces, blend:    │
│  - Age (rounded integer)      │
│  - Age range (min, max)       │
│  - Gender confidence          │
│  - Face confidence            │
│  - Bounding box coordinates   │
│  alpha = 0.3 (30% new frame)  │
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│  Step 3: Gender Majority Vote │
│  Maintain buffer of last 8    │
│  gender predictions per face. │
│  Require 6/8 (75%) agreement  │
│  to flip gender label.        │
│  Otherwise, retain previous.  │
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│  Step 4: New Face Registration│
│  Unmatched detections become  │
│  new tracked faces with       │
│  initial values.              │
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│  Step 5: Stale Face Pruning   │
│  Unmatched tracked faces get  │
│  framesMissed++. Pruned after │
│  10 consecutive missed frames.│
└───────────┬───────────────────┘
            │
            ▼
    Return smoothed FaceResult[]
```

**Constants:**

| Constant               | Value | Purpose                                         |
|------------------------|-------|--------------------------------------------------|
| `ALPHA`                | 0.3   | EMA blending factor (30% new, 70% history)       |
| `STALE_THRESHOLD`      | 10    | Frames before a lost face is pruned               |
| `IOU_THRESHOLD`        | 0.3   | Minimum IoU to consider two boxes the same face   |
| `GENDER_VOTE_BUFFER`   | 8     | Number of recent gender votes to retain           |
| `GENDER_FLIP_THRESHOLD`| 6     | Minimum votes (of 8) needed to change gender label|

**IoU Computation:** The `computeIoU` function calculates Intersection over Union for `[x, y, w, h]` bounding boxes, supporting both normalized (0-1) and pixel coordinate formats.

---

## 5. Library Modules

### 5.1 api.ts -- HTTP Client

**Path:** `src/lib/api.ts`

Two exported functions for backend communication:

**`compressImage(base64, maxWidth?, quality?)`**
- Resizes images exceeding `maxWidth` while preserving aspect ratio
- Re-encodes as JPEG at the specified quality
- 10-second timeout to prevent hanging on corrupt images
- Used for both uploads and payload size enforcement

**`analyzeImage(base64, signal?)`**
- Sends a POST request to the analyze endpoint with `{ image: base64 }`
- Automatic compression if payload exceeds 4 MB (Vercel's 4.5 MB limit minus safety margin)
- Returns `{ data, error }` tuple (never throws)
- 15-second fetch timeout via internal `AbortController`
- Chains with external `AbortSignal` for request cancellation
- Handles HTTP errors, network errors, timeouts, and oversized payloads gracefully

### 5.2 canvas.ts -- Overlay Renderer

**Path:** `src/lib/canvas.ts` (393 lines)

Renders the sci-fi styled detection overlay on an HTML `<canvas>` element. The `drawOverlay` function is the main entry point, called after each successful analysis.

**Visual Elements:**

| Element              | Function                | Description                                     |
|---------------------|-------------------------|-------------------------------------------------|
| Corner brackets      | `drawCornerBrackets()`  | Sci-fi style L-shaped corners around each face   |
| Bounding box         | Inline                  | Semi-transparent rectangle outline               |
| Scan line            | `drawScanLine()`        | Animated laser line bouncing vertically (streaming only) |
| Age/Gender label     | Inline                  | Header label: "Male, 24 +/-3 yrs" with gender icon |
| Confidence bars      | `drawConfidenceBars()`  | Dual-bar chart (blue=Male, pink=Female) adjacent to box |
| Focus ring           | `drawFocusRing()`       | Dashed border indicating prediction reliability  |
| Emotion indicator    | `drawEmotionIndicator()`| Expression label below the bounding box          |
| Timestamp            | `drawTimestamp()`       | Live HH:MM:SS.mmm clock (streaming only)         |
| Processing badge     | `drawProcessingBadge()` | Latency readout with color coding (streaming only)|

**Coordinate Handling:** Supports both normalized (0-1) and absolute pixel coordinates. Detects the format by checking if all region values are <= 1, then scales to canvas dimensions accordingly.

**Color Scheme:**

| Gender  | Primary Color | Hex       |
|---------|---------------|-----------|
| Male    | Blue          | `#3b82f6` |
| Female  | Pink          | `#ec4899` |

**Focus Ring Color Coding:**

| Condition                          | Color  | Meaning              |
|------------------------------------|--------|----------------------|
| Gender confidence < 60%            | Red    | Low confidence       |
| Expressive face (> 50% confidence) | Yellow | May affect accuracy  |
| Default                            | Green  | Reliable prediction  |

**Scan Line Animation:** Uses `performance.now()` with a triangle wave function for smooth 2-second bounce cycles. Multi-pass rendering creates a glow effect (wide soft pass, medium pass, sharp core line).

### 5.3 types.ts -- Type Definitions

**Path:** `src/lib/types.ts`

```typescript
interface FaceResult {
  age: number;
  age_min?: number;               // Lower bound of predicted age range
  age_max?: number;               // Upper bound of predicted age range
  gender: string;                  // "Male" or "Female"
  gender_confidence?: number;      // Gender prediction confidence (0-1)
  confidence: number;              // Face detection confidence (0-1)
  region: [number, number, number, number]; // [x, y, width, height]
  emotion?: string;                // Detected facial expression
  emotion_confidence?: number;     // Expression detection confidence (0-1)
}

interface AnalyzeResponse {
  results: FaceResult[];           // Array of detected faces
  face_count: number;              // Total faces detected
  processing_time_ms: number;      // Server-side processing duration
}
```

**Helper Functions:**
- `getAgeRange(face)` -- Resolves optional `age_min`/`age_max` with a +/-5 year fallback.
- `getGenderConfidence(face)` -- Resolves optional `gender_confidence` with a fallback to `confidence`.

### 5.4 constants.ts -- Application Constants

**Path:** `src/lib/constants.ts`

| Constant                     | Value    | Purpose                                       |
|------------------------------|----------|------------------------------------------------|
| `API_URL`                    | env or `/api/analyze` | Backend endpoint (configurable via `NEXT_PUBLIC_API_URL`) |
| `HEALTH_URL`                 | derived  | Health check endpoint (derived from `API_URL`)  |
| `MAX_PAYLOAD_BYTES`          | 4 MB     | Safety margin below Vercel's 4.5 MB body limit  |
| `COMPRESS_MAX_WIDTH`         | 800 px   | Max width for automatic pre-send compression    |
| `COMPRESS_QUALITY`           | 0.7      | JPEG quality for standard compression           |
| `COMPRESS_QUALITY_AGGRESSIVE`| 0.6      | JPEG quality for oversized payload compression  |
| `UPLOAD_MAX_WIDTH`           | 1024 px  | Max width for user-uploaded images              |
| `STREAM_INTERVAL_MS`         | 100 ms   | Base streaming interval                         |
| `FETCH_TIMEOUT_MS`           | 15000 ms | API request timeout                             |
| `COMPRESS_TIMEOUT_MS`        | 10000 ms | Image compression timeout                       |

---

## 6. Data Flow

### 6.1 End-to-End Pipeline

```
                              FRONTEND                                      BACKEND
┌─────────────────────────────────────────────────────────────┐   ┌──────────────────────┐
│                                                             │   │                      │
│  ┌──────────┐    ┌───────────┐    ┌────────────┐            │   │                      │
│  │ getUserMedia │─>│ <video>   │─>│ <canvas>   │            │   │                      │
│  │ (webcam)  │    │ (live)    │    │ (capture)  │            │   │                      │
│  └──────────┘    └───────────┘    └─────┬──────┘            │   │                      │
│                                         │                    │   │                      │
│       OR                                │ toDataURL()        │   │                      │
│                                         │ (JPEG, q=0.8)     │   │                      │
│  ┌──────────┐    ┌───────────┐          │                    │   │                      │
│  │ File Drop │─>│ FileReader │──────────┤                    │   │                      │
│  │ / Select  │    │ + Compress│          │                    │   │                      │
│  └──────────┘    └───────────┘          │                    │   │                      │
│                                         ▼                    │   │                      │
│                               ┌─────────────────┐           │   │                      │
│                               │ analyzeImage()  │           │   │                      │
│                               │ - Size check    │           │   │                      │
│                               │ - Auto compress │           │   │                      │
│                               │ - Timeout (15s) │           │   │                      │
│                               └────────┬────────┘           │   │                      │
│                                        │                     │   │                      │
│                                        │ POST /api/analyze   │   │                      │
│                                        │ {image: base64}     │   │                      │
│                                        │─────────────────────┼──>│  Face Detection      │
│                                        │                     │   │  Age Prediction       │
│                                        │  AnalyzeResponse    │   │  Gender Classification│
│                                        │<────────────────────┼───│  Emotion Analysis     │
│                                        │                     │   │                      │
│                                        ▼                     │   └──────────────────────┘
│                             ┌──────────────────┐             │
│                             │ useTemporalSmoothing │         │
│                             │ (streaming only)  │            │
│                             │ - IoU face matching│           │
│                             │ - EMA age smoothing│           │
│                             │ - Gender voting   │            │
│                             └────────┬─────────┘            │
│                                      │                       │
│                    ┌─────────────────┼──────────────┐        │
│                    ▼                 ▼              ▼        │
│           ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│           │ drawOverlay()│  │ ResultsPanel │  │ State    │  │
│           │ (canvas)     │  │ (HTML cards) │  │ Update   │  │
│           └──────────────┘  └──────────────┘  └──────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Streaming Flow (Step by Step)

1. User clicks "Live Detect" -- `toggleStream()` sets streaming state and calls `scheduleNextFrame()`.
2. `scheduleNextFrame()` sets a `setTimeout` of 200ms minimum.
3. When timeout fires, `captureFrame()` is invoked:
   a. Guard check: if `busyRef` is true, skip (prevents overlap).
   b. Draw current video frame to the hidden capture canvas.
   c. Export canvas as base64 JPEG at 80% quality.
   d. Call `analyze(base64)` -- sends POST to `/api/analyze`.
   e. On response, pass results through `smoothResults()` for temporal stabilization.
   f. Call `drawOverlay()` to render bounding boxes on the visible overlay canvas.
   g. Update the `ResultsPanel` with smoothed data.
4. After `captureFrame()` completes, `scheduleNextFrame()` recurses.
5. User clicks "Stop Live" -- clears the timeout, resets smoothing state.

### 6.3 Upload Flow

1. User drops an image file or selects via file picker.
2. `useFileUpload` reads the file with `FileReader.readAsDataURL()`.
3. Image is compressed (max 1024px width, JPEG quality 0.7).
4. Preview is displayed in the `DropZone`.
5. Compressed base64 is sent to `analyze()`.
6. Results are rendered by `ResultsPanel` and overlay canvas.
7. No temporal smoothing is applied (single-shot analysis).

---

## 7. Page Structure & Routing

### 7.1 Root Layout (`layout.tsx`)

- Font: Inter (Google Fonts, swap display strategy)
- Viewport: device-width, initial-scale 1, theme-color `#0a0a0a`
- Metadata: title "Lite-Vision", OpenGraph tags for social sharing
- Body: `antialiased` class for font rendering

### 7.2 Home Page (`page.tsx`)

Single-page application structure:

- `<main>` container: full viewport height, centered content
- Heading: "Lite-Vision" with subtitle "Real-time age & gender detection"
- `<Suspense>` boundary wrapping the `<Camera>` component with a skeleton fallback

The `CameraFallback` component renders a pulsing skeleton with a spinner, matching the dimensions of the camera view area.

### 7.3 Error Boundary (`error.tsx`)

Client component that catches runtime errors:

- Displays a warning icon, "Something went wrong" message, and a "Try again" button
- Logs errors to the console
- `reset()` callback re-renders the error boundary's children

### 7.4 Loading Page (`loading.tsx`)

Streaming SSR loading skeleton:

- Title placeholder (animated pulse)
- Camera view area skeleton (aspect-video with dashed border)
- Control button skeletons (two pill-shaped placeholders)

### 7.5 Not Found Page (`not-found.tsx`)

Custom 404 page:

- Large "404" heading with "Page not found" subtitle
- Link back to the home page

---

## 8. Security & Middleware

**Path:** `src/middleware.ts`

The middleware runs on every request except static assets and sets the following security headers:

| Header                    | Value                                                    | Purpose                          |
|---------------------------|----------------------------------------------------------|----------------------------------|
| `X-Frame-Options`        | `DENY`                                                    | Prevents clickjacking            |
| `X-Content-Type-Options` | `nosniff`                                                 | Prevents MIME-type sniffing      |
| `Referrer-Policy`        | `origin-when-cross-origin`                                | Controls referrer information     |
| `Permissions-Policy`     | `camera=(self), microphone=()`                            | Restricts API access to camera only, denies microphone |
| `Content-Security-Policy`| `default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self' https: http://localhost:*; media-src 'self' blob:; frame-ancestors 'none'` | Comprehensive CSP policy |

**Matcher exclusions:** `_next/static`, `_next/image`, `favicon.ico`

---

## 9. Configuration

### 9.1 next.config.ts

```typescript
{
  output: "standalone",          // Optimized for Docker/Vercel deployment
  rewrites: [
    { source: "/api/:path*",     // Proxy API calls to the backend
      destination: "${BACKEND_URL}/api/:path*" }
  ]
}
```

The `BACKEND_URL` environment variable (defaults to `http://localhost:8000`) determines where API requests are proxied. This allows the frontend to call `/api/analyze` without CORS issues.

### 9.2 Environment Variables

| Variable              | Required | Default              | Description                          |
|-----------------------|----------|----------------------|--------------------------------------|
| `BACKEND_URL`         | No       | `http://localhost:8000` | Backend API base URL (server-side rewrite) |
| `NEXT_PUBLIC_API_URL` | No       | `/api/analyze`       | Client-side API endpoint override    |

### 9.3 Scripts

| Script          | Command                    | Description                    |
|-----------------|----------------------------|--------------------------------|
| `dev`           | `next dev --turbopack`     | Development server with Turbopack |
| `build`         | `next build`               | Production build               |
| `start`         | `next start`               | Start production server        |
| `lint`          | `next lint`                | Run ESLint                     |
| `test`          | `vitest run`               | Run tests once                 |
| `test:watch`    | `vitest`                   | Run tests in watch mode        |
| `test:coverage` | `vitest run --coverage`    | Run tests with coverage report |

---

## 10. Testing

| Tool                         | Purpose                              |
|------------------------------|--------------------------------------|
| Vitest                       | Test runner (compatible with Jest API)|
| @testing-library/react       | Component rendering and queries      |
| @testing-library/jest-dom    | DOM assertion matchers               |
| @testing-library/user-event  | User interaction simulation          |
| jsdom                        | Browser environment for Node.js      |
| @vitejs/plugin-react         | React JSX transform for Vite/Vitest  |

**Test locations:**
- `src/components/__tests__/` -- Component integration tests
- `src/test/` -- Test utilities and shared setup

---

## 11. Deployment

### 11.1 Vercel Configuration

The frontend is deployed on Vercel with the following architecture:

```
┌────────────────────────────────────────────┐
│              Vercel Edge Network            │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │  Next.js Frontend                    │  │
│  │  - SSR / Static pages                │  │
│  │  - API rewrites to BACKEND_URL       │  │
│  │  - Middleware (security headers)     │  │
│  └──────────────┬───────────────────────┘  │
│                 │ /api/:path* rewrite       │
│                 ▼                           │
│  ┌──────────────────────────────────────┐  │
│  │  Backend (FastAPI)                   │  │
│  │  - Separate Vercel deployment        │  │
│  │  - Or external server                │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
```

### 11.2 Build Optimizations

- **Standalone output:** Generates a minimal production bundle with only necessary dependencies, reducing deployment size.
- **Turbopack (dev):** Uses Rust-based bundler for fast development rebuilds.
- **Font optimization:** Inter font loaded via `next/font/google` with `display: swap` for zero layout shift.
- **Payload compression:** Automatic JPEG compression ensures requests stay under Vercel's 4.5 MB body limit.

### 11.3 Performance Characteristics

| Metric                    | Value          | Notes                                |
|---------------------------|----------------|--------------------------------------|
| Max streaming FPS         | ~5 fps         | 200ms minimum frame gap              |
| Frame capture quality     | JPEG 80%       | Balance of quality vs. payload size  |
| Upload compression target | 1024px width   | Reduces phone camera images          |
| API timeout               | 15 seconds     | Prevents indefinite loading states   |
| Compression timeout       | 10 seconds     | Prevents hung image processing       |
| Max payload size          | 4 MB           | Safety margin below Vercel's limit   |
| EMA smoothing factor      | 0.3            | 30% new frame, 70% history          |
| Gender flip threshold     | 75% (6/8)      | Prevents flickering labels           |
| Stale face timeout        | 10 frames      | Cleans up lost face tracks           |

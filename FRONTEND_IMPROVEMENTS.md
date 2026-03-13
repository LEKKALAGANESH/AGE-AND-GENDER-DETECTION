# Frontend Improvement Areas — Lite-Vision

> Audit performed on the Next.js 15 + React 19 + Tailwind CSS frontend.
> Reviewed files: `page.tsx`, `Camera.tsx`, `layout.tsx`, `globals.css`, `next.config.ts`, `package.json`, `tsconfig.json`

---

## 1. Architecture & Component Structure

| Issue                                                                                                                                                         | Where                  | Severity |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | -------- |
| **Monolithic component** — `Camera.tsx` is ~450 lines handling API calls, state, canvas drawing, drag-and-drop, debug panel, and all UI rendering             | `Camera.tsx`           | High     |
| **No separation of concerns** — business logic (API, compression, overlay drawing) is co-located with presentation                                            | `Camera.tsx`           | High     |
| **No custom hooks extraction** — camera lifecycle, API communication, and file handling should be isolated hooks (`useCamera`, `useAnalyze`, `useFileUpload`) | `Camera.tsx`           | High     |
| **No error boundary** — a runtime error in the Camera component crashes the entire app with no recovery                                                       | `layout.tsx` / app dir | Medium   |
| **Missing Next.js conventions** — no `error.tsx`, `loading.tsx`, or `not-found.tsx` in the app directory                                                      | `src/app/`             | Medium   |

### Recommended Decomposition

```
src/
  hooks/
    useCamera.ts          # webcam start/stop/stream lifecycle
    useAnalyze.ts         # API call + compression logic
    useFileUpload.ts      # drag-and-drop + file input
  components/
    Camera/
      CameraView.tsx      # video + overlay canvas
      Controls.tsx        # buttons for capture/stream/stop
      ResultsPanel.tsx    # face detection results display
      DebugPanel.tsx      # debug tools (conditionally loaded)
      DropZone.tsx        # drag-and-drop wrapper
    ErrorBoundary.tsx
  lib/
    api.ts                # fetch wrapper, types, compression
    canvas.ts             # overlay drawing utilities
    constants.ts          # magic numbers consolidated
```

---

## 2. State Management

| Issue                                                                                                                                                                                                    | Where                | Severity |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | -------- |
| **8 independent `useState` calls** — `active`, `streaming`, `results`, `error`, `uploadPreview`, `dragOver`, `debug`, `debugMsg` make state transitions hard to reason about                             | `Camera.tsx:56-63`   | High     |
| **No state machine** — camera has implicit states (idle / starting / active / streaming / error) with no formal transitions, making impossible states possible (e.g., `streaming=true` + `active=false`) | `Camera.tsx`         | High     |
| **`busyRef` is a concurrency hack** — using a mutable ref to prevent overlapping requests is fragile; should use `AbortController` or a proper request queue                                             | `Camera.tsx:54`      | Medium   |
| **Stale closure risk** — `toggleStream` captures `captureFrame` via `useCallback` deps, but `setInterval` holds a stale reference if deps change                                                         | `Camera.tsx:192-200` | Medium   |

### Recommendation

Use `useReducer` with a discriminated union state or adopt a lightweight state machine (XState or a simple finite-state reducer):

```ts
type CameraState =
  | { status: "idle" }
  | { status: "starting" }
  | { status: "active"; streaming: boolean }
  | { status: "error"; message: string }
  | { status: "upload"; preview: string };
```

---

## 3. Performance

| Issue                                                                                                                                                                                                                                 | Where                | Severity |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | -------- |
| **`setInterval` at 100ms ignores backpressure** — if the API takes 500ms, frames queue up behind `busyRef` gate; should use recursive `setTimeout` or `requestAnimationFrame` to send the next frame only after the previous response | `Camera.tsx:198`     | High     |
| **`compressImage` creates throwaway DOM elements** — every call creates a new `Image` + `Canvas` element; should reuse a pooled offscreen canvas or `OffscreenCanvas`                                                                 | `Camera.tsx:12-33`   | Medium   |
| **No `AbortController` on fetch** — navigating away or stopping the camera doesn't cancel in-flight requests                                                                                                                          | `Camera.tsx:111`     | Medium   |
| **Array index as React key** — `key={i}` on face results causes unnecessary re-renders and incorrect reconciliation if the list changes                                                                                               | `Camera.tsx:420`     | Low      |
| **No memoization** — result cards could benefit from `React.memo` when streaming at high frequency                                                                                                                                    | `Camera.tsx:419-442` | Low      |
| **Redundant JSON serialization for size check** — `new Blob([JSON.stringify({ image })]).size` serializes the payload just to measure it, then serializes again for fetch                                                             | `Camera.tsx:101`     | Low      |

---

## 4. Error Handling & Resilience

| Issue                                                                                                                                                           | Where                | Severity |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | -------- |
| **`compressImage` never rejects** — the `Promise` has no `onerror` handler on the `Image` element; a corrupt/invalid image will hang forever                    | `Camera.tsx:17`      | High     |
| **No fetch timeout** — API calls can hang indefinitely with no abort signal or timeout                                                                          | `Camera.tsx:111-115` | High     |
| **FileReader `onerror` not handled** — if file reading fails, nothing happens (silent failure)                                                                  | `Camera.tsx:207-218` | Medium   |
| **Non-null assertions (`!`)** — `canvas.getContext("2d")!` assumes success; `getContext` can return null in specific browser contexts (e.g., too many canvases) | `Camera.tsx:27`      | Medium   |
| **No retry/backoff for transient errors** — a single network blip during streaming kills the session with no auto-recovery                                      | `Camera.tsx:121-124` | Medium   |
| **Error state is never auto-cleared** — once an error appears, it persists even after successful subsequent requests (only manually cleared in some paths)      | `Camera.tsx:59`      | Low      |

---

## 5. Accessibility (a11y)

| Issue                                                                                                                                            | Where                     | Severity |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------- | -------- |
| **No ARIA labels** — buttons have text but no `aria-label` for icon-only states; upload `<label>` wrapping an `<input>` lacks proper association | `Camera.tsx:282-333`      | High     |
| **Color-only gender distinction** — Male=blue, Female=pink with no icon/pattern/text differentiation for colorblind users                        | `Camera.tsx:148, 423-427` | High     |
| **No live region for results** — screen readers get no announcement when detection results appear or update                                      | `Camera.tsx:414-448`      | High     |
| **No keyboard navigation plan** — custom buttons work, but the drag-and-drop area is mouse-only with no keyboard alternative                     | `Camera.tsx:232-244`      | Medium   |
| **No focus management** — switching between camera/upload modes doesn't move focus logically                                                     | `Camera.tsx`              | Medium   |
| **Video has no `<track>` element** — browsers may warn about missing captions                                                                    | `Camera.tsx:247-253`      | Low      |
| **Insufficient color contrast** — `text-zinc-500` on dark backgrounds may fail WCAG AA (4.5:1 ratio)                                             | Multiple                  | Medium   |

---

## 6. TypeScript & Type Safety

| Issue                                                                                                                                                                                                                                                           | Where                            | Severity |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | -------- |
| **Frontend/backend type drift** — frontend `region` is `[number, number, number, number]` (tuple) while backend sends `list[int]` with pixel values; also backend `region` contains `[x, y, w, h]` but frontend checks for normalized values — no shared schema | `Camera.tsx:39` vs `main.py:125` | High     |
| **Non-null assertions** — `getContext("2d")!` used in `compressImage` without null check                                                                                                                                                                        | `Camera.tsx:27`                  | Medium   |
| **No API response validation** — `res.json()` is trusted without runtime validation (e.g., Zod)                                                                                                                                                                 | `Camera.tsx:120`                 | Medium   |
| **Loose `string` type for `gender`** — should be a union `"Male" \| "Female"` to match backend                                                                                                                                                                  | `Camera.tsx:37`                  | Low      |

---

## 7. Testing

| Issue                                                                                                       | Severity |
| ----------------------------------------------------------------------------------------------------------- | -------- |
| **Zero test files** — no unit tests, integration tests, or e2e tests exist in the project                   | Critical |
| **No test configuration** — no Jest, Vitest, Playwright, or Cypress setup                                   | Critical |
| **No testable architecture** — business logic embedded in components makes unit testing extremely difficult | High     |
| **No MSW or API mocking setup** — cannot test API interactions without a running backend                    | Medium   |

### Minimum Test Coverage Needed

- `lib/api.ts` — compression logic, fetch wrapper, error handling (unit)
- `lib/canvas.ts` — overlay coordinate math (unit)
- `useCamera` hook — lifecycle states (hook testing with `renderHook`)
- `Camera` component — render states, user interactions (component)
- Upload flow — file selection, drag-drop (integration)
- Full detection pipeline — e2e with mocked API (Playwright/Cypress)

---

## 8. Security

| Issue                                                                                                                                                  | Where               | Severity |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------- | -------- |
| **No security headers** — missing `Content-Security-Policy`, `X-Frame-Options`, etc.; no Next.js middleware for headers                                | `next.config.ts`    | High     |
| **File upload accepts any "image/\*"** — no validation of actual file content (magic bytes), file size limit, or dimensions before processing          | `Camera.tsx:204`    | Medium   |
| **Base64 data passed without sanitization** — large payloads sent directly to API; no client-side size limit UX (user sees error only after upload)    | `Camera.tsx:97-124` | Medium   |
| **CORS wildcard on backend** — `allow_origins=["*"]` in production is overly permissive                                                                | `main.py:109`       | Medium   |
| **`output: "standalone"` with rewrites** — rewrite rules don't apply on Vercel's Edge Network for standalone builds, potentially exposing backend URLs | `next.config.ts`    | Low      |

---

## 9. UX & User Experience

| Issue                                                                                                                                              | Where                | Severity |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | -------- |
| **No loading indicator for uploads** — after selecting a file, users see nothing until results appear                                              | `Camera.tsx:203-219` | High     |
| **No camera initialization feedback** — "Start Camera" button has no loading/spinner while waiting for `getUserMedia` permission                   | `Camera.tsx:66-82`   | High     |
| **"Stream 10fps" is technical jargon** — end users won't understand fps; use "Live Detection" or "Auto-Detect"                                     | `Camera.tsx:323`     | Medium   |
| **No image paste support** — users can't Ctrl+V an image from clipboard, a common expectation                                                      | `Camera.tsx`         | Medium   |
| **Debug panel ships to production** — should be behind `NODE_ENV` check or feature flag, or dynamically imported                                   | `Camera.tsx:337-404` | Medium   |
| **No results history** — previous detection results are lost when a new analysis is triggered                                                      | `Camera.tsx:58`      | Low      |
| **Drag zone doesn't indicate supported formats** — "drop an image" doesn't tell users which formats are accepted                                   | `Camera.tsx:273`     | Low      |
| **No mobile-specific UX** — no camera switching (front/back), buttons may be too small on mobile                                                   | `Camera.tsx:70`      | Medium   |
| **Overlay doesn't render on uploaded images** — `drawOverlay` is called for webcam frames but uploaded images only show results in the panel below | `Camera.tsx:214-216` | Medium   |

---

## 10. DevOps, Tooling & Configuration

| Issue                                                                                                                                           | Where                              | Severity |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | -------- |
| **No ESLint configuration** — no linting rules enforced; `next lint` would fail without config                                                  | project root                       | High     |
| **No Prettier / formatting config** — no consistent code formatting enforced                                                                    | project root                       | Medium   |
| **No `.env.example`** — `NEXT_PUBLIC_API_URL` and `BACKEND_URL` are referenced but undocumented                                                 | `Camera.tsx:7`, `next.config.ts:3` | Medium   |
| **No CI/CD pipeline** — no GitHub Actions for lint, type-check, test, build                                                                     | project root                       | High     |
| **No pre-commit hooks** — no Husky/lint-staged to catch issues before commit                                                                    | project root                       | Low      |
| **`autoprefixer` in devDeps but no `.browserslistrc`** — Tailwind v3 handles prefixing; autoprefixer is redundant without a browserslist target | `package.json:24`                  | Low      |

---

## 11. Next.js Best Practices

| Issue                                                                                                                                                                                    | Where              | Severity |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | -------- |
| **No metadata API usage beyond basics** — missing Open Graph, Twitter Card, theme-color, viewport, icons                                                                                 | `layout.tsx:4-7`   | Medium   |
| **No `Suspense` boundaries** — the Camera component is `"use client"` but never wrapped in `Suspense` for progressive loading                                                            | `page.tsx`         | Medium   |
| **No middleware** — should add security headers, rate limiting indicators, or geo-redirect via `middleware.ts`                                                                           | `src/`             | Medium   |
| **Standalone output may not suit Vercel** — `output: "standalone"` is for Docker/self-hosted; Vercel uses its own build output                                                           | `next.config.ts:6` | Low      |
| **No `next/font` usage** — using system fonts implicitly, but no explicit font optimization via Next.js built-in font system                                                             | `layout.tsx`       | Low      |
| **No image optimization** — uploaded preview uses raw `<img>` instead of `next/image`; though dynamic images are tricky, a `blurDataURL` placeholder would improve perceived performance | `Camera.tsx:257`   | Low      |

---

## 12. Styling & Design System

| Issue                                                                                                                                                       | Where                | Severity |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | -------- |
| **Hardcoded colors in CSS** — `#0a0a0a` and `#ededed` should be Tailwind theme tokens or CSS custom properties                                              | `globals.css:5-8`    | Medium   |
| **No dark/light mode support** — hardcoded dark theme with no toggle or `prefers-color-scheme` respect                                                      | `globals.css`        | Medium   |
| **Very long className strings** — some elements have 80+ character class strings; consider `cva` (class-variance-authority) or component-level abstractions | `Camera.tsx:284-285` | Low      |
| **Tailwind theme not extended** — `theme.extend` is empty; project-specific colors (brand blue, brand pink, bg dark) should be defined as semantic tokens   | `tailwind.config.ts` | Low      |
| **No animation/transition consistency** — some buttons have `transition-colors`, others don't                                                               | `Camera.tsx`         | Low      |

---

## Priority Matrix

| Priority          | Items                                                                                                                                                                                              |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **P0 — Critical** | Add tests & test infrastructure, decompose Camera.tsx into hooks + components                                                                                                                      |
| **P1 — High**     | State machine for camera lifecycle, fix error handling (Promise rejection, fetch timeout, AbortController), add accessibility (ARIA, live regions, keyboard), add loading states, security headers |
| **P2 — Medium**   | Extract API layer with Zod validation, add ESLint + Prettier, CI/CD pipeline, `.env.example`, mobile UX improvements, streaming backpressure fix                                                   |
| **P3 — Low**      | Design system tokens, Next.js metadata enrichment, `next/font`, result history, clipboard paste support                                                                                            |

---

_Generated by a senior frontend architecture review._

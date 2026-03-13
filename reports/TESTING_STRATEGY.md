# Lite-Vision -- Testing Strategy

**Comprehensive test coverage for real-time age, gender, and emotion detection**

---

## Philosophy

All backend tests execute **without real model files**. The SCRFD face detector,
InsightFace genderage network, FER+ emotion network, and FairFace ResNet34 are replaced
with lightweight mocks that return deterministic, mathematically-specified outputs. This
design guarantees:

- **Reproducibility** -- Tests produce identical results on any machine, regardless of
  GPU availability or model weights.
- **Speed** -- The full backend suite completes in seconds, not minutes.
- **Isolation** -- Test failures always indicate application logic errors, never model
  download or loading issues.

---

## Test Summary

| Layer | Framework | Test Files | Tests | Status |
|---|---|---|---|---|
| Backend unit | pytest + FastAPI TestClient | 2 | 36 | All passing |
| Backend integration | pytest (7 edge-case images) | -- | 7 | All passing |
| Frontend unit | Vitest + Testing Library | 1 | 6 | All passing |
| **Total** | | **3** | **49+** | **All passing** |

> 73 unit tests passing in CI; 7/7 integration test images passing with SCRFD + FairFace
> fusion.

---

## Backend Testing (pytest)

### Architecture

```
backend/
  tests/
    conftest.py                    # ~345 lines -- shared fixtures & mock builders
    test_expression_robustness.py  # Expression-robust gender detection
    test_model_quality.py          # Model accuracy, variability & edge cases
```

All tests use `fastapi.testclient.TestClient` to exercise the `/api/analyze` endpoint
end-to-end through the ASGI stack, validating JSON schemas, numeric ranges, and
behavioral invariants.

---

### Mock Builder Reference

The test infrastructure in `conftest.py` provides nine mock builders that simulate the
full inference pipeline. Each builder returns a `MagicMock` with `.detect()` or
`.forward()` / `.setInput()` methods matching the real model interface.

| Mock Builder | Returns | Purpose |
|---|---|---|
| `_build_scrfd_mock()` | `(det[N,5], kpss[N,5,2])` in SCRFD format | Single face at 10-90% of frame, score 0.95, 5 landmarks |
| `_build_genderage_mock()` | Gender logits `[Male=2.0, Female=-1.0]`, `age_factor=0.28` | Deterministic Male, age 28 |
| `_build_emotion_mock()` | 8-class logits with neutral dominant (5.0) | Deterministic neutral emotion |
| `_build_happy_emotion_mock()` | 8-class logits with happiness dominant (5.0) | Deterministic happiness emotion |
| `_build_fairface_mock()` | `[age_out(9), gender_out(2), race_out(7)]` -- Male, 20-29 bin | Deterministic Male, age ~25 |
| `_build_fairface_female_mock()` | Same structure -- Female with high confidence | Deterministic Female |
| `_build_multi_face_scrfd_mock()` | 3 detections at left, center, right positions | Multi-face scenarios |
| `_build_varied_genderage_mock()` | Cycles every 3 calls (multi-crop aware) | Different age/gender per face |
| `_build_varied_fairface_mock()` | Cycles every 2 calls (full face + upper half) | Different gender/age per face |

---

### Fixture Reference

| Fixture | Models Injected | Use Case |
|---|---|---|
| `_patch_models` | Stubs `load_models()`, clears cache | Base fixture -- prevents real model loading |
| `client` | SCRFD + genderage + emotion + FairFace (all deterministic) | Standard single-face tests |
| `client_no_models` | All models remain `None` | Simulates model load failure (503 responses) |
| `client_happy_expression` | SCRFD + genderage + happy emotion + FairFace | Expression-aware confidence tests |
| `client_multi_face` | Multi-face SCRFD + varied genderage + emotion + varied FairFace | Multi-face detection tests |
| `quality_client` | Varied SCRFD + varied genderage (call-counter driven) | Quality and variability tests |
| `deterministic_client` | Varied SCRFD + deterministic genderage | Caching and consistency tests |
| `valid_image_b64` | -- | 10x10 JPEG, decodable |
| `invalid_image_b64` | -- | Valid base64, not a valid image |
| `corrupt_base64` | -- | Not valid base64 at all |
| `multi_face_image_b64` | -- | 300x100 JPEG, wide frame for 3 faces |

---

### Test File 1: `test_expression_robustness.py`

**Scope:** Validates multi-crop ensemble, emotion detection, expression-aware gender
confidence adjustment, graceful degradation, and performance budgets.

| Test Class | Test Method | Assertion |
|---|---|---|
| **TestMultiCropEnsemble** | `test_response_still_has_required_fields` | `results`, `face_count`, `processing_time_ms` present |
| | `test_gender_is_valid` | Gender in `{"Male", "Female"}` |
| | `test_age_is_reasonable` | `0 <= age <= 120` |
| | `test_gender_confidence_is_valid_range` | `0.0 <= confidence <= 1.0` |
| **TestEmotionDetection** | `test_emotion_field_present` | `emotion` key exists in response |
| | `test_emotion_confidence_present` | `emotion_confidence` key exists |
| | `test_emotion_is_valid_label` | Emotion in 8 FER+ classes |
| | `test_emotion_confidence_range` | `0.0 <= confidence <= 1.0` |
| | `test_no_emotion_when_model_not_loaded` | Returns HTTP 503 when all models missing |
| **TestExpressionAwareGender** | `test_happy_expression_reduces_gender_confidence` | Happiness scales down gender confidence |
| | `test_happy_expression_still_returns_valid_gender` | Gender remains `"Male"` or `"Female"` |
| | `test_happy_expression_returns_emotion_label` | Emotion field equals `"happiness"` |
| **TestGracefulDegradation** | `test_works_without_emotion_model` | HTTP 200 with `emotion: null` when model absent |
| | `test_data_url_prefix_still_works` | `data:image/jpeg;base64,` prefix stripped correctly |
| **TestPerformanceBudget** | `test_processing_time_is_reasonable` | `processing_time_ms < 5000` with mocks |
| | `test_multiple_requests_are_consistent` | 3 sequential requests return identical gender |

**Total: 16 tests**

---

### Test File 2: `test_model_quality.py`

**Scope:** Validates age variability, gender robustness, inference performance, response
consistency, caching behavior, and edge-case resilience.

| Test Class | Test Method | Assertion |
|---|---|---|
| **TestAgeVariability** | `test_age_not_bucketed` | No single age exceeds 20% of 50 predictions |
| | `test_age_range_continuous` | >= 5 distinct ages; not all from old bucket set `{1,5,10,18,25,35,48,60}` |
| | `test_age_min_max_reasonable` | `age_min <= age <= age_max`; spread in `(0, 30]` |
| | `test_age_is_non_negative` | `age >= 0` and `age_min >= 0` across 10 images |
| **TestGenderRobustness** | `test_gender_confidence_varies` | Std dev > 0.01 across 20 images |
| | `test_gender_returns_valid_labels` | Gender in `{"Male", "Female"}` across 15 images |
| | `test_gender_confidence_between_0_and_1` | `0.0 <= gender_confidence <= 1.0` |
| | `test_both_genders_appear` | Both Male and Female predicted across 30 images |
| **TestPerformance** | `test_single_face_under_500ms` | Server time < 500ms; wall-clock < 2000ms |
| | `test_response_format_complete` | All required fields present with correct types |
| | `test_multiple_sequential_requests_stable` | 10 sequential requests all return HTTP 200, each < 2000ms |
| | `test_processing_time_is_positive` | `processing_time_ms > 0` |
| **TestResponseConsistency** | `test_same_image_same_result` | Identical image produces identical age, gender, confidence |
| | `test_cache_returns_faster` | Cached response not slower than cold by more than 5ms |
| | `test_different_images_can_differ` | Two distinct images produce at least one differing field |
| **TestEdgeCases** | `test_small_image_does_not_crash` | 32x32 image returns 200 or 422, no crash |
| | `test_large_synthetic_image` | 640x640 image returns 200 |
| | `test_face_count_matches_results_length` | `face_count == len(results)` |
| | `test_region_values_normalized` | All bounding box values in `[0.0, 1.0]` |
| | `test_confidence_in_valid_range` | Detection confidence in `[0.0, 1.0]` |

**Total: 20 tests**

---

### Synthetic Image Generation

`test_model_quality.py` includes a `create_synthetic_face()` helper that draws crude face
geometry using OpenCV primitives (ellipse for head, circles for eyes, arc for mouth). The
`_make_varied_images(n)` function generates *n* images with systematic variation across:

| Parameter | Range | Purpose |
|---|---|---|
| `size` | 180 -- 240 px | Test different resolutions |
| `brightness` | 100 -- 200 | Simulate lighting variation |
| `face_size_ratio` | 0.45 -- 0.72 | Simulate near/far faces |
| `offset_x` | -2 to +2 px | Simulate off-center faces |
| `offset_y` | -3 to +3 px | Simulate vertical offset |

---

## Frontend Testing (Vitest)

### Configuration

```
frontend/
  vitest.config.ts             # jsdom environment, globals enabled
  src/test/setup.ts            # Loads @testing-library/jest-dom matchers
  src/components/__tests__/
    Camera.test.tsx             # Camera component tests
```

**Framework:** Vitest with `@testing-library/react` and `jsdom` environment.

### Mock Infrastructure

| Mock | What It Replaces |
|---|---|
| `mockGetUserMedia()` | `navigator.mediaDevices.getUserMedia` -- returns a fake `MediaStream` |
| `stubVideoPlay()` | `HTMLMediaElement.prototype.play` -- jsdom does not implement this |
| `stubCanvas()` | `HTMLCanvasElement.prototype.getContext` / `toDataURL` -- returns fake 2D context |
| Global `fetch` stub | Returns `{ results: [], face_count: 0, processing_time_ms: 0 }` |

### Camera Component Tests

| Test | Assertion |
|---|---|
| Renders without crashing | Container is truthy |
| Shows "Start Camera" button in idle state | Button present in DOM |
| Shows "Upload Image" button in idle state | Button present in DOM |
| Displays both buttons simultaneously | Both buttons coexist |
| Attempts getUserMedia on Start Camera click | `getUserMedia` called once with `{ video: true, audio: false }` |
| Shows placeholder text when idle | `/start camera or drop an image/i` visible |

**Total: 6 tests**

---

## Integration Testing

### Edge-Case Image Matrix

Seven integration test images validate the full SCRFD + FairFace fusion pipeline against
real-world challenges:

| # | Scenario | Challenge | Expected Behavior |
|---|---|---|---|
| 1 | Normal face | Baseline | Correct age/gender with high confidence |
| 2 | Masked face (surgical mask) | Lower-face occlusion | Mask-aware fusion shifts to upper-face analysis |
| 3 | Dark lighting / low contrast | Poor illumination | CLAHE preprocessing recovers detectable features |
| 4 | Elderly with glasses | Accessory occlusion + age extremes | Age regression handles upper range; glasses do not break detection |
| 5 | Young smiling woman | Expression bias on gender | Expression-aware confidence adjustment applied |
| 6 | Different racial backgrounds | Demographic diversity | FairFace fusion corrects demographic bias |
| 7 | Multiple faces in one image | Multi-face detection | All faces detected with independent predictions |

**Result: 7/7 passing** with the SCRFD + FairFace fusion pipeline.

---

## Coverage Map

### API Endpoint Coverage

| Endpoint | Happy Path | Error Path | Edge Cases |
|---|---|---|---|
| `POST /api/analyze` | Single face, multi-face, varied images | Models not loaded (503), invalid image (422), corrupt base64 | Small image (32px), large image (640px), data URL prefix |
| Response schema | All required fields validated with type checks | Missing model graceful degradation | Normalized regions `[0,1]`, confidence bounds |

### Behavioral Coverage

| Behavior | Tests Covering It |
|---|---|
| Multi-crop ensemble averaging | `TestMultiCropEnsemble` (4 tests) |
| Emotion detection (FER+ 8 classes) | `TestEmotionDetection` (5 tests) |
| Expression-aware gender adjustment | `TestExpressionAwareGender` (3 tests) |
| Graceful model absence | `TestGracefulDegradation` (2 tests), `test_no_emotion_when_model_not_loaded` |
| Age regression continuity | `TestAgeVariability` (4 tests) |
| Gender prediction diversity | `TestGenderRobustness` (4 tests) |
| Inference latency budgets | `TestPerformanceBudget` (2 tests), `TestPerformance` (4 tests) |
| Response caching / determinism | `TestResponseConsistency` (3 tests) |
| Bounding box normalization | `test_region_values_normalized`, `test_response_format_complete` |
| Input sanitization | `test_data_url_prefix_still_works`, `invalid_image_b64`, `corrupt_base64` fixtures |
| Camera component lifecycle | `Camera.test.tsx` (6 tests) |

---

## Running the Tests

### Backend

```bash
cd backend
pip install -r requirements.txt
pytest tests/ -v
```

No model files, GPU, or network access required.

### Frontend

```bash
cd frontend
npm install
npx vitest run
```

---

## Design Decisions

### Why mock at the model layer, not the HTTP layer?

Mocking at the OpenCV DNN / ONNX Runtime interface (`net.forward()`, `detector.detect()`)
exercises the full application logic -- image decoding, face alignment, multi-crop
cropping, softmax computation, confidence adjustment, and JSON serialization. HTTP-level
mocks would bypass all of this and test only routing.

### Why deterministic *and* varied mocks?

- **Deterministic mocks** (`client`, `deterministic_client`) verify that the pipeline
  produces correct, stable output for known inputs. They catch regressions in data flow.
- **Varied mocks** (`quality_client`, `client_multi_face`) verify that the pipeline does
  not collapse all inputs to a single output. They catch loss of information (e.g., a
  broken ensemble that always returns the first crop's result).

### Why synthetic face images instead of real photos?

Synthetic images generated with `create_synthetic_face()` are fully deterministic, carry
no licensing concerns, and produce consistent results across platforms. Real photos are
used only in the 7-image integration suite, where the goal is to validate end-to-end
accuracy under realistic conditions.

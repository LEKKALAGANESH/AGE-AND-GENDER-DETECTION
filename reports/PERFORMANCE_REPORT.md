# Lite-Vision Performance Report

> **Version:** 4.0 (SCRFD + FairFace)
> **Date:** March 2026
> **Test Suite:** 85 automated tests (73 backend pytest + 12 frontend Vitest), all run against mocked inference
> **Status:** All tests passing
>
> **Scope note:** This document reports **pipeline and integration behaviour, not model
> accuracy.** Every automated test mocks the ONNX networks, so no number here measures how
> well the models predict age or gender. Accuracy figures quoted for SCRFD and FairFace are
> **published by their upstream authors** and were not reproduced or measured in this project.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Manual Edge-Case Spot-Check](#manual-edge-case-spot-check-not-an-accuracy-benchmark)
3. [Version Comparison](#version-comparison)
4. [Inference Performance](#inference-performance)
5. [Architecture Performance](#architecture-performance)
6. [Model Inventory](#model-inventory)
7. [Unit Test Results](#unit-test-results)
8. [Key Techniques](#key-techniques)

---

## Executive Summary

Lite-Vision v4.0 replaces YuNet with SCRFD for face detection and adds FairFace for gender
classification. In a **manual, by-eye spot-check** of a handful of hard sample images, the
v4.0 pipeline produced sensible output on scenarios where v3.0 visibly failed — masked
faces, low light, and cross-racial gender calls. This was an informal developer check, not
a scored benchmark, and it is not reproducible from this repository.

---

## Manual Edge-Case Spot-Check (not an accuracy benchmark)

> **Read this before quoting anything below.** The table records a **one-off manual
> inspection** of sample images by the developer. It is **not** an accuracy benchmark and
> carries no statistical meaning:
>
> - **No automated test loads these images.** The scenarios below were checked by hand and
>   are not covered by any test in `backend/tests/`.
> - **"Pass" means "output looked plausible to a human"**, not that it was scored against
>   ground-truth labels.
> - **A handful of hand-picked images cannot measure model accuracy.** Any real figure would
>   need a held-out labelled dataset, which this project does not have.
> - **The sample set is not fully archived.** `testing images/` ships 5 images; the 7 rows
>   below describe scenarios exercised during development, so the table cannot be replayed
>   as-is.
>
> No aggregate score is reported here, because a hand-picked sample does not support one.

| Sample Image | Challenge | Mechanism exercised | Observed by eye |
|---|---|---|---|
| Normal face | Baseline | Detection + fusion | Plausible |
| Masked face | Surgical mask occlusion | SCRFD + mask-aware age fusion | Plausible |
| Dark lighting | Low contrast / exposure | CLAHE + 50/50 age blend | Plausible |
| Elderly + glasses | Occlusion + aging | SCRFD detection | Plausible |
| Young smiling woman | Expression bias on gender | FairFace fusion | Plausible |
| Racial diversity | Cross-racial gender call | FairFace fusion | Plausible |
| Multiple faces | Multi-face detection | Per-face independent prediction | Plausible (3 faces found) |

---

## Version Comparison

Rationale for moving from v3.0 (YuNet) to v4.0 (SCRFD + FairFace):

| Metric | v3.0 (YuNet) | v4.0 (SCRFD + FairFace) | Source |
|---|---|---|---|
| Face detection (WIDERFace Hard) | 70.8% AP | 82.8% AP | **Upstream published figures** (InsightFace/SCRFD authors) -- not measured here |
| Gender accuracy | -- | 95.7% | **Upstream published figure** (FairFace paper) -- not measured here |
| Masked face detection | Visibly failed | Looked correct | Manual spot-check |
| Dark lighting detection | Visibly failed | Looked correct | Manual spot-check |

The two headline percentages are the **model authors' own benchmark results on their own
datasets**. They are the reason these models were selected; they are **not** measurements of
this project and say nothing about how the fused pipeline performs on your data.

---

## Inference Performance

Benchmarks collected with mocked model inference to isolate pipeline overhead:

| Metric | Value |
|---|---|
| End-to-end processing (mocked models) | < 5,000 ms (test budget) |
| Result consistency | Deterministic across repeated requests |
| Cache hit latency | < 1 ms (near-instant) |

Mocked-model testing ensures reproducible CI results without requiring real ONNX model files on the test runner.

---

## Architecture Performance

| Parameter | Configuration |
|---|---|
| Concurrency | 4 simultaneous inference threads |
| Cache strategy | LRU with 60s TTL, 100 entries max |
| Cache key | SHA-256 hash of image bytes |
| Rate limit (analyze) | 300 requests / min |
| Rate limit (upload) | 30 requests / min |
| Image resize threshold | Auto-downscale images > 4,096 px |
| Memory footprint | ~512 MB with all 4 models loaded |

The LRU cache uses SHA-256 content hashing for deduplication, ensuring that identical images submitted within the TTL window return cached results without triggering redundant inference.

---

## Model Inventory

| Model | File | Size |
|---|---|---|
| SCRFD 10G KPS | `scrfd_10g_kps.onnx` | 16.9 MB |
| InsightFace GenderAge | `genderage.onnx` | ~5 MB |
| FER+ Emotion | `emotion-ferplus-8.onnx` | ~35 MB |
| FairFace ResNet34 | `fairface.onnx` | 85.2 MB |
| **Total** | | **~102 MB** |

All models are loaded at startup and held in memory for the lifetime of the process. Inference runs through **OpenCV's DNN module** (`cv2.dnn.readNetFromONNX`) -- the `onnxruntime` package is **not** a dependency of this project.

---

## Unit Test Results

| Stat | Value |
|---|---|
| Backend tests (pytest) | 73 |
| Frontend tests (Vitest) | 12 |
| **Total** | **85** |
| Passing | 85 |
| Failing | 0 |

**Backend test files (`backend/tests/`):**

- `test_analyze.py` (15) -- `/api/analyze` request handling and response contract.
- `test_expression_robustness.py` (16) -- consistent response shape across expression and edge-case inputs.
- `test_health.py` (4) -- `/api/health` readiness and model-status reporting.
- `test_model_quality.py` (20) -- response **structure, variability, and determinism** (see caveat below).
- `test_multi_face.py` (9) -- multi-face response handling.
- `test_validation.py` (9) -- input validation and rejection of malformed payloads.

**Frontend test files (`frontend/src/`):**

- `components/__tests__/Camera.test.tsx` (6) -- camera component rendering and controls.
- `lib/__tests__/api.test.ts` (6) -- API client behaviour.

All tests run with **mocked models**, requiring no real ONNX model files on the test runner. This allows the full suite to execute in CI environments without large binary dependencies.

> **Caveat on `test_model_quality.py`:** despite the name, it does **not** measure model
> accuracy. The mocked networks return fixed, deterministic outputs regardless of input, and
> the inputs are synthetic shapes drawn with OpenCV primitives -- not real faces. These tests
> assert that the API returns well-formed, varied, non-bucketed values; they cannot and do
> not validate prediction correctness.

---

## Key Techniques

The following techniques contribute to Lite-Vision v4.0's accuracy and performance gains:

1. **CLAHE Preprocessing**
   Contrast Limited Adaptive Histogram Equalization is applied to input images before detection, boosting face visibility in dark and low-contrast scenes.

2. **Multi-Crop Ensemble**
   Three crop variants of each detected face are generated and averaged to produce a robust gender prediction that is less sensitive to alignment and framing.

3. **FairFace Fusion**
   The FairFace ResNet34 model, trained on a racially-balanced dataset, provides a gender correction signal that is fused with the InsightFace GenderAge output to reduce demographic bias.

4. **Mask-Aware Age Fusion**
   When occlusion (e.g., a surgical mask) is detected, the age estimator shifts to upper-face analysis, using periorbital features to estimate age through the obstruction.

5. **ArcFace Alignment**
   A similarity transform aligns detected faces to a canonical coordinate frame using five facial landmarks, ensuring consistent input geometry for all downstream classifiers.

6. **LRU Caching with SHA-256 Deduplication**
   Each incoming image is hashed with SHA-256. Repeated submissions within the 60-second TTL window are served from cache, eliminating redundant inference and reducing average response time.

---

*Generated for Lite-Vision v4.0 -- SCRFD + FairFace architecture.*

# Changelog

All notable changes to Lite-Vision will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0] — Multi-Model Fusion Architecture (Current)

### Added

- SCRFD 10G KPS face detector, replacing YuNet (WIDERFace Hard AP: 70.8% to 82.8%).
- FairFace ResNet34 model for racially-balanced gender classification (95.7% accuracy).
- Multi-crop ensemble (3 crop variants) for expression-robust gender detection.
- Mask-aware age fusion: FairFace upper-face analysis detects aging through masks.
- CLAHE preprocessing in SCRFD detection for dark and low-contrast images.
- FER+ emotion detection with expression-aware gender confidence adjustment.
- SCRFDDetector class with NMS, anchor caching, and letterbox resize.
- `_predict_fairface()` with ImageNet normalization and 3 output heads.
- `_align_face()` with ArcFace similarity transform.
- 73 unit tests passing across the test suite.

### Changed

- Gender fusion logic: InsightFace + FairFace agreement produces higher confidence;
  disagreement defers to FairFace.
- Age fusion: 50/50 InsightFace + FairFace blend under normal conditions,
  15/85 InsightFace + upper-face blend when a mask is detected.
- Test score improved from 4/7 to 7/7 on edge case test images.
- API version bumped to 4.0.0.

### Removed

- YuNet face detector (replaced by SCRFD 10G KPS).

## [3.x] — Vercel Deployment Architecture

### Added

- Vercel-compatible backend with normalized bounding boxes.

### Changed

- Rebuilt as Lite-Vision: FastAPI + Next.js 15 full-stack app.

### Fixed

- Vercel build pipeline for two-deployment architecture.
- "Failed to fetch" error: graceful model loading, payload compression,
  and route config.

## [2.x] — Enhanced Features

### Added

- Emotion detection via FER+ model.
- Multi-crop ensemble for gender robustness.
- Expression-aware gender confidence adjustment.
- Concurrency semaphore for inference.
- LRU cache with TTL.
- Rate limiting (slowapi).
- Structured JSON logging.
- Correlation ID middleware.

## [1.0] — Initial Release

### Added

- Face detection with YuNet.
- InsightFace genderage model for age and gender prediction.
- FastAPI backend with base64 image analysis endpoint.
- Next.js frontend with real-time webcam capture.
- Canvas overlay with bounding boxes.

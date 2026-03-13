# Lite-Vision -- ML Models Documentation

**Comprehensive reference for the ONNX inference models powering Lite-Vision's
detection and classification pipeline**

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Model 1 -- SCRFD 10G KPS (Face Detection)](#model-1----scrfd-10g-kps-face-detection)
3. [Model 2 -- InsightFace GenderAge (Age + Gender)](#model-2----insightface-genderage-age--gender)
4. [Model 3 -- FER+ Emotion (Expression Detection)](#model-3----fer-emotion-expression-detection)
5. [Model 4 -- FairFace ResNet34 (Racially-Balanced Gender + Age)](#model-4----fairface-resnet34-racially-balanced-gender--age)
6. [Model Size Summary](#model-size-summary)
7. [Fusion Logic](#fusion-logic)
8. [Old vs New Model Comparison](#old-vs-new-model-comparison)

---

## Pipeline Overview

Lite-Vision processes each frame through a four-stage inference pipeline:

```
Input Frame
    |
    v
[1] SCRFD 10G KPS  ──>  face bounding boxes + 5 facial landmarks
    |
    v
[2] InsightFace GenderAge  ──>  primary age estimate + gender logits
    |
    v
[3] FER+ Emotion  ──>  8-class expression probabilities
    |
    v
[4] FairFace ResNet34  ──>  bias-corrected gender + age bin distribution
    |
    v
Fused Output (age, gender, confidence, emotion)
```

All models are in ONNX format and are auto-downloaded on first startup. Total
download size is approximately **102 MB**.

---

## Model 1 -- SCRFD 10G KPS (Face Detection)

| Property | Value |
|---|---|
| **File** | `scrfd_10g_kps.onnx` |
| **Size** | ~16.9 MB |
| **Source** | InsightFace / SCRFD paper (Guo et al., 2021) |
| **Architecture** | Single-stage anchor-based detector with keypoint head |
| **Input** | 640 x 640 BGR |
| **Normalization** | `(pixel - 127.5) / 128.0` |
| **Output** | 9 tensors: 3 strides [8, 16, 32] x 3 [scores, bboxes, keypoints] |
| **Keypoints** | 5 facial landmarks: left eye, right eye, nose, left mouth corner, right mouth corner |
| **NMS threshold** | 0.4 (configurable) |
| **Confidence threshold** | 0.5 (configurable) |

### Preprocessing

- **Letterbox resize** -- The input image is resized to 640x640 while preserving
  aspect ratio. Padding is applied to fill the remaining area, preventing
  geometric distortion of facial features.
- **CLAHE** -- Contrast-Limited Adaptive Histogram Equalization is applied to the
  detection input before inference, improving detection accuracy under dark or
  low-contrast lighting conditions.

### Decoding

The model uses an **anchor-based detection** scheme with three feature map strides
(8, 16, 32). Raw outputs are decoded using:

- **Distance-to-bbox** -- Each anchor predicts four distances (left, top, right,
  bottom) from the anchor center to the bounding box edges.
- **Distance-to-kps** -- Each anchor predicts ten values (five x-y pairs) encoding
  the offset from the anchor center to each facial landmark.

### Benchmark

| Metric | SCRFD 10G KPS | YuNet (previous) |
|---|---|---|
| WIDERFace Hard AP | **82.8%** | 70.8% |
| Keypoint output | Yes (5 landmarks) | Yes (5 landmarks) |
| Resize strategy | Letterbox (aspect-preserving) | Direct resize |

---

## Model 2 -- InsightFace GenderAge (Age + Gender)

| Property | Value |
|---|---|
| **File** | `genderage.onnx` |
| **Source** | InsightFace `buffalo_l` model pack |
| **Architecture** | Lightweight CNN for simultaneous gender and age regression |
| **Input** | 96 x 96 RGB |
| **Normalization** | Baked into the ONNX graph (no external preprocessing required) |
| **Output** | Tensor of shape `[1, 3]` |

### Output Interpretation

| Index | Meaning |
|---|---|
| `[0:2]` | Gender logits: index 0 = Male, index 1 = Female |
| `[2]` | Age factor (multiply by 100 to obtain predicted age) |

### Face Alignment

Input faces are aligned using **ArcFace similarity transform**: a 2D similarity
transformation (rotation, scale, translation) is computed from the 5 detected
landmarks to a standard ArcFace face template. This produces a tightly-aligned
96x96 crop that the model was trained on.

### Multi-Crop Ensemble (Gender)

To improve gender prediction robustness, the model runs three inference passes
per face:

| Variant | Description |
|---|---|
| **Aligned** | Standard ArcFace-aligned crop (96 x 96) |
| **15%-padded** | Aligned crop with 15% additional padding on all sides |
| **25%-padded** | Aligned crop with 25% additional padding on all sides |

Gender logits from all three crops are aggregated to produce a more stable
prediction that is less sensitive to expression changes and partial occlusions.

### Age Prediction

Age uses **only the aligned face prediction** (no ensemble). The model was
trained on ArcFace-aligned inputs, and padded crops degrade age regression
accuracy. The age is computed as:

```
predicted_age = age_factor * 100
```

This is a regression-based continuous prediction, not a binned classification.

---

## Model 3 -- FER+ Emotion (Expression Detection)

| Property | Value |
|---|---|
| **File** | `emotion-ferplus-8.onnx` |
| **Source** | ONNX Model Zoo |
| **Architecture** | CNN trained on FER+ dataset with soft labels |
| **Input** | 64 x 64 grayscale |
| **Normalization** | Raw pixel values (normalization baked into ONNX graph) |
| **Output** | 8-class probability distribution |

### Emotion Classes

| Index | Class |
|---|---|
| 0 | Neutral |
| 1 | Happiness |
| 2 | Surprise |
| 3 | Sadness |
| 4 | Anger |
| 5 | Disgust |
| 6 | Fear |
| 7 | Contempt |

### Role in the Pipeline

The FER+ model serves a dual purpose:

1. **Expression label** -- The top predicted emotion class is reported alongside
   age and gender in the API response.
2. **Gender confidence adjustment** -- Expressive emotions (happiness, surprise,
   contempt) are known to shift facial geometry in ways that reduce the
   reliability of gender classification. When one of these emotions is detected,
   gender confidence is reduced by up to **15%**, signaling to downstream
   consumers that the gender prediction is less certain.

---

## Model 4 -- FairFace ResNet34 (Racially-Balanced Gender + Age)

| Property | Value |
|---|---|
| **File** | `fairface.onnx` |
| **Size** | ~85.2 MB |
| **Source** | FairFace project (Karkkainen & Joo, UCLA) |
| **Architecture** | ResNet-34 with three output heads |
| **Gender accuracy** | 95.7% overall, max 5.5% racial disparity |
| **Input** | 224 x 224 RGB |
| **Normalization** | ImageNet standard: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] |

### Output Heads

| Head | Shape | Description |
|---|---|---|
| `age_output` | `(1, 9)` | 9-bin age distribution |
| `gender_output` | `(1, 2)` | Gender logits: index 0 = Male, index 1 = Female |
| `race_output` | `(1, 7)` | 7-class race distribution (used internally, not exposed in API) |

### Age Bins

| Index | Age Range | Midpoint |
|---|---|---|
| 0 | 0--2 | 1 |
| 1 | 3--9 | 6 |
| 2 | 10--19 | 14.5 |
| 3 | 20--29 | 24.5 |
| 4 | 30--39 | 34.5 |
| 5 | 40--49 | 44.5 |
| 6 | 50--59 | 54.5 |
| 7 | 60--69 | 64.5 |
| 8 | 70+ | 75 |

The predicted age is computed as a **weighted midpoint average** across all bins,
producing a continuous estimate from the discrete distribution.

---

## Model Size Summary

| Model | File | Size |
|---|---|---|
| SCRFD 10G KPS | `scrfd_10g_kps.onnx` | ~16.9 MB |
| InsightFace GenderAge | `genderage.onnx` | < 1 MB |
| FER+ Emotion | `emotion-ferplus-8.onnx` | < 1 MB |
| FairFace ResNet34 | `fairface.onnx` | ~85.2 MB |
| **Total** | | **~102 MB** |

All models are auto-downloaded on first startup and cached locally.

---

## Fusion Logic

### Gender Fusion (InsightFace + FairFace)

The system combines gender predictions from both models using a trust-based
fusion rule:

| Condition | Fusion Strategy |
|---|---|
| InsightFace and FairFace **agree** | Use the prediction with **higher confidence** |
| InsightFace and FairFace **disagree** | **Trust FairFace** (trained on racially-balanced data) |

This approach leverages InsightFace's speed and general accuracy while falling
back to FairFace's superior cross-demographic calibration when predictions
conflict.

### Age Fusion (Mask-Aware)

The system detects masked faces by comparing FairFace predictions on the full
face versus the upper half only:

| Condition | Interpretation | Fusion Formula |
|---|---|---|
| `upper_age - full_age > 15` | Face is **masked** | `0.15 * InsightFace_age + 0.85 * upper_age` |
| `upper_age - full_age <= 15` | Face is **unmasked** | `0.50 * InsightFace_age + 0.50 * FairFace_age` |

When a mask is detected, the full-face FairFace prediction is discarded because
the lower face is occluded. The system shifts weight heavily toward the
upper-face crop, which sees the forehead, eyes, and eyebrows -- regions that
remain visible and carry age-discriminative features. InsightFace retains a small
weight (0.15) as a regularizer.

### Expression-Aware Gender Confidence

When FER+ detects an expressive emotion (happiness, surprise, or contempt),
gender confidence is reduced by up to **15%**. This reflects the empirical
observation that strong facial expressions distort the features that gender
classification models rely on, producing less reliable predictions.

---

## Old vs New Model Comparison

### Face Detection

| Attribute | YuNet (v1--v3) | SCRFD 10G KPS (v4) |
|---|---|---|
| **Model file** | `face_detection_yunet.onnx` | `scrfd_10g_kps.onnx` |
| **Model size** | ~337 KB | ~16.9 MB |
| **WIDERFace Hard AP** | 70.8% | **82.8% (+12.0 pp)** |
| **Resize strategy** | Direct resize (distorts aspect ratio) | Letterbox (preserves aspect ratio) |
| **Preprocessing** | None | CLAHE (low-light robustness) |
| **Detection decoding** | OpenCV built-in API | Custom anchor-based (distance-to-bbox, distance-to-kps) |
| **Keypoints** | 5 landmarks | 5 landmarks |
| **Integration** | `cv2.FaceDetectorYN` | Direct ONNX inference via `cv2.dnn.readNetFromONNX` |

### Age + Gender Classification

| Attribute | Single GenderAge (v1--v3) | FairFace Fusion (v4) |
|---|---|---|
| **Models used** | InsightFace `genderage.onnx` only | InsightFace `genderage.onnx` + FairFace `fairface.onnx` + FER+ `emotion-ferplus-8.onnx` |
| **Total model size** | < 1 MB | ~86 MB |
| **Gender method** | Single aligned crop | Multi-crop ensemble (3 variants) + FairFace fusion |
| **Gender bias mitigation** | None | FairFace cross-demographic calibration (max 5.5% racial disparity) |
| **Expression awareness** | None | FER+ confidence adjustment (up to -15% for expressive emotions) |
| **Age method** | `age_factor * 100` (single prediction) | Dual-model fusion with mask-aware weighting |
| **Mask handling** | None | Upper-face crop detection (FairFace upper vs full comparison) |
| **Age representation** | Continuous regression | Continuous regression (InsightFace) + weighted bin average (FairFace), fused |
| **Emotion detection** | Not available | 8-class FER+ classification |

### Summary of Improvements (v3 to v4)

| Metric | Before | After |
|---|---|---|
| Face detection accuracy (WIDERFace Hard) | 70.8% | 82.8% |
| Gender classification | Single model, single crop | Multi-model fusion, multi-crop ensemble |
| Racial bias in gender | Uncontrolled | Max 5.5% disparity (FairFace) |
| Mask robustness | None | Upper-face age estimation |
| Expression robustness | None | FER+ confidence adjustment |
| Emotion output | None | 8-class expression labels |
| Low-light robustness | None | CLAHE preprocessing |
| Total model payload | ~337 KB | ~102 MB |

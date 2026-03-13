# Lite-Vision Inference Pipeline

> Detailed technical reference for the end-to-end inference pipeline used by the
> Lite-Vision age, gender, and emotion detection system.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Flowchart](#flowchart)
3. [Stage 1 -- Image Input](#stage-1----image-input)
4. [Stage 2 -- Face Detection (SCRFD 10G KPS)](#stage-2----face-detection-scrfd-10g-kps)
5. [Stage 3 -- Face Alignment](#stage-3----face-alignment)
6. [Stage 4 -- Multi-Crop Ensemble (InsightFace GenderAge)](#stage-4----multi-crop-ensemble-insightface-genderage)
7. [Stage 5 -- FairFace Gender + Age Fusion](#stage-5----fairface-gender--age-fusion)
8. [Stage 6 -- Emotion Detection (FER+)](#stage-6----emotion-detection-fer)
9. [Stage 7 -- Response Assembly](#stage-7----response-assembly)

---

## Pipeline Overview

The Lite-Vision inference pipeline transforms a raw base64-encoded image into a
structured JSON response containing per-face bounding boxes, age ranges, gender
classifications, and emotion labels. The pipeline is composed of seven sequential
stages, each building on the outputs of its predecessors.

| Property              | Value                                      |
|-----------------------|--------------------------------------------|
| Face detector         | SCRFD 10G KPS (ONNX)                      |
| Age/Gender model      | InsightFace GenderAge + FairFace (fusion)  |
| Emotion model         | FER+ 8-class (ONNX)                       |
| Input resolution      | Arbitrary (capped at 4096px per dimension) |
| Detection resolution  | 640 x 640 letterboxed                      |
| Alignment template    | ArcFace 96 x 96                           |
| Cache strategy        | SHA-256 LRU, 100 entries, 60 s TTL        |

---

## Flowchart

```
 INPUT (base64 string)
  |
  v
+-----------------------------------------------+
| STAGE 1: IMAGE INPUT                          |
|  base64 decode --> magic byte check -->       |
|  SHA-256 hash --> cache lookup -->            |
|  cv2.imdecode --> conditional resize          |
+-----------------------------------------------+
  |                          |
  | (cache miss)             | (cache hit)
  v                          v
+---------------------------+   +--> return cached response
| STAGE 2: FACE DETECTION   |
|  letterbox 640x640        |
|  CLAHE preprocessing      |
|  SCRFD forward pass       |
|  anchor decode + NMS      |
|  --> det[N,5], kpss[N,5,2]|
+---------------------------+
  |
  v
+---------------------------+     +-----------------------------+
| STAGE 3: FACE ALIGNMENT  |     |                             |
|  5-point landmarks        |     |  (for each detected face)   |
|  similarity transform     |---->|                             |
|  warp to 96x96 aligned    |     +-----------------------------+
+---------------------------+                |
  |                                          |
  +------------------+-----------------------+
  |                  |                       |
  v                  v                       v
+---------------+ +----------------+ +----------------+
| STAGE 4       | | STAGE 5        | | STAGE 6        |
| InsightFace   | | FairFace       | | FER+           |
| Multi-Crop    | | Gender + Age   | | Emotion        |
| Ensemble      | | Fusion         | | Detection      |
| (3 variants)  | | (mask-aware)   | | (8 classes)    |
+---------------+ +----------------+ +----------------+
  |                  |                       |
  |   age (aligned)  |  fused age/gender     |  emotion label
  |   gender (avg)   |                       |  expression adj.
  |                  |                       |
  +------------------+-----------------------+
                     |
                     v
          +-------------------------+
          | STAGE 7: RESPONSE       |
          |  normalize bboxes       |
          |  compute age range      |
          |  cache result           |
          |  return JSON            |
          +-------------------------+
                     |
                     v
              JSON RESPONSE
```

---

## Stage 1 -- Image Input

**Purpose.** Accept a raw base64-encoded image, validate it, and decode it into
an OpenCV BGR matrix suitable for downstream processing.

### 1.1 Base64 Decoding

The incoming payload contains a base64-encoded image string. If the standard
data-URI prefix is present (`data:image/...;base64,`), it is stripped before
decoding.

### 1.2 Magic Byte Validation

The first bytes of the decoded buffer are inspected to confirm a supported image
format:

| Format | Magic Bytes           |
|--------|-----------------------|
| JPEG   | `\xff\xd8\xff`       |
| PNG    | `\x89PNG`             |

Payloads that do not match either signature are rejected immediately.

### 1.3 SHA-256 Cache Lookup

A SHA-256 digest of the raw byte buffer is computed and used as a cache key
against an in-memory LRU cache.

| Parameter   | Value       |
|-------------|-------------|
| Max entries | 100         |
| TTL         | 60 seconds  |

On a cache hit the pipeline short-circuits and returns the stored response
directly, avoiding all subsequent stages.

### 1.4 OpenCV Decode

The validated byte buffer is converted to a NumPy array and decoded via
`cv2.imdecode`, producing a BGR matrix.

### 1.5 Conditional Resize

If either spatial dimension exceeds **4096 pixels**, the image is
proportionally downscaled using `cv2.INTER_AREA` interpolation so that the
larger dimension equals 4096. This bounds GPU/CPU memory consumption for
unusually large inputs.

---

## Stage 2 -- Face Detection (SCRFD 10G KPS)

**Purpose.** Locate all faces in the image and extract bounding boxes with
five-point facial landmarks.

### 2.1 Letterbox Resize

The image is resized to **640 x 640** while preserving the original aspect
ratio. The resized image is pasted into the top-left corner of a zero-filled
640 x 640 canvas (letterboxing).

### 2.2 CLAHE Preprocessing

Contrast-Limited Adaptive Histogram Equalization is applied to improve
detection under varying lighting conditions:

1. Convert BGR to LAB colour space.
2. Apply CLAHE to the **L** (lightness) channel:
   - `clipLimit = 2.0`
   - `tileGridSize = (8, 8)`
3. Convert back to BGR.

### 2.3 Blob Creation

The preprocessed image is normalized into a network-compatible blob:

```
blob = (pixel - 127.5) / 128.0
```

### 2.4 Forward Pass

The SCRFD 10G KPS ONNX model produces **9 output tensors**, organized as three
groups of three across three stride levels:

| Stride | Outputs                           |
|--------|-----------------------------------|
| 8      | scores, bboxes, keypoints         |
| 16     | scores, bboxes, keypoints         |
| 32     | scores, bboxes, keypoints         |

### 2.5 Anchor-Based Decoding

Raw network outputs are decoded into image-space coordinates using two internal
functions:

- **`_distance2bbox`** -- converts per-anchor distance predictions
  `(left, top, right, bottom)` into `(x1, y1, x2, y2)` bounding boxes.
- **`_distance2kps`** -- converts per-anchor offset predictions into absolute
  `(x, y)` coordinates for each of the five facial landmarks.

### 2.6 Non-Maximum Suppression

Overlapping detections are suppressed using standard NMS with an IoU threshold
of **0.4**.

### 2.7 Optional Max-Faces Filtering

When a `max_faces` limit is specified, the detector retains only the top-N
faces ranked by a composite score:

```
score = face_area - distance_from_image_center
```

This heuristic favours large, centrally located faces.

### 2.8 Outputs

| Tensor      | Shape     | Description                                        |
|-------------|-----------|----------------------------------------------------|
| `det`       | `[N, 5]`  | `(x1, y1, x2, y2, confidence)` per face           |
| `kpss`      | `[N, 5, 2]` | Five landmark `(x, y)` coordinates per face      |

---

## Stage 3 -- Face Alignment

**Purpose.** Produce a frontalized, alignment-normalized face crop using a
similarity transform to a canonical template.

### 3.1 Landmark Extraction

The five keypoints from Stage 2 are extracted for the current face:

| Index | Landmark         |
|-------|------------------|
| 0     | Left eye         |
| 1     | Right eye        |
| 2     | Nose tip         |
| 3     | Left mouth corner  |
| 4     | Right mouth corner |

### 3.2 Similarity Transform

An affine partial transformation matrix is estimated via
`cv2.estimateAffinePartial2D`, mapping the detected landmarks to the canonical
**ArcFace reference template**.

### 3.3 Template Scaling

The standard ArcFace template is defined at **112 x 112** resolution. For this
pipeline the template is scaled down to **96 x 96** to match the InsightFace
GenderAge model's expected input size.

### 3.4 Warp

The original image is warped using the computed affine matrix, producing a
**96 x 96** aligned face crop.

---

## Stage 4 -- Multi-Crop Ensemble (InsightFace GenderAge)

**Purpose.** Predict age and gender using three complementary crop variants to
improve robustness against partial occlusion and alignment errors.

### 4.1 Crop Variants

| Variant | Source                         | Size   | Usage             |
|---------|--------------------------------|--------|-------------------|
| 1       | Landmark-aligned face (Stage 3) | 96x96  | Primary (age)    |
| 2       | 15%-padded raw crop            | 96x96  | Gender ensemble   |
| 3       | 25%-padded raw crop            | 96x96  | Gender ensemble   |

The padded raw crops extend each side of the detection bounding box by the
stated percentage before resizing to 96 x 96.

### 4.2 Preprocessing

Each variant is converted to a blob with `swapRB=True` (BGR to RGB conversion).
Raw pixel values are used without further normalization.

### 4.3 Forward Pass

Each variant is independently passed through the InsightFace GenderAge ONNX
model, yielding:

| Output            | Shape  | Description                              |
|-------------------|--------|------------------------------------------|
| `gender_logits`   | `[2]`  | Male / female logits                     |
| `age_factor`      | `[1]`  | Normalized age factor in the range [0, 1] |

### 4.4 Age Estimation

Age is derived **exclusively from Variant 1** (the landmark-aligned face),
because the InsightFace GenderAge model was trained on ArcFace-aligned inputs:

```
age = round(age_factor * 100)
```

### 4.5 Gender Estimation

Gender is computed by averaging the softmax probabilities across **all three
variants**:

```
gender_prob = mean(softmax(variant_1), softmax(variant_2), softmax(variant_3))
```

This ensemble averaging reduces sensitivity to crop framing.

---

## Stage 5 -- FairFace Gender + Age Fusion

**Purpose.** Fuse predictions from the racially balanced FairFace model with
InsightFace outputs, and apply mask-aware age correction using upper-face
analysis.

### 5.1 Preprocessing

A **15%-padded** crop of the detected face is resized to **224 x 224** and
normalized per ImageNet conventions:

```
pixel = pixel / 255.0
R = (R - 0.485) / 0.229
G = (G - 0.456) / 0.224
B = (B - 0.406) / 0.225
```

### 5.2 Forward Pass

The FairFace ONNX model produces three output heads:

| Output          | Shape  | Description                     |
|-----------------|--------|---------------------------------|
| `age_output`    | `[9]`  | 9-class age bracket logits      |
| `gender_output` | `[2]`  | Male / female logits            |
| `race_output`   | `[7]`  | 7-class race logits (unused)    |

### 5.3 Gender Fusion

The gender predictions from InsightFace (Stage 4) and FairFace are reconciled
as follows:

| Condition                          | Action                          |
|------------------------------------|---------------------------------|
| Both models agree                  | Use the higher confidence score |
| Models disagree                    | Trust FairFace (racially balanced training set) |

### 5.4 Upper-Face Analysis

An additional inference pass is performed on the **upper half** of the face
crop (the eye region). This crop is resized to 224 x 224 and run through
FairFace to obtain `ff_upper_age`.

### 5.5 Mask-Aware Age Fusion

The upper-face age is compared to the full-face age to detect lower-face
occlusion (e.g., masks):

| Condition                              | Fusion Formula                                |
|----------------------------------------|-----------------------------------------------|
| `ff_upper_age - ff_full_age > 15` (masked) | `age = 0.15 * InsightFace + 0.85 * ff_upper_age` |
| Otherwise (unmasked)                   | `age = 0.50 * InsightFace + 0.50 * FairFace`     |

**Rationale.** When a face mask is present, the full-face FairFace prediction
is unreliable because the lower face is occluded. In this case the pipeline
shifts weight toward the upper-face prediction (85%) and retains a small
contribution from InsightFace (15%), which is more robust to occlusion due to
its alignment-based preprocessing.

---

## Stage 6 -- Emotion Detection (FER+)

**Purpose.** Classify the dominant facial expression into one of eight emotion
categories and optionally adjust gender confidence based on expression
intensity.

### 6.1 Preprocessing

1. Extract a **20%-padded** crop around the detected face.
2. Convert to **grayscale**.
3. Resize to **64 x 64**.
4. Create a raw pixel blob (no explicit normalization; normalization is baked
   into the ONNX graph).

### 6.2 Forward Pass

The FER+ model produces an **8-class softmax** distribution:

| Index | Emotion    |
|-------|------------|
| 0     | Neutral    |
| 1     | Happiness  |
| 2     | Surprise   |
| 3     | Sadness    |
| 4     | Anger      |
| 5     | Disgust    |
| 6     | Fear       |
| 7     | Contempt   |

### 6.3 Expression-Aware Gender Adjustment

Strong facial expressions can subtly shift perceived gender features (e.g., a
wide smile changes jaw geometry). To account for this, a post-hoc correction is
applied:

**Trigger conditions (both must be true):**

- Detected emotion is **expressive**: happiness, surprise, or contempt.
- Emotion confidence exceeds **0.5**.

**Adjustment:**

```
gender_confidence *= (1.0 - emotion_confidence * 0.15)
```

This dampens gender confidence by up to 15% when a strong expression is
present, reflecting the reduced reliability of gender cues under expressive
deformation.

---

## Stage 7 -- Response Assembly

**Purpose.** Package all per-face predictions into a normalized JSON response,
cache it, and return it to the caller.

### 7.1 Bounding Box Normalization

Raw pixel-space bounding boxes `(x1, y1, x2, y2)` are normalized to the range
**[0.0, 1.0]** relative to the original image dimensions. This makes
coordinates resolution-independent and simplifies frontend rendering.

### 7.2 Age Range

The point estimate from the age fusion stage is expanded into a symmetric
range:

```
age_range = [age - 3, age + 3]
```

### 7.3 Cache Storage

The complete response is stored in the SHA-256-keyed LRU cache (see
[Stage 1.3](#13-sha-256-cache-lookup)) so that identical images submitted within
the TTL window are served instantly.

### 7.4 JSON Response

The final response is returned as a JSON payload containing an array of
per-face objects, each including:

- Normalized bounding box coordinates
- Age range
- Gender classification with confidence
- Dominant emotion with confidence

---

*Document generated for the Lite-Vision inference pipeline.*

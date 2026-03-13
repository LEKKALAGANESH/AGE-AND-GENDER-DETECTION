# Model Architecture Improvements — Lite-Vision

> Audit of the current age & gender detection pipeline with upgrade recommendations.

---

## 1. Current Model Stack Analysis

### Face Detection — SSD ResNet-10 (2017)

| Attribute     | Value                                        |
| ------------- | -------------------------------------------- |
| Architecture  | ResNet-10 + Single Shot Detector             |
| Input size    | 300x300 (fixed)                              |
| Format        | Caffe (.caffemodel + .prototxt)              |
| Model size    | 10.2 MB                                      |
| Inference     | ~20-30ms CPU                                 |
| Training data | Unknown (likely WIDER Face)                  |
| WIDER Face AP | ~87% Easy / ~83% Med / ~65% Hard (estimated) |

**Limitations:**

- Fixed 300x300 input — misses small faces and loses detail on high-res images
- Struggles with profile/occluded faces (65% AP Hard)
- 2017-era architecture, significantly behind current SOTA
- Caffe format is deprecated — no ecosystem support

### Age Prediction — CaffeNet/AlexNet (2015)

| Attribute      | Value                               |
| -------------- | ----------------------------------- |
| Architecture   | AlexNet variant (Conv1-3 + FC6-FC8) |
| Input size     | 227x227                             |
| Format         | Caffe                               |
| Model size     | 43.5 MB                             |
| Output         | 8-class softmax (age buckets)       |
| Training data  | Likely Adience/IMDB-WIKI subset     |
| Real-world MAE | ~8-12 years (estimated)             |

**Age Bucket Mapping (Current):**

| Bucket | Range  | Midpoint Returned |
| ------ | ------ | ----------------- |
| 0      | 0-2    | 1                 |
| 1      | 4-6    | 5                 |
| 2      | 8-12   | 10                |
| 3      | 15-20  | 17                |
| 4      | 25-32  | 28                |
| 5      | 38-43  | 40                |
| 6      | 48-53  | 50                |
| 7      | 60-100 | 80                |

**Critical Problems:**

- Only 8 discrete buckets — a 26-year-old and a 31-year-old both return "28"
- Gaps between buckets (e.g., ages 3, 7, 13-14, 21-24, 33-37, 44-47, 54-59 have NO bucket)
- Final bucket (60-100) spans 40 years with midpoint 80 — a 62-year-old gets labeled 80
- Uses argmax only — discards all softmax probability information
- No uncertainty/confidence for age prediction
- Model trained on ~2015 data, likely biased toward Western/Caucasian faces

### Gender Prediction — CaffeNet/AlexNet (2015)

| Attribute    | Value                                    |
| ------------ | ---------------------------------------- |
| Architecture | AlexNet variant (identical to age model) |
| Input size   | 227x227                                  |
| Format       | Caffe                                    |
| Model size   | 43.6 MB                                  |
| Output       | 2-class softmax                          |
| Accuracy     | ~86% (Adience benchmark)                 |

**Problems:**

- Binary only (Male/Female)
- 86% accuracy is well below modern standards (~97%)
- Heavily influenced by hair/presentation rather than facial structure
- 43.6 MB for a binary classifier is extremely oversized

### Pipeline Summary

| Metric                   | Current Value      | Problem                          |
| ------------------------ | ------------------ | -------------------------------- |
| Total model size         | **~97 MB**         | Slow cold starts on Vercel       |
| Age MAE                  | ~8-12 years        | Unacceptable for production      |
| Gender accuracy          | ~86%               | Well below modern ~97%           |
| Face detection AP (Hard) | ~65%               | Misses many real-world faces     |
| Age output               | 8 discrete values  | Not continuous, has gaps         |
| Model format             | Caffe (deprecated) | No tooling, no community support |
| Training data era        | 2015-2017          | Outdated, likely biased          |

---

## 2. Modern Alternatives Comparison

### Face Detection Models

| Model                   | AP Easy   | AP Med    | AP Hard   | Size       | Speed (CPU) | Format |
| ----------------------- | --------- | --------- | --------- | ---------- | ----------- | ------ |
| SSD ResNet-10 (current) | ~87%      | ~83%      | ~65%      | 10.2 MB    | ~20ms       | Caffe  |
| **YuNet 2023**          | 88.4%     | 86.6%     | 75.0%     | **337 KB** | **1-5ms**   | ONNX   |
| SCRFD-500M              | 90.6%     | 88.1%     | 68.5%     | 2.4 MB     | ~4ms        | ONNX   |
| **SCRFD-2.5G**          | **93.8%** | **92.0%** | **77.1%** | 3.1 MB     | ~8ms        | ONNX   |
| SCRFD-10G               | 95.4%     | 94.0%     | 82.8%     | 16.1 MB    | ~15ms       | ONNX   |
| RetinaFace-R50          | 94.2%     | 93.2%     | 83.6%     | 100+ MB    | ~50ms       | ONNX   |

### Age & Gender Models

| Model                     | Age MAE       | Gender Acc. | Size             | Format        | Continuous Age? |
| ------------------------- | ------------- | ----------- | ---------------- | ------------- | --------------- |
| CaffeNet (current)        | ~8-12 yrs     | ~86%        | 87 MB (2 models) | Caffe         | No (8 bins)     |
| **InsightFace genderage** | **~4.65 yrs** | **~97%**    | **~1 MB**        | **ONNX**      | **Yes**         |
| onnx-community ViT        | ~4.5 yrs      | ~94.3%      | ~50 MB           | ONNX          | Yes             |
| DeepFace (VGG-Face)       | ~4.65 yrs     | ~97.4%      | ~500 MB          | PyTorch       | Yes             |
| MiVOLO v2 (SOTA)          | ~4.0 yrs      | SOTA        | ~200 MB          | PyTorch       | Yes             |
| DEX (20-ensemble)         | ~2.68 yrs     | N/A         | ~3 GB            | Caffe/PyTorch | Yes             |
| FairFace                  | Binned        | ~97%+       | ~100 MB          | PyTorch       | No (9 bins)     |

### Age Estimation Approaches

| Approach                       | Description                                       | Pros                                       | Cons                                                              |
| ------------------------------ | ------------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------- |
| **Classification** (current)   | Softmax over N bins, argmax                       | Simple to train                            | Coarse, treats errors equally, loses ordering info                |
| **Direct Regression**          | Single continuous output                          | Simple architecture                        | Regression-to-mean bias, overestimates young / underestimates old |
| **Deep Expectation (DEX)**     | 101-class softmax, compute E[age] = sum(p_i \* i) | Smooth continuous output, proven SOTA      | Large model needed for best results                               |
| **Ordinal Regression (CORAL)** | K binary classifiers: "is age > k?"               | Rank-consistent, no inversions, elegant    | Slightly more complex training                                    |
| **Distribution Learning**      | Predict mean + variance                           | Captures uncertainty, probabilistic output | Specialized loss functions needed                                 |

---

## 3. Recommended Upgrade Path

### Option A: Drop-in ONNX Replacement (Recommended — Fastest Impact)

Replace all 3 Caffe models with 2 ONNX models:

| Component      | Current                        | Replacement                                  |
| -------------- | ------------------------------ | -------------------------------------------- |
| Face Detection | SSD ResNet-10 (10.2 MB, Caffe) | **YuNet 2023** (337 KB, ONNX)                |
| Age + Gender   | 2x AlexNet (87 MB, Caffe)      | **InsightFace genderage.onnx** (~1 MB, ONNX) |

**Impact:**

| Metric                | Before                | After                    | Improvement               |
| --------------------- | --------------------- | ------------------------ | ------------------------- |
| Total model size      | 97 MB                 | **~1.3 MB**              | **75x smaller**           |
| Age MAE               | ~8-12 years           | ~4.65 years              | **2-3x more accurate**    |
| Gender accuracy       | ~86%                  | ~97%                     | **+11 percentage points** |
| Face detection (Hard) | ~65%                  | ~75%                     | **+10 percentage points** |
| Age output            | 8 discrete values     | Continuous integer       | **Smooth predictions**    |
| Cold start time       | Slow (97 MB download) | Near-instant             | **Massively faster**      |
| Format                | Caffe (deprecated)    | ONNX (industry standard) | **Future-proof**          |

**Download Sources:**

- YuNet: `https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`
- InsightFace genderage: Extract `genderage.onnx` from `https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip`

**Integration:**

- YuNet works with `cv2.FaceDetectorYN` (built into OpenCV 4.5.4+) — no extra deps
- genderage.onnx works with `cv2.dnn.readNetFromONNX()` or `onnxruntime` — no extra deps
- Both models use standard ONNX format, compatible with existing OpenCV DNN pipeline

### Option B: Maximum Accuracy (Requires GPU or Larger Server)

| Component      | Replacement                  |
| -------------- | ---------------------------- |
| Face Detection | SCRFD-10G (16.1 MB, ONNX)    |
| Age + Gender   | MiVOLO v2 (PyTorch, ~200 MB) |

**Pros:** SOTA accuracy (MAE ~4.0, 95%+ detection), uses body context for age estimation
**Cons:** Requires PyTorch, ~200 MB model, needs GPU for real-time, won't fit Vercel serverless

### Option C: Balanced (Best Accuracy within Serverless Constraints)

| Component      | Replacement                                             |
| -------------- | ------------------------------------------------------- |
| Face Detection | SCRFD-2.5G (3.1 MB, ONNX) + 5-point landmarks           |
| Age + Gender   | onnx-community/age-gender-prediction-ONNX (ViT, ~50 MB) |

**Pros:** Better accuracy than Option A (4.5 MAE, 93.8% face AP), includes face alignment via landmarks
**Cons:** 50 MB ViT model may be tight for Vercel's size limit

---

## 4. Architecture Improvements Beyond Model Swap

### 4.1 Face Alignment (High Impact)

Current pipeline feeds raw face crops to age/gender models. Adding alignment significantly improves accuracy:

```
Current:  Detect face box → Crop → Resize 227x227 → Predict
Improved: Detect face box + landmarks → Align (affine warp) → Crop → Resize → Predict
```

- SCRFD and InsightFace models output 5-point landmarks (eyes, nose, mouth corners)
- Affine transformation to canonical face position normalizes pose variation
- Expected improvement: 1-2 year MAE reduction for age, 2-3% for gender

### 4.2 Deep Expectation Instead of Argmax (High Impact, No New Model Needed)

Even with current models, replace argmax with expected value computation:

```python
# Current (lossy):
age_idx = int(age_preds[0].argmax())
age = AGE_MIDPOINTS[age_idx]  # returns 28

# Improved (smooth):
probs = age_preds[0]
age = sum(p * midpoint for p, midpoint in zip(probs, AGE_MIDPOINTS))  # returns 26.4
age = int(round(age))
```

This uses ALL softmax probabilities instead of just the max, producing smoother predictions at zero cost. Can be applied to existing Caffe models immediately.

### 4.3 Preprocessing Enhancements

| Enhancement                             | Description                               | Impact                                                |
| --------------------------------------- | ----------------------------------------- | ----------------------------------------------------- |
| Adaptive histogram equalization (CLAHE) | Normalize lighting before inference       | Better accuracy in low-light/overexposed conditions   |
| Face padding                            | Expand crop by 20-30% around detected box | Includes forehead/chin context, improves age accuracy |
| Multi-scale detection                   | Run face detector at multiple scales      | Better small face detection                           |
| Bilateral filtering                     | Reduce noise while preserving edges       | Cleaner input for age model                           |

### 4.4 Confidence & Uncertainty

Current system returns only point estimates. Improvements:

| Feature                 | Implementation                                                                     |
| ----------------------- | ---------------------------------------------------------------------------------- |
| Age confidence interval | Return `age_min`, `age_max` from softmax distribution (e.g., 10th/90th percentile) |
| Gender confidence       | Already available in softmax — return the probability, not just the class          |
| Detection quality score | Face size + sharpness + angle as a quality metric                                  |
| "Uncertain" flag        | When softmax entropy is high, flag the prediction as low-confidence                |

### 4.5 Temporal Smoothing (Live Detection Mode)

For webcam streaming, smooth predictions across frames:

```python
# Exponential moving average across frames for the same tracked face
smoothed_age = alpha * current_age + (1 - alpha) * previous_age  # alpha = 0.3
```

- Reduces jitter (age jumping between 28 and 40 frame-to-frame)
- Requires simple face tracking (IoU-based matching between frames)
- Convergence to stable prediction within 5-10 frames

### 4.6 Batch Inference

Current pipeline processes each face sequentially:

```
Current:  face1 → gender → age → face2 → gender → age → ...
Improved: [face1, face2, face3] → gender_batch → age_batch (single forward pass)
```

- OpenCV DNN and ONNX Runtime both support batched input
- 2-3x speedup for multi-face images

---

## 5. Bias & Fairness Considerations

### Known Issues with Current Models

- Trained primarily on Western/Caucasian faces (Adience dataset)
- Higher error rates on darker skin tones, East Asian faces, elderly faces
- Gender model correlates with hair length/facial hair rather than facial structure
- Age model has large errors on children (0-2 bucket) and elderly (60-100 bucket)

### Mitigation Strategies

| Strategy                       | Description                                                                 |
| ------------------------------ | --------------------------------------------------------------------------- |
| **FairFace-balanced training** | Use FairFace dataset (108K images, 7 race groups, balanced) for fine-tuning |
| **Per-demographic evaluation** | Measure MAE separately per ethnicity/gender to detect bias                  |
| **Calibration**                | Apply post-hoc calibration curves per demographic group                     |
| **Uncertainty flagging**       | When model is unsure, return a range instead of a point estimate            |

---

## 6. Priority Roadmap

### Phase 1 — Quick Wins (No Model Change)

| Improvement                                                    | Effort          | Impact                        |
| -------------------------------------------------------------- | --------------- | ----------------------------- |
| Replace argmax with deep expectation (weighted sum of softmax) | 5 lines of code | ~1-2 year MAE reduction       |
| Add age confidence interval from softmax distribution          | 10 lines        | Better UX, honest uncertainty |
| Add face crop padding (20% expansion)                          | 5 lines         | Better context for age model  |
| Return gender probability instead of just label                | 2 lines         | More nuanced output           |

### Phase 2 — Drop-in Model Swap (Option A)

| Improvement                                              | Effort           | Impact                            |
| -------------------------------------------------------- | ---------------- | --------------------------------- |
| Replace SSD ResNet-10 with YuNet                         | Medium (new API) | 75x smaller, 10% better detection |
| Replace Caffe age+gender with InsightFace genderage.onnx | Medium           | 2-3x better accuracy, 87x smaller |
| Add face alignment via landmarks                         | Medium           | 1-2 year MAE improvement          |
| Total model size: 97 MB → 1.3 MB                         | —                | Instant cold starts               |

### Phase 3 — Advanced Features

| Improvement                           | Effort | Impact                                 |
| ------------------------------------- | ------ | -------------------------------------- |
| Temporal smoothing for live detection | Medium | Stable, non-jittery predictions        |
| Batch inference for multi-face images | Medium | 2-3x speedup                           |
| CLAHE preprocessing                   | Low    | Better low-light accuracy              |
| Per-demographic calibration           | High   | Fairer predictions across demographics |

### Phase 4 — SOTA (If GPU Available)

| Improvement                              | Effort | Impact                           |
| ---------------------------------------- | ------ | -------------------------------- |
| MiVOLO v2 integration                    | High   | SOTA accuracy (~4.0 MAE)         |
| Body context for age estimation          | High   | Works even when face is occluded |
| Custom fine-tuning on target demographic | High   | Domain-specific accuracy gains   |

---

## 7. Model Sources & References

### Pre-trained Models (Ready to Download)

| Model                                           | Link                                                                                                         |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| YuNet 2023 (face detection, 337 KB)             | [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)                     |
| InsightFace buffalo_l (includes genderage.onnx) | [insightface releases v0.7](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) |
| SCRFD models (multiple sizes)                   | [insightface SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)                  |
| onnx-community age-gender ViT                   | [Hugging Face](https://huggingface.co/onnx-community/age-gender-prediction-ONNX)                             |
| MiVOLO v2                                       | [Hugging Face](https://huggingface.co/iitolstykh/mivolo_v2)                                                  |

### Research Papers

| Paper                                                  | Year | Key Contribution                                                          |
| ------------------------------------------------------ | ---- | ------------------------------------------------------------------------- |
| Gil Levi & Tal Hassner — Age and Gender Classification | 2015 | Original models currently used                                            |
| DEX: Deep EXpectation (Rothe et al.)                   | 2015 | Expected value from softmax, 2.68 MAE                                     |
| CORAL-CNN (Cao, Mirjalili & Raschka)                   | 2020 | Rank-consistent ordinal regression                                        |
| FairFace (Karkkainen & Joo)                            | 2021 | Bias-aware training, <1% demographic gap                                  |
| MiVOLO (Kuprashevich & Tolstykh)                       | 2023 | Multi-input transformer, face+body, ~4.0 MAE                              |
| JAM — Just Age Model                                   | 2024 | Distribution-based with confidence, outperforms MiVOLO on real-world data |
| YuNet (Wu et al.)                                      | 2023 | 337 KB face detector, built into OpenCV                                   |
| SCRFD (Guo et al.)                                     | 2022 | Computation redistribution, best accuracy/size tradeoff                   |

### Libraries

| Library                                                                  | Use Case                                                   |
| ------------------------------------------------------------------------ | ---------------------------------------------------------- |
| [insightface](https://github.com/deepinsight/insightface)                | Face detection + recognition + age/gender                  |
| [deepface](https://github.com/serengil/deepface)                         | All-in-one face analysis (age, gender, emotion, ethnicity) |
| [coral-pytorch](https://github.com/Raschka-research-group/coral-pytorch) | Ordinal regression for any PyTorch backbone                |

---

_Generated by a senior model architecture review._

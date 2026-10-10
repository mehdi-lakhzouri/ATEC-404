# Results

> **Provisional.** The labels are automated (S2-H) and not human-verified, and the data are synthetic renders only. Every number
> below is read from `artifacts/` (hash-verified copies of the experiment outputs).

## Frozen model

| Item | Value |
|---|---|
| Weights | `models/artemis_rock_yolo11n.pt`, SHA-256 `413d0481a728a310f5b919d069676ee5ddd2091e81b4b663831e1e977a0752eb` |
| Source | E.1c `best.pt`, epoch 111 |
| Architecture | YOLO11n, 2,590,035 parameters, class `{0: rock}` |
| Inference | imgsz 736 · conf **0.20** · NMS IoU 0.60 · max_det 300 · boxes < 50 px² removed |

## Training (E.1c)

**Setup:** Config A (1,903 training images); 200 epochs; batch 16; imgsz 736; SGD (lr0 0.01, momentum 0.937, weight decay 5e-4,
cosine); 3 warm-up epochs; AMP; seed 42; close_mosaic 10. Hardware: RTX 5060 Laptop GPU.

**Run:**
- 68.7 min wall-clock;
- median 125 images/s;
- training-window GPU utilisation median 87%;
- peak VRAM 5.5 GB;
- max temperature 71 °C;
- no OOM, NaN or thermal throttling.

**Convergence:**
- Validation losses reached their minimum between epochs 104 and 138.
- Ignore-aware AP50 plateaued at 0.75–0.775 from about epoch 70, with a slight decline after epoch 150. This is mild overfitting.

## Model selection (validation, 923 images, primary policy)

| Checkpoint | AP50 | AP50-95 | P@0.20 | R@0.20 | F1 | FP/img | Large recall |
|---|---|---|---|---|---|---|---|
| **best.pt (epoch 111)** | **0.776** | 0.432 | 0.696 | 0.775 | 0.734 | 9.47 | 0.807 |
| epoch 120 | 0.775 | 0.433 | 0.715 | 0.768 | 0.741 | 8.54 | 0.824 |
| epoch 130 | 0.769 | 0.429 | 0.652 | 0.792 | 0.715 | 11.83 | 0.818 |
| epoch 140 | 0.775 | 0.433 | 0.719 | 0.766 | 0.742 | 8.39 | 0.812 |

**Threshold grid for `best.pt`:**

| conf | 0.05 | 0.10 | 0.15 | **0.20** | 0.25 |
|---|---|---|---|---|---|
| Precision | 0.416 | 0.542 | 0.629 | **0.696** | 0.747 |
| Recall | 0.869 | 0.837 | 0.807 | **0.775** | 0.740 |
| F1 | 0.563 | 0.658 | 0.707 | **0.734** | 0.743 |
| FP/img | 34.1 | 19.8 | 13.3 | **9.5** | 7.0 |

## Once-only test evaluation (917 images, 25,987 VALID boxes)

| Policy | Precision | Recall | F1 | FP/img | AP50 | AP50-95 |
|---|---|---|---|---|---|---|
| **Primary (REVIEW ignored)** | **0.698** | **0.774** | **0.734** | **9.48** | **0.777** | **0.428** |
| Secondary (+ proposed sub-threshold) | 0.779 | 0.774 | 0.777 | 6.21 | 0.803 | 0.440 |
| No ignores | 0.534 | 0.774 | 0.632 | 19.17 | 0.664 | 0.370 |

**Recall by size (primary, conf 0.20):**

| Size bin | n (test) | Test | Val |
|---|---|---|---|
| small < 400 px² | 17,110 | 0.743 | 0.740 |
| medium 400–2,500 px² | 7,544 | 0.842 | 0.849 |
| large 2,500–10,000 px² | 1,109 | 0.782 | 0.815 |
| very large ≥ 10,000 px² | 224 | 0.821 | 0.760 |
| **all large ≥ 2,500 px²** | 1,333 | **0.788** | 0.807 |

**Error analysis (test, primary):**
- **False negatives (5,879):** 61% were found below the threshold, 21% were localisation errors (IoU 0.1–0.5), and 18% were
  near-total misses.
- **False positives (8,694):** 465 duplicates, 3,747 localisation errors, and 4,482 on background. Under the secondary policy,
  background false positives drop to 1,617, which suggests many are small unlabelled rocks.
- **Hard scenes:** 194 of the 490 test images that contain a large rock miss at least one.

**Interpretation:**
- Test reproduces validation within ±0.004 on every headline metric, so selection did not overfit the validation split.
- That agreement is **not** evidence of generalisation to real lunar imagery: both splits share one synthetic distribution and
  the same automated labels.
- The gap of about 0.12 AP50 between the primary and no-ignore policies comes from the evaluation policy, not from the model.

**Sources:** `artifacts/evaluation/test_results.json`, `selection_analysis.json`, `ignore_aware_val_22_checkpoints.json`, and
the reports `artifacts/reports/E1C_TRAINING_REPORT.md`, `F_MODEL_SELECTION_REPORT.md` and `F_TEST_EVALUATION_REPORT.md`.

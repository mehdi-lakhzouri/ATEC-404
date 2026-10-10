# Task F: Validation-Based Model Selection and Recall Analysis

**Status: model selection and configuration FROZEN on validation only. The test split has not been read.**
**Labels:** automated S2-H boxes, not human-verified; all figures are PROVISIONAL.
**Dataset:** `yolo-rock-v1.1`, still PROVISIONAL and NOT FROZEN; only the model selection is frozen.

## 1. Frozen selection (`FROZEN_MODEL_CONFIG.json`)

| Item | Frozen value |
|---|---|
| Model file | `model/artemis_rock_yolo11n_e1c_s42.pt`: a byte-identical copy of E.1c `weights/best.pt`; the original is untouched |
| SHA-256 | `413d0481a728a310f5b919d069676ee5ddd2091e81b4b663831e1e977a0752eb` |
| Architecture | YOLO11n, 2,590,035 params, classes `{0: rock}`, epoch 111 of the E.1c Config A seed-42 run |
| Image size | 736 (native 720×480 render, RGB) |
| Confidence threshold | **0.20** |
| NMS | IoU **0.6**, max_det **300**, class-aware |
| Post-filter | drop boxes < **50 px²**, matching the evaluator's `min_pred_area` |
| Evaluation policy | **primary:** ignore REVIEW regions (`eval_ignore_regions.json`, sha in the config), match IoU 0.5, ignore IoA 0.5, 101-point AP, size bins < 400 / 400–2500 / ≥ 2500 px²; the sub-threshold PROPOSED ignores are reported **as secondary only** |

**Validation at the frozen configuration** (923 images, 25,822 VALID ground-truth boxes):

| Metric | Value |
|---|---|
| Precision | 0.696 |
| Recall | 0.775 |
| F1 | 0.734 |
| F2 | 0.758 |
| False positives per image | 9.47 |
| AP50 (threshold-independent) | 0.776 |
| AP50-95 (threshold-independent) | 0.432 |

**Large rocks (≥ 2500 px², n = 1,409):** recall 0.807, precision 0.852.

## 2. Selection rules

These rules were fixed in code (`backend/training/task_f_selection.py → RULES`) **before any threshold metric was computed**. The checkpoint AP values were already known from E.1c.

- **Checkpoint:** keep `best.pt`, the provisional primary you named, unless a comparison candidate beats its primary AP50 by more than 0.005.
  - **Result:** none did. Epoch 120 scored −0.0011, epoch 130 −0.0066, epoch 140 −0.0013. **`best.pt` is kept.**
- **Threshold, recall-oriented:** take the lowest grid threshold whose F1 is within 0.02 of the best F1 on the grid.
  - The best F1 is 0.7433, at 0.25. At 0.20, F1 is 0.7336 (−0.0097), so it qualifies; 0.15 (0.7067) does not.
  - **Result: 0.20.** Against 0.25, this gains +3.5 points of recall (+913 rocks found) for +2.45 false positives per image.
- **NMS and pre-processing:** unchanged from the evaluation that produced the candidates. Not tuned.

## 3. Confidence-threshold analysis (val, primary policy; `selection_analysis.json`, `threshold_analysis.png`)

Each threshold's figures come from the cached conf ≥ 0.001 predictions of E.1c.

**Is filtering cached predictions the same as predicting at the threshold?**
- The match is greedy in score order, so yes, unless `max_det` cut off boxes above the threshold. That happened in 0 images for conf ≥ 0.10, and in 5 of 923 images at 0.05 for `best.pt` and epoch 130. So the 0.05 row is very slightly pessimistic for those two.
- **Live check:** live inference with the frozen config on 50 validation images reproduced the cached result **exactly**: 1,904 of 1,904 boxes, IoU > 0.99.
- At 0.25 the table reproduces the evaluator's TP/FP counts (an assertion in the code).

**`best.pt`** (AP50 0.776, AP50-95 0.432, at every threshold):

| Conf | Precision | Recall | F1 | F2 | FP / image | TP | FN |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.416 | 0.869 | 0.563 | 0.714 | 34.1 | 22,450 | 3,372 |
| 0.10 | 0.542 | 0.837 | 0.658 | 0.755 | 19.8 | 21,620 | 4,202 |
| 0.15 | 0.629 | 0.807 | 0.707 | **0.764** | 13.3 | 20,840 | 4,982 |
| **0.20** | **0.696** | **0.775** | **0.734** | 0.758 | **9.5** | **20,021** | **5,801** |
| 0.25 | 0.747 | 0.740 | **0.743** | 0.741 | 7.0 | 19,108 | 6,714 |

**Comparison candidates at conf 0.20:**

| Checkpoint | AP50 | AP50-95 | Precision | Recall | F1 | FP / image | Large-rock recall | Very-large (≥ 10k px²) recall |
|---|---|---|---|---|---|---|---|---|
| **`best.pt` (epoch 111)** | **0.776** | 0.432 | 0.696 | 0.775 | 0.734 | 9.47 | 0.807 | 0.760 |
| epoch 120 | 0.775 | 0.433 | 0.715 | 0.768 | 0.741 | 8.54 | **0.824** | **0.784** |
| epoch 130 | 0.769 | 0.429 | 0.652 | 0.792 | 0.715 | 11.83 | 0.818 | 0.784 |
| epoch 140 | 0.775 | **0.433** | 0.719 | 0.766 | 0.742 | 8.39 | 0.812 | 0.775 |

**Observations:**
- The four candidates are within the single-seed noise of one another.
- **Epoch 120 is about 1.7 points better on large-rock recall** with slightly fewer false positives. Under the pre-declared rule it doesn't displace `best.pt`.
  - If you value large-obstacle recall over the rule, epoch 120 is the alternative to consider. It needs your decision **before** the test evaluation, and the frozen file would then be regenerated.

## 4. Recall by rock size (`best.pt`)

| Size bin (px²) | GT | Recall @0.05 | @0.10 | @0.15 | **@0.20** | @0.25 | Precision @0.20 |
|---|---|---|---|---|---|---|---|
| small < 400 | 17,001 | 0.851 | 0.816 | 0.779 | **0.740** | 0.698 | 0.641 |
| medium 400–2500 | 7,412 | 0.912 | 0.887 | 0.869 | **0.849** | 0.828 | 0.818 |
| large 2500–10000 | 1,205 | 0.877 | 0.849 | 0.829 | **0.815** | 0.794 | 0.871 |
| very large ≥ 10000 | 204 | 0.824 | 0.784 | 0.765 | **0.760** | 0.745 | 0.767 |
| **all large ≥ 2500** | 1,409 | 0.869 | 0.840 | 0.820 | **0.807** | 0.787 | 0.852 |

**Recall falls for the biggest objects.** Very-large rocks (0.760) trail medium (0.849) and large (0.815). These are the most safety-relevant obstacles, so this is the most important finding.

At 0.20, 181 of the 491 validation images that contain a large rock miss at least one large rock.

## 5. False-negative analysis at conf 0.20 (`best.pt`; `per_image_fn_selected.csv`, `fn_scenes/`)

**5,801 false negatives in total:**

| Category | Count | Share | What it means |
|---|---|---|---|
| Found below threshold | 3,503 | 60% | A prediction with conf < 0.20 matches the rock (IoU ≥ 0.5); recoverable by threshold |
| Localisation error | 1,316 | 23% | Best prediction ≥ 0.20 has IoU 0.1–0.5: wrong extent, or merged with or split from a neighbour |
| Near-total miss | 982 | 17% | No prediction ≥ 0.20 with IoU ≥ 0.1 |

| Size | Below threshold | Localisation | Near miss |
|---|---|---|---|
| small | 2,619 | 977 | 817 |
| medium | 725 | 262 | 129 |
| large 2.5k–10k | 134 | 62 | 27 |
| very large ≥ 10k | 25 | 15 | 9 |

**Visual inspection** of 12 difficult validation scenes (`fn_scenes/*_contact_sheet.jpg`) and 4 representative ones (`qualitative/`). Qualitative and limited to 16 images, so these are hypotheses:

1. **Label artefacts make up a visible share of the "missed large rocks":**
   - oversized or merged boxes covering a boulder plus terrain (render8425, 3024);
   - boxes on shadow regions rather than rock (render2063);
   - boxes on ridges or slope highlights (render2369);
   - thin slivers cut off at the image edge (render2369, 5901).

   These inflate the large and very-large false-negative counts and are **candidates for human review (E.1b)**, not model faults.
2. **Genuine misses:**
   - **Dark or low-illumination scenes:** render0240 (recall 0.08), 1966 (0.00), 3371 (0.04). Rocks sit in shadow on crater rims.
   - **Large shadowed boulders silhouetted against the black sky:** render6352. This is safety-relevant.
   - **Dense boulder fields:** render5211, 8539. Adjacent boulders get merged, producing localisation errors.
3. **Distant small-rock clusters near the horizon** make up most small-rock misses. These matter less for near-field hazard avoidance.
4. Many predictions land on REVIEW (yellow) regions that visibly contain rocks. This supports the ignore policy over counting them as false positives.

## 6. Qualitative and compatibility checks (`inference_check.json`)

- The frozen model loads with names `{0: 'rock'}`.
- Live inference parity: 50 of 50 images with exact box counts; 1,904 of 1,904 boxes match the cached analysis.
- **Latency: median 15.3 ms per image** (p95 17.2 ms), one 720×480 image on the RTX 5060, including file I/O, at batch 1. This is not a Webots measurement.
- Overlays for validation scenes 0115, 5589, 0575 and 6352 are in `qualitative/val_scene_*.jpg`.

## 7. Limitations

- **Labels:** automated and not human-validated; selection metrics inherit their errors (§5).
- **Seeds:** one seed, so checkpoint differences of about ±0.01 AP50 can't be told apart from noise.
- **Domain:** synthetic renders only; no real lunar imagery (NASA POLAR) has been evaluated.
- **Threshold grid:** 0.05–0.25 as instructed. 0.25 was the F1 maximum on the grid, and higher thresholds weren't explored.
- **What "frozen" covers:** the model, threshold, NMS and policy for the test evaluation. It doesn't freeze the dataset.

## 8. Ready for authorisation

The selection is complete and frozen; see `FROZEN_MODEL_CONFIG.json` and `CHECKSUMS.sha256`. Pending your authorisation:

1. **The once-only test evaluation** (917 images) with exactly the frozen model, inference settings and primary policy, reporting the secondary and naive scores separately.
2. **Webots integration** of `artemis_rock_yolo11n_e1c_s42.pt` with the frozen inference settings.

**Decision before the test run:** keep `best.pt` (rule outcome), or switch to epoch 120 for about +1.7 points of large-rock recall at the same AP. It must be settled before the test evaluation.

**Not done, as instructed:** no test-split access, no training, no dataset changes, no checkpoint deletion.

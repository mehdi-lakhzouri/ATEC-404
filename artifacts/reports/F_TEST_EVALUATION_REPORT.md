# Task F: Once-Only Test Evaluation of the Frozen Rock Detector

> **The labels are automated (S2-H) and NOT human-verified. All metrics are PROVISIONAL.**
> The test split (917 synthetic renders) was evaluated **once**, on 2026-10-09 between 23:14:06 and 23:15:17, using the frozen configuration exactly. Nothing was tuned on these results. They don't establish performance on real lunar imagery, or readiness for rover deployment.

## 1. What was evaluated

| Item | Value |
|---|---|
| Frozen configuration | `experiments/task-f/model-selection-001/FROZEN_MODEL_CONFIG.json`, sha256 `efdc5802…44bf4` |
| Model | `artemis_rock_yolo11n_e1c_s42.pt`: YOLO11n, E.1c `best.pt` (epoch 111), sha256 `413d0481a728a310f5b919d069676ee5ddd2091e81b4b663831e1e977a0752eb`, classes `{0: rock}` |
| Inference | imgsz 736; **conf 0.20**; NMS IoU 0.6; max_det 300; boxes < 50 px² dropped |
| Protocol | identical to validation: predict once at conf 0.001 (for AP); the operating-point metrics use the frozen conf 0.20. `max_det` didn't truncate any box ≥ 0.20 (0 of 917 images affected) |
| Matching | IoU 0.5, greedy by score; ignore = IoU or IoA ≥ 0.5 with an ignore region; 101-point AP; AP50-95 over IoU 0.50:0.95 |
| Test set | 917 images, 25,987 VALID ground-truth boxes |

**Verification before the run** (all passed; otherwise the script refuses to run):
- The frozen config's SHA-256 matches `CHECKSUMS.sha256`.
- The config says `frozen: true` and `test_split_used: false`.
- The model SHA-256 matches both the config and the checksums.
- The primary ignore file and the split manifest match the config.
- Test images and labels number 917 each.
- No earlier test run exists.

**Once-only enforcement:**
- `TEST_RUN_LOCK.json` now blocks a second run.
- The scoring pipeline was first dry-run on the validation predictions. It reproduced the frozen validation numbers exactly, so a code bug couldn't waste the single test run.

**Integrity after the run:**
- The protected trees are unchanged: source datasets, Task C/D v1.1, archived v1 and the earlier experiments.
- The model-selection and E.1c checksums verify.
- Predictions were written only to `predictions/test/labels/` (917 files).

## 2. Headline results at the frozen operating point (conf 0.20)

**Primary = the approved policy: REVIEW regions ignored.** The secondary policy also ignores the *proposed, unapproved* sub-threshold regions. The reference applies no ignores at all. **Recall is the same under all three policies**: ignore regions affect only how predictions are counted, never the ground truth.

| Policy | Precision | Recall | F1 | FP / image | AP50 | AP50-95 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| **PRIMARY (REVIEW ignored)** | **0.698** | **0.774** | **0.734** | **9.48** | **0.777** | **0.428** | 20,108 | 8,694 | 5,879 |
| Secondary (+ sub-threshold, PROPOSED) | 0.779 | 0.774 | 0.777 | 6.21 | 0.803 | 0.440 | 20,108 | 5,697 | 5,879 |
| Reference (no ignores) | 0.534 | 0.774 | 0.632 | 19.17 | 0.664 | 0.370 | 20,108 | 17,579 | 5,879 |

**Test against validation** (same frozen configuration, primary policy):

| | Precision | Recall | F1 | FP / image | AP50 | AP50-95 |
|---|---|---|---|---|---|---|
| Validation (923 images, used for selection) | 0.696 | 0.775 | 0.734 | 9.47 | 0.776 | 0.432 |
| **Test (917 images, once)** | **0.698** | **0.774** | **0.734** | **9.48** | **0.777** | **0.428** |

**Test reproduces validation to within ±0.004 on every headline metric.** The selection didn't overfit the validation split. Both splits come from the same synthetic distribution and the same automated labelling.

## 3. Recall by rock size (primary policy, conf 0.20)

The ± values are 95% binomial intervals, treating boxes as independent. Boxes within one image are correlated, so the true uncertainty is larger.

| Size (px²) | GT (test) | Recall test | ±95% | Recall val | Precision test | AP50 test |
|---|---|---|---|---|---|---|
| small < 400 | 17,110 | 0.743 | 0.007 | 0.740 | 0.647 | 0.726 |
| medium 400–2500 | 7,544 | 0.842 | 0.008 | 0.849 | 0.813 | 0.870 |
| large 2500–10000 | 1,109 | 0.782 | 0.024 | 0.815 | 0.847 | 0.835 |
| **very large ≥ 10000** | 224 | **0.821** | 0.050 | 0.760 | 0.760 | 0.817 |
| **all large ≥ 2500** | 1,333 | **0.788** | 0.022 | 0.807 | 0.828 | 0.829 |

**Large and very-large rocks:**
- **About one large rock in five is missed** at the frozen threshold: recall 0.788 for all large rocks.
- Of the 490 test images that contain a large rock, **194 (40%) miss at least one**. On validation it was 181 of 491.
- The very-large bin swung from 0.760 (val, n = 204) to 0.821 (test, n = 224). That is within the sampling uncertainty of about ±0.05 at this size, so **neither number on its own is reliable**. Pooled over both splits, very-large recall is about 0.79.
- Medium rocks are detected best. Large-rock recall sits *below* medium-rock recall on both splits, which is the reverse of what you'd expect from object size alone.

**Bottom-edge (near-field) rocks:** visually, near-field rocks cut off at the bottom of the image were often missed. Counting them (`large_rock_edge_analysis.json`, descriptive only):
- large rocks touching the bottom edge: recall 0.751 (n = 173) on test, 0.801 (n = 166) on validation;
- interior large rocks: 0.792 on test, 0.806 on validation.

The gap is within the uncertainty (±0.065), so **the near-field truncation effect is not confirmed**. It's still worth a targeted check with human-verified labels, because the nearest rocks matter most to a rover.

## 4. Confusion analysis (single class, conf 0.20; `test_confusion_analysis.png`)

**Predictions (28,802 kept after the 50 px² filter, before ignores):**

| Outcome | Primary | Reference (no ignores) | Meaning |
|---|---|---|---|
| True positive | 20,108 | 20,108 | IoU ≥ 0.5 with an unmatched VALID box |
| Ignored (REVIEW) | 8,885 | 0 | Lands on an uncertain REVIEW region: neither TP nor FP |
| FP: duplicate | 465 | 467 | IoU ≥ 0.5 with a ground-truth box that is already matched |
| FP: localisation | 3,747 | 3,966 | IoU 0.1–0.5 with a ground-truth box: wrong extent, merged or split |
| FP: background | 4,482 | 13,146 | IoU < 0.1 with any VALID box |

Under the primary policy, about 1.4 FP per image are duplicates or localisation errors on real rocks, and about 4.9 per image are background. The secondary policy cuts background FPs from 4,482 to 1,617. That suggests many "background" detections are small rocks below the labelling threshold rather than hallucinations, but this is unconfirmed until the labels are reviewed.

**False negatives (5,879):**

| Category | All | small | medium | large | very large |
|---|---|---|---|---|---|
| Found below threshold (a prediction < 0.20 matches) | 3,584 (61%) | 2,639 | 782 | 144 | 19 |
| Localisation (best prediction ≥ 0.20 has IoU 0.1–0.5) | 1,233 (21%) | 891 | 275 | 57 | 10 |
| Near-total miss (IoU < 0.1) | 1,062 (18%) | 870 | 139 | 42 | 11 |

Validation proportions were 60 / 23 / 17%, so the error profile matches between splits.

**Visual inspection** of the 6 test scenes with the most missed large rocks (`fn_scenes/`). This is qualitative, so these are hypotheses:
- Several "missed large rocks" are **label artefacts**: a crater or shadow region boxed as a rock (render3996); cliff or slope faces boxed as rocks (render1158); terrain slabs cut off at the image edge (render4752).
- **Clusters of touching boulders** labelled as one box (render3197, 0085) count as localisation misses even when the individual rocks are detected.
- **Genuine misses:** isolated near-field boulders at the bottom of the frame (render3197, 3500) and rocks in dark shadow.

## 5. Interpretation and limitations

**What is supported by the data:**
- At conf 0.20 the frozen model finds about 77% of the automatically labelled rocks, with about 9.5 false positives per image counted against VALID labels (primary policy).
- Results are stable between validation and test.

**Limitations:**
- **Labels:** automated and not human-validated. Visible artefacts (craters, slopes, merged clusters) inflate the miss counts for large rocks; other errors could inflate or deflate any metric. **The true accuracy is unknown until human review (E.1b).**
- **Policy dependence:** AP50 spans 0.664–0.803 depending on the ignore policy. Only the primary policy is approved; the secondary row is not an endorsement of the proposed policy.
- **Domain:** synthetic renders only. No real lunar imagery was evaluated (NASA POLAR is out of scope), and no Webots imagery has been tested yet.
- **Statistics:** one training seed; size-bin intervals assume independent boxes; the very-large bin is small.
- **Safety:** about 21% of large rocks are missed in this synthetic test. **This detector alone isn't sufficient for obstacle avoidance.** A rover pipeline would need redundancy, such as depth or stereo hazard detection, temporal tracking across frames and conservative path margins, before any deployment claim.

## 6. Status

- **Test evaluation: COMPLETE (once-only, locked).** The model and configuration are unchanged; nothing was tuned on test.
- **Webots integration: NOT STARTED.** The authorisation message lists only Task 1. The integration inputs would be the frozen model and config:
  - model `experiments/task-f/model-selection-001/model/artemis_rock_yolo11n_e1c_s42.pt`;
  - inference settings: imgsz 736, conf 0.20, NMS IoU 0.6, max_det 300, 50 px² post-filter.
  - Webots camera images will differ from the training renders, so a domain-gap check on Webots frames should come first.

## Files

| File | Content |
|---|---|
| `test_results.json` | All metrics: three policies, size bins, confusion breakdown, FN analysis, verification, integrity |
| `test_evaluation_figures.png` | PR curve with the operating point; recall by size (val vs test); policy comparison |
| `test_confusion_analysis.png` | Prediction outcomes and FN categories |
| `per_image_test_primary.csv` | Per-image GT, FN and large-rock FN |
| `large_rock_edge_analysis.json` | Edge vs interior recall for large rocks |
| `fn_scenes/` | Overlays of the 6 hardest large-rock scenes |
| `predictions/test/labels/` | Raw predictions (conf ≥ 0.001), 917 files |
| `TEST_RUN_LOCK.json` | Once-only lock: start and finish time, config and model SHA-256 |
| `test_eval_console.txt` | Run log |
| `CHECKSUMS.sha256` | Checksums of every file in this folder |

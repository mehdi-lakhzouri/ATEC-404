# REVIEW_REQUIRED Handling — Decision Record (AI-01.D)

**Status: PENDING HUMAN DECISION.** Neither configuration is validated, and no model has been trained.

## 1. The problem

- **Scale:** Task C left 192,826 REVIEW_REQUIRED boxes. On the eligible frames that is 40% of all non-rejected boxes.
- **What they contain:** in the Task C review sample, 5 of 12 "low-visibility" REVIEW boxes and 1 of 10 "possible merge" boxes were in fact visible rocks.
- **Why it matters:** YOLO (Ultralytics) has no ignore class. An unlabelled region is treated as background. Training on VALID labels alone would teach the model that many visible rocks are background, which is exactly what the task forbids.

## 2. The two prepared configurations (train split only; val/test identical and unmasked)

| | Config A: conservative scenes | Config B: masked uncertain regions |
|---|---|---|
| Rule | Train scenes with REVIEW / (VALID + REVIEW) ≤ 0.25 and no CSV rock withheld in shadow | All train scenes; every REVIEW box inpainted (OpenCV Telea, radius 5) in a derived copy |
| Train scenes | 1,903 (26% of train) | 7,353 (100%) |
| VALID boxes (exported labels) | 50,060 | 209,430 |
| REVIEW boxes still treated as background | **12,161** (unmasked, unlabelled) | **0**; 149,983 masked, 4,352 left visible where they overlap a protected VALID box |
| VALID targets altered | none | none: VALID boxes plus 2 px are protected, and 0 pixels outside the mask changed (verified on every image) |
| Mean masked area | — | 3.8% of the image (p95 12.4%; 23 scenes ≥ 25%) |
| Hard scenes / no-record negatives | 81 / 121 | 287 / 121 |
| Small boxes (< 400 px²) among VALID | ≈ 69% | ≈ 66% |

**Trade-off for Config A's threshold** (train split):

| Threshold | Scenes | VALID boxes | REVIEW boxes left as background |
|---|---|---|---|
| 0.10 | 574 | 3,251 | 160 |
| 0.20 | 1,220 | 23,106 | 4,150 |
| **0.25 (chosen)** | **1,913 → 1,903 after the shadow rule** | **50,579** | **12,296** |
| 0.30 | 2,735 | 86,513 | 26,043 |
| 0.50 | 5,400 | 176,595 | 82,829 |

## 3. Limitations, observed in `previews/config_B_before_after.jpg`

**Config A**
- It still contains 12k unlabelled REVIEW boxes, about 6 per image.
- It discards 74% of the training scenes.
- It shifts towards sparse scenes: empty plus 1–15-rock scenes are 48% of A versus 38% of all train, and 36+-rock scenes 31.5% versus 34.3%.
- It is not biased towards easy lighting: hard scenes are 4.3% of A versus 3.9% of train.

**Config B**
- **Visible inpainting artefacts:** smooth smudges (frames 5541, 1401, 5623) and detached shadow fragments where a rock was masked but its shadow was not (4970, 5623). A model could learn "smudge = no rock", and it never sees those artefacts at test time. This shift between training and test images is the main risk.
- **Fewer rocks in the scene:** masking removes real rocks, so the model sees less cluttered terrain than at evaluation.
- **Partial masking:** REVIEW boxes that overlap VALID boxes are only partly masked (4,352 boxes stay fully visible).

**Both**
- Evaluation labels are VALID-only and provisional. In soft-lit scenes (e.g. val render6832) many visible rocks are REVIEW. Without the ignore regions, precision and recall would be misleading.

## 4. Recommendation for Task E (to be approved)

1. **Train A and B as a paired ablation**, with the same hyperparameters and the same evaluation:
   - VALID labels;
   - detections matching `eval_ignore_regions.json` excluded;
   - results reported per difficulty and density stratum, and separately on the NASA subset.
2. **Prefer a third option if time allows:** an ignore-aware loss. Keep images unmasked and drop REVIEW-region anchors from the loss (a custom Ultralytics trainer). This avoids both the artefacts of B and the scene loss of A. It is *not* implemented in Task D.
3. Choose the final configuration only from Task E metrics, not from this record.

| Option | What it means |
|---|---|
| CONFIG_A | Conservative scenes only |
| CONFIG_B | Masked uncertain regions |
| BOTH_AS_ABLATION | Train both and compare (recommended) |
| NEITHER | Choose another approach |

Record the decision in the review gallery under policy question `review_handling`.

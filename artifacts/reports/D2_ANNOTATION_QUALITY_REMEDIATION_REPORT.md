# Annotation Quality Remediation

**Evidence:**
- per-scene sheets in `scene_investigation/scene_*.jpg`;
- box features in `scene_investigation/evidence_boxes.csv`;
- the decision queue in `scene_investigation/adjudication_queue.csv` (54 entries).

All visual judgements here are the AI assistant's, not a human's. **No annotation was changed in this task:** every correction is a *proposal* for human adjudication.

## 1. Findings per priority scene

| Frame | Split | Audit concern | What the evidence shows | Classification |
|---|---|---|---|---|
| 0977 | train | Merged rocks, heavy masking | REVIEW boxes 103 / 106 / 107 (14–18k px, solidity 0.56–0.77) each contain **several boulders** the semantic mask joins; watershed cannot cut them. Config B masks **29.3%**, erasing the main foreground obstacles. 105 and 099 are single visible rocks withheld by visibility thresholds. | Merges: correct per policy, need manual instance boxes. 105 / 099: policy limitation. Masking: a real problem. |
| 8894 | train | Visible REVIEW rocks | 038 / 058 / 064 / 023 / 052 / 066 / 051 are clearly visible pale rocks with solidity 0.95–1.0. They fail only because edge 33–88 < 100 and contrast 6–20 < 25: **soft, diffuse lighting**. split-068 is not a rock. | Policy limitation (visibility rule) |
| 4357 | test | High uncertainty | 155 REVIEW vs 31 VALID. 170 / 187 / 189 / 192 are visible single rocks; 181 is a cluster (correct); split-105 is texture and 097 is shadow (correct). | Policy limitation + **evaluation risk** |
| 4465 | train | Zero labels despite small rocks | All rock components are ≤ 9×9 px, even at class threshold 40. | Consistent with policy (out of scope), but **not a verified rock-free negative** |
| 5311 | test | Zero labels despite small rocks | **Not the area floor:** its distant rocks have *dim mask colour* (blue median 62). At class threshold 127 the largest is 41 px; at 80 rocks reach 150 px (19×11). The render shows them clearly (edge 189). | **Extraction limitation** (class-intensity threshold) |
| 6352 | val | Bad geometry of a large rock | The mask shows **two touching boulders** (two peaks). CSV boxes 000 (VALID) and 001 are the two instances, so the geometry is correct. 001 is REVIEW because its box is ≥ 85% black against dark surroundings (shadow plus black sky); only its base outline is visible. 100 (edge 87, contrast 24.7) is visible but withheld. | Geometry correct; 001 correct per policy (adjudicate); 100 a policy limitation |
| 6864 | val | Over-conservative visibility | 68 REVIEW vs 8 VALID; 070 / 066 / 060 / 045 / 046 / 049 are visible rocks; 044 is two rocks (correct). | Policy limitation |
| 4138 | val | Over-conservative visibility | 036 / 030 / 024 / 026 / 016 / 002 are visible (002 has contrast 24.9 vs the 25 threshold); 004 / 007 / 033 are rightly uncertain. | Policy limitation |

**Adjudication queue totals:**

| Classification | Entries |
|---|---|
| Policy limitation (all promotion candidates for human confirmation) | 27 |
| Consistent with policy | 15 |
| Uncertain | 9 |
| Masking problem | 1 |
| Evaluation risk | 1 |
| Extraction limitation | 1 |

**No policy *violation* (a bug against the stated rules) was found in these scenes.** The one rule bug found in D.2 is the duplicate split pieces, fixed in v1.1.

## 2. Root causes

1. **The absolute visibility rule (edge ≥ 100 or contrast ≥ 25) fails in soft, bright lighting.** It was calibrated on 78 Task B rocks and keeps 90% of visible rocks overall. In bright low-contrast scenes it withholds most visible rocks: 60 of 70 in 8894, 155 of 186 in 4357. A scene-relative rule tested in Task C scored worse on the small calibration set (Youden J 0.60 vs 0.75), so the real fix is a **larger, lighting-stratified human-labelled calibration set**, not another heuristic.
2. **The semantic mask merges touching rocks of one class.** Watershed recovers only 39% of merged CSV instances. Large merged blobs (0977) need human instance boxes.
3. **The class-intensity threshold (> 127) drops distant rocks with dim mask colour.** At threshold 80, **65 of the 153 "no-record" frames** gain at least one ≥ 50 px component.

## 3. Proposed corrections (not applied)

| ID | Proposal | Effect | Needs |
|---|---|---|---|
| P1 | Human-confirm the 27 promotion candidates; promote only confirmed ones | Fewer visible rocks treated as unknown | Human review |
| P2 | Manual instance boxes for 0977-103/106/107, 4357-181, 6864-044 | Merged clusters become trainable | Human annotation (~5 min/scene) |
| P3 | Recalibrate the visibility rule on ≥ 300 human-labelled candidates stratified by lighting | Addresses the systematic soft-light loss | Human labels |
| P4 | Tag the 153 no-record frames `negative_unverified`; never use them as true negatives in evaluation | Prevents false "rock-free" claims | Policy approval |
| P5 | Use the proposed sub-threshold / dim-mask ignore regions in evaluation | Detections of real but out-of-scope rocks are not counted as FP | Approval (`EVALUATION_PROTOCOL_REPORT.md`) |
| P6 | Exclude 0977-type heavily masked scenes from config B | See `MASKING_ABLATION_RECOMMENDATION.md` | Approval |

## 4. The 50 px minimum and the in-scope definition (item 4)

**Data:** `threshold_study/threshold_study.json`, `threshold_curves.png`, 3 visual samples.

| Rule (class threshold, min area) | Components / image | Visible by proxy | Low-solidity (merge risk) | Box side < 6 px | Frames with none |
|---|---|---|---|---|---|
| 127, ≥ 10 | 99.0 | 53% | 4.8% | 42% | 44 |
| 127, ≥ 25 | 67.0 | 59% | 7.1% | 17% | 83 |
| **127, ≥ 50 (current)** | **46.7** | **62%** | 9.6% | 2.8% | 153 |
| 80, ≥ 50 | 52.6 | 54% | 10.9% | 2.4% | 88 |

Visual samples (AI assistant, 20 per stratum), share that are a visible single rock:

| Stratum | Visible single rock |
|---|---|
| 10–24 px | 15% |
| 25–49 px | 40% |
| ≥ 50 px only at class threshold 80 | 25% (3 of 20 were merged rows) |
| VALID ≥ 50 px (Task C, for comparison) | 69–88% |

- **Compute:** extraction takes about 0.06 s per frame at any threshold. Training cost grows with box density: going from 50 to 25 px means +43% targets per image.

**Recommendation: keep `MIN_AREA = 50` and class threshold 127 for *labels*.** Lowering either adds mostly ambiguous, dim or merged targets. The official threshold is **unchanged**.

**Proposed in-scope definition for Artemis-X detection and evaluation:**

| Category | Definition | Treatment |
|---|---|---|
| In scope | A rock whose mask component is ≥ 50 px at native 720×480 (median box side 12 px; at `imgsz 736` no downscaling) | Training and evaluation target (VALID) |
| Out of scope, ignored | Components 10–49 px, and components ≥ 50 px visible only at threshold 80 | Detections there are neither TP nor FP; never false negatives |
| Prediction scope in evaluation | Predicted boxes < 50 px² | Dropped |
| Reported size bins | small < 400 px², medium 400–2,500 px², large ≥ 2,500 px² | Box area at native resolution |

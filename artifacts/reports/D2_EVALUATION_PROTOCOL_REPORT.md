# Evaluation Protocol

## 1. Does the YOLO evaluation code support `eval_ignore_regions.json`?

**No.** Ultralytics 8.4.174 `DetectionValidator` has no ignore or crowd handling: `iscrowd` appears only as a hard-coded 0 in its COCO JSON export. Any Ultralytics `val` metric on this dataset therefore counts detections on REVIEW regions as false positives. It must be labelled **PROVISIONAL / reference only**.

## 2. Ignore-aware evaluator (implemented and tested)

`backend/analysis/evaluation/ignore_eval.py`, wired in `backend/training/evaluate.py`.

**Matching:** COCO-style, greedy by confidence, one ground-truth box per prediction.

**Ignore rule** (COCO crowd semantics):
- A prediction that matches no ground-truth box but has IoU ≥ 0.5 with an ignore region, or ≥ 50% of its area inside one, is **neither TP nor FP**.
- Ignore regions never produce false negatives.

**Metrics:**
- AP50 and AP50:95, using 101-point interpolation;
- precision, recall, TP, FP and FN at a confidence threshold;
- size-binned AP50 and recall (small < 400, medium 400–2,500, large ≥ 2,500 px² at 720×480), with COCO handling of other-size ground truth;
- an optional prediction scope (`min_pred_area`, 50 px²).

**Inputs:**
- predictions from `predict(..., save_txt=True, save_conf=True, conf=0.001)` on **unmasked** val/test images;
- ground truth from `yolo-rock-v1.1/labels/{val,test}`;
- ignore files:
  - `eval_ignore_regions.json`: REVIEW_REQUIRED boxes, the policy (38,478 on val/test in v1.1: val 19,411, test 19,067);
  - `eval_ignore_regions_subthreshold_PROPOSED.json`: out-of-scope rocks, a proposal. On val: 48,306 sub-threshold plus 7,379 dim-mask regions. On test: 47,900 plus 7,054.

**Tests** (`backend/tests/test_ignore_eval.py`, 16 tests, all pass):
- *Matching:*
  - perfect predictions;
  - an ignored REVIEW detection is not a false positive, and removing the ignore makes it one;
  - ignore regions create no false negatives;
  - the IoA rule;
  - recall with a missed ground-truth box;
  - duplicate detections;
  - a duplicate that falls in an ignore region.
- *Scope and size:* minimum prediction area; confidence threshold; size bins.
- *AP computation:* a hand-computed 101-point AP; agreement with **Ultralytics `ap_per_class` within 0.02** when there are no ignores.
- *YOLO parsing.*
- *Real v1.1 validation split:* using the labels as predictions gives AP50 = 1.0. Adding a "detection" on every REVIEW box gives **0 FP with ignore regions, and precision < 0.6 without them.**

## 3. Protocol for Task E (proposed)

1. Val and test images stay **unmasked**. Split assignments and leakage groups are those of `yolo-rock-v1` (verified identical in v1.1).
2. **Primary metric:** ignore-aware AP50 on val (for checkpoint selection) and test (once per arm), with REVIEW plus sub-threshold ignores, `min_pred_area = 50`.
3. **Also report:** AP50:95, size-binned AP50, recall@0.25, and the mean ± spread over 3 seeds.
4. Ultralytics `val` mAP is reported as **PROVISIONAL**. `best.pt` selected by Ultralytics is *not* used. Checkpoints are selected with the ignore-aware evaluator.
5. The 153 no-record frames are never treated as verified negatives. Detections there are judged against the sub-threshold ignore regions.
6. **NASA POLAR:** the 92-image subset stays **blocked** until the 12→8-bit conversion is approved (open since Task B).

## 4. Human-verified test benchmark (recommendation)

All current val/test labels are automated (S2-H) and **not human-verified**. A small gold benchmark is needed before any claim of model quality.

**Size and selection**
- 150 test images (16% of test).
- Stratified on the existing strata: difficulty × density × uncertainty.
- Oversample hard (≥ 30), dense (≥ 40) and high-uncertainty (≥ 50) scenes.
- Include 10 of the 153 no-record frames.

**Annotation protocol**
- Exhaustive boxes for **every visible rock in scope** (≥ ~8×6 px).
- Separate "uncertain" boxes, used as ignore regions.
- Instance-level boxes for touching rocks.
- Start from S2-H VALID + REVIEW pre-annotations, which the annotator must accept, edit or delete.

**Quality control**
- 30 images double-annotated: target matched-box F1 ≥ 0.85 at IoU 0.5.
- Disagreements adjudicated by a third person.

**Effort**
- About 4 minutes per image, so 10–12 person-hours including QC.
- The v2 gallery can collect decisions; box editing needs a tool such as CVAT or Label Studio (not installed).

**Use**
- Report every model on the gold set alongside the automated test set. Any gap between the two quantifies the label noise.

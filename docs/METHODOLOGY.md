# Methodology

This page summarises the historical pipeline. Each stage links to its full report in `artifacts/reports/` and to a notebook.

| Task | Question | Method | Outcome | Notebook |
|---|---|---|---|---|
| A | What data exist and how good are they? | Inventory, decoding, statistics, archive-vs-extraction CRC checks, quality register | CSV labels cover only large rocks; 573 anomaly frames excluded; 376 near-duplicate pairs | 01 |
| B | How should boxes be derived? | S1 (CSV) vs S2 (mask components) on a stratified sample, AI-assisted visual review (not human) | HYBRID S2-H, single class `rock` | 02 |
| C / C v1.1 | Which boxes are trustworthy? | Deterministic rules: size floor, watershed split, visibility proxy, three statuses, duplicate removal | curated-dataset-v1.1 | 03 |
| D | How to train and evaluate without leakage? | Grouped, stratified split; YOLO export; Config A (low-uncertainty scenes) vs Config B (REVIEW masked) | yolo-rock-v1.1 | 04 |
| D.2 | Is the data ready, and how to evaluate fairly? | Size-floor study, masking audit, scene investigation, **ignore-aware evaluator** | Evaluation protocol; Config A preferred for first training | 02–04, 06 |
| E.0 / E.1a | Does the pipeline run safely and fast? | 3-epoch smoke test; 1-epoch benchmark with telemetry | Fixed the class-name bug (`item` → `rock`); 736 px, batch 16, 4 workers | 05 |
| E.1c | Train the model | YOLO11n, Config A, 200 epochs, seed 42, SGD + cosine, AMP; guarded wrapper; telemetry; 42 verified checkpoints | Converged by about epoch 100–120; mild overfitting later | 05 |
| F | Select and test | Ignore-aware validation of 22 checkpoints; **rules fixed before threshold metrics**; threshold grid 0.05–0.25; freeze; **once-only** test | Epoch 111, conf 0.20; test F1 0.734 | 06, 07 |

## Ignore-aware evaluation

**Matching:**
- Predictions under 50 px² are removed.
- The rest are matched to VALID boxes greedily by confidence at IoU ≥ 0.5.
- An unmatched prediction whose IoU **or** IoA (intersection ÷ prediction area) with a REVIEW region is ≥ 0.5 is *ignored*.
- Other predictions are false positives.

**Metrics:**
- AP uses 101-point interpolation.
- AP50-95 averages IoU thresholds 0.50:0.95:0.05.
- Size bins: small < 400 px², medium 400–2,500 px², large ≥ 2,500 px².

**Policies** (always reported separately):
- **primary:** REVIEW regions ignored (approved);
- **secondary:** also ignores the *proposed* sub-threshold regions (not approved);
- **reference:** no ignores.

## Selection rules (fixed in code before computing threshold metrics)

- **Checkpoint:** keep `best.pt`, the provisional primary named by the owner, unless a candidate beats its primary AP50 by
  more than 0.005.
- **Threshold:** the lowest grid value whose F1 is within 0.02 of the best F1 on the grid. This is recall-oriented, because a
  missed obstacle costs more than a false alarm.
- **NMS and pre-processing:** unchanged from the evaluation that produced the candidates.

The checkpoint AP values were already known from the E.1c evaluation when the rules were written. The threshold metrics were not.

## Integrity controls

- **Pre- and post-flight fingerprints** of the source datasets, Task C/D outputs and earlier experiments for every GPU run.
- **Content-hash registry** (`VERSIONS.json`) for each dataset version.
- **Checksums** for every run folder, the frozen selection and the test evaluation.
- **Test evaluation guard:** refuses to run unless the frozen config, model, ignore file and split manifest hashes match, and
  leaves a lock file behind.

# Dataset Engineering Report — `yolo-rock-v1` (AI-01.D)

**Status:** PROVISIONAL · **NOT FROZEN** · not approved for training · no model trained
**Code:** `backend/analysis/yolo_dataset/` · **Run** (about 8 min, deterministic):

```
cd backend
uv run --no-sync python -m analysis.yolo_dataset --workers 8
```

## 1. Task C validation (read-only; `task_c_validation.json`)

All 13 checks pass. Nothing in Task C was modified: its file fingerprint is identical before and after.

1. **Status counts:** 261,385 VALID / 192,826 REVIEW_REQUIRED / 49,817 REJECTED, matching `counts.json`.
2. **Integrity:** annotation IDs are unique, statuses are in the allowed set, and every non-VALID record has a reason.
3. **Geometry:** all 504,028 boxes are inside 720×480, with w = x1 − x0 and h = y1 − y0.
4. **Derived files:** `rock_valid.csv`, `review_required.csv` and the COCO export each match their subset of `annotations_all.csv`.
5. **CSV preservation:** all 17,685 eligible CSV rows are present with exact coordinates (X − 0.5).
6. **Per-scene counts** in `scenes.csv` match the annotations, and annotations appear only on eligible frames.
7. **Source hashes:** the SHA-256 of all 9,766 render and ground files was recomputed and matches `scenes.csv`.

**The 281 vs 128 discrepancy is a flag-definition gap, not a data error.**
- Task C only sets `no_valid_annotation` on frames that have at least one record.
- 153 eligible frames have **no records at all**: no CSV box and no mask component ≥ 50 px. 128 + 153 = 281.
- 24 of the 153 use palette-mode masks. I confirmed those decode to pure class colours: they are near-empty masks the encoder stored compactly.
- 16 random no-record frames are rock-free slopes; 3 show only tiny distant rocks below the 50 px policy floor.
- **Handling:** Task D adds a derived `annotation_state` (`has_valid` / `records_no_valid` / `no_records`) to its own manifest. Task C files are unchanged.

## 2. Export layout (`dataset/processed/yolo-rock-v1/`, 4.8 GB)

| Path | Content |
|---|---|
| `images/{val,test}`, `labels/{val,test}` | Shared evaluation set: unmasked, byte-identical source copies (923 / 917) |
| `config_A/{images,labels}/train` | 1,903 conservative train scenes, unmasked |
| `config_B/{images,labels}/train` | 7,353 train scenes with REVIEW regions inpainted |
| `data_A.yaml`, `data_B.yaml` | Ultralytics configs (`nc: 1`, `names: {0: rock}`); absolute `path`; header marked PROVISIONAL |
| `manifests/split_manifest.csv` | Per frame: split, leakage group, stratum, `annotation_state`, counts, config membership, masked fraction, source hash |
| `manifests/eval_ignore_regions.json` | 38,491 REVIEW boxes on val/test for an ignore-aware evaluation |
| `manifests/config_B_masking.csv` | Per-image masking statistics |
| `PROVISIONAL.md`, `manifests/dataset_status.json` | `frozen: false`, `approved_for_training: false` |

**Labels:**
- Format: `0 cx cy w h`, normalised, 6 decimals, VALID boxes only. Frames with no VALID box get an empty label file.
- REVIEW and REJECTED boxes are never written as labels.
- Images are the source renders: RGB with nearly identical channels, 720×480.

## 3. Export validation (`split_statistics.json` → `export_validation`)

- **Image-label correspondence:** every image has exactly one label file and vice versa, in all four sets. Every image is 720×480.
- **Labels:** 0 invalid lines. Each file's line count equals its frame's exported VALID count. 0 pairs of near-identical boxes (IoU > 0.9) in any image. The maximum round-trip error versus the source boxes is **0.0005 px**.
- **Unmasked copies:** all 3,743 (val, test, config A train) are byte-identical to the source renders (SHA-256). No masking reached evaluation images.
- **Config B:** 0 pixels changed outside the mask, across all 7,353 images.
- **Independent check with Ultralytics 8.4.174:**
  - Its own path mapping (`img2label_paths`) resolves every image to a label file inside the export.
  - Its own verifier (`verify_image_label`) reports, per set:

| Set | Images | Boxes | Missing labels | Corrupt |
|---|---|---|---|---|
| A train | 1,903 | 50,060 | 0 | 0 |
| B train | 7,353 | 209,430 | 0 | 0 |
| val | 923 | 25,822 | 0 | 0 |
| test | 917 | 25,987 | 0 | 0 |

  - The verifier also emits no warnings. A first run had flagged 13 duplicate label lines, which led to the defect described in §3a.
  - No dataset check that could download files was used, and no label cache was written.
- **Previews:** `previews/labels_*.jpg` draw boxes back from the exported `.txt` files; `previews/config_B_before_after.jpg` shows the masking.

## 3a. Task C defect found and corrected at export (Task C files untouched)

- **Defect:** Task C applied its duplicate rule D05 ("duplicate of a CSV instance") to mask components *before* the watershed split, but never to the split pieces. When a split recovers a CSV rock, both boxes stay VALID.
- **How it was found:** Ultralytics' verifier reported identical label lines (11 exact duplicates). An IoU audit then showed 97 CSV / blue-piece pairs with IoU > 0.5, plus 2 green-piece pairs.
- **Correction:** D05 is applied to VALID split pieces at export time, with the same thresholds as Task C (blue: ≥ 50% inside a CSV box or IoU ≥ 0.3; green: IoU ≥ 0.5).
  - **146 pieces excluded**, each listed with its CSV twin in `manifests/export_exclusions.csv`.
  - The rock stays labelled through its CSV box, and is therefore not masked in config B.
  - Exported labels: 261,239 = 261,385 VALID − 146.
- **Recommendation:** fix it at the source in a Task C v1.1 (curated-dataset-v1.1) after approval.

## 4. Splits, configs and uncertainty

- Splits and leakage control: `SPLIT_VALIDATION_REPORT.md`.
- Configurations A and B and their limitations: `REVIEW_HANDLING_DECISION.md`.

## 5. Human review (`review_gallery/index.html`)

**Opening it**
- Open the file directly in a browser. It works offline: images load from the dataset by relative path.
- 36 scenes: the 5 required frames (0676, 0742, 0977, 4191, 8894), the duplicate pair 1 / 6658, and stratified picks across train, val and test.
- The picks cover hard scenes, dense scenes, high and low uncertainty, negative frames, split pieces, rocks withheld in shadow, and held-back merges.

**What you see**
- Boxes in distinct colours *and* line styles:

| Box | Style |
|---|---|
| VALID from the CSV | cyan, solid |
| VALID from the mask | green, solid |
| REVIEW | orange, dashed |
| REJECTED | red, dotted |

- Toggles show or hide each status, show the config B masked image, and zoom 2×. Hovering a box shows its ID, source and reasons.

**Recording decisions**
- For each scene, choose NOT_REVIEWED, APPROVE or FLAG. You can add flag categories and notes.
- Four policy questions cover REVIEW handling, the S2-H labels, provisional evaluation labels, and freezing the dataset.
- Progress autosaves in the browser. **Export JSON** (and CSV) downloads your decisions.
- Put the exported `review_decisions.json` into `experiments/yolo-dataset/human_review/`. The next pipeline run records it in `dataset_status.json`.
- **Current state:** 36/36 NOT_REVIEWED, policy UNDECIDED.

**Testing caveat:** the gallery's data and paths were validated and its script passes `node --check`. Its buttons were not exercised in a real browser by me, so please confirm export works on first use.

## 6. Reproducibility and integrity

- **Source data unchanged:** fingerprint `fecca0cf…` (38,590 files), identical to Tasks A–C.
- **Task C artefacts unchanged:** fingerprint identical before and after the run.
- **Two consecutive runs:** byte-identical outputs, see §8. That covers labels, manifests, yaml files, config B PNGs, previews and the gallery.
- **Seeds:** RANSAC uses a fixed RNG seed; splits use 20261011.
- **Side effect outside the project:** importing Ultralytics created its standard settings file in `%APPDATA%\Ultralytics\settings.json`.

## 7. GO / NO-GO for Task E — **CONDITIONAL GO**

The export is technically sound and leakage-controlled. Before any training run, the conditions below must hold:

1. You review the gallery (at least 20 scenes) and export your decisions.
2. You choose REVIEW handling: A, B, both as an ablation (recommended), or neither. The ignore-aware-loss alternative is described in `REVIEW_HANDLING_DECISION.md`.
3. You accept, or replace, the provisional val/test labels. Evaluation must use `eval_ignore_regions.json`.
4. You approve freezing the dataset. Until then `frozen` stays `false`.
5. NASA POLAR evaluation needs its own 12-bit to 8-bit conversion decision. That decision is still open from Task B.

## 8. Determinism check

Two consecutive full runs (about 10 min each) produced **22,217 byte-identical files**: the whole export plus this folder, logs excluded. Both runs: all checks ok, 0 errors.

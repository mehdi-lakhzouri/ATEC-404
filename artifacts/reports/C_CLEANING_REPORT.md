# Cleaning Report — AI-01.C

**Date:** 2026-10-09 · **Policy:** S2-H/1.0 (`ANNOTATION_POLICY.md`) · **Status:** DONE, pending human review
**Code:** `backend/analysis/dataset_cleaning/` (reuses the Task A and Task B parsers, mask-class logic and drawing code)
**Run:** about 140 s, deterministic

```
cd backend
uv run --no-sync python -m analysis.dataset_cleaning --workers 8
```

## 1. Counts (9,193 eligible frames; full dataset; automated)

| | Boxes | Per image (mean / median) |
|---|---|---|
| S1: CSV boxes only | 17,685 | 1.92 / 1 |
| Raw S2: CSV + every V1 component ≥ 50 px | 447,357 | 48.7 / 46 |
| **Cleaned S2-H, VALID** | **261,385** | **28.4 / 23** |
| REVIEW_REQUIRED | 192,826 | — |
| REJECTED | 49,817 | — |

- 281 frames have no VALID box.
- REVIEW_REQUIRED + REJECTED add up to more than raw minus VALID, because split pieces create new records.

**By source:**

| Source | VALID | REVIEW | REJECTED |
|---|---|---|---|
| `csv_large` | 17,518 | 167 | 0 |
| `mask_green` | 182,735 | 165,565 | 31,000 |
| `mask_green_split` | 36,896 | 15,856 | 1,042 |
| `mask_blue` | 21,952 | 10,663 | 17,757 |
| `mask_blue_split` | 2,284 | 575 | 18 |

- **All 17,685 CSV instances are preserved.** 167 are marked REVIEW because the rock is invisible in black shadow. 191 dark rocks that are still visible as silhouettes stay VALID with the flag `dark_silhouette`.
- **Reasons** (one record can carry several):

| Reason | Records |
|---|---|
| `ambiguous_low_visibility` | 159,732 |
| `possible_merge_borderline_split` | 28,327 |
| `replaced_by_split_pieces` | 25,405 |
| `low_solidity_possible_merge` | 20,862 |
| `duplicate_of_csv_instance` | 16,120 |
| `invisible_black_shadow` | 8,117 |
| `mostly_in_shadow` | 3,501 |
| `thin_artifact` | 181 |
| `csv_box_in_black_shadow` | 167 |
| `large_blue_without_csv_likely_merged` | 132 |
| `overlaps_csv_large_rock` | 25 |

- **Not listed:** 1,318,681 mask components under 50 px (64,717 blue, 1,253,964 green).

## 2. Merged rocks: watershed, measured against CSV instance ground truth

- **Test set (blue components ≥ 200 px):**
  - 1,742 contain ≥ 2 CSV box centres, i.e. known merges holding 3,934 CSV instances;
  - 13,180 match exactly one CSV box (IoU ≥ 0.7), i.e. known single rocks.
- **Results by neck threshold:**

| Neck threshold | Instances recovered (IoU ≥ 0.5) | Over-split singles |
|---|---|---|
| No split | 1,004 (25.5%) | — |
| 0.4 | 1,356 | 1.1% |
| 0.6 | 1,457 | 1.5% |
| **0.7 (selected)** | **1,539 (39.1%)** | **2.1%** |
| 0.8 | 1,640 | 5.4% (above the 5% limit) |

- **Effect at 0.7:** 416 of 1,742 merges (24%) get split.
- **Applied to all mask components:** 25,405 parents are split into 56,671 pieces, of which 39,180 are VALID. 49,273 possible merges that were not split go to REVIEW_REQUIRED instead of being accepted.
- **Visual check** (`before_after/splits_accepted.jpg`, 24 random accepted splits):
  - most separate visibly distinct rocks (e.g. f3142, f6612, f6665);
  - a few cut one rock along a texture line (f6343, f6380); the dark half then lands in REVIEW;
  - pieces on shadowed or textured ground mostly become REVIEW.
- **Limit:** watershed recovers only 39% of merged instances. 61% of merged regions stay merged; most of those are REVIEW_REQUIRED rather than VALID.

## 3. Rule revision made during this task (recorded, not hidden)

- **First rule:** CSV boxes with ≥ 85% dark pixels were sent to REVIEW (358 boxes).
- **Problem:** in the first independent sample, 6 of those 8 boxes were clearly visible black rocks against lit ground (evidence: `assistant_review/csv_shadow_rule_v0_evidence.csv`).
- **Fix:** the rule now also requires dark surroundings (mean < 30). That leaves 167 REVIEW boxes.
- **Check:** the re-drawn sample of 8 contains 0 visible rocks (5 not visible, 3 ambiguous).
- **Sample stability:** the other 122 review items were identical between the two versions.

## 4. Reproducibility and integrity

- **Source data unchanged.** The source fingerprint, taken over `dataset/` excluding `.git` internals and `dataset/processed/`, was identical before and after every run: `fecca0cf…`, 38,590 files. This is the same value as in Task A, so the source data is unchanged since Task A.
- **Deterministic output.** Two consecutive runs produced 41 byte-identical files (experiment outputs + curated dataset, logs excluded).
- **Logs.** 0 errors logged. The run log is `logs/run.log`.
- **Out of scope.** No YOLO export, no train/val/test split, no training.

## 5. Generated files

**`dataset/processed/curated-dataset-v1/`** (243 MB):

| Folder | Files |
|---|---|
| `annotations/` | `annotations_all.csv` (all 504,028 records with status, reasons, features, provenance), `rock_valid.csv`, `rock_valid_coco.json` (VALID only), `review_required.csv` |
| `manifests/` | `scenes.csv` (9,766 frames: eligibility, flags, hashes, leakage group), `excluded_frames.csv`, `duplicate_groups.csv` |
| `metadata/` | `policy.json`, `calibration.json`, `provenance.json`, `counts.json` |

**`experiments/dataset-cleaning/`:**
- the three reports;
- `quality_metrics.json`, `cleaning_decisions.csv`, `review_required.csv`, `review_sample.csv`;
- `before_after/`: 16 scene sheets, the accepted splits, the unsplit merges, and 6 review-sample sheets;
- `assistant_review/`.

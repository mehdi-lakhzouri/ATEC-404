# Task C v1.1 — Duplicate Split-Piece Fix

**Status:** applied at the source, as a new version. v1 is archived unchanged.
**Policy:** S2-H/1.0 → **S2-H/1.1**
**Code:** `backend/analysis/dataset_cleaning/build.py` (`duplicates_csv`)

## 1. Defect

- **What went wrong:** rule D05 rejects a mask box that duplicates a CSV rock instance (blue: ≥ 50% inside or IoU ≥ 0.3; green: IoU ≥ 0.5). v1 applied it only to *parent* mask components, before the watershed split. When a split recovered a CSV rock, the piece stayed VALID next to the CSV box.
- **How it was found:** Ultralytics' label verifier during Task D.
- **v1 workaround:** an export-time exclusion of 146 pieces, with Task C files left untouched.

## 2. Fix

- **Rule factoring:** D05 is now one helper, `duplicates_csv(box, class, csv_boxes)`, used for parents (behaviour unchanged) **and** for every split piece.
- **Annotation IDs stay stable:** a duplicate piece keeps its `ann_id` and becomes `REJECTED / duplicate_of_csv_instance`, flagged `split_piece_dedup`.
- **Versioned output:**
  - `dataset/processed/curated-dataset-v1.1/`
  - `experiments/dataset-cleaning-v1.1/`
- **Review evidence:** the review sample and its verdicts are reused from v1 and **matched by `ann_id`**, never by position. No sampled item changed status.

## 3. Before / after (`metadata/counts.json`)

| Status | v1 | v1.1 | Δ |
|---|---|---|---|
| VALID | 261,385 | 261,239 | −146 (exactly the 146 export-time exclusions) |
| REVIEW_REQUIRED | 192,826 | 192,784 | −42 (REVIEW pieces duplicating a CSV rock; the v1 export fix never examined these) |
| REJECTED | 49,817 | 50,005 | +188 |

- **What changed:** all 188 changed records are split pieces with unchanged coordinates. Nothing else changed.
- **Records:** 504,028 annotation IDs in both versions.
- **CSV rocks:** all 17,685 CSV rocks are field-for-field identical.
- **YOLO export rebuilt from v1.1 (`yolo-rock-v1.1`):**
  - Export-time exclusions needed: **0**.
  - All label files are **content-identical** to `yolo-rock-v1` (labels tree sha256 `e2a4b406b05e…` in both).
  - 8 config B images changed slightly, because 42 former REVIEW pieces are no longer masked.

## 4. Regression tests (`backend/tests/test_task_c_v11.py`, 11 tests, all pass)

**Unit tests of the D05 helper:**
- a blue piece inside a CSV box counts as a duplicate;
- a blue piece that is 25% inside with low IoU does not;
- green needs IoU ≥ 0.5;
- with no CSV boxes, nothing is a duplicate.

**Integration tests against both versions:**
- each of the 146 previously excluded IDs is now `REJECTED / duplicate_of_csv_instance / split_piece_dedup`;
- VALID(v1) − VALID(v1.1) equals exactly those 146, and no new VALID appears;
- the same set of annotation IDs exists in both versions;
- every changed record is a split piece with identical geometry and parent;
- all CSV instances are identical;
- **no VALID split piece in all of v1.1 duplicates a CSV instance**;
- the status counts and policy version are as expected.

Full suite run: `experiments/remediation/test_results.txt` (38 tests, OK).

## 5. Versions and hashes (`dataset/processed/VERSIONS.json`)

| Version | Status | `annotations_all.csv` sha256 | `rock_valid.csv` sha256 |
|---|---|---|---|
| curated-dataset-v1 | ARCHIVED, read-only, byte-identical after all D.2 runs | `764eff122451…` | `a62688ea448c…` |
| curated-dataset-v1.1 | CURRENT, provisional | `87213054874f…` | `b61e1e60257d…` |

**Idempotency:** re-running Task C v1.1 and then the v1.1 YOLO rebuild changed no content among 22,253 output files. The only differing file was `dataset_status.json`, which records file mtimes.

**Storage:** +242 MB (curated v1.1).

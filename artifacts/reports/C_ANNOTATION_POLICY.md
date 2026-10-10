# Annotation Policy S2-H/1.0 — `curated-dataset-v1`

**Class:** single class `rock` (id 0). **Images:** Artificial Lunar `render` images at their native 720×480 size.
**Status:** automated policy. **No annotation has been validated by a human.**

The policy is machine-readable in `dataset/processed/curated-dataset-v1/metadata/policy.json`. Calibration evidence is in `metadata/calibration.json`.

## 1. Scene eligibility

- **Eligible:** 9,193 frames.
- **Excluded:** 573 frames on any of the dataset authors' four anomaly lists (`cam_anomaly`, `mismatch`, `shadow`, `ground_facing`). Their reasons are in `manifests/excluded_frames.csv`.
- **Kept but flagged:** difficult scenes stay in the dataset. Their flags are in `manifests/scenes.csv`:

| Flag | Meaning |
|---|---|
| `very_dark` | Mean render < 40 |
| `heavy_shadow` | ≥ 40% of non-sky pixels are dark |
| `no_sky_ground_facing_like` | No sky visible |
| `palette_ground_mask` | Ground mask stored as a palette PNG |
| `review_heavy` | More REVIEW_REQUIRED boxes than VALID |
| `no_valid_annotation` | No VALID box in the scene |

## 2. Box sources

| `source` | Meaning |
|---|---|
| `csv_large` | Original `bounding_boxes.csv` instance, converted exactly: x0 = X − 0.5, y0 = Y − 0.5, x1 = x0 + Length, y1 = y0 + Height. The original line is kept in `csv_line`. |
| `mask_blue` | V1_cyan blue (large-class) mask component that does not duplicate a CSV box |
| `mask_green` | V1_cyan green (small/medium-class) mask component |
| `*_split` | Piece of a mask component separated by watershed; `parent_id` names the original component |

Coordinates are 0-based pixel edges, with x1 and y1 exclusive.

## 3. Statuses

- **VALID:** a training target.
- **REVIEW_REQUIRED:** **not** a training target and **not** background. A later step must treat these as ignore or uncertain regions until a human decides.
- **REJECTED:** not a target. The reason is always recorded.

## 4. Rules, applied in order

The same rules, with counts, are in `cleaning_decisions.csv`.

1. **CSV instances are always kept.** A CSV box becomes REVIEW_REQUIRED only when at least 85% of its pixels are darker than 15 **and** its non-sky surroundings average below 30, meaning the rock is invisible in black shadow.
   - A dark rock against lit ground is a visible silhouette: it stays VALID and gets the flag `dark_silhouette`.
2. **Minimum size:** mask components smaller than 50 px are not listed; they are only counted.
   - Evidence: in Task B, only 31% of 25–50 px components were visible rocks.
3. **Duplicates:** these mask components are REJECTED as `duplicate_of_csv_instance`:
   - a blue component at least 50% inside a CSV box, or with IoU ≥ 0.3 to one;
   - a green component with IoU ≥ 0.5 to a CSV box.
4. **Artefacts:** a box narrower or shorter than 4 px is REJECTED as `thin_artifact`.
5. **Invisible:** a component whose inside and surroundings both average below 15 is REJECTED as `invisible_black_shadow`.
6. **Mostly dark:** a component with at least 85% dark pixels becomes REVIEW_REQUIRED (`mostly_in_shadow`).
7. **Low visibility (calibrated):** if the boundary gradient is below 100 **and** the inside-versus-surrounding contrast is below 25, the component becomes REVIEW_REQUIRED (`ambiguous_low_visibility`).
   - Calibration used 78 rocks labelled in Task B: the rule keeps 90% of visible rocks and 15% of ambiguous ones.
   - A scene-relative version was also tested and scored worse (Youden J 0.60 vs 0.75).
8. **Merged rocks, distance transform + watershed (calibrated):**
   - **Markers:** distance peaks ≥ 0.35 × the maximum and ≥ 2.5 px.
   - **Split:** two basins are separated when the neck between them is ≤ 0.7 × the smaller peak. The parent is recorded as REJECTED `replaced_by_split_pieces`, and each piece is judged by rules 4–7.
   - **Borderline:** a neck in (0.7, 0.85], or solidity below 0.8 with no split, makes the component REVIEW_REQUIRED (`possible_merge_*`). It is never silently accepted as a single rock.
   - **Blue components ≥ 1000 px that were not split** become REVIEW_REQUIRED.
9. **Overlap:** a green box at least 90% inside a CSV box and at least 30% of its area becomes REVIEW_REQUIRED (`overlaps_csv_large_rock`).

## 5. Leakage control

- `manifests/scenes.csv` gives every frame a `leakage_group`. Exact duplicate renders share a group: frames 1 and 6658 form group D001.
- Any future split must keep each leakage group within a single partition.

## 6. Provenance

- **Per image:** `scenes.csv` records the SHA-256 of the render and the mask.
- **Run:** `provenance.json` records the code hash, the source data fingerprint, the inputs, the command and the seed.
- **To regenerate:**

```
cd backend
uv run --no-sync python -m analysis.dataset_cleaning
```

# Split Validation Report — `yolo-rock-v1` (AI-01.D)

**Seed:** 20261011. **Unit of assignment:** a *leakage group*, never a single frame.
**Data:** `split_statistics.json`, `dataset/processed/yolo-rock-v1/manifests/split_manifest.csv`.

## 1. Result

| Split | Frames | Share | Task C VALID | Exported labels | REVIEW boxes | VALID per image | Hard scenes | No-record frames |
|---|---|---|---|---|---|---|---|---|
| train | 7,353 | 79.98% | 209,545 | 209,430 | 154,335 | 28.5 | 287 | 121 |
| val | 923 | 10.04% | 25,834 | 25,822 | 19,417 | 28.0 | 37 | 16 |
| test | 917 | 9.97% | 26,006 | 25,987 | 19,074 | 28.4 | 34 | 16 |

The exported-label column excludes 146 duplicate split pieces; see the engineering report §3a.

- All 9,193 eligible frames are assigned exactly once.
- The 573 anomaly-listed frames are not exported.
- **NASA POLAR files in the export: 0.** NASA stays a separate, real-domain evaluation set for Task E.

## 2. Leakage control

**Frame-ID ordering**
- Neighbouring frame IDs are no more alike than random pairs: thumbnail correlation is 0.370 at lag 1 versus 0.364 for random pairs, and it is flat over lags 1–50.
- So random group assignment is valid; IDs do not encode terrain sequences.

**Why whole-image similarity is unusable**
- Visual inspection of the highest-correlation pairs (correlation 0.98–0.99) showed **different** scenes that only share horizon geometry.

**Near-duplicate detection used instead**
- **Candidates:** for each frame, its 15 nearest neighbours by gradient-detail and its 15 nearest by thumbnail correlation. That gives 232,990 pairs.
- **Verification:** ORB matching with RANSAC homography.
- **Calibration:**

| Pair | RANSAC inliers |
|---|---|
| Identical frames 1 / 6658 | 600 |
| The same frame shifted and zoomed | 217 |
| High-correlation but different scenes | 0–8 |
| Random pairs | max 5 |

- **Threshold:** 40 inliers counts as a duplicate. Pairs with 15–39 inliers are grouped conservatively.

**Results**

| Inliers | Pairs |
|---|---|
| ≥ 40 (duplicates) | 1: frames 1 / 6658 |
| 15–39 | 2: 4247 / 7643 (20) and 4363 / 5039 (16). Visually different scenes (repeated regolith texture), grouped anyway. |
| 5–14 | 4,484 |
| < 5 | 228,503 |

**Retrieval audit**
- 60 random frames were compared against all 9,193 frames: 551,520 pairs.
- The maximum was 12 inliers; no pair reached 15.
- So the candidate retrieval missed no duplicate in that sample.

**Leakage groups**
- Groups combine Task C's `leakage_group` with the pairs above, merged by union-find.
- 0 groups span more than one split. Frames 1 and 6658 are both in **train**.

## 3. Stratification

- **Strata:** scene difficulty (hard = `very_dark` or `heavy_shadow`) × VALID-density bin (0 / 1–15 / 16–35 / 36+) × uncertainty bin (REVIEW / (VALID + REVIEW) ≤ 0.25).
- **Balance:** every stratum with ≥ 40 frames is within 1.5 percentage points of 80/10/10.
- The largest deviation, 0.15, is a 4-frame stratum (hard, empty, high uncertainty), where integer rounding makes it unavoidable.
- **Distributions:** box size and boxes-per-image distributions overlap across splits (`figures/split_box_distributions.png`, `figures/split_strata.png`).

## 4. Limits

- Leakage control covers exact duplicates and same-terrain views. Renders that share procedural texture or style but not geometry are not grouped, and do not need to be.
- Val and test labels are **provisional**: automated and not human-verified. `manifests/eval_ignore_regions.json` lists the 38,491 REVIEW boxes on evaluation images, so Task E can exclude them instead of counting them as background.

# Dataset

## Sources

| Source | Content | Used for | Licence status |
|---|---|---|---|
| Artificial Lunar Landscape (Kaggle) | 9,766 synthetic scenes at 720×480: photorealistic *render*, semantic *ground* mask, *clean* mask; CSV of large-rock boxes; author anomaly lists; 36 real lunar photos (PCAM/TCAM) | Training, validation, test | **Not confirmed**: no licence file was found in the downloaded archive. Check the Kaggle page before redistributing anything derived from the images |
| NASA POLAR (3 terrains) | Real stereo images of a lunar-analogue testbed: 12-bit data in 16-bit PNG, exposure brackets, LiDAR | Exploration only. Excluded from every metric until a 12-bit → 8-bit conversion is validated | **Not confirmed**: no terms file was found locally |
| POLAR-Sim labels | YOLO boxes and polygons for POLAR (github.com/uwsbel/POLAR-Sim, commit `593ed65`) | Annotation reference | CC0 1.0 (`LICENSE`). The README still mentions BSD-3. Cite the POLAR-Sim paper |

**Obtaining the data** (not redistributed here):
1. Download the Artificial Lunar Landscape dataset from Kaggle and extract it to `dataset/artificial-lunar/` (expected:
   `images/{render,ground,clean}/`, `bounding_boxes.csv`, `*_IDs.txt`, `real_moon_images/`).
2. Optional: download the NASA POLAR terrains to `dataset/nasa-polar/Terrain*/`, and clone POLAR-Sim to `dataset/polar-sim-source/`.
3. To use another location, set `ARTEMIS_DATA_ROOT`.

Processed datasets are *derived* from the sources by the Task C/D code in `backend/analysis/`. Their content hashes are in
`artifacts/provenance/VERSIONS.json`.

## Versions

| Version | Status | Content |
|---|---|---|
| `curated-dataset-v1` | ARCHIVED | S2-H/1.0 annotations |
| **`curated-dataset-v1.1`** | **CURRENT** (provisional, not frozen) | S2-H/1.1: VALID 261,239 · REVIEW_REQUIRED 192,784 · REJECTED 50,005. Fixes duplicate rule D05 on watershed pieces |
| `yolo-rock-v1` | ARCHIVED | YOLO export of v1 |
| **`yolo-rock-v1.1`** | **CURRENT** (provisional, NOT frozen) | Config A (1,903 train), Config B (7,353 masked train), val 923, test 917, split and ignore manifests |

`yolo-rock-v1.1` reuses identical image files from v1 through NTFS **hard links**. Copying it with ordinary tools duplicates the data.

## Annotation policy (S2-H, summary)

| Step | Rule |
|---|---|
| CSV large-rock boxes | Kept as authoritative instances |
| Mask components | Added when ≥ 50 px |
| Merged blobs | Split by watershed (neck ratio 0.7) |
| Visibility | Edge ≥ 100 or contrast ≥ 25 |

**Statuses:**
- **VALID** boxes are training and evaluation targets.
- **REVIEW_REQUIRED** boxes are never background: they are excluded or masked in training and ignored in evaluation.
- **REJECTED** boxes are discarded.

**Exclusions:** 573 author-flagged anomaly frames, leaving 9,193 eligible frames. **Split:** grouped by near-duplicate clusters,
stratified 80/10/10, seed 20261011. The full policy is in `artifacts/reports/C_ANNOTATION_POLICY.md`.

## Known limitations

- **No human verification.** AI-assisted reviews (Task B verdicts, Task D.2 audit of 36 scenes) are recorded as automated
  evidence, and the human review (E.1b) was postponed.
- **Label artefacts.** Crater rims, cliff faces and shadows are sometimes boxed as rocks, and touching boulders are sometimes merged.
- **Out of scope.** Rocks below 50 px or invisible in shadow are not labelled.
- **Synthetic domain only.**

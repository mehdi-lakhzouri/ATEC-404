# Artemis-X — Dataset Exploration & Quality Assessment (AI-01.A)

**Date:** 2026-10-09 · **Status:** DONE · **Next task:** AI-01.B (needs approval)
**Code:** `backend/analysis/dataset_exploration/` · **Run:** `cd backend; uv run --no-sync python -m analysis.dataset_exploration` (~130 s)

Every number below comes from the generated JSON files in this folder. How sure each statement is:

- **[V] Verified**: measured on all files, or checked visually.
- **[I] Inferred**: supported by evidence but not proven.
- **[R] Recommendation**: a proposal, not a finding.

---

## 1. Executive summary

1. **[V] All three datasets are complete and readable.**
   - Every PNG decodes: 29,298 Artificial Lunar images and 1,224 NASA images, with 0 failures.
   - Every zip member exists on disk with the right size.
   - Running the analysis did not change any source data file.
2. **[V] Artificial Lunar is the only dataset that can train a detector now.** It contains 9,766 scenes in three aligned forms:
   - `render`: photorealistic grayscale image. This is the detector input.
   - `ground`: full semantic mask.
   - `clean`: the same mask with small rocks removed.
3. **[V] `bounding_boxes.csv` only covers large rocks.**
   - It was generated from the *blue* (large-rock) connected components of the `ground` mask.
   - 5,555 boxes match a component to the pixel. 83.2% match at IoU ≥ 0.5.
   - No box is smaller than 21 px in height or 441 px² in area.
   - Medium and small rocks (green) are never boxed: about 39 green components of ≥ 50 px per frame (median).
   - **This label incompleteness is the main training risk.**
4. **[V] The dataset authors' anomaly lists mark real defects** (573 frames, no overlap between lists). They should be excluded from training.
5. **[V] NASA images and POLAR-Sim labels can be matched without ambiguity.**
   - A strict 6-field key gives 627/1,224 detection matches, 0 ambiguous and 0 orphan labels.
   - Segmentation gives 592 matches and 2 ambiguous.
   - An independent stereo check confirms the left/right assignment (98.8% positive disparity).
6. **[V] NASA POLAR is useful for evaluation but highly redundant.**
   - Images are 12-bit data stored in 16-bit single-channel PNGs.
   - There are only 6 camera viewpoints per terrain.
   - In 93% of exposure brackets, the rock boxes are identical at every exposure.
7. **[V] Spatial localisation can use the LiDAR ground truth**, provided the camera–LiDAR transform is applied *inverted*. The literal reading puts 0% of points in the image.
8. **GO for Task B**, with the conditions in §11.

## 2. Dataset inventory (`dataset_inventory.json`)

| Directory | Files (excl. `.git`) | Size | Content |
|---|---|---|---|
| `artificial-lunar/` | 29,379 | 10.09 GiB | 3×9,766 PNG, 3 CSV, 5 ID lists, `archive.zip` (5.39 GB), 72 real-moon PNG |
| `nasa-polar/` | 1,239 | 5.91 GiB | 1,224 PNG, 3 `stereo.calibration`, 9 `.xyz`, 3 zips |
| `polar-sim-source/` | 7,972 | 0.14 GiB | 7,962 `.txt` labels, README, `yolo2bitmap.py`, demo `.cpp` |
| `polar-sim/`, `processed/` | 0 | — | empty |

Archive and repository checks:
- **Zips vs extracted files [V]:** `archive.zip` has 29,378 members; the NASA zips have 340, 448 and 448. None is missing on disk and none differs in size.
- **POLAR-Sim checkout [V]:** a second copy at `./polar-sim-source` (project root) is identical by path and size.
- **Partial clone [V]:** `dataset/polar-sim-source` is a partial clone (`blob:none`, sparse checkout of `Labels`, `SegmentationLabels`).
- **IDE downloads [V]:** during this session the IDE's git tooling (`.git/gk/`) downloaded 144 pack files into its `.git` (now 472 MB). Working-tree files did not change. *This was not done by the analysis.*
- **Licence:** the latest commit message reads "LICENSE changed from BSD-3 to CC0", while the README still says BSD-3.

Nothing required by the brief is missing. Not present: POLAR-Sim meshes, POLAR images for terrains 02–05, 07–10 and 12–13, and POLAR HDR files.

## 3. Artificial Lunar Landscape (`artificial_lunar_analysis.json`)

**Structure [V]**
- IDs 0001–9766 exist in all three folders, with no gaps and no naming violations.
- All images are 720×480 with 8 bits per channel.
- `render` is stored as RGB but is effectively grayscale (channels differ by ≤ 3 levels).
- `clean` is strictly binary (9,766/9,766 images).
- `ground` is RGB in 9,718 images and palette (`P`) mode in 48. Anti-aliasing produces cyan/magenta edge pixels.

**Semantics [V]**

| Folder | Content | Evidence |
|---|---|---|
| `render` | Photorealistic grayscale render | Visual check |
| `ground` | Mask: red = sky, green = small/medium rocks, blue = large rocks | Visual check |
| `clean` | `ground` with small rocks removed and colours binarised | Visual check |

- **Same scene across the three folders [V]:** sky IoU between `ground` and `clean` has a median of 0.999. 98.7% of ground-sky pixels are black in the render (median).
- **CSV schema [V]:**
  - Header is `Frame,TopLeftCornerX,TopLeftCornerY,Length,Height`, with 18,867 rows and 0 parse errors.
  - The Frame column is sorted and every value maps to an existing ID (`render%04d.png`).
  - Every X and Y value ends in .5. The 0-based left/top edge is X − 0.5, which is the MATLAB `regionprops` convention.
  - This convention gives a median offset of 0 px against the mask components.

**Box statistics [V]**

| Metric | Value |
|---|---|
| Boxes | 18,867 |
| Frames with ≥ 1 box | 6,208 |
| Frames with 0 boxes | 3,558 |
| Boxes per frame | median 1, p95 7, max 80 (frame 7281, which is in `cam_anomaly`) |
| Width | min 21, median 53, max 720 px |
| Height | min 21, median 34, max 480 px |
| Area | median 1,833 px² (0.53% of the image), p99 20% |
| Boxes touching the image border | 3,374 |

- No box is invalid, negative, zero-area, out of bounds, or an exact duplicate.
- No pair of boxes overlaps at IoU > 0.8.

**Label coverage [V]**
- Matching CSV boxes to `ground` blue components: 15,690/18,867 match (83.2%).
- 3,177 boxes in 1,577 frames do not match and are listed as *uncertain*.
  - In frames 11, 13 and 15 the boxes are on real rocks. The mismatch comes from anti-aliased mask edges splitting one rock into several components [V, sample only].
  - Other unmatched boxes are in anomaly frames.
  - How the 3,177 split between these two causes was **not** measured.
- Unboxed rocks:
  - 744 frames without any CSV box still contain blue components in `clean`.
  - 2,387 such frames contain green rocks of ≥ 50 px.
  - Distant large rocks below about 21 px are not boxed.
- **Conclusion [I]:** the CSV is "large rocks above a size threshold". A frame with no boxes does **not** mean the frame has no rocks.

**Anomaly lists [V]**
- The four quality lists do not overlap each other. Together they cover 573 frames.

| List | Frames | Visual finding (4 sampled frames each) | Metric support |
|---|---|---|---|
| `cam_anomaly` | 19 | Corrupted masks: sky labelled as rock, foreground rock as sky, speckle; up to 80 bogus boxes | Only 61% of render-black pixels fall in labelled sky (unlisted frames: 96%) |
| `mismatch` | 278 | Render geometry does not match the masks (horizon and boxes off-target) | 46% of render-black pixels fall in labelled sky (unlisted frames: 96%) |
| `shadow` | 226 | Rocks boxed inside fully black shadow | Dark fraction outside the sky 0.79 (unlisted: 0.014) |
| `ground_facing` | 50 | Camera faces the ground or a rock face; no horizon | Sky fraction 0.001 |
| `top200_largerocks` | 200 | Selection list, not a defect list | 165/200 are also in the top 200 by largest CSV box |

**Other findings**
- **Near-duplicate renders [V]:** dHash finds 10 exact collisions, for example frames 1 and 6658, which are visually identical. Pairs at Hamming distance 1–3 (376 in total) are false positives on mostly black frames.
- **Recommended detector input [R]:** `render` only. `ground` and `clean` are labels, never model inputs.

**`real_moon_images/` (not in the brief) [V]**
- 14 PCAM and 22 TCAM real lunar colour photos (source not documented in the folder), each with a `g_*` mask.
- `PCAM_bounding_boxes.csv`: 143 boxes (78 green, 65 blue). `TCAM_bounding_boxes.csv`: 181 boxes (not analysed in detail).
- 6 images differ in size from their mask (for example TCAM1 is 2352×1728 but its mask is 777×571).
- Annotation is sparse (PCAM7 has unboxed rocks).
- This is the only *real-lunar* evaluation material available.

## 4. NASA POLAR (`nasa_polar_analysis.json`)

| | Terrain01_LargeOnly | Terrain06_Smooth | Terrain11_FreshCrater |
|---|---|---|---|
| Images | 336 | 444 | 444 |
| Position + light folders | 5 (PosC_Lon missing) | 6 | 6 |
| Exposures (ms, from filenames) | 16–1024 (7) | 32–2048, plus 4096/8192 for NoSun | same as T06 |
| Brackets × size | 48×7 | 48×7 + 12×9 | 48×7 + 12×9 |
| Camera viewpoints (position × L/R) | 6 | 6 | 6 |
| LiDAR plane-fit residual p99 (A/B/C) | 9.0 / 9.6 / 8.8 cm | 4.1 / 3.6 / 9.2 cm | 16.6 / 17.7 / 16.9 cm |
| POLAR-Sim detection labels | 280 files, 3,899 rock / 3,224 shadow boxes | none | 347 files, 2,230 rock / 1,451 shadow boxes |

**Images [V]**
- All images are `I;16` single-channel PNGs at 1936×1216, with values 11–4094 (12-bit effective) and no embedded metadata.
- Stereo pairs are complete: every left image has a right image with the same timestamp and exposure.
- No exposure repeats within a bracket.
- Lighting:
  - Sun azimuths are 30, 180, 270 and 350, plus NoSun.
  - The rover light is on or off.
  - NoSun with light off is nearly black (median value 17).
  - Sun 30 and Sun 350 are brightest. Sun 180 shows strong lamp flare in the frame.
- The best exposure (largest share of well-exposed pixels) is 512–2048 ms. Exposures of 16–64 ms have < 1% well-exposed pixels.
- The 4096/8192 ms points in `figures/nasa_exposure_response.png` look dark only because those exposures exist only for NoSun shots.

**Redundancy [V]**
- Adjacent exposures in a bracket correlate at a median of 0.99 or more (log thumbnails).
- Different sun azimuths of the same view correlate at about 0.46.
- Each bracket is therefore one scene. The meaningful variation is lighting.

**Scene domain [V]**
- These are indoor lab sandbox images: walls, curtains, lamp flare and, at PosB, a tripod with a light.
- They differ strongly from the Artificial Lunar renders and from real lunar photos.

**Calibration [V]**
- Values are identical in all three terrains except `CAM_LIDAR_*`, which differs for T06 and T11.
- Left camera: fx = 2068.4, fy = 2064.3, cx = 964.4, cy = 593.5.
- The calibrated image size matches the images. Distortion is k1 = −0.108, k2 = 0.158.
- Field of view is 50.2° horizontal and 32.8° vertical.
- Stereo baseline is 0.302 m (if the units are metres), with a 0.15° relative rotation.
- Disparity is 156 px at 4 m and 78 px at 8 m.

**Ground truth [V/I]**
- One ASCII `.xyz` per position: 0.70–4.59 million rows, 6 columns, no NaN.
- Columns 3–5 are XYZ in metres [I].
- Columns 1, 2 and 6 are integers; their meaning is **unresolved**. Possibly scan indices and intensity.
- Projection test:
  - Xc = Rᵀ(X − t) puts 100% of points in front of the camera and 81–100% inside the image. Median depth is 3.3 m (PosA/C) and 6.1 m (PosB).
  - Xc = R X + t puts 0% inside the image.
  - Overlays (`samples/nasa_lidar_projection_*.png`) show the crater depth structure matching the image.
- Ground truth is per *position*, not per image. It suits depth and position evaluation of detections; it does not provide per-rock identities.

**Terrain role hypotheses**
- **T01 "rock-rich" [V]:** supported by 3,899 rock boxes.
- **T11 "crater" [V]:** supported by the largest LiDAR residuals and a visible crater.
- **T06 "negative" [I]:** plausible but **unverified**:
  - Visually smooth sand with footprints, small residuals at PosA/B.
  - No detection labels exist for it.
  - The POLAR-Sim demo code uses `rock_num = 0` for terrain 6.
  - Its 6 segmentation files contain only background and ground polygons.

## 5. POLAR-Sim annotations (`polar_sim_analysis.json`)

**Detection (`Labels/`) [V]**

Files in the folder:

| File type | Count |
|---|---|
| Canonical label files | 3,872 |
| `classes.txt` (metadata, not annotations) | 12 |
| `.DS_Store` | 6 |
| Variant copies (`… copy.txt`, `…good.txt`, `…txt.txt`) | 101 |

- Variant copies: 6 differ from their canonical file. None of the 6 is in T01 or T11.
- There is no `Terrain06` folder.
- Classes:
  - All 12 `classes.txt` files read `rock, shadow`, so 0 = rock and 1 = shadow, matching the README.
  - Totals: 68,756 rock and 49,621 shadow boxes.
- Malformed files:
  - Exactly 1: `Labels/Terrain09/09_B_off_180_R_0032.txt`, line 1, where a filename is concatenated into the class token.
  - 1 empty file: `Labels/Terrain11/11_A_on_180_L_1024.txt`.
  - No out-of-range class, no box outside the image, no duplicate line.
- Rock box sizes at 1936×1216: median 53×37 px (2.7% of image width), minimum 10×7 px.
- **Labels reused across exposures [V]:**
  - In 543 of 581 brackets, the rock boxes are identical at every exposure.
  - Mean rock count is 17.8 per file at every exposure from 32 to 2048 ms.
  - So labels were made once per scene (HDR) and copied to every exposure, including frames where the rocks are invisible.

**Segmentation (`SegmentationLabels/`) [V]**
- 3,977 files give 3,972 unique keys.
- There is no class metadata file. The mapping 0 = background, 1 = ground, 2 = rocks, 3 = shadows comes from the README and `yolo2bitmap.py`, and class IDs 0–3 are observed.
- Polygons: 76,763 rocks, 70,175 shadows, 8,158 background and 7,645 ground.
- Polygons cover the image: median unlabelled area is 0%, p99 is 1.1%.
- Defects:
  - 28 degenerate polygons (< 1 px²), e.g. `terrain02/labels/02_A_off_270_R_*.txt` line 14.
  - 5 files are in another terrain's folder.
  - 3 keys are duplicated with **different** content, e.g. `11_B_off_no_L_2048` exists in both `terrain07/` and `terrain11/`.
- **Feasibility [I]:** good for T01 (182 files) and T11 (406 files), with overlays aligning visually. Not usable for T06 (6 files).

## 6. Dataset matching (`annotation_matching_report.json`)

**Algorithm [V]**
1. Parse the NASA path strictly into the key `TT_P_on|off_az|no_L|R_exp`.
2. Index only canonical POLAR-Sim filenames by the same key.
3. Never use variant copies as a primary match.
4. All six fields must agree. Exposure alone is never used.
5. The key is unique: 1,224 keys for 1,224 images.

Example: `Terrain11_FreshCrater/PosA_1500_Loff/Sun_270/CamL_281395797_1024.png` ↔ `Labels/Terrain11/11_A_off_270_L_1024.txt`.

| | Unambiguous | Ambiguous | NASA image without a label file | Labels without an image (T01/T06/T11) |
|---|---|---|---|---|
| Detection | 627 (T01: 280, T11: 347) | 0 | 597 (T06: all 444, T01: 56, T11: 97) | 0 |
| Segmentation | 592 | 2 (11_B/11_C NoSun 2048 duplicates) | 630 | 0 |

- Labels for the 10 terrains not downloaded are out of scope: 3,245 detection and 3,378 segmentation files.
- **Images without a label file** are, according to the README, mostly dark images ("nothing visible"). They are recorded as *unknown*, **not** as negatives.
- **Validation [V]:**
  - 2,684 rock-box pairs between L and R label files of the same scene (T01/T11) were compared.
  - 98.8% have positive disparity, with a median of 155 px.
  - The implied depth is median 4.0 m (p5–p95: 2.7–6.6 m), consistent with LiDAR (median 3.3–6.1 m).
  - Across all 12 terrains, 95% of pairs are positive.
  - Visual overlays (`samples/match_*.png`) align.

## 7. Quality issues (`data_quality_report.json`: 0 critical, 4 high, 8 medium, 8 low)

| ID | Severity | Issue |
|---|---|---|
| AL-01 | HIGH | CSV boxes only cover large rocks (≥ 21 px); medium and small rocks are unlabelled |
| NP-01 | HIGH | NASA images are 12-bit single-channel; they need an explicit conversion to 8-bit before YOLO |
| NP-02 | HIGH | Only 6 viewpoints per terrain; brackets and lighting inflate the count of "independent" images |
| PS-05 | HIGH | Rock boxes are copied across exposures, so dark frames have boxes on invisible rocks |
| AL-02 | MEDIUM | 3,177 CSV boxes with no mask match (mostly a fragmentation artefact; split not measured) |
| AL-04 | MEDIUM | 573 anomaly-listed frames with real render or mask defects |
| NP-04 | MEDIUM | LiDAR is per position; the camera–LiDAR extrinsics must be inverted |
| PS-01 | MEDIUM | 101 copy/variant files, 12 metadata files and 6 `.DS_Store` files mixed into the label folders |
| PS-02 | MEDIUM | 1 malformed line (Terrain09) |
| PS-04 | MEDIUM | No detection labels for T06; its negative status is unverified |
| MT-01 | MEDIUM | 597 NASA images without labels must not count as negatives |
| HK-03 | MEDIUM | 5 logged warnings (segmentation duplicate keys) |
| AL-03, AL-05, AL-06, NP-03, PS-03, PS-06, HK-01, HK-02 | LOW | Geometric validity (clean), real-moon size mismatch, 10 duplicate renders, missing T01 PosC_Lon, 28 degenerate polygons, misfiled segmentation files, duplicate checkout, empty directories |

## 8. Dataset suitability

| Question | Answer |
|---|---|
| 1. Which dataset to train on first? | **Artificial Lunar** [V]. It is the only one with images and labels at scale (9,193 frames after exclusions, 17,685 CSV boxes, 5,880 frames with boxes). |
| 2. Which image representation? | **`render`**, converted to 1-channel or 3× replicated gray [V]. Inference on NASA images needs a fixed 12→8-bit mapping [R]. |
| 3. Rocks only, or rocks and shadows? | **Rocks only** [R]. Artificial Lunar has no shadow class. POLAR-Sim shadows can only be used for evaluation. |
| 4. Are the boxes reliable enough to fine-tune? | **Yes, for the "large rock" definition** [V]: geometrically clean and exactly derived from the mask. **No, as "all rocks"** [V]. Decide the target definition in Task B. |
| 5. Can NASA POLAR be an independent test set? | **Yes, with conditions** [V]: T01 + T11 have 627 matched images. Evaluate one well-exposed exposure per bracket and report per terrain and position. The domain is a lab sandbox. |
| 6. Which annotations support segmentation? | POLAR-Sim segmentation for T01/T11 (588 files, 4 classes) [V]. Artificial Lunar `ground`/`clean` masks for synthetic training (sky / small rock / large rock) [V]. |
| 7. What supports spatial localisation? | `stereo.calibration` (intrinsics, distortion, 0.30 m baseline) and `PosA/B/C.xyz` with the inverted extrinsics [V]. Stereo depth from box disparity already agrees with LiDAR to within metres [V]. |
| 8. Synthetic-to-real risks | [I] Renders vs lab sandbox (walls, flare, tripod, curtains), 8-bit vs 12-bit, different relative rock scale (Artificial Lunar median box 7.4% of width vs POLAR-Sim 2.7%), black sky vs lab background, shadow appearance, and the fact that Artificial Lunar boxes only large rocks while POLAR-Sim boxes all rocks. |
| 9. How to handle near-duplicates? | NASA: one exposure per bracket, never split a bracket or a position across train and test [R]. Artificial Lunar: keep exact duplicates (e.g. 1 and 6658) in the same split [R]. |
| 10. Top data-quality problems | AL-01 (label definition), PS-05 (dark-exposure boxes), NP-01 (bit depth), NP-02 (redundancy), AL-04 (anomaly frames). |

## 9. Recommended training strategy [R]

- **Training set:**
  1. Start from the Artificial Lunar `render` images.
  2. Exclude the 573 frames in the four quality lists.
  3. Keep the shadow frames separately as a hard-case set.
- **Detection target, pending Task B review.** Two options:
  - **(a)** "Large rock" from the CSV: fastest, consistent, honest.
  - **(b)** Boxes from `ground` mask components (blue plus green above a size threshold), to match POLAR-Sim's "all visible rocks".
  - Option (b) better fits the obstacle-avoidance goal and the evaluation labels. It needs component merging because of the anti-aliased edges.
- **Split:**
  1. Random by frame ID with a fixed seed: 80/10/10.
  2. Stratify by box count (0 / 1–3 / ≥ 4).
  3. Keep exact dHash duplicates together.
  4. Do not use NASA images for training. They are the test set.
- **Test sets:**
  - NASA T01 + T11, unambiguous matches only, one exposure in 512–2048 ms per bracket. That gives 92 images: 46 lit scenes × 2 cameras.
  - Metrics reported per terrain, position and lighting.
  - T06 as a false-positive probe once Task B confirms it has no rocks.
  - Real-moon PCAM/TCAM as a qualitative real-lunar check.
- **Preprocessing:** a documented 12→8-bit mapping for NASA images, either a fixed scale (÷16) or a per-image percentile stretch. Compare the two in Task C.

## 10. Remaining uncertainties

- What fraction of the 3,177 unmatched CSV boxes are mask-fragmentation artefacts versus real errors? Only 3 cases were inspected.
- T06 contains no rocks: likely, not confirmed.
- NASA images without a label file: unlabelled because nothing is visible, or simply never annotated?
- The `.xyz` integer columns 1, 2 and 6, and the meaning of the position suffixes `_1500` / `_4000`.
- The units of the exposure values in NASA filenames (the README says ms).
- The relation of the real-moon set to the Kaggle release; TCAM boxes were not analysed.
- Whether the dark-exposure boxes in PS-05 still cover visible rocks after display stretching. They appeared visible at 16 ms in one sample.

## 11. GO / NO-GO for Task B — **GO**

Conditions for Task B (AI-01.B Annotation Visualisation & Human Validation):
1. Review AL-01 and pick the target definition: CSV large rocks, or mask-derived boxes.
2. Review a stratified sample of the 3,177 uncertain CSV boxes.
3. Confirm T06 contains no rocks.
4. Check the PS-05 dark-exposure boxes.
5. Decide whether to fix the IDE fetching blobs into `dataset/polar-sim-source/.git` (e.g. by disabling git auto-fetch for that folder).

No blocking (critical) issue was found.

## Deliverables and reproducibility

- **JSON:** `dataset_inventory.json`, `artificial_lunar_analysis.json`, `nasa_polar_analysis.json`, `polar_sim_analysis.json`, `annotation_matching_report.json`, `data_quality_report.json`.
- **Figures:** `figures/` (10). **Samples:** `samples/` (19 sheets, more than 60 frames inspected visually). **Log:** `logs/run.log`.
- **Determinism:** two consecutive runs produced byte-identical analysis JSONs (4 of 6) and all 29 PNGs.
  - `dataset_inventory.json` and `data_quality_report.json` differ only in the `.git` metadata counts that the IDE changes.
  - The data fingerprint (`fecca0cf…`, 38,590 files) was unchanged before and after each run.
  - No file under `dataset/` (outside `.git`) has a modification time later than the first run.

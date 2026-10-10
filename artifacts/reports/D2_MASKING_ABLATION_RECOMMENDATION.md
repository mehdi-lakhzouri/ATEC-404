# Config B Masking Audit and Recommendation

**Data:** `masking_audit/masking_audit_summary.json`, `masking_audit_per_image.csv` (7,353 v1.1 training images), `fill_comparison.jpg`.
**Nothing was replaced.** Config B is unchanged and **not scientifically validated**.

## 1. Measurements (full training export)

| Check | Result |
|---|---|
| Modified pixels per image | median 2.4%, mean 3.8%, p95 12.4%, p99 19.8%, max 44.6% |
| Images ≥ 5 / 10 / 15 / 20 / 25% masked | 1,992 / 645 / 207 / 72 / 23 |
| Foreground (bottom-third) masked fraction | median 2.9%, p95 19.9% |
| Pixels changed inside VALID boxes | **0** (2 px protection margin works) |
| Pixels changed outside the mask | **0** |
| Texture retained in filled areas (mean \|Laplacian\| filled / original) | **median 0.11** (p95 0.18); 6,866 of 6,877 measurable images < 0.5: the fill is a smooth smear |
| REVIEW boxes erased | 149,975 |
| …large (≥ 1,000 px²) | **21,553, in 5,044 images (69% of train)** |
| …flagged as possible merges (mostly real rock groups) | 38,246 |

**Audit bug found and fixed:**
- **The bug:** the first pass read images through OpenCV's *grayscale* path. That path applies the source PNGs' `gAMA` chunk, which config B files lack, so it reported large false "changes".
- **The fix:** reading in colour confirms 0 changed pixels.
- **Residual fact:** config B PNGs carry no `gAMA` / `cHRM` chunks. Ultralytics loads in colour and sees identical pixels, but gamma-aware viewers may display them slightly differently.

## 2. Problems confirmed

1. **Unnatural texture.** Every masked region becomes a low-texture smear unlike anything in evaluation images. A model can learn "smear = no rock".
2. **Recognisable rocks are erased.** REVIEW includes many visible rocks (see the remediation report). B removes them from the image instead of labelling them. 0977 and 8894 lose their most prominent foreground obstacles.
3. **Shadow fragments.** A rock is masked but its shadow is not (frames 7449, 5623).

## 3. Alternatives evaluated

| Option | Evidence | Pros | Cons |
|---|---|---|---|
| a. Cap the masked fraction (drop scenes) | Cap trade-off below | Removes the worst smears (0977 at 29%) | Smears remain in kept scenes; drops dense, rocky scenes |
| b. Noise-matched fill (prototype; Telea + re-injected ground texture) | Texture ratio 0.06–0.10 → **0.63–0.96** on 8 worst frames (7449 overshoots to 1.9); see `fill_comparison.jpg` | Much less smear | Low-frequency blur and shadow remnants stay; still **erases real rocks**; not validated |
| c. **Ignore-aware loss** (no image editing) | Ultralytics `v8DetectionLoss.get_assigned_targets_and_loss` exposes `fg_mask` / `target_scores`; anchors whose centres fall in REVIEW boxes and are not assigned to a VALID box can be excluded from the classification loss | Images untouched (no train/test shift); REVIEW is neither positive nor negative | Needs a custom trainer (~150 lines plus tests); not implemented |
| d. Config A (conservative scenes) | 1,903 scenes, 12,161 REVIEW boxes still treated as background | Simple, no artefacts | Discards 74% of training scenes |

**Cap trade-off** (train split):

| Cap | Scenes kept | VALID boxes kept |
|---|---|---|
| 5% | 5,362 | 129,255 |
| 10% | 6,709 | 185,353 |
| 15% | 7,146 | 202,510 |
| 20% | 7,281 | 207,353 |
| 25% | 7,330 | 208,596 |
| none | 7,353 | 209,430 |

## 4. Recommendation (for approval)

1. **Do not use Config B as-is as the main arm.** Keep it only as an ablation, with a **15% masked-fraction cap**: it drops 207 scenes (2.8%) and 6,920 VALID boxes (3.3%), and removes 0977-type cases.
2. **Implement the ignore-aware loss (option c) as arm C in Task E**, with tests on synthetic batches before any training. It is the only option that neither alters images nor treats REVIEW as background.
3. The noise-matched fill is **not** recommended for adoption without a controlled comparison. It is a prototype.
4. Planned comparison: A vs B (capped) vs C, on the same ignore-aware evaluation (`EVALUATION_PROTOCOL_REPORT.md`).

# Artemis-X — Annotation Validation & Strategy Selection (AI-01.B)

**Date:** 2026-10-09 · **Status:** DONE, pending human review · **Recommendation:** HYBRID, target ontology `rock`, confidence medium
**Training authorisation:** NOT GRANTED · **Next task:** AI-01.C (requires explicit approval)

**Code:** `backend/analysis/annotation_validation/` (reuses Task A's `dataset_exploration` parsers, matching and fingerprinting).

**Run** (about 130 s; deterministic):

```
cd backend
uv run --no-sync python -m analysis.annotation_validation --workers 8
```

Every visual verdict below comes from the **AI assistant's own visual review**, kept in `assistant_review/*.csv`. **None of it is human validation.** All human fields in `human_review_manifest.csv` are `NOT_REVIEWED`.

Tags: **[F]** full dataset, automated; **[R]** sampled review by the AI assistant.

## 1. Summary

1. **S1 (CSV) is clean but answers the wrong question.**
   - It labels *physically* large rocks (blue renderer class, ≥ ~21 px).
   - Visible nearby rocks of the "small" class are unlabelled, even when they look bigger than labelled rocks.
   - 28,519 such green components exceed the median CSV box area [F].
   - S1 misses clearly visible rocks in 41/62 assessed candidate-subset scenes, and partially in 13 more [R].
2. **S2 (mask-derived) adds the missing rocks but has quality problems.**
   - Mask components merge touching rocks: 9/10 large blue components without a CSV box were merged groups [R].
   - Components under 50 px are mostly not identifiable: 5/16 visible [R].
   - Among additional candidates ≥ 50 px, 58% are clean visible rocks and 19% are merged groups [R].
3. **The 3,177 "uncertain" CSV boxes are mainly a matching artefact.** 3,034 cover part of a larger merged mask region [F]. In clean frames, 16/28 sampled boxes are valid, 2 are wrong and 10 are undeterminable [R]. Real errors concentrate in anomaly-listed frames.
4. **Terrain 06 is an UNVERIFIED_NEGATIVE.** Exclude it from quantitative false-positive rates.
5. **Dark NASA exposures:** POLAR-Sim boxes are recoverable with contrast enhancement but not with a fixed 12→8-bit view. 1024–2048 ms exposures clip sunlit rocks. Exposure-selection policy: 92 images, with 96.5% of rock boxes visible in the fixed view [F].
6. **Decision: HYBRID.** Train an S1 baseline first. Build S2-H (CSV for large + filtered, split green/blue components for medium) in AI-01.C. Evaluate both on the `rock` ontology with size-stratified recall. Details: `STRATEGY_COMPARISON.md`.

## 2. Inputs reused and integrity

- **Reused from Task A:**
  - the CSV half-pixel convention, anomaly lists and palette-mask handling;
  - the NASA image index and strict six-field matches (`annotation_matching_report.json`);
  - the POLAR-Sim parsers and the source fingerprint.
- **Not changed:** no dataset or Task A report was modified, and no dependency or config changed.
- **Source data fingerprint:** `fecca0cf…` (38,590 files), identical before and after every run and identical to Task A's.
- **Determinism:** two consecutive runs produced 560 byte-identical output files (JSON, CSV, JPEG, PNG, HTML).

## 3. Sampling (B1)

- `sampling_manifest.csv`: 117 distinct scenes; 88 from the candidate subset, 29 anomaly-listed. Seed 20261009.
- Strata (primary category): random 14, zero-CSV 8, one-rock 8, multiple-rocks 8, high-density 8, small-near-large 6, touching 6, very-dark 6, heavy-shadow 6, border 6, fragmentation 6, uncertain-match 8, cam_anomaly 5, mismatch 6, shadow 6, ground_facing 4, exact-duplicate pairs 6.
- Every frame also records all the other strata it qualifies for, plus a numeric rationale.

## 4. S1 visualisation (B2)

- **Per scene:** `s1_csv_overlays/fNNNN_s1.jpg` (render + CSV) and `fNNNN_raw_s1_ground.jpg` (raw | S1 | ground mask). Native 720×480, no resizing, JPEG quality 92.
- **Raw images** are referenced at their dataset paths in the manifest rather than copied.
- **Boxes** use x0 = X − 0.5 and y0 = Y − 0.5, with the right/bottom edge exclusive.

Findings [R]:
- **Correct:** boxes sit tightly on large rocks and separate touching boulders (0742, 0977).
- **Incomplete:** visible medium rocks, including 60–100 px foreground rocks, are not boxed (0676, 2042, 2875, 4044, 8570, 8894). 5 of the 8 zero-CSV frames contain clearly visible rocks.
- **Difficult:** boxes over rocks hidden in black shadow even in frames not on the shadow list (4191, 5409, 7220, 6133).

## 5. S2 extraction (B3)

- **Classes:** pixel channels > 127. Red = sky; blue = large rock; green = small/medium. Cyan (blue+green anti-aliasing) is assigned to large rock in V1.
- **Masks:** the 48 palette-mode masks are converted to RGB.
- **Visibility proxy:** |mean(rock) − mean(surrounding ground)| ≥ 6 grey levels, and the region must not be dark (both < 15).
- **Variants [F]:** see `STRATEGY_COMPARISON.md` §3. V1 (cyan → large, no closing) is chosen: closing cuts splits but merges more rocks and lowers CSV agreement.
- **Thresholds [F], candidate subset, additional candidates per image:**

| Min area | Green | Blue without CSV |
|---|---|---|
| ≥ 25 px | 60.1 | 5.2 |
| ≥ 50 px | 41.3 | 3.9 |
| ≥ 100 px | 26.4 | 2.5 |
| ≥ 200 px | 15.5 | 1.2 |

- **Chosen threshold: 50 px.** Visible-rock rate is 31% at 25–50 px versus 81% at 50–100 px [R].
- **Invalid geometry:** none. Connected-component boxes are valid by construction; components under 10 px are counted but not stored.
- **Output:** candidates are provisional, kept in memory and in overlays only. No YOLO files were written.

## 6. Mandatory comparison (B4)

Generated outputs:
- `comparison_sheets/scenes/fNNNN_compare.jpg` (117 scenes): A raw | B S1 | C S2 | D diagnostic, with a legend.
- 33 contact sheets of 5 scenes each.
- `gallery/index.html`: a static page with category, anomaly and assistant-flag filters.

Box styles (colour plus line style plus tag):

| Tag | Colour / line | Meaning |
|---|---|---|
| C | cyan, solid | CSV box |
| C? | orange, dashed | CSV box with no S2 match |
| L | blue, thin | S2 large rock matching CSV |
| L+ | purple, dashed | S2 large rock without CSV |
| S | green, thin | S2 small/medium rock |
| X | red, dotted | Rejected candidate |

Assistant answers over 62 candidate-subset scenes [R]:
1. **Does S1 miss relevant rocks?** Yes 41, partial 13, no 8.
2. **Does S2 recover them?** Yes 45, partial 12, n/a 5.
3. **Are S2 boxes tight?** Mostly 55, yes 5, no 2.
4. **Does S2 split single rocks?** Not observed at scene scale (0/62).
5. **Does S2 merge adjacent rocks?** Yes in 18/62, mostly horizon rows and touching boulders.
6. **Does S2 produce segmentation artefacts?** In 10/62, mostly in dark terrain.
7. **Are small rocks visible?** Yes 40, partial 17, no 5. Below 50 px they mostly are not (§5).
8. **Do very small rocks matter?** Not as targets at 720×480: under 50 px they are mostly unidentifiable.
9. **Which matches the evaluation ontology?** S2/S2-H; POLAR-Sim labels all visible rocks.
10. **Is quality sufficient for training?** S1 yes as `large_rock`. Raw S2 no; S2-H after the split step needs to be re-measured.

Preferred per scene: S2 in 42, equal in 13, S1 for the large rocks in 4, S1 in 3. All 10 assessed anomaly scenes: neither.

## 7. Uncertain CSV boxes (B5)

**Automated triage of all 3,177 [F]:**

| Mechanism | Boxes |
|---|---|
| Mask component extends beyond the box | 3,034 |
| Fragmented mask | 54 |
| Mask pixels present, other | 41 |
| No rock pixels in box (all in anomaly frames) | 32 |
| Matched once cyan pixels count as large rock | 12 |
| Partial mask support | 4 |

The full list is in `uncertain_boxes/uncertain_all_auto_triage.csv`.

**46 reviewed [R]:**

| Stratum | Result |
|---|---|
| Clean frames (28) | 12 valid / matcher failed, 4 valid / imperfect mask, 2 annotation errors (partial or offset box), 10 undetermined |
| mismatch (6) | 6 frame mismatch |
| cam_anomaly (6) | 5 annotation errors (corrupted mask), 1 valid / imperfect mask |
| shadow (6) | 6 undetermined (render fully black) |

**Remaining uncertainty:** the error rate in clean frames is not estimable from 28 items. No box was modified.

## 8. NASA (B6, B7)

**Terrain 06 → UNVERIFIED_NEGATIVE.** 18 images were reviewed: positions A/B/C × L/R × three lighting conditions.

| Result | Images | Notes |
|---|---|---|
| Apparently rock-free | 8 | |
| Rock-like or ambiguous | 2 | PosC sunlit: small dark clods |
| Unusable | 8 | No-sun/no-light images are noise; PosB rover-light shows almost no floor |

- PosB is dominated by a lamp tripod, which makes it a useful *qualitative* false-positive probe.
- All six Terrain 06 segmentation files contain only background and ground polygons.
- Do not compute quantitative false-positive rates on Terrain 06.

**Dark exposures.** Visualisation modes:
- **Fixed:** `raw >> 4`, clipped to 255.
- **Enhanced:** per-image 0.5–99.5 percentile stretch, then gamma 1/2.2.

Automated proxy over all 6,129 rock boxes on the 627 matched images [F]: share of boxes visible in the fixed view.

| Exposure | Visible in fixed view |
|---|---|
| 16 ms | 5% (90% visible only after enhancement) |
| 32 ms | 26% |
| 64 ms | 70% |
| 128 ms | 92% |
| 256 ms | 96% |
| 512 ms | 98% |
| 1024 ms | 99% |
| 2048 ms | 91% |

Review of two brackets [R]:
- 16–64 ms: rocks are identifiable only after enhancement.
- 128–512 ms: clean.
- 1024–2048 ms: sunlit rocks saturate.

**Exposure-selection policy:**
- For each labelled bracket, take the 512/1024/2048 ms image with the largest well-exposed fraction.
- Result: 92 images (Terrain 01: 40, Terrain 11: 52), with 96.5% of rock boxes visible in the fixed view.
- Brackets lit with no sun and no light are excluded.
- Boxes on invisible rocks are never scored as fair false negatives.

## 9. Metrics (B8)

- `strategy_metrics.json` separates `full_dataset_automated` [F] from `sampled_review` [R].
- Every rate states its denominator (reviewed items per stratum).
- Strata are not population-weighted, and no confidence intervals are claimed.

## 10. Decision (B9)

See `strategy_decision.json` and `STRATEGY_COMPARISON.md`.

- **Recommended strategy:** HYBRID.
- **Ontology:** `rock`; the S1 baseline is explicitly `large_rock`, evaluated with size-binned recall.
- **Confidence:** medium.
- **Approved for training:** false.

## 11. Limitations and open questions

- The visual review is the AI assistant's work at display resolution; small-rock calls on zoomed crops are uncertain.
- The visibility proxy is weak: of 111 candidates it marked "ok", only 59% were visible rocks.
- The merged-component split step for S2-H is untested.
- The effect of a 24× denser target on YOLO is unmeasured.
- NASA exposure findings rest on an automated contrast proxy plus 2 reviewed brackets.

## 12. Required human reviews

1. Approve or reject the ontology `rock` (versus `large_rock`).
2. Review at least 20 scenes in `gallery/index.html` and fill in `human_review_manifest.csv`, which has 424 items, all NOT_REVIEWED.
3. Spot-check the candidate crops to confirm the 50 px threshold.
4. Confirm the Terrain 06 status.
5. Approve the NASA exposure policy and the 12→8-bit conversion.
6. Authorise AI-01.C.

## Deliverables

| Path | Content |
|---|---|
| `ANNOTATION_VALIDATION_REPORT.md`, `STRATEGY_COMPARISON.md` | Reports |
| `sampling_manifest.csv`, `human_review_manifest.csv` | Manifests; human-edited manifests are never overwritten (`.generated.csv` instead) |
| `strategy_metrics.json`, `quality_findings.json`, `strategy_decision.json` | Metrics, findings, decision |
| `s1_csv_overlays/` (234), `s2_mask_overlays/` (117 + 5 candidate sheets) | Per-scene overlays |
| `comparison_sheets/` (117 scenes + 33 contact sheets) | A/B/C/D comparisons |
| `uncertain_boxes/` (2 sheets + 2 CSV) | Uncertain-box review |
| `nasa_terrain06/` (4 + 18 full-resolution) | Terrain 06 review |
| `nasa_exposure_review/` (12) | Exposure brackets and crops |
| `figures/` (2), `gallery/index.html` | Charts and review gallery |
| `assistant_review/` | AI-assistant assessments; pipeline inputs, not human review |
| `logs/run.log`, `logs/integrity.json` | Run log and integrity check |

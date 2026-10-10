# S1 vs S2 — Annotation Strategy Comparison (AI-01.B)

**Recommendation: HYBRID.** Train an S1 baseline first. Separately build **S2-H**:
- CSV boxes for large rocks,
- plus mask-derived boxes for medium rocks and distant large rocks.

Compare the two models on one shared evaluation ontology, `rock`.

This is a recommendation by the AI assistant; it is **not** human-approved. Training is **not** authorised.

Where each number comes from:
- **[F]** full dataset, automated.
- **[R]** sampled visual review by the AI assistant (not human).
- "Candidate subset" means the 9,193 frames that remain after removing the 573 anomaly-listed frames.

## 1. The two strategies as found in the data

| | S1: CSV boxes | S2: mask-derived (`ground`, variant V1_cyan) |
|---|---|---|
| What is labelled | **Physically large** rocks (blue renderer class) with box width and height ≥ ~21 px [F] | Blue + green mask components (large + small/medium classes) |
| Boxes per image (candidate subset) | 1.92 mean, 1 median [F] | +41.3 green and +3.9 blue at ≥ 50 px; S2-H total 46.1 mean [F] |
| Instance separation | Separates touching boulders (built per instance) [R] | Merges touching rocks of the same class [R] |
| Geometry | 0 invalid, 0 out-of-bounds (Task A) [F] | Valid by construction; anti-aliased edges handled by assigning cyan to large rock |
| Completeness for obstacles | Misses clearly visible rocks in 41/62 scenes (partial in 13) [R] | Recovers them in 45/62 scenes (partial in 12) [R] |
| Effort | Ready now | About half a day: merge-splitting, filters, conversion |

## 2. Why "large rock" is the wrong target on its own

- The CSV marks rocks the renderer considers physically large, whatever their apparent size.
- In the candidate subset, **28,519 green ("small") components appear larger in the image than the median CSV box** (1,836 px²) [F]. All CSV boxes together number 17,685.
- 76,168 green components are at least 21×21 px [F].
- Visual examples (frames 0676, 2875, 4044, 8894):
  - 60–100 px foreground rocks have no box;
  - a 21 px distant boulder does [R].
- A detector trained on S1 must infer *physical* size from a monocular image. Every visible near rock becomes negative evidence.

## 3. Why raw S2 is not good enough either

Per-candidate visual review [R] covered 114 candidates in stratified bins (strata are not population-weighted).

| Stratum | Reviewed | Visible rock | Merged | Ambiguous | Not visible |
|---|---|---|---|---|---|
| Green 25–50 px | 16 | 5 (31%) | 0 | 8 | 3 |
| Green 50–100 px | 16 | **13 (81%)** | 0 | 3 | 0 |
| Green 100–200 px | 16 | 10 (62%) | 0 | 5 | 1 |
| Green 200–1000 px | 16 | 8 (50%) | 5 | 2 | 1 |
| Green ≥ 1000 px | 12 | 8 (67%) | 2 | 2 | 0 |
| Blue without CSV, 25–200 px | 10 | 7 | 0 | 3 | 0 |
| Blue without CSV, 200–1000 px | 10 | 6 | 0 | 4 | 0 |
| **Blue without CSV, ≥ 1000 px** | 10 | 1 | **9** | 0 | 0 |
| Blue matching CSV (control) | 8 | 8 | 0 | 0 | 0 |

Across all additional candidates ≥ 50 px: 50/86 are visible rocks (58%), 16 merged (19%), 18 ambiguous (21%), 2 not visible.

**Merged-group mechanism [F]:**
- 3,034 of the 3,177 "uncertain" CSV boxes from Task A cover only part of a larger, merged blue-mask region.
- In those cases the CSV box is right and the mask is coarse.

**Extraction variants [F], candidate subset:**

| Variant | CSV match rate (IoU ≥ 0.5) | CSV boxes split into ≥ 2 components | Components covering ≥ 2 CSV centres |
|---|---|---|---|
| V0 strict | 0.837 | 477 | 1,705 |
| **V1 cyan → large** (chosen) | 0.834 | **350** | 1,744 |
| V2 V1 + 3×3 closing | 0.801 | 260 | **1,898** |

Closing fixes splits but merges more rocks and lowers agreement with CSV, so it is rejected.

## 4. Proposed S2-H label policy (for AI-01.C to build and validate)

1. **Large rocks:** CSV instance boxes, using X − 0.5 and Y − 0.5.
2. **Distant large rocks below the CSV cutoff:** V1 blue components of 50–999 px that are not ≥ 50% inside a CSV box and that pass the visibility proxy.
3. **Medium rocks:** V1 green components ≥ 50 px that pass the visibility proxy, **after** a split step for merged components (e.g. distance-transform watershed). The split step is not built yet.
4. **Frames:** candidate subset only.

Result [F]: 46.1 boxes per image (CSV 1.9 + blue 3.7 + green 40.5); 159 of 9,193 frames have no box.

## 5. Evaluation ontology (avoid comparing `large_rock` with `rock`)

- **Reference ontology = `rock`:** any visible rock with a box of about 50 px area at 720×480. This matches what POLAR-Sim labels on NASA images.
- **S1 baseline:** score it against `rock` ground truth with **recall per box-size bin**. Low recall on small rocks is expected by construction and is not a model failure.
- **Synthetic hold-out:** score both models against both S1 and S2-H labels.
- **NASA:** use the 92-image exposure-selected subset, rock class only (§6 of the main report). Terrain 06 is excluded from quantitative false-positive rates.

## 6. Scorecard

| Criterion | S1 | S2 raw | S2-H |
|---|---|---|---|
| Completeness for obstacle avoidance | Low | High | High |
| Geometric correctness | High | Medium (merges) | High for large, medium for small |
| Visual fidelity (box on a visible rock) | High, except shadow boxes | 58% of additional ≥ 50 px | Expected > raw; split step to be measured |
| Class consistency | Inconsistent with apparent size | Consistent | Consistent |
| Matches the NASA/POLAR-Sim ontology | Partial | Yes | Yes |
| Implementation effort | None | Low | Medium (about half a day) |
| Hackathon feasibility | Immediate baseline | Not recommended alone | Feasible as the second model |

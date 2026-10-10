# Dataset Quality Report — `curated-dataset-v1` (AI-01.C)

Tags used below:
- **[F]** full dataset, automated.
- **[R]** visual review by the AI assistant. **This is not human validation.** All rates come from small stratified samples, and no confidence interval is claimed.

## 1. Raw S2 vs cleaned S2-H

| Metric | Raw S2 | Cleaned S2-H (VALID) | Denominator / source |
|---|---|---|---|
| Boxes | 447,357 [F] | 261,385 [F] | eligible frames (9,193) |
| Visible-rock rate | 58% (50/86) [R, Task B] | **76%** [R, weighted] | Task B: 86 additional candidates ≥ 50 px. Task C: 92 VALID mask annotations in 6 strata, weighted by stratum population |
| Merged-rock rate | 19% (16/86) | **6%** | same |
| Ambiguous rate | 21% (18/86) | 18% | same |
| Not visible | 2% (2/86) | 0% | same |
| Over-split fragments | — | 0% (0 seen) | same |
| Rock coverage | 7.1% for S1 [F] | 54.9% VALID; 94.8% VALID + REVIEW [F] | 429,672 mask components ≥ 50 px; a component counts as covered when ≥ 50% of it lies inside a box |
| Median box area | 288 px² | 242 px² | S1 median is 1,836 px²; histograms in `quality_metrics.json` |

**Comparison caveat:** the two samples were drawn differently. Task B sampled raw candidates by area bin; Task C sampled VALID records by stratum, using an independent seed (20261010). The direction of the change is clear, but its exact size is not established.

## 2. Task C independent review sample

130 items with seed 20261010; per-stratum details in `quality_metrics.json` → `review_after_task_c`.

**VALID strata:**

| Stratum | Items | Visible rock | Merged | Ambiguous / not visible |
|---|---|---|---|---|
| green 50–100 px | 16 | 69% | 0% | 31% |
| green 100–200 px | 16 | 81% | 6% | 12% |
| green 200–1000 px | 16 | 88% | 12% | 0% |
| green ≥ 1000 px | 12 | 83% | 8% | 8% |
| watershed pieces | 20 | 65% | 5% | 30% |
| blue without CSV | 12 | 75% | 8% | 17% |

**Withheld strata (REVIEW and REJECTED):**

| Stratum | Items | What the review found |
|---|---|---|
| REVIEW: possible merge | 10 | 7 were merged groups, so withholding them is correct |
| REVIEW: low visibility | 12 | **5 were visible rocks**, so the rule is conservative and loses real rocks |
| REJECTED: invisible in shadow | 8 | 7 not visible, 1 ambiguous; the rule is correct |
| REVIEW: CSV box in shadow, revised rule | 8 | 0 visible; the rule is correct |

## 3. Unresolved problems

1. **REVIEW_REQUIRED is large: 192,826 boxes.**
   - The low-visibility rule sends visible rocks there in flat, softly lit scenes (frame 8894: 10 VALID versus 60 REVIEW).
   - A scene-relative rule did not do better on the calibration labels.
   - **Consequence:** REVIEW boxes must not be used as background. If the training tool cannot ignore regions, this needs a decision before training (see §4).
2. **About 18–30% of VALID small rocks and split pieces are ambiguous.** This is the main remaining label noise.
3. **Watershed recovers only 39% of merged instances.** The rest stay merged and mostly go to REVIEW.
4. **The visibility calibration rests on 78 rocks reviewed in Task B** and is not validated by a human.
5. **The test data shift is not addressed by cleaning.** The synthetic renders differ from the NASA lab-sandbox photos.

## 4. GO / NO-GO for Task D — **GO, with conditions**

The cleaned VALID set is cleaner than raw S2: visible-rock rate 76% versus 58%, merged rate 6% versus 19%. It preserves every CSV instance, and it is reproducible and leakage-annotated.

Conditions:

1. **REVIEW_REQUIRED regions:** Task D must choose explicitly how to handle them; the choice needs approval. Options:
   - **(a)** grey-fill REVIEW regions in a *derived* training copy of the images;
   - **(b)** train on VALID only and accept the extra false-negative noise;
   - **(c)** run an ablation with REVIEW counted as positive.
2. **Splits** must respect `leakage_group`.
3. **Human review:** a human spot-checks at least 20 scenes in `before_after/` and the 6 review-sample sheets before the labels are frozen.
4. **Training remains unauthorised** until it is approved.

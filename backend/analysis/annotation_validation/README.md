# Annotation validation (Task AI-01.B)

Read-only S1 (CSV) vs S2 (mask-derived) comparison → `experiments/annotation-validation/`.

```powershell
cd backend
uv run --no-sync python -m analysis.annotation_validation --workers 8   # ~2 min, deterministic
```

| Module | Scope |
|---|---|
| `config.py` | extraction variants, thresholds, sampling quotas, drawing styles, NASA view parameters |
| `s2_extract.py` | per-frame mask classes, connected components, render-visibility proxy, per-CSV-box mask evidence |
| `al_analysis.py` | full-dataset S1/S2 statistics, variant comparison, uncertain-box triage, stratified sampling |
| `al_outputs.py` | per-scene overlays, A/B/C/D comparisons, contact sheets, candidate / uncertain crop grids |
| `nasa_review.py` | Terrain06 negative-set sheets, exposure brackets, per-box visibility proxy, exposure-selection policy |
| `review_io.py` | manifests (never overwrites human edits), assistant-review ingestion, static gallery |
| `summarize.py` | metrics (full-dataset vs sampled review), figures, decision file |

`experiments/annotation-validation/assistant_review/*.csv|json` are visual assessments written by the AI assistant and read
as inputs; they are reported separately from human review. `approved_for_training` is never set by code.

# YOLO dataset engineering (Task AI-01.D)

Provisional, leakage-controlled YOLO export of `curated-dataset-v1` → `dataset/processed/yolo-rock-v1/` + `experiments/yolo-dataset/`.

```powershell
cd backend
uv run --no-sync python -m analysis.yolo_dataset --workers 8   # ~10 min, deterministic; no training
```

| Module | Scope |
|---|---|
| `validate.py` | read-only checks of Task C outputs (counts, hashes, geometry, CSV preservation, zero-VALID discrepancy) |
| `similarity.py`, `neardup.py` | thumbnails, ID-sequence test, ORB + RANSAC near-duplicate detection with exhaustive audit |
| `splits.py` | leakage groups (union-find), stratified 80/10/10 group assignment, config A rule |
| `export.py` | YOLO labels, unmasked eval/config A copies, config B inpainting, post-export validation |
| `ultralytics_check.py` | consumer-side verification with Ultralytics' own path mapping and label verifier (offline) |
| `gallery.py` | interactive review page (NOT_REVIEWED by default, JSON/CSV export) |
| `__main__.py` | orchestration, export-time D05 correction for split pieces, previews, integrity |

Human decisions exported from the gallery go in `experiments/yolo-dataset/human_review/`; the run reports them but never sets
`frozen` or `approved_for_training`.

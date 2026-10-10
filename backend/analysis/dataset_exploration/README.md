# Dataset exploration (Task AI-01.A)

Read-only analysis of `dataset/` → `experiments/dataset-exploration/`.

```powershell
cd backend
uv run --no-sync python -m analysis.dataset_exploration            # ~2-3 min, 8 workers
uv run --no-sync python -m analysis.dataset_exploration --workers 4 --out ..\experiments\tmp
```

Exit code `0` = done and source data fingerprint unchanged, `2` = a data file changed during the run.

| Module | Scope |
|---|---|
| `inventory.py` | A1: counts/sizes/extensions, zip-vs-extracted check, source fingerprint |
| `artificial_lunar.py` | A2: full decode of all 3×9,766 PNGs, CSV validation, CSV↔mask matching, anomaly lists, real-moon set |
| `nasa_polar.py` | A3: 16-bit image stats, exposure brackets, calibration parsing, LiDAR `.xyz` + projection test |
| `polar_sim.py` | A4: YOLO box / polygon validation, file classification (canonical / copy / metadata) |
| `matching.py` | A5: deterministic NASA↔POLAR-Sim key matching + stereo-disparity validation |
| `quality.py` | Severity-ranked issue list built from the measured results |

Guarantees: JSON is written with sorted keys and rounded floats, samples are drawn from a fixed seed, and nothing under `dataset/` is opened for writing.
Known non-determinism: `dataset_inventory.json` / `data_quality_report.json` also record `.git` metadata counts, which IDE git tooling can change between runs.

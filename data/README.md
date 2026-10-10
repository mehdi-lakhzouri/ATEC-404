# data/

This folder holds **metadata only**. No images and no box coordinates are published, because the redistribution licence of the
source images has not been confirmed (see `docs/DATASET.md`).

| File | Content |
|---|---|
| `manifests/split_manifest.csv` | One row per eligible frame (9,193). It records the split (train/val/test), leakage group, strata, VALID/REVIEW/REJECTED counts, Config A/B membership, the source render path relative to the dataset root, and the render SHA-256. Byte-identical to the registered manifest (hash in `artifacts/provenance/VERSIONS.json`) |
| `manifests/data_A.yaml`, `data_B.yaml` | Ultralytics dataset definitions. The absolute `path:` is replaced by `<repo>/dataset/processed/yolo-rock-v1.1`; point it at your local copy to use them |

`samples/` is intentionally absent: no redistribution-cleared images exist yet. To view examples locally:
1. obtain the dataset;
2. set `ARTEMIS_SHOW_LOCAL_IMAGES=1`;
3. run the notebooks.

# Dataset Consolidation Plan

**Status: PLAN, nothing moved or deleted.** Sizes are hard-link aware: "frees" means bytes actually released.

## Principle

Keep datasets **where they are**. Every processed version is registered by content hash, the code addresses them through
`artemis_x.config.paths` (with the `ARTEMIS_DATA_ROOT` override), and the YOLO trees are NTFS hard links. Moving or copying them
with ordinary tools would materialise the links and add up to about 5 GiB. Consolidation is therefore done by **retiring superseded
copies**, not by relocating the canonical ones.

## Canonical layout after consolidation

```text
dataset/
├── artificial-lunar/            # source (extracted): images/{render,ground,clean}, CSV, ID lists, real_moon_images/   KEEP
├── nasa-polar/Terrain{01,06,11}_*/   # source (extracted)                                                        KEEP
├── polar-sim-source/            # CC0 labels repo clone (re-clonable)                                            KEEP
└── processed/
    ├── VERSIONS.json            # registry (kept, also records retired versions)                                  KEEP
    ├── curated-dataset-v1.1/    # canonical curated dataset                                                       KEEP
    └── yolo-rock-v1.1/          # canonical YOLO dataset: config_A, images, labels, manifests (+ config_B, decision) KEEP
```

## Retained (PROTECTED)

| Path | Reason |
|---|---|
| `dataset/artificial-lunar/images`, `real_moon_images`, CSV and ID lists | Only extracted source; input to Tasks A–D and to any rebuild |
| `dataset/nasa-polar/Terrain*/` | Real-domain reference for future evaluation |
| `dataset/polar-sim-source/` | Referenced by code; licence and citation source |
| `dataset/processed/VERSIONS.json` | Content-hash registry |
| `dataset/processed/curated-dataset-v1.1/` | Canonical annotations |
| `dataset/processed/yolo-rock-v1.1/{config_A,images,labels,manifests}`, `data_*.yaml` | Trained data, val/test, split and ignore manifests |

## Retirement candidates

| Path | Why it can go | What is lost / mitigation | Frees |
|---|---|---|---|
| `dataset/artificial-lunar/archive.zip` | CRC-32 identical to the extraction (29,378/29,378) | The pristine original download. **Delete only if an external copy exists** (Kaggle re-download or your own backup) | 5.02 GiB |
| `dataset/nasa-polar/Terrain{01,06,11}_*.zip` | CRC-32 identical to the extractions (340/448/448) | Same condition | 2.44 GiB |
| `polar-sim-source/` (root) | Duplicate clone, same commit `593ed65`; not referenced by code | None (the canonical clone stays) | 0.71 GiB |
| `dataset/polar-sim/` | Empty | None | 0 |
| `dataset/processed/curated-dataset-v1/` | Superseded by v1.1; counts, policy and hashes are already in `VERSIONS.json` | Byte-level v1 annotations (re-buildable with the Task C code at policy S2-H/1.0). **Code update needed:** `preflight.py` PROTECTED list, `dataset_cleaning/config.py` provenance constant | 0.24 GiB |
| `dataset/processed/yolo-rock-v1/` | Superseded by v1.1 (identical labels, split differs only by v1.1 dedup) | **Code update needed:** `remediation/ai_review.py` reads its split manifest (one-off Task D.2 step), `yolo_dataset --link-from` default, `preflight.py`, `collect_taskd_review_images.ps1` | **0.02 GiB** (images are hard-linked into v1.1) |
| `dataset/processed/yolo-rock-v1.1/config_B/` | *Only if Config B is abandoned* | The untrained masked arm; regenerable deterministically by the Task D exporter | ~0 while v1 exists; **3.03 GiB** if v1 is also removed |

**Config B recommendation: KEEP.** It is the second arm of the pre-registered A-versus-B comparison (D.2), its masking has been
audited, and it costs nothing extra while v1 exists.

## Before / after (physical, hard links resolved)

| State | `dataset/` physical |
|---|---|
| Now | 25.9 GiB (10.09 Artificial Lunar + 5.91 NASA POLAR + 0.60 POLAR-Sim + ~9.3 unique processed; processed includes 3.03 Config B and the v1 sharing) |
| After retiring v1 trees + empty dir | about 25.6 GiB |
| + duplicate archives (only with an external copy) | about **18.2 GiB** |
| + Config B (only if abandoned) | about 15.1 GiB |

The root POLAR-Sim duplicate (0.71 GiB) sits outside `dataset/` and is counted in `CLEANUP_DRY_RUN.md`.

## Procedure (after approval)

1. Update the code references listed above, pointing them at v1.1 or making them optional, and run both test suites.
2. Remove the approved paths with an explicit, logged command, not a GUI move to the recycle bin (which would keep the space).
3. Update `VERSIONS.json` statuses of retired versions to `RETIRED (deleted <date>; metadata only)`, keeping their hashes.
4. Re-run `scripts/verify_artifacts.py`, both test suites and `scripts/audit_workspace.py`, then compare free disk space against
   the baseline (371 GB free on 2026-10-09).

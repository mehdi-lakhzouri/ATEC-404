# Storage Analysis (hard-link aware)

| | |
|---|---|
| Source | `raw/storage_scan.json` (217,700 files, scanned in 21 s) and `RETENTION_MANIFEST.json` |
| Disk C: | 925 GB, 555 GB used, **371 GB free** (2026-10-09) |

## 1. Why apparent size is misleading here

The YOLO exports and the training data views are **NTFS hard links**: one physical file appears under several paths.

| Total | Value |
|---|---|
| Apparent size (every path counted) | **51.0 GiB** |
| Physical size (each file record once) | **42.5 GiB** |
| Of which hard-linked from outside the workspace (uv cache, pnpm store) | 3.7 GiB. Deleting the workspace copy frees ~0 of this |

**Reclaimable** in this document means bytes freed when every hard link of a file is inside the deleted set. Sizes are file sizes, not cluster-rounded allocation.

## 2. Physical usage by area

| Area | Apparent (GiB) | Physical (GiB) | Reclaimable if deleted alone (GiB) |
|---|---|---|---|
| `.git/objects` (dangling blobs) | 14.69 | 14.69 | **14.69** |
| `dataset/artificial-lunar` (5.02 extracted + 5.02 zip) | 10.09 | 10.09 | 10.09 |
| `dataset/processed` (all versions) | 9.91 | ≈ 5.0 unique | see §4 |
| `dataset/nasa-polar` (3.47 extracted + 2.44 zips) | 5.91 | 5.91 | 5.91 |
| `experiments/task-e` | 4.54 | ≈ 0.74 unique | 0.72 |
| `backend/.venv` | 3.33 | 3.33 | **0.07** (hard-linked from the uv cache) |
| `polar-sim-source` (root duplicate) | 0.71 | 0.71 | 0.71 |
| `dataset/polar-sim-source` | 0.60 | 0.60 | 0.60 |
| `frontend` | 0.43 | 0.42 | **0.00** (pnpm store) |
| `experiments/yolo-dataset` (0.415 is the ORB cache) | 0.42 | 0.42 | 0.42 |
| other experiments (A–D.2, F) | 0.34 | 0.34 | 0.34 |
| root ZIPs and `weights/` | 0.08 | 0.08 | 0.08 |

## 3. The `.git` finding

`git count-objects -vH` reports 47,954 loose objects totalling 14.69 GiB, with 0 packs. `git fsck --unreachable` lists them as **dangling blobs**: there is no `.git/index`, no commit and no ref.

- **Interpretation:** these are copies of workspace files (largely dataset images) from a `git add` whose staging was later discarded.
- **Recovery:** `git gc --prune=now` (or `git prune`) frees about **14.7 GB** without touching any working file.
- **Caveat:** a few blobs may be earlier versions of report files that have since been regenerated. Nothing references them, and the current versions are on disk.

### Update at 23:40: external Git maintenance doubled the garbage

**These changes were made outside this migration.**

| Time | Change | Evidence |
|---|---|---|
| 23:28 | `.git/config` gained `remote "origin" = https://github.com/mehdi-lakhzouri/ATEC.git` | `.git/config` |
| 23:28 | GitKraken ran repository maintenance | `.git/gk/config`: `maintenanceLastRun = 22:28:51Z` |
| 23:40 | `.git/objects/pack/loose-d5c95e94….pack` appeared (15.65 GB): every dangling loose blob is now **also** packed | `git count-objects -vH`: `in-pack: 47954`, `prune-packable: 47954` |
| 23:29 | `dataset/polar-sim-source/.git` was deleted. The label files are intact; the root `polar-sim-source/` clone (commit `593ed65`) is now the only local copy with Git history | directory mtime |

**Consequences:**
- `.git` now holds **≈ 29.3 GiB**. `git gc --prune=now` would free all of it: nothing reachable exists.
- The source-dataset fingerprint used by `training/preflight.py` has changed. Future runs record a new pre-flight baseline, so this
  only matters when comparing against E.0/E.1a/E.1c records.

## 4. Hard-link sharing between dataset versions

Physical GiB shared between the units listed:

| Shared physical files | Between |
|---|---|
| 3.01 | `yolo-rock-v1/config_B` ↔ `yolo-rock-v1.1/config_B` |
| 0.86 | `yolo-rock-v1{,.1}/config_A` ↔ all three `task-e/*/data_view` |
| 0.42 | `yolo-rock-v1{,.1}/images` ↔ all three `data_view` |
| 0.41 | `yolo-rock-v1/images` ↔ `yolo-rock-v1.1/images` |

**Consequences:**
- Deleting `yolo-rock-v1` **alone** frees **0.02 GiB**, although it lists as 4.7 GiB.
- Deleting all three `data_view` folders frees **0.006 GiB**: only the Ultralytics label caches.
- Config B is a real **3.03 GiB** cost only once v1 is gone. It is shared until then, so deleting just one of the two Config B trees frees ~0.
- **Copying or moving** these trees with ordinary copy tools (Explorer, `Copy-Item`, `robocopy` without `/SL`, zip/unzip) **materialises every link** and would add up to ~5 GiB. Any relocation must use rename/move on the same volume, or keep paths and add configuration aliases.

## 5. Deletion scenarios (combined sets, links resolved)

| Scenario | Contents | Frees (GiB) | Category |
|---|---|---|---|
| S1 | empty `dataset/polar-sim`, 2 root share ZIPs, `weights/yolo26n.pt` | 0.08 | SAFE |
| S2 | dangling git blobs, loose + packed since 23:40 (`git gc --prune=now`) | **29.27** (14.69 before the external maintenance run) | REVIEW |
| S3 | 4 dataset ZIPs (CRC-identical to their extractions) | **7.46** | REVIEW: only with a confirmed external copy |
| S4 | duplicate root `polar-sim-source` clone | 0.71 | REVIEW |
| S5 | `curated-dataset-v1`, `yolo-rock-v1`, 3 `data_view` folders | 0.26 | REVIEW |
| S6 | 20 `epochN.pt` (0-based), ORB feature cache, E.1a weights | 0.64 | REVIEW |
| S7 | *only if Config B is abandoned*: `yolo-rock-v1.1/config_B` + `yolo-rock-v1` | 3.05 | REVIEW (decision) |

**Totals:**

| Selection | Frees |
|---|---|
| S1–S6 together | ≈ 38.4 GiB (was 23.8 GiB before the external Git maintenance) |
| Plus S7 | ≈ 41.5 GiB |
| Workspace after S1–S6 | 56.6 → ≈ 18.2 GiB physical |

Nothing has been deleted. The per-path list is in `CLEANUP_DRY_RUN.md`, written after the migration.

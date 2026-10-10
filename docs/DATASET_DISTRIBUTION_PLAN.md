# Prepared-Dataset Distribution Plan: `yolo-rock-v1.1-configA`

**Status: NOT PUBLISHED.** No archive has been built or uploaded and no download URL exists. `scripts/download_dataset.py` fails
with a clear message until an approved source is configured.

## Goal

Let users reproduce training **without** downloading and re-processing the original source datasets. They download the
already-cleaned, already-annotated YOLO dataset, verify it file by file, and run notebook 09.

## Release contents

| Part | Files | Uncompressed |
|---|---|---|
| Config A training images + YOLO labels (`config_A/images/train`, `config_A/labels/train`) | 1,903 + 1,903 | 920.3 MB |
| Validation images + labels (`images/val`, `labels/val`) | 923 + 923 | 447.6 MB |
| Test images + labels (`images/test`, `labels/test`) | 917 + 917 | 440.8 MB |
| Manifests: `split_manifest.csv`, `eval_ignore_regions.json`, `eval_ignore_regions_subthreshold_PROPOSED.json`, `dataset_status.json` | 4 | 10.5 MB |
| **Total** | **7,490** | **1.82 GB** (PNG does not compress further: archive ≈ 1.8 GB) |

**Excluded:**
- original source datasets (Artificial Lunar renders, ground and clean masks, CSVs; NASA POLAR; POLAR-Sim);
- Config B (masked) training images;
- curated CSV annotations;
- experiment outputs.

**Optional split:** a `train+val` archive (≈ 1.37 GB) and a separate `test` archive (≈ 0.44 GB), for users who only reproduce
training.

## Integrity

| Element | Where | Role |
|---|---|---|
| Per-file SHA-256 manifest | `data/release/yolo-rock-v1.1-configA.files.sha256` (committed to Git; manifest SHA-256 `4e117ee3…`) | **Trust anchor**: every downloaded file must match |
| Release record | `data/release/dataset_release.json` | Sizes, contents, status, registry proof; `archives: []` until published |
| Registry proof | Built by `scripts/build_release_manifest.py` | The local source dataset's full label tree (`e2a4b406…`), image tree (`1f58b56f…`) and split manifest match `VERSIONS.json` |
| Archive SHA-256 | Added to `archives[]` when the archive is built | Checked before extraction |

**Verification commands:**

```powershell
uv run --project backend python scripts/download_dataset.py --check-only            # verify an existing local copy (full hash)
uv run --project backend python scripts/download_dataset.py --source <approved https URL | local ZIP | extracted folder>
```

The script refuses non-HTTPS URLs and blocks path traversal in archives. It never overwrites an unverified folder, and it writes
a machine-specific `data/local/data_A.local.yaml` (git-ignored) instead of editing the dataset.

## Hosting options (for your decision)

| Host | Fit | Notes |
|---|---|---|
| **Zenodo** | Archival, citable DOI, immutable versions | 50 GB per record by default. A licence must be chosen at upload, and restricted access is possible |
| **Hugging Face Datasets** | Easy programmatic download, dataset card, gated access | Needs a licence and a dataset card. Large PNG sets work; Git LFS / Xet storage is handled by the platform |
| GitHub Release asset | Close to the code | 2 GB per-file limit: the 1.82 GB archive fits, but just barely |

## Blocking requirements before any publication

1. **Licence of the source renders.** The images in the release are re-encoded Artificial Lunar Landscape renders, and the labels
   are derived from that dataset's masks and CSV boxes. **Redistribution is allowed only if that dataset's licence permits
   derivatives and redistribution.** No licence file was found in the downloaded archive, and an online search did not surface the
   Kaggle licence. Check the Kaggle page ("Artificial Lunar Rocky Landscape Dataset", Romain Pessia) and record the licence and
   attribution text.
2. **Owner approval** of the host and of the dataset licence statement, which must be compatible with the source licence.
3. **A dataset card** stating that the labels are automated (S2-H/1.1) and not human-verified, the evaluation policy, the split
   seed, and the provenance.
4. **Then:** build the archive from the manifest list, compute its SHA-256, upload it, and add
   `{"url", "sha256", "bytes"}` to `archives[]`. The download script will use it automatically.

## Provenance kept in the repository

- `artifacts/provenance/VERSIONS.json`: registry hashes.
- `data/manifests/split_manifest.csv`: split membership, leakage groups, source render SHA-256.
- `docs/DATASET.md`: sources and versions.
- `artifacts/reports/C_ANNOTATION_POLICY.md`: the S2-H policy.

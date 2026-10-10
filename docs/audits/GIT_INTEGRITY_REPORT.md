# Git Integrity and Oversized-Object Report

| | |
|---|---|
| Investigated | 2026-10-10 |
| Method | Read-only commands only: `status`, `show-ref`, `reflog`, `count-objects`, `cat-file --batch-check --batch-all-objects`, `ls-tree`, `config`, `ls-remote` (with prompts disabled) |
| Changes | **No object was removed, no ref or remote changed, nothing committed or pushed.** |

## 1. Repository state

| Item | Finding |
|---|---|
| Branch | `master`, **no commits** |
| Refs (`show-ref`), reflogs (`.git/logs`) | **none** |
| Index (`.git/index`) | **absent**: nothing is staged |
| Tracked files | none (`git status` lists everything as untracked) |
| Remote | `origin = https://github.com/mehdi-lakhzouri/ATEC.git`, added 2026-10-09 23:28. `ls-remote` without credentials → *authentication required*: the repository is private or does not exist yet. Its contents could not be verified |
| Objects | 47,954: **47,743 blobs + 211 trees**, 17.39 GB of object payload |
| Storage | Loose: 47,954 files, 14.69 GiB. Pack: `loose-d5c95e94….pack`, 15.65 GB (14.58 GiB) + `.idx` + `.rev`. **Total ≈ 29.27 GiB** |
| Reachability | **0 objects reachable**: there are no refs, so every object is dangling. `count-objects` reports `prune-packable: 47954`, meaning every loose object also exists in the pack |
| Git config | `core.autocrlf=true` (system: `Program Files/Git/etc/gitconfig`); GitKraken `maintenanceLastRun = 2026-10-09T22:28:51Z` |

## 2. What the objects are

The 211 trees form **one root tree** whose entries are the whole workspace:
`artemis-comparisons.zip`, `audit-artemis.ps1`, `backend`, `collect_taskd_review_images.ps1`, `dataset`, `experiments`,
`frontend`, `polar-sim-source`, `taskd-review-images.zip`. They describe 75,878 staged paths.

**Interpretation:** between 18:37 and 20:25 on 2026-10-09 (loose-object timestamps), the entire folder was staged (`git add`),
including the raw datasets. The staging area was later discarded (no index exists), and no commit was ever made.

Each blob was compared with the current workspace by recomputing Git blob IDs for 131,410 working files:

| Category | Objects | Payload | Unique information? |
|---|---|---|---|
| Byte-identical to a current file | 47,625 | 15.36 GB | No |
| Identical to a current file after CRLF→LF normalisation (`core.autocrlf=true`: NASA `.xyz` point clouds, JSON reports, HTML galleries) | 96 | 0.92 GB | No |
| `dataset/artificial-lunar/archive.zip`, truncated: blob = 1,095,569,597 bytes = 5,390,536,893 − 2³² (Git for Windows wrapped the size at 4 GiB) | 1 | 1.10 GB | **No: a corrupt, unusable copy.** The real 5.39 GB ZIP is intact on disk |
| Small text blobs that differ from the current files (earlier versions of `experiments/{dataset-cleaning-v1.1,yolo-dataset-v1.1}/logs/run.log` and `yolo-dataset-v1.1/dataset_status.json`, plus 18 blobs not referenced by the final tree) | 21 | **126 KB** | Yes, but trivial: superseded logs and status files |
| Trees | 211 | — | Directory listings only |

**Conclusion:** apart from 126 KB of superseded log and status text, nothing in `.git` is unique. **No data needs Git for recovery.**

## 3. Why GitKraken created a 15.65 GB pack

GitKraken runs Git's background maintenance. The `loose-objects` task:
1. packs up to 50,000 loose objects into a pack named `loose-<hash>.pack`;
2. deletes loose objects only when they are *already* packed (`prune-packed`), and that happens at the **next** run.

The 47,954 objects fit in one batch, so all of them were copied into the pack and the loose originals stay until the next
maintenance run. PNG images and ZIPs do not compress, so the pack is as large as the loose objects. The result is temporarily
**two copies** of the same unreachable data. Even after a future `prune-packed`, the 14.6 GiB pack would remain until garbage
collection with pruning.

## 4. Is `ATEC.git` the right repository?

It cannot be verified without your credentials. `ATEC` matches the hackathon name (ATEC 2026). Choosing between a shared
hackathon repository and a dedicated `artemis-x` repository is **your decision**. Nothing has been changed or pushed.

## 5. What would enter the first commit

With the new `.gitignore`, `git ls-files --others --exclude-standard` lists the files in `FIRST_COMMIT_FILES.txt`. See
`FIRST_COMMIT_READINESS.md`. **No dataset, experiment output, ZIP, environment or frontend file is included.**

## 6. Recommended resolution (requires your approval)

| Step | Action | Size / effect |
|---|---|---|
| 0 | **Close GitKraken**, so that maintenance does not run concurrently | — |
| 1 | **Back up only the unique content:** export the 21 differing blobs plus the full object listing to `experiments/_git_orphans_backup/` (git-ignored) | ≈ 126 KB + ≈ 3 MB listing |
| 2 | `git gc --prune=now` (equivalently, since nothing is reachable, `git prune` then removal of the unreachable `loose-*` pack) | Frees **≈ 29.27 GiB (≈ 31.4 GB)**. `.git` drops to a few KB |
| 3 | Verify: `git count-objects -vH` shows 0, `git fsck` is clean, the remote and config are unchanged, and the workspace files are untouched (`scripts/verify_artifacts.py`) | — |
| 4 | Decide the remote: keep `ATEC.git`, or create a dedicated repository | your choice |

**Not recommended:** a full backup of `.git`. It would copy about 31 GB of data that already exists in the workspace.

**Alternative:** delete `.git` and run `git init`. The effect is the same, but it also discards the GitKraken settings and the remote,
which would then have to be re-added.

**Realistic saving:** the whole 29.27 GiB, because none of it is reachable or uniquely needed. The repository metadata that remains
(config, remote) is a few KB.

## 7. Preventing a recurrence

- Keep the root `.gitignore`, which excludes `dataset/`, `experiments/`, ZIPs, `.venv` and `frontend/`, before staging anything.
  Check with `git status --short` or `git add --dry-run .` first.
- Git for Windows mishandles files above 4 GiB (the truncated ZIP above). Never stage large archives; use a dataset release
  (Zenodo or Hugging Face) instead of Git.
- `.gitattributes` keeps artifacts byte-exact (`-text`), so their recorded SHA-256 values remain valid after checkout.

# Model Privacy and Licensing Decision Report

**Nothing has been re-frozen, replaced, licensed or published.** This report sets out the facts and options for the owner's
decisions.

## 1. Embedded local paths in the frozen model

`models/artemis_rock_yolo11n.pt` (SHA-256 `413d0481…0752eb`, byte-identical to the historical E.1c `best.pt`) contains these
metadata strings with the training machine's user directory:

| Field | Content |
|---|---|
| `train_args.data` / `model.args.data` | absolute path to the E.1c `data_view.yaml` |
| `train_args.project` | absolute path to `experiments/task-e` |
| `train_args.save_dir` | absolute path to the E.1c run folder |
| `git.root` | absolute path of the workspace |

These fields come from Ultralytics' checkpoint format. **None of them affects inference.**

### Proposed transformation (`scripts/sanitize_checkpoint.py`)

The script reads the frozen model (SHA-verified) and writes a **separate** copy. In that copy:
- `data`, `project` and `save_dir` are replaced by repository-relative placeholders;
- `git.root` is set to `None`;
- a `sanitised_from_sha256` field is added.

**Why the SHA-256 changes:** the file's serialised metadata bytes change, so its hash changes even though the weights are bit-for-bit
the same.

### Feasibility test (run in a temporary folder, not published)

| Check | Result |
|---|---|
| Every tensor in `model.state_dict()` | bit-identical |
| Class names, all inference-relevant arguments | identical |
| Predictions (boxes and scores, frozen settings) on 20 validation images | **20/20 identical** |
| Local paths remaining | none |
| New SHA-256 | `050e840372019e1c9ac50bf93ad015a0f3b7a4b139ba147209a9c2f0f33096ff` (5,521,109 bytes) |
| Frozen model after the test | unchanged (`413d0481…`) |

### Options

| Option | What happens | Provenance impact |
|---|---|---|
| **A. Publish the sanitised copy (recommended if privacy matters)** | Add `models/artemis_rock_yolo11n_public.pt` (new SHA-256) as the published file. Keep the historical frozen file local (git-ignored), with its SHA-256 in `FROZEN_MODEL_CONFIG.json` | Needs a documented **distribution addendum**: `public_sha256`, `derived_from_sha256 = 413d0481…`, the equivalence evidence above, and `RockDetector` accepting either verified hash. The frozen config and the test results remain unchanged |
| B. Publish the historical file as is | No change | The user-directory name is visible to anyone who inspects the checkpoint |
| C. Do not publish weights | Users train their own via notebook 09 | Workflow B (inference demo) unavailable without the weights |

**Inference-equivalence verification for option A:**
1. Tensor equality.
2. Prediction equality on all 923 validation images (not only 20).
3. The ignore-aware validation metrics recomputed from the sanitised model's predictions must equal the frozen ones exactly.
4. The test split is **not** touched.

## 2. Distribution obligations

| Component | Licence / terms (as found) | Obligation | Confidence |
|---|---|---|---|
| **Ultralytics YOLO11 code** (dependency) | AGPL-3.0 (`ultralytics` package; checkpoint metadata `license: AGPL-3.0`) | Distributing software that uses it, or offering it over a network, triggers AGPL source-availability duties. An Ultralytics Enterprise licence removes them | High |
| **Pretrained `yolo11n.pt`** (initialisation, not redistributed) | AGPL-3.0 (Ultralytics release asset) | Not committed (git-ignored); users download it from Ultralytics | High |
| **Trained Artemis-X weights** | Ultralytics states that models trained with its code are AGPL-3.0 by default | Publishing them means publishing under AGPL-3.0 terms, unless you hold an Enterprise licence | Medium-high (from Ultralytics' licence pages; confirm the full terms) |
| **Artificial Lunar Landscape dataset** (Kaggle) | **Unknown**: no licence file in the archive; not found online | Do **not** redistribute images or derived labels until the Kaggle licence is confirmed. The weights are also derived from it, so check whether the licence allows publishing trained models | Unverified |
| **NASA POLAR** | **Unknown**: the NASA Ames dataset pages show a citation, but no licence was found | Not used in training or evaluation; nothing derived from it is published. Cite it if mentioned | Unverified |
| **POLAR-Sim labels** | CC0 1.0 (`LICENSE`; the README still says BSD-3) | No obligation under CC0. Citing the paper is good practice | High (licence file) |
| **Derived artifacts** (metrics, plots, split manifest, release file manifest) | Metadata and statistics only; no pixels or boxes | Low risk. The split manifest includes source render SHA-256s and file names, not content | Medium |
| **Project source code** (`src/`, `backend/`, `scripts/`, notebooks) | **Not chosen** | Needs the owner's decision. Because the code imports Ultralytics (AGPL-3.0), **AGPL-3.0** is the simplest compatible choice. A permissive licence is possible for your own code, but the combined distribution still carries AGPL obligations | — |

## 3. Decisions requested

1. **Project licence:** AGPL-3.0 (recommended for compatibility), another licence, or keep deciding later. Until then the
   repository is all-rights-reserved.
2. **Model publication:** option A, B or C above.
3. **Dataset licence:** confirm the Kaggle licence and attribution text. This blocks the dataset release and possibly the
   publication of the weights.

**Sources consulted:**
- [Ultralytics licence](https://ultralytics.com/license)
- [Ultralytics licence request](https://www.ultralytics.com/request-license)
- [NASA Ames POLAR dataset page](https://ti.arc.nasa.gov/dataset/IRG_PolarDB/)
- [POLAR-Sim paper](https://arxiv.org/pdf/2309.12397)
- [Planetside forum post announcing the Artificial Lunar dataset](https://planetside.co.uk/forums/index.php?msg=265038)

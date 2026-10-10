"""Generate, execute and validate the eight Artemis-X research notebooks (notebooks/0*.ipynb).

The notebooks read only published artifacts (artifacts/, data/manifests/, models/). None of them trains, rebuilds a
dataset or evaluates the held-out test split; the builder refuses to save a notebook whose source contains a call to
the trainer or the test evaluator, or whose outputs contain absolute local paths.

Usage (repo root; notebook tooling is layered on the locked environment without changing uv.lock):
    uv run --project backend --no-sync --with nbformat --with nbclient --with ipykernel python scripts/build_notebooks.py [--no-execute]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
FORBIDDEN_SOURCE = [r"training\.train\b", r"task_f_test_eval", r"\.train\(", r"images/test", r"images\\\\test", r"build_ignore", r"yolo_dataset\.export"]
FORBIDDEN_OUTPUT = [r"[A-Za-z]:\\\\Users", r"[A-Za-z]:\\Users", r"[A-Za-z]:/Users", re.escape(Path.home().name)]  # user name read at runtime

BOOT = '''import os, sys
from pathlib import Path
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "artemis_x").is_dir())
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from artemis_x.config import paths
from artemis_x.reporting import artifacts as A, plots as P
P.use_style()
pd.set_option("display.precision", 3); pd.set_option("display.max_colwidth", 90)
# Dataset pixels are never committed (redistribution licence unconfirmed). Set ARTEMIS_SHOW_LOCAL_IMAGES=1 to view
# local images when the dataset folder is present.
SHOW_LOCAL_IMAGES = os.environ.get("ARTEMIS_SHOW_LOCAL_IMAGES") == "1" and paths.local_dataset_available()
FIG = paths.ARTIFACTS / "figures"
print("artemis_x ready | local dataset present:", paths.local_dataset_available(), "| showing local images:", SHOW_LOCAL_IMAGES)'''

DISCLAIMER = ("> **Provisional results.** All labels were produced by an automated pipeline (S2-H policy) and have **not** been "
              "verified by humans. Metrics are measured on synthetic renders only and do not establish performance on real lunar "
              "imagery or readiness for rover deployment.")


def md(s: str):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s: str):
    return nbf.v4.new_code_cell(s.strip("\n"))


def header(n: int, title: str, objective: str, questions: list[str], inputs: list[str]) -> list:
    q = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(questions))
    inp = "\n".join(f"- `{x}`" for x in inputs)
    return [md(f"# {n:02d} · {title}\n\n**Artemis-X - AI perception for lunar rock and obstacle detection**\n\n"
               f"**Objective.** {objective}\n\n**Research questions**\n\n{q}\n\n**Inputs (published, pixel-free artifacts)**\n\n{inp}\n\n{DISCLAIMER}"),
            md("## Setup\nThe notebook imports the `artemis_x` package from `src/` and reads only published artifacts. "
               "It never trains a model, rebuilds a dataset or evaluates the held-out test split."),
            code(BOOT)]


def footer(conclusion: str, nxt: str | None, prov: list[str]) -> list:
    cells = [md("## Provenance\nEach artifact below is a verified copy (or path-sanitised derivative) of a historical experiment "
                "output; `scripts/verify_artifacts.py` re-checks every hash."),
             code(f"A.provenance({prov!r})"),
             md(f"## Conclusion\n{conclusion}")]
    if nxt:
        cells.append(md(f"**Next:** [{nxt}]({nxt})"))
    return cells


# ---------------------------------------------------------------------------------------------------------------------
def nb01():
    c = header(1, "Dataset Exploration", "Characterise the data sources available for lunar rock detection and identify the problems a "
               "detector must cope with before any modelling decision is made (Task A).",
               ["What data are available, at what resolution and in what quantity?", "How complete and how reliable are the original annotations?",
                "Which data-quality problems constrain the later pipeline?"],
               ["artifacts/data/artificial_lunar_analysis.json", "artifacts/data/data_quality_report.json", "artifacts/data/dataset_inventory.json",
                "artifacts/figures/a_*.png"])
    c += [md("""## 1. Background and data sources
Three sources were inventoried (Task A):

| Source | Content | Role in Artemis-X |
|---|---|---|
| **Artificial Lunar Landscape** (Kaggle release) | 9,766 synthetic lunar scenes, each with a photorealistic *render*, a semantic *ground* mask (sky / small rocks / large rocks) and a *clean* mask; a CSV of large-rock boxes | **Primary training and evaluation data** |
| **NASA POLAR** (3 terrains) | Real stereo images of a lunar-analogue testbed: 12-bit in 16-bit PNG, exposure brackets, LiDAR | Real-domain reference; *not* used for training or evaluation yet |
| **POLAR-Sim** labels (github.com/uwsbel/POLAR-Sim, CC0) | YOLO boxes and segmentation polygons for POLAR | Annotation reference for the real domain |

The Artificial Lunar renders are 720 x 480 px, 8-bit RGB, rendered with sky, terrain and rocks of all sizes."""),
          code("A.dataset_overview()"),
          md("""## 2. Original annotations are incomplete
The Kaggle CSV boxes only cover **large rocks**. The *ground* masks also contain thousands of smaller rocks that a rover camera sees
and that a detector would otherwise learn as background. The table summarises the CSV box geometry."""),
          code('''a = A.load_json("data/artificial_lunar_analysis.json")
bb = a["bounding_boxes"]
pd.DataFrame({k: bb[k] for k in ("area_px", "area_frac_of_image", "aspect_w_over_h")}).T[["count", "min", "p05", "median", "mean", "p95", "max"]]'''),
          code('P.show_png(FIG / "a_bbox_distributions.png", "Original CSV boxes: size, position and shape (Task A)")'),
          code('P.show_png(FIG / "a_objects_per_frame.png", "Objects per frame: CSV boxes vs mask components (Task A)")'),
          md("""## 3. Rendering conditions and author-flagged anomalies
The dataset authors list frames with known problems (camera anomalies, ground-facing views, render/mask mismatches, heavy shadow).
These **573 frames were excluded**, leaving **9,193 eligible frames**."""),
          code('P.show_png(FIG / "a_render_intensity_and_sky.png", "Render brightness and sky fraction (Task A)")'),
          code('P.show_png(FIG / "a_anomaly_list_metrics.png", "Metrics of the author-provided anomaly lists (Task A)")'),
          md("## 4. Data-quality register\nEvery issue found in Task A, with its severity. HIGH issues shaped the later design decisions."),
          code('''q = A.quality_issues()
display(q["severity"].value_counts().rename("issues").to_frame().T)
q'''),
          md("""## 5. Local-only visual examples
Sample renders are displayed only when the local dataset is present **and** `ARTEMIS_SHOW_LOCAL_IMAGES=1`, because the
redistribution licence of the images has not been confirmed. Committed outputs therefore contain no dataset pixels."""),
          code('''if SHOW_LOCAL_IMAGES:
    from PIL import Image
    import matplotlib.pyplot as plt
    ids = [189, 5589, 586]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    for a_, i in zip(ax, ids):
        a_.imshow(Image.open(paths.RAW_RENDERS / f"render{i:04d}.png")); a_.set_title(f"render{i:04d}"); a_.axis("off")
    plt.show()
else:
    print("Dataset images not shown (licence unconfirmed or dataset absent). See docs/DATASET.md to obtain the data.")''')]
    c += footer("""- The synthetic Artificial Lunar renders are the only source large and consistently labelled enough for training.
- The **original CSV labels cover only large rocks** (HIGH issue AL-01), so the rock masks must be used to label smaller rocks. This
  is the subject of the annotation analysis.
- NASA POLAR is a valuable *real-domain* check but needs a validated 12-bit to 8-bit conversion first. It is excluded from all
  reported metrics.
- 376 near-duplicate render pairs mean that splits must be **grouped**, which is handled in notebook 04.""",
                "02_annotation_analysis.ipynb", ["data/artificial_lunar_analysis.json", "data/data_quality_report.json", "figures/a_bbox_distributions.png"])
    return c


def nb02():
    c = header(2, "Annotation Analysis", "Decide how rock boxes should be derived and which boxes are trustworthy enough to train on (Task B, D.2).",
               ["Should labels come from the original CSV (S1), from the semantic masks (S2) or from both?", "How are uncertain boxes identified and handled?",
                "What has, and has not, been verified by humans?"],
               ["artifacts/data/strategy_decision.json", "artifacts/data/strategy_metrics.json", "artifacts/data/annotation_distributions_v1.1.json",
                "artifacts/provenance/VERSIONS.json"])
    c += [md("""## 1. Two candidate labelling strategies
- **S1 - CSV boxes:** the original large-rock boxes. They are precise but cover only about 1.9 rocks per image.
- **S2 - mask components:** a box around every connected rock region in the *ground* mask. This gives about 46 boxes per image,
  including small and distant rocks, but some components merge adjacent rocks or are barely visible.

Task B compared both strategies on a stratified sample and selected a **HYBRID (S2-H)** policy. CSV instances are kept as
authoritative. Mask components of at least 50 px are added, merged regions are split by a watershed (neck ratio 0.7), and each box gets
a **visibility** check (edge strength >= 100 or contrast >= 25)."""),
          code('''d = A.load_json("data/strategy_decision.json")
k = d["key_numbers"]
pd.Series({"recommended strategy": d["recommended_strategy"], "evaluation ontology": d["evaluation_ontology"],
           "CSV-to-mask match rate (strict)": k["csv_match_rate_V0_strict"],
           "additional S2 candidates >= 50 px per image": k["additional_s2_candidates_ge_50px_per_image_mean"],
           "approved for training (at decision time)": d["approved_for_training"]}).to_frame("value")'''),
          code('P.show_png(FIG / "b_s1_vs_s2_counts.png", "Boxes per image: S1 (CSV) vs S2 (mask components), Task B")'),
          md("""## 2. Three annotation states
Each candidate box receives exactly one status:

| Status | Meaning | Use |
|---|---|---|
| **VALID** | Clear, visible, unmerged rock | Training target and evaluation ground truth |
| **REVIEW_REQUIRED** | Plausible rock but uncertain (low visibility, possible merge, shadow) | **Never treated as background**: excluded from training scenes (Config A) or masked (Config B), and ignored in evaluation |
| **REJECTED** | Not a usable rock box (degenerate, duplicate, invisible) | Discarded |"""),
          code("A.curated_counts()"),
          md("Why boxes were sent to REVIEW (training split, v1.1):"),
          code('''dist = A.load_json("data/annotation_distributions_v1.1.json")
rr = pd.DataFrame({s: dist[s]["review_reasons"] for s in ("train", "val", "test")}).fillna(0).astype(int)
rr.sort_values("train", ascending=False)'''),
          code('''src = pd.DataFrame({s: dist[s]["by_source"] for s in ("train", "val", "test")})
src.loc["total"] = src.sum(); src'''),
          md("""## 3. Known annotation weaknesses
- **Merged objects:** clusters of touching boulders can form one mask component. Watershed splitting reduces this, and remaining
  low-solidity components go to REVIEW.
- **Duplicates:** overlapping CSV and mask boxes for the same rock were found in Task D. They were removed at the source in curated
  v1.1 (rule D05).
- **Dark and small rocks:** rocks in black shadow or under 50 px are often invisible. They are REVIEW or out of scope, and the
  evaluation ignores them rather than counting them as misses.
- **Mask artefacts:** crater rims, slopes and shadow regions are sometimes labelled as rocks. These were found again in the Task F
  error analysis."""),
          md("## 4. Human versus automated review"),
          code('''st = A.load_json("data/dataset_status_v1.1.json")
pd.DataFrame([
    {"review": "Task B visual verdicts", "by": "AI assistant (not human)", "items": d["human_review_status"]["rows"], "human decisions": 0},
    {"review": "Task D.2 AI audit of 36 scenes", "by": "AI assistant (not human)", "items": 36, "human decisions": 0},
    {"review": "Human gallery review (E.1b)", "by": "human", "items": st["human_review"]["scenes_reviewed"], "human decisions": st["human_review"]["scenes_reviewed"]},
]).assign(status=lambda x: ["automated only", "automated only", "postponed (hackathon timeline)"])'''),
          md("""**No annotation has been verified by a human.** The AI-assisted reviews are recorded as *automated* evidence with
`human_review: false`. The dataset status remains **PROVISIONAL, NOT FROZEN**.""")]
    c += footer("""- S2-H labels about 15 times more rocks per image than the original CSV. That makes small-rock detection learnable, but also
  makes the labels noisier.
- The three-state policy keeps uncertain rocks out of both the positive and the negative training signal. The ignore-aware
  evaluation in notebook 06 applies the same principle.
- All labels remain automated. The human review (E.1b) is the main open validation gap.""",
                "03_data_preprocessing.ipynb", ["data/strategy_decision.json", "data/annotation_distributions_v1.1.json", "provenance/VERSIONS.json"])
    return c


def nb03():
    c = header(3, "Data Preprocessing and Cleaning", "Explain the S2-H cleaning pipeline that produced curated-dataset-v1.1, and quantify its effect "
               "without re-running it (Tasks C, C v1.1, D.2).",
               ["How does the pipeline turn raw mask components into VALID / REVIEW / REJECTED boxes?", "What changed between v1 and v1.1?",
                "Were the original data preserved?"],
               ["artifacts/data/cleaning_quality_metrics_v1.1.json", "artifacts/data/threshold_study.json", "artifacts/provenance/VERSIONS.json"])
    c += [md("""## 1. Pipeline (implemented in `backend/analysis/dataset_cleaning`)
```mermaid
flowchart LR
  R[render + ground mask + CSV] --> X[extract mask components]
  X --> F[size floor >= 50 px]
  F --> W[watershed split of merged blobs<br/>neck ratio 0.7]
  W --> V[visibility rule<br/>edge >= 100 or contrast >= 25]
  V --> S{status}
  S --> VA[VALID]
  S --> RE[REVIEW_REQUIRED]
  S --> RJ[REJECTED]
  C[CSV large-rock boxes] --> VA
```
The rules are deterministic, and every decision is recorded with its reason (`cleaning_decisions.csv`, local-only)."""),
          code('''q = A.load_json("data/cleaning_quality_metrics_v1.1.json")["automated_full_dataset"]
pd.Series(q["status_counts"]).rename("boxes").to_frame().assign(share=lambda x: x["boxes"] / x["boxes"].sum())'''),
          code('''sbs = pd.Series(q["status_by_source"])
sbs.index = pd.MultiIndex.from_tuples([tuple(k.split("|")) for k in sbs.index], names=["source", "status"])
sbs.unstack(fill_value=0)[["VALID", "REVIEW_REQUIRED", "REJECTED"]]'''),
          md("## 2. Effect on box sizes (raw S2 components vs cleaned VALID)"),
          code('''import numpy as np, matplotlib.pyplot as plt
bins = ["0-100", "100-400", "400-1600", "1600-6400", "6400-inf"]
raw = [q["box_sizes"]["raw_s2"]["area_bins"][b] for b in bins]; val = [q["box_sizes"]["cleaned_valid"]["area_bins"][b] for b in bins]
x = np.arange(len(bins)); fig, ax = plt.subplots(figsize=(10, 3.8))
ax.bar(x - .2, raw, .4, label="raw S2 components"); ax.bar(x + .2, val, .4, label="cleaned VALID")
ax.set_xticks(x); ax.set_xticklabels([b + " px$^2$" for b in bins]); ax.set_ylabel("boxes"); ax.set_title("Box area distribution before and after cleaning")
ax.legend(); plt.show()'''),
          md("""## 3. Version history
`curated-dataset-v1.1` fixed one defect found in Task D: rule **D05** (duplicate CSV and mask boxes) had not been applied to
watershed-split pieces. The change is small and fully registered by content hash."""),
          code('''c = A.curated_counts().set_index("version")[["VALID", "REVIEW_REQUIRED", "REJECTED"]]
c.loc["change v1 -> v1.1"] = c.iloc[1] - c.iloc[0]; c'''),
          md("## 4. Calibration of the size floor (Task D.2 threshold study)\nLowering the 50 px floor would add many tiny, mostly invisible components."),
          code('''t = A.load_json("data/threshold_study.json")
pd.DataFrame(t["table"]).T[["per_image", "median_box_side_px", "visible_by_proxy_rate", "min_side_lt_6px_rate", "low_solidity_merge_risk_rate"]]'''),
          code('P.show_png(FIG / "d2_threshold_curves.png", "Component yield vs size floor (Task D.2)")'),
          md("## 5. Source preservation\nThe source dataset fingerprint was recorded before and after cleaning:"),
          code('''i = A.load_json("data/cleaning_quality_metrics_v1.1.json")["integrity"]
pd.Series({"files": i["source_fingerprint_before"]["files"], "bytes": i["source_fingerprint_before"]["bytes"],
           "sha256 before": i["source_fingerprint_before"]["sha256"][:16] + "...", "sha256 after": i["source_fingerprint_after"]["sha256"][:16] + "...",
           "unchanged": i["unchanged"]}).to_frame("source dataset fingerprint")''')]
    c += footer("""- Cleaning is rule-based, deterministic and fully logged. Original images and annotations were never modified.
- About 52% of candidate boxes are VALID and 38% go to REVIEW. Such a large uncertain share is itself a finding: **label noise is the
  main scientific risk** of this project.
- The 50 px floor is a calibrated compromise. Rocks below it are out of scope and are ignored, not penalised, at evaluation
  time.""", "04_yolo_dataset_preparation.ipynb", ["data/cleaning_quality_metrics_v1.1.json", "data/threshold_study.json", "figures/d2_threshold_curves.png"])
    return c


def nb04():
    c = header(4, "YOLO Dataset Preparation", "Explain how curated-dataset-v1.1 became the leakage-free YOLO dataset yolo-rock-v1.1, why Config A "
               "was trained, and how uncertain regions are handled (Tasks D, D.2).",
               ["How were train/validation/test splits built without leakage?", "What distinguishes Config A from Config B?",
                "Which assumptions does the YOLO format impose?"],
               ["data/manifests/split_manifest.csv", "artifacts/data/split_statistics_v1.1.json", "artifacts/data/masking_audit_summary.json"])
    c += [md(r"""## 1. YOLO label format
Each image has a text file with one line per VALID rock: `0 cx cy w h`. Here 0 is the single class `rock`, and the coordinates are
normalised to the 720 x 480 frame:
$$c_x = \frac{x_0 + x_1}{2W},\quad c_y = \frac{y_0 + y_1}{2H},\quad w = \frac{x_1 - x_0}{W},\quad h = \frac{y_1 - y_0}{H}.$$
REVIEW boxes are **not** written as labels. Images are re-encoded losslessly, and identical files are shared between dataset
versions through NTFS hard links."""),
          md("""## 2. Leakage-free grouped split (80 / 10 / 10)
Near-identical renders would leak information between splits. Frames were therefore grouped by:
- the dataset's duplicate lists;
- a 64-bit difference hash;
- ORB feature matching with RANSAC inliers.

Each **group** was assigned to one split, stratified by difficulty, density and uncertainty (seed 20261011). The split is fixed in
`split_manifest.csv`, which is published as metadata only."""),
          code('''from artemis_x.datasets import registry as R
sc = R.split_counts(); sc'''),
          code("A.split_table()"),
          code('''sv = A.load_json("data/split_statistics_v1.1.json")["split_validation"]
nd = sv["near_duplicates"]["audit"]
pd.Series({"eligible frames": sv["frames"], "leakage groups spanning > 1 split": len(sv["groups_spanning_multiple_splits"]),
           "image pairs checked (ORB audit)": nd["pairs_checked"], "pairs with >= 15 inliers": nd["pairs_ge_15"],
           "missed by retrieval": nd["pairs_ge_15_missed_by_retrieval"], "NASA files in export": sv["nasa_files_in_export"],
           "anomaly frames in export": sv["excluded_anomaly_frames_in_export"]}).to_frame("check")'''),
          code('P.show_png(FIG / "d_split_strata.png", "Stratification of the three splits (Task D)")'),
          md("""## 3. Two training configurations
| | **Config A - conservative (trained)** | Config B - masked (prepared, not trained) |
|---|---|---|
| Training scenes | 1,903 scenes with REVIEW / (VALID + REVIEW) <= 0.25 | all 7,353 training scenes |
| REVIEW regions | absent from training scenes by selection | painted out with local background texture |
| Risk | fewer images; some uncertain rocks remain unlabelled | in-painting artefacts; erased real rocks |
| Validation / test | identical unmasked images and labels for both | |

**Why Config A was trained first.** Config A changes no pixels. The Task D.2 masking audit showed that Config B erases about 150,000
REVIEW boxes, including about 21,500 large ones, and that the in-painted texture is measurably flatter than real terrain. These are
risks a model could learn as a shortcut. A full A-versus-B comparison is still possible because Config B is preserved."""),
          code('''m = A.load_json("data/masking_audit_summary.json")
pd.Series({"REVIEW boxes erased (Config B)": m["erased_review_boxes_total"], "of which large": m["erased_large_review_boxes_total"],
           "changed pixels inside VALID boxes": m["changed_pixels_inside_valid_boxes_total"],
           "images with low texture ratio (< 0.5)": m["images_texture_ratio_lt_0.5"],
           "median masked fraction of an image": m["masked_fraction"]["median"]}).to_frame("Config B masking audit")'''),
          code('P.show_png(FIG / "d_split_box_distributions.png", "Box-size distributions per split (Task D)")'),
          md("## 4. Integrity of the local dataset copy (if present)\nThe content hashes pinned in `VERSIONS.json` are compared with the files on this machine."),
          code('''R.verify_local_yolo_dataset() if paths.local_dataset_available() else "local dataset not present - registry hashes are in artifacts/provenance/VERSIONS.json"''')]
    c += footer("""- Splits are grouped and stratified. No leakage group spans two splits, and the ORB audit found no missed near-duplicates.
- Validation and test contain **all** eligible scenes with VALID labels. REVIEW regions are not labels, and the evaluation treats
  them as ignore regions.
- Config A (1,903 scenes) was trained. Config B is preserved for a future controlled comparison.""",
                "05_model_training_analysis.ipynb", ["data/split_statistics_v1.1.json", "data/masking_audit_summary.json", "figures/d_split_strata.png"])
    return c


def nb05():
    c = header(5, "Model Training Analysis", "Document the single authorised YOLO11n training run (E.1c) and analyse its convergence and hardware "
               "behaviour from the saved logs. **This notebook never starts training.**",
               ["Which architecture and settings were used, and why?", "Did training converge, and did it overfit?", "Was the run technically sound?"],
               ["artifacts/training/results.csv", "artifacts/training/e1c_args.yaml", "artifacts/training/e1c_summary.json",
                "artifacts/training/checkpoint_inventory.json"])
    c += [md("""## 1. Model and settings
**YOLO11n** (Ultralytics 8.4.174) is a single-stage, anchor-free detector with about 2.6 M parameters. It was fine-tuned from the
official COCO-pretrained `yolo11n.pt` (sha256 `0ebbc80d…`). It was chosen for its latency budget: a rover-side detector must run in
real time on modest hardware.

Training was launched once, through the guarded wrapper `backend/training/train.py`. That wrapper:
- requires an explicit authorisation flag and environment variable;
- verifies the effective arguments before the first batch;
- blocks Ultralytics' automatic OOM batch-halving, NaN epoch replay and time-based epoch changes;
- records telemetry every second.

Historical command, shown for reference and **not executed here**:
```
ARTEMIS_TRAINING_AUTHORIZED=1 uv run --no-sync python -m training.train --arm A_steps --seed 42 --authorized --fresh --data-view \\
    --run-dir ../experiments/task-e/full-training-config-a-200-s42 --set epochs=200 save_period=10 imgsz=736 batch=16 workers=4 ...
```"""),
          code('''import yaml
args = yaml.safe_load((paths.ARTIFACTS / "training/e1c_args.yaml").read_text(encoding="utf-8"))
keys = ["model", "pretrained", "epochs", "imgsz", "batch", "workers", "optimizer", "lr0", "lrf", "momentum", "weight_decay", "cos_lr",
        "warmup_epochs", "amp", "seed", "deterministic", "single_cls", "close_mosaic", "mosaic", "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v",
        "patience", "save_period", "cache"]
pd.DataFrame({"value": [str(args.get(k)) for k in keys]}, index=keys)   # str: no display rounding'''),
          code('''s = A.training_summary(); t = s["telemetry"]
pd.Series({"status": s["status"], "epochs": s["epochs_recorded"], "wall-clock (h)": s["wall_clock_h_epochs"],
           "epoch duration median (s)": s["epoch_duration_s"]["median"], "training throughput median (img/s)": s["img_per_s_epoch"]["median"],
           "GPU util, training windows, median (%)": t["gpu_util_train_window"]["median"], "peak VRAM (MiB)": t["vram_mib"]["max"],
           "max GPU temperature (C)": t["temp_c"]["max"], "thermal-throttle samples": t["thermal_slowdown_samples"],
           "min available RAM (GB)": t["ram_available_gb"]["min"],
           "hardware": "NVIDIA RTX 5060 Laptop GPU (8 GB), AMP fp16"}).to_frame("E.1c run")'''),
          md("## 2. Losses"),
          code('''h = A.training_history()
P.training_losses(h)'''),
          md("""Training losses decrease throughout. Validation losses reach their minimum between epochs 104 and 138 and then drift
slightly upward. That is **mild overfitting**, possibly to label noise. The step at epoch 191 is the planned switch-off of mosaic
augmentation for the last 10 epochs."""),
          code("P.validation_metrics(h)"),
          md("""Ultralytics' built-in validation counts detections on REVIEW regions as false positives, so its mAP is **provisional** and
was not used for model selection (see notebook 06)."""),
          code('''m = pd.DataFrame(s["milestones"]).T
m[["box", "cls", "vbox", "vcls", "P", "R", "m50", "m5095", "lr_pg0"]].rename(columns={"vbox": "val box", "vcls": "val cls", "m50": "mAP50", "m5095": "mAP50-95"})'''),
          md("## 3. Hardware behaviour"),
          code('''e = A.epoch_performance()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
ax[0].plot(e["epoch"], e["train_s"], label="training"); ax[0].plot(e["epoch"], e["val_save_s"], label="validation + save"); ax[0].set_title("Seconds per epoch"); ax[0].legend()
ax[1].plot(e["epoch"], e["img_per_s_epoch"]); ax[1].set_title("Training throughput (images/s)")
for a in ax: a.set_xlabel("epoch")
plt.show()'''),
          code('P.show_png(FIG / "e_hardware_telemetry.png", "GPU utilisation, VRAM, temperature/power and RAM over the 69-minute run")'),
          md("""Two short slow phases (epochs 124-132 and 161-169) coincided with **lower** GPU power and temperature but **higher**
CPU use. They are attributed to host-side slowdown; that is a hypothesis, because CPU clocks were not logged. There was no OOM, NaN,
repeated epoch or thermal throttling."""),
          md("## 4. Checkpoints\n42 checkpoints were verified (loadable, finite, class `rock`). `epochN.pt` follows Ultralytics' 0-based naming (N+1 completed epochs); the `epoch_completed_*` copies are aligned to completed epochs."),
          code('''inv = A.checkpoint_inventory()
inv[inv["file"].isin(["best.pt", "last.pt", "epoch_completed_050.pt", "epoch_completed_100.pt", "epoch_completed_120.pt", "epoch_completed_200.pt", "epoch10.pt"])]''')]
    c += footer("""- One 200-epoch run of YOLO11n on Config A (seed 42) completed in 68.7 minutes. It was technically clean: no OOM, NaN or
  throttling, and every checkpoint was verified.
- The model **converged by about epoch 100-120**, with mild overfitting afterwards. Training longer is not justified by these data.
- Ultralytics' own validation metrics are provisional. Checkpoint choice uses the ignore-aware evaluation in the next notebook.""",
                "06_model_selection.ipynb", ["training/results.csv", "training/e1c_summary.json", "training/e1c_args.yaml", "figures/e_hardware_telemetry.png"])
    return c


def nb06():
    c = header(6, "Model Selection", "Show how the deployed checkpoint (epoch 111) and its operating point (confidence 0.20) were selected "
               "using the validation split only, and why (Task F). **No new selection is made here.**",
               ["How do the 22 saved checkpoints compare under a fair, ignore-aware metric?", "Which confidence threshold balances recall and false alarms?",
                "How well are large obstacles detected?"],
               ["artifacts/evaluation/ignore_aware_val_22_checkpoints.json", "artifacts/evaluation/selection_analysis.json",
                "artifacts/provenance/FROZEN_MODEL_CONFIG.json"])
    c += [md(r"""## 1. Ignore-aware evaluation
Standard COCO/Ultralytics evaluation counts a detection on an uncertain (REVIEW) rock as a false positive. That penalises the model for
finding real rocks that the automated labels did not confirm. The Artemis-X protocol (Task D.2, `artemis_x.evaluation.ignore_aware`)
follows COCO *crowd* semantics:

- a prediction matching a VALID box with IoU >= 0.5 (greedy by confidence) is a **TP**;
- an unmatched prediction with $\mathrm{IoU} \ge 0.5$ or $\mathrm{IoA} = \frac{|p \cap r|}{|p|} \ge 0.5$ against any REVIEW region $r$ is **ignored**;
- every other prediction is an **FP**, and every unmatched VALID box is an **FN**;
- AP uses 101-point interpolation. AP50-95 averages IoU thresholds 0.50:0.95:0.05. Predictions under 50 px$^2$ are removed.

Three policies are always reported **separately**:
- **primary:** REVIEW regions ignored (approved);
- **secondary:** also ignores the *proposed* sub-threshold regions (not approved);
- **no ignores:** for reference."""),
          code('''cmp = A.checkpoint_comparison()
P.checkpoint_curve(cmp)
cmp'''),
          md("""All three policies tell the same story: performance rises quickly until about epoch 40 and then **plateaus**. The gap of
about 0.12 AP50 between the primary policy and the no-ignore policy comes from the **evaluation policy, not from the model**."""),
          md("## 2. Selection rules (fixed in code before any threshold metric was computed)"),
          code('''pd.Series(A.selection_rules()).to_frame("rule")'''),
          code('''A.candidate_summary(0.20)'''),
          md("""No comparison candidate beat `best.pt` (epoch 111) by more than 0.005 AP50, so `best.pt` was kept. **Epoch 120** has the
same AP and about 1.7 points better large-rock recall. It was presented to the project owner, who chose to keep epoch 111."""),
          md("## 3. Operating point"),
          code('''tab = A.threshold_table("best.pt")
P.threshold_tradeoff(tab, 0.20)
tab[["conf", "precision", "recall", "f1", "f2", "fp_per_image", "tp", "fp", "fn"]]'''),
          md("""The best F1 on the grid is at 0.25. The **recall-oriented rule** selects the lowest threshold whose F1 is within 0.02 of
that maximum: **0.20**. Compared with 0.25 it finds about 900 more rocks (+3.5 points recall) for about 2.5 more false alarms per image.
For a rover, a missed obstacle costs more than a false alarm."""),
          md("## 4. Recall by rock size (validation)"),
          code('''sel = A.load_json("evaluation/selection_analysis.json")["candidates"]["best.pt"]["size_recall"]
pd.DataFrame(sel).T[["gt", "recall@0.1", "recall@0.15", "recall@0.2", "recall@0.25", "precision@0.2"]].loc[A.SIZE_ORDER]'''),
          code('''fn = A.load_json("evaluation/selection_analysis.json")["fn_analysis_selected"]
pd.DataFrame(fn["by_size"]).T.assign(total=lambda x: x.sum(axis=1))'''),
          md("""About 60% of misses are rocks the model *did* find below the threshold, about 23% are localisation errors (merged or
split boxes), and about 17% are near-total misses. Visual inspection (local-only) found that many "missed large rocks" are label
artefacts, such as crater rims, slopes and merged boulders.

Genuine failures concentrate in:
- dark, low-illumination scenes;
- dense boulder fields;
- large shadowed boulders against the black sky."""),
          md("## 5. The frozen configuration"),
          code('''f = A.frozen_config()
pd.concat([pd.Series(f["inference"]).drop("note"), pd.Series({"model sha256": f["model"]["sha256"], "selected on": f["selected_on"],
          "test split used for selection": f["test_split_used"]})]).to_frame("frozen value")''')]
    c += footer("""- Selection used **only the validation split** and pre-declared rules.
- Frozen model: `best.pt`, epoch 111. Operating point: confidence 0.20, NMS IoU 0.60, max 300 detections, boxes under 50 px$^2$
  removed. Evaluation policy: REVIEW regions ignored.
- Validation at that operating point: precision 0.696, recall 0.775, F1 0.734, about 9.5 false positives per image, AP50 0.776.
- Large-obstacle recall (0.807) is lower than medium-rock recall (0.849). That is the most safety-relevant weakness.""",
                "07_final_evaluation.ipynb", ["evaluation/ignore_aware_val_22_checkpoints.json", "evaluation/selection_analysis.json", "provenance/FROZEN_MODEL_CONFIG.json"])
    return c


def nb07():
    c = header(7, "Final Evaluation (once-only test)", "Present the single, locked evaluation of the frozen model on the 917-image held-out test split. "
               "**This notebook reads saved results only; it never calls the evaluator and never reads test images.**",
               ["How does the frozen model perform on unseen test scenes?", "Do test results agree with validation?",
                "Where does the detector fail, especially on large rocks?"],
               ["artifacts/evaluation/test_results.json", "artifacts/provenance/TEST_RUN_LOCK.json", "artifacts/figures/f_test_*.png"])
    c += [md("""## 1. Protocol and lock
The test split was evaluated **once**, after the model, threshold, NMS and evaluation policy had been frozen and their SHA-256 hashes
verified. A lock file prevents any second run, and test results were never used for tuning."""),
          code('''import json
lock = A.load_json("provenance/TEST_RUN_LOCK.json"); r = A.test_results()
pd.Series({"started": lock["started"], "finished": lock["finished"], "frozen config sha256": lock["frozen_config_sha256"][:16] + "...",
           "model sha256": lock["model_sha256"][:16] + "...", "test images": 917, "protocol": r["protocol"],
           "max_det truncation affecting conf 0.20": r["max_det_check"]["images_where_cap_affects_conf_0.20"],
           "protected data unchanged after run": all(r["protected_trees_unchanged"].values())}).to_frame("once-only test evaluation")'''),
          md("## 2. Headline results (confidence 0.20)"),
          code("A.test_policy_table()"),
          md("""The **primary** row is the official result. Recall is identical under every policy, because ignore regions change how
predictions are counted, never the ground truth. Test agrees with validation to within ±0.004 on every headline metric, so the selection
did not overfit the validation split.

That agreement is **not** evidence of generalisation to real lunar imagery. Both splits come from the same synthetic distribution and
the same automated labelling."""),
          code('P.show_png(FIG / "f_test_evaluation.png", "Test: PR curve with frozen operating point; recall by size; policy comparison")'),
          md("## 3. Recall by rock size, with 95% Wilson intervals"),
          code('''from artemis_x.evaluation.metrics import wilson_interval
st = A.test_size_table()
ci = [wilson_interval(round(r * n), n) for r, n in zip(st["recall test"], st["GT (test)"])]
st["95% CI"] = [f"{lo:.3f}-{hi:.3f}" for lo, hi in ci]
P.size_recall(st)
st'''),
          md("""About **one large rock in five is missed** (recall 0.788 for rocks of at least 2,500 px$^2$). The very-large bin is small
(n = 224), so its val/test difference (0.760 vs 0.821) is within sampling noise. The intervals assume independent boxes, so the true
uncertainty is larger."""),
          md("## 4. Error analysis"),
          code('''pb = pd.DataFrame({k: v["prediction_breakdown"] for k, v in r["policies"].items()})
pb'''),
          code('''fn = r["fn_analysis_primary"]
pd.DataFrame(fn["by_size"]).T.assign(total=lambda x: x.sum(axis=1))'''),
          code('''edge = A.load_json("evaluation/large_rock_edge_analysis.json")["results"]
pd.DataFrame({s: {k: f"{v['recall']:.3f} (n={v['gt']})" for k, v in d.items()} for s, d in edge.items()})'''),
          code('P.show_png(FIG / "f_test_confusion_analysis.png", "Prediction outcomes and false-negative categories (test, primary policy)")'),
          md("""**Qualitative failure cases** (local-only overlays, not committed):
- craters, cliff faces and touching-boulder clusters labelled as a single rock (label artefacts);
- isolated near-field boulders at the bottom of the frame;
- rocks in deep shadow.

The near-field (bottom-edge) effect seen visually is **not** confirmed statistically: 0.751 vs 0.792, within the uncertainty."""),
          md("""## 5. Limitations
1. **Labels:** automated and not human-validated. Visible artefacts inflate the large-rock miss counts, and other label errors could
   bias any metric in either direction.
2. **Policy dependence:** AP50 ranges from 0.664 to 0.803 across policies. Only the primary policy is approved.
3. **Domain:** synthetic renders only. No real lunar imagery and no Webots imagery have been evaluated.
4. **Statistics:** one training seed; size-bin intervals assume independent boxes.
5. **Safety:** missing about 21% of large rocks means this detector alone is **not sufficient** for obstacle avoidance. A rover needs
   redundant hazard sensing (stereo or depth), temporal tracking and conservative margins.""")]
    c += footer("""- On the held-out test split, the frozen detector reaches precision 0.698, recall 0.774, F1 0.734, 9.48 false positives per
  image, AP50 0.777 and AP50-95 0.428. These figures use the primary policy and are **provisional**.
- Results reproduce validation closely. The dominant open questions are label quality and real-domain transfer, not overfitting
  during model selection.""",
                "08_inference_demo.ipynb", ["evaluation/test_results.json", "provenance/TEST_RUN_LOCK.json", "figures/f_test_evaluation.png"])
    return c


def nb08():
    c = header(8, "Inference Demonstration", "Load the frozen detector and run it with exactly the frozen settings, on GPU when available and "
               "otherwise on CPU. Only a small demonstration inference is performed: no training and no dataset processing.",
               ["How is the frozen model loaded and verified?", "What does its output look like?", "How fast is it?"],
               ["models/artemis_rock_yolo11n.pt", "artifacts/provenance/FROZEN_MODEL_CONFIG.json"])
    c += [md("""## 1. Load and verify the frozen model
`load_frozen_spec()` reads `FROZEN_MODEL_CONFIG.json` and refuses to continue if the weights' SHA-256 differs from the frozen value."""),
          code('''from artemis_x.ml.model_registry import load_frozen_spec
from artemis_x.ml.inference import RockDetector, default_device
spec = load_frozen_spec()
pd.Series({"weights": paths.rel(spec.weights), "sha256 (verified)": spec.sha256, "class names": spec.class_names,
           "imgsz": spec.imgsz, "confidence": spec.conf, "NMS IoU": spec.nms_iou, "max detections": spec.max_det,
           "min box area (px^2)": spec.min_box_area_px2, "selected epoch": spec.epoch, "device": default_device()}).to_frame("frozen detector")'''),
          code('''det = RockDetector(spec)          # CUDA if available, else CPU
det.warmup()'''),
          md("""## 2. Example inputs
**No redistribution-cleared lunar images are bundled with this repository.** The licence of the Artificial Lunar Landscape renders
has not been confirmed, so no dataset pixels are committed. To try the model on real inputs:
1. download the Artificial Lunar Landscape dataset (see `docs/DATASET.md`) into `dataset/artificial-lunar/`, or
2. point `IMAGE_PATHS` below at your own 720 x 480 lunar images.

Run with `ARTEMIS_SHOW_LOCAL_IMAGES=1` to draw the detections. Without images, the cell below runs a **synthetic** frame only, to
demonstrate the API and timing. A blank frame is expected to produce no detections."""),
          code('''import numpy as np, time
IMAGE_PATHS = []
if paths.local_dataset_available():
    # validation scenes only - the held-out test split is never used for demonstrations
    IMAGE_PATHS = [paths.YOLO_DATASET / "images" / "val" / f"render{i:04d}.png" for i in (115, 5589, 575)]
rows = []
for p in IMAGE_PATHS:
    res = det.predict(p)
    c = res.as_array()[:, 4]
    rows.append({"image": p.name, "detections": len(res.detections), "removed < 50 px^2": res.removed_small_boxes,
                 "max conf": round(float(c.max()), 3) if len(c) else None, "median conf": round(float(np.median(c)), 3) if len(c) else None,
                 "latency (ms)": round(res.inference_ms, 1), "device": res.device})
    if SHOW_LOCAL_IMAGES:
        from PIL import Image
        P.draw_detections(np.asarray(Image.open(p).convert("RGB")), res.as_array(), f"{p.name}: {len(res.detections)} rocks @ conf >= {spec.conf}")
synthetic = det.predict(np.full((480, 720, 3), 90, dtype=np.uint8))
rows.append({"image": "synthetic grey frame", "detections": len(synthetic.detections), "removed < 50 px^2": synthetic.removed_small_boxes,
             "max conf": None, "median conf": None, "latency (ms)": round(synthetic.inference_ms, 1), "device": synthetic.device})
pd.DataFrame(rows)'''),
          md("## 3. Latency\nMedian of 20 single-image runs. This includes Ultralytics pre- and post-processing, but not disk I/O."),
          code('''frame = np.random.default_rng(0).integers(0, 255, (480, 720, 3), dtype=np.uint8)
def bench(d, n=20):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); d.predict(frame); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 95))
out = {det.device: bench(det)}
if det.device != "cpu":
    out["cpu"] = bench(RockDetector(spec, device="cpu"), n=5)
pd.DataFrame(out, index=["median ms", "p95 ms"]).T.round(1)'''),
          md("""Historical reference: on the RTX 5060 Laptop GPU, the Task F parity check measured a median of 15.3 ms per 720 x 480 image,
including file I/O. Live inference reproduced the cached validation predictions exactly (1,904 of 1,904 boxes on 50 images).

**Using the detector in code**
```python
from artemis_x.ml.inference import RockDetector
det = RockDetector()                    # verifies the frozen SHA-256, picks CUDA or CPU
result = det.predict("frame.png")       # frozen imgsz/conf/NMS/max_det + 50 px^2 post-filter
for d in result.detections: print(d.x0, d.y0, d.x1, d.y1, d.confidence)
```""")]
    c += footer("""- The frozen detector loads with checksum verification and runs on CPU or GPU through one call.
- Future Webots integration (planned, not implemented) will feed simulator camera frames to the same `RockDetector`. A domain-gap
  check on Webots frames must come first; see `future/WEBOTS_INTEGRATION_PLAN.md`.""",
                None, ["provenance/FROZEN_MODEL_CONFIG.json", "evaluation/inference_check.json"])
    return c


def nb09():
    c = header(9, "Reproduce the YOLO11n Training (opt-in)", "Let a user with the prepared dataset and an NVIDIA GPU re-run the historical "
               "200-epoch E.1c experiment with exactly the same configuration, through the same guarded trainer. **Training is "
               "disabled by default. Run All only performs checks and a dry run.**",
               ["What exactly was trained, and with which settings and initialisation?", "Is this machine and dataset ready to reproduce it?",
                "How is a new run launched safely, and how should its results be compared?"],
               ["artifacts/training/e1c_train_config.json", "artifacts/training/e1c_args.yaml", "data/release/yolo-rock-v1.1-configA.files.sha256",
                "backend/training/train.py (guarded trainer)"])
    c += [md("""## 1. The historical experiment (E.1c)
| Item | Value |
|---|---|
| Model | YOLO11n, initialised from the official COCO-pretrained `yolo11n.pt` (SHA-256 `0ebbc80d…44ee1`) |
| Data | `yolo-rock-v1.1` **Config A**: 1,903 training scenes; validation 923 |
| Schedule | 200 epochs, batch 16, imgsz 736, SGD (lr0 0.01, momentum 0.937, wd 5e-4), cosine LR, 3 warm-up epochs, mosaic closed for the last 10 epochs |
| Precision / seed | AMP fp16, `deterministic=true`, seed 42 |
| Hardware / time | RTX 5060 Laptop GPU (8 GB): **68.7 min**, peak VRAM 5.5 GB, peak system RAM 25.9 GB |
| Outcome | Converged by about epoch 100-120; selected checkpoint at epoch 111 |"""),
          code('''from artemis_x.ml import training_repro as T
h = T.historical_config()
pd.Series({"arm": h["arm"]["id"], "seed": h["seed"], "status": h["status"], "epochs completed": h["epochs_completed"],
           **{f"override: {k}": str(v) for k, v in h["overrides"].items()}}).to_frame("E.1c")'''),
          code('''keys = ["model", "pretrained", "epochs", "batch", "imgsz", "optimizer", "lr0", "lrf", "momentum", "weight_decay", "cos_lr",
        "warmup_epochs", "close_mosaic", "mosaic", "fliplr", "hsv_v", "amp", "seed", "deterministic", "single_cls", "patience", "save_period"]
pd.DataFrame({"E.1c effective value": [str(h["effective_args"].get(k)) for k in keys]}, index=keys)   # str: no rounding'''),
          md("""## 2. Requirements
- **Dataset:** the prepared release `yolo-rock-v1.1-configA` (about 1.8 GB). It is **not published yet**, pending licence
  approval; see `docs/DATASET_DISTRIBUTION_PLAN.md`. Once a release exists:
  `uv run --project backend python scripts/download_dataset.py --source <approved source>`.
  It verifies all 7,490 files against the committed SHA-256 manifest.
- **Pretrained weights:** `backend/yolo11n.pt`, the official Ultralytics asset. Its SHA-256 is checked below.
- **Hardware:** an NVIDIA GPU with at least 8 GB VRAM (5.5 GB peak observed), at least 32 GB system RAM recommended (validation
  peaked at 25.9 GB used), and about 1.5 GB of free disk per run (checkpoints every 10 epochs). CPU-only training is technically
  possible but was not measured and would be impractically slow.
- **Software:** the locked environment (`backend/uv.lock`: torch 2.14.1 + CUDA 13.0, Ultralytics 8.4.174)."""),
          md("## 3. Preflight (no training)\nDataset verification hashes all labels and manifests and every 50th image (`full_hash=True` hashes everything)."),
          code('''pf = T.preflight(full_hash=False)
ds = pf["dataset"]
display(pd.Series({"dataset files in manifest": ds.get("files_in_manifest"), "hashed": ds.get("hashed"), "missing": ds.get("n_missing"),
                   "mismatched": ds.get("n_mismatched"), "split counts as expected": ds.get("counts_match_expected"), "dataset ok": ds.get("ok")}).to_frame("prepared dataset"))
display(pd.Series(pf["pretrained"]).to_frame("pretrained initialisation"))
display(pd.Series(pf["environment"]).to_frame("environment"))
print("READY TO REPRODUCE:", pf["ready"])'''),
          md("""## 4. Plan the run and resolve the exact arguments (dry run)
Outputs go to `experiments/reproductions/<RUN_NAME>/`. The planner refuses to write into `experiments/task-e`,
`experiments/task-f`, `models/` or `artifacts/`, and refuses to reuse a run folder that already holds checkpoints. A reproduction
appends to `experiments/reproductions/reproductions_registry.jsonl`, never to the historical registry. The dry run asks the real
trainer to resolve its arguments without touching the GPU."""),
          code('''RUN_NAME = "repro-config-a-200-s42"
plan = None
if pf["dataset"].get("ok"):
    plan = T.plan_run(name=RUN_NAME)
    resolved = T.dry_run(plan)
    a = resolved["ultralytics_args"]
    display(pd.Series({"run folder": resolved["run_dir"], **{k: a[k] for k in ("data", "epochs", "batch", "imgsz", "workers", "seed", "optimizer",
                       "lr0", "cos_lr", "amp", "deterministic", "single_cls", "save_period", "close_mosaic")}}).to_frame("resolved by the trainer"))
else:
    print("Prepared dataset not available - see section 2. Nothing planned.")'''),
          md("""## 5. Launch (disabled by default)
Training starts only if **both** values below are changed by hand: `RUN_TRAINING = True`, and `AUTHORISATION` set to the exact
phrase `I authorise one 200-epoch YOLO11n training run`. The trainer then additionally requires its own authorisation flag and
environment variable, which are set for that single subprocess only. It refuses to resume, change the batch size after an OOM, replay
NaN epochs, or exceed 200 epochs. **Under Run All this cell does nothing.**"""),
          code('''RUN_TRAINING = False      # must be changed by hand
AUTHORISATION = ""        # must equal T.AUTH_PHRASE exactly
exit_code = T.launch(plan, run=RUN_TRAINING, authorisation=AUTHORISATION) if plan else None
exit_code'''),
          md(r"""## 6. Comparing a reproduction with the original
Compare on the **validation split only**: the held-out test split has already been used once and must not be used again for
model choice.

1. Plot the reproduced `results.csv` against `artifacts/training/results.csv` (cell below).
2. Score the reproduced checkpoints with the ignore-aware evaluator
   (`backend/training/evaluate.py --weights <run>/weights/best.pt --split val`).
3. Treat differences of about ±0.01 AP50 as noise: single-seed checkpoint variability was of that order in E.1c.

**Numerical reproducibility limits:**
- GPU kernels (cuDNN, AMP) are not bitwise deterministic, even with `deterministic=true` and seed 42.
- Results also depend on the GPU model, driver, CUDA, PyTorch and Ultralytics versions, and on the number of data-loader workers.
- Expect similar curves and metrics, not identical weights. The frozen model (`models/artemis_rock_yolo11n.pt`) is never
  replaced by a reproduction."""),
          code('''import matplotlib.pyplot as plt
hist = A.training_history()
runs = sorted((paths.EXPERIMENTS / "reproductions").glob("*/results.csv")) if (paths.EXPERIMENTS / "reproductions").exists() else []
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].plot(hist["epoch"], hist["metrics/mAP50(B)"], label="E.1c (historical)", lw=2)
ax[1].plot(hist["epoch"], hist["val/box_loss"], label="E.1c (historical)", lw=2)
for r in runs:
    d = pd.read_csv(r); d.columns = [x.strip() for x in d.columns]
    ax[0].plot(d["epoch"], d["metrics/mAP50(B)"], label=r.parent.name); ax[1].plot(d["epoch"], d["val/box_loss"], label=r.parent.name)
ax[0].set_title("Ultralytics val mAP50 (PROVISIONAL)"); ax[1].set_title("Validation box loss")
for a_ in ax: a_.set_xlabel("epoch"); a_.legend(fontsize=8)
plt.show()
print(f"{len(runs)} reproduction(s) found")''')]
    c += footer("""- The historical configuration is fully specified and resolved by the same guarded trainer that produced E.1c.
- Reproducing it needs the prepared dataset release (pending licence approval), the official pretrained weights, an 8 GB NVIDIA GPU
  and about 70 minutes.
- Reproductions are isolated from the historical evidence, and the frozen model is never replaced.""",
                None, ["training/e1c_train_config.json", "training/e1c_args.yaml", "training/results.csv"])
    return c


NOTEBOOKS = {"01_dataset_exploration.ipynb": nb01, "02_annotation_analysis.ipynb": nb02, "03_data_preprocessing.ipynb": nb03,
             "04_yolo_dataset_preparation.ipynb": nb04, "05_model_training_analysis.ipynb": nb05, "06_model_selection.ipynb": nb06,
             "07_final_evaluation.ipynb": nb07, "08_inference_demo.ipynb": nb08, "09_reproduce_training.ipynb": nb09}


def check_source(name: str, nb) -> None:
    src = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    bad = [p for p in FORBIDDEN_SOURCE if re.search(p, src)]
    if bad:
        raise SystemExit(f"{name}: forbidden pattern(s) in code: {bad}")


def check_outputs(name: str, nb) -> None:
    for i, c in enumerate(nb.cells):
        for o in c.get("outputs", []):
            text = str(o.get("text", "")) + str(o.get("data", {}).get("text/plain", "")) + str(o.get("data", {}).get("text/html", "")) + str(o.get("traceback", ""))
            for p in FORBIDDEN_OUTPUT:
                if re.search(p, text):
                    raise SystemExit(f"{name}: cell {i} output contains a local path ({p})")
            if o.get("output_type") == "error":
                raise SystemExit(f"{name}: cell {i} raised {o.get('ename')}: {o.get('evalue')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-execute", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    os.environ.pop("ARTEMIS_SHOW_LOCAL_IMAGES", None)      # committed outputs never contain dataset pixels
    NB_DIR.mkdir(exist_ok=True)
    for name, fn in NOTEBOOKS.items():
        if a.only and not any(name.startswith(o) for o in a.only):
            continue
        nb = nbf.v4.new_notebook()
        nb.cells = fn()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["language_info"] = {"name": "python"}
        check_source(name, nb)
        if not a.no_execute:
            from nbclient import NotebookClient
            NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}}).execute()
            check_outputs(name, nb)
        nbf.validate(nb)
        nbf.write(nb, NB_DIR / name)
        size = (NB_DIR / name).stat().st_size
        print(f"{name}: {len(nb.cells)} cells, {size / 1024:.0f} KB, executed={not a.no_execute}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

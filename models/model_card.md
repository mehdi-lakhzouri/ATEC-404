# Model card: `artemis_rock_yolo11n.pt`

> **The training and evaluation labels are automated and not human-verified. All metrics are provisional and synthetic-only.**

## Identity

| Item | Value |
|---|---|
| File | `models/artemis_rock_yolo11n.pt` (5.48 MB, stripped inference checkpoint) |
| SHA-256 | `413d0481a728a310f5b919d069676ee5ddd2091e81b4b663831e1e977a0752eb` |
| Source | Training run E.1c `full-training-config-a-200-s42`, `best.pt`; the Ultralytics fitness peak is epoch 111 |
| Architecture | YOLO11n (Ultralytics 8.4.174), 2,590,035 parameters, 1 class `{0: "rock"}` |
| Initialisation | Official COCO-pretrained `yolo11n.pt` (SHA-256 `0ebbc80d…44ee1`) |
| Frozen configuration | `artifacts/provenance/FROZEN_MODEL_CONFIG.json` (SHA-256 `efdc5802…44bf4`) |

## Intended use

- Research and demonstration of rock and obstacle detection in **synthetic lunar rover imagery** at about 720×480.
- Input to a future Webots simulation (planned).

**Not intended for:**
- safety-critical navigation on its own;
- real lunar imagery without validation;
- any deployment claim.

## Inference settings (frozen)

| Setting | Value |
|---|---|
| imgsz | 736 |
| confidence | 0.20 |
| NMS IoU | 0.60 |
| max detections | 300 |
| post-filter | remove boxes < 50 px² |

Use `artemis_x.ml.inference.RockDetector`, which applies all of these.

## Training data

- **Data:** `yolo-rock-v1.1` Config A: 1,903 synthetic scenes from the Artificial Lunar Landscape dataset, with automated S2-H
  VALID boxes. Scenes with uncertain (REVIEW) rocks were excluded from training.
- **Validation:** 923 scenes. **Test:** 917 scenes (evaluated once).
- **Training settings:** 200 epochs, batch 16, SGD + cosine, AMP, seed 42.

## Performance (held-out synthetic test, primary ignore-aware policy, conf 0.20)

| Metric | Value |
|---|---|
| Precision | 0.698 |
| Recall | 0.774 |
| F1 | 0.734 |
| FP / image | 9.48 |
| AP50 | 0.777 |
| AP50-95 | 0.428 |

| Size bin | Recall |
|---|---|
| small < 400 px² | 0.743 |
| medium 400–2,500 px² | 0.842 |
| large ≥ 2,500 px² | 0.788 |

**Known failure modes:** dark or shadowed scenes, dense boulder fields, large shadowed boulders against the sky, and merged
clusters of touching rocks.

## Licensing and provenance notes

- **Ultralytics licence.** The checkpoint's own metadata records `license: AGPL-3.0 (https://ultralytics.com/license)`. Models
  trained with Ultralytics YOLO are subject to the AGPL-3.0 unless an Ultralytics Enterprise licence applies. The owner should
  confirm the publication terms.
- **Training data licence.** The licence of the training data (Artificial Lunar Landscape) has **not been confirmed**. Whether
  publishing weights trained on it is permitted must be checked.
- **Embedded paths.** The checkpoint embeds Ultralytics' `train_args`, which include **absolute paths of the training machine**
  (local user directory). Removing them would change the frozen SHA-256, so the file was left unchanged. Review this before public
  release.

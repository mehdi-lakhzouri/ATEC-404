# Limitations

## Scientific

1. **Automated labels.** Every training and evaluation label comes from the S2-H rules. No annotation has been verified by a
   human; the AI-assisted reviews are not human validation. The true precision and recall of the detector are therefore unknown,
   and visible label artefacts (craters, slopes, merged boulders) can bias metrics in either direction.
2. **Policy-dependent metrics.** AP50 on test ranges from 0.664 to 0.803 depending on how uncertain regions are counted. Only the
   primary policy (REVIEW regions ignored) is approved.
3. **Synthetic domain only.** No real lunar imagery (NASA POLAR, the PCAM/TCAM photos) and no Webots imagery has been evaluated.
   Agreement between validation and test is not evidence of real-world generalisation.
4. **One seed, one configuration.** Differences between checkpoints of about ±0.01 AP50 cannot be distinguished from noise. Config B
   (masked REVIEW regions) has not been trained.
5. **Statistics.** Size-bin confidence intervals assume independent boxes. Boxes within an image are correlated, so the
   uncertainty is underestimated.
6. **Dataset status.** `yolo-rock-v1.1` is PROVISIONAL and NOT FROZEN.

## Safety

- About **21% of large rocks (≥ 2,500 px²) are missed** at the frozen threshold.
- At least one large rock is missed in 40% of the test images that contain one.
- Failures concentrate in dark scenes, dense boulder fields and large shadowed boulders against the sky.
- The detector outputs 2-D boxes only. It does not estimate range, size or traversability.

**This detector must not be the sole basis for obstacle avoidance.** A rover pipeline needs redundant hazard sensing (stereo,
depth or LiDAR), temporal fusion across frames, conservative clearance margins, and validation on real or simulator imagery
before any deployment claim.

## Engineering

- **Licences.** The redistribution licences of the Artificial Lunar Landscape and NASA POLAR data are unconfirmed, so no dataset
  pixels are published. The project licence has not been chosen.
- **Model terms.** The model was trained with Ultralytics YOLO (AGPL-3.0).
- **Embedded paths.** The frozen checkpoint embeds the training arguments recorded by Ultralytics, including absolute local paths
  of the training machine. The file cannot be altered without breaking the frozen SHA-256.
- **Environment.** The locked environment targets CUDA 13.0 wheels. The CPU-only path is documented but unverified.

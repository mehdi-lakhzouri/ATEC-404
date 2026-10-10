# Roadmap

Nothing in this roadmap is implemented yet. The completed work ends at the frozen, test-evaluated detector.

| Phase | Goal | Exit criterion |
|---|---|---|
| R1 | **Human annotation review (E.1b).** Review the gallery and REVIEW-policy decisions; correct label artefacts | Human-verified evaluation subset, re-scored with the frozen model; no retuning on test |
| R2 | **Webots camera integration.** Feed simulator frames to `RockDetector` (see `future/WEBOTS_INTEGRATION_PLAN.md`) | Domain-gap report on Webots frames with simulator ground truth |
| R3 | **Spatial localisation.** Project boxes to rover-frame positions using camera intrinsics, extrinsics and stereo or depth | Position error versus simulator ground truth |
| R4 | **Hazard map and navigation.** Fuse detections over time into an occupancy or hazard map; plan conservatively | Collision-free traverses in Webots scenarios |
| R5 | **Mission Control.** Dashboard of telemetry, detections and hazard map (the `frontend/` scaffold) | Live end-to-end demo |
| R6 | **Model improvements** (each needs explicit approval) | Multi-seed training, Config B ablation, larger model, real-domain fine-tuning | Improvement on the human-verified subset |

**Rules carried forward:**
- The frozen model and its configuration are not modified.
- The held-out test split is never re-evaluated.
- Any new model goes through its own validation-only selection and a new, separately locked test.

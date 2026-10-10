# Webots Integration Plan (not started)

**Status: PLAN ONLY.** Webots R2025a is installed on the development machine. No controller, world or integration code exists yet.

## Goal

Run the frozen rock detector on a simulated rover's camera in Webots, then measure how well it transfers to simulator imagery
**before** any navigation logic depends on it.

## Interface

```python
from artemis_x.ml.inference import RockDetector

det = RockDetector()                    # frozen weights, SHA-256 verified, CUDA or CPU

# inside the Webots controller loop
img = camera.getImageArray()            # convert BGRA -> RGB uint8 HxWx3
result = det.predict(rgb_frame)         # imgsz 736, conf 0.20, NMS 0.60, max_det 300, area >= 50 px^2
publish(result.detections)              # boxes + confidence + latency
```

The detector API is synchronous. At a median of about 9–15 ms per frame on the RTX 5060 it can run every frame. On CPU, roughly
25 ms or more per frame, the controller should process every N-th frame.

## Steps

1. **Camera parity.** Configure the Webots camera to 720×480 with a field of view close to the training renders. Record intrinsics.
2. **Domain-gap check (no training):**
   - generate scenes with known rock placements;
   - export frames together with simulator ground-truth boxes;
   - score them with the same ignore-aware evaluator, on a Webots split separate from all existing splits.
3. **Latency budget.** Measure end-to-end controller loop time on the target hardware.
4. **Localisation.** Back-project box bottoms to the ground plane, or use a depth or stereo camera, to get rover-frame positions.
5. **Hazard map.** Temporal fusion and conservative inflation of obstacles. The detector alone is not sufficient: it misses about
   21% of large rocks on synthetic test data.
6. **Decision gate.** If the domain gap is large, propose (for approval) a fine-tuning experiment on Webots frames with its own
   validation-only selection and its own locked test.

## Constraints

- Do not modify the frozen model or configuration.
- Never use the existing held-out test split for Webots tuning.
- Every new training run needs explicit authorisation and goes through the guarded wrapper.

# Webots R1 + R2 Report: Artemis-X Lunar Rover Simulator v0.1

| | |
|---|---|
| Date | 2026-10-10 |
| Environment | Windows 11, Webots R2025a (`C:\Program Files\Webots`), system Python 3.12.0 (Webots `pythonCommand = python`) |
| Scope | Simulation infrastructure only. **No YOLO, no training, no navigation.** |

## 1. Inspection results (R1)

| Item | Finding (inspected, not assumed) |
|---|---|
| Installation | `C:\Program Files\Webots`: `msys64/mingw64/bin/webots.exe`, Python API in `lib/controller/python/controller` |
| Official world | `projects/samples/demos/worlds/moon.wbt`. Lunar gravity 1.62, textured ground mesh (2 km, flat within about 8 m of the origin, **no collision geometry**), Earth backdrop, two Koala robots running the C controller `moon` (Braitenberg obstacle avoidance plus keyboard control) |
| Assets | 3 `EXTERNPROTO` URLs (TexturedBackground, TexturedBackgroundLight, Koala) and texture URLs on `raw.githubusercontent.com/cyberbotics/webots/R2025a/...`; resolved from the local Webots asset cache. All preserved verbatim |
| Koala (`Koala.proto`, cached copy) | Motors `"left wheel motor"` and `"right wheel motor"` on the middle wheels (axis +y); encoders `"left wheel sensor"` and `"right wheel sensor"`; wheel radius 0.0425 m; wheel centres y = ±0.1425 m; 16 IR `DistanceSensor` `ds0`–`ds15` (lookup 0 m → 1024, 0.5 m → 0); `extensionSlot`; **no camera, no Physics node** (kinematic motion) |
| Python API | `from controller import Supervisor` (Webots R2025a); `Motor.setPosition/setVelocity/getMaxVelocity` (max 10 rad/s), `Camera.getImage` (BGRA), `RangeFinder.getRangeImage` |

The Webots installation was **not modified**. The official `moon.wbt` stays available unchanged.

## 2. What was built (R2)

| File | Purpose |
|---|---|
| `artemis-webots/worlds/artemis_moon.wbt` | Generated copy of `moon.wbt`. Same terrain, textures, background and asset URLs. **One** Koala `artemis_rover` at a deterministic pose (0, 0, 0), yaw 0. Controller `artemis_rover`, `supervisor TRUE` for ground-truth pose. Camera and RangeFinder in `extensionSlot`. 3 static test rocks. Follow-camera viewpoint, `basicTimeStep` 32 ms |
| `artemis-webots/tools/make_world.py` | Regenerates the world from the installation (read-only) |
| `artemis-webots/controllers/artemis_rover/artemis_rover.py` | Thin Webots entry point |
| `src/artemis_x/sim/drive.py` | Wheel commands (forward, backward, left, right, stop), proportional velocity limiting, twist ↔ wheel conversion, scripted sequence |
| `src/artemis_x/sim/sensors.py` | Device initialisation with error handling, configurable period, BGRA→RGB, depth statistics, intrinsics, timestamped export |
| `src/artemis_x/sim/rover.py` | Motor setup (velocity mode), demo loop, ground-truth pose, telemetry CSV, self-checks, graceful stop (`finally` → motors at 0), optional quit |
| `scripts/run_webots_smoke.py` | Automated headless run with independent checks |
| `tests/test_sim.py` | 7 pure-logic tests + 1 opt-in end-to-end Webots test |

## 3. Results

**Automated runs:** `scripts/run_webots_smoke.py`, both `--no-rendering` and `--rendering`, with the system Python that a GUI launch
uses. Each takes about 2.6 s wall-clock for 11.6 s of simulation.

| Check | Result |
|---|---|
| World loads, controller starts (`Starting controller: python.exe -u artemis_rover.py`), Webots exit 0 | **PASS** |
| Motors receive commands; rover position changes (supervisor ground truth) | **PASS**: forward_1 0.475 m, forward_2 0.471 m (ideal 0.510 m) |
| Left turn | **PASS**: +83.2° (ideal kinematic target 90°; six-wheel kinematics, open-loop timing) |
| Stops | **PASS**: after a stop command the rover coasts 0.031 m during the motor ramp-down (≈ 0.2 s), then moves 0.0000 m |
| Camera output dimensions | **PASS**: 3 frames, 720×480 |
| RGB pixel-format conversion | **PASS**: raw RGB is byte-identical to Webots' own PNG export (3/3 frames) |
| Depth finite where valid | **PASS**: 64–68 % valid pixels; sky = `inf` (invalid); nearest ground 0.499 m, as the mounting geometry predicts |
| RGB/depth spatial consistency | **PASS**: 4 rock observations projected from ground-truth pose; measured vs analytic depth error **1.6–3.8 mm** |
| Depth semantics | Measured: **optical-axis depth** (straight-line-range hypothesis off by 3.6–5.2 cm) |
| Determinism | Telemetry and depth frames **byte-identical** across two runs |
| Device-missing handling | Implemented (`SensorSuite` logs and continues) and unit-tested for format errors; not exercised in a live run |
| Regression | Root tests **37/37 OK** (29 existing + 8 new, including Webots end-to-end); backend **52/52 OK**; `verify_artifacts.py` **8/8 PASS** (frozen model, test lock and evidence unchanged) |

## 4. Interactive GUI verification

| Item | Result |
|---|---|
| Visual check | **Done by the project owner on 2026-10-10**: the world opens in the Webots GUI and the scripted rover motion was observed |
| GUI runs | The three runs (01:43–01:45) each wrote diagnostics with **6/6 self-checks PASS** |
| Determinism | Their `telemetry.csv` files are byte-identical to the headless runs |
| GUI layout | Webots saved its layout to `worlds/.artemis_moon.wbproj` (camera and RangeFinder overlays opened). It is per-user GUI state and is git-ignored |

During the automated runs, another Webots instance was already open (port 1234 busy). The tests used port 1240 and were unaffected.

## 5. Known limitations

1. **Kinematic rover:** there are no rover dynamics, slip or terrain contact. The ground has no collision geometry, so the rover cannot
   physically hit rocks; the demo path avoids them.
2. **Open-loop turning:** it reaches 83°, not 90°. Closed-loop heading control (encoders or IMU) is future work.
3. **Simple test rocks:** they are spheres, not photorealistic. The scene lighting is dim (mean pixel value ≈ 42/255), unlike the training
   renders. **No claim is made that YOLO detects Webots rocks**; that is untested.
4. Only the 3 frames captured during stop phases are exported, by design. Continuous capture is available through
   `SensorSuite.capture()`.
5. The Koala IR distance sensors are not used yet.

## 6. Dependencies

- Webots R2025a (asset URLs pinned to the R2025a tag, resolved from the local cache).
- The controller needs only the Python ≥ 3.10 standard library.
- The smoke runner uses the project environment (numpy, Pillow).

## 7. Next step: R3, YOLO integration (not started)

1. Run the controller with the project interpreter: point Webots' Python command at `backend/.venv/Scripts/python.exe`, or use an
   extern controller. That interpreter is needed for `torch` and `ultralytics`.
2. Feed `SensorSuite.capture()` RGB frames to `artemis_x.ml.inference.RockDetector` (frozen settings), every N-th frame.
3. **Domain-gap check first:** label Webots frames with simulator ground truth (rock poses are known) and score them with the
   ignore-aware evaluator, on a Webots-only split. The held-out test split is never used.
4. Project detections to rover-frame positions with the measured optical-axis depth and the intrinsics above.
   Distances are never inferred from box size.

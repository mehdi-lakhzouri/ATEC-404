# artemis-webots: Artemis-X Lunar Rover Simulator v0.1 (R1 + R2)

A Webots R2025a project: one Koala rover on the official lunar terrain, driven by a Python controller, with a forward RGB
camera and a pixel-aligned depth sensor. R3 adds frozen-YOLO11n detection through a separate local inference server; see below and
`docs/WEBOTS_R3_YOLO_REPORT.md`.

```text
artemis-webots/
├── worlds/artemis_moon.wbt              # generated from the official moon.wbt (do not hand-edit; regenerate)
├── controllers/artemis_rover/
│   └── artemis_rover.py                 # thin Webots entry point -> src/artemis_x/sim/rover.py
├── tools/make_world.py                  # regenerates the world from the Webots installation (read-only on it)
└── output/                              # run diagnostics (git-ignored)
```

Code lives in the package `src/artemis_x/sim/`:

| Module | Content |
|---|---|
| `drive.py` | Koala geometry, wheel commands, limits, scripted demo |
| `sensors.py` | Camera and RangeFinder capture, conversion, export |
| `rover.py` | Webots runtime, ground-truth pose, self-checks |

## Run it in the Webots GUI

1. Start **Webots R2025a**.
2. **File → Open World…** → `artemis-webots/worlds/artemis_moon.wbt`.
3. The controller `artemis_rover` starts automatically. Webots runs it with the `python` configured in
   **Tools → Preferences → Python command**; the default `python` works. Only the standard library is needed, no numpy.
4. Press **Run** (▶). The rover performs:

   | Step | Duration |
   |---|---|
   | stationary | 1 s |
   | forward | 3 s |
   | stop | 1 s |
   | left turn | ≈ 2.6 s |
   | forward | 3 s |
   | stop | — |

   It then stays stopped. The viewpoint follows the rover.
5. Diagnostics are written to `artemis-webots/output/<timestamp>/`: `summary.json`, `telemetry.csv` and `frames/`. The console prints
   the PASS/FAIL self-checks.

To open the camera and depth views in the GUI, double-click the rover, then **Overlays → Camera Devices / RangeFinder Devices**.

## Automated (headless) run

```powershell
uv run --project backend --no-sync python scripts/run_webots_smoke.py              # --no-rendering, minimised, ~3 s
uv run --project backend --no-sync python scripts/run_webots_smoke.py --rendering
$env:ARTEMIS_RUN_WEBOTS="1"; uv run --project backend --no-sync python -m unittest tests.test_sim
```

The runner launches `webots.exe --batch --mode=fast --minimize` with the system Python, so the controller quits the
simulation when the demo ends. It then independently checks the RGB conversion and the depth-to-geometry consistency
(`output/smoke-*/smoke_result.json`).

## Rover, motors and sensors (from the real `Koala.proto`, Webots R2025a)

| Item | Value |
|---|---|
| Drive | `RotationalMotor` "left wheel motor" / "right wheel motor" (middle wheels only; the other 4 wheels are passive). Velocity control: `setPosition(inf)` + `setVelocity(rad/s)`. Max 10 rad/s (`getMaxVelocity`) |
| Encoders | `PositionSensor` "left wheel sensor" / "right wheel sensor" |
| Geometry | Wheel radius 0.0425 m, track 0.285 m, forward = +x |
| Built-in sensors | 16 infra-red `DistanceSensor` `ds0`–`ds15` (≈ 0–0.5 m). **No camera** in the PROTO |
| Added (`extensionSlot`) | `Camera` "front_camera" and `RangeFinder` "front_depth". Same pose (x 0.16, z 0.24 m, pitched 0.12 rad down), same FOV (1.0 rad), same 720×480 resolution, so pixel (u, v) in both images is the same ray |
| Physics | The Koala PROTO has no `Physics` node, so Webots moves it **kinematically** from wheel rotation. The ground mesh has no collision geometry. Test rocks are static solids, and the rover would pass through them (the demo path avoids them) |
| World | `basicTimeStep` 32 ms, lunar gravity 1.62 m/s². Sensors sample every 64 ms (`ARTEMIS_SENSOR_MS`) |

**Data formats:**
- **RGB:** `Camera.getImage()` returns BGRA, which is converted to RGB. Previews are exported with Webots' own PNG encoder, raw frames as
  `*_rgb.raw` (uint8, H×W×3).
- **Depth:** `RangeFinder.getRangeImage()` gives metres along the **optical axis** (verified within 4 mm against analytic geometry).
  Sky and no-return pixels are `inf` and are treated as invalid. Raw frames are `*_depth.f32` (float32, H×W).
- **Intrinsics:** `fx = fy = 658.98 px`, `cx = 359.5`, `cy = 239.5` (`summary.json`).
- **Frames:** sensor frame x forward, y left, z up; image u to the right, v downwards. Each frame stores its simulation time and
  the ground-truth rover pose.

## Options (environment variables)

| Variable | Meaning |
|---|---|
| `ARTEMIS_WEBOTS_OUT` | Output folder |
| `ARTEMIS_WEBOTS_QUIT=1` | Quit at the end of the demo (automated runs) |
| `ARTEMIS_SENSOR_MS` | Camera/depth sampling period |

## R3: YOLO camera integration

The controller (Webots' system Python, stdlib only) sends camera frames over a versioned local TCP protocol
(`127.0.0.1:5761`) to a separate inference server. The server runs in the ML environment with the **frozen** YOLO11n
(SHA-256 `413d0481…`, imgsz 736, conf 0.20, NMS 0.60, max_det 300, boxes < 50 px² removed). Communication is asynchronous and
bounded:
- one pending frame (older frames are dropped);
- one request in flight;
- frame-ID matching, staleness flags and timeouts.

If the server is unavailable, the R1/R2 scripted motion continues unchanged. That is **not** autonomous navigation.

**Steps:**

```powershell
# 1. Start the inference server (ML environment) and keep the window open
uv run --project backend --no-sync python scripts/inference_server.py            # --device cpu to force CPU

# 2. Open a world in Webots: File -> Open World...
#    artemis-webots/worlds/artemis_moon.wbt            R1/R2 scene (baseline)
#    artemis-webots/worlds/artemis_moon_enhanced.wbt   irregular rocks + low-angle sun

# 3. Press Run. The rover performs the R1/R2 sequence while detections stream back. Inspect:
#    artemis-webots/output/demo-<timestamp>/inference/detections.jsonl
#    (boxes, confidence, depth validity, surface point in camera and rover frames, latency)

# Automated runs (server + headless Webots, managed for you)
uv run --project backend --no-sync python scripts/run_webots_r3.py demo
uv run --project backend --no-sync python scripts/run_webots_r3.py robustness     # no server / slow server
uv run --project backend --no-sync python scripts/run_webots_r3.py eval           # 40 paired captures + analysis + overlays
uv run --project backend --no-sync python scripts/run_webots_r3.py course         # test course v2: 11 viewpoints x 2 suns, per-zone results
uv run --project backend --no-sync python scripts/run_webots_r3.py tour           # scripted 9-stop course tour + stop accuracy vs ground truth
```

| Module (`src/artemis_x/sim/`) | Role | Runs in |
|---|---|---|
| `protocol.py` | Wire format (versioned, size limits) | both |
| `client.py` | Async, bounded inference client | Webots (stdlib) |
| `perception.py` | Frame / depth / pose association, demo link, evaluation captures | Webots (stdlib) |
| `projection.py` | Box + optical-axis depth → visible-surface point (camera and rover frames) | Webots (stdlib) |
| `inference_server.py` | Frozen-model server | ML environment |

**Localisation semantics:** the reported point is the median of the back-projected *foreground* depth pixels in the central
half of the box. It is a point on the **visible rock surface**, not the rock centre. Boxes without enough valid depth are marked
invalid; a distance is never invented.

## Regenerating the world

```powershell
python artemis-webots/tools/make_world.py              # R1/R2 world (byte-stable) + enhanced demo world
python artemis-webots/tools/make_world.py --eval 0 1 2 3 4   # paired evaluation worlds + manifests (git-ignored)
# reads <WEBOTS_HOME>/projects/samples/demos/worlds/moon.wbt; never writes there
```

## Test course v2 (`worlds/artemis_test_course.wbt`, `worlds/artemis_test_course_highsun.wbt`)

A ready-made, deterministic environment for perception tests. It contains:

- 39 Poly Haven CC0 rocks and boulders on a cratered height-field regolith (≈ 43 × 9 m test area);
- 5 non-rock objects;
- two lighting variants: low sun (≈ 20°, long shadows) and high sun (≈ 55°).

Fetch the assets first (see `assets/README.md`), then run `python artemis-webots/tools/make_world.py`.

| Zone | x (m) | Content | What it tests |
|---|---|---|---|
| Z1 isolated | 3.1–3.4 | 3 rocks, 0.55–0.7 m | Clean detections, distinct shapes |
| Z2 small scatter | 6.0–7.3 | 8 pebbles, 8–18 cm (all seven scans) | Small and distant objects |
| Z3 large boulder | 10.0–10.2 | 1.4 m boulder scan + 0.45 m rock | Near-field, frame-filling obstacle |
| Z4 occlusion / cluster | 13.0–13.8 | Rock behind rock + 3 touching rocks | Occlusion, merged boxes, occluded depth |
| Z5 non-rock distractors | 16.6–17.6 | Lander crate, fuel tank, antenna mast, solar panel, white ball | False positives on engineered objects |
| Z6 shadowed | 20.0–20.9 | 2 small rocks in a 1.2 m boulder's shadow | Low-light detection |
| Z7 craters | 23.9–26.3 | 2.4 m crater with 3 rim rocks + 1 in the bowl; an empty 1.1 m crater | Terrain relief; FP probe on an empty crater |
| Z8 boulder field | 29.0–33.0 | 1.1 m and 1.8 m boulders + 8 mixed rocks | Clutter, large and small together |
| Z9 half-buried | 36.0–37.4 | 4 rocks sunk 45–60 % into the regolith | Partial outlines, low relief |

**Ground truth:** `worlds/artemis_test_course*.json` (generated, git-ignored). Each object has:
- a class (`rock` / `non_rock`);
- a placement (asset, position, yaw, scale, terrain height, sink fraction), from which the analyser rebuilds the exact mesh;
- a unique segmentation colour.

The manifests also list the craters, the 11 camera viewpoints and the tour plan.

**How to use it:**
- **Interactive:** open either world in Webots with the inference server running and press Run.
  - The rover starts on a clear lane (y = −2.6 m) and drives the embedded **9-stop tour**: drive to the stop, turn to face the
    zone, observe for 2 s, turn back.
  - Control is closed-loop on wheel-encoder odometry; detections are logged, never used for control.
  - See `docs/ROVER_BEHAVIOUR.md`.
- **Automated:**

  ```powershell
  uv run --project backend --no-sync python scripts/run_webots_r3.py course   # 11 viewpoints x 2 suns -> output/course-<ts>/
  uv run --project backend --no-sync python scripts/run_webots_r3.py tour     # headless tour -> output/tour-<ts>/
  ```

  - `course` scores detections against the segmentation references, per zone and per sun, and writes overlays.
  - `tour` reports per-stop position and heading error against ground truth, plus the detection-stream statistics.

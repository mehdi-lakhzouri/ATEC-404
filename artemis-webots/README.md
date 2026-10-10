# artemis-webots: Artemis-X Lunar Rover Simulator v0.1 (R1 + R2)

A Webots R2025a project: one Koala rover on the official lunar terrain, driven by a Python controller, with a forward RGB
camera and a pixel-aligned depth sensor. **No YOLO integration yet**: that is release R3.

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

## Regenerating the world

```powershell
python artemis-webots/tools/make_world.py     # reads <WEBOTS_HOME>/projects/samples/demos/worlds/moon.wbt; never writes there
```

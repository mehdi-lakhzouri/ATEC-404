# Artemis-X rover: how the simulated robot behaves today

This page describes what the Webots rover actually does in the current code (R1, R2, R3 and the test course v2). It is a
description of behaviour, not a plan. Every number comes from a headless run on 2026-10-10. **Short version:** the rover follows
fixed scripts. It sees rocks, but it does not react to them.

## 1. The robot

| Item | Value | Source |
|---|---|---|
| Platform | K-Team Koala (six wheels, two driven middle wheels) from Webots R2025a | `Koala.proto` |
| Motors | `left wheel motor`, `right wheel motor`, velocity mode, limited to 10 rad/s | `src/artemis_x/sim/rover.py` |
| Geometry | wheel radius 0.0425 m, track 0.285 m | `src/artemis_x/sim/drive.py` |
| Encoders | `left/right wheel sensor` (wheel angle, rad) | used for odometry in the tour |
| Camera | `front_camera`: 720 × 480 RGB, 1.0 rad horizontal field of view, 0.16 m ahead and 0.24 m above the base, pitched 0.12 rad down | extension slot |
| Depth | `front_depth`: RangeFinder with the same pose, field of view and resolution as the camera (pixel-aligned); 0.05–15 m range, optical-axis depth | extension slot |
| IR sensors | `ds0`–`ds15` from the Koala model | **not used** |
| Physics | **none**: the rover is kinematic and the ground has no collision | see §5 |

## 2. The controller chooses one of three behaviours

`artemis-webots/controllers/artemis_rover` calls `artemis_x.sim.rover.main()`. That function picks a mode in this order:

| Condition | Behaviour | Used by |
|---|---|---|
| `ARTEMIS_MODE=eval` | **Evaluation**: the rover is teleported to fixed viewpoints, one frame is taken at each, then it quits | the 40-capture R3 evaluation and the test-course evaluation |
| The world's `customData` holds `{"script": "course_tour", ...}` | **Course tour**: closed-loop scripted drive along a clear lane (§4) | `artemis_test_course*.wbt` |
| Otherwise | **R1/R2 demo**: open-loop timed motion (§3) | `artemis_moon.wbt` and the enhanced/eval worlds outside eval mode |

In every mode the YOLO detections are **logged only**. Ground truth comes from the supervisor API and is **logged for evaluation
only**; neither feeds back into the motion.

## 3. R1/R2 demo: open loop, timed

Six steps run on a clock (`drive.demo_sequence`), with no feedback:

| Step | Command | Duration | What happens (measured) |
|---|---|---|---|
| start_stationary | stop | 1 s | camera and depth frame captured |
| forward_1 | both wheels 4 rad/s | 3 s | 0.475 m travelled (ideal 0.510 m; the motor ramps up) |
| stop_1 | stop | 1 s | coasts 0.031 m, then stands still; frame captured |
| turn_left | wheels −2 / +2 rad/s | π/2 ÷ ideal yaw rate | **83.2°** instead of 90°: the six-wheel Koala turns less than ideal differential-drive kinematics predict |
| forward_2 | both wheels 4 rad/s | 3 s | 0.471 m |
| stop_2 | stop | 1 s | coasts 0.031 m; frame captured |

The rover ends about half a metre to the side of where it started, facing roughly +y. **There is no position control.** A different
time step or speed would give a different path.

## 4. Course tour: closed loop on wheel encoders (new)

In the test course the rover starts on a lane at y = −2.6 m. All 39 rocks and 5 non-rock objects sit at least 0.85 m off that lane
(rock edges) or 1.3 m off it (distractors). The world carries the plan in `customData`.

**How the plan is made** (`make_world.course_layout`, precomputed, not decided by the robot):
- For each zone, the generator tries lane stops every 0.1 m.
- From each stop it extends a straight approach toward the zone centre in 2 cm steps. The approach ends when the rover is 1.8 m
  from the centre, or when any object or crater rim would come within **0.6 m** of the rover centre (Koala half-length ≈ 0.25 m).
- It keeps the stop whose approach ends closest to the zone.

| Zone | Approach | Observes from (zone centre) | Before the change |
|---|---|---|---|
| Z1–Z4, Z6, Z9 | 0.8–2.4 m | 1.80–1.82 m | 3.0–3.8 m |
| Z5 distractors | 2.2 m | 1.99 m | 3.8 m |
| Z7 craters | 0.8 m | 2.49 m (the rocks surround the crater; the rover may not enter the bowl) | 4.0 m |
| Z8 boulder field | 0.7 m | 2.09 m (dense clutter) | 3.7 m |

**At each stop the rover:**

1. **drives to the stop's absolute lane position.** The distance comes from encoder odometry only. It slows down over the last
   15 cm, never below 15 % of 4 rad/s, and stops within 2 mm.
2. **turns in place to face the zone** (31°–88°). It slows down over the last 0.25 rad and stops within 0.004 rad.
3. **drives straight toward the zone** for the planned approach distance.
4. **pauses 2 s.** It captures one RGB and depth frame. Detections keep streaming from the inference server.
5. **reverses the same distance** back to the lane.
6. **turns back onto the lane heading.**

`drive.Odometry` integrates the two encoders. Distance uses the wheel radius as is. Yaw is scaled by **0.958**, calibrated
once from R2 ground truth (encoder 1.5148 rad vs true 1.4516 rad).

Measured, from the latest tour run (`run_webots_r3.py tour`):

| | Result |
|---|---|
| Moves completed | 55 / 55 (9 stops × 6 moves + final pause) |
| Observation-point accuracy vs ground truth | position error ≤ 7.3 mm, heading error ≤ 0.21° at every stop |
| Clearance along the true path | min **0.601 m** from any object edge or crater rim over 16,005 simulation steps (required 0.6 m) |
| Detection stream during the tour | 1,996 frames submitted; 1,994 answered OK, 0 failed, 0 stale; 5,994 detections, 5,872 localised with depth; end-to-end latency p50 19 ms / p95 34 ms |
| Controller time per step | p50 0.25 ms; max 199 ms on the steps that write a captured frame to disk |

**Why it is this accurate (and why that will not transfer):**
- The rover is kinematic, so wheels never slip.
- Odometry and ground truth agree to the millimetre because the simulator moves the robot exactly as the wheels turn.
- On real regolith, wheel slip, sinkage and rocks would make encoder odometry drift. This result validates the control logic, not
  real-world localisation.

## 5. What the rover does NOT do

- **No collision.** There is no Physics node and no ground collision. If a script drove it into a boulder, it would pass through.
  The tour avoids objects only because the lane and every approach line are checked clear by the generator.
- **No obstacle avoidance or reaction to detections.** Rocks found by YOLO (with 3-D positions from depth) are written to
  `detections.jsonl` and never used to steer, stop or re-plan.
- **IR sensors are unused.**
- **No terrain following.** The rover stays at z = 0. The course terrain varies by about ±1 cm along the lane, so the wheels sink
  or float by up to 1 cm. The rover never enters a crater bowl (−25 cm); crater rims count as obstacles.
- **No path planning, mapping or navigation.** These remain out of scope, as agreed for R3.

## 6. Perception attached to the motion (R3)

| Stage | How it works |
|---|---|
| Sampling | Every 64 ms of simulated time the controller reads one camera image and one depth image from the same sensor step. |
| Hand-off | The pair goes to a bounded asynchronous client: one pending slot, one frame in flight. The newest frame replaces an older unsent one. Results are matched by frame ID; old ones are flagged stale. |
| Inference | A separate process (`scripts/inference_server.py`, 127.0.0.1:5761) runs the **frozen YOLO11n** (SHA-256 `413d0481…`, conf 0.20, NMS 0.60, unchanged). |
| Localisation | Each box is turned into a visible-surface 3-D point using the aligned depth, in camera, rover and world frames. |
| Fault tolerance | If the server is missing or slow, frames are dropped and the motion is unaffected. The R3 robustness runs measured this. |

## 7. Evaluation mode (teleporting)

For the paired 40-capture R3 evaluation and the test-course evaluation, the rover is moved with the supervisor:

- It is teleported to each manifest viewpoint.
- It waits a few steps, captures, and submits the frame.
- Reference boxes come from Webots recognition segmentation.

This is **not driving**: it isolates detection quality from motion.

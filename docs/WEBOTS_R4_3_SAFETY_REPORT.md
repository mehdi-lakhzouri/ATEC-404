# R4.3 release report: safety layers, obstacle inflation and uncertainty management

| | |
|---|---|
| Date | 2026-10-10 |
| Status | Implemented and tested; **not committed** (awaiting review) |
| Evaluation run | `artemis-webots/output/r4-safety-20261010-090802` (8 scenarios, full course tour each) |
| Scope kept | No navigation, no motor command from health or cost; frozen YOLO, datasets, R1–R3 worlds unchanged; no supervisor ground truth in live code (test-enforced) |

**Reproduce:**

```powershell
uv run --project backend --no-sync python scripts/run_webots_r4.py safety                      # 8 scenarios + analysis (about 9 min)
uv run --project backend --no-sync python scripts/run_webots_r4.py analyse-safety artemis-webots/output/r4-safety-<ts>
```

## 1. R4.2 audit (prerequisite)

| Check | Result |
|---|---|
| R4.2 implemented | Yes (in the working tree; R4.1 and R4.2 are uncommitted by your instruction) |
| Unit tests (Webots' own Python) | 33/33 pass |
| Integration (YOLO-off tour) | Re-run reproduced R4.2 exactly: free precision 0.9968, occupied 0.9254, recall 32/33, alignment shift (0, 0), recentring 0 mismatches |
| Geometry, serialisation, timestamps, UNKNOWN | Covered by existing tests (cell indexing, round trip, sample-time alignment, blind zone / shadow / hidden pit stay non-free) |
| **Defect found** | The R4.2 **evaluation runner** had a syntax error: a figure-label edit made through a shell heredoc. The `analyse-map` call right after it failed silently (its output was discarded), so `r4_2_map_final.png` never received the new markers. Fixed, the repository was swept for syntax errors (none other), and the evaluation re-run. The live mapping was not affected |
| Blockers for a safety layer | None. Open R4.2 items (small crater, overhang cells) are UNKNOWN, not FREE, which the cost layer keeps non-traversable |

## 2. What was implemented

| Feature | Files | Summary |
|---|---|---|
| A. Footprint and inflation | `mapping/costmap.py` (new), `config.py` | Koala footprint measured from `Koala.proto` (see 2.1). Cost per cell for the rover centre: 0 free / 1–252 graded / 253 inscribed / 254 lethal / 255 unknown. UNKNOWN and LETHAL are never lowered. Each hazard is inflated by its own 2σ_rel. Kernels are precomputed per 5 cm of margin. Updates are incremental by 1 m tiles; a recentre shifts the arrays instead of rebuilding |
| B. Health monitor | `mapping/health.py` (new) | OK / DEGRADED / STOP from 13 reason codes, with timestamp and measurements. A reason is raised in one cycle and cleared after 2 clean cycles (hysteresis). Advisory only: the tour never reacts |
| C. Uncertainty | `mapping/uncertainty.py` (new), `pose.py`, `drive.py` | Local relative model replaces the start-relative R4.1 model for safety decisions (section 4) |
| D. Fault injection | `mapping/faults.py` (new), runner | Off unless `ARTEMIS_FAULTS` is set (JSON). It is logged in `mapping_config.json` and on every telemetry line. Depth dropout, freeze, invalid, cycle overrun, odometry scale / turn scale / heading drift. Inference faults (absent, killed, delayed) are produced by the runner |
| E. R5 interface | `mapping/safety.py` (new) | `health()`, `get_stop_reasons()`, `cost_at()`, `footprint_cost()`, `corridor_ahead()`, `corridor_clear()`. Uses the plan's `MapHealth` fields, and snapshots gain an additive `cost` layer + `health` header (schema stays `artemis.map.v1`) |
| Integration | `mapping/mapper.py`, `grid.py`, `depth_scan.py`, `snapshot.py`, `perception.py`, `rover.py` | Freeze detection (byte-identical depth while moving → frame rejected, never mapped). Lower-image valid fraction. Inference status from the R3 link. Cost and health each cycle |
| Evaluation | `scripts/run_webots_r4.py` (`safety`, `safety-run`, `analyse-safety`), `scripts/r4_safety_analysis.py` (new) | 8 scenarios. Ground-truth comparison offline only |
| Tests | `tests/test_mapping_safety.py` (new, 18) | Footprint, kernel radii and grading, cost semantics, negative hazards, uncertainty growth and cap, incremental = full rebuild, recentre shift = full rebuild, reset, history lookup, default odometry unchanged, hysteresis, corridor (unknown / clear / obstacle / stale by age / stale by σ), fault injector (off by default, deterministic, windowed), frozen and invalid frames never change the map |

### 2.1 Footprint (Koala.proto R2025a, fetched and read)

| Part | Extent |
|---|---|
| Collision body | x −0.125…0.175 m, y ±0.14 m (two boxes + two front cylinders) |
| Six wheels (r 0.0425, 0.04 wide) | x ±0.1325, y ±0.1625 |
| Visual front | 0.19 m |
| **Circumscribed radius** | **0.210 m** (wheel corners) |
| Collision radius used | **0.25 m** (+0.04 m allowance, plan decision D3) |
| Safety band | 0.15 m |
| Uncertainty margin | 2σ_rel, capped at 0.20 m |
| Camera blind zone | Ground from 0.19 m (rover front) to about 0.64 m ahead is never in view: it is known only if seen earlier, otherwise UNKNOWN |

## 3. Results (8 scenarios, full 55-move tour each)

### 3.1 Fault detection (simulation time)

| Scenario / fault | Reason raised | Detection latency | Recovery after fault end | Target |
|---|---|---|---|---|
| 30 % depth dropout (45–55 s) | `depth_degraded` | **0.12 s** | 0.36 s | ≤ 0.5 s ✓ |
| Depth frozen 2 s (104–106 s) | `depth_frozen` (+ `depth_stale`) | **0.00 s** (first frozen cycle) | 0.30 s | ✓ |
| Invalid (inf) lower half of the image (160–163 s) | `depth_invalid` | **0.06 s** | 0.39 s | ✓ |
| Cycle overrun +60 ms (210–214 s) | `cycle_overrun` | **0.24 s** | 0.59 s | ✓ |
| YOLO server absent from start | `inference_unavailable` | from the first cycle | — | ✓ |
| YOLO server killed mid-tour (at sim 220.0 s) | `inference_unavailable` | **0.48 s** | — (never restarted) | ✓ |
| YOLO server +1.5 s per answer | `inference_unavailable` after the 1.0 s answer timeout; `inference_stale` when the first late answer arrives (18.4 s) | bounded by the timeout | — | **Not met as written**: a slow and a dead server look identical until a late answer arrives. The run is about 8× faster than real time, so 1.5 s wall ≈ 12 s simulated |
| Encoder scale ±3 %, heading bias | none expected (no external reference) | — | — | See 3.4 |

**Map integrity under faults.** The depth-fault run's final map is identical to the baseline: 82 false-free cells, 39 unsafe cells, same alignment. 0 frozen frames were integrated (8 rejected). Invalid or NaN depth never creates FREE space (unit-tested; confirmed in Webots).

**Depth hazards with YOLO offline.** The YOLO-absent, YOLO-killed and depth-fault runs produce the same occupancy and cost maps as the baseline. Mapping and the safety layer depend only on depth.

### 3.2 Corridor decisions against ground truth (offline)

Per cycle: does the true 1 m corridor ahead of the true pose cross the true collision zone (centre within 0.25 m of a true hazard)?

| Scenario | Truly blocked cycles | `obstacle_in_corridor` recall | Precision | **Dangerous: truly blocked, reported clear** |
|---|---|---|---|---|
| Baseline | 99 / 2,001 | 1.00 | 0.48 | **0** |
| +3 % scale | 140 / 1,951 | 1.00 | 0.86 | **0** |
| −3 % scale | 158 / 2,053 | 1.00 | 0.69 | **0** |
| Heading bias | 112 / 1,996 | 1.00 | 0.53 | **0** |

The low precision is deliberate conservatism: the 2σ margin, the half-cell rounding (up to 7 cm), and the 0.15 m band shown as graded cost.

### 3.3 Health over the baseline tour (advisory; the rover did not react)

| State | Time | Main reasons |
|---|---|---|
| OK | 276 s (54 %) | — |
| STOP | 236 s (46 %) | `stale_corridor` 40 % of cycles, `obstacle_in_corridor` 10 %, `unobserved_corridor` 5.5 % |
| DEGRADED | 0 s | — |

- **`stale_corridor` (782 of 796 cycles are the age rule).** FREE evidence on the corridor is older than 10 s. This happens in two situations:
  - after each turn back onto the lane, the cells under the rover were last seen about 30 s earlier, before the out-and-back excursion;
  - while reversing out of an approach, the camera faces the zone, not the ground being reversed over.

  This is the conservative rule working as specified, not a fault. R5 must re-observe the near field (or carry a rear sensor) before moving.
- **`unobserved_corridor`.** At the tour start nothing has been observed, so the blind zone is UNKNOWN: STOP.
- **`obstacle_in_corridor`.** At every observation stop the zone lies 1.8 m ahead, inside the 1 m corridor plus inflation.

### 3.4 Uncertainty model (Feature C)

**Finding.** The R4.1 model charged 2 % of the distance since the start. Its σ was 1.33 m after the tour, while the measured odometry error was 0.5 mm, and the planned STOP at 0.3 m would have fired after about 15 m. The coefficients were not the problem; the reference point was. A local map needs the error since each cell was last seen.

**Policy implemented:**

| Item | Definition |
|---|---|
| σ_rel | √((k_s·ΔD)² + (ρ·(k_turn·ΔA + k_dist·ΔD))²) since the cell's last observation, where ρ is the rover-to-cell distance (the heading lever arm) |
| Coefficients | k_s = 0.02, k_turn = 0.02, k_dist = 0.002 rad/m: a **design envelope** whose 2σ covers ±3 % errors, not a measurement |
| Inflation | Each hazard is inflated by 2σ_rel (capped at 0.20 m) |
| Corridor staleness | Corridor cells with σ_rel > 0.10 m count as stale. That is half the cap: beyond it, inflation can no longer absorb the error |
| Absolute σ | Still reported (32 m after the tour, turns charged worst-case) but **not a health reason**. It is unbounded with odometry only, and matters only for R5's global goals, which need absolute localisation first |

**Measured vs model** (relative error between capture-time pose pairs, ground truth offline):

| Scenario | Measured p95 error, 0–1 m / 2–4 m / 7–10.5 m driven | Model σ p50 (same bins) | Pairs within 2σ |
|---|---|---|---|
| Baseline (simulator) | 0.1 mm / 0.8 mm / 2.5 mm | 1.0 / 7.2 / 29 cm | **100 %** in every bin |
| +3 % distance scale | 2.7 / 10.5 / 22.8 cm | 1.0 / 7.1 / 28 cm | **100 %** |
| −3 % distance scale | 2.7 / 11.1 / 24.2 cm | 1.0 / 7.2 / 30 cm | **100 %** |
| Heading (turn scale +3 %, 0.0005 rad/m drift) | 1.4 / 7.3 / 22.6 cm | 1.0 / 7.2 / 29 cm | **100 %** |

The simulator's own odometry is essentially exact (no slip), so the envelope is conservative by two orders of magnitude here. It was chosen for robustness to the injected faults, not tuned to make a test pass.

**Effect on the cost map and safety.** Measured in the window within 3 m of the rover, with the map placed at the rover's true pose (offline): true collision-zone cells the costmap would allow.

| Scenario | Without the σ margin | With the σ margin (as implemented) |
|---|---|---|
| Baseline | 2.10 % | **0.84 %** |
| +3 % scale | 6.79 % | **2.14 %** |
| −3 % scale | 3.99 % | **1.75 %** |
| Heading bias | 3.01 % | **1.12 %** |

**Real physical consequence of odometry faults.** The scripted tour drives on odometry. With ±3 % scale the rover's true path came within **0.05–0.13 m** of obstacle footprints (0.60 m nominal), and the whole-course map is shifted 2–3 cells against ground truth. The health monitor still flagged every truly blocked corridor (table 3.2). Health transitions are otherwise unchanged by the faults, because nothing in the local data reveals a pure scale error. Detecting it needs an external reference (visual odometry or landmarks): R5+.

### 3.5 Cost layer against ground truth (baseline)

| Metric | Value |
|---|---|
| True collision-zone cells observed | 2,340 |
| …of which the cost layer allows (cost < 253) | **39 (1.7 %)**, only 1 inside a hazard cell |
| Attribution | rock 19 (Z6, in shadow) 11, rock 07 (Z2 pebble 15.6 cm) 10, rock 34 (Z8) 9, small empty crater rim 7, others 2 |

Cause: parts of these objects that depth never mapped as hazards. Inflation cannot invent hazards that were not seen; R4.4 YOLO fusion targets exactly these.

### 3.6 Performance (wall clock per mapping cycle, Webots fast mode)

| Scenario | Scan p95 | Grid p95 | Cost p50 / p95 | Health p95 | **Cycle p50 / p95** | Busiest step p95 | Memory (process) |
|---|---|---|---|---|---|---|---|
| Baseline (YOLO on) | 8.2 ms | 4.8 ms | 1.4 / 6.1 ms | 0.25 ms | **12.1 / 19.0 ms** | 11.6 ms | 74 MB |
| YOLO off runs | 6.8–7.6 ms | 4.0–4.7 ms | 1.3–1.7 / 5.1–6.1 ms | 0.20 ms | **9.8–10.4 / 15.0–15.8 ms** | 9.1–9.8 ms | 52–54 MB |

- **Target p95 ≤ 15 ms: missed** by 0–1 ms without YOLO and by about 4 ms with YOLO on.
- **Root cause:** pure-Python depth scan (p95 about 8 ms), plus the cost update when many tiles change (p95 about 6 ms). With YOLO on, the RGB conversion adds about 1.5 ms (read p50 2.0 vs 0.6 ms).
- **The 32 ms simulation step was never at risk:** busiest step p95 ≤ 11.6 ms.
- **Recentring:** 40 shifts and 1 full rebuild, down from 41 rebuilds (max 20 ms) before the shift optimisation.
- **Options:** fewer depth samples, or decision D1-b (NumPy in the controller interpreter).

### 3.7 Regression

| Check | Result |
|---|---|
| Root suite | 124 tests: **114 pass**; the 10 errors are `tests/test_mapping_fusion.py`, written ahead for R4.4, whose configuration did not exist yet |
| Backend | 52 OK |
| `verify_artifacts` | 8/8 |
| R1/R2 smoke | all pass |
| Committed worlds | Byte-identical |
| Mapping tests under Webots' Python | 51 OK |
| R3 and R4.1/R4.2 behaviour | Tour 55/55 with the same stops and 0.601 m clearance; maps identical to R4.2 |

## 4. Figures (`docs/figures/webots/`)

| File | Shows |
|---|---|
| `r4_3_occupancy_vs_cost.png` | Occupancy before inflation vs cost layer after, whole course |
| `r4_3_cost_zooms.png` | Safety margins around the Z3 boulder, Z8 boulder field and Z6 boulder; crater bowl and rim as lethal hazards |
| `r4_3_unknown_blind_zone.png` | Live window at 3 moments: footprint, 1 m corridor, camera FOV, blind zone, UNKNOWN cells, health state |
| `r4_3_cost_under_odometry_faults.png` | Accumulated cost map for baseline, ±3 % scale and heading bias |
| `r4_3_uncertainty_measured_vs_model.png` | Measured relative error vs the 2σ envelope |
| `r4_3_health_timelines.png` | States and reasons over time with the fault windows and the YOLO kill marked |

## 5. Known limitations

1. **No collision physics.** The rover is kinematic and the rocks have no collision geometry. Safety is judged geometrically against ground truth offline. Nothing here proves physical avoidance (plan section 10).
2. **Advisory only.** STOP does not stop the rover; R5 must wire it.
3. **Pure scale or heading errors are invisible to the health monitor.** They are only bounded by the σ envelope, which is a design assumption to be re-measured on real terrain (slip, sinkage).
4. **The absolute σ envelope is unbounded** with odometry only.
5. **Freeze detection needs motion.** A frozen camera on a stationary rover is indistinguishable from a static scene.
6. **The per-cycle CPU target (15 ms p95) is missed** (section 3.6).
7. **Depth-unseen parts of objects stay uncovered** (section 3.5).

## 6. Prerequisites for R4.4 and R5

| For | Prerequisites |
|---|---|
| R4.4 | `MappingConfig` fusion and tracker fields, and the link-to-mapper callback. The cost layer and health are reused unchanged |
| R5 | Act on `get_stop_reasons()`. Re-observe the blind zone and stale corridors before moving (rotate-to-look or a rear sensor). Absolute localisation for global goals. A collision-capable world and contact monitor (plan D6). Decide D1-b if CPU headroom is needed |

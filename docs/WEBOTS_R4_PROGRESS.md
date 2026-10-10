# Webots R4 progress notes

One short note per sprint of [WEBOTS_R4_MAPPING_PLAN.md](WEBOTS_R4_MAPPING_PLAN.md). The numbers come from the headless runs
named below. Ground truth is used only by the offline evaluator (`scripts/run_webots_r4.py`, `src/artemis_x/evaluation/map_eval.py`).

Decisions in force (approved with "start implementing the plan", 2026-10-10):

| Decision | Choice |
|---|---|
| D0 | R3 baseline committed and pushed (`c425125`) |
| D1 | Mapping runs in the Webots controller, stdlib only |
| D2 | 0.10 m grid (used from R4.2) |
| D3 | Thresholds: ground ±0.03 m, obstacle ≥ 0.05 m, hazard ≤ −0.08 m |
| Others | Recommended options, applied when their sprint comes |

---

## R4.1 Frames, estimated pose and measurement pipeline: done, awaiting review

**What now exists**

| Item | Detail |
|---|---|
| `src/artemis_x/sim/mapping/` (stdlib only, imports under Webots' system Python 3.12.0) | `config.py` (all thresholds, hashed into each run). `frames.py` (`Pose2D`, SE(2), base↔odom). `pose.py` (`PoseEstimator` = the tour's own encoder odometry + 1σ model; `PoseBuffer` with interpolation; sensor `sample_time`). `depth_scan.py` (fast depth read from the C buffer; 5,040 precomputed rays; ground / positive / negative classification; per-column free and hit ranges). `mapper.py` (cycle every 0.256 s on a sensor sampling instant, telemetry) |
| Controller | The tour creates the mapper (`ARTEMIS_MAPPING=0` disables it; the R1/R2 demo keeps it off unless `ARTEMIS_MAPPING=1`). The tour's control uses the same odometry object, so the motion is unchanged |
| Frames and timing | The mapper captures each frame once, and the inference link submits that same frame |
| Pose in the detection log | `InferenceLink` now logs the **estimated** pose at the frame's sample time. Ground truth moves to a separate `gt_pose_eval_only` key, and the old `rover_pose_ground_truth` key is gone |
| Telemetry, per cycle | `tour/mapping/mapping.jsonl` (pose, valid fraction, class counts, columns, timings, ground-truth pose for evaluation only). `obstacle_points.jsonl` (≤ 150 points per cycle). `mapping_config.json` |
| Ground-truth guard | `tests/test_mapping_frames.py` fails if any module in `mapping/` touches the supervisor API, or uses a passed ground-truth value for anything except logging |

**Measured** (run `r4-measure-20261010-080610`, course tour, low sun, YOLO on)

| Check | Target (plan) | Result |
|---|---|---|
| Tour unchanged | 55/55, same stop accuracy | **55/55**; stops within 3.0 mm and 0.21°; true-path clearance 0.601 m (same as R3) |
| Mapping cycles | every 0.256 s | **2,001 cycles**, median period 0.256 s, 0 without pose, 0 without depth |
| Positive obstacle points vs true surfaces (169,477 points) | p95 ≤ 0.05 m | **p50 4.3 mm, p95 15 mm, max 40 mm; 100 % within 5 cm** |
| Negative (crater bowl) points (1,070) | — | p50 3.3 mm, p95 5.8 mm from the true terrain; 1,042 on the crater bowl, 28 on the rock inside it |
| Capture-time check | — | Pose at the sample instant: p95 14.8 mm. Pose one period (64 ms) earlier: p95 24.3 mm. The timing logic is right |
| Measurement cost per cycle (depth scan) | p95 ≤ 6 ms | **scan p95 4.8 ms**. Whole cycle, including the RGB conversion the R3 link already did: p50 6.4 ms, p95 10.4 ms, max 34 ms |
| Estimated vs true pose | — | max 0.5 mm and 0.03°. The simulator has no wheel slip, so this says nothing about real drift (plan risk R5) |
| Detection log | estimated pose only | 1,949 records: all with `rover_pose_estimated`, 0 with the old ground-truth key |
| Regression | green | Root 81 OK (63 + 18 new mapping tests), backend 52 OK, `verify_artifacts` 8/8, R1/R2 smoke pass, committed worlds regenerate byte-identical |

Figure: `docs/figures/webots/r4_1_obstacle_points.png` (obstacle points over the true rocks, crater and path).

**Findings to carry into the next sprints**

1. **Small pebbles are mostly invisible to depth.** Most of the Z2 pebbles (8–18 cm wide) are lower than the 0.05 m obstacle
   height and produced few or no points, as the plan anticipated. R4.2 will measure recall by height; the YOLO small-rock rule
   is R4.4.
2. **The empty small crater produced no hazard points.** It is 12 cm deep at its centre and only seen at a grazing angle from
   the lane, so almost no pixel falls below −0.08 m. The large crater's bowl and rim are mapped well. To revisit with the
   threshold or the grid evidence in R4.2/R4.3.
3. **The planned σ model is too pessimistic for the STOP rule.** σ_xy = 2 % of the distance travelled reached 1.33 m after the
   tour (about 66 m with all approach and return legs). The planned STOP at σ_xy > 0.3 m would fire after about 15 m. The model
   or the threshold must be revisited in R4.3; no change made now.
4. **Fewer detections answered than in R3.** 1,949 of 2,001 frames were answered, latency p95 47 ms, versus 1,994/1,996 in R3.
   An interactive Webots session was running on the same GPU during this run. Dropped frames are by design (one frame in
   flight); this is noted, not a fault.
5. **Port clash with an open Webots window.** The default port 5761 can be held by an open Webots GUI session, whose controller
   keeps reconnecting. Headless R4 runs therefore use port 5771 (`run_webots_r3.run_tour` gained a `port` argument).
6. **Slowest controller step.** The single slowest step (298 ms) is the PNG export at an observation stop, which predates R4.
   It is not mapping: the mapping cycle max is 34 ms.

**How to reproduce**

```powershell
uv run --project backend --no-sync python scripts/run_webots_r4.py measure       # tour + analysis (about 2 min)
uv run --project backend --no-sync python scripts/run_webots_r4.py analyse artemis-webots/output/r4-measure-<ts>
```

---

## R4.2 Local occupancy grid (depth only): done, awaiting review

**What now exists**

| Item | Detail |
|---|---|
| `mapping/grid.py` | `LogOddsGrid`: 0.10 m cells, 120 × 120 (12 × 12 m) rolling window in `odom`. Log-odds +0.85 / −0.40, clamped to [−2, 3.5]; OCCUPIED above 0.85, FREE below −0.85 (only if ever observed), otherwise UNKNOWN; NEGATIVE when the latest hit was below the ground plane. At most one update per cell per cycle, hits first. Recentring is an integer-cell shift once the rover is 1 m off-centre: cells keep their exact odom footprint, cells entering start UNKNOWN, cells leaving are dropped. Storage: stdlib `array`, 183 KB |
| Free-space evidence (`depth_scan.py`) | **Deviation from the plan, on purpose.** The plan marked free space from the sensor to the last ground return. That would have marked the camera blind zone and occluded crater bowls FREE. Now: (1) a segment joins two *pixel-consecutive* ground samples only if the range grows by about what flat ground gives (gap rule), and any obstacle, uncertain point or larger jump breaks the chain; (2) **overhang rule**: a ground sample with a nearer obstacle (≤ 0.40 m high) *above* it in the image lies under that obstacle and is not free evidence. The rule was found in the tour: the 24 cm-high camera saw ground under bulging boulders, the lying fuel tank and the tilted panel |
| Mapper | Each cycle is split over two simulation steps (capture + scan, then grid update on the next step). Snapshots go every 2 s to `tour/mapping/snapshots/`; a background thread writes them, so disk I/O never runs on the simulation thread. Telemetry per cycle in `mapping.jsonl`; process memory traced every 200 cycles |
| `mapping/snapshot.py` (`artemis.map.v1`) | JSON header + raw little-endian layers: `log_odds` float32, `last_obs_s` float32, `negative` uint8. Readers derive `state` with the `state_rule` stored in the header (stdlib reader included). **Deviation from the plan's layer list:** converting 14,400 cells per layer in pure Python cost about 44 ms per snapshot, `tobytes()` costs microseconds. The contract is frozen in R4.5 |
| Evaluator (`map_eval.py` + `run_webots_r4.py map / analyse-map`) | Ground-truth raster from the manifest. Hazard = rendered rock surface ≥ 5 cm above the local terrain, engineered objects, crater rims ≥ 5 cm; negative = terrain ≤ −8 cm; rocks < 5 cm = traversable by policy D3. Then: accumulated whole-course map (each world cell keeps its latest *observed* snapshot state), cell precision, false-free hazard cells per object, per-object recall by height for objects in view within 4 m, crater report, registration shift search, recentring consistency between snapshots, figures |
| Tests | `tests/test_mapping_grid.py` (14): log-odds arithmetic and thresholds; saturation and reversal; one update per cell per cycle with hits winning; negative kind; pose transform; recentring keeps odom positions and drops/adds cells; full chain on rendered depth (blind zone stays UNKNOWN, box shadow never FREE, hidden pit never FREE); overhang slab (with the rule disabled as a control); snapshot round trip. Plus a `SensorSuite.capture` regression test |

**Measured** (run `r4-map-20261010-083443`, full course tour, **YOLO disabled**, low sun; ground truth offline only)

| R4.2 acceptance (plan) | Target | Result |
|---|---|---|
| Free-cell precision (observed area) | ≥ 0.99 | **0.9968** (0.9998 excluding cells that touch a hazard edge) |
| Occupied-cell precision (observed area) | ≥ 0.90 | **0.925** strict; 1.000 within one cell |
| Recall, obstacles ≥ 0.10 m high in view within 4 m | ≥ 0.95 | **32/33 = 0.97**. The miss is the white ball, which is **occluded**: along its bearing from the Z5 stop the depth reads 1.5 m (the fuel tank) against 2.86 m to the ball. The in-view test does not model occlusion yet |
| Recall, 0.05–0.10 m | reported | **5/5** |
| Rocks < 0.05 m (Z2: rocks 03, 06, 08, 10) | traversable by policy | 0/4 mapped, by design; 16 FREE cells over them |
| Craters appear as hazards | large and small | **Large:** bowl detected (69 NEGATIVE cells, 196 UNKNOWN, 5 FREE wall-edge cells), rim 221/257 cells OCCUPIED. **Small (12 cm, empty): bowl NOT detected as negative.** All 33 bowl cells stay UNKNOWN (0 FREE), and its rim is mapped (30/33 OCCUPIED) |
| False-free hazard cells | reported | **82** strict, almost all single boundary cells of rocks. Only **6** lie inside an object (all 4 neighbours hazard). Before the overhang rule: 189 strict, 86 interior |
| Map alignment | — | Best registration shift **(0, 0)** cells (781 overlapping hazard cells) |
| Recentring correctness | — | 40 window shifts; **215,752** untouched cells compared across shifts, **0** mismatched |
| Mapping frequency | ≥ 3.9 Hz | **3.906 Hz** (2,001 cycles, 0 without pose or depth) |
| Mapping CPU per cycle | p50 ≤ 8 ms, p95 ≤ 15 ms | **p50 9.6 ms, p95 17.2 ms**, max 53 ms: slightly over. Read 0.6, scan 4.5, grid 4.2, snapshot hand-off 0.18 ms (p50) |
| Controller step | ≤ 20 ms excluding disk writes | Busiest mapping step p50 5.3 ms, **p95 10.9 ms**, max 44 ms (rare outliers). Whole-controller step max 282 ms: the pre-existing PNG export at observation stops (disk) |
| Memory | mapping ≤ 20 MB | Grid arrays 183 KB. Controller process 24 → 50 MB with YOLO off. With YOLO on: 66 MB at cycle 200, then flat 68–74 MB to cycle 2,000, so **no leak**; that run also keeps 8 RGB+depth frames for YOLO (about 19 MB). Mapping alone cannot be isolated from the process figure |
| YOLO disabled | mapping works | Yes; the YOLO-on run (`r4-measure-20261010-083610`) gives the **identical map**, cell for cell |
| R3 / R4.1 behaviour | preserved | Tour 55/55, stops within 3.0 mm and 0.21°, clearance 0.601 m; YOLO 2,000/2,001 frames OK, all with the estimated pose; obstacle points p95 15 mm (unchanged) |
| Regression | green | Root 96 OK (+15), backend 52 OK, `verify_artifacts` 8/8, R1/R2 smoke pass, committed worlds byte-identical |

Figures:

| File | Shows |
|---|---|
| `docs/figures/webots/r4_2_map_sequence.png` | The live 12 m window at 8 moments |
| `docs/figures/webots/r4_2_map_progress.png` | The accumulated map at 25, 50, 75 and 100 % of the tour |
| `docs/figures/webots/r4_2_map_final.png` | Final map, plus the error map: green correct, orange false hazard (none), magenta false free, grey never mapped, red ✗ missed, grey ○ below 5 cm |

**Bugs found and fixed during R4.2**

1. **No RGB images (introduced in this sprint).** `SensorSuite.capture` returned no RGB for every call because its new `rgb` parameter was shadowed by a local variable. Caught by the first YOLO-off tour (crash at an observation stop) before any YOLO run used it; regression test added.
2. **Snapshots overwriting each other.** `Path.with_suffix()` cut snapshot names at the decimal point (`map_00000.064`), so snapshots within the same second overwrote each other. Names are now integer milliseconds, the writer no longer uses `with_suffix`, and the test name contains a dot.
3. **Cell-boundary disagreement.** Cell indexing differed between `//` and `floor(/)` on exact boundaries (`7.0 // 0.1 == 69`). One expression is now used everywhere.
4. **float32 threshold.** A single hit (0.85f > 0.85) counted as occupied; the thresholds now carry an epsilon.
5. **Ground-truth guard caught a hand-off.** The guard flagged ground truth carried between steps inside a tuple. It is now carried only as an explicit `{"gt_pose_eval_only": …}` log record, which the guard accepts.

**Carried into R4.3 (not changed in R4.2)**

1. **σ model vs STOP threshold** (R4.1 finding 3): σ_xy reaches 1.33 m over the tour; the planned STOP at 0.3 m would fire after about 15 m. Untouched, as instructed.
2. **Small empty crater.** Not detected as a negative hazard. It is UNKNOWN, not FREE, and its rim is OCCUPIED. R4.3's unknown-cost policy must keep it non-traversable.
3. **Large-crater wall cells.** 5 bowl-wall cells are FREE (10 cm cells straddling the −3 cm band); R4.3 can inflate negative hazards.
4. **Cells under overhangs** are now UNKNOWN; the cost layer must treat them as non-traversable.
5. **CPU.** Per-cycle CPU is about 15 % over the plan target. Options: fewer depth samples, or decision D1-b (NumPy in the controller interpreter). To decide.
6. **Evaluator.** The in-view test should model occlusion (white ball) by R4.6.

**How to reproduce**

```powershell
uv run --project backend --no-sync python scripts/run_webots_r4.py map               # YOLO-off tour + grid evaluation (about 2 min)
uv run --project backend --no-sync python scripts/run_webots_r4.py analyse-map artemis-webots/output/r4-map-<ts>
uv run --project backend --no-sync python scripts/run_webots_r4.py measure           # YOLO-on tour + R4.1 measurement checks
```

---

## R4.3 Safety layers, inflation and uncertainty: done, awaiting review

Full release report: [WEBOTS_R4_3_SAFETY_REPORT.md](WEBOTS_R4_3_SAFETY_REPORT.md). Run `r4-safety-20261010-090802`, 8 scenarios.

| Item | Result |
|---|---|
| R4.2 audit | Complete and reproduced. One defect fixed: a syntax error in the R4.2 evaluation runner (figure-label edit), which had kept `r4_2_map_final.png` stale |
| Footprint | Measured from `Koala.proto`: circumscribed radius 0.210 m; collision radius 0.25 m, safety band 0.15 m, 2σ_rel margin up to 0.20 m |
| Cost layer | 0 / 1–252 / 253 / 254 / 255; UNKNOWN never lowered. Incremental tiles, shift on recentre (1 full rebuild per tour instead of 41) |
| Fault detection | Dropout 0.12 s, freeze 0.00 s, invalid depth 0.06 s, overrun 0.24 s, YOLO kill 0.48 s, all ≤ 0.5 s. A slow YOLO server is detected by the 1.0 s answer timeout (not ≤ 0.5 s) |
| Integrity | Faults never changed the map; frozen frames rejected; no FREE from invalid depth |
| Corridor vs ground truth | 0 dangerous "clear while blocked" cycles in all scenarios, including ±3 % odometry; obstacle recall 1.0 |
| Uncertainty (R4.1 finding 3, fixed) | Relative σ since each cell's last observation replaces the start-relative σ for safety. Measured error inside 2σ in 100 % of pairs (baseline and ±3 % / heading faults). The σ margin cuts unsafe near-field cells 2.5–3×. The absolute σ is reported, not a health reason |
| Health (baseline) | OK 54 %, STOP 46 % (stale corridor after excursions, obstacle at observation stops, unobserved start), DEGRADED 0 %. Advisory: the tour ran 55/55 unchanged |
| CPU per cycle | p95 15.0–15.8 ms (YOLO off), 19.0 ms (YOLO on): **target 15 ms missed** by 0–4 ms; busiest 32 ms step p95 ≤ 11.6 ms |
| Tests | +18 (`test_mapping_safety.py`); root 114 pass (the 10 errors are the R4.4 tests written ahead), backend 52, `verify_artifacts` 8/8, smoke pass, worlds unchanged |

---

## R4.4 YOLO + depth fusion and obstacle tracking: done, awaiting review

**Prerequisite.** R4.3 verified before starting: gate passed, report [WEBOTS_R4_3_SAFETY_REPORT.md](WEBOTS_R4_3_SAFETY_REPORT.md).

Evaluation run `artemis-webots/output/r4-fusion-20261010-095447`, four full course tours on the same code. Ground truth is used only in `scripts/r4_fusion_analysis.py`.

**Reproduce:**

```powershell
uv run --project backend --no-sync python scripts/run_webots_r4.py fusion        # 4 scenarios (about 13 min; the delay run is real time)
uv run --project backend --no-sync python scripts/run_webots_r4.py analyse-fusion artemis-webots/output/r4-fusion-<ts>
```

**What now exists**

| Item | Detail |
|---|---|
| `mapping/fusion.py` (new) | Each fresh YOLO detection → `RockMeasurement`. It uses the frame's OWN depth and the ESTIMATED pose at that frame's capture time (from the R3 ring buffer). It carries the visible-surface point, the footprint centre (surface + radius along the viewing ray), radius, height, range and σ = 0.03 + 0.02 × range. It also records the number of depth samples in the box at least 2 cm above the ground ("support"). **Occlusion:** a depth gap of more than 0.30 m inside the box marks it bimodal; the dominant cluster is used and σ doubles. **Small-rock rule:** depth points 2–5 cm high inside a *confirmed* rock's box become grid hits. YOLO only lowers the threshold where it points: the hits are measured geometry, and YOLO never lowers or clears a cell |
| `mapping/tracker.py` (new) | Persistent IDs. Association: greedy nearest neighbour, gate 0.25 + 0.05 × range + the larger radius. Inverse-variance position update, exponential moving average for radius, height and confidence. Lifecycle: tentative → confirmed after 2 *depth-supported* hits within 2 s; unconfirmed tracks expire after 3 s. Misses are counted only when the track should be visible (inside the FOV, ≤ 5 m, a depth return at its projected position, not behind nearer depth) → lost after 4 misses; lost tracks are kept and flagged. Merging of overlapping tracks |
| Track refinement against the depth map (~1 Hz) | Each confirmed track is snapped to its connected OCCUPIED blob. Blobs ≥ 0.5 m wide get a circle fit (a boulder's centre and radius from its visible arc); tracks sharing a blob ≥ 0.8 m wide are merged as fragments of one boulder. **Terrain guard:** blobs wider than 3 m, or with NEGATIVE (crater-bowl) cells in their bounding box, are never fitted or merged. This rule was added after a run where the crater rim was treated as a boulder and the rim rocks' tracks were absorbed |
| Integration | `InferenceLink` calls the mapper for every fresh (non-stale) result, including empty ones so misses can be counted. Track IDs are written into `detections.jsonl`. Snapshots carry confirmed and lost tracks in the header. The summary has fusion and tracker counters, capture→fusion latency and fusion CPU time |
| Unchanged | Cost layer and health monitor (identical health timeline in the fusion run), UNKNOWN handling, R3 tour (55/55) |
| Tests | `tests/test_mapping_fusion.py` (15): measurement geometry, capture-time pose, bimodal occlusion, small-rock points, no-depth rejection, one confirmed track from noisy repeats, two rocks stay two tracks, tentative expiry, merging, misses only when visible (open ground → lost, sky or occluder → no miss), unsupported boxes never confirm, boulder fragments merge with a fitted centre, rocks on a crater rim stay separate, touching small rocks are not merged, and the mapper `fuse()` rules (stale rejected, one persistent ID, small-rock hits, fusion never lowers a cell) |

**Measured** (ground truth offline; "in view" = within 4 m and inside the FOV at some point)

| R4.4 target | Target | Fusion run | 400 ms delay (real time) | Server killed mid-tour |
|---|---|---|---|---|
| Confirmed-track centre error, median | ≤ 0.10 m | **0.028 m** | 0.032 m | 0.023 m |
| Confirmed-track centre error, p95 | ≤ 0.25 m | **0.167 m** (max 0.217) | 0.191 m (max 0.217) | 0.151 m (max 0.160) |
| Duplicate rate (rocks with > 1 track) | ≤ 5 % | **3.2 %** (1 of 31: rock 19) | 3.1 % | 0 % |
| Distinct rocks tracked | — | 28 | 29 | 16 (YOLO stopped at 173 s) |
| False tracks on bare terrain | — | 1 | 0 | 0 |
| Tracks on crater rims (real raised hazards, labelled "rock") | — | 3 | 2 | 0 |
| Engineered objects labelled "rock" (YOLO false positives) | — | 3 (lander crate, fuel tank, mast) | 3 | 2 |
| Capture → fused latency, sim p50 / p95 | — | 0.128 / 0.288 s | 0.544 / 0.672 s | 0.128 / 0.256 s |
| YOLO answers | — | 2,000/2,001, 0 stale | 1,255 (one frame in flight), 0 stale, 0 evicted | 675, then server gone |

**Depth-only vs fusion maps** (same course, same code):

| Metric | Depth only | YOLO + depth |
|---|---|---|
| Obstacles ≥ 0.10 m mapped | 33/34 (miss: white ball, occluded) | 33/34 |
| 0.05–0.10 m mapped | 5/5 | 5/5 |
| < 0.05 m mapped as OCCUPIED | 0/4 | **1/4** (rock 10) |
| < 0.05 m tracked | — | **3/4** (rocks 06, 08, 10; rock 03 is never detected by YOLO) |
| Occupied-cell precision (strict / incl. low rocks) | 0.925 / 0.925 | 0.921 / 0.925 |
| False-free hazard cells | 82 | 78 |
| Unsafe collision-zone cells (cost < 253) | 39 | 39 |
| Health | — | Identical timeline (fusion does not change the safety layer) |

**Other findings:**
- **Occlusion (Z4, rock 14 behind rock 13).** Confirmed and tracked to **2.8 cm**; 84 of its 441 measurements were flagged bimodal and resolved to the dominant surface. R3 had placed it up to 0.66 m too near.
- **Disconnection.** `inference_unavailable` was raised **0.28 s** after the server was killed. Mapping continued for 1,326 cycles. The 16 confirmed tracks at the kill were kept; 2 were merged later by depth refinement, and none was deleted.
- **400 ms delay, run in real time.** In fast mode, 400 ms wall clock is about 3 s simulated: every result came back stale and nothing was fused. Run in real time, track accuracy is unchanged: median 0.032 m vs 0.028, p95 0.191 vs 0.167, an increase of 0.02 m against the plan's 0.05 m budget. Throughput halves because only one frame is in flight.

**CPU (wall clock, Webots fast mode)**

| Run | Cycle p50 / p95 | Busiest 32 ms step p95 | Fusion per result p50 / p95 |
|---|---|---|---|
| Depth only | 10.1 / 16.3 ms | 10.4 ms | — |
| With fusion | 16.2 / **24.3 ms** | 16.8 ms | 1.1 / 3.5 ms (in the inference link's step) |

The plan's 15 ms p95 target is missed (it already was in R4.3). The extra cost is refinement plus the larger set of touched cells. Every step still fits in 32 ms.

**Bugs found and fixed during R4.4** (each caught by a run or a test, then covered by a test):
1. **No misses were ever counted.** The occlusion test treated the ground in front of a rock as an occluder, and frames with no detections never reached the tracker. Now the track is projected into the image and the depth is checked there, and empty results are passed through.
2. **Boulder fragments.** Fragment boxes on boulders produced small radii, centres up to 0.72 m off, and a 22 % duplicate rate. Fixed by the size-aware gate and depth-blob refinement.
3. **False tracks without depth support.** Repeated weak YOLO boxes on bare terrain confirmed. Confirmation now needs depth support.
4. **Crater rim treated as a boulder.** The circle fit absorbed the rim rocks into one 0.84 m track (terrain guard added). The first guard (2.2 m) then rejected the real Z8 boulder blob (2.24 m with its neighbours); it was widened to 3 m, the NEGATIVE-cell rule handles craters, and the search radius now scales with the track radius.

**Annotated evidence** (`docs/figures/webots/`):

| File | Shows |
|---|---|
| `r4_4_annotated_stops.jpg` | Camera at the 9 observation stops with YOLO boxes, track IDs, confidence and range |
| `r4_4_tracks_and_maps.png` | Depth-only map vs fusion map with the confirmed tracks, coloured by what they really are |
| `r4_4_small_rocks_zoom.png` | Z2 pebbles: tracked by fusion, mostly not occupied |

**Limitations**
- **YOLO's recall bounds the gain.** Rock 03 is never detected; most sub-5 cm rocks are tracked but not made OCCUPIED, because depth still decides and YOLO adds evidence only where depth sees ≥ 2 cm.
- **Labels on engineered objects.** The crate, tank and mast are labelled "rock" (YOLO false positives, frozen model). Depth maps them as obstacles anyway, so safety is unaffected.
- **Track radius on unseen sides.** A boulder seen from one side has its radius estimated from the visible arc only.
- **Refinement assumptions.** The merge thresholds (0.8 m merge, 3 m terrain) are scene-scale assumptions for this course.
- **CPU target missed** (see above).

**Regression**

| Check | Result |
|---|---|
| Root suite | **129 OK** (+15 fusion tests; includes the live Webots test) |
| Backend | 52 OK |
| `verify_artifacts` | 8/8 |
| R1/R2 smoke | Pass |
| Committed worlds | Byte-identical |
| Mapping tests under Webots' Python | All OK |
| R3 tour | 55/55 in all four runs |

---

## Performance and release-readiness review (between R4.4 and R4.5): done, awaiting review

**Scope.** Profile and speed up the full R4.4 mapping pipeline without changing what it outputs. Investigate small-rock misses and false tracks. Re-run every regression and the complete course. Motor control, the frozen YOLO model, the datasets and the worlds are unchanged.

**Final evaluation runs:**
- `artemis-webots/output/r4-safety-20261010-144458` (8 scenarios);
- `artemis-webots/output/r4-fusion-20261010-145112` (4 scenarios).

The `r4_3_*` and `r4_4_*` figures in `docs/figures/webots/` were regenerated from these runs. The tables in sections R4.3 and R4.4 above (and in the R4.3 report) still describe the accepted runs.

**Profiling.** Enable it with `ARTEMIS_PROFILE=1` (`src/artemis_x/sim/profiling.py`). It writes cProfile output plus GC pause timings to the tour's output folder; it is off by default.

The profile of the R4.4 fusion tour showed five costs:

| Cost | Cause |
|---|---|
| Grid integration | 48 M `math.floor` calls |
| Depth scan | A second pass to build free segments |
| Inflation levels | Re-evaluated on every cycle |
| RGB conversion | A Python BGRA → RGB conversion of about 1.2 ms per frame |
| Garbage collection | Gen-2 pauses of up to 7.3 ms, 252 of them inside mapping steps |

**Changes** (stdlib only, no new dependencies):

| Change | Output |
|---|---|
| The depth scan builds free-space chains in the same pass as classification (`Column.free_chains`; `Column.segments()` still gives the R4.2 form) | **Identical** segments, points, counts and grid log-odds. Checked old against new on synthetic, noisy and recorded depth frames, and covered by `TestChainsEqualSegments` |
| Grid integration converts each chain point to a cell once, with an `int()` fast path | Identical |
| Fusion `measure()` uses inline math | Identical measurements |
| Camera frames are sent to the inference server as BGRA (`bgra8`); the server converts them with numpy | Identical detections (`test_bgra8_frames_give_identical_detections`) |
| Inflation levels are re-evaluated only after the rover moves ≥ 0.05 m or turns ≥ 0.05 rad, and are always computed with that headroom | Conservative: inflation is never smaller than before |
| Track refinement runs on idle simulation steps instead of inside the mapping cycle | Same ~1 Hz schedule |
| Garbage collection: automatic GC is off; young generations are collected on idle steps, with a full collection every 2 s on an idle step; restored in `finish()` | No GC pause inside a mapping step |

**CPU, wall clock** (fast mode, back-to-back runs on the same machine; ranges are 2 runs before and 3 runs after). The "before" runs used a copy of the R4.4 source taken before any optimisation.

| Run | Cycle p50 | Cycle p95 | Cycle max | Busiest step p95 / max | Memory at end |
|---|---|---|---|---|---|
| Fusion, before | 12.7–14.6 ms | 18.3–21.6 ms | 26.6–35.0 ms | 10.8–13.4 / 19.7–21.9 ms | 71.7–72.5 MB |
| Fusion, after | 7.7–7.9 ms | **11.4–12.0 ms** | 18.7–21.2 ms | 8.2–8.5 / 15.0–18.3 ms | 72.9–74.1 MB |
| Depth only, before | 9.8–10.2 ms | 15.3–15.8 ms | 23.7–24.4 ms | 9.2–9.5 / 15.3–15.7 ms | 50.7–50.8 MB |
| Depth only, after | 6.7–7.1 ms | **10.2–10.7 ms** | 17.6–18.0 ms | 7.6–7.8 / 14.4–15.4 ms | 49.0–49.2 MB |

Fusion stage p95, before → after:

| Stage | Before | After |
|---|---|---|
| Depth read | 2.4–3.4 ms | 1.0–1.2 ms |
| Scan | 8.2–9.4 ms | 2.9 ms |
| Grid | 4.5–5.6 ms | 3.4–3.5 ms |
| Cost | 6.0–7.3 ms | 5.2–5.3 ms |
| Fusion per result | 3.1–4.0 ms | 2.0 ms |

Full GC collections now take p50 1.4 ms / max 5.2 ms, always on idle steps. The plan's **15 ms p95 target is met** in every final run: safety and fusion batches range from 9.9 to 11.7 ms. Run-to-run machine variance is about ±15 %.

**Small-rock investigation and change (plan D3 / 6.4).**

Track data showed that rocks 06 and 08 (2.8 and 4.4 cm high) were *tracked* to within 2 cm and 0.7 cm. They were not mapped because:
- their tracks went **lost** once YOLO stopped detecting them;
- later depth passes then cleared their cells through the ±3 cm ground band.

Two changes, both behind `fusion_protect_small_rocks` (default on):
1. Confirmed small-rock tracks (radius ≤ 0.15 m) withhold FREE evidence from their footprint (track radius + 5 cm), **also once lost**. The rock is still physically there.
2. Each association adds +0.4 log-odds to the footprint of a confirmed small rock.

YOLO still never clears space. Tentative tracks protect nothing. Covered by `test_confirmed_small_rock_is_protected_even_when_lost`.

| Fusion run | Before | After |
|---|---|---|
| < 5 cm rocks mapped as OCCUPIED | 1/4 | **3/4** (rock 03 is never detected by YOLO) |
| False-free hazard cells | 78 | 70 |
| Unsafe collision-zone cells | 39 | **27** (0 inside a hazard) |
| Occupied precision (strict / incl. low rocks) | 0.921 / 0.925 | 0.917 / 0.925 |

The small drop in strict precision is the pebble footprints. They are real rocks but are not counted as hazards in the strict ground truth.

**False tracks (reported, not changed).**
- **Crater rim.** 3 unmatched tracks sit on the large crater's rim and 1 on its inner bowl wall. They are real raised terrain labelled "rock" by YOLO. Confidence does not separate them (one has 0.695), and the rim rocks share their depth blob, so a geometric relabel risks deleting real rocks.
- **Engineered objects.** 3 engineered objects are labelled "rock" (YOLO false positives, frozen model). Depth maps them as obstacles anyway, so safety is unaffected.

**Complete course on the final code**

All 12 tours completed 55/55 moves. Safety batch compared with the accepted R4.3 run:

| Scenario | Fault detection | Corridor (dangerous clear while blocked) | Map / health |
|---|---|---|---|
| Depth-only scenarios | Identical | — | Identical maps and health timelines |
| Depth faults | Identical latencies: 0.12 / 0.00 / 0.064 / 0.24 s | 0 | — |
| Odometry faults (±3 % scale, heading) | — | 0 | Same results, within 2 % |
| YOLO disconnected | `inference_unavailable` raised 0.40 s after the kill | 0 | — |
| YOLO stale | `inference_stale` 22.5 s (18.4 s before) | 0 | Health DEGRADED throughout, as before |

The YOLO-stale latency is a **fast-mode wall-clock artefact**, not a regression: the 1.5 s delay is wall time and the simulation now advances further per wall second. `inference_unavailable` is active from the start in both runs.

Fusion batch compared with the accepted R4.4 run:

| R4.4 target | Fusion | 400 ms delay (real time) | Server killed |
|---|---|---|---|
| Median centre error ≤ 0.10 m | 0.034 m (was 0.028) | 0.038 m (was 0.032) | 0.035 m |
| p95 centre error ≤ 0.25 m | 0.162 m (was 0.167) | 0.187 m (was 0.191) | 0.228 m (max 0.30) |
| Duplicate rate ≤ 5 % | **0 %** (was 3.2 %) | 0 % | 0 % |
| Rocks tracked | 28 | 29 | 18 |

- The slightly higher median comes from the footprint evidence: refinement now snaps to slightly larger blobs. Optimisation alone left tracking identical (0.028 m).
- The server was killed later in simulated time (244 s vs 173 s): the kill is wall-clock based. More tracks were therefore tracked, and the p95 includes harder rocks; it is still within the target.
- `inference_unavailable` was raised 0.64 s after the kill, within the 1 s timeout.
- No confirmed track was deleted after the disconnection.
- The occluded rock 14 is still tracked to 3.8 cm.

**Regression on the final code**

| Check | Result |
|---|---|
| Root suite with the live Webots test | **134 OK** |
| Backend | 52 OK |
| Mapping and R3 tests under Webots' Python 3.12 (stdlib only) | 84 OK (1 skipped: the live inference-server tests need the ML environment; they pass in the root suite, which has no skips) |
| `verify_artifacts` | All pass |
| R1/R2 smoke | Pass |
| Committed worlds | Byte-identical |
| Frozen model SHA | `413d0481…` unchanged |

**Remaining limitations**
- **Small-rock recall is bounded by YOLO.** Rock 03 is never detected; depth alone cannot resolve rocks under 5 cm.
- **Protected small-rock cells stay OCCUPIED for the whole session**, including lost tracks. This is conservative; a false track of radius ≤ 0.15 m would leave a small permanent obstacle. None were seen on open ground in these runs.
- **Crater-rim tracks and engineered objects labelled "rock" remain** (see above).
- **Timings are wall-clock in Webots fast mode** on one machine, with ±15 % variance. Busiest-step maxima of about 15–18 ms still fit the 32 ms step. The controller's single first step (~150–190 ms, setup and server handshake) is unchanged.
- **The inflation-level gate** trades exactness for conservatism: levels can be up to 5 cm / 0.05 rad "ahead" of the rover's real uncertainty.

### Release conditions (small-rock protection, approved for the static lunar simulation)

**Persistence and reset.**
- Within a mission, a confirmed small rock (radius ≤ 0.15 m) stays an obstacle, also after its track is lost.
- `Mapper.reset_map(reason)` is the only thing that clears it. It:
  - sets every cell back to UNKNOWN, so nothing is traversable;
  - drops all tracks; ids keep increasing and are never reused;
  - clears protected cells and provenance;
  - rebuilds the cost layer;
  - discards any scan captured before the reset.
- The reset is logged as a `map_reset` event in `mapping.jsonl`, and counted in the summary (`map_resets`, `grid.resets`, `tracker.resets`).
- Nothing calls the reset automatically; it is an explicit mission or operator action. R4.5 must decide who may call it.

**Provenance.** Each grid cell carries bit flags for the evidence that raised it since the last reset. They are exported as the additive snapshot layer `provenance` (`artemis.map.v1`, uint8):

| Flag | Meaning |
|---|---|
| 1 | Depth hit |
| 2 | YOLO-selected small-rock depth points |
| 4 | YOLO small-rock footprint |
| 8 | Currently protected from FREE evidence |

The summary splits OCCUPIED cells into `depth_only`, `yolo_only` and `depth_and_yolo` (`grid.occupied_by_source`). The flags only describe the evidence; they are never read by the mapping decisions.

**Regression tests** (`tests/test_mapping_fusion.py`, `TestReleaseSafety`):

| Test | What it checks |
|---|---|
| Persistent false track | A confirmed false track on bare ground is never cleared (it stays UNKNOWN and non-traversable), also once lost. It is flagged protected and not depth-based. After `reset_map` depth clears it |
| Lost then re-detected rock | It stays OCCUPIED while lost, even though depth sees only ground. Its provenance is YOLO-only. Re-detection reuses the same track (no duplicate) and re-confirms it |
| Map reset | Map, tracks, protection, provenance and cost are cleared, and mapping continues from depth only. Ids are not reused; the event and counters are logged |
| Snapshot provenance | Depth and YOLO hazards are distinguishable in a written snapshot, and every hazard cell is attributed |

**Known residuals.** Measured on `r4-fusion-20261010-145112`, fusion run; ground truth offline.

- **70 false-free hazard cells.** These are true hazard cells whose latest map state is FREE.
  - 64 of the 70 lie on the boundary of a rasterised ground-truth footprint, spread over 25 objects with at most 7 cells each (boulder 25, h 1.31 m). The remaining 6 are interior footprint cells; I did not attribute them further.
  - **All 70 carry cost ≥ 253** (inscribed or lethal) through the inflation of neighbouring OCCUPIED cells. None is a traversable cell for the rover centre.
- **27 unsafe collision-zone cells.** These are cells within 0.25 m of a true hazard (but not inside one) that the cost layer leaves below 253. 21 have cost 0; the others are graded (12–161). All lie 0.10–0.22 m from the true footprint; **0 are inside a hazard**.

| Object | Cells | Cause |
|---|---|---|
| `crs_rock_34` (Z8, h 0.12 m) | 9 | Observed only from beyond the 4 m "in view" range, where the sparse depth rows leave its rim as ground |
| `crs_rock_07` (Z2, h 0.16 m, 0.18 m wide) | 9 | Mapped, but its mapped footprint is smaller than or offset from the true one by up to about 0.2 m; the exact cause was not investigated |
| `crater_small_empty` (0.12 m deep) | 7 | The shallow outer bowl is above the −0.08 m hazard threshold (decision D3) |
| `crs_rock_18`, `crs_rock_11` (large boulders) | 1 each | — |

  **27 is the floor, not a constant.** Across the 9 YOLO-on tours of the two final batches the count was 27 (3 runs) or 37–38 (6 runs). The extra 10–11 cells all belong to `crs_rock_19` (Z6 shadowed, true radius 0.15 m, exactly the protection limit):
  - when its estimated track radius ends up ≤ 0.15 m, the rock is protected and its cells drop out;
  - otherwise they remain, exactly as in depth-only mapping (39 cells, 11 of them on rock 19).

  Which case happens depends on the asynchronous YOLO timing of the run (fast mode is not bit-reproducible). 0 cells inside a hazard in every fusion-on run.

  The rover's own corridor check had 0 dangerous clears in every run. These cells are the margin that R4.5 planning must respect (plan decision D3: collision radius 0.25 m).
- **0.64 s inference-disconnect detection.**
  - `inference_unavailable` is raised either when the socket to the server fails or when no answer has arrived for `inference_timeout_s` = 1.0 s (simulation time). The design bound is therefore at most 1.0 s plus one 0.256 s cycle, about 1.26 s.
  - Measured over 5 kill runs: 0.28 s (accepted R4.4), 0.40 and 0.64 s (performance-review batches), 0.58 and 0.44 s (release batches).
  - The spread comes from where in the cycle the kill lands, and from converting the runner's wall-clock kill time to simulation time, which has a resolution of about one cycle. In fast mode, 0.64 s of simulation time is under 0.1 s of wall time.
  - While undetected, the map is still depth-driven: YOLO never clears space, so the only loss is that a few small-rock observations are not fused.

**Release regression** (after the changes above; runs `r4-safety-20261010-153012` and `r4-fusion-20261010-153714`)

| Check | Result |
|---|---|
| Root suite with the live Webots test | **138 OK** (+4 release tests) |
| Backend | 52 OK |
| Stdlib tests under Webots' Python | 88 OK (1 skipped: the live inference-server class needs the ML environment; it runs in the root suite) |
| `verify_artifacts` | All pass |
| R1/R2 smoke | Pass |
| Committed worlds | Byte-identical |
| Complete course | 12/12 tours 55/55 |
| Safety scenarios | Fault-detection latencies identical; 0 dangerous corridor clears |
| Fusion | Track median 0.034–0.037 m, p95 0.165–0.228 m, 0 % duplicates, < 5 cm rocks 2–3/4 |
| Cycle p95 | 10.2–13.3 ms |

The provenance flags add no measurable time: grid integration is 2.2 ms per frame with and without them in a stdlib micro-benchmark. The 1 ms higher p95 in the safety batch is machine variance.

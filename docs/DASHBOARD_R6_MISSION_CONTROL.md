# R6 — Mission Control dashboard on the live ground station

The existing Mission Control dashboard (`frontend/apps/web`, Next.js 16, React 19, Tailwind 4, shadcn/ui) now shows
real ground station data (R5.5 API v1). The layout, panels, palette and navigation are unchanged. The mock data was
removed, and each panel now reads REST and WebSocket data. This is still the only dashboard.

**Rule: the dashboard never shows a value it did not receive.** No simulated telemetry, no interpolated positions and
no Webots ground truth. Before any data arrives, a panel shows an explicit empty, loading or error state.

## Screens (same panel slots as before)

| Screen | Panel | Data |
|---|---|---|
| Mission Overview | Header (freshness pill, mission, rover clock), *Mission Overview*, *Mission Analytics* | `GET /missions/{id}`, WS `mission.snapshot`, `health.changed`, `link.status` |
| Live Lunar Map | *Lunar Surface Map* (fit mission / follow rover, expandable), *Mission Extent* | `GET /missions/{id}/positions` (paginated), `GET /missions/{id}/obstacles`, WS `rover.position`, `obstacle.*`, `map.reset` |
| Obstacle picture | *Obstacle Picture* (range, confidence, uncertainty, source, tracking status) | same as the map |
| Telemetry Monitor | *Telemetry* (x/y, heading, σxy, packets/s, gaps·dup·late, data age), *Mission Log* | WS `rover.position`, `telemetry.received`, `link.status`; `GET /missions/{id}/events` |
| Lunar Network Control | *Lunar Network* (preset buttons, link state, loss, latency, throughput, blackout and recovery, rate and loss chart) | `POST /network/preset`, confirmed by `GET /network/status`; `GET /missions/{id}/link`; WS `link.status` |
| Mission History | *Mission History* (paginated, "Follow latest") | `GET /missions?limit&offset` |

The rover card and the decorative artwork (sidebar terrain, header moon) are static platform content, not telemetry.
The old camera, YOLO overlay and depth images were sampled from the design reference. They were replaced because protocol v1
carries no video downlink, and showing those images as "LIVE" would have been fabricated data.

### Map semantics (odom frame, x right, y up)

The map follows the design reference's style:

- glowing explored area;
- dashed glowing path;
- rock and hazard badges;
- rover halo with its camera cone;
- START flag, scale bar and legend.

The terrain texture is decorative noise inside the explored area, not imagery. Every shape comes from received telemetry. The map supports:

- wheel or `+`/`−` to zoom;
- drag to pan;
- **Fit**, for the whole mission;
- **Follow**, centred on the rover.

The axes widget shows odom `+x`/`+y`: rover odometry has no absolute north.

- **Unknown** (dark, hatched background): the occupancy grid is not downlinked, so no area is drawn as free by default.
- **Explored (camera)**: the camera footprint (FOV 1.0 rad × 6 m mapping range) swept from received poses. It shows
  where the sensors looked, not proven free space, because obstacles can shadow it.
- **Traversed (free)**: the rover's footprint along received poses. Only space the rover itself occupied counts as free.
- **Received path**: received poses only, ordered by observation time. A gap of more than 3 s in `t_obs_s` (packets
  that never arrived) breaks the line, so no position is invented between two receptions.
- **Obstacles**: the reconciled ground picture.
  - Fill opacity follows confidence.
  - A dashed ring shows `radius + uncertainty`.
  - The stroke shows the tracking status: confirmed is solid, observed is dashed, lost is faint.
  - Badges keep a constant on-screen size: a green triangle for a rock, a red one for a negative hazard and an amber
    diamond for a depth-only obstacle. A lost track is grey.
  - Ids are labelled when there are 14 obstacles or fewer, on hover, and on selection.
- **Obstacle removal**: an obstacle disappears only on `obstacle.removed`, on `map.reset`, or on a `mission.snapshot`
  resync, which is the ground station's persisted reconciled picture. On resync, detections newer than the snapshot
  (not yet written by the asynchronous writer) are kept.

## Freshness: two clocks, never mixed

- `t_obs_s` is the rover observation time in **simulation seconds**. It orders poses and is shown as "Rover clock" and "Last observation".
- `rx_utc` is the ground reception time on the **UTC wall clock**. Every age and the freshness state come from it.
- The browser↔ground clock offset is estimated from server timestamps (`meta.generated_utc`, `link.status.rx_utc`).

| State | Condition |
|---|---|
| **LIVE** | WebSocket open and a rover packet reached the ground ≤ 3 s ago |
| **STALE** | WebSocket open, last reception older (blackout, link loss, rover stopped). The last map is kept, greyed, with a STALE banner and its age |
| **DISCONNECTED** | No live stream from the ground station, whatever the data age |
| **NO DATA** | Connected, nothing received yet for this mission |

`link.status` samples are produced by the ground itself, so they never refresh freshness. Only receptions of rover packets do.

## Real-time behaviour

1. **Initial REST load** (`lib/ground/use-mission.ts`) loads, in parallel:
   - the mission;
   - the latest 5,000 poses, paginated through `meta.pagination.next_offset`;
   - every present obstacle (paginated);
   - 120 link samples;
   - 100 stored events.
2. **WebSocket** `ws://…/ws/v1/missions/{id}`:
   - Its first message is `mission.snapshot` (resync), followed by live events.
   - Events are buffered and applied every **200 ms** in one state update, so a telemetry burst costs one render instead of one per packet.
3. **Reconnect**:
   - Exponential backoff from 0.5 s to 10 s, plus an immediate retry on the browser's `online` event.
   - After a reconnect, the server snapshot resyncs obstacles and a REST backfill (`since_t_obs_s`) adds the poses persisted meanwhile, including store-and-forward deliveries.
4. **No duplicates, no mixing**:
   - Poses are unique by `t_obs_s` and late poses are inserted in observation order.
   - Obstacles are keyed by `obstacle_id`, and log lines have stable keys.
   - Every event whose `mission_id` differs from the selected mission is dropped.
   - A mission switch closes the socket, discards pending events and starts from an empty state.
5. **Mission selection**: `?mission=<id>` pins a mission. Otherwise "Follow latest" picks the newest active mission and switches when a new rover session starts.
6. **Network control**:
   - `POST /network/preset` is reported as applied only after a fresh `GET /network/status` returns that preset.
   - Status is polled every 2 s while the tab is visible.
   - Blackout duration is measured by the emulator (`blackout_total_s`).
   - Recovery is the time from leaving BLACKOUT to the next rover packet received.
   - These endpoints only answer local clients (R5.5), so the dashboard must run on the same machine as the ground station.
7. **Errors**:
   - Every non-2xx response becomes an `ApiError(status, code, message, request_id)` taken from the `{error, meta}` envelope.
   - The affected panel shows it with a *Retry* action. Examples: ground station unreachable, `mission_not_found (404)`, `emulator_unavailable (503)`.

## Code map (`frontend/apps/web`)

| File | Role |
|---|---|
| `lib/ground/types.ts` | API v1 contracts (REST envelopes, mission, pose, obstacle, link, WS events) |
| `lib/ground/config.ts` | `NEXT_PUBLIC_GROUND_API`, `NEXT_PUBLIC_GROUND_WS` |
| `lib/ground/api.ts` | Typed REST client, `ApiError`, pagination helper |
| `lib/ground/mission-state.ts` | Pure state: hydrate (REST), apply events (WS), dedup, mission isolation, log, link samples |
| `lib/ground/freshness.ts` | LIVE / STALE / DISCONNECTED / NO_DATA from UTC |
| `lib/ground/trajectory.ts` | Gap-aware trajectory segments, render decimation |
| `lib/ground/use-mission.ts` | REST load + WebSocket, batching, reconnect, resync |
| `lib/ground/use-network.ts` | Emulator control (POST + GET confirmation), mission history |
| `components/mission/lunar-map.tsx` | SVG odom map |
| `app/page.tsx` | The existing dashboard, wired to the above |

## Running it

```powershell
# 1-5: the R5.5 stack (docs/TELEMETRY_R5_5_GROUND_BACKEND.md section 5): MongoDB, ground backend on :8000,
#      lunar link emulator (--reverse), YOLO inference server, Webots with ARTEMIS_TELEMETRY=1 / _DTN=1 on
#      artemis-webots\worlds\artemis_r4_monitor.wbt (the course WITH the onboard 2D LOCAL MAP panel)
# 6. Dashboard (Node >= 24, pnpm 11)
cd frontend
pnpm install
copy apps\web\.env.example apps\web\.env.local      # optional: defaults are http://127.0.0.1:8000 / ws://127.0.0.1:8000
pnpm --filter web dev                                # http://localhost:3000
```

The backend CORS default allows `http://localhost:3000` and `http://127.0.0.1:3000`
(`ARTEMIS_GROUND_CORS_ORIGINS`). `NEXT_PUBLIC_*` values are inlined at build time, so rebuild after changing them.

### Checks
```powershell
cd frontend
pnpm --filter web test          # node --test: state, dedup, isolation, freshness, gaps (16 tests)
pnpm --filter web lint
pnpm --filter web check-types
pnpm --filter web build
# full stack, real processes + headless Edge driving the dashboard's own controls (dashboard dev server on :3000):
uv run --project ground_station --no-sync python scripts/r6_dashboard_e2e.py --out artemis-webots/output/r6_dashboard
```

## Validation (2026-10-10/11, real stack)

**Unit and build checks.**
- `pnpm --filter web test`: 16/16 pass.
- `lint`, `check-types` and `build` for `web` pass.
- `@repo/ui#lint` (the untouched shared starter package, ESLint 10.9.1) crashes inside ESLint with `scopeManager.addGlobals is not a function`. This predates R6 and is independent of the dashboard.

**Full stack.** `scripts/r6_dashboard_e2e.py` runs MongoDB 8 (docker), the FastAPI ground backend, the lunar link emulator (`--reverse`), the YOLO inference server and Webots (realtime, DTN telemetry, `artemis_r4_monitor.wbt`). The dashboard is driven by headless Edge through the dashboard's own buttons.
- **Result: 12/12 checks pass.**
- **0 browser console errors.**
- Screenshots: [`docs/screenshots/r6/`](screenshots/r6/).

| Check | Result |
|---|---|
| Backend down | Explicit `ground station unreachable` error with Retry. Header shows DISCONNECTED, no data drawn ([00](screenshots/r6/00_backend_down.jpg)) |
| History from MongoDB | 6 stored missions listed before the rover starts ([01](screenshots/r6/01_history_from_mongodb.jpg)) |
| Live mission | New session followed automatically and LIVE 8.6 s after Webots start: pose, heading, σxy, 2 packets/s ([02](screenshots/r6/02_live_mission.jpg)) |
| Obstacle detection → dashboard | Dashboard ids equal `GET /obstacles` ids (8/8): YOLO+depth tracks and depth-only obstacles ([03](screenshots/r6/03_obstacle_detected.jpg)) |
| Map zoom / follow | Zoom-in narrows the view from 17.07 m to 12.19 m; Follow centres on the rover ([03b](screenshots/r6/03b_map_follow.jpg)) |
| BLACKOUT (UI button) | Emulator confirms BLACKOUT. Header and panels go STALE, the map is kept and greyed with "no telemetry for 10.0 s", and the pose count stays frozen at 18 → 18 ([04](screenshots/r6/04_blackout_stale.jpg)) |
| RESTORE (UI button) | Back to LIVE. Poses 18 → 35, equal to REST (35). Blackout 10.3 s (emulator) and first packet 8.8 s after restore. The path is not bridged across the blackout ([05](screenshots/r6/05_restored_live.jpg)) |
| Backend killed | DISCONNECTED banner, last picture kept, "Reconnecting in 1 s" ([06](screenshots/r6/06_backend_down_disconnected.jpg)) |
| Backend restarted | WebSocket reconnects (1 reconnect). Snapshot and backfill resync to 50 poses = REST 50. Obstacle ids unique ([07](screenshots/r6/07_reconnected.jpg)) |
| Mission switch | Older mission: 82 poses = REST, 11 obstacles = REST, unchanged over 4 s while the live mission flows (no mixing) ([08](screenshots/r6/08_mission_switch.jpg)) |
| Dashboard restart | New browser session, `?mission=`: mission reloaded from the API (68 → 75 poses) ([09](screenshots/r6/09_dashboard_restart.jpg)) |
| Unknown mission | `mission_not_found (404)` shown, no stream opened, nothing drawn ([10](screenshots/r6/10_unknown_mission_404.jpg)) |

**Onboard Webots panels during the same run.** The R4.5 monitor saved its four panels at t = 45 s with telemetry, DTN and the dashboard all active:
- [RGB](screenshots/r6/webots_monitor_rgb.jpg)
- [AI DETECTION](screenshots/r6/webots_monitor_ai.jpg)
- [DEPTH](screenshots/r6/webots_monitor_depth.jpg)
- [LOCAL MAP (2D)](screenshots/r6/webots_monitor_map.jpg)

The 2D map panel exists only in `artemis_r4_monitor.wbt`. That world is the test course plus the four displays. `artemis_test_course.wbt` shows the camera, range-finder and `yolo_view` overlays, but no map.

**Shell.**
- The header is sticky, so the sidebar stays aligned when the page scrolls.
- Below 900 px of height the decorative landscape and motto are hidden, so navigation items are never covered.
- The active navigation item follows the panel in view (scroll spy).

**Regressions.**
- Repository suite: 200 tests, 199 pass and 1 Webots test is skipped. One error remains: the pre-existing `test_notebook_smoke` import error (`nbformat` not installed).
- Ground station suite: 9/9.
- No backend, protocol, model or world file was changed.

### Known limitations
- **No video downlink.** Telemetry protocol v1 carries no camera or depth images, so the dashboard has no live camera panel.
- **Free space.** The occupancy grid is not downlinked: only the traversed corridor is shown as free. *Explored (camera)* is sensor coverage, not free space.
- **Delay figures** are the emulator's cumulative p50 and its configured latency, not an end-to-end per-packet measurement.
- **Recovery time** is measured on the browser clock from the confirmed RESTORE to the next rover packet.
- **Blackout duration.** When the page opens in the middle of a blackout, the ongoing duration counts from page load. The completed duration needs the pre-blackout emulator total.
- **Trajectory size.** The first load is capped at the latest 5,000 poses (≈ 42 min at 2 Hz) and the in-memory trajectory at 20,000.

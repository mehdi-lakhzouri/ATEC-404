# Artemis-X Ground Station Backend (R5.5)

| | |
|---|---|
| Status | Implemented; validated end to end with real Webots telemetry through the lunar link emulator |
| Project | `ground_station/`, a dedicated uv project. `backend/pyproject.toml` and `backend/uv.lock` are protected reproducibility files (frozen ML environment), so they are left untouched |
| Stack | FastAPI 0.143, Pydantic 2, PyMongo 4.18 (native async API), structlog 26, uvicorn; MongoDB 8.0.4 in Docker |
| Reuses | `artemis_x.telemetry`: protocol v1, `GroundReceiver` (UDP, CRC, strict decode, ACKs), `GroundMap` (reconciliation), the R5.3 emulator control port. One additive hook: `GroundReceiver(on_packet=...)` |
| API | **Versioned**: REST under `/api/v1`, WebSocket under `/ws/v1`, every response in a `{data, meta}` envelope, OpenAPI at `/docs` |
| Tests | `ground_station/tests/test_ground_backend.py` (9 tests against the real MongoDB and real localhost UDP) |
| End to end | `scripts/run_webots_r4.py ground` → `scripts/r5_ground_e2e_analysis.py`; run `artemis-webots/output/r5-ground-20261010-231241` |

## 1. Architecture

```
Webots --R5.1 binary--> R5.4 DTN sender --udp 5791--> R5.3 emulator --udp 5790--> GroundReceiver (thread, unchanged R5.4 code)
   ^ rover udp 5793 <-- emulator reverse <-- udp 5794 <-- selective ACKs ------------------+   | on_packet(pk) for VALID packets only
                                                                                               v
                                       per-session GroundMap (dedup, observation-time order, full snapshots, resets) -> delta
                                                                                               | call_soon_threadsafe (never blocks)
                                                                                               v
       WebSocket hub (bounded fan-out) <-- live events -- event loop -- bounded asyncio queue --> single MongoDB writer (retry/backoff)
             /ws/v1/missions/{id}                                                                missions · rover_positions · obstacles
                                                       REST /api/v1/...  <---------------------- telemetry_events · link_metrics
                                                       POST /api/v1/network/preset --udp 5792--> the real emulator
```

**Key properties:**
- **Only delivered packets.** Only packets the receiver actually decoded (CRC and strict decode passed) are ingested. A packet the emulator dropped never reaches the backend.
- **Reception never waits on MongoDB.** UDP, ACKs and reconciliation run in the receiver thread; the database writer is a separate task fed through a bounded queue (20,000 items; overflow is dropped **and counted**).
- **Database outages are absorbed.** The writer retries the same item with backoff, up to 5 s; the queue absorbs the outage. WebSocket events are published at reception, so the live view keeps working while the database is down.
- **One reconciled picture per mission.** Each session gets its own R5.4 `GroundMap` (in-memory LRU of 16 sessions; MongoDB keeps every mission). This was a bug found in the live run, now covered by a test: a single shared map let a second mission's obstacles (same IDs) update and remove the first mission's.
- **Idempotent.** Packet-derived documents use `_id = "<mission>:<seq>"` plus unique `(mission_id, seq)` indexes, so retransmissions are stored once and counted as duplicates.
- **Two clocks, never mixed.** `t_obs_s` is the rover's observation time in **simulation seconds**; `*_utc` is the ground reception time in **UTC**. Freshness (`last_rx_age_s`, `link_up`) uses UTC only.
- **Logging (structlog).** JSON lines with UTC ISO timestamps. Each HTTP request carries an `X-Request-ID` (returned in the header and in `meta.request_id`) and logs a line with method, path, status and duration. uvicorn and PyMongo logs go through the same pipeline.

## 2. MongoDB (`docker/compose.yml`)

| Item | Setting |
|---|---|
| Image | `mongo:8.0.4` (pinned) |
| Container | `artemis-mongodb`, `restart: unless-stopped` |
| Port | `127.0.0.1:27017` (localhost only) |
| Volume | `artemis-mongodb-data` (named) |
| Credentials | From `docker/.env` (git-ignored; template `docker/.env.example`) |
| Health check | `mongosh ... db.adminCommand('ping')` |

**Prerequisites (Windows):** Docker Desktop with the WSL 2 backend, running. Check with `docker version` and `docker compose version`.

```powershell
Copy-Item docker/.env.example docker/.env                 # then set a long random MONGO_ROOT_PASSWORD
docker compose -f docker/compose.yml up -d
docker compose -f docker/compose.yml ps                   # STATUS: "Up ... (healthy)"
docker compose -f docker/compose.yml down                 # stop (data kept in the volume; add -v to delete it)
```

**Collections** in `artemis_mission`; the indexes are created at startup:

| Collection | Content | Indexes |
|---|---|---|
| `missions` | `_id` = mission ID (the rover's telemetry session), session, status, `created_utc`, `last_rx_utc`, first/last `t_obs_s`, counters (packets, pose, obstacles, event, duplicates, map_resets), latest health, config (frame, receiver, ACK target, protocol) | `session` (unique), `(status, last_rx_utc)` |
| `rover_positions` | Received POSE: x, y, yaw, heading, sigma (with saturation flag), frame `odom`, seq, `t_obs_s`, `rx_utc` | `(mission_id, seq)` unique, `(mission_id, t_obs_s)`, `(mission_id, rx_utc)` |
| `obstacles` | **Latest reconciled state** per obstacle: identity (persistent `trk-` or per-snapshot `dep-`), type, position, radius, confidence, uncertainty, source, status, protected, linked track; `present` and `removed: {reason: full_snapshot / map_reset, t_obs, rx_utc}`; first/last times | `(mission_id, obstacle_id)` unique, `(mission_id, present, type)`, `(mission_id, last_rx_utc)` |
| `telemetry_events` | One document per received packet: type, kind, seq, `t_obs_s`, `rx_utc`, `seq_flag`, provenance (transport, receiver, validation); events (health, map_reset) in full, snapshot metadata (full, part/parts, ids) | `(mission_id, seq)` unique, `(mission_id, kind, rx_utc)`, `(mission_id, t_obs_s)` |
| `link_metrics` | Every 2 s: receiver counters (valid, invalid, gaps, missing, duplicates, reordered, ACKs), packets/s, `last_rx_age_s`, `link_up`, the **emulator's live status** (preset, link state, configured loss, delivery ratio, delay, blackout time, drops), ingestion health | `(mission_id, rx_utc)` |

## 3. REST API v1 (OpenAPI: `http://127.0.0.1:8000/docs`)

**Envelope:** `{"data": ..., "meta": {"api_version": "v1", "request_id", "generated_utc", "pagination": {limit, offset, total, next_offset} | null}}`.

**Errors:** `{"error": {"code", "message", "details"}, "meta": {...}}`. Codes include `mission_not_found` (404), `validation_error` (422), `local_only` / `network_control_disabled` (403), `emulator_unavailable` (503), `emulator_rejected` (400).

| Endpoint | Content |
|---|---|
| `GET /health` (unversioned, also `/api/v1/health`) | `status` ok/degraded, `mongodb` up/down, receiver counters, ingestion stats (queue, drops, DB errors, retries), WebSocket clients |
| `GET /api/v1/missions?status=&limit=&offset=` | Missions, newest first; `status` active/idle computed from UTC freshness |
| `GET /api/v1/missions/{id}` | One mission |
| `GET /api/v1/missions/{id}/overview` | Mission, latest pose, obstacle picture counts (by type and source), latest link metric, health, freshness |
| `GET /api/v1/missions/{id}/positions?since_t_obs_s=&order=&limit=&offset=` | Received poses |
| `GET /api/v1/missions/{id}/obstacles?present=&source=&type=&limit=&offset=` | Reconciled obstacles (default: present only; `present=false`: removed, with reason) |
| `GET /api/v1/missions/{id}/telemetry?packet_type=` | Packet log (validated packets, each stored once) |
| `GET /api/v1/missions/{id}/events?kind=health\|map_reset` | Health transitions and map resets |
| `GET /api/v1/missions/{id}/link` | Link metrics history |
| `GET /api/v1/missions/{id}/stats` | Counters, document counts, present obstacles, latest receiver and emulator stats, ingestion stats |
| `GET /api/v1/network/status` (local only) | The emulator's live status, read through its UDP control port 5792 |
| `POST /api/v1/network/preset {"preset": "NORMAL\|DEGRADED\|SEVERE\|BLACKOUT\|RESTORE"}` (local only) | Applies the preset **on the emulator**; returns its reply; fails if the emulator did not apply it |

Local-only means the client host must be `127.0.0.1` or `::1`, and uvicorn is bound to `127.0.0.1`. Can be disabled with `ARTEMIS_GROUND_NETWORK_CONTROL_ENABLED=false`.

## 4. WebSocket v1: `ws://127.0.0.1:8000/ws/v1/missions/{id}` (`*` = all missions)

**Envelope:** `{"type", "api_version": "v1", "mission_id", "t_obs_s" (simulation s, or null), "rx_utc" (UTC ISO "Z"), "data"}`.

**Reconnect-friendly:** the first message after every (re)connect is `mission.snapshot`: latest pose, present obstacles and the mission document, read from MongoDB (`source: "unavailable"` if the database is down).

**Events:**
- `rover.position`
- `obstacle.detected`, `obstacle.updated` (only when the content changed), `obstacle.removed` (`reason`: full_snapshot or map_reset)
- `map.reset`
- `health.changed`
- `telemetry.received` (every valid packet: seq, type, seq_flag)
- `link.status`, `mission.stats` (every 2 s)

Each client has a bounded queue of 512 events. A slow client loses its oldest events (counted); it never blocks ingestion or other clients. A minimal Next.js client is in `docs/examples/nextjs_ground_client.ts`.

## 5. Running it (five terminals)

```powershell
# 1. MongoDB
docker compose -f docker/compose.yml up -d
# 2. Ground backend (receiver on udp 5790, ACKs to the emulator, REST/WS on :8000). Config: ground_station/.env (template .env.example)
uv sync --project ground_station
uv run --project ground_station uvicorn artemis_ground.main:app --app-dir ground_station --host 127.0.0.1 --port 8000
# 3. Lunar link emulator with the reverse (ACK) path
python scripts/lunar_link.py serve --reverse --log artemis-webots/output/link/link.jsonl
# 4. YOLO inference server
uv run --project backend --no-sync python scripts/inference_server.py --port 5761
# 5. Webots with DTN telemetry through the emulator. artemis_r4_monitor.wbt is the test course WITH the onboard
#    Mission-Control panels (RGB CAMERA / AI DETECTION / DEPTH SENSOR / LOCAL MAP 2D map); artemis_test_course.wbt has
#    no map panel. A closed panel is reopened from Webots: Overlays > Display Devices.
$env:ARTEMIS_TELEMETRY = "1"; $env:ARTEMIS_TELEMETRY_PORT = "5791"; $env:ARTEMIS_TELEMETRY_DTN = "1"
& "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" --mode=realtime artemis-webots\worlds\artemis_r4_monitor.wbt
```

Then open `http://127.0.0.1:8000/docs`, call `GET /api/v1/missions`, or switch the link live: `POST /api/v1/network/preset {"preset": "BLACKOUT"}`.

The backend replaces the standalone `ground_receiver.py` here: both listen on udp 5790, so do not run them together.

**Automated end to end** (MongoDB must be up): `uv run --project backend --no-sync python scripts/run_webots_r4.py ground`, then `uv run --project ground_station python scripts/r5_ground_e2e_analysis.py <run>/ground_e2e`.

## 6. Validation

**Automated tests** (`ground_station/tests/test_ground_backend.py`, real MongoDB, real UDP; skipped, and saying so, when MongoDB is unreachable): **9/9 pass**.

| Test | Checks |
|---|---|
| Indexes, mission, poses, obstacles, REST | Unique indexes exist; mission created from incoming telemetry; poses (newest first, odom, `rx_utc`); all 48 obstacles of a real R4.6 export with their position and uncertainty; pagination and `next_offset`; events, telemetry, overview, stats; 404 and 422 envelopes |
| Duplicates | Same packet ×3 → 1 telemetry document, 1 position, `counters.duplicates = 2` |
| Out of order, full snapshot, map reset | A late older observation does not win; an obstacle missing from a newer complete full snapshot is marked removed (`full_snapshot`); a map reset removes everything (`map_reset`), and a late pre-reset packet is ignored |
| Two missions in a row | Identical obstacle IDs never cross-talk; each mission keeps its own picture |
| Invalid datagrams | CRC error and truncated datagram counted by the receiver, never persisted |
| WebSocket | `mission.snapshot`, then `telemetry.received`, `rover.position`, `obstacle.detected` / `removed`, `health.changed`, `map.reset` with complete envelopes |
| Network control | `POST` SEVERE / BLACKOUT / RESTORE changes the **real emulator's** configuration; status read back; invalid preset 422; remote client 403 `local_only`; emulator absent 503 |
| No database at all | Health `degraded` / `down`; the receiver keeps sending **ACKs**; WebSocket still streams live; `db_errors` and `last_db_error` observable |

**End to end** (`r5-ground-20261010-231241`): Webots → R5.4 DTN sender → emulator → backend receiver → MongoDB + WebSocket.
- **Presets** switched **through the backend REST API** (BLACKOUT at about 15 s, RESTORE at about 27 s wall clock).
- **MongoDB stopped for 10 s** (`docker stop`, then `docker start`) at about 36 s.

| Check | Result |
|---|---|
| Rover tour | 55/55; mapping cycle p95 12.0 ms |
| Rover unique packets → forwarded by the emulator → decoded by the receiver | 1,111 → **1,073** → 1,073 (38 dropped during the blackout, never forwarded) |
| **Persisted in MongoDB** (unique packets) | **1,073**: every forwarded packet; **0 persisted packets that the emulator did not forward**; 0 forwarded-but-not-persisted |
| Database outage | 7 write errors, 4 retries, `mongodb_recovered`, **0 packets lost** (queue never full, `dropped_queue_full = 0`) |
| Poses (800) vs the rover's own odometry at the same `t_obs` | Max 0.5 mm (protocol quantisation) |
| Obstacles in MongoDB vs the rover's final export (deduplicated) | **41/41**, 0.0 mm, 0 stale extras |
| WebSocket (one connection for the whole run) | 3,658 events, all with complete envelopes: 1 `mission.snapshot`, 1,073 `telemetry.received`, 800 `rover.position`, 127 `obstacle.detected`, 1,414 `obstacle.updated`, 86 `obstacle.removed`, 45 `health.changed`, 56 `link.status`, 56 `mission.stats` |
| Network control | BLACKOUT and RESTORE requested via `POST /api/v1/network/preset`, both acknowledged; the emulator log shows the same two configuration changes |
| `link_metrics` documents | 53 |

**Bugs found by the live run and fixed:**
- **Shared ground map.** A single `GroundMap` shared by all sessions mixed missions. Now one map per session, with a regression test.
- **Orphaned backends on Windows.** Terminating `uv run` left an orphaned uvicorn, so a stale backend answered the next run. The runner now kills the process tree and refuses to start if port 8000 already answers.
- **`obstacle.updated` noise.** These events were emitted for identical re-sends (full snapshots). They are now emitted only when the content changes: 2,350 → 1,414.

## 7. Known limitations

- **Single writer.** Simple and ordered, but throughput is bounded by sequential round trips to MongoDB. That is ample for this rover (about 20 packets/s) but not tuned for many rovers.
- **Bounded memory, not persisted.** The ingestion queue (20,000 items) and the per-session `GroundMap`s (16) live in memory. A backend crash loses what is queued; the rover's R5.4 queue does not help, because ACKs are sent at reception.
- **ACKs mean received, not persisted.** The ground ACKs at **reception** (R5.4 semantics), not after the MongoDB write. During a database outage, data is safe only in the backend's memory queue.
- **Local only, no authentication.** Network control is restricted to local clients and there is no authentication. Do not expose the API beyond localhost.
- **Two missions after a backend restart.** A backend restarted mid-mission rebuilds its `GroundMap` for that session from the next packets. Reconciliation resumes fully only with the next complete full snapshot (about 10 s).
- **Not an R4 change.** The `obstacle.updated` volume reflects real tracker refinements (confidence EMA, refinement); throttling is a dashboard (R6) concern.

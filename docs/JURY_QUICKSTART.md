# Artemis-X — jury quickstart

This is a simulator-only R6/R6.1 prototype, not autonomous obstacle avoidance. The frozen model and scientific evidence
are tracked. Webots/Edge/Docker and downloaded CC0 scene assets are external prerequisites. Training data is **not published**
and is not required for this demo. See [final audit](FINAL_REPOSITORY_AUDIT.md) for freshly measured outcomes and open decisions.

## 1. Prepare a new Windows checkout

Install Webots **R2025a**, Python **3.12** via uv, Node **≥24**, pnpm **11.25.0**, Microsoft Edge and Docker Desktop
with Compose v2. Start Docker Desktop yourself. A CUDA-compatible GPU is recommended for real-time YOLO; the locked ML
environment installs CUDA 13 wheels and needs several GB. The Windows orchestration is not a Linux validation claim.

```powershell
git clone https://github.com/mehdi-lakhzouri/ATEC.git
cd ATEC
uv sync --project backend --locked
uv sync --project ground_station --locked --group dev
Push-Location frontend
pnpm install --frozen-lockfile
Pop-Location
uv run --project backend --no-sync python artemis-webots/tools/fetch_polyhaven.py --locked
uv run --project backend --no-sync python scripts/verify_artifacts.py
```

`--locked` downloads the **committed URLs**, verifies SHA256, converts meshes and verifies derived textures without changing
`ASSETS.json`. A checksum mismatch is a failure, not permission to replace the manifest. The original dynamic fetch mode
is retained for historical asset-maintenance workflows. No new weights are trained and no held-out test evaluation is run.

On a **new** checkout only, copy the templates:

```powershell
Copy-Item -LiteralPath docker/.env.example -Destination docker/.env
Copy-Item -LiteralPath ground_station/.env.example -Destination ground_station/.env
Copy-Item -LiteralPath frontend/apps/web/.env.example -Destination frontend/apps/web/.env.local
```

Edit the private Docker username/password and the matching ground `ARTEMIS_GROUND_MONGO_URI`; replace both example
passwords with the **same** secret. URI credentials must be percent-encoded when necessary. Database name defaults to
`artemis_mission`. Do not overwrite an existing configuration: Mongo's named volume keeps its original authentication.
Never commit `.env` files. `NEXT_PUBLIC_GROUND_API=http://127.0.0.1:8000` and
`NEXT_PUBLIC_GROUND_WS=ws://127.0.0.1:8000` are the frontend defaults and are inlined by Next.js builds.

For nonstandard installations set `WEBOTS_HOME` to the Webots installation directory and `ARTEMIS_EDGE_EXE` to Edge's
executable. Paths to the repository are calculated from script locations, not from the developer's username.

## 2. Verify the environment and tests

```powershell
.\scripts\jury.ps1 -Step preflight
docker compose -f docker/compose.yml up -d mongodb
docker compose -f docker/compose.yml ps mongodb
.\scripts\jury.ps1 -Step unit
.\scripts\jury.ps1 -Step test
```

Wait for Mongo **healthy** before `test` or `demo`. Preflight is read-only; missing Mongo is a WARN because `start` can
start it. Occupied ports are also reported, but actual launch refuses conflicting services. Port presence alone is not
service readiness. Example credentials, missing weights/assets/dependencies or Docker failure are actionable FAILs.

`unit` executes runtime/model/notebook and historical research regressions plus frontend tests/lint/type/build. It needs
no external server but does load the tracked YOLO model. It explicitly labels Mongo/Webots/browser integrations **NOT RUN**.
Notebook tooling comes from hash-pinned `tests/requirements.txt` overlaid by `uv run --with-requirements`; protected
`backend/pyproject.toml` and `backend/uv.lock` are unchanged.

`test` additionally enables the R1/R2 Webots smoke, runs the real-Mongo ground suite and starts the actual R6 browser/rover
scenario, waiting for its **55/55** course summary. Mongo skips cause a nonzero exit; optional historical dataset/checkpoint
skips are listed by test ID/reason, not counted as passed. Every command has a log and exit code under `.jury/runs/test-*`;
the linked demo has separate `.jury/runs/demo-*` evidence. A full run takes approximately 10–15 minutes on the tested GPU,
not a guaranteed duration on another machine. Keep the interactive stack stopped while running integrations.

## 3. Interactive complete platform

```powershell
.\scripts\jury.ps1 -Step start
.\scripts\jury.ps1 -Step status
```

The owned background worker starts/reuses Mongo, then ground, emulator **with reverse ACK**, inference, dashboard and
Webots. It returns success only after authenticated Mongo/ground/emulator readiness and **real received rover packets**.
Webots runs the existing monitor course immediately. Browse `http://127.0.0.1:3000`; API health is
`http://127.0.0.1:8000/api/v1/health`, OpenAPI is `http://127.0.0.1:8000/docs`.

| Service | Local endpoint |
|---|---|
| Dashboard / FastAPI / Mongo | TCP 3000 / 8000 / 27017 |
| Frozen inference | TCP 5761 (R6 automated validation uses 5771) |
| Rover → emulator → ground | UDP 5791 → 5790 |
| ACK ground → emulator → rover | UDP 5794 → 5793 |
| Emulator control | UDP 5792 |
| R6 headless Edge debug | TCP 9333 |

There must be **one receiver on 5790**. Do not launch `ground_receiver.py` alongside FastAPI's receiver.
If the existing Artemis dashboard alone occupies 3000, explicitly use `-ReuseDashboard` with `start`, `test` or `demo`;
its `/settings` response is checked and the reused process is never stopped. Other occupied ports are not automatically
killed. The newly launched dashboard binds localhost; a reused older server's bind address remains its operator's responsibility.

## 4. Automated jury demonstration

```powershell
.\scripts\jury.ps1 -Step stop
.\scripts\jury.ps1 -Step demo
```

The demo starts owned services and a fresh retained `artemis_jury_*` Mongo database. It first creates a short historical
session using **real Webots telemetry**, then starts a new full course. Edge clicks the application's real controls:

1. Backend unavailable: explicit error / **DISCONNECTED**, no fabricated values.
2. Mongo history and live rover: received odometry, genuine YOLO+depth obstacles with spatial coordinates; map zoom/follow.
3. **NORMAL → DEGRADED → SEVERE**: real emulator confirmation, configured impairments and increasing packet counters.
4. **BLACKOUT**: no new poses, increasing UTC reception age and **STALE** while the ground WebSocket remains open.
5. **RESTORE**: newly received telemetry, retained DTN updates and **LIVE** again.
6. Owned API shutdown/restart: **DISCONNECTED**, retained picture, reconnect/resync with no duplicate obstacle IDs.
7. Historical mission isolation, fresh browser reload, dedicated pages, clean/unified setting persistence, unknown mission 404.
8. Wait for the full **55/55** Webots tour and successful frozen inference; preserve evidence before owned cleanup.

Look at `r6_dashboard_e2e.json`, `screens/`, `link.jsonl`, `tour/summary.json`, raw inference/mapping logs and service logs
in the printed demo directory. The historical seed is intentionally interrupted and is **not** represented as a complete tour.
Browser preference setup selects a UI mode only; no telemetry is injected and no API response is mocked.

Distinguish **STALE** (radio data stopped; ground connection still open) from **DISCONNECTED** (API/WS transport stopped).
Simulator timestamps and UTC reception timestamps are different clocks; never subtract them to claim one-way latency.
DTN retries are bounded and low-priority poses are coalesced; recovery is not a zero-loss/all-poses replay guarantee.
Receiver ACK is not proof of Mongo durability, particularly during DB failure.

For the separate video project: `cd demo`, `npm ci`, then follow [capture/render/QA instructions](../demo/README.md).
Generated film and raw media are not included in Git; existing video QA is historical evidence, not this audit's new E2E result.

## 5. Stop safely

```powershell
.\scripts\jury.ps1 -Step stop
```

Only recorded owned PID **and creation-time identities** can be terminated. The worker normally handles a stop request and
its own process handles; exact identities provide crash recovery. Existing dashboard/Mongo services are preserved. A Mongo
container started by this workflow is stopped by recorded container ID; its named volume is never deleted. `stop` does not
stop independently launched manual/demo processes: the standalone demo cleans its own children in `finally`.
Do not use `taskkill /IM python.exe`, port-based mass kills or `docker compose down -v`.

## Troubleshooting / boundaries

- Missing or changed assets: rerun the locked fetch; do not silently replace committed hashes.
- Mongo authentication mismatch: check private credentials and retained volume initialization; no auto-reset/destructive fix.
- Startup failed: inspect `.jury/active.json` and the printed worker/service logs, then run `stop` before retrying.
- Port conflict: identify the listener, close it yourself, or explicitly reuse the verified dashboard. No duplicate UDP listener.
- Historical parity skipped: old datasets/checkpoints are not all shipped; [distribution plan](DATASET_DISTRIBUTION_PLAN.md)
  remains blocked by owner/licensing decisions. Runtime evidence is reproducible without those pixels.
- Motion remains scripted; health STOP is advisory, the ground UI has no video/occupancy-grid downlink, synthetic labels are
  provisional, real hardware has not been validated, and the project license is still an owner decision.

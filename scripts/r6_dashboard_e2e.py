"""R6 end-to-end validation of the Mission Control dashboard against the REAL stack:

    MongoDB (docker) -> ground backend (FastAPI) <- lunar link emulator <- Webots rover (YOLO server, DTN telemetry)
                                    |
                                    +-> Next.js dashboard (already running: pnpm --filter web dev) <- headless Edge (CDP)

The browser is driven through the Chrome DevTools Protocol: the checks click the dashboard's own controls (preset
buttons, mission history) and read what the dashboard renders. Nothing is injected into the dashboard.

    uv run --project ground_station --no-sync python scripts/r6_dashboard_e2e.py --out artemis-webots/output/r6_dashboard

Steps: backend down (API error state) -> backend up (history from MongoDB) -> Webots live (pose + obstacles) ->
BLACKOUT from the UI (STALE, no invented poses) -> RESTORE (LIVE, DTN backfill) -> backend restart (WebSocket reconnect,
no duplicates) -> mission switching (no mixing) -> dashboard restart (persistence) -> unknown mission (404 shown).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8000"
DASH = os.environ.get("ARTEMIS_DASHBOARD_URL", "http://localhost:3000/")
EDGE = Path(os.environ.get("ARTEMIS_EDGE_EXE", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"))
WEBOTS = Path(os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")) / "msys64" / "mingw64" / "bin" / "webots.exe"
# the test course WITH the R4.5 Mission-Control panels (RGB / AI DETECTION / DEPTH / LOCAL MAP): same course, + displays
WORLD = ROOT / "artemis-webots" / "worlds" / "artemis_r4_monitor.wbt"
CDP_PORT = 9333
INFER_PORT = 5771


def clean_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    venv = str(Path(sys.executable).parent).lower().rstrip("\\/")
    env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep) if p.lower().rstrip("\\/") != venv)
    for key, value in (extra or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    return env


def kill_tree(p: subprocess.Popen | None) -> None:
    if p is not None and p.poll() is None:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            pass


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def get(path: str, timeout: float = 3.0):
    with urllib.request.urlopen(API + path, timeout=timeout) as r:
        return json.loads(r.read())


class Stack:
    def __init__(self, out: Path):
        self.out = out
        self.backend = self.emulator = self.infer = self.webots = None
        self.tour_name = "tour"

    def start_backend(self) -> None:
        if port_open(8000):
            raise RuntimeError("something already answers on 127.0.0.1:8000 - stop the old ground backend first")
        log = open(self.out / f"backend_{int(time.time())}.log", "w", encoding="utf-8")
        self.backend = subprocess.Popen(
            ["uv", "run", "--project", str(ROOT / "ground_station"), "--no-sync", "uvicorn", "artemis_ground.main:app",
             "--app-dir", str(ROOT / "ground_station"), "--host", "127.0.0.1", "--port", "8000"],
            env=clean_env({"ARTEMIS_GROUND_RECEIVER_ENABLED": "true", "ARTEMIS_GROUND_UDP_HOST": "127.0.0.1",
                           "ARTEMIS_GROUND_UDP_PORT": "5790", "ARTEMIS_GROUND_ACK_TO": "127.0.0.1:5794",
                           "ARTEMIS_GROUND_NETWORK_CONTROL_ENABLED": "true",
                           "ARTEMIS_GROUND_EMULATOR_CONTROL": "127.0.0.1:5792", "ARTEMIS_GROUND_LINK_METRICS_PERIOD_S": "1.0"}),
            stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
        end = time.time() + 90
        while time.time() < end:
            try:
                h = get("/api/v1/health")["data"]
                if h["mongodb"] == "up" and h["receiver"]["enabled"]:
                    return
            except OSError:
                pass
            time.sleep(0.5)
        raise RuntimeError("ground backend did not become healthy")

    def stop_backend(self) -> None:
        kill_tree(self.backend)
        self.backend = None
        end = time.time() + 15
        while port_open(8000) and time.time() < end:
            time.sleep(0.3)

    def start_emulator(self) -> None:
        log = open(self.out / "emulator.log", "w", encoding="utf-8")
        self.emulator = subprocess.Popen(
            ["uv", "run", "--project", str(ROOT / "backend"), "--no-sync", "python", str(ROOT / "scripts" / "lunar_link.py"),
             "serve", "--reverse", "--log", str(self.out / "link.jsonl")],
            env=clean_env(), stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
        end = time.time() + 15
        while time.time() < end:
            if self.emulator.poll() is not None:
                raise RuntimeError("emulator exited during startup")
            try:
                if get("/api/v1/network/status")["data"]["preset"] == "NORMAL":
                    return
            except OSError:
                pass
            time.sleep(0.3)
        raise RuntimeError("emulator control did not become ready")

    def start_inference(self) -> None:
        log = open(self.out / "inference.log", "w", encoding="utf-8")
        self.infer = subprocess.Popen(
            ["uv", "run", "--project", str(ROOT / "backend"), "--no-sync", "python", str(ROOT / "scripts" / "inference_server.py"),
             "--port", str(INFER_PORT)], env=clean_env(), stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
        end = time.time() + 180
        while time.time() < end and not port_open(INFER_PORT):
            if self.infer.poll() is not None:
                raise RuntimeError("inference server exited during start-up")
            time.sleep(1.0)
        if not port_open(INFER_PORT):
            raise RuntimeError("inference server did not become ready")

    def start_webots(self) -> None:
        log = open(self.out / f"{self.tour_name}_webots.log", "w", encoding="utf-8")
        env = clean_env({"ARTEMIS_TELEMETRY": "1", "ARTEMIS_TELEMETRY_PORT": "5791", "ARTEMIS_TELEMETRY_DTN": "1",
                         "ARTEMIS_MODE": "demo", "ARTEMIS_INFERENCE": "1", "ARTEMIS_MAPPING": "1",
                         "ARTEMIS_TELEMETRY_HOST": "127.0.0.1", "ARTEMIS_WEBOTS_QUIT": "1",
                         "ARTEMIS_TELEMETRY_SESSION": None, "ARTEMIS_FAULTS": None, "ARTEMIS_PROFILE": None,
                         "ARTEMIS_MEMTRACE": None, "ARTEMIS_MAP_RESET_AT": None,
                         "ARTEMIS_DTN_MAX_ENTRIES": None, "ARTEMIS_DTN_MAX_BYTES": None,
                         "ARTEMIS_TELEMETRY_ACK_PORT": "5793", "ARTEMIS_INFER_PORT": str(INFER_PORT),
                         "ARTEMIS_WEBOTS_OUT": str(self.out / self.tour_name),
                         # the onboard 2D map panel keeps working with telemetry on: panels saved mid-mission as evidence
                         "ARTEMIS_MONITOR_SAVE": str(self.out / "monitor_panels"), "ARTEMIS_MONITOR_SAVE_AT": "20,45"})
        self.webots = subprocess.Popen([str(WEBOTS), "--batch", "--minimize", "--mode=realtime", "--stdout", "--stderr",
                                        "--port=1244", str(WORLD)], env=env, stdout=log, stderr=subprocess.STDOUT)

    def stop_all(self) -> None:
        for p in (self.webots, self.infer, self.emulator):
            kill_tree(p)
        self.stop_backend()


class Browser:
    """Minimal CDP client for one headless Edge page."""

    _instances = 0

    def __init__(self, out: Path, workspace: str | None = "unified"):
        """workspace: display mode stored before every page load ("unified" = the R6 command center the original
        checks drive); None = a first visit with no stored preference (the default clean workspace)."""
        self.out = out
        self.workspace = workspace
        Browser._instances += 1
        self.instance = Browser._instances
        self.proc = None
        self.ws = None
        self.msg_id = 0
        self.pending: dict[int, asyncio.Future] = {}
        self.console: list[dict] = []
        self.reader = None

    async def start(self) -> None:
        # absolute, outside the repo, fresh per browser session (no localStorage carried between sessions)
        profile = Path(os.environ.get("TEMP", str(self.out))) / f"artemis-r6-edge-{os.getpid()}-{self.instance}"
        self.proc = subprocess.Popen([str(EDGE), "--headless=new", f"--remote-debugging-port={CDP_PORT}",
                                      f"--user-data-dir={profile}", "--window-size=1920,1080", "--no-first-run",
                                      "--disable-extensions", "--hide-scrollbars", "about:blank"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        end = time.time() + 30
        target = None
        while time.time() < end and target is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json/list", timeout=2) as r:
                    target = next((t for t in json.loads(r.read()) if t["type"] == "page"), None)
            except OSError:
                await asyncio.sleep(0.5)
        if target is None:
            raise RuntimeError("Edge DevTools endpoint not available")
        self.ws = await websockets.connect(target["webSocketDebuggerUrl"], max_size=64 * 2**20)
        self.reader = asyncio.create_task(self._read())
        await self.call("Runtime.enable")
        await self.call("Page.enable")
        if self.workspace:
            await self.call("Page.addScriptToEvaluateOnNewDocument",
                            source=f"try {{ localStorage.setItem('artemis.workspace', '{self.workspace}') }} catch (e) {{}}")
        await self.call("Emulation.setDeviceMetricsOverride", width=1920, height=1080, deviceScaleFactor=1, mobile=False)

    async def _read(self) -> None:
        async for raw in self.ws:
            msg = json.loads(raw)
            if "id" in msg and msg["id"] in self.pending:
                self.pending.pop(msg["id"]).set_result(msg)
            elif msg.get("method") == "Runtime.exceptionThrown":
                self.console.append({"kind": "exception", "text": msg["params"]["exceptionDetails"].get("text"),
                                     "detail": str(msg["params"]["exceptionDetails"].get("exception", {}).get("description", ""))[:300]})
            elif msg.get("method") == "Runtime.consoleAPICalled" and msg["params"]["type"] in ("error", "warning"):
                self.console.append({"kind": msg["params"]["type"],
                                     "text": " ".join(str(a.get("value", a.get("description", "")))[:200] for a in msg["params"]["args"])})

    async def call(self, method: str, **params):
        self.msg_id += 1
        fut = asyncio.get_running_loop().create_future()
        self.pending[self.msg_id] = fut
        await self.ws.send(json.dumps({"id": self.msg_id, "method": method, "params": params}))
        res = await asyncio.wait_for(fut, 60)
        if "error" in res:
            raise RuntimeError(f"{method}: {res['error']}")
        return res["result"]

    async def goto(self, url: str) -> None:
        await self.call("Page.navigate", url=url)
        await asyncio.sleep(3.0)

    async def js(self, expr: str):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"js error: {r['exceptionDetails']}")
        return r["result"].get("value")

    async def shot(self, name: str) -> str:
        r = await self.call("Page.captureScreenshot", format="png")
        p = self.out / "screens" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(base64.b64decode(r["data"]))
        return p.relative_to(ROOT).as_posix()

    async def close(self) -> None:
        try:
            if self.ws:
                await self.ws.close()
        finally:
            kill_tree(self.proc)


DOM = r"""(() => {
  const t = (s) => document.querySelector(s)?.textContent?.trim() ?? null;
  const all = (s) => [...document.querySelectorAll(s)];
  const map = document.querySelector('#surface .lunar-map svg, [data-page=map] .lunar-map svg')?.getAttribute('aria-label') ?? null;
  const m = map && map.match(/(\d+) received poses, (\d+) obstacles/);
  return {
    fresh: t('.mission-header .freshness-pill'),
    mission: document.querySelector('.context-day p')?.getAttribute('title') ?? null,
    poses: m ? Number(m[1]) : null,
    mapObstacles: m ? Number(m[2]) : null,
    obstacleIds: all('.obstacle-row:not(.is-head) > span:first-child').map(e => e.textContent),
    staleBanner: t('.stale-banner'),
    feedback: t('.network-feedback'),
    network: document.querySelector('.network-stats')?.innerText ?? null,
    status: all('.status-grid > div').map(d => d.innerText.replace(/\n/g, ': ')),
    errors: all('.api-error').map(e => e.innerText),
    history: all('.history-row').map(e => e.innerText.replace(/\n/g, ' ')),
    following: t('.follow-latest'),
    log: all('.log-row').slice(0, 10).map(e => e.innerText.replace(/\n/g, ' ')),
    telemetry: all('.telemetry-tile').map(e => e.innerText.replace(/\n/g, ' ')),
    empty: all('.empty-state strong').map(e => e.textContent),
    page: document.querySelector('[data-page]')?.getAttribute('data-page') ?? null,
    unified: !!document.querySelector('.dashboard-layout'),
    stored: (() => { try { return localStorage.getItem('artemis.workspace'); } catch (e) { return 'blocked'; } })(),
    chip: t('.workspace-chip'),
    path: location.pathname,
    href: location.href,
    activeNav: t('.navigation-item.is-active span'),
    rail: t('.cw-rail-mid'),
    kpis: all('[data-page] .cw-kpi').length,
    presets: all('.network-preset-grid button').length,
    presetFeedback: t('.network-control-feedback'),
    cleanStale: t('.cw-stale'),
    detectionRows: all('[data-page=detection] tbody tr').length,
    missionRows: all('[data-page=missions] tbody tr').length,
    feed: all('[data-page=logs] .logs-interactive-timeline li').length,
    logRows: all('[data-page=logs] tbody tr').length,
    switch: (() => { const s = document.querySelector('[data-setting=unified-mode] [role=switch]');
      return s ? s.getAttribute('aria-checked') === 'true' : null; })(),
  };
})()"""


async def wait_for(b: Browser, pred, timeout: float, what: str, every: float = 0.5) -> dict:
    end = time.time() + timeout
    last = None
    while time.time() < end:
        last = await b.js(DOM)
        try:
            if pred(last):
                return last
        except (TypeError, KeyError):
            pass
        await asyncio.sleep(every)
    raise TimeoutError(f"timeout waiting for {what}; last DOM: {json.dumps(last)[:1500]}")


async def click(b: Browser, selector: str) -> None:
    ok = await b.js(f"(() => {{ const e = document.querySelector({json.dumps(selector)}); if (!e) return false; e.click(); return true; }})()")
    if not ok:
        raise RuntimeError(f"no element {selector}")


async def main(out: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("evidence directory is not empty; choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    stack, b = Stack(out), Browser(out)
    res: dict = {"dashboard": DASH, "started_utc": datetime.now(timezone.utc).isoformat(), "steps": {}, "screens": []}
    step = res["steps"]

    def record(name: str, ok: bool, **kw) -> None:
        step[name] = {"ok": ok, **kw}
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(kw)[:400]}", flush=True)

    try:
        if port_open(8000):
            raise RuntimeError("a ground backend is already running on :8000; stop it first")
        if port_open(INFER_PORT) or port_open(CDP_PORT):
            raise RuntimeError("inference / browser debug port already occupied")
        # Binding each port detects conflicts without stopping an existing service.
        for port in (5790, 5791, 5792, 5793, 5794):
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.bind(("127.0.0.1", port))
        processes = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                                   "@(Get-Process webots,webots-bin -ErrorAction SilentlyContinue).Count"],
                                  capture_output=True, text=True, check=True)
        if int(processes.stdout.strip()):
            raise RuntimeError("Webots already running; refusing to replace it")
        await b.start()

        # 1. backend down: explicit API error, no data, DISCONNECTED/NO DATA, nothing fabricated
        await b.goto(DASH)
        d = await wait_for(b, lambda d: d["errors"], 20, "API error state")
        res["screens"].append(await b.shot("00_backend_down.png"))
        record("api_error_backend_down", any("unreachable" in e for e in d["errors"]) and d["poses"] is None,
               errors=d["errors"], fresh=d["fresh"])

        # 2. backend up: mission history from MongoDB (persisted earlier missions)
        stack.start_backend()
        stack.start_emulator()
        stack.start_inference()
        # A fresh MongoDB must work too: create the historical session with REAL
        # Webots packets, not a synthetic seed or a developer's pre-existing DB.
        if not get("/api/v1/missions?limit=1")["data"]:
            stack.tour_name = "history_seed"
            stack.start_webots()
            await wait_for(b, lambda d: d["fresh"] == "Live" and (d["poses"] or 0) >= 5, 240, "history seed telemetry")
            kill_tree(stack.webots)
            stack.webots = None
            stack.tour_name = "tour"
            await asyncio.sleep(3)
        missions_before = get("/api/v1/missions?limit=200")["meta"]["pagination"]["total"]
        d = await wait_for(b, lambda d: d["history"] and not d["errors"], 30, "mission history")
        res["screens"].append(await b.shot("01_history_from_mongodb.png"))
        record("history_from_mongodb", len(d["history"]) > 0, missions_in_db=missions_before, rows=d["history"][:5], fresh=d["fresh"])

        # 3. live mission: Webots -> emulator -> backend -> dashboard
        stack.start_webots()
        t0 = time.time()
        d = await wait_for(b, lambda d: d["fresh"] == "Live" and (d["poses"] or 0) >= 5, 240, "LIVE with poses")
        live_mission = d["mission"]
        res["screens"].append(await b.shot("02_live_mission.png"))
        rest_m = get(f"/api/v1/missions/{live_mission}")["data"]
        record("live_mission_followed", rest_m["status"] == "active", mission=live_mission, poses=d["poses"],
               seconds_to_live=round(time.time() - t0, 1), telemetry=d["telemetry"])

        d = await wait_for(b, lambda d: len(d["obstacleIds"]) > 0, 300, "first obstacle")
        await asyncio.sleep(2.0)
        d = await b.js(DOM)
        rest_obs = get(f"/api/v1/missions/{live_mission}/obstacles?present=true&limit=1000")
        ids_rest = sorted(o["obstacle_id"] for o in rest_obs["data"])
        res["screens"].append(await b.shot("03_obstacle_detected.png"))
        common = [o for o in rest_obs["data"] if o["obstacle_id"] in d["obstacleIds"]]
        record("obstacle_detected_flows_to_dashboard",
               len(d["obstacleIds"]) == len(set(d["obstacleIds"])) and bool(common)
               and any(o.get("source") == "yolo+depth" and all(isinstance(o.get(k), (int, float)) for k in ("x_m", "y_m")) for o in common),
               dashboard_ids=sorted(d["obstacleIds"]), rest_ids=ids_rest, spatial_records=common[:10], log=d["log"][:4])

        # 3b. live map interaction: zoom in narrows the view, follow centres on the rover
        vb = "document.querySelector('#surface .lunar-map svg').getAttribute('viewBox')"
        w0 = float((await b.js(vb)).split()[2])
        await click(b, '#surface .map-zoom button[aria-label="Zoom in"]')
        await asyncio.sleep(0.5)
        w1 = float((await b.js(vb)).split()[2])
        await click(b, '#surface .map-zoom button[title="Follow the rover"]')
        await asyncio.sleep(1.0)
        res["screens"].append(await b.shot("03b_map_follow.png"))
        await click(b, '#surface .map-zoom button[title="Fit the whole mission"]')
        record("map_zoom_follow", w1 < w0, view_width_m=[round(w0, 2), round(w1, 2)])

        # Actual emulator impairments, driven by the application's controls.
        for preset, loss in (("NORMAL", 0.0), ("DEGRADED", 0.3), ("SEVERE", 0.5)):
            await click(b, f'.preset-buttons button[title*="{preset}"]')
            await wait_for(b, lambda d, p=preset: p + " confirmed" in (d["feedback"] or ""), 15, preset + " confirmation")
            before_net = get("/api/v1/network/status")["data"]
            await asyncio.sleep(8)
            after_net = get("/api/v1/network/status")["data"]
            record("preset_" + preset.lower(), after_net["preset"] == preset
                   and after_net["received"] > before_net["received"] and after_net["configured_loss"] == loss,
                   preset=after_net["preset"], expected_loss=loss, status=after_net)

        # 4. BLACKOUT from the dashboard's own control; confirmed by GET status; freshness ages; no invented poses
        await click(b, '.preset-buttons button[title*="BLACKOUT"]')
        d = await wait_for(b, lambda d: "BLACKOUT confirmed" in (d["feedback"] or ""), 15, "BLACKOUT confirmation")
        net = get("/api/v1/network/status")["data"]
        d = await wait_for(b, lambda d: d["fresh"] == "Stale", 20, "STALE during blackout")
        poses_a = d["poses"]
        await asyncio.sleep(6.0)
        d = await b.js(DOM)
        res["screens"].append(await b.shot("04_blackout_stale.png"))
        record("blackout_ages_freshness", net["preset"] == "BLACKOUT" and d["fresh"] == "Stale" and d["staleBanner"] is not None
               and d["poses"] == poses_a, emulator_preset=net["preset"], link_state=net["link_state"], banner=d["staleBanner"],
               poses_frozen=[poses_a, d["poses"]], network=d["network"])
        blackout_rest_poses = get(f"/api/v1/missions/{live_mission}/positions?limit=1")["meta"]["pagination"]["total"]

        # 5. RESTORE: LIVE again; newer pose / retained obstacle updates arrive.
        # Low-priority poses are coalesced; this is NOT a complete pose replay.
        await click(b, '.preset-buttons button[title*="RESTORE"]')
        d = await wait_for(b, lambda d: "RESTORE confirmed" in (d["feedback"] or ""), 15, "RESTORE confirmation")
        d = await wait_for(b, lambda d: d["fresh"] == "Live", 30, "LIVE after restore")
        await asyncio.sleep(8.0)
        d = await b.js(DOM)
        res["screens"].append(await b.shot("05_restored_live.png"))
        after_rest = get(f"/api/v1/missions/{live_mission}/positions?limit=1")["meta"]["pagination"]["total"]
        record("restore_resumes", d["fresh"] == "Live" and d["poses"] > poses_a, poses=[poses_a, d["poses"]],
               rest_positions=[blackout_rest_poses, after_rest], network=d["network"], log=d["log"][:6])

        # 6. backend restart: WebSocket reconnect + resync, no duplicate obstacles, trajectory unique
        stack.stop_backend()
        d = await wait_for(b, lambda d: d["fresh"] == "Disconnected", 20, "DISCONNECTED")
        res["screens"].append(await b.shot("06_backend_down_disconnected.png"))
        poses_down = d["poses"]
        record("ws_disconnect_shown", d["staleBanner"] is not None and d["poses"] is not None,
               banner=d["staleBanner"], status=d["status"])
        stack.start_backend()
        d = await wait_for(b, lambda d: d["fresh"] == "Live", 60, "LIVE after backend restart")
        await asyncio.sleep(5.0)
        d = await b.js(DOM)
        rest_total = get(f"/api/v1/missions/{live_mission}/positions?limit=1")["meta"]["pagination"]["total"]
        res["screens"].append(await b.shot("07_reconnected.png"))
        record("ws_reconnect_resync", d["fresh"] == "Live" and len(d["obstacleIds"]) == len(set(d["obstacleIds"]))
               and d["poses"] >= poses_down and d["mission"] == live_mission,
               poses=[poses_down, d["poses"]], rest_positions_total=rest_total,
               reconnect_status=[s for s in d["status"] if "stream" in s.lower()])

        # 7. mission switching: an older mission is shown alone (its own poses, frozen while the live one flows)
        missions = get("/api/v1/missions?limit=20")["data"]
        other = next((m for m in missions[:10] if m["mission_id"] != live_mission and m["counters"].get("pose", 0) > 0), None)
        if other:
            ok = await b.js("(() => { const r = [...document.querySelectorAll('.history-row')].find(e => "
                            f"e.querySelector('.history-id')?.textContent === '#{other['mission_id']}'); r?.click(); return !!r; }})()")
            if not ok:
                raise RuntimeError(f"mission {other['mission_id']} not listed in the history panel")
            d = await wait_for(b, lambda d: d["mission"] == other["mission_id"] and d["poses"] is not None, 30, "switched mission")
            await asyncio.sleep(4.0)
            d1 = await b.js(DOM)
            await asyncio.sleep(4.0)
            d2 = await b.js(DOM)
            rest_poses = get(f"/api/v1/missions/{other['mission_id']}/positions?limit=1")["meta"]["pagination"]["total"]
            rest_obs = get(f"/api/v1/missions/{other['mission_id']}/obstacles?present=true&limit=1000")["meta"]["pagination"]["total"]
            res["screens"].append(await b.shot("08_mission_switch.png"))
            record("mission_switch_isolated", d1["mission"] == other["mission_id"] and d1["poses"] == d2["poses"]
                   and d2["poses"] == min(rest_poses, 5000) and len(d2["obstacleIds"]) == rest_obs,
                   mission=other["mission_id"], poses=[d1["poses"], d2["poses"]], rest_poses=rest_poses,
                   obstacles=[len(d2["obstacleIds"]), rest_obs], fresh=d2["fresh"])
            await click(b, ".follow-latest")
            d = await wait_for(b, lambda d: d["mission"] == live_mission and d["poses"] is not None,
                               30, "back to the live mission with hydrated map")
        else:
            record("mission_switch_isolated", False, reason="no other mission with poses in MongoDB")

        # 8. dashboard restart: a fresh browser session reloads the mission from the ground station
        before = await wait_for(b, lambda d: d["mission"] == live_mission and d["poses"] is not None,
                                30, "hydrated live map before browser restart")
        await b.close()
        b = Browser(out)
        await b.start()
        await b.goto(f"{DASH}?mission={live_mission}")
        d = await wait_for(b, lambda d: d["mission"] == live_mission and (d["poses"] or 0) >= before["poses"], 40, "reload after restart")
        res["screens"].append(await b.shot("09_dashboard_restart.png"))
        record("persistence_after_dashboard_restart", True, poses=[before["poses"], d["poses"]],
               obstacles=[len(before["obstacleIds"]), len(d["obstacleIds"])], history_rows=len(d["history"]))

        # 9. CLEAN WORKSPACE (R6.1): a first visit with no stored preference, live mission still running
        await b.close()
        b = Browser(out, workspace=None)
        await b.start()
        await b.goto(DASH)
        d = await wait_for(b, lambda d: d["page"] == "overview" and d["mission"] == live_mission, 60, "clean overview")
        res["screens"].append(await b.shot("11_clean_overview.png"))
        record("clean_first_visit_default", d["unified"] is False and d["stored"] is None and d["chip"] == "Clean workspace",
               page=d["page"], stored=d["stored"], chip=d["chip"], rail=d["rail"])

        # each sidebar item opens its own page with its own content
        pages = [("map", "Mapping", "12_clean_map.png", lambda d: d["poses"] is not None),
                 ("detection", "Detection", "13_clean_detection.png", lambda d: d["detectionRows"] > 0),
                 ("telemetry", "Telemetry", "14_clean_telemetry.png", lambda d: d["kpis"] >= 4),
                 ("network", "Communication", "15_clean_network.png", lambda d: d["presets"] == 5),
                 ("missions", "Missions", "16_clean_missions.png", lambda d: d["missionRows"] > 1),
                 ("logs", "Logs", "17_clean_logs.png", lambda d: d["logRows"] > 0),
                 ("settings", "Settings", "18_clean_settings.png", lambda d: d["switch"] is not None)]
        seen = {}
        for route, label, shot, ok in pages:
            await click(b, f'a[href^="/{route}"]' if route == "missions" else f'a.navigation-item[href^="/{route}"]')
            d = await wait_for(b, lambda d, r=route, f=ok: d["page"] == r and f(d), 60, f"clean page {route}")
            await asyncio.sleep(1.0)
            res["screens"].append(await b.shot(shot))
            seen[route] = {"path": d["path"], "active": d["activeNav"], "mission": d["mission"]}
        record("clean_pages_own_content", all((v["active"] == lbl or r == "missions") and v["mission"] == live_mission and v["path"] == f"/{r}"
                                              for (r, lbl, *_), v in zip(pages, seen.values())), pages=seen)

        # The new logs page defaults to its paginated shadcn ledger, not the old
        # timeline. Verify both real views through the visible tab control.
        await click(b, 'a.navigation-item[href^="/logs"]')
        ledger = await wait_for(b, lambda d: d["page"] == "logs" and d["logRows"] > 0, 30, "event ledger")
        ok = await b.js("(() => { const e = [...document.querySelectorAll('[role=tab]')].find(e => e.textContent.includes('Timeline')); e?.click(); return !!e; })()")
        if not ok:
            raise RuntimeError("event timeline tab not found")
        timeline = await wait_for(b, lambda d: d["feed"] > 0, 30, "event timeline")
        record("clean_log_ledger_and_timeline", True, ledger_rows=ledger["logRows"], timeline_events=timeline["feed"])

        # live WebSocket data keeps flowing on a dedicated page (one session: no new socket per page)
        await click(b, 'a.navigation-item[href^="/map"]')
        d1 = await wait_for(b, lambda d: d["page"] == "map" and d["poses"] is not None, 30, "clean map")
        await asyncio.sleep(6.0)
        d2 = await b.js(DOM)
        record("clean_live_updates", d2["fresh"] == "Live" and d2["poses"] > d1["poses"], poses=[d1["poses"], d2["poses"]],
               ws_clients=get("/api/v1/health")["data"]["websocket_clients"])

        # BLACKOUT / RESTORE from the clean Network page; STALE shown on the map, poses frozen
        await click(b, 'a.navigation-item[href^="/network"]')
        await wait_for(b, lambda d: d["page"] == "network" and d["presets"] == 5, 30, "clean network")
        await click(b, '.network-preset-grid button[title*="BLACKOUT"]')
        await wait_for(b, lambda d: "BLACKOUT confirmed" in (d["presetFeedback"] or ""), 15, "clean BLACKOUT confirmation")
        d = await wait_for(b, lambda d: d["fresh"] == "Stale", 25, "clean STALE")
        res["screens"].append(await b.shot("19_clean_network_blackout.png"))
        await click(b, 'a.navigation-item[href^="/map"]')
        s1 = await wait_for(b, lambda d: d["page"] == "map" and d["cleanStale"] is not None, 20, "clean map stale banner")
        await asyncio.sleep(4.0)
        s2 = await b.js(DOM)
        res["screens"].append(await b.shot("20_clean_map_stale.png"))
        await click(b, 'a.navigation-item[href^="/network"]')
        await wait_for(b, lambda d: d["page"] == "network" and d["presets"] == 5, 30, "clean network")
        await click(b, '.network-preset-grid button[title*="RESTORE"]')
        await wait_for(b, lambda d: "RESTORE confirmed" in (d["presetFeedback"] or ""), 15, "clean RESTORE confirmation")
        d = await wait_for(b, lambda d: d["fresh"] == "Live", 40, "clean LIVE after restore")
        record("clean_blackout_restore", s1["poses"] == s2["poses"] and d["fresh"] == "Live",
               stale_banner=s1["cleanStale"], poses_frozen=[s1["poses"], s2["poses"]], fresh_after=d["fresh"])

        # mission switching from the Missions page: every page shows the picked mission only
        await click(b, 'a[href^="/missions"]')
        await wait_for(b, lambda d: d["page"] == "missions" and d["missionRows"] > 1, 30, "clean missions")
        missions = get("/api/v1/missions?limit=20")["data"]
        other = next(x for x in missions if x["mission_id"] != live_mission and x["counters"].get("pose", 0) > 0)
        ok = await b.js("(() => { const r = [...document.querySelectorAll('[data-page=missions] tbody tr')].find(e => "
                        f"e.textContent.includes('#{other['mission_id']}')); r?.click(); return !!r; }})()")
        if not ok:
            raise RuntimeError(f"mission {other['mission_id']} not listed on the Missions page")
        # R6.1 rows open an informative sheet; switching is a separate explicit
        # action, which the E2E must exercise instead of bypassing React state.
        await asyncio.sleep(1)
        ok = await b.js("(() => { const e = [...document.querySelectorAll('button')].find(e => "
                        "e.textContent.includes('Use this mission')); e?.click(); return !!e; })()")
        if not ok:
            raise RuntimeError("mission sheet switch action not found")
        await wait_for(b, lambda d: d["mission"] == other["mission_id"], 20, "clean mission switch")
        await click(b, 'a.navigation-item[href^="/map"]')
        m1 = await wait_for(b, lambda d: d["page"] == "map" and d["mission"] == other["mission_id"] and d["poses"] is not None, 40, "switched map")
        await asyncio.sleep(4.0)
        m2 = await b.js(DOM)
        rest_poses = get(f"/api/v1/missions/{other['mission_id']}/positions?limit=1")["meta"]["pagination"]["total"]
        record("clean_mission_switch_isolated", m1["poses"] == m2["poses"] == min(rest_poses, 5000)
               and f"mission={other['mission_id']}" in m2["href"],
               mission=other["mission_id"], poses=[m1["poses"], m2["poses"]], rest_poses=rest_poses, url=m2["href"])

        # Settings switch: Unified on (same mission), survives reload, off again
        await click(b, 'a.navigation-item[href^="/settings"]')
        await wait_for(b, lambda d: d["page"] == "settings" and d["switch"] is not None, 30, "settings")
        await click(b, '[data-setting=unified-mode] [role=switch]')
        d = await wait_for(b, lambda d: d["stored"] == "unified" and d["switch"] is True, 10, "unified stored")
        await click(b, 'a.navigation-item[href^="/settings"]')  # still a link to /settings in unified mode
        await b.js("document.querySelector('.brand-lockup').click()")
        u1 = await wait_for(b, lambda d: d["unified"] is True and d["mission"] == other["mission_id"], 30, "unified view")
        await b.call("Page.reload")
        await asyncio.sleep(3.0)
        u2 = await wait_for(b, lambda d: d["unified"] is True and d["mission"] == other["mission_id"], 40, "unified after reload")
        res["screens"].append(await b.shot("21_unified_after_switch.png"))
        await click(b, 'a.navigation-item[href^="/settings"]')
        await wait_for(b, lambda d: d["page"] == "settings" and d["switch"] is True, 30, "settings (unified)")
        await click(b, '[data-setting=unified-mode] [role=switch]')
        await wait_for(b, lambda d: d["stored"] == "clean", 10, "clean stored")
        await b.js("document.querySelector('.brand-lockup').click()")
        c1 = await wait_for(b, lambda d: d["page"] == "overview" and d["unified"] is False, 30, "clean overview again")
        record("mode_switch_persists_and_keeps_mission",
               u1["chip"] == "Unified console" and u2["unified"] and c1["mission"] == other["mission_id"],
               unified_chip=u1["chip"], after_reload=u2["unified"], back_to_clean=c1["page"], mission=c1["mission"])
        await b.close()
        b = Browser(out)
        await b.start()

        # 10. unknown mission: the API error is shown, nothing is fabricated
        await b.goto(f"{DASH}?mission=does-not-exist")
        d = await wait_for(b, lambda d: any("mission_not_found" in e for e in d["errors"]), 20, "404 error")
        res["screens"].append(await b.shot("10_unknown_mission_404.png"))
        record("api_error_unknown_mission", d["poses"] is None, errors=d["errors"], fresh=d["fresh"])

        # The live mission must finish the validated course, not be killed after
        # the UI checks. Keep the real stack alive until summary.json is written.
        end = time.time() + 700
        summary_path = out / "tour" / "summary.json"
        while not summary_path.exists() and time.time() < end:
            await asyncio.sleep(1)
        if not summary_path.exists():
            raise RuntimeError("course summary missing after completion deadline")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        res["tour"] = {k: summary.get(k) for k in ("final", "inference", "controller_work_ms_per_step", "wall_time_s", "sim_time_s")}
        record("webots_course_55_of_55", summary["final"]["completed_moves"] == summary["final"]["planned_moves"] == 55,
               final=summary["final"])
        record("course_yolo_without_failures", summary["inference"]["ok"] > 0 and summary["inference"]["failed"] == 0,
               inference=summary["inference"])
    except Exception as e:  # noqa: BLE001 - report the failing step with the last state
        res["error"] = f"{type(e).__name__}: {e}"
        print("ERROR", res["error"], flush=True)
        try:
            res["screens"].append(await b.shot("zz_failure.png"))
        except Exception:  # noqa: BLE001
            pass
    finally:
        res["console"] = b.console
        try:
            await b.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if stack.emulator is not None and stack.emulator.poll() is None:
                urllib.request.urlopen(urllib.request.Request(API + "/api/v1/network/preset", data=b'{"preset":"NORMAL"}',
                                                          headers={"Content-Type": "application/json"}, method="POST"), timeout=3)
        except OSError:
            pass
        stack.stop_all()
    res["passed"] = sum(s["ok"] for s in step.values())
    res["finished_utc"] = datetime.now(timezone.utc).isoformat()
    res["total"] = len(step)
    (out / "r6_dashboard_e2e.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artemis-webots" / "output" / ("r6_dashboard-" + time.strftime("%Y%m%d-%H%M%S"))))
    a = ap.parse_args()
    r = asyncio.run(main(Path(a.out).resolve()))
    print(json.dumps({"passed": r["passed"], "total": r["total"], "error": r.get("error"), "console": len(r["console"])}))
    sys.exit(0 if r["passed"] == r["total"] and not r.get("error") else 1)

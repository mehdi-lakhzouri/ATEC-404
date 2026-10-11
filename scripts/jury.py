"""Release orchestration, not a second simulator. Windows entry: scripts/jury.ps1.

Reuses R6.Stack, R1/R2 smoke and the existing regression suites. All output is
isolated under .jury/; no preset, process, database or old evidence is replaced
silently. Background-worker shutdown uses a request file and PID creation times.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".jury/active.json"
WEB = ROOT / "frontend/apps/web"
ML = ROOT / "backend/.venv/Scripts/python.exe"
GROUND = ROOT / "ground_station/.venv/Scripts/python.exe"
WEBOTS = Path(os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")) / "msys64/mingw64/bin/webots.exe"
EDGE = Path(os.environ.get("ARTEMIS_EDGE_EXE", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"))
PNPM = shutil.which("pnpm.cmd") or shutil.which("pnpm")


def write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_dir(prefix: str) -> Path:
    out = ROOT / ".jury/runs" / (prefix + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6])
    out.mkdir(parents=True, exist_ok=False)
    return out


def execute(cmd, cwd=ROOT, env=None, timeout=1800, log: Path | None = None) -> int:
    print("RUN", " ".join(str(c).replace(str(ROOT), ".") for c in cmd), flush=True)
    if log:
        with log.open("w", encoding="utf-8") as fh:
            p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
            try:
                code = p.wait(timeout)
            except subprocess.TimeoutExpired:
                import r6_dashboard_e2e as r6
                r6.kill_tree(p)
                code = 124
    else:
        code = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, timeout=timeout).returncode
    print("EXIT", code, flush=True)
    return code


def process_identity(pid: int) -> int | None:
    """Windows PID creation FILETIME, preventing accidental PID-reuse kills."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel.GetProcessTimes.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    times = [ctypes.c_ulonglong() for _ in range(4)]
    try:
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            return None
        return times[0].value
    finally:
        kernel.CloseHandle(handle)


def http(url: str):
    with urllib.request.urlopen(url, timeout=3) as response:
        return json.loads(response.read())


def preflight() -> int:
    checks = []
    def check(name, ok, fix, warning=False):
        state = "PASS" if ok else "WARN" if warning else "FAIL"
        checks.append({"name": name, "status": state, "message": "available" if ok else fix})
        print(state, name, "" if ok else fix, flush=True)
    check("Windows", os.name == "nt", "Official orchestrator requires Windows; manual commands are documented.")
    check("Python 3.12", sys.version_info[:2] == (3, 12), "uv sync --project ground_station --locked --group dev")
    for command in ("uv", "node", "docker", "git"):
        check(command, bool(shutil.which(command)), "Install " + command + " and reopen PowerShell.")
    check("pnpm", bool(PNPM), "Install pnpm 11.25.0 (frontend/package.json).")
    if shutil.which("node"):
        version = subprocess.check_output(["node", "--version"], text=True).strip()
        check("Node >=24", int(version.lstrip("v").split(".")[0]) >= 24, "Install Node >=24.")
    if PNPM:
        version = subprocess.check_output([PNPM, "--version"], cwd=ROOT / "frontend", text=True).strip()
        check("pnpm 11.25.0", version == "11.25.0", "Use the pinned frontend packageManager.")
    check("Webots R2025a executable", WEBOTS.is_file(), "Install R2025a or set WEBOTS_HOME.")
    check("Edge", EDGE.is_file(), "Install Edge or set ARTEMIS_EDGE_EXE.")
    for path in ("docker/.env", "ground_station/.env"):
        check(path, (ROOT / path).is_file(), "Copy the matching .env.example and configure matching private Mongo credentials.")
    if (ROOT / "ground_station/.env").exists():
        check("credentials configured", "change-me" not in (ROOT / "ground_station/.env").read_text(),
              "Replace the example password; credentials are never printed.")
    for python, modules in ((ML, ["torch", "ultralytics", "numpy", "PIL"]), (GROUND, ["uvicorn", "fastapi", "pymongo", "websockets"])):
        ok = python.exists() and subprocess.run([str(python), "-c", "import importlib.util,sys; sys.exit(not all(importlib.util.find_spec(m) for m in sys.argv[1:]))", *modules], capture_output=True).returncode == 0
        check(python.parent.parent.parent.name + " dependencies", ok, "Run uv sync --project " + python.parent.parent.parent.name + " --locked.")
    check("frontend dependencies", (WEB / "node_modules/next/dist/bin/next").exists(), "cd frontend; pnpm install --frozen-lockfile")
    expected = json.loads((ROOT / "artifacts/provenance/FROZEN_MODEL_CONFIG.json").read_text())["model"]["sha256"]
    model = ROOT / "models/artemis_rock_yolo11n.pt"
    check("frozen model SHA256", model.exists() and hashlib.sha256(model.read_bytes()).hexdigest() == expected,
          "Restore the tracked frozen model; never substitute newly trained weights.")
    asset_root = ROOT / "artemis-webots/assets/polyhaven"
    assets = json.loads((asset_root / "ASSETS.json").read_text())["assets"]
    missing = []
    changed = []
    for asset in assets.values():
        for entry in asset["files"]:
            path = asset_root / entry["file"]
            if not path.exists(): missing.append(entry["file"])
            elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]: changed.append(entry["file"])
        for name in asset.get("converted", []):
            if not (asset_root / name).exists(): missing.append(name)
        if asset.get("derived"):
            entry = asset["derived"]
            path = asset_root / entry["file"]
            if not path.exists(): missing.append(entry["file"])
            elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]: changed.append(entry["file"])
    check("world asset availability + source hashes", not missing and not changed,
          f"Missing {len(missing)}, mismatched {len(changed)}. Run uv run --project backend --no-sync python artemis-webots/tools/fetch_polyhaven.py --locked")
    if shutil.which("docker"):
        check("Docker Desktop", subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True).returncode == 0, "Start Docker Desktop.")
        check("Docker Compose", subprocess.run(["docker", "compose", "version"], capture_output=True).returncode == 0, "Install/enable Compose v2.")
        check("Compose configuration", subprocess.run(["docker", "compose", "-f", str(ROOT / "docker/compose.yml"), "config", "--quiet"], capture_output=True).returncode == 0, "Configure docker/.env; do not print resolved Compose credentials.")
    try:
        sys.path.insert(0, str(ROOT / "ground_station"))
        from artemis_ground.config import Settings
        from pymongo import MongoClient
        with MongoClient(Settings().mongo_uri, serverSelectionTimeoutMS=2000) as client:
            client.admin.command("ping")
        mongo_ok = True
    except Exception:
        mongo_ok = False
    check("Mongo authenticated ping", mongo_ok, "docker compose -f docker/compose.yml up -d mongodb; wait for healthy", warning=True)
    import r6_dashboard_e2e as r6
    for port in (3000, 8000, 5761, 5771, 9333):
        check(f"TCP {port} free", not r6.port_open(port), "Already occupied; close it yourself or explicitly reuse the verified dashboard.", warning=True)
    for port in (5790, 5791, 5792, 5793, 5794):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock: sock.bind(("127.0.0.1", port))
            free = True
        except OSError: free = False
        check(f"UDP {port} free", free, "Resolve the existing listener; do not launch duplicate receivers.", warning=True)
    print("Preflight is read-only. WARN is not proof of readiness; start/demo enforce prerequisites.")
    out = run_dir("preflight")
    write(out / "preflight.json", checks)
    return int(any(c["status"] == "FAIL" for c in checks))


def frontend_start(out: Path, reuse: bool):
    import r6_dashboard_e2e as r6
    if r6.port_open(3000):
        if not reuse: raise RuntimeError(":3000 occupied; use --reuse-dashboard only for the existing Artemis dashboard.")
        with urllib.request.urlopen("http://localhost:3000/settings", timeout=20) as response:
            if response.status != 200 or b"Artemis" not in response.read():
                raise RuntimeError("existing :3000 is not the expected dashboard")
        return None
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "frontend.log").open("w", encoding="utf-8")
    process = subprocess.Popen([shutil.which("node"), str(WEB / "node_modules/next/dist/bin/next"), "dev", "--hostname", "127.0.0.1", "--port", "3000", str(WEB)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    try:
        end = time.time() + 90
        while time.time() < end:
            if process.poll() is not None: raise RuntimeError("frontend launch failed")
            try:
                with urllib.request.urlopen("http://127.0.0.1:3000/settings", timeout=5) as response:
                    if response.status == 200: return process
            except OSError: pass
            time.sleep(.5)
        raise RuntimeError("dashboard health timeout")
    except BaseException:
        r6.kill_tree(process)
        raise


def worker(out: Path, reuse: bool) -> int:
    import r6_dashboard_e2e as r6
    r6.INFER_PORT = 5761
    stack = r6.Stack(out)
    frontend = None
    processes = []
    container_owned = False
    container_id = None
    data = {"repo": str(ROOT), "worker": {"pid": os.getpid(), "created": process_identity(os.getpid())},
            "run": str(out), "status": "starting", "processes": processes}
    def save():
        data["processes"] = [{"name": name, "pid": p.pid, "created": process_identity(p.pid)} for name, p in processes if p.poll() is None]
        data["container"] = {"owned": container_owned, "id": container_id}
        write(STATE, data)
    try:
        # R6's pre-existing-process checks apply to both official start and demo.
        for port in (8000, 5761):
            if r6.port_open(port): raise RuntimeError(f"TCP {port} already occupied")
        for port in (5790, 5791, 5792, 5793, 5794):
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe: probe.bind(("127.0.0.1", port))
        proc = subprocess.run(["powershell.exe", "-NoProfile", "-Command", "@(Get-Process webots,webots-bin -ErrorAction SilentlyContinue).Count"], capture_output=True, text=True, check=True)
        if int(proc.stdout.strip()): raise RuntimeError("Webots already running")
        if r6.port_open(3000) and not reuse: raise RuntimeError("TCP 3000 already occupied")
        save()
        running = subprocess.check_output(["docker", "ps", "--filter", "name=^/artemis-mongodb$", "--format", "{{.ID}}"], text=True).strip()
        if execute(["docker", "compose", "-f", ROOT / "docker/compose.yml", "up", "-d", "mongodb"]):
            raise RuntimeError("Mongo startup failed")
        container_owned = not bool(running)
        container_id = subprocess.check_output(["docker", "inspect", "--format", "{{.Id}}", "artemis-mongodb"], text=True).strip()
        save()
        end = time.time() + 90
        while time.time() < end:
            health = subprocess.check_output(["docker", "inspect", "--format", "{{.State.Health.Status}}", container_id], text=True).strip()
            if health == "healthy": break
            time.sleep(1)
        else: raise RuntimeError("Mongo health timeout")
        # Register immediately, even if a readiness check raises, so cleanup owns
        # only the actual Popen handles that this worker created.
        try: stack.start_backend()
        finally:
            if stack.backend: processes.append(("ground", stack.backend)); save()
        try: stack.start_emulator()
        finally:
            if stack.emulator: processes.append(("emulator", stack.emulator)); save()
        try: stack.start_inference()
        finally:
            if stack.infer: processes.append(("inference", stack.infer)); save()
        frontend = frontend_start(out, reuse)
        if frontend: processes.append(("frontend", frontend)); save()
        stack.start_webots()
        processes.append(("webots", stack.webots)); save()
        end = time.time() + 180
        while time.time() < end:
            health = r6.get("/api/v1/health")["data"]
            if health["receiver"]["valid"] > 0: break
            if stack.webots.poll() is not None: raise RuntimeError("Webots exited before telemetry")
            time.sleep(1)
        else: raise RuntimeError("no real rover telemetry received")
        data["status"] = "ready"; save()
        print("READY http://127.0.0.1:3000 | API http://127.0.0.1:8000/api/v1/health", flush=True)
        while not (out / "stop.request").exists():
            if any(p.poll() is not None for name, p in processes if name != "webots"):
                raise RuntimeError("a service exited; see per-service logs")
            time.sleep(.5)
    except BaseException as error:
        data["status"] = "failed"
        data["error"] = str(error)
        print("FAIL", error, flush=True)
    finally:
        stack.stop_all()
        r6.kill_tree(frontend)
        if container_owned and container_id:
            subprocess.run(["docker", "stop", container_id], capture_output=True)
        data["status"] = "failed" if data.get("error") else "stopped"
        save()
    return int(bool(data.get("error")))


def start(reuse: bool) -> int:
    if STATE.exists():
        old = json.loads(STATE.read_text())
        if process_identity(old["worker"]["pid"]) == old["worker"]["created"]:
            raise RuntimeError("an owned stack worker is already running; use status/stop")
        if old["status"] not in ("stopped", "failed"):
            raise RuntimeError("stale ownership manifest; run stop before starting another stack")
    if preflight(): return 1
    out = run_dir("stack")
    with (out / "worker.log").open("w", encoding="utf-8") as log:
        command = [str(GROUND), str(Path(__file__).resolve()), "worker", "--out", str(out)]
        if reuse: command.append("--reuse-dashboard")
        p = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    end = time.time() + 360
    while time.time() < end:
        if STATE.exists():
            state = json.loads(STATE.read_text())
            if state["worker"]["pid"] == p.pid:
                if state["status"] == "ready":
                    print("PASS real telemetry ready | http://127.0.0.1:3000 | logs", out.relative_to(ROOT)); return 0
                if state["status"] == "failed": raise RuntimeError(state.get("error", "worker failed"))
        if p.poll() is not None: raise RuntimeError("worker exited; inspect worker.log")
        time.sleep(.5)
    (out / "stop.request").touch()
    raise RuntimeError("startup deadline reached; shutdown requested; inspect worker.log")


def stop() -> int:
    if not STATE.exists(): print("No owned stack."); return 0
    data = json.loads(STATE.read_text())
    out = Path(data["run"]).resolve()
    if data["repo"] != str(ROOT) or not out.is_relative_to(ROOT / ".jury/runs"):
        raise RuntimeError("ownership manifest is outside this repository")
    (out / "stop.request").touch()
    owner = data["worker"]
    end = time.time() + 45
    while process_identity(owner["pid"]) == owner["created"] and time.time() < end: time.sleep(.3)
    if process_identity(owner["pid"]) == owner["created"]:
        raise RuntimeError("worker still shutting down; no unrelated process was killed; retry stop")
    # Crash recovery: only exact PID + creation-time identities in this worker's
    # manifest may be stopped. Never select processes merely by name or port.
    for process in reversed(data["processes"]):
        if process["created"] is not None and process_identity(process["pid"]) == process["created"]:
            subprocess.run(["taskkill", "/PID", str(process["pid"]), "/T", "/F"], capture_output=True)
    container = data.get("container", {})
    if container.get("owned") and container.get("id"):
        subprocess.run(["docker", "stop", container["id"]], capture_output=True)
    data["status"] = "stopped"; write(STATE, data)
    print("PASS owned processes stopped. Existing services preserved; Mongo volume NOT deleted.")
    return 0


def demo(reuse: bool, out: Path | None = None) -> int:
    import r6_dashboard_e2e as r6
    out = out or run_dir("demo")
    # Empty evidence dir accepted by R6; a private database is retained, never
    # dropped, and proves this workflow does not depend on existing mission data.
    env = dict(os.environ, ARTEMIS_GROUND_MONGO_DB="artemis_jury_" + out.name.replace("-", "_"))
    frontend = None
    try:
        frontend = frontend_start(out.parent / (out.name + "-frontend"), reuse)
        return execute([GROUND, ROOT / "scripts/r6_dashboard_e2e.py", "--out", out], env=env, log=out.parent / (out.name + ".log"))
    finally: r6.kill_tree(frontend)


def tests(full: bool, reuse: bool) -> int:
    out = run_dir("test")
    results = []
    def task(name, command, cwd=ROOT, env=None):
        start_time = time.monotonic()
        code = execute(command, cwd=cwd, env=env, log=out / (name + ".log"))
        results.append({"name": name, "command": [str(c).replace(str(ROOT), ".") for c in command],
                        "exit_code": code, "elapsed_s": round(time.monotonic() - start_time, 3)})
        write(out / "commands.json", results)
        return code
    overlay = ["uv", "run", "--project", "backend", "--no-sync", "--with-requirements", "tests/requirements.txt", "python"]
    env = dict(os.environ, ARTEMIS_RUN_WEBOTS="1" if full else "0")
    task("runtime", [*overlay, "scripts/jury_unittest.py", "--suite", "runtime", "--out", out / "runtime.json"], env=env)
    task("research", [*overlay, "scripts/jury_unittest.py", "--suite", "research", "--out", out / "research.json"])
    if full:
        task("ground", [GROUND, "scripts/jury_unittest.py", "--suite", "ground", "--require-integrations", "--out", out / "ground.json"])
    else:
        results.append({"name": "Mongo/Webots/browser integrations", "status": "NOT RUN", "reason": "unit mode; run -Step test for integrations"})
    task("frontend-tests", [PNPM, "--filter", "web", "test"], cwd=ROOT / "frontend")
    task("frontend-quality", [PNPM, "exec", "turbo", "run", "lint", "check-types", "build", "--force", "--ui=stream"], cwd=ROOT / "frontend")
    if full and not any(r.get("exit_code", 0) for r in results):
        code = demo(reuse)
        results.append({"name": "R6 real-stack E2E + 55/55", "exit_code": code})
    elif full:
        results.append({"name": "R6 real-stack E2E + 55/55", "status": "NOT RUN", "reason": "earlier checks failed"})
    write(out / "commands.json", results)
    print("Evidence:", out.relative_to(ROOT))
    return int(any(r.get("exit_code", 0) != 0 for r in results) or (full and any(r.get("status") == "NOT RUN" for r in results)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=("preflight", "unit", "test", "start", "stop", "demo", "status", "worker"))
    ap.add_argument("--reuse-dashboard", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    os.chdir(ROOT)
    if os.name != "nt": raise RuntimeError("Use the documented manual commands on non-Windows hosts.")
    if args.step == "preflight": return preflight()
    if args.step == "start": return start(args.reuse_dashboard)
    if args.step == "stop": return stop()
    if args.step == "worker": return worker(args.out, args.reuse_dashboard)
    if args.step == "demo": return demo(args.reuse_dashboard)
    if args.step in ("unit", "test"): return tests(args.step == "test", args.reuse_dashboard)
    print(STATE.read_text() if STATE.exists() else "No owned stack.")
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print("FAIL", exc, file=sys.stderr)
        raise SystemExit(1)

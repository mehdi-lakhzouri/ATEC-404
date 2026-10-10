"""R5.2: evaluation of the live Webots -> UDP -> ground receiver runs (scripts/run_webots_r4.py telemetry).

Per scenario: packets sent (controller send log) vs received and decoded (ground JSONL); decoded POSE vs the controller's
own odometry at the same simulation time; the reconstructed obstacle picture vs the run's final export; first reception
vs mission end; send -> receive latency (same machine, same wall clock) and throughput; and the mapping / tour impact
against the telemetry-off reference run of the same batch.
"""

from __future__ import annotations

import bisect
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def q(xs, p):
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(p * (len(s) - 1)))], 3) if s else None


def analyse_run(d: Path) -> dict:
    tour = d / "tour"
    summ = json.loads((tour / "summary.json").read_text(encoding="utf-8"))
    mp = summ["mapping"]
    res = {"tour": f"{summ['final']['completed_moves']}/{summ['final']['planned_moves']}",
           "cycle_p95_ms": mp["timing_ms"]["cycle_ms"]["p95"], "step_max_ms": mp["timing_ms"]["step_max_ms"]["max"],
           "controller_mem_end_mb": mp["process_memory_mb"]["at_end"],
           "yolo_ok": (summ.get("inference") or {}).get("ok"), "yolo_submitted": (summ.get("inference") or {}).get("submitted")}
    tel = summ.get("telemetry")
    if tel is None:
        return res
    res["sent"] = tel["sender"]
    res["adapter"] = tel["adapter"]
    cyc = [json.loads(x) for x in (tour / "mapping" / "mapping.jsonl").read_text(encoding="utf-8").splitlines() if '"wall_s"' in x]
    mission_end_wall = max(c["wall_s"] for c in cyc)
    log = json.loads((tour / "telemetry_send_log.json").read_text(encoding="utf-8"))["rows"]
    gstats = json.loads((d / "ground_stats.json").read_text(encoding="utf-8")) if (d / "ground_stats.json").exists() else {}
    res["ground_receivers"] = [{k: r[k] for k in ("datagrams", "valid", "invalid", "gaps", "missing", "duplicates", "reordered", "by_type")}
                               for r in gstats.get("receivers", [])]
    gfile = d / "ground.jsonl"
    rx = [json.loads(x) for x in gfile.read_text(encoding="utf-8").splitlines()] if gfile.exists() else []
    valid = [r for r in rx if "invalid" not in r]
    res["received_valid"] = len(valid)
    res["received_invalid"] = len(rx) - len(valid)
    if not valid:
        return res
    # sent vs received by type
    sent_by = {}
    for seq, ptype, *_ in log:
        sent_by[ptype] = sent_by.get(ptype, 0) + 1
    rx_by = {}
    for r in valid:
        rx_by[r["type"]] = rx_by.get(r["type"], 0) + 1
    names = {1: "pose", 2: "obstacles", 3: "event"}
    res["by_type"] = {names[k]: {"sent": sent_by.get(k, 0), "received": rx_by.get(k, 0)} for k in names}
    # latency: send wall (controller log) -> reception wall (ground), joined on seq (single session)
    send_wall = {row[0]: row[4] for row in log if row[5]}
    lat = [1000 * (r["rx_wall_s"] - send_wall[r["seq"]]) for r in valid if r["seq"] in send_wall]
    res["latency_ms"] = {"n": len(lat), "p50": q(lat, 0.5), "p95": q(lat, 0.95), "max": round(max(lat), 3) if lat else None}
    res["first_rx_wall_s"] = valid[0]["rx_wall_s"]
    res["first_rx_sim_t_obs_s"] = valid[0]["t_obs_s"]
    res["mission_end_wall_s"] = mission_end_wall
    res["received_before_mission_end"] = sum(1 for r in valid if r["rx_wall_s"] <= mission_end_wall)
    span = valid[-1]["rx_wall_s"] - valid[0]["rx_wall_s"]
    res["throughput"] = {"wall_span_s": round(span, 1), "packets_per_s": round(len(valid) / span, 2) if span else None,
                         "bytes_total": tel["sender"]["bytes"], "bytes_per_s_wall": round(tel["sender"]["bytes"] / span, 1) if span else None,
                         "bytes_per_sim_s": round(tel["sender"]["bytes"] / max(1e-9, cyc[-1]["stamp_s"]), 1)}
    # decoded POSE vs the controller's odometry at the same simulation time (telemetry.csv)
    rows = list(csv.DictReader((tour / "telemetry.csv").open(encoding="utf-8")))
    ts = [float(r["sim_time_s"]) for r in rows]
    errs = []
    for r in valid:
        if r["type"] != 1:
            continue
        i = bisect.bisect_left(ts, r["t_obs_s"] - 1e-6)
        if i < len(ts) and abs(ts[i] - r["t_obs_s"]) < 0.0011:
            o = rows[i]
            errs.append(max(abs(r["body"]["x"] - float(o["odo_x"])), abs(r["body"]["y"] - float(o["odo_y"]))))
    res["pose_vs_odometry_max_abs_m"] = {"n": len(errs), "max": round(max(errs), 6) if errs else None}
    # obstacles: reconstruct the ground picture (latest record per id since the last full snapshot) vs the final export
    final = json.loads((tour / "mapping" / "obstacles_final.json").read_text(encoding="utf-8"))
    picture = {}
    for r in valid:
        if r["type"] == 2:
            for o in r["body"]["obstacles"]:
                picture[o["id"]] = o
    from artemis_x.telemetry.adapter import dedupe
    expect, _ = dedupe(final)
    exp_ids = {o["id"] for o in expect}
    got = {i: picture[i] for i in exp_ids if i in picture}
    dev = [max(abs(got[o["id"]]["x_m"] - o["x_m"]), abs(got[o["id"]]["y_m"] - o["y_m"])) for o in expect if o["id"] in got]
    res["obstacles_final_vs_received"] = {"final_deduplicated": len(exp_ids), "received": len(got),
                                          "missing_ids": sorted(exp_ids - set(got))[:10],
                                          "max_position_diff_m": round(max(dev), 6) if dev else None,
                                          "frame": final["frame"], "session": tel["session"]}
    if (d / "link_stats.json").exists():
        res["link"] = link_report(d)
    return res


def link_report(d: Path) -> dict:
    """Emulator view: totals, per-preset phases (from link.jsonl), configured vs observed loss, blackout and resumption."""
    st = json.loads((d / "link_stats.json").read_text(encoding="utf-8"))
    lines = [json.loads(x) for x in (d / "link.jsonl").read_text(encoding="utf-8").splitlines()]
    from artemis_x.telemetry.emulator import PRESETS
    phases = {}
    for r in lines:
        if "action" not in r:
            continue
        ph = phases.setdefault(r["preset"], {"received": 0, "forwarded": 0, "dropped_loss": 0, "dropped_blackout": 0,
                                             "dropped_queue": 0, "dropped_shutdown": 0, "delay_ms": []})
        if r["action"] == "forwarded":
            ph["forwarded"] += 1
            ph["delay_ms"].append(r["delay_ms"])
        else:
            ph[r["action"]] += 1
    for name, ph in phases.items():
        n = ph["forwarded"] + ph["dropped_loss"] + ph["dropped_blackout"] + ph["dropped_queue"] + ph["dropped_shutdown"]
        ph["received"] = n
        cfg = PRESETS.get(name)
        ph["configured_loss"] = cfg.loss if cfg else None
        # loss stage only: of the packets that reached the random-loss stage, the fraction it dropped
        stage = n - ph["dropped_blackout"]
        ph["observed_random_loss"] = round(ph["dropped_loss"] / stage, 4) if stage and not (cfg and cfg.blackout) else None
        ph["delivery_ratio"] = round(ph["forwarded"] / n, 4) if n else None
        dl = ph.pop("delay_ms")
        ph["delay_ms"] = {"p50": q(dl, 0.5), "p95": q(dl, 0.95), "max": round(max(dl), 1) if dl else None}
    restore = next((r for r in lines if r.get("event") == "config" and r.get("preset") == "RESTORE"), None)
    resumed = None
    if restore:
        first = next((r for r in lines if r.get("action") == "forwarded" and r["t_s"] >= restore["t_s"]), None)
        resumed = round(first["t_s"] - restore["t_s"], 3) if first else None
    gs = json.loads((d / "ground_stats.json").read_text(encoding="utf-8"))
    return {"totals": {k: st[k] for k in ("received", "forwarded", "dropped_loss", "dropped_blackout", "dropped_queue",
                                          "dropped_shutdown", "invalid_crc_seen", "delivery_ratio", "blackout_total_s",
                                          "throughput_Bps", "delay_ms", "elapsed_s")},
            "phases": phases, "switches": gs.get("switches", []), "resumed_after_restore_s": resumed}


def analyse_telemetry(root: Path) -> dict:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    out = {"root": root.relative_to(ROOT).as_posix(), "scenarios": {}}
    for d in sorted(p for p in root.iterdir() if (p / "tour" / "summary.json").exists()):
        out["scenarios"][d.name] = analyse_run(d)
    (root / "r5_2_telemetry_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys
    print(json.dumps(analyse_telemetry(Path(sys.argv[1]).resolve()), indent=1))

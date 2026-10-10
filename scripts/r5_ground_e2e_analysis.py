"""R5.5 end-to-end verification: what the ground backend persisted / streamed vs what the lunar link actually delivered.

    uv run --project ground_station python scripts/r5_ground_e2e_analysis.py artemis-webots/output/r5-ground-<ts>/ground_e2e

Checks (MongoDB read directly, emulator log, WebSocket capture, rover logs):
  * every packet persisted in MongoDB was FORWARDED by the emulator (data comes exclusively through the link);
  * persisted unique packets == packets the receiver decoded (no ingestion loss, also across the MongoDB restart);
  * poses in MongoDB match the rover's own odometry at the same observation time;
  * the reconciled obstacle collection matches the rover's final export (deduplicated);
  * WebSocket event counts / envelope fields; presets applied through the REST API reached the emulator.
"""

from __future__ import annotations

import bisect
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ground_station"))
sys.path.insert(0, str(ROOT / "src"))
from artemis_ground.config import Settings  # noqa: E402
from artemis_x.telemetry.adapter import dedupe  # noqa: E402
from pymongo import MongoClient  # noqa: E402


def main(d: Path) -> dict:
    tour = d / "tour"
    summ = json.loads((tour / "summary.json").read_text(encoding="utf-8"))
    session = summ["telemetry"]["session"]
    mid = str(session)
    link = [json.loads(x) for x in (d / "link.jsonl").read_text(encoding="utf-8").splitlines()]
    forwarded = {r["seq"] for r in link if r.get("action") == "forwarded" and r.get("session") == session}
    dropped_only = {r["seq"] for r in link if str(r.get("action", "")).startswith("dropped") and r.get("session") == session} - forwarded
    s = Settings()
    with MongoClient(s.mongo_uri) as c:
        db = c[s.mongo_db]
        persisted = {e["seq"] for e in db["telemetry_events"].find({"mission_id": mid}, {"seq": 1})}
        poses = list(db["rover_positions"].find({"mission_id": mid}, {"_id": 0}))
        obs = {o["obstacle_id"]: o for o in db["obstacles"].find({"mission_id": mid, "present": True}, {"_id": 0})}
        mission = db["missions"].find_one({"_id": mid})
        link_docs = db["link_metrics"].count_documents({"mission_id": mid})
    health = json.loads((d / "backend_health.json").read_text(encoding="utf-8"))["data"]
    rows = list(csv.DictReader((tour / "telemetry.csv").open(encoding="utf-8")))
    ts = [float(r["sim_time_s"]) for r in rows]
    perr = []
    for p in poses:
        i = bisect.bisect_left(ts, p["t_obs_s"] - 1e-6)
        if i < len(ts) and abs(ts[i] - p["t_obs_s"]) < 0.0011:
            perr.append(max(abs(p["x_m"] - float(rows[i]["odo_x"])), abs(p["y_m"] - float(rows[i]["odo_y"]))))
    final = json.loads((tour / "mapping" / "obstacles_final.json").read_text(encoding="utf-8"))
    exp = {o["id"]: o for o in dedupe(final)[0]}
    odiff = [max(abs(obs[i]["x_m"] - o["x_m"]), abs(obs[i]["y_m"] - o["y_m"])) for i, o in exp.items() if i in obs]
    ws = [json.loads(x) for x in (d / "ws_events.jsonl").read_text(encoding="utf-8").splitlines()] if (d / "ws_events.jsonl").exists() else []
    wtypes = {}
    for w in ws:
        wtypes[w["event"]["type"]] = wtypes.get(w["event"]["type"], 0) + 1
    env_ok = all({"type", "api_version", "mission_id", "t_obs_s", "rx_utc", "data"} <= set(w["event"]) for w in ws)
    gstats = json.loads((d / "ground_stats.json").read_text(encoding="utf-8"))
    bev = json.loads((d / "backend_events.json").read_text(encoding="utf-8")) if (d / "backend_events.json").exists() else []
    cfg_events = [(r["preset"], r["wall_s"]) for r in link if r.get("event") == "config"]
    res = {
        "mission_id": mid, "tour": f"{summ['final']['completed_moves']}/{summ['final']['planned_moves']}",
        "mapping_cycle_p95_ms": summ["mapping"]["timing_ms"]["cycle_ms"]["p95"],
        "packets": {"rover_sent_unique": len({r[0] for r in json.loads((tour / "telemetry_send_log.json").read_text(encoding='utf-8'))['rows']}),
                    "emulator_forwarded_unique": len(forwarded), "emulator_dropped_never_forwarded": len(dropped_only),
                    "receiver_valid": health["receiver"]["valid"], "mongodb_persisted_unique": len(persisted),
                    "persisted_not_forwarded": sorted(persisted - forwarded)[:10],
                    "persisted_but_only_dropped": sorted(persisted & dropped_only)[:10],
                    "forwarded_not_persisted": len(forwarded - persisted)},
        "ingestion": {k: health["ingestion"][k] for k in ("persisted", "duplicates_ignored", "dropped_queue_full", "db_errors", "db_retries", "db_ok")},
        "mission_doc": {"counters": mission.get("counters"), "status_field": mission.get("status"), "health": mission.get("health")},
        "poses": {"persisted": len(poses), "vs_odometry_max_abs_m": round(max(perr), 6) if perr else None, "compared": len(perr)},
        "obstacles": {"present_in_db": len(obs), "final_export_dedup": len(exp), "matched": sum(i in obs for i in exp),
                      "missing": sorted(set(exp) - set(obs))[:10], "stale_extra": len(set(obs) - set(exp)),
                      "max_position_diff_m": round(max(odiff), 4) if odiff else None},
        "link_metrics_docs": link_docs,
        "websocket": {"events": len(ws), "by_type": wtypes, "envelopes_complete": env_ok},
        "network_control": {"switches_via_backend": gstats.get("switches"), "emulator_config_events": cfg_events},
        "db_restart": bev,
    }
    (d / "r5_5_ground_e2e.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return res


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]).resolve()), indent=1, default=str))

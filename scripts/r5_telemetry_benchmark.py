"""R5.1: binary protocol v1 vs compact JSON on REAL artemis.obstacles.v1 exports (every obstacles_final.json under a
run directory, e.g. the R4.6 validation). Sizes are measured byte counts - nothing is estimated. zlib-compressed JSON is
reported as a measured reference only (the protocol itself does not compress).

    uv run --project backend --no-sync python scripts/r5_telemetry_benchmark.py artemis-webots/output/r4-validate-<ts>
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from artemis_x.telemetry import protocol as P  # noqa: E402
from artemis_x.telemetry.adapter import dedupe  # noqa: E402


def compact(x) -> bytes:
    return json.dumps(x, separators=(",", ":")).encode("utf-8")


SENT = ("id", "type", "x_m", "y_m", "radius_m", "confidence", "uncertainty_m", "timestamp_s", "source", "status", "protected")


def equivalent(doc: dict, obs: list) -> dict:
    """The same information the binary snapshot carries, as compact JSON (contract precision, transmitted fields only)."""
    return {"type": "obstacles", "session": 1, "seq": 0, "t_obs": doc["stamp_s"], "frame": "odom", "map_resets": doc["map_resets"],
            "obstacles": [{**{k: o[k] for k in SENT}, **({"linked_track": o["track_ids"][0]} if o.get("track_ids") else {})} for o in obs]}


def bench(root: Path) -> dict:
    rows = []
    for f in sorted(root.rglob("obstacles_final.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        obs = doc["obstacles"]
        if not obs:
            continue
        kept, dropped = dedupe(doc)
        b_all = sum(len(p) for p in P.encode_obstacles(1, 0, doc))
        b_ded = sum(len(p) for p in P.encode_obstacles(1, 0, doc, kept))
        j_doc, j_obs, j_eq = compact(doc), compact(obs), compact(equivalent(doc, obs))
        rows.append({"run": f.relative_to(root).parts[0] + "/" + f.relative_to(root).parts[1], "obstacles": len(obs),
                     "json_equivalent_B": len(j_eq), "json_document_B": len(j_doc), "json_obstacles_B": len(j_obs), "json_obstacles_zlib9_B": len(zlib.compress(j_obs, 9)),
                     "binary_B": b_all, "binary_packets": len(P.encode_obstacles(1, 0, doc)),
                     "deduplicated": dropped, "binary_dedup_B": b_ded,
                     "json_per_obstacle_B": round(len(j_obs) / len(obs), 1), "binary_per_obstacle_B": round(b_all / len(obs), 1)})
    tot = {k: sum(r[k] for r in rows) for k in ("obstacles", "json_equivalent_B", "json_document_B", "json_obstacles_B", "json_obstacles_zlib9_B",
                                                 "binary_B", "binary_dedup_B", "deduplicated")}
    tot["saving_vs_json_equivalent"] = round(1 - tot["binary_B"] / tot["json_equivalent_B"], 4)
    tot["ratio_json_equivalent_to_binary"] = round(tot["json_equivalent_B"] / tot["binary_B"], 2)
    tot["saving_vs_json_obstacles"] = round(1 - tot["binary_B"] / tot["json_obstacles_B"], 4)
    tot["saving_vs_json_document"] = round(1 - tot["binary_B"] / tot["json_document_B"], 4)
    tot["ratio_json_obstacles_to_binary"] = round(tot["json_obstacles_B"] / tot["binary_B"], 2)
    tot["binary_vs_zlib9_json"] = round(tot["binary_B"] / tot["json_obstacles_zlib9_B"], 3)
    # pose and event packets vs the equivalent compact JSON
    doc = json.loads(sorted(root.rglob("obstacles_final.json"))[0].read_text(encoding="utf-8"))
    pose = doc["pose"]
    pose_json = compact({"type": "pose", "session": 1, "seq": 0, "t_obs": pose["stamp_s"], "frame": "odom",
                         "x": pose["x"], "y": pose["y"], "yaw": pose["yaw"], "sigma_xy": pose["sigma_xy"]})
    ev = {"type": "event", "session": 1, "seq": 0, "t_obs": 12.5, "event": "health", "state": "STOP",
          "reasons": ["stale_corridor", "inference_unavailable"], "value": 0}
    return {"source": root.relative_to(ROOT).as_posix() if root.is_relative_to(ROOT) else str(root), "runs": rows, "total": tot,
            "pose": {"binary_B": len(P.encode_pose(1, 0, pose)), "json_B": len(pose_json)},
            "event": {"binary_B": len(P.encode_event(1, 0, 12.5, "health", "STOP", ["stale_corridor", "inference_unavailable"])),
                      "json_B": len(compact(ev))},
            "record_B": P.RECORD.size, "packet_overhead_B": P.HEADER.size + P.OBS_HEAD.size + P.CRC.size, "max_packet_B": P.MAX_PACKET}


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    res = bench(root)
    (root / "r5_1_telemetry_benchmark.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))

"""R5.4: evaluation of the store-and-forward runs (scripts/run_webots_r4.py dtn).

Delivery is judged at the GROUND (what the receiver actually decoded), never from the rover's sends. Per run:
critical packets generated / delivered / lost and, for those first sent during a blackout, the recovery time after the
RESTORE; per-priority delivery; EVENT packets first sent during a blackout and delivered (comparable with the no-DTN
reference run); the reconciled ground map vs the rover's final export; rover DTN stats (retries, expiry, eviction,
queue memory, detected outages); plus the R5.2 / R5.3 fields (tour, mapping p95, YOLO, pose correctness).
"""

from __future__ import annotations

import bisect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from r5_telemetry_analysis import analyse_run, q  # noqa: E402


def windows(d: Path) -> list:
    """[(blackout_wall_start, restore_wall)] from the emulator's config events."""
    out, start = [], None
    for line in (d / "link.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("event") != "config":
            continue
        if r["preset"] == "BLACKOUT" and start is None:
            start = r["wall_s"]
        elif r["preset"] != "BLACKOUT" and start is not None:
            out.append((start, r["wall_s"]))
            start = None
    if start is not None:
        out.append((start, None))
    return out


def sim_to_wall(tour: Path):
    """Simulation time -> wall time of the rover, interpolated from the mapping cycles (both are logged per cycle)."""
    cyc = [json.loads(x) for x in (tour / "mapping" / "mapping.jsonl").read_text(encoding="utf-8").splitlines() if '"wall_s"' in x]
    ts, ws = [c["stamp_s"] for c in cyc], [c["wall_s"] for c in cyc]

    def f(t):
        i = bisect.bisect_left(ts, t)
        if i <= 0:
            return ws[0]
        if i >= len(ts):
            return ws[-1]
        a = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
        return ws[i - 1] + a * (ws[i] - ws[i - 1])
    return f


def dtn_report(d: Path) -> dict:
    tour = d / "tour"
    gen_wall = sim_to_wall(tour)
    summ = json.loads((tour / "summary.json").read_text(encoding="utf-8"))
    tel = summ["telemetry"]
    log = json.loads((tour / "telemetry_send_log.json").read_text(encoding="utf-8"))
    cols, rows = log["columns"], log["rows"]
    has_prio = "prio" in cols
    first_send, prio, ptype, t_obs = {}, {}, {}, {}
    for r in rows:
        seq = r[0]
        first_send.setdefault(seq, r[4])
        ptype[seq] = r[1]
        t_obs[seq] = r[2]
        if has_prio:
            prio[seq] = r[7]
    rx_first, seen_ids = {}, {}
    for line in (d / "ground.jsonl").read_text(encoding="utf-8").splitlines():
        g = json.loads(line)
        if "invalid" in g or g.get("session") != tel["session"]:
            continue
        rx_first.setdefault(g["seq"], g["rx_wall_s"])
        if g["type"] == 2:
            for o in g["body"]["obstacles"]:
                seen_ids.setdefault(o["id"], g["rx_wall_s"])
    wins = windows(d)

    def in_blackout(w):
        return next(((a, b) for a, b in wins if a <= w and (b is None or w < b)), None)

    res = {"blackouts": [[round(b - a, 1) if b else None] for a, b in wins], "sent_unique": len(first_send),
           "delivered_unique": sum(1 for s in first_send if s in rx_first)}
    def generated_in_blackout(s):            # generation = observation time on the rover, mapped to wall time
        return in_blackout(gen_wall(t_obs[s]))

    # EVENT packets (comparable with or without DTN)
    ev = [s for s, t in ptype.items() if t == 3]
    ev_bo = [s for s in ev if generated_in_blackout(s)]
    res["events"] = {"sent": len(ev), "delivered": sum(s in rx_first for s in ev), "generated_in_blackout": len(ev_bo),
                     "of_those_delivered": sum(s in rx_first for s in ev_bo)}
    if has_prio:
        names = {0: "critical", 1: "normal", 2: "low"}
        res["by_priority"] = {}
        for p, n in names.items():
            seqs = [s for s, x in prio.items() if x == p]
            res["by_priority"][n] = {"sent": len(seqs), "delivered": sum(s in rx_first for s in seqs)}
        crit = [s for s, x in prio.items() if x == 0]
        crit_bo = [s for s in crit if generated_in_blackout(s)]
        rec = []
        for s in crit_bo:
            a, b = generated_in_blackout(s)
            if s in rx_first and b is not None:
                rec.append(rx_first[s] - b)
        # hazards: was every newly announced hazard id received by the ground (by ANY obstacle packet)?
        hz = {int(k): v for k, v in log.get("critical_hazards", {}).items()}
        ids = {i: s for s, v in hz.items() for i in v}
        ids_bo = {i for i, s in ids.items() if s in t_obs and generated_in_blackout(s)}
        hz_rec = [seen_ids[i] - generated_in_blackout(ids[i])[1] for i in ids_bo
                  if i in seen_ids and generated_in_blackout(ids[i])[1] is not None and seen_ids[i] >= generated_in_blackout(ids[i])[1]]
        res["hazards"] = {"announced": len(ids), "received_at_ground": sum(i in seen_ids for i in ids),
                          "missing": sorted(i for i in ids if i not in seen_ids)[:20],
                          "announced_in_blackout": len(ids_bo), "of_those_received": sum(i in seen_ids for i in ids_bo),
                          "received_after_restore_s": {"n": len(hz_rec), "p50": q(hz_rec, 0.5), "max": round(max(hz_rec), 3) if hz_rec else None}}
        res["critical"] = {"sent": len(crit), "delivered": sum(s in rx_first for s in crit),
                           "lost": sorted(s for s in crit if s not in rx_first)[:20],
                           "generated_in_blackout": len(crit_bo), "of_those_delivered": sum(s in rx_first for s in crit_bo),
                           "recovery_after_restore_s": {"n": len(rec), "p50": q(rec, 0.5), "p95": q(rec, 0.95),
                                                        "max": round(max(rec), 3) if rec else None}}
        res["rover_dtn"] = {k: tel["dtn"][k] for k in ("offered", "acked", "expired", "evicted", "superseded", "coalesced", "retries",
                                                       "probes", "sent", "pending", "peak_entries", "peak_bytes", "rto_s", "srtt_s",
                                                       "rto_backoffs", "covered_by_full", "redundant_evicted",
                                                       "acks_received", "outages_detected", "delivery_latency_s", "link_state")}
    gm = d / "ground_map.json"
    if gm.exists():
        from artemis_x.telemetry.adapter import dedupe
        final = json.loads((tour / "mapping" / "obstacles_final.json").read_text(encoding="utf-8"))
        exp = {o["id"]: o for o in dedupe(final)[0]}
        got = {o["id"]: o for o in json.loads(gm.read_text(encoding="utf-8"))["obstacles"]}
        diff = [max(abs(got[i]["x_m"] - o["x_m"]), abs(got[i]["y_m"] - o["y_m"])) for i, o in exp.items() if i in got]
        res["ground_map"] = {"expected": len(exp), "present": sum(i in got for i in exp), "missing": sorted(set(exp) - set(got))[:10],
                             "stale_extra": len(set(got) - set(exp)), "max_position_diff_m": round(max(diff), 4) if diff else None,
                             "reconcile_stats": json.loads(gm.read_text(encoding="utf-8"))["stats"]}
    return res


def analyse_dtn(root: Path) -> dict:
    out = {"root": root.relative_to(ROOT).as_posix(), "scenarios": {}}
    for d in sorted(p for p in root.iterdir() if (p / "tour" / "summary.json").exists()):
        r = analyse_run(d)
        if (d / "link.jsonl").exists():
            r["dtn_eval"] = dtn_report(d)
        out["scenarios"][d.name] = r
    (root / "r5_4_dtn_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(analyse_dtn(Path(sys.argv[1]).resolve()), indent=1))

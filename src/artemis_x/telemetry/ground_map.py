"""R5.4 ground-side reconciliation of the rover's obstacle picture (standard library only).

Packets may arrive duplicated (retransmissions whose ACK was lost), late and out of order (retries after a blackout,
jitter). The ground picture is therefore ordered by OBSERVATION time (``t_obs``), never by arrival:
    duplicates        (session, seq) already applied -> ignored
    map reset         EVENT map_reset at t_r: every entry observed before t_r is removed; later packets observed before
                      t_r belong to the old map and are ignored
    changes           an obstacle record replaces the stored one only if its packet is not older (newest wins)
    full snapshot     records are upserted as parts arrive; when ALL parts of a full snapshot are in, entries observed
                      before it and absent from it are removed (they no longer exist on the rover) - unless a newer full
                      snapshot was already applied (then the late one only upserts where it is newer)
    pose              newest observation time wins
"""

from __future__ import annotations

from . import protocol as P


class GroundMap:
    def __init__(self, seen_max: int = 200000):
        self.obs: dict = {}                     # id -> (record, packet t_obs)
        self.pose = None                        # (t_obs, body)
        self.reset_t = float("-inf")
        self.last_full_t = float("-inf")
        self.parts: dict = {}                   # (session, t_obs) -> {"parts": n, "got": set, "ids": set}
        self.seen: set = set()
        self.seen_max = seen_max
        self.st = {"packets": 0, "duplicates_ignored": 0, "stale_ignored": 0, "pre_reset_ignored": 0, "records_applied": 0,
                   "full_snapshots_completed": 0, "full_snapshots_late": 0, "removed_by_full": 0, "resets_applied": 0,
                   "removed_by_reset": 0}

    def apply(self, pk: P.Packet) -> None:
        key = (pk.session, pk.seq)
        if key in self.seen:
            self.st["duplicates_ignored"] += 1
            return
        if len(self.seen) < self.seen_max:
            self.seen.add(key)
        self.st["packets"] += 1
        t = pk.t_obs_s
        if pk.type == P.T_POSE:
            if self.pose is None or t >= self.pose[0]:
                self.pose = (t, pk.body)
        elif pk.type == P.T_EVENT and pk.body["event"] == "map_reset":
            if t >= self.reset_t:
                self.reset_t = t
                gone = [i for i, (_, tp) in self.obs.items() if tp < t]
                for i in gone:
                    del self.obs[i]
                self.st["resets_applied"] += 1
                self.st["removed_by_reset"] += len(gone)
        elif pk.type == P.T_OBSTACLES:
            if t < self.reset_t:
                self.st["pre_reset_ignored"] += 1
                return
            for o in pk.body["obstacles"]:
                cur = self.obs.get(o["id"])
                if cur is None or cur[1] <= t:
                    self.obs[o["id"]] = (o, t)
                    self.st["records_applied"] += 1
                else:
                    self.st["stale_ignored"] += 1
            if pk.body.get("full"):
                self._full(pk, t)

    def _full(self, pk: P.Packet, t: float) -> None:
        k = (pk.session, t)
        s = self.parts.setdefault(k, {"parts": pk.body["parts"], "got": set(), "ids": set()})
        s["got"].add(pk.body["part"])
        s["ids"].update(o["id"] for o in pk.body["obstacles"])
        if len(s["got"]) < s["parts"]:
            return
        del self.parts[k]
        if t <= self.last_full_t:
            self.st["full_snapshots_late"] += 1
            return
        self.last_full_t = t
        gone = [i for i, (_, tp) in self.obs.items() if tp < t and i not in s["ids"]]
        for i in gone:
            del self.obs[i]
        self.st["full_snapshots_completed"] += 1
        self.st["removed_by_full"] += len(gone)
        for old in [x for x in self.parts if x[1] < t]:       # older incomplete full snapshots can no longer matter
            del self.parts[old]

    def snapshot(self) -> dict:
        return {"obstacles": sorted((o for o, _ in self.obs.values()), key=lambda o: o["id"]),
                "pose": None if self.pose is None else {"t_obs_s": self.pose[0], **self.pose[1]},
                "reset_t_obs_s": None if self.reset_t == float("-inf") else self.reset_t,
                "last_full_t_obs_s": None if self.last_full_t == float("-inf") else self.last_full_t, "stats": dict(self.st)}

"""Live monitoring panels for the Webots GUI (R4.5, display part only): a 2 x 2 Mission-Control layout of four Display
devices - RGB CAMERA (the camera attached as background), AI DETECTION (drawn by the inference link), DEPTH SENSOR
(colourised range image) and LOCAL MAP (the live R4 occupancy / cost / provenance layers, tracks, rover and camera FOV).

Read-only on the mapping state: nothing here changes the map, the tracker or the health monitor, and ground truth is
never used. All work runs on IDLE simulation steps (steps without a mapping cycle or grid update), in bounded row chunks
(at most one chunk per idle step), so the mapping cycle's own timing is untouched. Standard library only.

Panels are optional: a world without the ``rgb_view`` / ``depth_view`` / ``map_view`` displays gets no monitor at all
(R1-R3 worlds are unchanged), and ARTEMIS_MONITOR=0 disables it.
"""

from __future__ import annotations

import array
import math
import os
import time
from pathlib import Path

from .mapping.grid import NEVER, SRC_DEPTH, SRC_YOLO_FOOTPRINT, SRC_YOLO_POINTS
from .projection import SENSOR_FOV_RAD

W, H = 720, 480                       # every panel: same size and aspect as the 720 x 480 camera
TITLE_H = 32
# ONE font per display, set once. Webots R2025a allocates native memory in the simulator process (webots-bin) on every
# Display.setFont() that changes the font and never releases it: switching sizes a few times per frame grew webots-bin by
# > 1 GB/s until the machine ran out of memory (R4.5 memory investigation). Hierarchy uses colour and a drawn bold instead.
FONT = ("Tahoma", 15, True)
CHAR_W = 10                           # advance of FONT in display pixels, measured on screen (9.7): layout estimates
_FONT_SET: set = set()


def use_font(d) -> None:
    """Set the panel font on display ``d`` exactly once for the life of the controller."""
    if id(d) not in _FONT_SET:
        d.setFont(*FONT)
        _FONT_SET.add(id(d))


def bold_text(d, text: str, x: int, y: int) -> None:
    d.drawText(text, x, y)
    d.drawText(text, x + 1, y)

# Mission-Control palette (0xRRGGBB)
BG, BAR, RULE = 0x0B1118, 0x0E1A26, 0x22313F
TEXT, DIM, ACCENT = 0xE6EDF3, 0x8B98A5, 0x3FC1FF
OK_C, DEGR_C, STOP_C = 0x2ECC71, 0xF5A623, 0xFF4D4D
TRACK_C, LOST_C, ROVER_C = 0xFFD54A, 0x9AA4AE, 0x00E5FF

# map cell classes -> colour
C_UNKNOWN, C_FREE, C_BAND, C_NOGO, C_OCC_D, C_OCC_Y, C_OCC_DY, C_NEG, C_PROT = range(9)
CELL_RGB = {
    C_UNKNOWN: 0x161C23,     # never observed / undecided: NOT traversable, never drawn as free
    C_FREE: 0x3E5468,        # observed free ground
    C_BAND: 0x5B4A2C,        # graded inflation band (cost 1-252)
    C_NOGO: 0x8A5A1C,        # inscribed / lethal inflation (cost 253-254): rover centre may not enter
    C_OCC_D: 0xF2F4F5,       # obstacle from depth evidence only
    C_OCC_Y: 0xE040FB,       # obstacle from YOLO-protected small-rock evidence only
    C_OCC_DY: 0xFF9DEB,      # obstacle with depth AND YOLO evidence
    C_NEG: 0xFF3B3B,         # negative hazard (crater bowl)
    C_PROT: 0x5E2A68,        # protected small-rock cell, not (yet) occupied: withheld from FREE evidence
}
LEGEND = [(C_FREE, "Free (observed)"), (C_UNKNOWN, "Unknown - no-go"), (C_OCC_D, "Obstacle - depth"),
          (C_OCC_Y, "Obstacle - YOLO-protected"), (C_OCC_DY, "Obstacle - depth + YOLO"), (C_NEG, "Negative hazard"),
          (C_NOGO, "Inflation - no-go"), (C_BAND, "Inflation - cost band")]
HEALTH_C = {"OK": OK_C, "DEGRADED": DEGR_C, "STOP": STOP_C}


IMAGES = {"new": 0, "deleted": 0}      # Display image handles created / released (memory diagnostics)


def paste_image(d, data: bytes, fmt: int, w: int, h: int, x: int, y: int) -> None:
    """One Display image: create, paste, always release (the handle is native memory in the controller library)."""
    img = d.imageNew(data, fmt, w, h)
    IMAGES["new"] += 1
    try:
        d.imagePaste(img, x, y, False)
    finally:
        d.imageDelete(img)
        IMAGES["deleted"] += 1


def _rgb(c: int) -> bytes:
    return bytes(((c >> 16) & 255, (c >> 8) & 255, c & 255))


def depth_lut(max_m: float = 6.0) -> list:
    """256 colours, near -> far: white-yellow -> orange -> magenta -> blue -> dark blue (perceptually ordered)."""
    stops = [(0.0, (255, 252, 220)), (0.15, (255, 214, 64)), (0.35, (255, 120, 40)), (0.55, (214, 48, 120)),
             (0.75, (110, 50, 190)), (1.0, (24, 40, 110))]
    out = []
    for q in range(256):
        f = q / 255
        for (f0, c0), (f1, c1) in zip(stops, stops[1:]):
            if f <= f1:
                a = (f - f0) / (f1 - f0)
                out.append(bytes(int(round(c0[i] + a * (c1[i] - c0[i]))) for i in range(3)))
                break
    return out


def classify(lo: float, last: float, neg: int, src: int, cost: int, protected: bool, occ_th: float, free_th: float) -> int:
    """One cell's display class. UNKNOWN is never shown as FREE; inflation is drawn only over observed-free cells."""
    if lo > occ_th:
        if neg:
            return C_NEG
        d, y = src & SRC_DEPTH, src & (SRC_YOLO_POINTS | SRC_YOLO_FOOTPRINT)
        return C_OCC_DY if d and y else C_OCC_Y if y else C_OCC_D
    if protected:
        return C_PROT
    if lo < free_th and last != NEVER:
        return C_NOGO if cost >= 253 else C_BAND if cost > 0 else C_FREE
    return C_UNKNOWN


def _edges(n: int, size: int) -> list:
    """Integer pixel boundaries of n cells spread over ``size`` pixels (cells differ by at most one pixel)."""
    return [round(i * size / n) for i in range(n + 1)]


def _clip(x0, y0, x1, y1, r):
    """Liang-Barsky: clip a segment to rectangle r = (xmin, ymin, xmax, ymax); None if outside."""
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - r[0]), (dx, r[2] - x0), (-dy, y0 - r[1]), (dy, r[3] - y0)):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return None
            t0 = max(t0, t)
        else:
            if t < t0:
                return None
            t1 = min(t1, t)
    return x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy


def title_bar(d, title: str, right: str = "", chip: tuple | None = None) -> None:
    """Panel header: dark bar, accent stripe, title left, status right (optionally a coloured state chip)."""
    d.setAlpha(0.92)
    d.setColor(BAR)
    d.fillRectangle(0, 0, W, TITLE_H)
    d.setAlpha(1.0)
    d.setColor(ACCENT)
    d.fillRectangle(0, 0, 4, TITLE_H)
    d.setColor(RULE)
    d.drawLine(0, TITLE_H - 1, W - 1, TITLE_H - 1)
    use_font(d)
    d.setColor(TEXT)
    bold_text(d, title, 14, 8)
    x = W - 34                                                        # Webots draws the overlay's close button here
    if chip is not None:
        label, col = chip
        cw = 18 + CHAR_W * len(label)
        x -= cw
        d.setColor(col)
        d.fillRectangle(x, 5, cw, TITLE_H - 10)
        d.setColor(0x081016)
        bold_text(d, label, x + 8, 8)
        x -= 12
    if right:
        d.setColor(DIM)
        d.drawText(right, max(200, x - CHAR_W * len(right) - 4), 8)


class _Rate:
    """Events per second of simulation time over a sliding window."""

    def __init__(self, window_s: float = 4.0):
        self.window, self.samples = window_s, []

    def add(self, t: float, count: int) -> float:
        self.samples.append((t, count))
        while len(self.samples) > 2 and t - self.samples[0][0] > self.window:
            self.samples.pop(0)
        (t0, c0), (t1, c1) = self.samples[0], self.samples[-1]
        return (c1 - c0) / (t1 - t0) if t1 > t0 else 0.0


class MapRenderer:
    """LOCAL MAP panel. A frame is a consistent copy of the live layers (the same layers MapSnapshot writes), rendered in
    row chunks over several idle steps, then pasted with the vector overlays and the status / legend sidebar."""

    MAP_X, MAP_Y = 10, TITLE_H + 6
    MAP_S = H - MAP_Y - 6
    SIDE_X = MAP_X + MAP_S + 14

    def __init__(self, display, rows_per_chunk: int = 30):
        self.d = display
        self.rows_per_chunk = rows_per_chunk
        self.px = {c: _rgb(v) for c, v in CELL_RGB.items()}
        self.job = None
        self.frames = 0
        self.last = None                    # header info of the last completed frame (tests, saved stills)
        self.rate = _Rate()
        self.banner = None                  # (text, until_t)

    def begin(self, mapper, t: float) -> None:
        g, cm, tr = mapper.grid, mapper.costmap, mapper.tracker
        n = g.n
        self.job = {
            "t": t, "n": n, "res": g.res, "ox": g.origin_x, "oy": g.origin_y, "occ": g.occ_th, "free": g.free_th,
            "lo": array.array("f", g.logodds), "last": array.array("f", g.last_obs), "neg": g.neg.tobytes(),
            "src": g.src.tobytes(), "cost": cm.cost.tobytes(), "prot": frozenset(g.protected),
            "tracks": [(x.id, x.x, x.y, x.radius_m, x.confidence, x.status) for x in tr.tracks if x.status != "tentative"],
            "pose": mapper.estimator.pose(), "health": mapper.health_monitor.latest,
            "depth_stamp": mapper.last_depth_stamp, "cycles": mapper.stats["cycles"], "resets": g.resets,
            "inference": dict(mapper.inference), "next_j": n - 1, "rows": [], "edges": _edges(n, self.MAP_S), "ms": 0.0,
        }

    def step(self) -> bool:
        """Render one chunk of rows; True when the frame was completed (and pasted) by this call."""
        jb = self.job
        if jb is None:
            return False
        t0 = time.perf_counter()
        n, e, px = jb["n"], jb["edges"], self.px
        lo, last, neg, src, cost, prot = jb["lo"], jb["last"], jb["neg"], jb["src"], jb["cost"], jb["prot"]
        occ, free = jb["occ"], jb["free"]
        widths = [e[i + 1] - e[i] for i in range(n)]
        j_end = max(-1, jb["next_j"] - self.rows_per_chunk)
        for j in range(jb["next_j"], j_end, -1):                   # image top = highest odom y
            b = j * n
            row = b"".join(px[classify(lo[k], last[k], neg[k], src[k], cost[k], k in prot, occ, free)] * widths[k - b]
                           for k in range(b, b + n))
            hj = e[n - j] - e[n - 1 - j]
            jb["rows"].append(row * hj)
        jb["next_j"] = j_end
        jb["ms"] += (time.perf_counter() - t0) * 1000
        if j_end >= 0:
            return False
        self._finish(jb)
        jb["ms"] += (time.perf_counter() - t0) * 1000
        self.last, self.job = jb, None
        self.frames += 1
        return True

    # ---- geometry --------------------------------------------------------------------------------------------------
    def to_px(self, jb: dict, x: float, y: float) -> tuple:
        span = jb["n"] * jb["res"]
        return (self.MAP_X + (x - jb["ox"]) / span * self.MAP_S, self.MAP_Y + self.MAP_S - (y - jb["oy"]) / span * self.MAP_S)

    def _line(self, jb, x0, y0, x1, y1) -> None:
        r = (self.MAP_X, self.MAP_Y, self.MAP_X + self.MAP_S - 1, self.MAP_Y + self.MAP_S - 1)
        c = _clip(x0, y0, x1, y1, r)
        if c is not None:
            self.d.drawLine(*(int(round(v)) for v in c))

    # ---- one completed frame -------------------------------------------------------------------------------------
    def _finish(self, jb: dict) -> None:
        d, S = self.d, self.MAP_S
        d.setAlpha(1.0)
        d.setColor(BG)
        d.fillRectangle(0, TITLE_H, W, H - TITLE_H)
        paste_image(d, b"".join(jb["rows"]), d.RGB, S, S, self.MAP_X, self.MAP_Y)
        span, mx, my = jb["n"] * jb["res"], self.MAP_X, self.MAP_Y
        # 1 m grid ticks on the frame, coordinates of the window corners, scale bar
        d.setColor(RULE)
        d.drawRectangle(mx - 1, my - 1, S + 2, S + 2)
        d.setColor(DIM)
        use_font(d)
        first = math.ceil(jb["ox"])
        for m in range(first, int(jb["ox"] + span) + 1):
            x = int(round(mx + (m - jb["ox"]) / span * S))
            d.drawLine(x, my + S, x, my + S + (5 if m % 2 == 0 else 3))
        for m in range(math.ceil(jb["oy"]), int(jb["oy"] + span) + 1):
            y = int(round(my + S - (m - jb["oy"]) / span * S))
            d.drawLine(mx - (5 if m % 2 == 0 else 3), y, mx, y)
        d.setColor(BG)
        d.fillRectangle(mx + 4, my + S - 42, 150, 38)
        d.setColor(TEXT)
        k1 = int(round(1.0 / span * S))
        d.fillRectangle(mx + 10, my + S - 14, k1, 3)
        d.drawText("1 m", mx + 14 + k1, my + S - 21)
        d.setColor(DIM)
        d.drawText(f"x {jb['ox']:+.1f}  y {jb['oy']:+.1f}", mx + 8, my + S - 38)
        d.setColor(BG)
        d.fillRectangle(mx + S - 146, my + 4, 142, 19)
        d.setColor(DIM)
        d.drawText(f"x {jb['ox'] + span:+.1f}  y {jb['oy'] + span:+.1f}", mx + S - 142, my + 7)
        # tracked rocks (persistent ids, confidence)
        p = jb["pose"]
        taken = []                                                    # label boxes already drawn (no overlapping labels)
        for tid, x, y, r, conf, status in sorted(jb["tracks"], key=lambda z: math.hypot(z[1] - p.x, z[2] - p.y)):
            if not (jb["ox"] <= x < jb["ox"] + span and jb["oy"] <= y < jb["oy"] + span):
                continue
            cx, cy = self.to_px(jb, x, y)
            rp = max(4, int(round(r / span * S)) + 2)
            d.setColor(TRACK_C if status == "confirmed" else LOST_C)
            d.drawOval(int(cx), int(cy), rp, rp)
            if status != "confirmed":
                continue
            lbl = f"{tid}:{conf:.2f}"
            lw, lh = CHAR_W * len(lbl) + 2, 15
            lx = int(cx) + rp + 2
            if lx + lw > mx + S:                                    # would run into the sidebar: put it on the left
                lx = int(cx) - rp - 2 - lw
            ly = int(cy) - 6
            box = (lx, ly, lx + lw, ly + lh)
            if lx < mx or any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in taken):
                continue
            taken.append(box)
            d.drawText(lbl, lx, ly)
        # camera field of view (mapping range) and the rover (estimated odometry pose)
        rx, ry = self.to_px(jb, p.x, p.y)
        rng = 6.0
        ends = [self.to_px(jb, p.x + rng * math.cos(p.yaw + s * SENSOR_FOV_RAD / 2), p.y + rng * math.sin(p.yaw + s * SENSOR_FOV_RAD / 2))
                for s in (-1, 1)]
        d.setColor(ROVER_C)
        d.setAlpha(0.75)
        for ex, ey in ends:
            self._line(jb, rx, ry, ex, ey)
        self._line(jb, ends[0][0], ends[0][1], ends[1][0], ends[1][1])
        d.setAlpha(1.0)
        foot = max(3, int(round(0.21 / span * S)))
        d.drawOval(int(rx), int(ry), foot, foot)
        hx, hy = math.cos(p.yaw), -math.sin(p.yaw)                  # screen y points down
        tip = (rx + hx * (foot + 9), ry + hy * (foot + 9))
        left = (rx - hy * 5, ry + hx * 5)
        right = (rx + hy * 5, ry - hx * 5)
        d.fillPolygon([int(tip[0]), int(left[0]), int(right[0])], [int(tip[1]), int(left[1]), int(right[1])])
        self._sidebar(jb)
        if self.banner and jb["t"] <= self.banner[1]:
            d.setColor(STOP_C)
            d.fillRectangle(mx + S // 2 - 110, my + 10, 220, 26)
            d.setColor(0xFFFFFF)
            d.drawText(self.banner[0], mx + S // 2 - 100, my + 15)

    def _sidebar(self, jb: dict) -> None:
        d, x = self.d, self.SIDE_X
        h = jb["health"]
        state = h.state if h is not None else "STARTING"
        rate = self.rate.add(jb["t"], jb["cycles"])
        age = (jb["t"] - jb["depth_stamp"]) if jb["depth_stamp"] is not None else None
        fresh = DIM if age is None else OK_C if age <= 0.6 else DEGR_C if age <= 1.0 else STOP_C
        title_bar(d, "LOCAL MAP", f"{jb['n'] * jb['res']:.0f}x{jb['n'] * jb['res']:.0f} m  {jb['res']:.2f} m/cell",
                  (state, HEALTH_C.get(state, DIM)))
        y = TITLE_H + 6
        p = jb["pose"]
        rows = [("ROVER (odometry estimate)", None),
                (f"x {p.x:+6.2f}  y {p.y:+6.2f} m", TEXT),
                (f"heading {math.degrees(p.yaw):+6.1f} deg", TEXT),
                ("MAPPING", None),
                (f"{rate:3.1f} Hz  cycle {jb['cycles']}", TEXT),
                (f"updated t {jb['t']:7.2f} s", TEXT),
                (f"depth age {age:5.2f} s" if age is not None else "depth  waiting", fresh)]
        reasons = [r for r in (h.reasons if h is not None else ()) if r != "inference_disabled"]
        inf = jb["inference"]
        ai = ("YOLO off - depth only" if not inf.get("enabled") else
              "YOLO online" if inf.get("connected") and "inference_unavailable" not in reasons else "YOLO OFFLINE - depth only")
        conf = sum(1 for z in jb["tracks"] if z[5] == "confirmed")
        rows += [("TRACKS / AI", None), (f"{conf} confirmed  {len(jb['tracks']) - conf} lost", TEXT),
                 (ai, OK_C if ai == "YOLO online" else DEGR_C)]
        if reasons:
            rows += [("HEALTH: " + reasons[0].replace("_", " ")[:20], HEALTH_C.get(state, DIM))]
        for text, col in rows:
            if col is None:
                y += 3
                d.setColor(ACCENT)
                bold_text(d, text, x, y)
                y += 18
            else:
                d.setColor(col)
                d.drawText(text, x, y)
                y += 18
        rh = 18
        y = max(y + 8, H - 6 - rh * (len(LEGEND) + 2))
        d.setColor(RULE)
        d.drawLine(x, y - 5, W - 10, y - 5)
        for c, label in LEGEND:
            d.setColor(CELL_RGB[c])
            d.fillRectangle(x, y + 2, 13, 12)
            d.setColor(RULE)
            d.drawRectangle(x, y + 2, 13, 12)
            d.setColor(TEXT)
            d.drawText(label, x + 20, y)
            y += rh
        d.setColor(TRACK_C)
        d.drawOval(x + 6, y + 8, 5, 5)
        d.setColor(TEXT)
        d.drawText("Track id:conf (grey = lost)", x + 20, y)
        y += rh
        d.setColor(ROVER_C)
        d.fillPolygon([x + 13, x + 1, x + 1], [y + 8, y + 3, y + 13])
        d.setColor(TEXT)
        d.drawText("Rover + camera FOV, 6 m", x + 20, y)


class DepthRenderer:
    """DEPTH SENSOR panel: the mapper's own depth frame, subsampled 3x, colourised 0-6 m (the mapping range), invalid black."""

    STRIDE = 3

    def __init__(self, display, rows_per_chunk: int = 40):
        self.d = display
        self.rows_per_chunk = rows_per_chunk
        s = self.STRIDE
        self.lut = [c * s for c in depth_lut()]
        self.far = _rgb(0x0E1420) * s                              # beyond the 6 m mapping range
        self.bad = b"\x00\x00\x00" * s                             # invalid / no return
        self.job = None
        self.frames = 0

    def begin(self, depth, stamp: float) -> None:
        self.job = {"depth": depth, "stamp": stamp, "v": 0, "rows": [], "ms": 0.0}

    def step(self) -> bool:
        jb = self.job
        if jb is None:
            return False
        t0 = time.perf_counter()
        s, dep, lut, far, bad = self.STRIDE, jb["depth"], self.lut, self.far, self.bad
        k = 255 / 6.0
        v_end = min(H, jb["v"] + self.rows_per_chunk * s)
        for v in range(jb["v"], v_end, s):
            seg = dep[v * W:(v + 1) * W:s]
            row = b"".join([lut[int(z * k)] if z < 6.0 else (far if z < 1e30 else bad) for z in seg])
            jb["rows"].append(row * s)
        jb["v"] = v_end
        jb["ms"] += (time.perf_counter() - t0) * 1000
        if v_end < H:
            return False
        d = self.d
        paste_image(d, b"".join(jb["rows"]), d.RGB, W, H, 0, 0)
        title_bar(d, "DEPTH SENSOR", f"t {jb['stamp']:.2f} s")
        self._colourbar()
        jb["ms"] += (time.perf_counter() - t0) * 1000
        self.last, self.job = jb, None
        self.frames += 1
        return True

    def _colourbar(self) -> None:
        d = self.d
        x0, y0, bw = W - 230, H - 30, 200
        d.setAlpha(0.85)
        d.setColor(BAR)
        d.fillRectangle(x0 - 10, y0 - 8, bw + 20, 34)
        d.setAlpha(1.0)
        lut = depth_lut()
        for i in range(bw):
            c = lut[int(i * 255 / (bw - 1))]
            d.setColor((c[0] << 16) | (c[1] << 8) | c[2])
            d.drawLine(x0 + i, y0, x0 + i, y0 + 8)
        use_font(d)
        d.setColor(TEXT)
        for m in (0, 2, 4, 6):
            d.drawText(f"{m}m", x0 + int(m / 6 * (bw - 1)) - (0 if m == 0 else 10), y0 + 11)


class Monitor:
    """The four panels plus their idle-step scheduler. ``tick`` does at most one bounded chunk of work per call."""

    def __init__(self, robot, sensors, mapper, period_s: float = 0.5, save_dir: Path | None = None, save_at: tuple = ()):
        from .rover import find_device
        self.mapper, self.sensors = mapper, sensors
        on = set(os.environ.get("ARTEMIS_MONITOR_PANELS", "rgb,ai,depth,map").split(","))   # diagnostics: panel subsets
        self.rgb = find_device(robot, "rgb_view") if "rgb" in on else None
        self.ai = find_device(robot, "yolo_view") if "ai" in on else None
        self.depth_d = find_device(robot, "depth_view") if "depth" in on else None
        self.map_d = find_device(robot, "map_view") if "map" in on else None
        self.map = MapRenderer(self.map_d) if self.map_d is not None else None
        self.depth = DepthRenderer(self.depth_d) if self.depth_d is not None else None
        self.period = period_s
        self.next_map = self.next_depth = self.next_hdr = 0.0
        self._resets = mapper.grid.resets
        self._depth_src = None
        self._last_begin = None
        self.save_dir, self.save_at = save_dir, sorted(save_at)
        self.saved = []
        self.stats = {"tick_ms": [], "map_frame_ms": [], "depth_frame_ms": [], "map_age_s": [], "map_frames": 0, "depth_frames": 0}
        if self.rgb is not None and sensors.camera is not None:
            self.rgb.setAlpha(0.0)
            self.rgb.fillRectangle(0, 0, W, H)                    # transparent: the attached camera shows through
            self.rgb.attachCamera(sensors.camera)
        for d, title in ((self.ai, "AI DETECTION"), (self.depth_d, "DEPTH SENSOR"), (self.map_d, "LOCAL MAP")):
            if d is not None:
                d.setAlpha(1.0)
                d.setColor(BG)
                d.fillRectangle(0, 0, W, H)
                title_bar(d, title, "waiting for data")

    @classmethod
    def create(cls, robot, sensors, mapper):
        if mapper is None or os.environ.get("ARTEMIS_MONITOR", "1") == "0":
            return None
        from .rover import find_device
        if find_device(robot, "map_view") is None:
            return None
        save = os.environ.get("ARTEMIS_MONITOR_SAVE")
        at = tuple(float(v) for v in os.environ.get("ARTEMIS_MONITOR_SAVE_AT", "").split(",") if v.strip())
        return cls(robot, sensors, mapper, save_dir=Path(save) if save else None, save_at=at)

    def _map_due(self, t: float) -> bool:
        return self.map is not None and t >= self.next_map

    def _depth_due(self, t: float) -> bool:
        return self.depth is not None and self._depth_src is not None and t >= self.next_depth

    def counters(self) -> dict:
        """Memory-trace counters: frames drawn, Display images created / deleted, pending jobs."""
        return {"map_frames": self.stats["map_frames"], "depth_frames": self.stats["depth_frames"],
                "images_new": IMAGES["new"], "images_deleted": IMAGES["deleted"],
                "map_job": self.map is not None and self.map.job is not None,
                "depth_job": self.depth is not None and self.depth.job is not None}

    def on_frame(self, frame) -> None:
        """The mapper's captured frame (its depth is what the map was built from)."""
        if frame is not None and frame.depth is not None:
            self._depth_src = (frame.depth, frame.sim_time_s)

    def tick(self, t: float) -> None:
        """Called on idle steps only. Priority: a pending chunk, then a due new frame, then headers / saves."""
        t0 = time.perf_counter()
        m = self.mapper
        if m.grid.resets != self._resets:                         # explicit map reset: drop the stale frame, redraw now
            self._resets = m.grid.resets
            if self.map is not None:
                self.map.job = None
                self.map.banner = ("MAP RESET", t + 5.0)
                self.next_map = t
        if self.map is not None and self.map.job is not None:
            if self.map.step():
                self.stats["map_frame_ms"].append(self.map.last["ms"])
                self.stats["map_frames"] += 1
                self.stats["map_age_s"].append(t - self.map.last["t"])   # data capture -> on screen (simulation time)
        elif self.depth is not None and self.depth.job is not None:
            if self.depth.step():
                self.stats["depth_frame_ms"].append(self.depth.last["ms"])
                self.stats["depth_frames"] += 1
        elif self._map_due(t) and (not self._depth_due(t) or self._last_begin != "map"):   # both due: alternate
            self.map.begin(m, t)
            self.next_map = t + self.period
            self._last_begin = "map"
        elif self._depth_due(t):
            self.depth.begin(*self._depth_src)
            self.next_depth = t + self.period
            self._last_begin = "depth"
        elif t >= self.next_hdr:
            self._headers(t)
            self.next_hdr = t + self.period
        elif self.save_dir is not None and self.save_at and t >= self.save_at[0]:
            self.save(t, f"t{int(self.save_at.pop(0)):04d}")
        self.stats["tick_ms"].append((time.perf_counter() - t0) * 1000)

    def _headers(self, t: float) -> None:
        if self.rgb is not None:
            title_bar(self.rgb, "RGB CAMERA", f"FOV {math.degrees(SENSOR_FOV_RAD):.0f} deg  t {t:.2f} s")
        inf = self.mapper.inference
        h = self.mapper.health_monitor.latest
        down = not inf.get("enabled") or not inf.get("connected") or (h is not None and "inference_unavailable" in h.reasons)
        if self.ai is not None and down:
            d = self.ai
            title_bar(d, "AI DETECTION", "YOLO11n (frozen)", ("OFFLINE", DEGR_C))
            d.setAlpha(0.9)
            d.setColor(BAR)
            d.fillRectangle(W // 2 - 190, H // 2 - 30, 380, 60)
            d.setAlpha(1.0)
            d.setColor(DEGR_C)
            bold_text(d, "YOLO unavailable", W // 2 - 175, H // 2 - 22)
            d.setColor(TEXT)
            d.drawText("depth mapping and safety continue", W // 2 - 175, H // 2 + 2)

    def save(self, t: float, tag: str) -> None:
        """Stills of the panels (mission-stage evidence); the RGB panel's camera background is saved by the camera."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        for name, d in (("ai", self.ai), ("depth", self.depth_d), ("map", self.map_d)):
            if d is not None:
                p = self.save_dir / f"{tag}_{name}.png"
                d.imageSave(None, str(p))
                self.saved.append(p.name)
        if self.sensors.camera is not None:
            p = self.save_dir / f"{tag}_rgb.png"
            self.sensors.camera.saveImage(str(p), 95)
            self.saved.append(p.name)

    def finish(self) -> dict:
        def q(v, f):
            s = sorted(v)
            return round(s[min(len(s) - 1, int(f * (len(s) - 1)))], 3) if s else None
        st = self.stats
        return {"map_frames": st["map_frames"], "depth_frames": st["depth_frames"],
                "tick_ms": {"p50": q(st["tick_ms"], 0.5), "p95": q(st["tick_ms"], 0.95), "max": q(st["tick_ms"], 1.0)},
                "map_frame_ms": {"p50": q(st["map_frame_ms"], 0.5), "p95": q(st["map_frame_ms"], 0.95)},
                "depth_frame_ms": {"p50": q(st["depth_frame_ms"], 0.5), "p95": q(st["depth_frame_ms"], 0.95)},
                "map_on_screen_age_s": {"p50": q(st["map_age_s"], 0.5), "p95": q(st["map_age_s"], 0.95), "max": q(st["map_age_s"], 1.0)},
                "map_resets_seen": self._resets,
                "saved": self.saved}

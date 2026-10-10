"""Generate Artemis-X Webots worlds from the official Webots R2025a `moon.wbt` sample (read-only on the installation).

The lunar ground mesh, textures, background and every EXTERNPROTO URL are copied verbatim. The two sample Koalas are
replaced by ONE Artemis-X rover (deterministic pose, Python controller, supervisor access for ground truth, forward
RGB camera + aligned RangeFinder in the PROTO's extensionSlot).

Worlds:
  worlds/artemis_moon.wbt            R1/R2 baseline (3 sphere test rocks, original lighting) - unchanged by R3
  worlds/artemis_moon_enhanced.wbt   R3 demo variant: seed-0 field of irregular procedural rocks + low-angle sun
  worlds/eval_s<seed>_<variant>.wbt  R3 evaluation worlds (git-ignored), with a JSON scene manifest each
  worlds/artemis_r4_monitor.wbt      R4.5 display variant of the test course (low sun): same scene, rover and tour, plus
                                     the rgb_view / depth_view / map_view displays of the 2 x 2 monitoring layout

Baseline vs enhanced evaluation worlds are PAIRED: same seed -> same rock centres, sizes and camera viewpoints; only rock
geometry/material and lighting differ. Rock meshes are generated here (no external assets); the rock texture is the
Webots moon-ground texture already used by the sample world.

Usage:  python artemis-webots/tools/make_world.py                   # R1/R2 + enhanced demo world
        python artemis-webots/tools/make_world.py --eval 0 1 2 3 4   # evaluation worlds + manifests
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "src"))
from artemis_x.sim.projection import MOUNT_PITCH_RAD, MOUNT_TRANSLATION, SENSOR_FOV_RAD, SENSOR_HEIGHT, SENSOR_WIDTH  # noqa: E402

WORLDS = HERE / "worlds"
OUT = WORLDS / "artemis_moon.wbt"
SAMPLE = Path("projects/samples/demos/worlds/moon.wbt")
MOON_TEXTURE = "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/samples/demos/worlds/textures/moon/moon_ground.jpg"

# Sensor geometry shared by the camera and the range finder (identical pose, FOV and resolution -> pixel-aligned)
SENSOR = {"translation": " ".join(f"{v:g}" for v in MOUNT_TRANSLATION), "rotation": f"0 1 0 {MOUNT_PITCH_RAD:g}",
          "fov": SENSOR_FOV_RAD, "width": SENSOR_WIDTH, "height": SENSOR_HEIGHT}


def rover(recognition: bool = False, pose: tuple = (0, 0, 0), custom_data: str = "", live_view: bool = False,
          monitor: bool = False) -> str:
    names = (["yolo_view"] if live_view else []) + (["rgb_view", "depth_view", "map_view"] if monitor else [])
    view = "".join(f"\n    Display {{\n      name \"{n}\"\n      width {SENSOR['width']}\n      height {SENSOR['height']}\n    }}"
                   for n in names)
    cd = f'  customData "{custom_data}"\n' if custom_data else ""
    rec = "\n      recognition Recognition {\n        maxRange 20\n        frameThickness 0\n        segmentation TRUE\n      }" if recognition else ""
    return f"""DEF ARTEMIS_ROVER Koala {{
  translation {pose[0]:g} {pose[1]:g} 0
  rotation 0 0 1 {pose[2]:g}
  name "artemis_rover"
  controller "artemis_rover"
  supervisor TRUE
{cd}  extensionSlot [
    Camera {{
      translation {SENSOR['translation']}
      rotation {SENSOR['rotation']}
      name "front_camera"
      fieldOfView {SENSOR['fov']}
      width {SENSOR['width']}
      height {SENSOR['height']}
      near 0.05{rec}
    }}
    RangeFinder {{
      translation {SENSOR['translation']}
      rotation {SENSOR['rotation']}
      name "front_depth"
      fieldOfView {SENSOR['fov']}
      width {SENSOR['width']}
      height {SENSOR['height']}
      near 0.05
      minRange 0.05
      maxRange 15
    }}{view}
  ]
}}
"""


ROVER = rover()


def rock(name: str, x: float, y: float, r: float) -> str:
    return f"""DEF {name.upper()} Solid {{
  translation {x} {y} {r * 0.6:.3f}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 0.55 0.53 0.5
        roughness 1
        metalness 0
      }}
      geometry Sphere {{
        radius {r}
        subdivision 2
      }}
    }}
  ]
  name "{name}"
  boundingObject Sphere {{
    radius {r}
  }}
}}
"""


ROCKS = [("test_rock_1", 1.6, 0.35, 0.15), ("test_rock_2", 2.3, -0.45, 0.25), ("test_rock_3", 1.2, 1.4, 0.2)]


# ---- R3: seeded rock fields, procedural meshes, lighting ---------------------------------------------------------------
def rock_field(seed: int, n: int = 10) -> list[dict]:
    rng = random.Random(1000 + seed)
    rocks = []
    while len(rocks) < n:
        r = math.exp(rng.uniform(math.log(0.08), math.log(0.45)))
        x, y = rng.uniform(1.0, 7.0), rng.uniform(-3.5, 3.5)
        if all(math.hypot(x - o["x"], y - o["y"]) > r + o["r"] + 0.25 for o in rocks):
            rocks.append({"name": f"rock_{len(rocks):02d}", "x": round(x, 3), "y": round(y, 3), "r": round(r, 3),
                          "color": [(16 * (len(rocks) + 1)) / 255, 200 / 255, 50 / 255]})
    return rocks


def viewpoints(seed: int) -> list[dict]:
    rng = random.Random(2000 + seed)
    base = [(0.0, 0.0, 0.0), (0.0, -1.2, 0.3), (0.0, 1.2, -0.3), (1.2, 0.0, 0.0)]
    return [{"id": i, "x": round(x + rng.uniform(-0.3, 0.3), 3), "y": round(y + rng.uniform(-0.3, 0.3), 3),
             "yaw": round(a + rng.uniform(-0.1, 0.1), 4)} for i, (x, y, a) in enumerate(base)]


def _icosphere(subdiv: int = 2):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    v = [tuple(c / math.sqrt(sum(k * k for k in p)) for c in p) for p in v]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(subdiv):
        cache, nf = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                p = [(v[a][i] + v[b][i]) / 2 for i in range(3)]
                n = math.sqrt(sum(c * c for c in p))
                v.append(tuple(c / n for c in p)); cache[k] = len(v) - 1
            return cache[k]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return v, f


def rock_mesh(rk: dict, seed: int) -> dict:
    """Irregular rock: icosphere with seeded anisotropic scaling and smooth radial bumps (world coordinates)."""
    rng = random.Random(3000 + seed * 100 + int(rk["name"][-2:]))
    v, f = _icosphere(2)
    sx, sy, sz = rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2), rng.uniform(0.55, 0.8)
    yaw = rng.uniform(0, 2 * math.pi)
    bumps = [(tuple(rng.gauss(0, 1) for _ in range(3)), rng.uniform(-0.18, 0.25), rng.uniform(2, 6)) for _ in range(6)]
    bumps = [(tuple(c / math.sqrt(sum(k * k for k in d)) for c in d), a, p) for d, a, p in bumps]
    cz = rk["r"] * 0.6 * sz / 0.8                     # embed the base like the paired sphere
    out = []
    for p in v:
        s = 1 + sum(a * max(0.0, sum(p[i] * d[i] for i in range(3))) ** k for d, a, k in bumps)
        x, y, z = p[0] * sx * s * rk["r"], p[1] * sy * s * rk["r"], p[2] * sz * s * rk["r"]
        c, si = math.cos(yaw), math.sin(yaw)
        out.append((round(rk["x"] + c * x - si * y, 5), round(rk["y"] + si * x + c * y, 5), round(cz + z, 5)))
    return {"vertices": out, "triangles": [list(t) for t in f]}


def sphere_node(rk: dict, recognition: bool) -> str:
    rc = f"\n  recognitionColors [ {rk['color'][0]:.6f} {rk['color'][1]:.6f} {rk['color'][2]:.6f} ]" if recognition else ""
    return f"""DEF {rk['name'].upper()} Solid {{
  translation {rk['x']} {rk['y']} {rk['r'] * 0.6:.3f}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 0.55 0.53 0.5
        roughness 1
        metalness 0
      }}
      geometry Sphere {{
        radius {rk['r']}
        subdivision 2
      }}
    }}
  ]
  name "{rk['name']}"{rc}
}}
"""


def mesh_node(rk: dict, mesh: dict, recognition: bool) -> str:
    pts = ", ".join(f"{x} {y} {z}" for x, y, z in mesh["vertices"])
    idx = ", ".join(f"{a}, {b}, {c}, -1" for a, b, c in mesh["triangles"])
    rc = f"\n  recognitionColors [ {rk['color'][0]:.6f} {rk['color'][1]:.6f} {rk['color'][2]:.6f} ]" if recognition else ""
    return f"""DEF {rk['name'].upper()} Solid {{
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 0.75 0.73 0.7
        baseColorMap ImageTexture {{
          url [
            "{MOON_TEXTURE}"
          ]
        }}
        roughness 1
        metalness 0
      }}
      geometry IndexedFaceSet {{
        coord Coordinate {{
          point [ {pts} ]
        }}
        coordIndex [ {idx} ]
        creaseAngle 0.9
      }}
    }}
  ]
  name "{rk['name']}"{rc}
}}
"""


def sample_head(webots_home: str, title: str, info: str) -> str:
    src = (Path(webots_home) / SAMPLE).read_text(encoding="utf-8")
    head, sep, _ = src.partition("\nKoala {")
    if not sep:
        raise SystemExit("unexpected moon.wbt layout: no Koala node found")
    head = head.replace('title "Koala Robot"', f'title "{title}"')
    head = re.sub(r"WorldInfo \{\n  info \[\n.*?\n  \]", 'WorldInfo {\n  info [\n' + info + '\n  ]', head, count=1, flags=re.S)
    head = head.replace("  gravity 1.62\n", "  basicTimeStep 32\n  gravity 1.62\n")
    head = re.sub(r"Viewpoint \{\n.*?\n\}", "Viewpoint {\n  orientation -0.140722 0.539298 0.830274 0.609005\n"
                  "  position -1.5 -1.0 1.0\n  follow \"artemis_rover\"\n}", head, count=1, flags=re.S)
    return head.rstrip("\n") + "\n"


def lunar_lighting(head: str, direction: str = "0.62 -0.72 -0.34") -> str:
    """Enhanced variant: dim the office-HDR fill light and add a low-elevation (~20 deg) sun with hard shadows."""
    head = head.replace('TexturedBackgroundLight {\n  texture "empty_office"\n}',
                        'TexturedBackgroundLight {\n  texture "empty_office"\n  luminosity 0.15\n}')
    sun = f"DirectionalLight {{\n  ambientIntensity 0\n  direction {direction}\n  intensity 2.6\n  castShadows TRUE\n}}\n"
    return head + sun


# ---- Poly Haven assets (CC0; fetched by tools/fetch_polyhaven.py, git-ignored) ------------------------------------------
ASSETS = HERE / "assets" / "polyhaven"
PH_ROCKS = ["moon_rock_01", "moon_rock_02", "moon_rock_03"]
_PH_CACHE: dict = {}


def polyhaven_available() -> bool:
    return all((ASSETS / a / f"{a}.obj").exists() for a in PH_ROCKS) and (ASSETS / "moon_01" / "moon_01_diff_2k.jpg").exists()


def ph_mesh(asset: str) -> dict:
    if asset not in _PH_CACHE:
        _PH_CACHE[asset] = json.loads((ASSETS / asset / f"{asset}_mesh.json").read_text(encoding="utf-8"))
    return _PH_CACHE[asset]


def ph_place(asset: str, x: float, y: float, yaw: float, scale: float, z0: float = 0.0, sink_frac: float = 0.03) -> dict:
    """World-frame vertices of a placed Poly Haven rock (same transform as the Webots node -> exact ground truth)."""
    m = ph_mesh(asset)
    v = [[a * scale for a in p] for p in m["vertices"]]
    sink = sink_frac * max(p[2] for p in v)               # bed the rock into the regolith (default 3 % of its height)
    c, si = math.cos(yaw), math.sin(yaw)
    out = [[round(x + c * p[0] - si * p[1], 5), round(y + si * p[0] + c * p[1], 5), round(z0 + p[2] - sink, 5)] for p in v]
    return {"vertices": out, "triangles": m["triangles"], "sink": sink}


def _rc(color) -> str:
    return f"\n  recognitionColors [ {color[0]:.6f} {color[1]:.6f} {color[2]:.6f} ]" if color else ""


def ph_rock_node(name: str, asset: str, x: float, y: float, yaw: float, scale: float, color=None, z0: float = 0.0,
                 sink_frac: float = 0.03) -> str:
    sink = ph_place(asset, x, y, yaw, scale, z0, sink_frac)["sink"]
    tex = f"../assets/polyhaven/{asset}/textures/{asset}"
    diff = f"{tex}_diff_1k_lunar.jpg" if (ASSETS / asset / "textures" / f"{asset}_diff_1k_lunar.jpg").exists() else f"{tex}_diff_1k.jpg"
    return f"""DEF {name.upper()} Solid {{
  translation {x} {y} {z0 - sink:.5f}
  rotation 0 0 1 {yaw:.5f}
  children [
    Transform {{
      scale {scale:.5f} {scale:.5f} {scale:.5f}
      children [
        Shape {{
          appearance PBRAppearance {{
            baseColorMap ImageTexture {{
              url [ "{diff}" ]
            }}
            normalMap ImageTexture {{
              url [ "{tex}_nor_gl_1k.jpg" ]
            }}
            roughness 1
            metalness 0
          }}
          geometry Mesh {{
            url [ "../assets/polyhaven/{asset}/{asset}.obj" ]
          }}
        }}
      ]
    }}
  ]
  name "{name}"{_rc(color)}
}}
"""


def regolith_ground(cx: float, cy: float, sx: float, sy: float) -> str:
    """Flat Poly Haven moon_01 regolith patch (3.1 m texture tile) laid over the sample terrain around the test area."""
    t = "../assets/polyhaven/moon_01/moon_01"
    return f"""DEF REGOLITH Solid {{
  translation {cx} {cy} 0.002
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColorMap ImageTexture {{
          url [ "{t}_diff_2k.jpg" ]
        }}
        normalMap ImageTexture {{
          url [ "{t}_nor_gl_2k.jpg" ]
        }}
        roughnessMap ImageTexture {{
          url [ "{t}_rough_1k.jpg" ]
        }}
        metalness 0
        textureTransform TextureTransform {{
          scale {sx / 3.1:.4f} {sy / 3.1:.4f}
        }}
      }}
      geometry Plane {{
        size {sx} {sy}
      }}
    }}
  ]
  name "regolith_patch"
}}
"""


def distractor_node(d: dict) -> str:
    """Non-rock objects (false-positive probes): simple engineered shapes with metallic / painted materials."""
    geo = {"box": f"Box {{\n        size {d.get('size', [0, 0, 0])[0]} {d.get('size', [0, 0, 0])[1]} {d.get('size', [0, 0, 0])[2]}\n      }}",
           "cylinder": f"Cylinder {{\n        height {d.get('height')}\n        radius {d.get('radius')}\n      }}",
           "sphere": f"Sphere {{\n        radius {d.get('radius')}\n        subdivision 3\n      }}"}[d["shape"]]
    r = d.get("rotation", [0, 0, 1, 0])
    return f"""DEF {d['name'].upper()} Solid {{
  translation {d['x']} {d['y']} {d['z']}
  rotation {r[0]} {r[1]} {r[2]} {r[3]}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor {d['rgb'][0]} {d['rgb'][1]} {d['rgb'][2]}
        roughness {d.get('roughness', 0.4)}
        metalness {d.get('metalness', 0.6)}
      }}
      geometry {geo}
    }}
  ]
  name "{d['name']}"{_rc(d['color'])}
}}
"""


def _camera_meta() -> dict:
    return {"mount_translation": MOUNT_TRANSLATION, "mount_pitch_rad": MOUNT_PITCH_RAD, "fov_rad": SENSOR_FOV_RAD,
            "width": SENSOR_WIDTH, "height": SENSOR_HEIGHT}


def build_world(webots_home: str, variant: str, seed: int, recognition: bool, pose: tuple, enhanced_style: str = "polyhaven") -> tuple[str, dict]:
    """Paired evaluation world. variant 'baseline' = spheres + sample lighting; 'enhanced' = Poly Haven photogrammetry rocks
    (fallback: procedural meshes) + regolith patch + low-angle sun. Same rock centres / footprint sizes in both."""
    rocks = rock_field(seed)
    style = enhanced_style if variant == "enhanced" else "spheres"
    if style == "polyhaven" and not polyhaven_available():
        style = "procedural"                               # documented fallback when assets are not fetched
    info = (f'    "Artemis-X R3 {variant} scene, seed {seed} ({style}): paired rock field (same centres/sizes in both variants)."\n'
            '    "Generated from the official Webots R2025a moon.wbt sample by artemis-webots/tools/make_world.py."')
    head = sample_head(webots_home, f"Artemis-X R3 {variant} s{seed}", info)
    manifest = {"seed": seed, "variant": variant, "style": style, "rocks": [], "distractors": [], "viewpoints": viewpoints(seed),
                "lighting": "sample default (office HDR light)" if variant == "baseline" else "low-angle sun (~20 deg), HDR fill 0.15, shadows",
                "rock_geometry": {"spheres": "sphere (Webots Sphere, subdivision 2)", "procedural": "procedural irregular mesh (icosphere + bumps)",
                                  "polyhaven": "Poly Haven CC0 photogrammetry rocks (moon_rock_01/02/03, 1K), scaled to the paired footprint"}[style],
                "camera": _camera_meta()}
    nodes = []
    rng = random.Random(4000 + seed)
    if style == "polyhaven":
        nodes.append(regolith_ground(4.0, 0.0, 14.0, 10.0))
    for rk in rocks:
        if style == "spheres":
            nodes.append(sphere_node(rk, recognition))
            manifest["rocks"].append({**rk, "shape": "sphere", "center": [rk["x"], rk["y"], round(rk["r"] * 0.6, 4)]})
        elif style == "procedural":
            mesh = rock_mesh(rk, seed)
            nodes.append(mesh_node(rk, mesh, recognition))
            manifest["rocks"].append({**rk, "shape": "mesh", "mesh": mesh})
        else:
            asset = PH_ROCKS[rng.randrange(3)]
            ext = ph_mesh(asset)["extent_m"]
            scale = 2 * rk["r"] / max(ext[0], ext[1])
            yaw = rng.uniform(0, 2 * math.pi)
            nodes.append(ph_rock_node(rk["name"], asset, rk["x"], rk["y"], yaw, scale, rk["color"] if recognition else None))
            manifest["rocks"].append({**rk, "shape": "mesh", "asset": asset, "scale": round(scale, 5), "yaw": round(yaw, 5),
                                      "mesh": {k: v for k, v in ph_place(asset, rk["x"], rk["y"], yaw, scale).items() if k != "sink"}})
    if variant == "enhanced":
        head = lunar_lighting(head)
    return head + rover(recognition, pose) + "".join(nodes), manifest


# ---- curated test course v2 ------------------------------------------------------------------------------------------------
LANE_Y = -2.6                                          # clear driving lane for the scripted tour (no object within 0.6 m)
SUN = {"lowsun": ("0.62 -0.72 -0.34", "low sun, ~20 deg elevation"), "highsun": ("0.373 -0.436 -0.819", "high sun, ~55 deg elevation")}


def _col(i: int, rock: bool) -> list:
    return [(6 * i + 6) / 255, (200 if rock else 90) / 255, (50 if rock else 160) / 255]


CRATERS = [{"name": "crater_large", "x": 25.0, "y": 0.4, "radius": 1.2, "depth": 0.25, "rim": 0.10},
           {"name": "crater_small_empty", "x": 23.9, "y": -1.25, "radius": 0.55, "depth": 0.12, "rim": 0.05}]


TERRAIN_CORE = (-3.0, 40.0, -4.5, 4.5)                 # flat-ish test area; beyond it the terrain tapers down to the sample ground
TERRAIN_EXTENT = (-6.0, 43.0, -7.5, 7.5)
GROUND_DROP_M = 0.35                                   # the sample moon ground is lowered by this much in the course worlds


def terrain_height(x: float, y: float) -> float:
    """Gentle regolith relief (+-1.2 cm) plus parabolic crater bowls with Gaussian rims; outside the core area the surface
    blends smoothly down to the lowered sample ground so no step is visible."""
    h = 0.006 * math.sin(0.9 * x + 0.3) * math.cos(0.7 * y) + 0.006 * math.sin(0.37 * x - 1.1 * y + 1.0)
    for c in CRATERS:
        rr = math.hypot(x - c["x"], y - c["y"]) / c["radius"]
        if rr < 1.0:
            h += -c["depth"] * (1 - rr * rr)
        h += c["rim"] * math.exp(-((rr - 1.0) / 0.22) ** 2)
    x0, x1, y0, y1 = TERRAIN_CORE
    out = max(x0 - x, x - x1, y0 - y, y - y1, 0.0) / (TERRAIN_EXTENT[1] - x1)
    w = min(1.0, out) ** 2 * (3 - 2 * min(1.0, out))  # smoothstep over the 3 m border
    return h * (1 - w) - (GROUND_DROP_M + 0.02) * w


def lower_sample_ground(head: str) -> str:
    """Course worlds only: wrap the sample's GROUND mesh in a Pose 0.35 m lower so crater bowls are not covered by it."""
    a, b = "DEF GROUND Shape {", "\nDEF EARTH Pose {"
    if head.count(a) != 1 or head.count(b) != 1:
        raise SystemExit("unexpected moon.wbt layout: GROUND/EARTH nodes not found")
    head = head.replace(a, f"Pose {{\n  translation 0 0 {-GROUND_DROP_M}\n  children [\n{a}")
    return head.replace(b, "\n  ]\n}" + b)


def terrain_node(step: float = 0.15) -> str:
    """Height-field regolith (Poly Haven moon_01 maps tiled every 3.1 m) with craters, as a compact ElevationGrid."""
    x0, x1, y0, y1 = TERRAIN_EXTENT
    nx, ny = int(round((x1 - x0) / step)) + 1, int(round((y1 - y0) / step)) + 1
    hs = [f"{0.002 + terrain_height(x0 + i * step, y0 + j * step):.3f}".replace("0.", ".", 1).replace("-.", "-.")
          for j in range(ny) for i in range(nx)]
    t = "../assets/polyhaven/moon_01/moon_01"
    return f"""DEF REGOLITH_TERRAIN Solid {{
  translation {x0} {y0} 0
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColorMap ImageTexture {{
          url [ "{t}_diff_2k.jpg" ]
        }}
        normalMap ImageTexture {{
          url [ "{t}_nor_gl_2k.jpg" ]
        }}
        roughnessMap ImageTexture {{
          url [ "{t}_rough_1k.jpg" ]
        }}
        metalness 0
        textureTransform TextureTransform {{
          scale {(x1 - x0) / 3.1:.3f} {(y1 - y0) / 3.1:.3f}
        }}
      }}
      geometry ElevationGrid {{
        xDimension {nx}
        xSpacing {step}
        yDimension {ny}
        ySpacing {step}
        height [ {" ".join(hs)} ]
      }}
    }}
  ]
  name "regolith_terrain"
}}
"""


COURSE_ASSETS = [f"moon_rock_0{i}" for i in range(1, 8)] + ["namaqualand_boulder_04", "namaqualand_boulder_05"]


def course_assets_available() -> bool:
    return polyhaven_available() and all((ASSETS / a / f"{a}.obj").exists() for a in COURSE_ASSETS)


def course_layout() -> dict:
    """Nine deterministic test zones along +x, all north of the tour lane. Every object: class, colour, exact geometry."""
    R = []

    def rock_(zone, asset, x, y, width, yaw, sink=0.03):
        R.append({"zone": zone, "asset": asset, "x": x, "y": y, "width_m": width, "yaw": yaw, "sink_frac": sink})
    # Z1 isolated medium rocks (one per scan family)
    rock_("Z1_isolated", "moon_rock_02", 3.2, -1.0, 0.55, 0.4); rock_("Z1_isolated", "moon_rock_01", 3.4, 0.1, 0.70, 1.9)
    rock_("Z1_isolated", "moon_rock_04", 3.1, 1.1, 0.60, 3.0)
    # Z2 small-rock scatter (8-18 cm), all seven lunar scans
    for i, (x, y, w) in enumerate([(6.0, -1.2, .12), (6.3, -0.4, .09), (6.1, 0.3, .15), (6.6, 0.9, .10), (6.9, -0.8, .18),
                                   (7.1, 0.1, .08), (7.3, 1.3, .13), (6.5, 1.6, .11)]):
        rock_("Z2_small_scatter", f"moon_rock_0{i % 7 + 1}", x, y, w, i * 0.7)
    # Z3 large near-field boulder (real boulder scan) with a medium rock beside it
    rock_("Z3_large_boulder", "namaqualand_boulder_04", 10.0, 0.6, 1.40, 0.9); rock_("Z3_large_boulder", "moon_rock_06", 10.2, -0.9, 0.45, 2.4)
    # Z4 occlusion line and a cluster of touching rocks
    rock_("Z4_occlusion_cluster", "moon_rock_03", 13.0, 0.0, 0.40, 0.2); rock_("Z4_occlusion_cluster", "moon_rock_02", 13.8, 0.05, 0.60, 1.1)
    rock_("Z4_occlusion_cluster", "moon_rock_01", 13.2, -1.3, 0.45, 0.0); rock_("Z4_occlusion_cluster", "moon_rock_07", 13.45, -1.05, 0.35, 2.0)
    rock_("Z4_occlusion_cluster", "moon_rock_05", 13.3, -1.6, 0.30, 4.0)
    # Z6 rocks in the shadow of a boulder (low-sun light travels toward +x, -y)
    rock_("Z6_shadowed", "namaqualand_boulder_05", 20.0, 0.8, 1.20, 0.5); rock_("Z6_shadowed", "moon_rock_03", 20.9, -0.05, 0.30, 1.3)
    rock_("Z6_shadowed", "moon_rock_04", 20.7, 0.2, 0.25, 2.2)
    # Z7 craters: rocks on the large crater's rim and in its bowl; the small crater is empty (false-positive probe)
    c = CRATERS[0]
    for k, (ang, w, a) in enumerate([(2.6, 0.35, "moon_rock_05"), (1.2, 0.45, "moon_rock_06"), (-0.6, 0.30, "moon_rock_07")]):
        rock_("Z7_craters", a, round(c["x"] + 1.05 * c["radius"] * math.cos(ang), 3), round(c["y"] + 1.05 * c["radius"] * math.sin(ang), 3), w, k * 1.7)
    rock_("Z7_craters", "moon_rock_02", c["x"] + 0.15, c["y"] - 0.1, 0.35, 0.8)
    # Z8 dense boulder field: two real boulder scans + lunar rocks of mixed size
    rock_("Z8_boulder_field", "namaqualand_boulder_04", 31.0, 1.3, 1.80, 2.0); rock_("Z8_boulder_field", "namaqualand_boulder_05", 29.6, -0.6, 1.10, 0.3)
    for i, (x, y, w) in enumerate([(29.0, 0.6, .45), (30.2, 0.1, .35), (30.5, -1.2, .50), (31.4, -0.7, .40), (32.2, 0.2, .60),
                                   (32.6, -1.3, .30), (33.0, 1.0, .55), (31.8, 2.4, .40)]):
        rock_("Z8_boulder_field", f"moon_rock_0{(i + 3) % 7 + 1}", x, y, w, i * 1.3)
    # Z9 half-buried rocks (45-60 % of the height below the regolith)
    for i, (x, y, w, s) in enumerate([(36.0, -1.0, .60, .45), (36.4, 0.2, .80, .55), (36.9, 1.2, .50, .60), (37.4, -0.3, .40, .50)]):
        rock_("Z9_half_buried", f"moon_rock_0{i + 1}", x, y, w, i * 0.9, s)
    rocks = []
    for i, r in enumerate(R):
        ext = ph_mesh(r["asset"])["extent_m"]
        rocks.append({**r, "name": f"crs_rock_{i:02d}", "class": "rock", "scale": round(r["width_m"] / max(ext[0], ext[1]), 5),
                      "z0": round(terrain_height(r["x"], r["y"]), 4), "color": _col(i, True)})
    D = [  # Z5 non-rock distractors (false-positive probes)
        {"name": "lander_crate", "shape": "box", "size": [0.9, 0.6, 0.5], "x": 16.8, "y": -1.0, "z": 0.25, "rgb": [0.75, 0.75, 0.72], "metalness": 0.7},
        {"name": "fuel_tank", "shape": "cylinder", "height": 0.9, "radius": 0.22, "x": 17.0, "y": 0.4, "z": 0.22, "rotation": [1, 0, 0, 1.5708],
         "rgb": [0.85, 0.82, 0.78], "metalness": 0.8},
        {"name": "antenna_mast", "shape": "cylinder", "height": 1.2, "radius": 0.03, "x": 16.6, "y": 1.3, "z": 0.6, "rgb": [0.3, 0.3, 0.32], "metalness": 0.9},
        {"name": "solar_panel", "shape": "box", "size": [0.8, 0.5, 0.02], "x": 17.4, "y": -0.2, "z": 0.12, "rotation": [0, 1, 0, 0.35],
         "rgb": [0.12, 0.14, 0.25], "metalness": 0.5, "roughness": 0.2},
        {"name": "white_ball", "shape": "sphere", "radius": 0.18, "x": 17.6, "y": 1.0, "z": 0.18, "rgb": [0.92, 0.92, 0.9], "metalness": 0.0,
         "roughness": 0.6},
    ]
    for j, d in enumerate(D):
        d.update({"zone": "Z5_non_rock_distractors", "class": "non_rock", "color": _col(j, False)})
    vps = [{"id": 0, "zone": "Z1_isolated", "x": 1.0, "y": 0.0, "yaw": 0.0}, {"id": 1, "zone": "Z2_small_scatter", "x": 4.6, "y": 0.0, "yaw": 0.0},
           {"id": 2, "zone": "Z3_large_boulder", "x": 8.3, "y": -0.2, "yaw": 0.15}, {"id": 3, "zone": "Z3_large_boulder", "x": 6.8, "y": 0.0, "yaw": 0.0},
           {"id": 4, "zone": "Z4_occlusion_cluster", "x": 11.0, "y": -0.3, "yaw": -0.15}, {"id": 5, "zone": "Z5_non_rock_distractors", "x": 14.4, "y": 0.1, "yaw": 0.0},
           {"id": 6, "zone": "Z6_shadowed", "x": 18.4, "y": -0.6, "yaw": 0.15}, {"id": 7, "zone": "overview", "x": -1.0, "y": 0.0, "yaw": 0.0},
           {"id": 8, "zone": "Z7_craters", "x": 22.2, "y": -0.6, "yaw": 0.12}, {"id": 9, "zone": "Z8_boulder_field", "x": 27.0, "y": 0.2, "yaw": 0.05},
           {"id": 10, "zone": "Z9_half_buried", "x": 34.2, "y": 0.0, "yaw": 0.0}]
    # scripted tour along the lane: stop ~2.4 m before each zone centre and face it
    zones = {}
    for o in rocks + D:
        zones.setdefault(o["zone"], []).append((o["x"], o["y"]))
    stops = []
    for z, pts in sorted(zones.items(), key=lambda kv: sum(p[0] for p in kv[1]) / len(kv[1])):
        zx, zy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
        obs = course_obstacles(rocks, D)
        best = None                                    # lane stop whose clear straight approach ends closest to the zone
        for k in range(46):
            sx = round(zx - 3.0 + 0.1 * k, 2)
            look = round(math.atan2(zy - LANE_Y, zx - sx), 3)
            if look > 1.55:                            # keep turns below 89 deg
                continue
            d = approach_distance(sx, look, (zx, zy), obs)
            rest = math.hypot(zx - sx - d * math.cos(look), zy - LANE_Y - d * math.sin(look))
            if best is None or rest < best[0] - 0.05:
                best = (rest, sx, look, d)
        _, sx, look, d = best
        stops.append({"zone": z, "x": sx, "look_rad": look, "approach_m": d,
                      "observe_x": round(sx + d * math.cos(look), 3), "observe_y": round(LANE_Y + d * math.sin(look), 3),
                      "observe_to_zone_centre_m": round(math.hypot(zx - sx - d * math.cos(look), zy - LANE_Y - d * math.sin(look)), 2)})
    return {"rocks": rocks, "distractors": D, "craters": CRATERS, "viewpoints": vps, "tour": {"script": "course_tour", "lane_y": LANE_Y, "stops": stops}}


ROVER_CLEARANCE_M = 0.6           # min distance from the rover centre to any object edge on the approach (Koala half-length ~0.25 m)
OBSERVE_RANGE_M = 1.8             # stop the approach this far from the zone centre


def course_obstacles(rocks: list, distractors: list) -> list:
    """Footprint discs (x, y, radius) of everything the rover must not touch: rocks, distractors, crater bowls + rims."""
    obs = [(r["x"], r["y"], r["width_m"] / 2) for r in rocks]
    for d in distractors:
        if d["shape"] == "box":
            rad = math.hypot(d["size"][0], d["size"][1]) / 2
        elif d["shape"] == "cylinder" and d.get("rotation", [0])[0] == 1:   # lying tank: its length is along y
            rad = math.hypot(d["height"] / 2, d["radius"])
        else:
            rad = d["radius"]
        obs.append((d["x"], d["y"], rad))
    obs += [(c["x"], c["y"], c["radius"] + 0.15) for c in CRATERS]
    return obs


def approach_distance(sx: float, look: float, centre: tuple, obstacles: list, step: float = 0.02) -> float:
    """How far the rover may drive from (sx, LANE_Y) along heading ``look``: until it is OBSERVE_RANGE_M from the zone centre,
    or one more step would bring any obstacle edge closer than ROVER_CLEARANCE_M to its centre."""
    d = 0.0
    while True:
        nx, ny = sx + (d + step) * math.cos(look), LANE_Y + (d + step) * math.sin(look)
        if math.hypot(centre[0] - nx, centre[1] - ny) < OBSERVE_RANGE_M:
            break
        if min(math.hypot(ox - nx, oy - ny) - r for ox, oy, r in obstacles) < ROVER_CLEARANCE_M:
            break
        d += step
    return round(d, 2)


def course_viewpoint(rover_xyz: tuple, offset=(-2.2, -1.8, 1.4), look_ahead=(1.0, 0.6, 0.0), follow_type: str = "") -> str:
    """GUI camera behind and to the right of the rover's start, looking past it toward the zones; it then follows the rover
    (Webots 'Tracking Shot' keeps this offset; ``follow_type`` "Mounted Shot" also turns with the rover). Orientation = yaw
    about z, then pitch down about y, as an axis-angle."""
    pos = [r + o for r, o in zip(rover_xyz, offset)]
    d = [r + a - p for r, a, p in zip(rover_xyz, look_ahead, pos)]
    yaw, pitch = math.atan2(d[1], d[0]), math.atan2(-d[2], math.hypot(d[0], d[1]))
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    w, v = cy * cp, (-sy * sp, cy * sp, sy * cp)
    n = math.sqrt(sum(c * c for c in v))
    ft = f'\n  followType "{follow_type}"' if follow_type else ""
    return ("Viewpoint {\n  orientation " + " ".join(f"{c / n:.6f}" for c in v) + f" {2 * math.acos(w):.6f}\n"
            "  position " + " ".join(f"{c:g}" for c in pos) + '\n  follow "artemis_rover"' + ft + "\n}")


# Overlay layout for the GUI (Webots stacks every overlay at the top-left otherwise): name;visible;pixel size;x;y (fractions)
COURSE_WBPROJ_OVERLAYS = ("renderingDevicePerspectives: artemis_rover:front_camera;1;0.55;0;0\n"
                          "renderingDevicePerspectives: artemis_rover:yolo_view;1;0.55;0.58;0\n"
                          "renderingDevicePerspectives: artemis_rover:front_depth;1;0.3;0;0.78\n")


# R4.5 monitor world: 2 x 2 grid of equal 720 x 480 panels in the right part of the 3D view; the scene and the rover
# (MONITOR_VIEWPOINT) stay visible in the left part. The raw camera / range-finder overlays are hidden: the panels show them.
# Entry format: visible;pixel size (screen pixels per device pixel);x;y (fractions of the 3D view)
MONITOR_PANEL_SCALE, MONITOR_MARGIN_PX, MONITOR_GAP_PX = 0.8, 10, 26   # Webots pushes apart overlays that (nearly) touch
MONITOR_VIEW_PX = (1646, 1000)         # 3D view of the maximised window, 1920 x 1200 screen (125 % scaling), physical px
# Chase camera mounted on the rover (turns with it): behind and to its right, aimed ahead-right, so the rover stays in the
# lower-left part of the view that the panels leave free
MONITOR_VIEWPOINT = {"offset": (-3.3, -1.4, 1.8), "look_ahead": (1.5, -1.06, 0.0), "follow_type": "Mounted Shot"}


def monitor_wbproj_overlays() -> str:
    """Panels anchored to the top-right corner of the 3D view, equal size, equal gaps (pixel size is in physical pixels)."""
    vw, vh = MONITOR_VIEW_PX
    pw, ph = 720 * MONITOR_PANEL_SCALE, 480 * MONITOR_PANEL_SCALE
    x0 = vw - MONITOR_MARGIN_PX - 2 * pw - MONITOR_GAP_PX
    out = ["renderingDevicePerspectives: artemis_rover:front_camera;0;0.5;0;0\n",
           "renderingDevicePerspectives: artemis_rover:front_depth;0;0.5;0;0\n"]
    for name, (c, r) in {"rgb_view": (0, 0), "yolo_view": (1, 0), "depth_view": (0, 1), "map_view": (1, 1)}.items():
        x = (x0 + c * (pw + MONITOR_GAP_PX)) / vw
        y = (MONITOR_MARGIN_PX + r * (ph + MONITOR_GAP_PX)) / vh
        out.append(f"renderingDevicePerspectives: artemis_rover:{name};1;{MONITOR_PANEL_SCALE:g};{x:.4f};{y:.4f}\n")
    return "".join(out)


def monitor_wbproj_base() -> list:
    """GUI layout of the monitor world: the course world's (scene tree kept), with the text-editor and console docks
    hidden so the 3D view gets the room. Qt saveState: the byte after each dock's name is its flags (1 = visible)."""
    p = WORLDS / ".artemis_test_course.wbproj"
    lines = [l for l in p.read_text(encoding="utf-8").splitlines(True) if not l.startswith("renderingDevicePerspectives:")] \
        if p.exists() else ["Webots Project File version R2025a\n"]
    hide = {"00540065007800740045006400690074006f0072",                      # "TextEditor"
            "0043006f006e0073006f006c00650041006c006c0041006c006c"}          # "ConsoleAllAll"
    out = []
    for l in lines:
        if l.startswith("perspectives: "):
            for h in hide:
                l = l.replace(h + "01", h + "00")
        out.append(l)
    return out


def write_course_wbproj(name: str, overlays: str = COURSE_WBPROJ_OVERLAYS, base: list | None = None) -> None:
    """Hidden per-world GUI settings file (git-ignored): keep Webots' own panel layout, replace only the overlay entries
    (``base``: a full layout to write instead, e.g. the monitor world's)."""
    p = WORLDS / f".{name}.wbproj"
    keep = base if base is not None else \
        [l for l in p.read_text(encoding="utf-8").splitlines(True) if not l.startswith("renderingDevicePerspectives:")] if p.exists() \
        else ["Webots Project File version R2025a\n"]
    data = ("".join(keep) + overlays).encode("utf-8")
    with open(p, "r+b" if p.exists() else "wb") as f:     # Webots marks the file hidden: Windows refuses mode "w" on it
        f.write(data)
        f.truncate()


def build_test_course(webots_home: str, lighting: str = "lowsun", recognition: bool = True, monitor: bool = False) -> tuple[str, dict]:
    lay = course_layout() if course_assets_available() else None
    if lay is None:
        raise SystemExit("Poly Haven assets missing: run artemis-webots/tools/fetch_polyhaven.py first")
    info = ('    "Artemis-X test course v2: nine zones (isolated, small scatter, large boulder, occlusion/cluster, non-rock distractors,"\n'
            f'    "shadowed, craters, boulder field, half-buried) - Poly Haven CC0 rocks on cratered regolith terrain, {SUN[lighting][1]}."')
    head = lunar_lighting(lower_sample_ground(sample_head(webots_home, f"Artemis-X test course ({lighting})", info)), SUN[lighting][0])
    vp = course_viewpoint((0.0, LANE_Y, 0.0), **MONITOR_VIEWPOINT) if monitor else course_viewpoint((0.0, LANE_Y, 0.0))
    head = re.sub(r"Viewpoint \{\n.*?\n\}", vp, head, count=1, flags=re.S)
    nodes = [terrain_node()]
    man = {"seed": "course", "variant": "test_course" if lighting == "lowsun" else f"test_course_{lighting}", "lighting_variant": lighting, "style": "polyhaven", "camera": _camera_meta(),
           "viewpoints": lay["viewpoints"], "lighting": SUN[lighting][1] + ", HDR fill 0.15, shadows", "rocks": [], "distractors": [],
           "craters": lay["craters"], "tour": lay["tour"]}
    for r in lay["rocks"]:
        nodes.append(ph_rock_node(r["name"], r["asset"], r["x"], r["y"], r["yaw"], r["scale"], r["color"] if recognition else None,
                                  z0=r["z0"], sink_frac=r["sink_frac"]))
        man["rocks"].append({**r, "shape": "mesh", "r": r["width_m"] / 2,
                             "placement": {"asset": r["asset"], "x": r["x"], "y": r["y"], "yaw": r["yaw"], "scale": r["scale"],
                                           "z0": r["z0"], "sink_frac": r["sink_frac"]}})
    for d in lay["distractors"]:
        nodes.append(distractor_node(d))
        man["distractors"].append(d)
    plan = json.dumps(lay["tour"], separators=(",", ":")).replace('"', '\\"')
    return head + rover(recognition, (0, LANE_Y, 0), custom_data=plan, live_view=True, monitor=monitor) + "".join(nodes), man


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--webots-home", default=os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots"))
    ap.add_argument("--eval", nargs="*", type=int, help="seeds for the paired evaluation worlds")
    ap.add_argument("--enhanced-style", choices=("polyhaven", "procedural"), default="polyhaven")
    a = ap.parse_args()
    if a.eval:
        for seed in a.eval:
            for variant in ("baseline", "enhanced"):
                vp = viewpoints(seed)[0]
                text, man = build_world(a.webots_home, variant, seed, recognition=True, pose=(vp["x"], vp["y"], vp["yaw"]),
                                        enhanced_style=a.enhanced_style)
                name = f"eval_s{seed}_{variant}"
                (WORLDS / f"{name}.wbt").write_text(text, encoding="utf-8", newline="\n")
                (WORLDS / f"{name}.json").write_text(json.dumps(man), encoding="utf-8", newline="\n")
                print(f"wrote worlds/{name}.wbt + .json ({man['style']})")
        return
    # R1/R2 baseline world (byte-stable)
    info = ('    "Artemis-X simulator v0.1: one Koala rover (Python controller) with a forward RGB camera and an aligned RangeFinder."\n'
            '    "Generated from the official Webots R2025a moon.wbt sample by artemis-webots/tools/make_world.py."')
    world = sample_head(a.webots_home, "Artemis-X lunar rover (v0.1)", info) + ROVER + "".join(rock(*r) for r in ROCKS)
    OUT.write_text(world, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(HERE.parent).as_posix()} ({len(world):,} bytes)")
    text, _ = build_world(a.webots_home, "enhanced", 0, recognition=False, pose=(0, 0, 0), enhanced_style=a.enhanced_style)
    (WORLDS / "artemis_moon_enhanced.wbt").write_text(text, encoding="utf-8", newline="\n")
    print("wrote artemis-webots/worlds/artemis_moon_enhanced.wbt")
    if course_assets_available():
        for lighting, name in (("lowsun", "artemis_test_course"), ("highsun", "artemis_test_course_highsun")):
            text, man = build_test_course(a.webots_home, lighting, recognition=True)
            (WORLDS / f"{name}.wbt").write_text(text, encoding="utf-8", newline="\n")
            (WORLDS / f"{name}.json").write_text(json.dumps(man), encoding="utf-8", newline="\n")
            write_course_wbproj(name)
            print(f"wrote artemis-webots/worlds/{name}.wbt ({len(man['rocks'])} rocks, {len(man['distractors'])} distractors, "
                  f"{len(man['craters'])} craters, {len(man['tour']['stops'])} tour stops)")
        text, _ = build_test_course(a.webots_home, "lowsun", recognition=True, monitor=True)
        (WORLDS / "artemis_r4_monitor.wbt").write_text(text, encoding="utf-8", newline="\n")
        write_course_wbproj("artemis_r4_monitor", monitor_wbproj_overlays(), monitor_wbproj_base())
        print("wrote artemis-webots/worlds/artemis_r4_monitor.wbt (R4.5 2 x 2 monitoring layout; manifest: artemis_test_course.json)")


if __name__ == "__main__":
    main()

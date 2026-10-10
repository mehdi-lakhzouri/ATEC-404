"""Generate worlds/artemis_moon.wbt from the official Webots R2025a `moon.wbt` sample (read-only on the installation).

The lunar ground mesh, textures, background and every EXTERNPROTO URL are copied verbatim. Only the two Koala robots
at the end of the file are replaced by ONE Artemis-X rover (deterministic pose, Python controller, supervisor access
for ground-truth pose, forward RGB camera + aligned RangeFinder in the PROTO's extensionSlot) and three static test
rocks are added in its field of view.

Usage:  python artemis-webots/tools/make_world.py [--webots-home "C:/Program Files/Webots"]
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "worlds" / "artemis_moon.wbt"
SAMPLE = Path("projects/samples/demos/worlds/moon.wbt")

# Sensor geometry shared by the camera and the range finder (identical pose, FOV and resolution -> pixel-aligned)
SENSOR = {"translation": "0.16 0 0.24", "rotation": "0 1 0 0.12", "fov": 1.0, "width": 720, "height": 480}

ROVER = f"""DEF ARTEMIS_ROVER Koala {{
  translation 0 0 0
  rotation 0 0 1 0
  name "artemis_rover"
  controller "artemis_rover"
  supervisor TRUE
  extensionSlot [
    Camera {{
      translation {SENSOR['translation']}
      rotation {SENSOR['rotation']}
      name "front_camera"
      fieldOfView {SENSOR['fov']}
      width {SENSOR['width']}
      height {SENSOR['height']}
      near 0.05
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
    }}
  ]
}}
"""


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--webots-home", default=os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots"))
    a = ap.parse_args()
    src = (Path(a.webots_home) / SAMPLE).read_text(encoding="utf-8")
    head, sep, _ = src.partition("\nKoala {")
    if not sep:
        raise SystemExit("unexpected moon.wbt layout: no Koala node found")
    head = head.replace('title "Koala Robot"', 'title "Artemis-X lunar rover (v0.1)"')
    head = re.sub(r"WorldInfo \{\n  info \[\n.*?\n  \]", 'WorldInfo {\n  info [\n'
                  '    "Artemis-X simulator v0.1: one Koala rover (Python controller) with a forward RGB camera and an aligned RangeFinder."\n'
                  '    "Generated from the official Webots R2025a moon.wbt sample by artemis-webots/tools/make_world.py."\n'
                  '  ]', head, count=1, flags=re.S)
    head = head.replace("  gravity 1.62\n", "  basicTimeStep 32\n  gravity 1.62\n")
    # third-person view behind/left of the rover, looking at the rover and the test rocks; follows the rover
    head = re.sub(r"Viewpoint \{\n.*?\n\}", "Viewpoint {\n  orientation -0.140722 0.539298 0.830274 0.609005\n"
                  "  position -1.5 -1.0 1.0\n  follow \"artemis_rover\"\n}", head, count=1, flags=re.S)
    world = head.rstrip("\n") + "\n" + ROVER + "".join(rock(*r) for r in ROCKS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(world, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(HERE.parent).as_posix()} ({len(world):,} bytes)")


if __name__ == "__main__":
    main()

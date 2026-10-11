"""Download the CC0 Poly Haven lunar assets used by the Artemis-X Webots scenes and convert the rock models for Webots.

Assets (https://polyhaven.com/license: CC0 1.0, no attribution required - authors are recorded anyway):
    moon_rock_01 ... moon_rock_07              lunar photogrammetry rock models (glTF 1K: mesh + diffuse/normal/ARM JPGs)
    namaqualand_boulder_04, _05                desert boulder scans (1-2.5 m) used as large boulders; a grey "_lunar" diffuse
                                               map is derived from their brown texture (recorded as a derivative)
    moon_01                                    regolith ground texture (diffuse + normal 2K, roughness 1K JPGs)

Downloads go to artemis-webots/assets/polyhaven/<asset>/ (git-ignored). Each file is checked against the MD5 published by
the Poly Haven API; SHA-256 values, URLs, licence and authors are written to artemis-webots/assets/polyhaven/ASSETS.json
(committed). Rock glTFs are converted (by this script, not by any downloaded code) to:
    <asset>.obj          Webots ``Mesh`` geometry, Webots axes (Z up), metres, with UVs and normals
    <asset>_mesh.json    the same vertices/triangles for ground truth (ray casting, projected reference boxes)
Meshes above 20,000 triangles (the two boulders) are simplified by vertex clustering so Webots can cast their shadows.

Usage:  uv run --project backend --no-sync python artemis-webots/tools/fetch_polyhaven.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np

DEST = Path(__file__).resolve().parents[1] / "assets" / "polyhaven"
API = "https://api.polyhaven.com"
ROCKS = ["moon_rock_01", "moon_rock_02", "moon_rock_03", "moon_rock_04", "moon_rock_05", "moon_rock_06", "moon_rock_07",
         "namaqualand_boulder_04", "namaqualand_boulder_05"]
LUNARIZE = {"namaqualand_boulder_04", "namaqualand_boulder_05"}   # brown desert boulders -> grey derived diffuse map (CC0 permits)
GROUND = {"asset": "moon_01", "maps": {"Diffuse": "2k", "nor_gl": "2k", "Rough": "1k"}}
MAX_TRIANGLES = 20000           # Webots casts no shadow from meshes above 21,845 triangles
UA = {"User-Agent": "artemis-x-webots-asset-fetch/1.0"}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:     # noqa: S310 - fixed https Poly Haven endpoints
        return r.read()


def _save(url: str, md5: str | None, path: Path, force: bool) -> dict:
    if path.exists() and not force and md5 and hashlib.md5(path.read_bytes()).hexdigest() == md5:
        data = path.read_bytes()
    else:
        data = _get(url)
        if md5 and hashlib.md5(data).hexdigest() != md5:
            raise SystemExit(f"MD5 mismatch for {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return {"file": path.relative_to(DEST).as_posix(), "url": url, "bytes": len(data), "md5": md5, "sha256": hashlib.sha256(data).hexdigest()}


# ---- minimal glTF 2.0 reader (positions, normals, UV0, indices, node transforms) --------------------------------------
_CT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _accessor(g: dict, buf: bytes, i: int) -> np.ndarray:
    a = g["accessors"][i]; bv = g["bufferViews"][a["bufferView"]]
    n, comps = a["count"], _NC[a["type"]]
    dt = np.dtype("<" + _CT[a["componentType"]])
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride", 0) or dt.itemsize * comps
    raw = np.frombuffer(buf, np.uint8, count=stride * (n - 1) + dt.itemsize * comps, offset=start)
    out = np.lib.stride_tricks.as_strided(raw, shape=(n, dt.itemsize * comps), strides=(stride, 1)).copy()
    return out.view(dt).reshape(n, comps)


def _node_matrix(nd: dict) -> np.ndarray:
    if "matrix" in nd:
        return np.array(nd["matrix"], float).reshape(4, 4).T
    t = np.array(nd.get("translation", [0, 0, 0]), float)
    x, y, z, w = nd.get("rotation", [0, 0, 0, 1])
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    s = np.diag(nd.get("scale", [1, 1, 1]))
    m = np.eye(4); m[:3, :3] = r @ s; m[:3, 3] = t
    return m


def gltf_to_webots(gltf_path: Path) -> dict:
    g = json.loads(gltf_path.read_text(encoding="utf-8"))
    buf = (gltf_path.parent / g["buffers"][0]["uri"]).read_bytes()
    P, N, T, F = [], [], [], []
    off = 0
    for ni, nd in enumerate(g.get("nodes", [])):
        if "mesh" not in nd:
            continue
        m = _node_matrix(nd)
        for prim in g["meshes"][nd["mesh"]]["primitives"]:
            at = prim["attributes"]
            p = _accessor(g, buf, at["POSITION"]).astype(float)
            n = _accessor(g, buf, at["NORMAL"]).astype(float) if "NORMAL" in at else np.zeros_like(p)
            uv = _accessor(g, buf, at["TEXCOORD_0"]).astype(float) if "TEXCOORD_0" in at else np.zeros((len(p), 2))
            idx = _accessor(g, buf, prim["indices"]).astype(int).reshape(-1, 3) if "indices" in prim else np.arange(len(p)).reshape(-1, 3)
            p = (m[:3, :3] @ p.T).T + m[:3, 3]
            n = (m[:3, :3] @ n.T).T
            P.append(p); N.append(n); T.append(uv); F.append(idx + off); off += len(p)
    P, N, T, F = np.concatenate(P), np.concatenate(N), np.concatenate(T), np.concatenate(F)
    # glTF is Y-up; Webots is Z-up: (x, y, z) -> (x, -z, y)
    P = np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1); N = np.stack([N[:, 0], -N[:, 2], N[:, 1]], 1)
    P[:, 2] -= P[:, 2].min()                                   # rest the model on z = 0
    P[:, :2] -= (P[:, :2].min(0) + P[:, :2].max(0)) / 2       # centre the footprint
    return {"P": P, "N": N, "T": T, "F": F}


def decimate(mesh: dict, target: int) -> dict:
    """Vertex-clustering simplification to at most ``target`` triangles (Webots casts no shadow from a mesh with more than
    21,845 triangles). Positions are merged per grid cell; every surviving triangle keeps its ORIGINAL corner UVs and normals
    (separate OBJ v / vt / vn indices), so the texture mapping is unchanged."""
    P, F = mesh["P"], mesh["F"]
    if len(F) <= target:
        return mesh
    lo, size = P.min(0), float((P.max(0) - P.min(0)).max())
    best, a, b = None, size / 4000, size / 20
    for _ in range(30):                                        # bisect the cell size for the largest count <= target
        s = (a * b) ** 0.5
        _, inv = np.unique(np.floor((P - lo) / s).astype(np.int64), axis=0, return_inverse=True)
        inv = inv.ravel(); c = inv[F]
        keep = (c[:, 0] != c[:, 1]) & (c[:, 1] != c[:, 2]) & (c[:, 0] != c[:, 2])
        _, first = np.unique(np.sort(c[keep], 1), axis=0, return_index=True)
        idx = np.nonzero(keep)[0][np.sort(first)]
        if len(idx) <= target:
            best, b = (inv, idx), s
        else:
            a = s
    inv, idx = best
    n = int(inv.max()) + 1
    Q = np.zeros((n, 3)); np.add.at(Q, inv, P); Q /= np.bincount(inv, minlength=n)[:, None]
    used = np.unique(inv[F[idx]])
    remap = np.full(n, -1); remap[used] = np.arange(len(used))
    Q = Q[used]
    Q[:, 2] -= Q[:, 2].min(); Q[:, :2] -= (Q[:, :2].min(0) + Q[:, :2].max(0)) / 2
    return {"P": Q, "N": mesh["N"], "T": mesh["T"], "F": remap[inv[F[idx]]], "FA": F[idx], "decimated_from": int(len(F))}


def write_obj(mesh: dict, path: Path, mtl_note: str) -> None:
    lines = [f"# {mtl_note}"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in mesh["P"]]
    lines += [f"vt {u:.6f} {1 - v:.6f}" for u, v in mesh["T"]]           # glTF UV origin top-left -> OBJ bottom-left
    lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in mesh["N"]]
    FA = mesh.get("FA", mesh["F"])                                         # UV / normal indices (differ after decimation)
    lines += [f"f {p[0] + 1}/{q[0] + 1}/{q[0] + 1} {p[1] + 1}/{q[1] + 1}/{q[1] + 1} {p[2] + 1}/{q[2] + 1}/{q[2] + 1}"
              for p, q in zip(mesh["F"], FA)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def lunarize(src: Path) -> dict:
    """Grey, slightly brightened copy of a diffuse map (lunar regolith is near-neutral grey). Original file is kept."""
    from PIL import Image, ImageOps
    im = ImageOps.grayscale(Image.open(src).convert("RGB"))
    im = ImageOps.autocontrast(im, cutoff=1).point(lambda v: int(40 + v * 0.75))
    dst = src.with_name(src.stem + "_lunar.jpg")
    Image.merge("RGB", (im, im, im)).save(dst, quality=92)
    return {"file": dst.relative_to(DEST).as_posix(), "from": src.relative_to(DEST).as_posix(),
            "operation": "greyscale + autocontrast(1%) + linear 40+0.75*v (derivative of a CC0 texture)",
            "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--locked", action="store_true", help="use committed URLs/SHA256; preserve ASSETS.json instead of querying latest API metadata")
    a = ap.parse_args()
    if a.locked:
        record = json.loads((DEST / "ASSETS.json").read_text(encoding="utf-8"))
        for name, asset in record["assets"].items():
            for entry in asset["files"]:
                path = DEST / entry["file"]
                if not path.resolve().is_relative_to(DEST.resolve()):
                    raise SystemExit("asset path outside destination")
                data = path.read_bytes() if path.exists() and not a.force else _get(entry["url"])
                if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                    if path.exists() and not a.force:
                        data = _get(entry["url"])
                    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                        raise SystemExit(f"SHA256 mismatch for {entry['file']}; committed provenance was NOT changed")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            if asset["type"] == "model":
                d = DEST / name
                mesh = decimate(gltf_to_webots(d / f"{name}_1k.gltf"), MAX_TRIANGLES)
                write_obj(mesh, d / f"{name}.obj", f"{name} (Poly Haven, CC0) converted from glTF 1K by fetch_polyhaven.py; Webots axes, metres, with UVs and normals")
                extent = (mesh["P"].max(0) - mesh["P"].min(0)).round(4).tolist()
                (d / f"{name}_mesh.json").write_text(json.dumps({"vertices": mesh["P"].round(5).tolist(), "triangles": mesh["F"].tolist(), "extent_m": extent}), encoding="utf-8")
                if asset.get("derived"):
                    derived = lunarize(d / "textures" / f"{name}_diff_1k.jpg")
                    if derived["sha256"] != asset["derived"]["sha256"]:
                        raise SystemExit(f"derived texture mismatch for {name}; use the locked ML environment")
            print(f"verified {name}")
        print("Locked assets verified and converted; committed ASSETS.json unchanged")
        return 0
    record = {"source": "Poly Haven (https://polyhaven.com)", "license": "CC0 1.0 Universal (https://polyhaven.com/license)",
              "fetched": time.strftime("%Y-%m-%d"), "assets": {}}
    for asset in ROCKS:
        info = json.loads(_get(f"{API}/info/{asset}"))
        files = json.loads(_get(f"{API}/files/{asset}"))
        g = files["gltf"]["1k"]["gltf"]
        d = DEST / asset
        entries = [_save(g["url"], g.get("md5"), d / Path(g["url"]).name, a.force)]
        for rel, f in g.get("include", {}).items():
            entries.append(_save(f["url"], f.get("md5"), d / rel, a.force))
        mesh = decimate(gltf_to_webots(d / Path(g["url"]).name), MAX_TRIANGLES)
        derived = None
        if asset in LUNARIZE:
            derived = lunarize(d / "textures" / f"{asset}_diff_1k.jpg")
        write_obj(mesh, d / f"{asset}.obj", f"{asset} (Poly Haven, CC0) converted from glTF 1K by fetch_polyhaven.py; Webots axes, metres")
        ext = (mesh["P"].max(0) - mesh["P"].min(0)).round(4).tolist()
        (d / f"{asset}_mesh.json").write_text(json.dumps({"vertices": mesh["P"].round(5).tolist(), "triangles": mesh["F"].tolist(),
                                                          "extent_m": ext}), encoding="utf-8")
        record["assets"][asset] = {"type": "model", "authors": info.get("authors"), "polycount": info.get("polycount"),
                                   "native_extent_m_xyz": ext, "triangles": int(len(mesh["F"])), "vertices": int(len(mesh["P"])),
                                   "files": entries, "converted": [f"{asset}/{asset}.obj", f"{asset}/{asset}_mesh.json"],
                                   "derived": derived, "decimated_from_triangles": mesh.get("decimated_from"),
                                   "textures_used": {"baseColorMap": f"{asset}/textures/{asset}_diff_1k" + ("_lunar" if derived else "") + ".jpg",
                                                     "normalMap": f"{asset}/textures/{asset}_nor_gl_1k.jpg",
                                                     "occlusionRoughnessMetalness(ARM)": f"{asset}/textures/{asset}_arm_1k.jpg"}}
        print(f"{asset}: {len(mesh['F'])} triangles, extent {ext} m")
    asset = GROUND["asset"]
    info = json.loads(_get(f"{API}/info/{asset}")); files = json.loads(_get(f"{API}/files/{asset}"))
    ents = []
    for kind, res in GROUND["maps"].items():
        f = files[kind][res]["jpg"]
        ents.append(_save(f["url"], f.get("md5"), DEST / asset / Path(f["url"]).name, a.force))
    record["assets"][asset] = {"type": "texture", "authors": info.get("authors"), "real_world_size_m": [v / 1000 for v in info.get("dimensions", [])],
                               "files": ents}
    total = sum(e["bytes"] for v in record["assets"].values() for e in v["files"])
    record["total_download_bytes"] = total
    (DEST / "ASSETS.json").write_text(json.dumps(record, indent=1), encoding="utf-8", newline="\n")
    print(f"total downloaded: {total / 1e6:.1f} MB -> {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

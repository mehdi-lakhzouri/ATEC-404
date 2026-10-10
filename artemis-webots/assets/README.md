# artemis-webots/assets: external scene assets

| Asset | Source | Licence | Used as |
|---|---|---|---|
| `moon_rock_01` … `moon_rock_07` | Poly Haven lunar photogrammetry models (glTF, 1K textures) | **CC0 1.0** ([polyhaven.com/license](https://polyhaven.com/license)) | Rocks in the enhanced evaluation scenes (01–03) and the test course (all seven) |
| `namaqualand_boulder_04`, `namaqualand_boulder_05` | Poly Haven boulder scans (1–2.5 m, desert) | **CC0 1.0** | Large boulders in the test course (zones Z3, Z6, Z8) |
| `moon_01` | Poly Haven regolith texture (diffuse and normal 2K, roughness 1K, JPG) | **CC0 1.0** | Ground patch in the enhanced scenes; cratered height-field terrain in the test course |

CC0 requires no attribution. The authors are recorded in `polyhaven/ASSETS.json` anyway, together with:

- every file's URL;
- its MD5, verified against the Poly Haven API;
- its SHA-256.

**The binary files are not committed** (≈ 26 MB downloaded, ≈ 50 MB after conversion). Fetch and convert them with:

```powershell
uv run --project backend --no-sync python -I artemis-webots/tools/fetch_polyhaven.py
```

What the script does:

| Step | Detail |
|---|---|
| Download | Each rock's glTF 1K (`.gltf` + `.bin` + diffuse, normal and ARM JPGs) and the ground's JPG maps |
| Convert | A small built-in glTF reader turns each rock into `<asset>.obj` (Webots axes, Z up, metres, resting on z = 0, with UVs and normals) and `<asset>_mesh.json` (the same vertices and triangles) |
| Simplify | Meshes above 20,000 triangles are reduced by vertex clustering. Each surviving triangle keeps its original UVs and normals. Only the two boulders are affected: 59,066 → 20,000 and 89,618 → 19,999 triangles, extents within 1 %. Reason: Webots casts **no shadow** from a mesh above 21,845 triangles, and zone Z6 depends on the boulder's shadow. At close range (< 2 m) the simplified boulder looks slightly faceted |
| Lunarise (derivative) | The boulders' brown desert diffuse maps get a grey copy, `<asset>_diff_1k_lunar.jpg`: greyscale, 1 % autocontrast, then 40 + 0.75·v. The original stays. The operation and the SHA-256 are recorded under `derived` in `ASSETS.json` |
| Check | Converted extents match Poly Haven's published dimensions (e.g. moon_rock_01: 0.212 × 0.112 × 0.075 m) |

The world generator scales and rotates the meshes with the **same transform** for the Webots node and for the ground-truth vertices.
This includes the terrain height `z0` and the sink depth. So reference boxes and surface ray-casts use exactly the rendered
geometry. The test-course manifests store each rock's placement (asset, x, y, yaw, scale, z0, sink fraction), not its vertices;
the analyser rebuilds the mesh with `make_world.ph_place`.

**If the assets are missing:**

- `make_world.py --eval` falls back to procedural rocks, and records `"style": "procedural"` in each manifest.
- The test course is not generated.

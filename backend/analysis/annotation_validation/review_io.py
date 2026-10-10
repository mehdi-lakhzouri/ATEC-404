"""Review manifests, assistant-review ingestion and the static HTML gallery.

Human review fields always default to NOT_REVIEWED. The assistant's own visual assessments are read from
`assistant_review/*.csv` (written by the AI assistant, never by this code) and kept in separate columns.
"""

from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..dataset_exploration.common import LOG

HUMAN_FIELDS = ["review_status", "reviewer_decision", "reviewer_notes"]
MANIFEST_FIELDS = ["review_item_id", "frame_id", "dataset", "sample_reason", "strategy", "raw_image_path", "overlay_path",
                   "comparison_path", "candidate_count", *HUMAN_FIELDS, "assistant_assessment", "assistant_notes"]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_assistant(out: Path) -> dict[str, dict[str, dict[str, str]]]:
    d = out / "assistant_review"
    spec = {"scenes": ("scene_review.csv", "frame_id"), "candidates": ("candidate_review.csv", "candidate_id"),
            "uncertain": ("uncertain_review.csv", "uncertain_id"), "t06": ("t06_review.csv", "image"),
            "exposure": ("exposure_review.csv", "item")}
    res = {}
    for k, (fn, key) in spec.items():
        res[k] = {r[key]: r for r in read_csv(d / fn)}
    LOG.info("assistant review rows: %s", {k: len(v) for k, v in res.items()})
    return res


def write_human_manifest(out: Path, rows: list[dict[str, Any]]) -> str:
    """Never overwrite human decisions: if the existing manifest has any reviewed row, write a .generated copy."""
    target = out / "human_review_manifest.csv"
    existing = read_csv(target)
    if any(r.get("reviewer_decision", "NOT_REVIEWED") not in ("", "NOT_REVIEWED") or r.get("reviewer_notes") for r in existing):
        alt = out / "human_review_manifest.generated.csv"
        write_csv(alt, rows, MANIFEST_FIELDS)
        LOG.warning("human_review_manifest.csv contains human edits; wrote %s instead", alt.name)
        return alt.name
    write_csv(target, rows, MANIFEST_FIELDS)
    return target.name


def human_review_status(out: Path) -> dict[str, Any]:
    rows = read_csv(out / "human_review_manifest.csv")
    return {"rows": len(rows), "decisions": dict(Counter(r.get("reviewer_decision", "") for r in rows))}


def gallery(out: Path, scenes: list[dict[str, Any]], extra_sections: dict[str, list[str]]) -> str:
    """Static, framework-free gallery with category / anomaly / assistant-flag filters."""
    data = json.dumps(scenes, sort_keys=True)
    cats = sorted({c for s in scenes for c in s["categories"]})
    sec_html = ""
    for title, files in extra_sections.items():
        links = "".join(f'<a href="../{html.escape(f)}" target="_blank"><img loading="lazy" src="../{html.escape(f)}"></a>' for f in files)
        sec_html += f"<details><summary>{html.escape(title)} ({len(files)})</summary><div class='strip'>{links}</div></details>"
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Annotation Review Gallery</title>
<style>
:root {{ --bg:#f6f6f4; --fg:#1b1b1b; --card:#fff; --muted:#666; --line:#ddd; --accent:#0b5cad; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#141414; --fg:#eee; --card:#1f1f1f; --muted:#aaa; --line:#333; --accent:#6aa9ff; }} }}
:root[data-theme="dark"] {{ --bg:#141414; --fg:#eee; --card:#1f1f1f; --muted:#aaa; --line:#333; --accent:#6aa9ff; }}
body {{ margin:0; padding:16px; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,Segoe UI,sans-serif; }}
h1 {{ font-size:20px; margin:0 0 4px; }} .note {{ color:var(--muted); max-width:70em; }}
.bar {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; position:sticky; top:0; background:var(--bg); padding:8px 0; z-index:2; }}
select,input,button {{ font:inherit; padding:4px 8px; background:var(--card); color:var(--fg); border:1px solid var(--line); border-radius:4px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:12px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:8px; }}
.card img {{ width:100%; height:auto; display:block; cursor:zoom-in; }}
.meta {{ font-size:12px; color:var(--muted); margin-top:6px; }} .tag {{ display:inline-block; border:1px solid var(--line); border-radius:3px; padding:0 4px; margin:1px; font-size:11px; }}
.links a {{ color:var(--accent); margin-right:8px; font-size:12px; }}
#viewer {{ position:fixed; inset:0; background:rgba(0,0,0,.92); display:none; overflow:auto; z-index:5; }} #viewer img {{ max-width:none; display:block; margin:auto; }}
details {{ margin:10px 0; }} .strip {{ display:flex; flex-wrap:wrap; gap:6px; }} .strip img {{ height:160px; }}
</style></head><body>
<h1>Artemis-X AI-01.B - S1 vs S2 annotation review</h1>
<p class="note">Each card = one sampled Artificial Lunar scene, panels A raw / B S1 CSV / C S2 mask-derived / D diagnostic.
Assistant flags are an AI assistant's visual assessment, <b>not</b> human validation. Record human decisions in <code>human_review_manifest.csv</code>.
Click an image for full resolution (Esc / click to close).</p>
<div class="bar"><select id="cat"><option value="">all categories</option>{''.join(f'<option>{html.escape(c)}</option>' for c in cats)}</select>
<select id="anom"><option value="">all frames</option><option value="clean">candidate subset only</option><option value="anom">anomaly-listed only</option></select>
<select id="flag"><option value="">any assistant assessment</option><option value="s1miss">S1 misses visible rocks</option><option value="s2issue">S2 split/merge/artefact noted</option></select>
<input id="q" placeholder="frame id or text"><span id="n" class="meta"></span></div>
<div id="g" class="grid"></div>
<h2>Other review material</h2>{sec_html}
<div id="viewer" onclick="this.style.display='none'"><img id="vimg" alt=""></div>
<script>
const S = {data};
const g = document.getElementById('g');
const esc = v => String(v ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[ch]);
function show() {{
  const c = cat.value, a = anom.value, fl = flag.value, q = document.getElementById('q').value.toLowerCase();
  let n = 0; g.innerHTML = '';
  for (const s of S) {{
    if (c && !s.categories.includes(c)) continue;
    if (a === 'clean' && s.anomaly_lists) continue;
    if (a === 'anom' && !s.anomaly_lists) continue;
    if (fl === 's1miss' && s.assistant.s1_misses_visible_rocks !== 'yes') continue;
    if (fl === 's2issue' && !(s.assistant.s2_splits === 'yes' || s.assistant.s2_merges === 'yes' || s.assistant.s2_artifacts === 'yes')) continue;
    if (q && !(String(s.frame_id).includes(q) || JSON.stringify(s).toLowerCase().includes(q))) continue;
    n++;
    const d = document.createElement('div'); d.className = 'card';
    d.innerHTML = `<img loading="lazy" src="../${{esc(s.comparison)}}" alt="frame ${{esc(s.frame_id)}}">
      <div class="meta"><b>frame ${{esc(s.frame_id)}}</b> ${{s.categories.map(x => `<span class="tag">${{esc(x)}}</span>`).join('')}}
      ${{s.anomaly_lists ? `<span class="tag">anomaly: ${{esc(s.anomaly_lists)}}</span>` : ''}}<br>
      S1 ${{esc(s.counts.csv)}} | S2 large ${{esc(s.counts.s2_large_matched)}}+${{esc(s.counts.s2_large_new)}} | S2 small ${{esc(s.counts.s2_small)}} | rejected ${{esc(s.counts.s2_rejected)}}<br>
      assistant (AI, not human): ${{esc(s.assistant.notes || 'not assessed')}}</div>
      <div class="links"><a href="../${{esc(s.s1_side_by_side)}}" target="_blank">raw|S1|ground</a><a href="../${{esc(s.s2_overlay)}}" target="_blank">S2</a>
      <a href="../${{esc(s.comparison)}}" target="_blank">comparison</a></div>`;
    d.querySelector('img').onclick = e => {{ vimg.src = e.target.src; viewer.style.display = 'block'; }};
    g.appendChild(d);
  }}
  document.getElementById('n').textContent = n + ' scenes';
}}
for (const id of ['cat', 'anom', 'flag', 'q']) document.getElementById(id).addEventListener('input', show);
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') viewer.style.display = 'none'; }});
show();
</script></body></html>
"""
    p = out / "gallery" / "index.html"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(page, encoding="utf-8")
    return "gallery/index.html"

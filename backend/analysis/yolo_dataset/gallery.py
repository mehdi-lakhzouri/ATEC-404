"""Interactive human-review gallery (static HTML, no framework, no network).

Every scene starts NOT_REVIEWED; the page never auto-approves. Decisions are exported by the reviewer as JSON/CSV and
can be placed in experiments/yolo-dataset/human_review/ to be reported by the next pipeline run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED = (676, 742, 977, 4191, 8894)


def select_scenes(props: dict[int, dict[str, Any]], split_of: dict[int, str], groups: dict[int, str],
                  anns_by_frame: dict[int, list[dict[str, Any]]], seed: int) -> list[tuple[int, str]]:
    rng = np.random.default_rng(seed)
    chosen: dict[int, str] = {f: "required (Task B/C reference scene)" for f in REQUIRED if f in split_of}
    first = sorted(g for g in set(groups.values()) if g.startswith("G"))[:1]  # exact-duplicate pair (G0001 = frames 1, 6658)
    for f in sorted(f for f, g in groups.items() if g in first):
        chosen.setdefault(f, f"duplicate leakage group {groups[f]}")

    def has(f: int, pred) -> bool:
        return any(pred(a) for a in anns_by_frame.get(f, []))
    pools = [
        ("hard scene (dark / heavy shadow), high density", lambda f: props[f]["difficulty"] == "hard" and props[f]["n_valid"] > 15),
        ("hard scene, sparse", lambda f: props[f]["difficulty"] == "hard" and props[f]["n_valid"] <= 15),
        ("high uncertainty (excluded from config A)", lambda f: props[f]["uncertainty"] > 0.5),
        ("low uncertainty (config A candidate)", lambda f: 0 < props[f]["uncertainty"] <= 0.25 and props[f]["n_valid"] > 10),
        ("no records at all (negative frame)", lambda f: props[f]["annotation_state"] == "no_records"),
        ("records but no VALID box", lambda f: props[f]["annotation_state"] == "records_no_valid"),
        ("contains watershed split pieces", lambda f: has(f, lambda a: a["source"].endswith("_split") and a["status"] == "VALID")),
        ("CSV rock withheld as invisible in shadow", lambda f: has(f, lambda a: a["reasons"] == "csv_box_in_black_shadow")),
        ("dense scene (36+ VALID)", lambda f: props[f]["density"] == "d3_36+"),
        ("possible merges held for review", lambda f: has(f, lambda a: "merge" in a["reasons"] and a["status"] == "REVIEW_REQUIRED")),
    ]
    for label, pred in pools:
        for split in ("train", "val", "test"):
            pool = sorted(f for f, s in split_of.items() if s == split and f not in chosen and pred(f))
            if pool:
                f = int(pool[int(rng.integers(len(pool)))])
                chosen[f] = label
                if len(chosen) >= 36:
                    break
    return sorted(chosen.items())


def build(out_html: Path, scenes: list[dict[str, Any]], policy_questions: list[dict[str, Any]],
          ai_reference: dict[str, Any] | None = None) -> None:
    data = json.dumps({"scenes": scenes, "policy": policy_questions, "ai_reference": ai_reference or {}}, separators=(",", ":"), sort_keys=True)
    data = data.replace("</", "<\\/")  # keep the inline JSON from closing the script tag
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(PAGE.replace("__DATA__", data), encoding="utf-8")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rock Label Review</title>
<style>
:root{--bg:#f5f5f2;--fg:#1a1a1a;--card:#fff;--muted:#666;--line:#d8d8d8;--accent:#0b5cad;--warn:#b25000}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#131313;--fg:#ececec;--card:#1e1e1e;--muted:#a8a8a8;--line:#333;--accent:#6aa9ff;--warn:#ffae5c}}
:root[data-theme="dark"]{--bg:#131313;--fg:#ececec;--card:#1e1e1e;--muted:#a8a8a8;--line:#333;--accent:#6aa9ff;--warn:#ffae5c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:12px;align-items:center}
h1{font-size:18px;margin:0}.muted{color:var(--muted)}.warn{color:var(--warn);font-weight:600}
.layout{display:grid;grid-template-columns:260px 1fr;min-height:calc(100vh - 60px)}
@media (max-width:860px){.layout{grid-template-columns:1fr}#list{max-height:220px}}
#list{border-right:1px solid var(--line);overflow:auto;padding:8px}
.item{display:flex;justify-content:space-between;gap:6px;padding:6px 8px;border-radius:4px;cursor:pointer;border:1px solid transparent}
.item:hover{border-color:var(--line)}.item.active{background:var(--card);border-color:var(--accent)}
.badge{font-size:11px;padding:1px 6px;border-radius:9px;border:1px solid var(--line);white-space:nowrap}
.b-NOT_REVIEWED{color:var(--muted)}.b-APPROVE{color:#1b8a3a;border-color:#1b8a3a}.b-FLAG{color:var(--warn);border-color:var(--warn)}
main{padding:12px 16px;min-width:0}
.viewer{position:relative;width:100%;max-width:1440px;background:#000;border:1px solid var(--line)}
.viewer img{display:block;width:100%;height:auto;image-rendering:auto}
.viewer.zoom img{image-rendering:pixelated}
.viewer svg{position:absolute;inset:0;width:100%;height:100%}
.controls,.decision{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:10px 0}
button,select,textarea,input{font:inherit;color:var(--fg);background:var(--card);border:1px solid var(--line);border-radius:4px;padding:5px 9px}
button.primary{border-color:var(--accent);color:var(--accent)}button:focus-visible,input:focus-visible{outline:2px solid var(--accent)}
textarea{width:100%;min-height:60px}.legend span{display:inline-flex;align-items:center;gap:5px;margin-right:12px}
.sw{width:26px;height:12px;display:inline-block;border:2px solid}
.box-meta{font-size:12px;color:var(--muted)}.policy{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px;margin-top:16px}
.policy label{display:block;margin:6px 0}
</style></head><body>
<header><h1>Rock label review</h1>
<span class="muted" id="progress"></span>
<span class="warn">Nothing here is human-validated until you set a decision. Default: NOT_REVIEWED.</span>
<span style="flex:1"></span>
<label>Reviewer <input id="reviewer" size="10" placeholder="name"></label>
<button class="primary" id="exportJson">Export JSON</button><button id="exportCsv">Export CSV</button>
<label class="badge" style="cursor:pointer">Import JSON<input type="file" id="importJson" accept=".json" hidden></label>
</header>
<div class="layout"><nav id="list" aria-label="Scenes"></nav><main>
<div class="muted" id="sceneTitle"></div>
<div class="controls">
<label><input type="checkbox" id="tV" checked> VALID</label>
<label><input type="checkbox" id="tR" checked> REVIEW_REQUIRED</label>
<label><input type="checkbox" id="tX" checked> REJECTED</label>
<label><input type="checkbox" id="tXd"> include REJECTED duplicates / split parents</label>
<label><input type="checkbox" id="tB"> show config B (masked) image</label>
<label><input type="checkbox" id="tZ"> 2x zoom</label>
<span class="legend"><span><i class="sw" style="border-color:#00e5ff"></i>VALID (CSV)</span><span><i class="sw" style="border-color:#22d822"></i>VALID (mask)</span>
<span><i class="sw" style="border-color:#ff9a1f;border-style:dashed"></i>REVIEW</span><span><i class="sw" style="border-color:#ff3030;border-style:dotted"></i>REJECTED</span></span>
</div>
<div class="viewer" id="viewer"><img id="img" alt=""><svg id="svg" viewBox="0 0 720 480" preserveAspectRatio="none"></svg></div>
<div class="box-meta" id="hover">Hover a box for its id, source and reasons.</div>
<div class="decision" role="group" aria-label="Scene decision">
<strong>Scene decision:</strong>
<label><input type="radio" name="dec" value="NOT_REVIEWED"> NOT_REVIEWED</label>
<label><input type="radio" name="dec" value="APPROVE"> APPROVE (labels acceptable)</label>
<label><input type="radio" name="dec" value="FLAG"> FLAG (problem)</label>
<button id="prev">&larr; prev</button><button id="next">next &rarr;</button>
</div>
<div class="decision" id="flags" aria-label="Flag categories"></div>
<textarea id="notes" placeholder="Notes for this scene (what is wrong, which box ids)"></textarea>
<section class="policy" id="aiPanel" aria-label="AI reference"><strong>AI-assisted audit (reference only &mdash; NOT a human decision, never counted)</strong><div id="aiBody" class="muted"></div></section>
<section class="policy"><strong>Policy questions</strong> <span class="muted">(dataset is not frozen until these are answered and exported)</span><div id="policy"></div></section>
</main></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const KEY = 'artemis-rock-review-v2';  // v2: provenance-separated (human_ui vs AI reference)
const FLAGS = ['missed_visible_rock', 'valid_box_not_a_rock', 'merged_rocks_in_one_box', 'bad_split', 'review_should_be_valid', 'review_should_be_rejected', 'masking_artifact_config_B', 'other'];
const blank = () => ({decisions: {}, policy: {}, reviewer: '', ai_imported: {}});
let st = blank();
try { const s = localStorage.getItem(KEY); if (s) st = Object.assign(blank(), JSON.parse(s)); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} renderList(); };
const dec = f => (st.decisions[f] && st.decisions[f].source === 'human_ui') ? st.decisions[f] : {decision: 'NOT_REVIEWED', flags: [], notes: ''};
let cur = 0;
const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';

function renderList() {
  const L = $('list'); L.replaceChildren();
  D.scenes.forEach((s, i) => {
    const d = dec(s.frame_id), el = document.createElement('div');
    el.className = 'item' + (i === cur ? ' active' : ''); el.tabIndex = 0;
    const a = document.createElement('span'); a.textContent = `f${String(s.frame_id).padStart(4, '0')} · ${s.split}`;
    const b = document.createElement('span'); b.className = 'badge b-' + d.decision; b.textContent = d.decision;
    el.append(a, b); el.onclick = () => show(i); el.onkeydown = e => { if (e.key === 'Enter') show(i); };
    L.append(el);
  });
  const n = D.scenes.filter(s => dec(s.frame_id).decision !== 'NOT_REVIEWED').length;
  $('progress').textContent = `${n} / ${D.scenes.length} scenes reviewed`;
}

function draw() {
  const s = D.scenes[cur], svg = $('svg'); svg.replaceChildren();
  const showB = $('tB').checked && s.masked_image;
  $('img').src = showB ? s.masked_image : s.image;
  $('viewer').classList.toggle('zoom', $('tZ').checked);
  $('viewer').style.width = $('tZ').checked ? '1440px' : '';
  for (const b of s.boxes) {
    const [x0, y0, x1, y1, status, source, reasons, id] = b;
    if (status === 'VALID' && !$('tV').checked) continue;
    if (status === 'REVIEW_REQUIRED' && !$('tR').checked) continue;
    if (status === 'REJECTED' && (!$('tX').checked || (!$('tXd').checked && /duplicate_of_csv_instance|replaced_by_split_pieces/.test(reasons)))) continue;
    const r = document.createElementNS(NS, 'rect');
    r.setAttribute('x', x0); r.setAttribute('y', y0); r.setAttribute('width', x1 - x0); r.setAttribute('height', y1 - y0);
    r.setAttribute('fill', 'none'); r.setAttribute('vector-effect', 'non-scaling-stroke');
    const col = status === 'VALID' ? (source === 'csv_large' ? '#00e5ff' : '#22d822') : status === 'REVIEW_REQUIRED' ? '#ff9a1f' : '#ff3030';
    r.setAttribute('stroke', col); r.setAttribute('stroke-width', status === 'VALID' ? 2 : 1.6);
    if (status === 'REVIEW_REQUIRED') r.setAttribute('stroke-dasharray', '6 3');
    if (status === 'REJECTED') r.setAttribute('stroke-dasharray', '2 3');
    r.style.pointerEvents = 'all';
    r.addEventListener('mouseenter', () => { $('hover').textContent = `${id} · ${status} · ${source}${reasons ? ' · ' + reasons : ''}`; });
    const t = document.createElementNS(NS, 'title'); t.textContent = `${id} ${status} ${source} ${reasons}`; r.append(t);
    svg.append(r);
  }
}

function show(i) {
  cur = (i + D.scenes.length) % D.scenes.length;
  const s = D.scenes[cur], d = dec(s.frame_id);
  $('sceneTitle').textContent = `frame ${s.frame_id} · split ${s.split} · group ${s.group} · ${s.reason} · VALID ${s.counts.VALID} / REVIEW ${s.counts.REVIEW_REQUIRED} / REJECTED ${s.counts.REJECTED} · ${s.config_note}`;
  $('tB').disabled = !s.masked_image;
  document.querySelectorAll('input[name=dec]').forEach(r => { r.checked = r.value === d.decision; });
  const F = $('flags'); F.replaceChildren();
  const lab = document.createElement('strong'); lab.textContent = 'Flags:'; F.append(lab);
  for (const f of FLAGS) {
    const l = document.createElement('label'), c = document.createElement('input');
    c.type = 'checkbox'; c.checked = d.flags.includes(f);
    c.onchange = () => { const x = dec(s.frame_id); x.flags = c.checked ? [...new Set([...x.flags, f])] : x.flags.filter(y => y !== f); set(s.frame_id, x); };
    l.append(c, document.createTextNode(' ' + f)); F.append(l);
  }
  $('notes').value = d.notes;
  const ai = st.ai_imported[s.frame_id] || (D.ai_reference.scenes || {})[s.frame_id];
  $('aiBody').textContent = ai ? `${ai.decision}${ai.flags && ai.flags.length ? ' · ' + ai.flags.join(', ') : ''} — ${ai.notes || ''} [${ai.reviewer || D.ai_reference.reviewer || 'AI'}]` : 'no AI reference for this scene';
  draw(); renderList();
}

function set(fid, x) { x.reviewed_at = new Date().toISOString(); x.source = 'human_ui'; st.decisions[fid] = x; save(); }
document.querySelectorAll('input[name=dec]').forEach(r => r.onchange = () => { const s = D.scenes[cur], x = dec(s.frame_id); x.decision = r.value; set(s.frame_id, x); });
$('notes').oninput = () => { const s = D.scenes[cur], x = dec(s.frame_id); x.notes = $('notes').value; set(s.frame_id, x); };
for (const id of ['tV', 'tR', 'tX', 'tXd', 'tB', 'tZ']) $(id).onchange = draw;
$('prev').onclick = () => show(cur - 1); $('next').onclick = () => show(cur + 1);
document.addEventListener('keydown', e => { if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight') show(cur + 1); if (e.key === 'ArrowLeft') show(cur - 1); });
$('reviewer').value = st.reviewer || ''; $('reviewer').oninput = () => { st.reviewer = $('reviewer').value; save(); };

const P = $('policy');
for (const q of D.policy) {
  const w = document.createElement('label'); w.append(document.createTextNode(q.question + ' '));
  const sel = document.createElement('select');
  for (const o of ['UNDECIDED', ...q.options]) { const op = document.createElement('option'); op.value = o; op.textContent = o; sel.append(op); }
  sel.value = st.policy[q.id] || 'UNDECIDED'; sel.onchange = () => { st.policy[q.id] = sel.value; save(); };
  w.append(sel); P.append(w);
}

function payload() {
  // only decisions entered in this UI are exported as human decisions; AI references are never exported as such
  return {gallery: 'artemis-rock-review-v2', provenance: 'human_ui', human_review: true, exported_at: new Date().toISOString(),
    reviewer: st.reviewer || '', ai_reference_shown: D.ai_reference.source_sha256 || null,
    scenes: D.scenes.map(s => { const d = dec(s.frame_id); return {frame_id: s.frame_id, split: s.split, decision: d.source === 'human_ui' ? d.decision : 'NOT_REVIEWED',
      flags: d.source === 'human_ui' ? d.flags : [], notes: d.source === 'human_ui' ? d.notes : '', reviewed_at: d.source === 'human_ui' ? d.reviewed_at : null, source: d.source === 'human_ui' ? 'human_ui' : null}; }),
    policy: Object.fromEntries(D.policy.map(q => [q.id, st.policy[q.id] || 'UNDECIDED']))};
}
function download(name, text, type) {
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], {type})); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
$('exportJson').onclick = () => download('review_decisions.json', JSON.stringify(payload(), null, 2), 'application/json');
$('exportCsv').onclick = () => {
  const q = v => '"' + String(v ?? '').replace(/"/g, '""') + '"';
  const rows = [['frame_id', 'split', 'decision', 'flags', 'notes', 'reviewed_at', 'reviewer'].join(',')];
  for (const s of payload().scenes) rows.push([s.frame_id, s.split, s.decision, s.flags.join(';'), s.notes, s.reviewed_at || '', st.reviewer || ''].map(q).join(','));
  download('review_decisions.csv', rows.join('\n') + '\n', 'text/csv');
};
$('importJson').onchange = async e => {
  try { const j = JSON.parse(await e.target.files[0].text());
    const human = j.gallery === 'artemis-rock-review-v2' && j.provenance === 'human_ui' && j.human_review === true;
    if (human) {  // resume a previous human session exported from this gallery version
      for (const s of j.scenes || []) if (s.source === 'human_ui') st.decisions[s.frame_id] = {decision: s.decision, flags: s.flags || [], notes: s.notes || '', reviewed_at: s.reviewed_at, source: 'human_ui'};
      st.policy = Object.assign(st.policy, j.policy || {}); st.reviewer = j.reviewer || st.reviewer;
    } else {      // anything else (AI audit, old v1 export) is loaded as read-only reference, never as human decisions
      for (const s of j.scenes || []) st.ai_imported[s.frame_id] = {decision: s.decision, flags: s.flags || [], notes: s.notes || '', reviewer: j.reviewer || 'unverified'};
      alert('Loaded as AI / unverified REFERENCE only. These are not counted as human decisions.');
    }
    save(); show(cur);
  } catch (err) { alert('Could not read file: ' + err.message); }
};
show(0);
</script></body></html>
"""

# Final repository audit — Artemis-X R6/R6.1

Audit date: 2026-10-11. Base commit: `6258db9` on `master`. Scope: source, existing user frontend changes,
Git/ignore state, dependencies, official workflows and actual simulator/DB/browser checks. The owner explicitly approved
including the existing frontend **after verification**. No training, held-out test reevaluation, model replacement,
R4 mapping change, protocol/schema change, dataset deletion or Windows system-setting change is authorized or performed.

**Verification is still in progress in this draft. Do not interpret pending E2E/checkout/launch checks as passed.**
This banner is replaced only after the evidence is collected. Existing R1–R6 reports and the earlier video report are historical.

## Structure and retention decisions

The baseline had 459 tracked files, no tracked dependency/build/output directories and substantial required untracked frontend
source. A tracked-only checkout did not contain the new focused workspaces. The release includes the verified application
source, components, frontend lock and agent guidance; private environment files and generated media remain excluded.
The recommended tree is the actual tree documented in [README](../README.md); no bulk directory relocation is necessary.

| Item | Classification / evidence / action |
|---|---|
| `artemis-webots/` | **KEEP**: R1/R2 smoke + R3/R4 regressions import `tools/make_world.py`; monitor world references CC0 assets. Worlds/controller behavior preserved. |
| `src/artemis_x/` | **KEEP**: imported by controller, inference, mapping, UDP/DTN and tests; public evaluation wrapper deliberately reuses proven historical evaluation code. |
| `backend/` | **KEEP**: Tasks A–F provenance and training guards; also the locked ML runtime. Protected `pyproject.toml`/`uv.lock` unchanged. |
| `ground_station/` | **KEEP**: actual FastAPI/Mongo ingestion and WS contracts; independent lock/env template. |
| `frontend/apps/web/` | **KEEP / REFACTOR**: approved required untracked routes/shared shell/shadcn/workspaces/tests now included; design and real API integration preserved; bind defaults restricted to loopback. |
| `frontend/apps/docs/` | **INVESTIGATE**, retained: generated Turborepo starter, not the jury dashboard, but discovered by workspace and root tasks; deleting it requires deliberate workspace/docs cleanup. Its lint still must work. |
| `frontend/packages/ui`, shared configs | **KEEP / REFACTOR**: workspace imports/exports and docs app depend on them; ESLint majors aligned to the shared compatible stack. |
| User-requested complete `components/ui/` collection | **KEEP / INVESTIGATE**: many are currently unreferenced; user explicitly requested all shadcn components and created them untracked. No deletion; typed/linted as part of the application. |
| `frontend/apps/web/app/page.module.css` | **REMOVE**: tracked default starter CSS; no web import/reference after the current shell rewrite. The docs app uses its own independent CSS. Removed, then frontend build/types/lint verified. Recoverable from Git. |
| Legacy web SVGs/design-reference media | **INVESTIGATE**, retained: static source scan cannot prove external usage; reference artwork can be intentional evidence, not telemetry. No media removed. |
| `models/`, `artifacts/`, `data/`, notebooks | **KEEP**: frozen SHA, artifact manifest, locked test evidence, training registry and published research notebook validations depend on them. Notebooks do not silently retrain. |
| `dataset/`, `experiments/`, `weights/`, local output | **KEEP locally / ignored**: research evidence or regenerable output, not an obsolete directory merely because it is not needed at runtime. Existing protected local sources were verified; no recursive deletion. |
| `docs/ARTEMIS_X_CURRENT_STATUS.md` | **KEEP unchanged**: user-created historical snapshot dated 2026-10-10 16:15, before later R5/R6 implementation; not a current capability specification. |
| Other R1–R6 reports / screenshots | **KEEP**: historical measurements and contracts, not duplicate disposable logs. README points to current entrypoints without falsifying older measurements. |
| `future/WEBOTS_INTEGRATION_PLAN.md` | **KEEP / INVESTIGATE**: historical plan referenced by docs; current integration is implemented. Label historical; optional archive relocation requires approval. |
| `demo/` | **KEEP**: separate optional capture/edit dependency lock, prior real footage QA and reproducible scripts; generated media/evidence excluded. Unconnected user V2 motion work is retained locally, not treated as current film evidence. |
| `.venv`, `node_modules`, `.next`, `.turbo`, `.jury`, generated captures | **Ignored generated state**: not needed in source submission. No cache purge; the audit avoids disturbing shared caches. |

Duplicate-text review found intentional exported `task_e_experiments.yaml` provenance, empty package markers, shared ESLint
configs and starter README/CSS copies. Exported provenance cannot be deduplicated without breaking the published hash
manifest. The web starter README was replaced with project-specific instructions. No independent duplicate runtime
implementation was removed without evidence. A static import absence is not proof a user-created file is disposable.

Two generic root npm scaffold files accidentally produced during a root package-manager invocation were quarantined in
ignored `.jury/rejected-root-scaffold/`; the authoritative Node workspace remains `frontend/`. They are recoverable, were
not in the initial Git/source inventory and are not user frontend work. No user-created untracked file was deleted.

## Dependency root causes and corrections

- Baseline `@repo/ui` lint: ESLint **10.9.1**, `TypeError: scopeManager.addGlobals is not a function` in JS SourceCode
  finalization using the shared Babel-parser stack. `apps/docs` reproduced the same error. Both consumers now use the same
  **9.39.5** as `apps/web` and shared config; `@eslint/js` is aligned to 9.39.5 too. No lint rule was disabled. pnpm lock
  regenerated without discarding existing approved frontend additions. `pnpm peers check`: no peer issues. ESLint 9 is
  registry-deprecated; future ESLint 10 migration requires a compatible parser stack, not merely bumping the consumer.
- Baseline root Python suite: **200 discovered, 1 import error, 2 skips**. The notebook smoke imported `build_notebooks`,
  which unconditionally imports `nbformat`; the local optional import/skip in the test could not protect that import.
  Required notebook-test tooling is now declared in `tests/requirements.in` and a fully pinned/hash-bearing compiled
  `tests/requirements.txt`, layered with `uv run --with-requirements`. **Neither protected ML file was changed.** Tests
  retain their assertions; no notebook validation is disabled. `nbclient`/`ipykernel` remain optional generation tools,
  not dependencies required to inspect the already executed committed notebooks.
- Ground dependencies are independently locked; dev HTTP/WebSocket tooling is installed by `uv sync --locked --group dev`.
- New scene acquisition `fetch_polyhaven.py --locked` uses committed source URLs and SHA256, retains the manifest and
  validates derived textures. The dynamic historical fetch remains explicit, not the jury acquisition path.
- Next/Turborepo bundled installed docs and managed `AGENTS.md` instructions were read before altering relevant commands.
  No Turbo configuration or application layout was rewritten for this audit.

## Execution-script audit

`scripts/release_inventory.py` reads every submission script source, parses Python ASTs, inventories argument definitions,
imports, subprocess/finally usage and source/doc references. Its generated inventory is review evidence, not proof that
all historical commands were dynamically rerun. Official entrypoints now isolate unique evidence and return nonzero failures.

| Scripts | Purpose / jury disposition / relevant boundary |
|---|---|
| `jury.ps1`, `jury.py` | **Official** Windows preflight/unit/test/start/stop/demo/status; repo-relative commands, unique outputs, checked readiness, explicit dashboard reuse, owned-process identities and preserved Mongo volume. |
| `jury_unittest.py` | **Official** exact pass/fail/error/skip JSON; missing Mongo integration cannot count as a passing full suite. |
| `r6_dashboard_e2e.py` | **Official real-stack regression**, reused and repaired: empty-DB real historical seed, real presets, non-tautological YOLO spatial check, hydrated-map synchronization, current shadcn sheet switch/navigation, 55/55 completion wait, no overwrite, owned cleanup. |
| `demo_atec.ps1` | Optional manual per-terminal filming service launcher; safe foreground operations and port checks. Not a second simultaneous 5790 receiver. |
| `inference_server.py`, `lunar_link.py` | Existing runtime entrypoints; frozen model, loopback inference and emulator/reverse ACK controls; retained. `--seed` keeps emulator RNG deterministic, not a guarantee of deterministic wall-clock scheduling. |
| `ground_receiver.py` | Standalone R5 receiver/diagnostics; **do not run alongside FastAPI** on 5790. Appended telemetry log and optional final stats/map files can overwrite their chosen paths; use a new run directory. |
| `r5_ws_capture.py` | Real WS diagnostic; explicit URL/output; not a synthetic telemetry seed. |
| `run_webots_smoke.py` | R1/R2 real simulator sensor/spatial smoke; enabled in full suite. Timestamped output. Historical timeout uses `subprocess.run`; wrapper enforces a larger overall deadline. |
| `run_webots_r3.py` | Historical inference/domain-gap/tour runner, port-specific server lifecycle, owned Webots timeout kill; keep regression CLI. Not the current jury stack launcher. |
| `run_webots_r4.py` | Historical mapping/safety/fusion/monitor/telemetry scenario orchestration; keep R4/R5 behavior. Some legacy startup/error branches kill an `uv` parent rather than prove all descendants exited; not used as official jury startup. Investigate hardening separately. |
| `r4_validation.py`, `r4_fusion_analysis.py`, `r4_monitor_analysis.py`, `r4_safety_analysis.py` | Offline R4 aggregators/plots/checks; consume their own historical run structure and can replace derived summaries/figures. Preserve originals; use copied/new evidence for reruns. |
| `r5_telemetry_analysis.py`, `r5_dtn_analysis.py`, `r5_ground_e2e_analysis.py` | Offline protocol/transport/DTN/ground evidence analysis; different stages, not interchangeable duplicate scripts. Require matching run layout. |
| `r5_telemetry_benchmark.py` | Binary protocol size/encode benchmark; can replace its chosen output summary. Historical performance comparison tool, not a live backend. |
| `verify_artifacts.py` | Read-only integrity by default; **do not use `--baseline` during release verification**, because that replaces the protected comparison baseline. |
| `inspect_environment.py`, `audit_workspace.py`, `retention_manifest.py`, `release_inventory.py` | Environment/storage/retention/Git diagnostic inventory; report outputs are writes, no automatic cleanup. Older reports are dated snapshots. |
| `demo_inference.py` | Frozen detector on supplied images, optional explicit output; not a training/test-split evaluation workflow. |
| `download_dataset.py` | Approved-source verification/extraction, destination protection and checksums; public dataset source currently unavailable by design. No invented download URL. |
| `build_notebooks.py` | Optional regeneration/execution with forbidden training/test calls and output privacy checks; replaces notebook files, so not run just to make a release test pass. |
| `build_release_manifest.py`, `export_artifacts.py` | Scientific release/export maintenance; may rewrite published metadata/evidence. **Not invoked** in runtime release preparation. |
| `sanitize_checkpoint.py` | Explicit separate distributable copy; cannot silently replace the frozen checkpoint/hash. Not invoked. |

Project package scripts were checked too: frontend dev/start now loopback, test uses Node's real unit runner, quality commands
run all workspace consumers uncached; docs starter still builds. `generate:component`/format are developer generators and
write sources, not release verification. Optional demo capture/render/QA require separately generated inputs; its TypeScript
checks passed, but this audit does not claim a new MP4 render. The existing film remains historical QA evidence.

## Fresh verification evidence

Initial command: `.\scripts\jury.ps1 -Step preflight`; actual prerequisites/model/assets/Docker/Mongo passed, with a WARN
for the operator's existing dashboard on 3000. That process is reused only by the explicit `-ReuseDashboard` option.

| Check | Fresh outcome |
|---|---|
| Runtime + enabled Webots smoke | 202 discovered, **201 passed**, 0 failures/errors, 1 optional historical parity skip |
| Historical research regression | **52/52**, 0 skips in the existing research workspace |
| Real Mongo/UDP/FastAPI/WS integration | **9/9**, no skips; disposable `artemis_test_*` databases owned by tests only |
| Frontend reducer/dashboard summaries | **23/23**, no skips |
| Workspace lint/types/build | **8/8 tasks**, uncached; web routes and retained docs starter build |
| Dependency peers | No peer dependency issues after ESLint alignment |
| Optional video source types | `cd demo; npx tsc --noEmit`: exit 0 |
| Locked assets + protected provenance | Locked conversion exit 0; 71 published manifest entries, 17 protected baseline files and frozen checkpoint hashes pass; no new training registry entry |
| First R6 attempt | 13 recorded checks passed, then **exit 1**: missing synchronization before browser reload; this is a failed attempt, not overall success |
| Repaired R6 + 55/55 tour | PENDING |
| Isolated checkout install/build | PENDING |
| Official persistent start/stop | PENDING |

The historical-parity skip's message is `local experiments/dataset not present`; this does **not** mean all local research
folders are absent. The published model and retained protected files passed integrity checks. No guarantee of complete
historical cached-prediction reproduction is made. Raw attempts/logs/screenshots remain in ignored `.jury/runs/`; compact
sanitized reports are retained with this audit after final verification. Fresh results above are not copied from older R6/video reports.

## Clean-checkout reproducibility and security boundaries

- Required formerly untracked frontend source must be committed; installed `node_modules` cannot substitute for source.
- A fresh checkout needs configured private env files, installed Webots/Edge/Docker, network-accessible package registries
  and locked downloaded CC0 assets. The model itself is a tracked 5.5 MB frozen file with SHA256
  `413d0481a728a310f5b919d069676ee5ddd2091e81b4b663831e1e977a0752eb`.
- Existing mission data is **not** required: automated demo uses a new DB and real short Webots historical seed. User
  mission DB and existing services are not dropped or force-replaced. Test-owned ephemeral databases are the only databases
  dropped by the ground suite; the jury demo DB is retained as evidence.
- Real `.env`, private-key and dependency/output trees are excluded by Git. The source inventory found no matching
  credential/token/private-key patterns in reviewed candidate text; this limited pattern scan is not a penetration test or
  a universal proof that every possible secret encoding is absent.
- Compose Mongo is bound to 127.0.0.1; official ground/inference/emulator/default dashboard bind locally. The original
  already-running dashboard was listening on `::`; it was **not** killed or silently replaced. The operator must close/restart
  that process to apply the new loopback default. Existing reused service exposure is explicitly not fixed by ownership cleanup.
- Network preset endpoints enforce local clients; the real ground tests verify refusal for a nonlocal TestClient peer.
- Frozen weights retain embedded historical training-machine paths. Changing the binary would invalidate its scientific
  hash; the existing separate sanitizer is an explicit alternative requiring a new distribution decision, not an audit shortcut.
- Project license is not selected; dataset redistribution approvals remain pending in `data/release/dataset_release.json`.
  CC0 scene authors/URLs/hashes are recorded. Existing model/vendor license declarations are retained; no new license was chosen.

## Remaining proposals and limits

1. Owner decision on project/model distribution terms and licensing. No claim of unrestricted commercial distribution.
2. Authorized historical dataset archive/hosting and cached predictions if complete training/parity reproduction is required.
   There is no approved public archive URL/checksum to pretend to supply; per-file release metadata is tracked.
3. Optional removal/archive of the monorepo docs starter, unused web starter SVGs and unused requested shadcn components:
   requires deliberate approval and reference/workspace checks; none are deleted here.
4. Legacy R3/R4 process lifecycle and analysis-output hardening can be reviewed independently. The new official path avoids
   selecting/killing unrelated processes; this audit does not certify every old CLI command under all timeout conditions.
5. Simulator-only scientific caveats remain: automated labels, synthetic-only benchmark, scripted noncollision navigation,
   advisory safety, no live camera/full-grid downlink, bounded/coalesced DTN queues and ACK-before-durability.

Commit/push outcome and final evidence are filled after the required regressions. Critical unresolved execution failures
must prevent a “jury ready” claim and push; licensing/research-data boundaries must remain visible even if runtime tests pass.

# R6.1 — Clean workspace and Unified Mission Control

Mission Control now has two display modes over the same live ground-station session.

| Mode | When | What you get |
|---|---|---|
| **Clean workspace** (default) | Settings › *Unified Mission Control* off | Focused pages: Overview, Rover, Mapping, Detection, Telemetry, Communication, Missions, Logs, Settings |
| **Unified Mission Control** | Settings › *Unified Mission Control* on | The R6 all-in-one command center on `/`, unchanged: map, overview, analytics, telemetry, network, history, log and rover panels together |

Every value in both modes still comes from the R5.5 ground station (REST + WebSocket). No mock or interpolated data is
shown, and the clock rules of R6 are unchanged:
- `t_obs_s` is the rover observation time (simulation seconds);
- `rx_utc` is the ground reception time (UTC);
- freshness (LIVE / STALE / DISCONNECTED / NO DATA) is computed from UTC only.

## Switching modes

*Settings › Workspace & Display Preferences › Unified Mission Control* is a shadcn/ui Switch. Its description reads:
"Display all mission systems in a single command center instead of separate focused pages."
- The change applies immediately. The header shows the active mode as a small chip ("Clean workspace" or "Unified console") that links to Settings.
- The preference is stored in `localStorage` (`artemis.workspace` = `clean` | `unified`) and restored on reload. If storage is blocked, the default (clean) applies.
- **No hydration mismatch and no flash.** The preference is read with `useSyncExternalStore`. Its server snapshot is `null`, so during server render and hydration `/` renders an empty placeholder instead of guessing a mode. The stored mode appears on the first client render.
- **The selected mission is preserved.** A pinned mission (`?mission=<id>`) and "follow the newest mission" live in the console-wide session, so they survive both page navigation and mode changes.

## Architecture (`frontend/apps/web`)

```
app/layout.tsx            <MissionProvider>            one session: selection, WebSocket, network control, clock, mode
                            <ConsoleShell>             R6 header + sidebar around every page (both modes)
app/page.tsx              "/" → OverviewPage (clean) | UnifiedDashboard (unified)
app/{rover,map,detection,telemetry,network,missions,logs,settings}/page.tsx → dedicated pages
components/mission/
  unified-dashboard.tsx   the R6 command center, panels moved verbatim from the old page.tsx
  shell.tsx               ConsoleShell: header (freshness pill, mode chip), sidebar (links or panel scroll)
  header.tsx              responsive header, live UTC clock, status/mode and compact mission-clock popover
  console-nav.tsx         navigation model: label → route (clean) / panel id (unified)
  shared.tsx              brand artwork, glyphs, formatting, Panel / pills / empty states
  lunar-map.tsx           the live map (shared by both modes)
components/workspace/     clean pages + primitives (page header, sections, KPI, signal rail, facts, meters)
lib/ground/mission-context.tsx   MissionProvider, useMission(), workspace preference store
app/workspace.css         clean-workspace visual system
app/header.css            shared responsive header; sidebar offset tracks the measured header height
```

- **One WebSocket for the whole console.** The provider lives in the root layout. App Router layouts persist across navigation, so changing pages never opens a second socket and never loses the picture already received.
- **Sidebar.**
  - Eight primary entries match the reference: Overview, Rover, Mapping, Detection, Telemetry, Communication, Logs, Settings. Mission history remains accessible from the header's mission link, Overview's mission record, and Rover's history action.
  - The dedicated lunar terrain asset, cyan active rail, icon placement, and footer typography follow the reference. Desktop collapse/tooltips and the mobile drawer remain available.
  - In the clean workspace, items are links to the dedicated pages. A pinned mission is kept in the query string.
  - In unified mode, on `/` the items scroll to the matching command-center panel, as in R6, with scroll spy. From any other page they link back to `/#panel`. Settings is a page in both modes.

### Responsive navbar

The shared navbar uses a fluid grid instead of fixed-width title/badge rows. Wide screens show UTC, mission and the
last received rover simulation clock inline. Below 1440 CSS pixels, a shadcn Popover exposes these facts and the
archive action. Below 1024 CSS pixels, navigation becomes a drawer and status/mode controls occupy a second row;
below 640 CSS pixels, the title becomes a compact two-line lockup. The mode link and freshness status stay visible.
Decorative lunar imagery is subdued and cannot obscure controls. A ResizeObserver keeps the desktop sidebar below
the header's actual height, including wrapping and browser zoom.

Navbar QA covered 59 browser assertions: viewport/DPR pairs equivalent to 80–250% zoom on four desktop widths,
responsive boundaries from 320 to 1440 pixels, long mission IDs, popover geometry/Escape/focus, UTC updates,
mobile navigation, mode changes, pinned links and a scrollable context panel on short zoomed viewports. TypeScript,
all 23 unit tests and the production build passed.

## Clean workspace pages

- **Visual system.**
  - Palette:

    | Role | Name | Value |
    |---|---|---|
    | Background | Void | `#050B14` |
    | Surface | Hull | `#0A1626` |
    | Borders | Seam | `#1B2E45` |
    | Accent | Signal | `#4CB4FF` |
    | Text | Bright | `#EEF4FB` |
    | Secondary text | Regolith | `#93A9C4` |
    | Status: nominal | | `#2FD69B` |
    | Status: caution | | `#F2B13D` |
    | Status: fault | | `#FF5A6E` |

  - Type: Geist Sans for titles and prose, Geist Mono for every readout and timestamp.
  - Layout: a 12-column grid, fewer and larger surfaces, one primary visual per page.
- **Signature, the signal rail (Overview):**
  - The rover's observation clock (sim) and the ground reception clock (UTC) sit at the two ends of a link line.
  - The line carries the reception age: a travelling pulse when LIVE, a hatched amber line when STALE, a broken red line when DISCONNECTED.
  - It shows the system's central rule, two clocks never mixed, at a glance.

| Page | Content |
|---|---|
| Overview | Signal rail. Four KPIs (obstacles on the map, rover health, distance along the received path, link delivery). Trajectory preview with a link to the live map. Rover facts. Recent activity (last 6 events). Mission record |
| Rover | Received health, odometry position, heading and uncertainty. Platform illustration explicitly labelled as decorative. Links to engineering telemetry, mapping and mission history; no invented hardware readings |
| Live map | Large interactive map with zoom, pan, fit and follow, a floating collapsible legend and a STALE / DISCONNECTED banner that keeps the last picture. Side panel with rover facts, the selected obstacle (confidence, uncertainty, provenance, observation and reception times) and the nearest obstacles |
| Detection | KPIs. Reception-time activity chart. Evidence-source composition. Searchable, filterable, sortable obstacle catalog. Per-obstacle evidence sheet (geometry relative to the last received pose, provenance, both clocks, related events). Retained detection history. States plainly that no video is downlinked |
| Telemetry | Position, heading, uncertainty, data age. Position over observation time (gaps are not bridged). Packet reception counters. Reception rate over UTC. Latest stored packets (REST, every 3 s). Expandable technical details (stream, resyncs, clock offset, ingest queue, receiver, protocol) |
| Communication `/network` | Live controller condition kept separate from the pinned mission's historical evidence. Five preset cards; confirmation requires a fresh `GET /network/status`. Loss/delay/throughput/blackout readouts, traffic and delay chart tabs, retained-sample windows, outcome donut, receiver facts, searchable/sortable/paginated evidence ledger and sample inspection sheet. Controls disabled on controller errors, including after a previously successful response |
| Missions `/missions` | Archive KPIs, session-comparison chart, current-console record and searchable/filterable/sortable/paginated shadcn Data Table. Row clicks open a dossier with overview, receiver and normalized record tabs without changing selection. Explicit “Use this mission”, follow-newest and refresh actions |
| Logs `/logs` | Retained-event KPIs, UTC severity-activity chart and interactive subsystem composition. Category/severity filters and search shared by a paginated shadcn Data Table and timeline. Event inspection sheet includes both clocks, same-category context and normalized record; counts describe the bounded retained window, not a complete audit archive |
| Settings | Workspace & Display Preferences (Unified Mission Control, motion effects), mission selection (follow newest), ground station endpoints and health |

## Validation

See the R6.1 section of `docs/DASHBOARD_R6_MISSION_CONTROL.md` for the full-stack run and screenshots.

The focused-page enhancement was additionally checked with TypeScript, a production build, 23 unit tests and 49 browser assertions over persisted mission reads. Browser checks covered desktop/mobile layouts, table and timeline pagination, filters/search, inspection sheets, explicit mission switching, pinned navigation, sidebar collapse and both display modes. A further 19 network assertions used isolated browser response fixtures for NORMAL, DEGRADED, SEVERE, BLACKOUT, RESTORE, pending/failed confirmation, controller failure and recovery; no preset request was sent to the real emulator during these checks. Eight final assertions checked explicit API-down states, unknown session counts, disabled controls and short-screen navigation. The temporary read-only QA API was then left stopped; the existing frontend development server was not stopped.

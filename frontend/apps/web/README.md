# Artemis-X Mission Control

The production application is `frontend/apps/web` (Next.js 16.3.8 / React 19, Tailwind 4, shadcn, Lucide, Recharts,
TanStack tables and Motion). It displays the real R5.5 ground API, not mock telemetry.

From `frontend/`: `pnpm install --frozen-lockfile`, `pnpm --filter web dev` (localhost:3000),
`pnpm --filter web test`. Shared lint/type/build: `pnpm exec turbo run lint check-types build --force --ui=stream`.
Use Node >=24 and the pinned pnpm 11.25.0. Do not run a second dashboard during the official stack workflow.

See [jury quickstart](../../../docs/JURY_QUICKSTART.md), [R6 contracts](../../../docs/DASHBOARD_R6_MISSION_CONTROL.md)
and [R6.1 workspaces](../../../docs/DASHBOARD_R6_1_WORKSPACES.md). Optional `.env.local` comes from `.env.example`;
public API/WS URLs are inlined at build time. Stale radio telemetry and disconnected ground transport are distinct states.
Sidebar artwork is decorative, not a live lunar camera image. `apps/docs` is a retained monorepo starter, not Mission Control.

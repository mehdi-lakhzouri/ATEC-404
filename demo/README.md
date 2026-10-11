# Artemis-X — reproducible product demo

The deliverable is `output/artemis-x-demo-final.mp4`: Full HD, 30 fps, H.264, captioned and music-free. It is generated from actual browser recordings, actual onboard panel captures and verified emulator controls. Media and run-specific evidence are ignored by Git; keep them with the submission, not in source control.

## Reproduce on this Windows workstation

Prerequisites: existing Python virtual environments in `backend/` and `ground_station/`, the frozen model in `models/`, installed Webots R2025a, Microsoft Edge, Node ≥24, installed frontend dependencies and healthy `artemis-mongodb` Docker container. Use the repository's `docs/DEMO_ATEC_26.md` for the service setup. Do not train or modify the model for filming.

```powershell
cd demo # from the repository root
npm ci
npm approve-scripts ffmpeg-static esbuild
npx playwright install ffmpeg
npm run capture
npm run render
npm run qa
```

Capture starts its own ground API, emulator, inference service and rover. It reuses an existing dashboard or starts one if port 3000 is free. It refuses to replace an API/inference service already listening. MongoDB must already be healthy. Cleanup confirms NORMAL and stops only owned processes. A full tour can take approximately nine minutes, longer than the edited film. The recording finishes first; capture waits for final rover evidence.

Browser automation uses real sidebar links, settings switches, detection rows, evidence tabs, map zoom, real preset buttons, historical mission selection and Unified Mission Control. It checks API confirmations and records actual WebSocket events. It does not inject telemetry, intercept responses, set React state, invent detections, or alter the tour.

## Webots footage and review

Capture records the desktop with its owned Webots window brought forward; avoid desktop interaction during filming. Inspect the raw native footage before including it. Window-based capture can omit OpenGL content; preserve any such attempt as rejected evidence and omit it from the edit. While capture's owned Webots is running, the optional supplemental desktop capture below brings that window forward and records 30 seconds.

```powershell
node scripts/capture-native.mjs --desktop
```

Approve only a visually verified take by saving a JSON object with its `file` path (relative to `demo/`) as `native-approved.json` inside that run's evidence directory. The successful take uses the metadata from `native-desktop.json`. The current crop assumes a 1920×1080 maximized Webots layout; adapt the crop in `prepare.mjs` if your layout differs. Actual camera, AI, depth and local-map PNGs are a labelled fallback; no screenshot is called a live stream. The standard film remains complete without approved native footage.

## Editable motion project

```powershell
npm run studio -- --props public/timeline.json
node scripts/render.mjs --preview
```

Source: `src/ArtemisDemo.tsx`. Titles use the existing Artemis-X SVG mark, a restrained cyan/navy palette, 20-frame ease-out entrances and hierarchy-based staggering. The requested Figma Motion skill informs this choreography; no Figma file was provided or altered. Figma motion APIs are not required for reproduction.

Generated `public/timeline.json` contains frame-accurate chapter positions. `edit-manifest.json` records source paths, source timestamps, trims, and speed (always 1×). Page navigation is edited out. Blackout and restore remain in source order. A historical mission is explicitly labelled and the ending returns to the new live mission.

## Evidence / acceptance

- `recordings/<take>/`: raw browser WebM and available native Webots MKV.
- `evidence/runs/<take>/capture.json`: screenshots, real preset confirmations, WebSocket evidence, pose freeze, restore timing and cleanup.
- `evidence/runs/<take>/link.jsonl`: emulator packet events, including dropped and retransmitted session/sequence identities.
- `evidence/runs/<take>/tour/summary.json`: final controller/DTN/monitor evidence when the tour completes.
- `output/qa/`: render previews, extracted final frames, metadata, full-decode and browser-playback checks.

Run QA after rendering, then visually inspect the final frames and watch the MP4. Verify small labels, contrast, sheet content, map uncertainty, source clock vs UTC, STALE during blackout, Live after RESTORE, historical selection, and the final unified view. Automated checks do not substitute for visual review.

Limitations are intentional: this is a simulation demonstration; the tour is scripted rather than perception-driven avoidance; downlinked maps are received odometry/obstacles rather than live camera or full occupancy grids. Low-priority poses are coalesced, not a complete blackout replay. ACK does not prove database durability. No zero-loss or exactly-once claim is made. The film has readable English captions and no audio track.

# Artemis-X — demo capture report

Successful take: `2026-10-11T01-14-29-226Z`. Live mission: `1791681282`. Historical mission: `1791681188`, explicitly labelled and pinned before returning to the new live session.

The browser recording ran approximately 191 seconds after its initial epoch (2026-10-11 01:14:37 UTC). The complete capture workflow, including startup and waiting for the rover's final report, took 559.6 seconds (9 min 20). The edited film is 177 seconds: 1920×1080, 30 fps, H.264. It includes a branded introduction, clean overview, detection inspection and evidence tabs, actual Webots footage, labelled onboard RGB/YOLO/depth/map snapshots, mapping selection and zoom, telemetry, constrained network presets, blackout, restore, archived mission and unified console, then a branded fade to black. All footage is played at 1× speed.

## Evidence from the real run

The UI confirmed NORMAL → DEGRADED → SEVERE → BLACKOUT → RESTORE against the emulator's real API. Ground freshness became STALE, not disconnected, during blackout. Two API overview snapshots taken five seconds apart contained the same latest received observation time. Their full payloads are retained in `capture.json`.

The emulator's cumulative blackout interval was approximately 20.4 seconds. The UI returned to Live approximately 0.50 seconds after RESTORE confirmation. This is a measured UI recovery interval for this run, not a promised network latency or a subtraction of simulation time from UTC.

The raw emulator journal identifies obstacle packet sequence `262`, session `1791681282`, observation time `100.48 s`, length `234 bytes`: repeatedly dropped during BLACKOUT and forwarded at 2026-10-11 01:16:58.939 UTC with preset RESTORE. This is genuine same-sequence retransmission evidence. It does not prove every observation was recovered or persisted exactly once. No blanket “critical” classification is asserted from a journal that does not encode priority.

The frozen model SHA-256 remains `413d0481a728a310f5b919d069676ee5ddd2091e81b4b663831e1e977a0752eb`. No model training, backend change, world modification, protocol change or schema change was used for the video.

## Capture adjustments and limitations

The first native capture failed because the Webots window's height was odd. Padding corrected the encoder error, but window-based GDI capture omitted the OpenGL viewport. Those rejected recordings are retained for provenance. Desktop capture with the actual Webots window brought forward succeeded; the approved excerpt is cropped to the simulator viewport. No desktop notification or terminal appears in that excerpt.

There is no usable native moving-rover clip from the blackout interval, so none is fabricated. The blackout chapter uses the genuine last-received ground map and its increasing UTC reception age. The onboard four-panel bridge is a clearly labelled snapshot triggered at ≥75 simulation seconds; panel headers preserve their individual sample times. It is not represented as a live downlink or perfectly synchronized sensor sample.

The course is scripted wheel-odometry motion, not perception-driven collision avoidance. Downlinked mapping shows received poses and obstacle records, not a full occupancy-grid or camera stream. Health STOP is a perception advisory, not proof the motor controller stopped. Low-priority poses are coalesced and are not all replayed after a blackout. Receiver ACK is not a database durability guarantee.

The edition is captioned in English and silent, with no unlicensed music or synthesized voice. Historical records can be stale; that is displayed honestly. Source navigation is edited out, and short dissolves smooth chapter boundaries without changing telemetry values. The original browser and native clips, timing manifest, backend confirmations, final controller report and QA artifacts are retained under the ignored media/evidence directories.

The requested Figma Motion skill influenced easing, hierarchy and staggered entrances in the editable Remotion composition. No Figma target file was supplied, so no Figma document or motion track was created or modified.

## Final validation

Accepted file: `output/artemis-x-demo-final.mp4`, 30,754,961 bytes, exactly 177 seconds / 5310 frames. All 15 automated acceptance checks passed. The entire stream decoded successfully and Microsoft Edge played it with a progressing video clock and no media error. Encoded-frame visual review covered the full-film contact sheet plus detection, simulator, map, network, blackout, restore, archive, unified and ending samples. No blank simulator viewport, browser error overlay, terminal, notification or unexpected mission ID appears in the selected footage. The brief's intentional final fade to black is retained.

`npm run qa` checks H.264, 1920×1080, 30 fps, duration, full-stream decoding, browser playback, capture errors, preset order, pose freeze, model integrity and same-sequence retransmission. It extracts beginning/middle/end frames of every chapter for visual review. Machine-readable results are in `output/qa/report.json`; visually reviewed images are in `output/qa/final/`.

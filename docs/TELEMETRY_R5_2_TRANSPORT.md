# Artemis-X Live Telemetry over UDP (R5.2)

| | |
|---|---|
| Status | Implemented and validated live in Webots. No database, dashboard, network emulator or retransmission (later sprints) |
| Code | `src/artemis_x/telemetry/transport.py` (`UdpSender`, `GroundReceiver`, `LiveTelemetry`); `adapter.py` (change filter); controller hook in `src/artemis_x/sim/rover.py` |
| Receiver | `scripts/ground_receiver.py` |
| Tests | `tests/test_telemetry_transport.py` (6 tests, real localhost UDP), plus the 16 R5.1 protocol tests |
| Live validation | `scripts/run_webots_r4.py telemetry` → `scripts/r5_telemetry_analysis.py`; run `artemis-webots/output/r5-telemetry-20261010-214035` |

## 1. Packet flow

```
Webots controller (stdlib)                                   Ground station (separate process / thread)
 mapping cycle (unchanged) --> R4 state (grid, tracks, health, pose)
 idle step --> LiveTelemetry.tick(t)
                 TelemetryAdapter: one job per tick
                   map_reset event > health event (on change) > POSE (2 Hz) > OBSTACLES (every 2 s, idle step)
                 protocol v1 encode (CRC16) --> bounded queue (drop-oldest)
                 UdpSender: non-blocking sendto, max 32 packets per tick --> UDP 127.0.0.1:5790 --> GroundReceiver
                                                                                   CRC check + strict decode
                                                                                   rx wall time, seq gap / dup / late
                                                                                   one JSON line per datagram
```

- **Opt-in.** `ARTEMIS_TELEMETRY=1` enables it. Without it the controller behaves exactly as in R4.6, and every validated run is unaffected.
- **Idle steps only.** All telemetry work runs on idle simulation steps (no mapping cycle and no grid update in that step), next to the R4.5 monitor. The mapping cycle is untouched.
- **Separate from YOLO.** UDP telemetry uses its own socket and port; the YOLO inference link (TCP 5761) is a separate connection.
- **Unchanged observations are not re-sent.** After a full snapshot, the adapter sends only obstacles that are new or changed: position (1 mm), radius, status, protection, confidence (0.01), uncertainty (1 cm) or track links. Re-detections that change nothing are skipped. A **full** snapshot goes out every 10 s, so a restarted receiver recovers the whole picture; also right after a map reset, and at mission end.
- **Duplicate physical observations** are removed as in R5.1: a depth cluster that is the same object as a track is not sent.

## 2. Commands

```powershell
# 1. Ground receiver (any terminal; ML environment not needed)
python scripts/ground_receiver.py --out artemis-webots/output/ground/telemetry.jsonl --stats artemis-webots/output/ground/stats.json
# 2. YOLO inference server (second terminal)
uv run --project backend --no-sync python scripts/inference_server.py --port 5761
# 3. Webots with telemetry on (third terminal)
$env:ARTEMIS_TELEMETRY = "1"            # optional: ARTEMIS_TELEMETRY_HOST / _PORT (default 127.0.0.1:5790) / _SESSION
& "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" --mode=realtime artemis-webots\worlds\artemis_test_course.wbt
# Ctrl+C on the receiver writes its stats. Automated live validation (4 tours, ~5 min):
uv run --project backend --no-sync python scripts/run_webots_r4.py telemetry
```

Each JSONL line contains:
- `rx_wall_s`, the reception time on the ground's wall clock;
- `session`, `seq`, `type`;
- `t_obs_s`, the observation time on the rover's simulation clock;
- `seq_flag`: `first`, `ok`, `gap:N`, `duplicate` or `late`;
- the decoded `body`: odom frame, ids, coordinates, radius, confidence, uncertainty, source and status.

An invalid datagram gives a line with its rejection reason instead.

## 3. Live validation (Webots → UDP → ground receiver)

Run `artemis-webots/output/r5-telemetry-20261010-214035`: 4 full course tours, YOLO on, Webots fast mode, same machine.

| | `fusion` (telemetry off) | `telemetry` | `telemetry_restart` | `telemetry_noreceiver` |
|---|---|---|---|---|
| Tour | 55/55 | 55/55 | 55/55 | 55/55 |
| Mapping cycle p95 | 12.15 ms | 11.71 ms | 12.17 ms | 11.59 ms |
| Busiest step max | 17.5 ms | 17.3 ms | 16.6 ms | 15.3 ms |
| Controller memory at end | 72.0 MB | 74.7 MB | 74.9 MB | 74.4 MB |
| YOLO answers | 2,001 / 2,001 | 1,994 / 2,001 | 1,997 / 2,001 | 1,996 / 2,001 |
| Packets sent (send errors) | — | 1,307 (0) | 1,307 (0) | 1,307 (0) |
| Received valid / invalid | — | **1,307 / 0** | 1,094 / 0 (two receivers) | — |
| Gaps / duplicates / late | — | 0 / 0 / 0 | 0 / 0 / 0 within each receiver; 213 packets sent while the receiver was down are lost | — |

- **Telemetry costs no measurable mapping time.** Cycle p95 is within run-to-run variation (11.6–12.2 ms against 12.2 ms with telemetry off). Controller memory is +2.5 to +2.9 MB, which is the send log.
- **YOLO.** Answers stay at 1,994–1,997 of 2,001, inside the range seen across R4 runs (1,993–2,001).
- **Receiver unavailable.** The mission completes normally (55/55). On Windows, localhost UDP to a closed port raised no send error in these runs; the sender would catch and count it if it did.

**Content correctness** (`telemetry` scenario):

| Check | Result |
|---|---|
| Packets by type, sent → received | POSE 1,001 → 1,001 · OBSTACLES 261 → 261 · EVENT 45 → 45 |
| Decoded POSE vs the controller's own odometry at the same simulation time | 1,001 poses, max difference **0.5 mm** (the 1 mm quantisation) |
| Ground obstacle picture vs the run's final export (deduplicated) | 41/41 obstacles present, max position difference **0.0 mm**, frame `odom`, session preserved |
| First reception | 0.032 s simulation time (the first pose); wall 1,791,664,908.16 s |
| Mission end (last mapping cycle) | wall 1,791,664,965.58 s, **57.4 s after the first reception** |
| Packets received before mission end | **1,303 of 1,307** (the last 4 are the end-of-mission flush) |

**Latency and throughput:**

| Measure | Result |
|---|---|
| Send → receive latency (send log vs `rx_wall_s`, same machine and clock) | **95.8 % ≤ 1 ms**; median below the ~1 ms resolution of the Windows wall clock; max 6.7 ms (restart run: max 7.3 ms) |
| Throughput (fast mode) | 22.8 packets/s, 1,957 B/s wall clock; **219 B per simulated second** (112,371 B for a 512 s mission) |
| Change filter | 4,005 unchanged obstacle observations not re-sent; 52 full snapshots, 200 changes-only snapshots |
| Deduplication | 1,528 depth-cluster observations not sent as duplicates of tracked objects |

The packet counts are identical across the three telemetry runs, as expected: the tour and the map are deterministic, and only the reception differs.

**Restart.** The first receiver stopped at about 15 s (wall) after receiving 273 packets. A second receiver started on the same port at about 25 s and received 821. The rover never stopped. The ground picture at mission end was complete (41/41 obstacles), thanks to the periodic full snapshots.

## 4. Known limitations

- **No retransmission or acknowledgement** (out of scope). Packets sent while the receiver is down are lost. Recovery relies on the 2 Hz pose and the 10 s full obstacle refresh.
- **Gap accounting is per receiver instance.** A restarted receiver starts a new sequence history, so the loss during the outage appears as sent minus received, not as a gap.
- **Localhost only.** No packet loss, delay or reordering was exercised over a real or emulated link (R5.3).
- **Latency precision is bounded by the ~1 ms Windows wall-clock resolution.** Observation-to-reception latency in simulation time is not meaningful in fast mode (the simulation runs faster than wall time).
- **No deletion message in v1.** An obstacle that disappears (a track merged by refinement, a cluster after a reset) is only dropped from the ground picture by the next full snapshot or a map-reset event.
- **Fast mode only in the automated runs.** In real time the wall-clock rates scale down, while the bytes per simulated second stay the same.
- **Controller memory** grows by the bounded send log (at most 20,000 rows).

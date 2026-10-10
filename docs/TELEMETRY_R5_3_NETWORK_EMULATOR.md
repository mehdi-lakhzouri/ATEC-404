# Artemis-X Lunar Network Emulator (R5.3)

| | |
|---|---|
| Status | Implemented; validated live with real Webots telemetry. No retransmission, acknowledgement or store-and-forward (R5.4) |
| Code | `src/artemis_x/telemetry/emulator.py` (`LunarLinkEmulator`, `LinkConfig`, `PRESETS`, `send_control`) |
| CLI | `scripts/lunar_link.py serve` (emulator with live console) and `scripts/lunar_link.py ctl <command>` |
| Tests | `tests/test_telemetry_emulator.py` (9 tests: statistical loss, real-UDP timing, bandwidth, queue, live control, blackout, shutdown) |
| Live validation | `scripts/run_webots_r4.py link` → run `artemis-webots/output/r5-link-20261010-215512` (4 full Webots tours) |
| Unchanged | R4 mapping, YOLO, odometry and motor control; the R5.1 protocol and the R5.2 sender and receiver. The rover is only pointed at another port |

## 1. Architecture

```
Webots controller --R5.1 encode--> R5.2 UdpSender --udp 127.0.0.1:5791--> LunarLinkEmulator --udp 127.0.0.1:5790--> GroundReceiver
                                                                               ^ control udp 127.0.0.1:5792 (live presets)
```

Pipeline per datagram, in arrival order:
1. **Blackout:** dropped. Nothing is buffered; sender-side store-and-forward is R5.4.
2. **Random loss:** dropped with probability `loss`, using a seeded RNG (reproducible; seed 2026 in the live runs).
3. **Bandwidth:** token bucket in bytes/s, with a burst of one maximum packet (1,066 B). Packets wait in a bounded FIFO (`queue_max`, default 64); a full FIFO drops the new packet (`dropped_queue`).
4. **Propagation:** delivered after `latency_ms` ± uniform `jitter_ms` (never negative; jitter can reorder packets).

**Properties:**
- Bytes are forwarded **unchanged**: payload and CRC16 are untouched. The emulator only peeks at the header and CRC to log them.
- Scheduling uses `time.perf_counter` (monotonic, high resolution; Windows `time.monotonic` only resolves 15.6 ms). The rover's simulated time (`t_obs` in the header) is logged but never used for scheduling.
- **Entering a blackout discards what is queued or in flight** (counted as `dropped_blackout`). Packets don't secretly arrive after the link went down.
- Shutdown counts the remaining packets as `dropped_shutdown` and closes the sockets.
- The emulator is a separate process (or thread). The Webots controller only does a non-blocking `sendto`, so emulator delays never block the rover.
- **Direct R5.2 mode is unchanged:** without the emulator, the rover sends straight to the receiver (`ARTEMIS_TELEMETRY_PORT=5790`, the default).

## 2. Presets (switchable live)

| Preset | Loss | Latency ± jitter | Bandwidth | Notes |
|---|---|---|---|---|
| NORMAL | 0 % | 0 ms | unlimited | Minimal latency |
| DEGRADED | 30 % | 200 ± 50 ms | unlimited | |
| SEVERE | 50 % | 1,300 ± 300 ms (about the Earth–Moon light time) | 1,000 B/s | |
| BLACKOUT | 100 % | — | — | Drops everything, empties the queue and in-flight packets |
| RESTORE | 0 % | 0 ms | unlimited | Same as NORMAL, logged as a restoration |

Custom settings: `SET loss=0.1 latency_ms=50 jitter_ms=10 bandwidth_Bps=2000 queue_max=32 blackout=false`. Invalid values (loss outside 0–1, negative latency, …) are rejected and the link stays unchanged.

## 3. Three-terminal demonstration

```powershell
# Terminal 1 - ground receiver (udp 5790)
python scripts/ground_receiver.py --out artemis-webots/output/ground/telemetry.jsonl --stats artemis-webots/output/ground/stats.json

# Terminal 2 - lunar link emulator (udp 5791 -> 5790); type presets at its prompt while Webots runs
python scripts/lunar_link.py serve --log artemis-webots/output/link/link.jsonl --stats artemis-webots/output/link/stats.json
#   NORMAL | DEGRADED | SEVERE | BLACKOUT | RESTORE | SET loss=0.1 ... | STATUS | QUIT
#   (or from any other terminal:  python scripts/lunar_link.py ctl SEVERE)

# Terminal 3 - YOLO server in the background, then Webots with telemetry routed through the emulator
Start-Process -NoNewWindow uv -ArgumentList "run","--project","backend","--no-sync","python","scripts/inference_server.py","--port","5761"
$env:ARTEMIS_TELEMETRY = "1"; $env:ARTEMIS_TELEMETRY_PORT = "5791"
& "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" --mode=realtime artemis-webots\worlds\artemis_test_course.wbt
```

**Automated live validation** (4 tours, about 5 min): `uv run --project backend --no-sync python scripts/run_webots_r4.py link`.

**Metrics** (`STATUS`, the `--stats` JSON, and one JSONL line per datagram and per configuration change):
- received; forwarded; dropped by loss, blackout, queue or shutdown;
- bytes forwarded; configured loss; delivery ratio;
- delay p50 / p95 / max (emulator arrival → forward, perf_counter);
- throughput (B/s);
- blackout total duration and current link state;
- per-preset counters.

## 4. Live Webots results (real telemetry, Webots fast mode, YOLO on)

Every tour completed **55/55**. Every rover sent the same 1,307 packets (1,001 POSE, 261 OBSTACLES, 45 EVENT; 112.3 kB), with 0 send errors.

| | NORMAL | DEGRADED | SEVERE | Live switching |
|---|---|---|---|---|
| Received by the emulator | 1,307 | 1,307 | 1,307 | 1,307 |
| Forwarded / received by ground (valid) | 1,307 / 1,307 | 934 / 934 | 645 / 645 | 826 / 826 |
| Dropped: loss · blackout · queue · shutdown | 0 · 0 · 0 · 0 | 373 · 0 · 0 · 0 | 661 · 0 · 0 · 1 | 206 · 275 · 0 · 0 |
| Configured → observed random loss | 0 → 0 % | 30 → **28.5 %** | 50 → **50.6 %** | per phase, below |
| Delivery ratio | 1.000 | 0.715 | 0.494 | 0.632 |
| Ground: gaps / missing / late / invalid | 0 / 0 / 0 / 0 | 347 / 372 / 127 / 0 | 258 / 660 / 325 / 0 | 118 / 481 / 71 / 0 |
| Rover send → ground receive latency p50 / p95 / max | ≤ 1 ms¹ / 1.5 / 6.7 ms | 206 / 253 / 262 ms | 1,363 / 2,227 / 4,389 ms² | — (mixed phases) |
| Emulator throughput (wall clock) | 1,712 B/s | 1,225 B/s | 854 B/s | 1,145 B/s |
| Decoded POSE vs controller odometry, max | 0.5 mm (1,001 poses) | 0.5 mm (712) | 0.5 mm (495) | 0.5 mm (623) |
| Ground obstacle map vs final export | 41/41, 0.0 mm | 41/41, 0.0 mm | 41/41, 0.0 mm | 41/41, 0.0 mm |
| Mapping cycle p95 · busiest step max | 12.07 · 16.9 ms | 11.78 · 16.9 ms | 11.48 · 14.8 ms | 11.29 · 14.5 ms |
| YOLO answers / 2,001 | 1,997 | 1,985 | 1,996 | 1,995 |

¹ Below the ~1 ms Windows wall-clock resolution.
² Includes 1,000 B/s bandwidth queueing on top of 1,300 ± 300 ms propagation.

The ground's `missing` count matches the emulator's drops: 372 against 373, 660 against 662, and 481 against 481. **The receiver detects the lost sequences.**

**Live switching run** (presets changed over the control port during the mission, wall-clock seconds after start: DEGRADED at 12 s, SEVERE at 24, BLACKOUT at 36, RESTORE at 48; all 4 switches acknowledged):

| Phase | Received | Forwarded | Dropped | Configured → observed loss | Delay p50 / max |
|---|---|---|---|---|---|
| NORMAL | 209 | 209 | 0 | 0 → 0 % | 0.06 / 0.5 ms |
| DEGRADED | 256 | 190 | 66 (loss) | 30 → 25.8 % | 208 / 263 ms |
| SEVERE | 270 | 119 | 140 loss + 11 blackout³ | 50 → 54.1 % | 1,296 / 1,601 ms |
| BLACKOUT (12.0 s) | 264 | **0** | 264 (blackout) | 100 % | — |
| RESTORE | 308 | 308 | 0 | 0 → 0 % | 0.06 / 0.4 ms |

³ Packets that were in flight when the blackout started; they are discarded, not delivered late.

- **The rover kept operating during the blackout:** 55/55 moves, mapping cycle p95 11.29 ms.
- **Communication resumed 23 ms after RESTORE** (first forwarded datagram).
- The ground received 826 valid packets, 822 of them before mission end. The first arrived at simulation time 0.032 s.

**Interpretation.**
- Observed loss rates fall within the binomial spread for these phase sizes.
- The steady 256-packet DEGRADED phase measured 25.8 % against 30 %, which is 1.6 σ.
- **The ground obstacle map was complete (41/41) in every run.** That comes from repetition: changes are sent as they happen, and a full snapshot goes out every 10 s, so each obstacle had many chances to get through. It is **not** a delivery guarantee. With longer outages or higher loss, obstacles can be missing until the next successful full snapshot.

## 5. Automated tests (`tests/test_telemetry_emulator.py`)

| Test | Checks |
|---|---|
| 0 % loss | 151 real packets forwarded, byte-identical (payload and CRC), all decode |
| 10 / 30 / 50 % loss | 4,000 packets each; observed rate within 4 σ of the binomial; accounting closes |
| 100 % loss, determinism | 0 forwarded; same seed gives the same drop pattern |
| Latency and jitter | 120 ± 30 ms configured: delays in [89, 175] ms (scheduling slack), real spread > 20 ms |
| Bandwidth | 4,000 B/s: measured rate within 15 %, no queue drops |
| Queue limit | 500 B/s with a 5-packet queue: ≥ 20 of 30 new packets dropped; accounting closes |
| Live control | NORMAL → SEVERE → BLACKOUT → RESTORE over the UDP control port; nothing delivered during the blackout; resumption; invalid `SET` rejected |
| Blackout and shutdown | In-flight packets discarded at blackout; packets in flight at shutdown counted `dropped_shutdown`; sockets closed |

## 6. Known limitations

- **No guaranteed delivery.** No retransmission, acknowledgement or store-and-forward (R5.4). Every packet lost to loss or blackout is gone. The ground recovers state only through later pose packets and the 10 s full snapshots.
- **Fast-mode bandwidth.** In fast mode the rover produces about 1.9 kB/s of wall-clock traffic. In real time it is about 219 B/s, so the 1,000 B/s SEVERE bandwidth only causes queueing in fast mode.
- **Loss model.** Losses are independent (Bernoulli); there are no burst or Gilbert-Elliott losses. Latency and jitter are uniform, with no long-tail distribution.
- **Single direction:** rover to ground only; there is no uplink.
- **Clock limits.** Emulator delays are measured with `perf_counter`. End-to-end latency uses the ~1 ms Windows wall clock.
- **Ports.** The emulator and the R5.2 test scenarios share the default ports. Don't run two demonstrations at the same time.

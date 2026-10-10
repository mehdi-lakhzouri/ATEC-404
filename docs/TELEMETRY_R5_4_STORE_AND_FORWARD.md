# Artemis-X DTN-Inspired Store-and-Forward (R5.4)

| | |
|---|---|
| Status | Implemented; validated live with real Webots telemetry through the emulated lunar link |
| Rover side | `src/artemis_x/telemetry/dtn.py` (`StoreForwardQueue`: pure logic, explicit clock; `DtnLink`: UDP glue), priorities in `adapter.py` |
| Ground side | `transport.py` `GroundReceiver(ack_to=..., reconcile=True)` (selective ACKs) and `ground_map.py` (`GroundMap` reconciliation) |
| Link | `emulator.py` reverse path: ground ACKs → udp 5794 → emulator (same blackout, loss, latency) → rover udp 5793 |
| Protocol | v1, additive: packet type 4 = ACK (selective, ≤ 64 seqs); bit 7 of the OBSTACLES frame byte = FULL snapshot. Older packets decode unchanged |
| Tests | `tests/test_telemetry_dtn.py` (17 tests: queue on a fake clock, reconciliation, ACK packet, real-UDP end to end) |
| Live validation | `scripts/run_webots_r4.py dtn` → `scripts/r5_dtn_analysis.py`; run `artemis-webots/output/r5-dtn-20261010-223409` (9 Webots tours) |
| Opt-in | `ARTEMIS_TELEMETRY_DTN=1`. Without it, R5.2 / R5.3 behave exactly as before |

## 1. Delivery claims

**A successful `sendto` proves nothing.** Delivery counts as confirmed only when the ground's ACK comes back over the simulated reverse link. That link has the same blackout and the same loss and latency as the downlink, so ACKs are lost too.

All results below are measured **at the ground** (what the receiver decoded), not inferred from rover sends.

**Nothing here guarantees delivery.** Messages that are still unconfirmed when their time-to-live or attempt cap runs out, or that are evicted by the memory limits, or that are pending when the mission ends, are reported as such.

## 2. Design

```
TelemetryAdapter --(packet + meta: kind, prio, seq, t_obs, hazards, ids, full)--> StoreForwardQueue --> UdpSender -> emulator -> ground
                                                                                       ^                                            |
                                                         ACK (selective, batched 100 ms) +--- rover udp 5793 <- emulator <- udp 5794 +
```

**Priorities** (decided by the adapter):

| Priority | What | Retention |
|---|---|---|
| CRITICAL | `map_reset` events; health **state** changes (OK / DEGRADED / STOP); obstacle packets carrying a hazard **never sent before** in this map epoch (a newly confirmed track, a new depth obstacle or negative hazard) | Until ACKed. TTL 600 s, cap of 60 attempts |
| NORMAL | Obstacle refreshes (full and changes-only); health **reason** changes | Until ACKed. TTL 60 s, cap of 8 attempts |
| LOW | Poses | Fire-and-forget, **coalesced**: only the newest pose waits; older ones are discarded as stale |

**Mechanisms** (`StoreForwardQueue`; every rule is unit-tested on a fake clock):
- **Superseding.** A FULL obstacle snapshot (any of its parts) makes older pending NORMAL obstacle packets obsolete, because it carries the current state. CRITICAL hazard packets are kept.
- **Coverage.** When a FULL snapshot is ACKed, pending CRITICAL hazard packets whose hazards it contains, and that are not newer than it, are purged (`covered_by_full`).
- **Retries.**
  - The RTO adapts (RFC 6298 smoothing) using **Karn's algorithm**: RTT samples are taken only from first transmissions, and the RTO itself backs off on a timeout retransmission, at most once per RTO.
  - Without that back-off, an RTO below the real RTT never grows: 1 s against a 2.6 s lunar round trip in the first live run.
  - Each entry backs off exponentially from the RTO it had at its first transmission, capped at 16 s.
- **Memory.** Limits of 400 entries and 192 KiB (configurable: `ARTEMIS_DTN_MAX_ENTRIES` / `ARTEMIS_DTN_MAX_BYTES`). Eviction order:
  1. LOW;
  2. NORMAL;
  3. CRITICAL, starting with *redundant* hazard packets (all their hazards are in a newer pending obstacle packet), then the oldest, with map-reset events evicted last.

  Every eviction is counted.
- **Link state.** The link is believed DOWN when no ACK has arrived for 2 s after a send. While DOWN, only **one probe per second** goes out (the most important pending packet, regardless of its own backoff). The first ACK marks the link UP again, and the backlog drains in priority order, at most 16 packets per tick.
- **Never blocking.** All rover work runs on idle simulation steps with bounded loops (≤ 64 ACK datagrams and ≤ 16 sends per tick), on a non-blocking socket. Mapping and motor control are never blocked.
- **End of mission.** The rover keeps draining and reading ACKs for up to 3 s (wall clock). Whatever is still pending is reported as undelivered.
- **Identity preserved.** Retransmissions are **byte-identical**: same session, sequence number, observation time and obstacle IDs.

**Ground** (`GroundReceiver` + `GroundMap`):
- **ACKs.**
  - Every valid packet is ACKed, duplicates included, since their first ACK may have been lost.
  - ACKs are batched every 100 ms, with at most 64 sequence numbers per packet.
- **Reconciliation:** the picture is ordered by **observation time**, never by arrival.
  - Duplicates (same session and seq) are ignored.
  - A record replaces the stored one only if it is not older.
  - A FULL snapshot removes vanished obstacles only once **all** its parts have arrived, and only if no newer full snapshot was already applied.
  - A `map_reset` removes everything observed before it and ignores late packets from the old map.
  - Poses: newest wins.
- **Sequence accounting:** gaps, duplicates and late packets are still counted, as in R5.2.

## 3. Commands (three terminals)

```powershell
# Terminal 1 - ground receiver with ACKs (to the emulator's reverse port) and the reconciled map
python scripts/ground_receiver.py --out artemis-webots/output/ground/telemetry.jsonl --ack-to 127.0.0.1:5794 --map-out artemis-webots/output/ground/map.json

# Terminal 2 - lunar link emulator carrying both directions; type NORMAL / DEGRADED / SEVERE / BLACKOUT / RESTORE live
python scripts/lunar_link.py serve --reverse --log artemis-webots/output/link/link.jsonl

# Terminal 3 - YOLO server, then Webots with store-and-forward telemetry through the emulator
$env:ARTEMIS_TELEMETRY = "1"; $env:ARTEMIS_TELEMETRY_PORT = "5791"; $env:ARTEMIS_TELEMETRY_DTN = "1"
& "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" --mode=realtime artemis-webots\worlds\artemis_test_course.wbt
```

**Automated live validation** (9 tours, about 10 min): `uv run --project backend --no-sync python scripts/run_webots_r4.py dtn`.

**Metrics exported for Mission Control** (`summary.json` → `telemetry.dtn`):
- queue size per priority, `pending_critical`, `queue_bytes`, peak entries and bytes;
- sent, `retries`, `probes`, RTO, SRTT and RTO back-offs;
- `acked`, `expired` and `evicted` per priority; `superseded`, `coalesced`, `covered_by_full`, `redundant_evicted`;
- ACK latency per priority (storage → confirmed delivery, including outages);
- detected outages `[down, up]` and the current `link_state`.

## 4. Live Webots results (real telemetry, fast mode, YOLO on; all 9 tours 55/55)

**Blackout scenarios.** "Generated in blackout" means the packet's observation time falls inside a blackout window; recovery is measured from the RESTORE command to the first ground reception.

| | No DTN (R5.3 reference) | DTN, one 12 s blackout | DTN, 3 × 6 s blackouts | DTN, 28 s blackout, 40-entry cap |
|---|---|---|---|---|
| EVENT packets generated in blackout → delivered | **0 / 7** | 5 / 5 | 12 / 12 | 15 / 15 |
| CRITICAL packets generated in blackout → delivered | — | **19 / 19** | **36 / 36** | **39 / 40** |
| Hazards announced in blackout → received by ground | — | 18 / 18 | 37 / 37 | 39 / 39 |
| Recovery after RESTORE, p50 / max | — | 0.12 / 0.13 s | 0.07 / 1.00 s | 0.33 / 0.33 s |
| All CRITICAL packets → delivered | — | 111 / 111 | 112 / 112 | 99 / 100 |
| Outage detected (DOWN → UP, s after start) | — | 14.1 → 24.2 | 9.2 → 13.3, 23.2 → 27.2, 37.2 → 42.2 | 11.3 → 37.6 |
| Queue peak: entries / bytes | — | 24 / 4.8 kB | 21 / 5.5 kB | 40 / 9.8 kB (at the cap) |
| Evicted LOW / NORMAL / CRITICAL | — | 0 / 0 / 0 | 0 / 0 / 0 | 130 / 20 / 13 (all 13 critical evictions were redundant) |
| Ground map vs final export | 41/41 (by repetition) | 41/41, 0.0 mm | 41/41, 0.0 mm | 41/41, 0.0 mm |

In the 40-entry overflow run, the one undelivered CRITICAL packet was evicted as redundant. Every hazard it announced was still received through a newer packet (39/39).

**Loss and restart scenarios:**

| | 30 % loss, 200 ± 50 ms | 50 % loss, 200 ± 50 ms | SEVERE stress: 50 %, 1.3 ± 0.3 s, 1 kB/s | Receiver restarted (down ~15–25 s wall) |
|---|---|---|---|---|
| CRITICAL packets → delivered | 107 / 111 | 97 / 109 | **82 / 110** | 111 / 111 |
| Hazards announced → received by ground | 118 / 119 | **118 / 118** | **110 / 120** | 120 / 120 |
| NORMAL / LOW delivered | 155 / 195 · 723 / 1,001 | 100 / 190 · 485 / 1,001 | 87 / 196 · 433 / 986 | 167 / 173 · 836 / 869 |
| Rover-confirmed (ACKed) CRITICAL | 100 | 62 | 51 | 111 |
| Still pending at mission end (critical / normal) | 3 / 4 | 17 / 7 | 49 / 15 | 0 / 0 |
| Retries · RTO back-offs · final RTO / SRTT | 141 · 45 · 2.73 / 0.49 s | 218 · 45 · 2.74 / 0.48 s | 156 · 10 · 10.0 / 3.81 s | 16 · 4 · 0.3 / 0.08 s |
| Ground duplicates (lost ACKs) / late | 51 / 165 | 48 / 124 | 31 / 317 | 0 / 1 |
| Ground map vs final export | 41/41 | 40/40, 17 stale extras | 40/41, 20 stale extras | 41/41 |
| Queue peak entries / bytes | 16 / 3.7 kB | 36 / 10.8 kB | 70 / 17.3 kB | 20 / 4.4 kB |

**Interpretation:**
- **Blackouts and restart.** Store-and-forward does what it is for: every critical observation generated during the tested blackouts reached the ground after restoration, mostly within 0.1–1 s, while the R5.3 reference lost all of them. With the receiver down for about 10 s, nothing critical was lost.
- **Random loss.** With 30–50 % loss in **both** directions (a round trip succeeds only 25–49 % of the time), most critical packets arrive and nearly all hazards reach the ground, but **not all**.
  - Packets still pending at mission end were cut off by the 3 s end-of-mission grace.
  - Lost ACKs make the rover's confirmed count (62 at 50 % loss) much lower than what the ground actually received (97).
- **Stale extras in the ground map.** At 50 % loss, full snapshots often lose a part, so vanished obstacles stay on the ground map until a complete newer snapshot arrives.
- **SEVERE is a stress case beyond the link's capacity.** In fast mode the rover produces about 1.9 kB/s of wall-clock traffic against a 1 kB/s link, with a 1.3 s one-way delay. The SRTT reached 3.8 s, the RTO hit its 10 s cap, and 10/120 hazards and 28 critical packets were not delivered. In real time the rover produces about 0.22 kB/s, so this preset would not be bandwidth-bound.

**Rover impact** (same batch):

| Measure | Result |
|---|---|
| Tours | 55/55 in all 9 runs |
| Controller memory | 73.1–75.6 MB (73–75 MB in R5.2 / R5.3) |
| DTN queue peak | 70 entries / 17.3 kB, against 192 KiB allowed |
| Mapping cycle p95, DTN runs | 11.2–14.5 ms |
| Mapping cycle p95, reference runs (no DTN) | 15.3 ms (`fusion`) and 11.3 ms (`nodtn_blackout`) |

The machine was loaded during this batch: the no-telemetry reference itself exceeded 15 ms, consistent with the R4.6 finding (up to 16.0 ms under load). The DTN runs stay within the established envelope.

**YOLO** answered **1,973–1,998** of 2,001 frames, against 1,985–2,001 in R4–R5.3. The lowest values came from the heavy-loss runs, where the test harness process (emulator, receiver and their logging) does more work on the same machine. The YOLO model and its settings are unchanged; the inference link is a separate TCP connection that DTN does not touch. Treat this as a test-harness CPU effect, not a functional change.

## 5. Automated tests (`tests/test_telemetry_dtn.py`, 17)

| Group | Tests |
|---|---|
| Queue (fake clock) | Priority order and pose coalescing (LOW fire-and-forget); exponential backoff with Karn's rule (no sample from retransmissions) and an adaptive RTO for a 1.3 s RTT; **Karn RTO back-off converging from 1 s to a 2.6 s RTT**; expiry by TTL and attempt cap, critical surviving a 5-minute blackout; memory limits by entries and bytes, eviction LOW → NORMAL → CRITICAL; **redundant critical evicted first, map reset kept**; **coverage purge on a FULL-snapshot ACK**; superseding by full snapshots; one probe per interval while DOWN, bounded drain once UP; duplicate and unknown ACKs; exported metric keys |
| Ground map | Duplicates and out-of-order arrival (newest observation wins); full snapshot reconciles only when complete and newer (a late older one never removes newer knowledge); map reset with late pre-reset packets ignored; newest pose wins |
| ACK packet | Round trip, 1–64 seqs, CRC rejection, FULL flag round trip |
| End to end, real localhost UDP through the emulator | Blackout: critical kept while DOWN (no ACKs), probes, stale poses coalesced, **recovered after RESTORE in < 3 s**; 50 % loss in both directions: every entry ends ACKed or expired, the ground receives ≥ 38/40 (expected 39.8), duplicates from lost ACKs are ignored by the map; receiver restart: critical recovered; queue overflow during blackout: memory bounded, no critical evicted while NORMAL can be, critical recovered |

## 6. Known limitations

- **No guaranteed delivery.** Critical data can still be lost through the attempt cap or TTL, memory eviction, or the end of the mission (3 s grace). This happened under 30–50 % loss and in the SEVERE stress case.
- **No congestion control.** Retries and backlog drains are budgeted per tick but not rate-limited to the link's bandwidth, so a bandwidth-starved link (SEVERE in fast mode) suffers queue drops.
- **No persistence.** The queue lives in controller memory: a controller crash loses it.
- **ACK reach.** The ACK reverse path uses the same impairment model as the downlink, without its own bandwidth limit.
- **Fixed policies.** The probe, TTL and backoff settings are tuned for this course, not derived from a real link budget.
- **Ground-map lag under heavy loss.** Vanished obstacles stay until a complete newer full snapshot arrives (17–20 stale extras at 50 % loss).
- **Single machine.** Measurements are in fast mode on one machine; YOLO answer counts and mapping p95 vary with machine load.

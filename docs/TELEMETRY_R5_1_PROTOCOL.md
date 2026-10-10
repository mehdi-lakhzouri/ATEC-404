# Artemis-X Lunar Telemetry Protocol v1 (R5.1)

| | |
|---|---|
| Status | Implemented and tested. No transport yet (R5.2) |
| Code | `src/artemis_x/telemetry/protocol.py` (encode / decode), `src/artemis_x/telemetry/adapter.py` (R5.2 interface) |
| Input | The R4.6 obstacle contract `artemis.obstacles.v1` (`src/artemis_x/sim/mapping/contract.py`), the estimated pose and the health monitor |
| Tests | `tests/test_telemetry.py` (16 tests, on a real R4.6 export: `tests/fixtures/obstacles_v1_fusion.json`) |
| Benchmark | `scripts/r5_telemetry_benchmark.py <run dir>`, which measures binary against JSON on real exports |
| Dependencies | Python standard library only. No compression, no network |

## 1. Packet layout

All multi-byte integers are **big-endian** (network order). Every packet is self-contained:

| Offset | Field | Type | Meaning |
|---|---|---|---|
| 0 | `magic` | 2 bytes | `"AX"` (0x41 0x58) |
| 2 | `version` | u8 | `1`. A decoder rejects any other version |
| 3 | `type` | u8 | 1 = POSE, 2 = OBSTACLES, 3 = EVENT. Others are rejected |
| 4 | `session` | u32 | Mission or session id, chosen by the rover |
| 8 | `seq` | u32 | Packet sequence number, +1 per packet, wraps modulo 2³² |
| 12 | `t_obs_ms` | u32 | **Observation** time, simulation clock, ms (range 0–49.7 days) |
| 16 | `payload_len` | u16 | Payload size in bytes |
| 18 | payload | — | See below |
| 18 + len | `crc16` | u16 | CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, no xor-out) over header + payload; check value "123456789" → 0x29B1 |

**Overhead:** 18-byte header + 2-byte CRC = 20 bytes per packet. **Largest packet:** 1,066 bytes (an OBSTACLES packet with 40 records). Larger snapshots are split into numbered parts.

**Observation time vs reception time.** `t_obs_ms` is set by the rover when the data was observed. Reception time is never in the packet: the receiver records it (`decode(data, rx_time_s=...)` stores it in `Packet.rx_time_s`), so link latency is `rx_time_s - t_obs_s`. Each obstacle record also carries its own observation time (last detection or last cell update), which can be older than the packet's `t_obs`.

**Frame.** Every position is in the rover's **odom** frame (frame code 0), the estimated-pose frame of R4. Any other frame code is rejected, on both encode and decode.

## 2. Payloads

### POSE (type 1, 13 bytes; packet 33 bytes)

| Field | Type | Quantisation | Range |
|---|---|---|---|
| `frame` | u8 | 0 = odom | — |
| `x`, `y` | i32 | 1 mm | ±2,147 km |
| `yaw` | i16 | 1e-4 rad, wrapped to [−π, π] | ±3.2767 rad |
| `sigma_xy` | u16 | 1 cm | 0–655.34 m; **0xFFFF = saturated (≥ 655.35 m)**, flagged on decode |

### OBSTACLES (type 2, 6 + 26 × n bytes, n ≤ 40)

Snapshot header (6 bytes): `frame` u8 · `part` u8 · `parts` u8 · `count` u8 · `map_resets` u16. Then `count` records:

| Field | Type | Quantisation / encoding | Range |
|---|---|---|---|
| `flags` | u8 | bit 0 id namespace (0 = track, 1 = depth cluster) · bits 1–2 type (0 rock, 1 obstacle, 2 negative_hazard) · bit 3 source (0 yolo+depth, 1 depth) · bits 4–5 status (0 confirmed, 1 lost, 2 observed) · bit 6 protected · bit 7 reserved (must be 0) | — |
| `id` | u32 | track: `trk-<n>` → n. Cluster: `dep-<i>_<j>` → (i & 0xFFFF) << 16 \| (j & 0xFFFF), with i, j signed int16 cell indices | tracks 0–2³²−1; cells ±32,767 (±3.2 km) |
| `x`, `y` | i32 | 1 mm, odom | ±2,147 km |
| `radius` | u16 | 1 mm | 0–65.535 m (larger raises an error) |
| `confidence` | u8 | round(c × 255) | 0–1 (resolution 0.0039) |
| `uncertainty` | u16 | 1 mm, 1-sigma | 0–65.534 m; **0xFFFF = saturated (≥ 65.535 m)**, flagged on decode |
| `t_obs` | u32 | 1 ms, simulation clock | 0–49.7 days |
| `link` | u32 | First linked track id of a depth cluster (`track_ids[0]`); 0 = none | — |

### EVENT (type 3, 8 bytes; packet 28 bytes)

| Field | Type | Encoding |
|---|---|---|
| `code` | u8 | 1 = health, 2 = map_reset, 3 = inference |
| `state` | u8 | 0 OK, 1 DEGRADED, 2 STOP |
| `reasons` | u32 | Bit *i* = the R4.3 health reason *i*: depth_stale, depth_frozen, depth_invalid, depth_degraded, no_pose, obstacle_in_footprint, obstacle_in_corridor, unobserved_corridor, stale_corridor, inference_disabled, inference_unavailable, inference_stale, cycle_overrun. Append-only within v1 |
| `value` | u16 | Code-specific (map_reset: number of resets) |

## 3. Validation rules

**Encode never clips silently.**
- A value that is not finite, or outside its range, raises `ProtocolError`.
- This applies to unknown enum values, a frame other than `odom`, an id/source namespace mismatch, and session or sequence numbers outside u32.
- Only the two uncertainty fields saturate, explicitly, at their maximum code.

**Decode is strict.** It rejects:
- a wrong magic, an unknown version or type;
- a packet shorter than 20 bytes or longer than 1,066 bytes;
- `payload_len` not matching the packet size, or a CRC mismatch;
- a payload size that doesn't match its type;
- more than 40 records, or an inconsistent part/parts count;
- an unknown frame, a set reserved bit, an enum out of range, or unknown reason bits.

The CRC is checked **before** any field is interpreted.

## 4. Obstacle ids and duplicate observations

- **Namespaces cannot collide.** `trk-65537` and `dep-1_1` share the same 32-bit value (0x00010001) but differ in the namespace bit. Ids round-trip exactly.
- **Track ids are persistent** within a mission and never reused.
- **Depth-cluster ids are not persistent**, as in the R4.6 contract: a cluster that grows or splits gets a new id. Receivers must not track them over time.
- **Duplicates** (`adapter.dedupe`):
  - A depth cluster linked to tracks (`track_ids`) and no larger than the largest linked track (radius + 0.2 m) is the **same physical object**, so it is not sent.
  - A larger linked cluster (a blob around a small tracked rock) carries geometry the track lacks, so it is sent, with `link` so the receiver can associate it.
  - Tracks are always sent.

## 5. R5.2 interface: `TelemetryAdapter`

```python
adapter = TelemetryAdapter(mapper, session_id, pose_period_s=0.5, obstacle_period_s=2.0, max_queue=256, dedup=True)
adapter.tick(t)            # controller, IDLE simulation steps only (like the R4.5 monitor): at most one job per call
packets = adapter.drain()  # R5.2 transport: list of bytes, oldest first
```

- **Job priority per tick:**
  1. a `map_reset` event, and the next snapshot is brought forward;
  2. a `health` event when the state or reasons change;
  3. a pose every `pose_period_s`;
  4. an obstacle snapshot every `obstacle_period_s`.
- **Bounded queue:** when it is full, the **oldest** packet is dropped and counted (`stats["dropped_queue_full"]`); fresh state is worth more than stale state.
- **Stats:** packets, bytes, per-type counts, deduplicated clusters.
- **Scope:** read-only on the mapping state, no ground truth, no network. It is **not yet wired into the controller**; R5.2 will call it next to `monitor.tick` on idle steps. Building a snapshot (`export_obstacles`) costs about 3–15 ms, which is why it runs only every `obstacle_period_s`, on an idle step.

## 6. Measured sizes on real R4.6 exports

Source: the 16 `obstacles_final.json` files of `artemis-webots/output/r4-validate-20261010-204718` (8 safety, 4 fusion and 4 monitor tours; 426 obstacles in total). All values are measured byte counts. "Equivalent JSON" is compact JSON (`separators=(",", ":")`) carrying exactly the transmitted fields at the contract's precision, plus the same header information.

| Snapshot | Obstacles | Equivalent JSON | Binary (all obstacles) | Binary, deduplicated | zlib-9 of the obstacle JSON (reference) |
|---|---|---|---|---|---|
| Fusion, nominal | 48 | 9,608 B | **1,300 B** (2 packets) | 1,118 B (7 clusters deduplicated) | 1,501 B |
| Depth only | 12 | 2,464 B | **338 B** | 338 B | 511 B |
| YOLO disconnected | 30 | 5,931 B | **806 B** | 806 B | 1,020 B |
| After a map reset | 35 | 7,104 B | **936 B** | 754 B | 1,183 B |
| **All 16 runs** | **426** | **85,721 B** | **11,622 B** | **10,504 B** (42 deduplicated) | 14,699 B |

| Packet | Binary | Equivalent compact JSON |
|---|---|---|
| POSE | 33 B | 115 B |
| EVENT (health, 2 reasons) | 28 B | 144 B |
| One obstacle record | 26 B (plus 26 B per packet) | about 200 B |

- **Saving vs equivalent JSON:** 86.4 % (7.4 × smaller).
- **Against the full contract document:** 87.6 %.
- **Compression:** the binary is not compressed, yet it is 21 % smaller than zlib-9-compressed JSON (11,622 against 14,699 B). Compressing the binary was not measured and is not claimed.

Several runs export the same final map (identical scenes), so the 16 snapshots are not independent samples. The per-snapshot rows are the meaningful ones.

## 7. Known limits

- **Precision:**
  - positions are quantised to 1 mm, angles to 1e-4 rad, confidence to 1/255;
  - the uncertainties saturate at 65.5 m (obstacles) and 655 m (pose);
  - the odometry absolute sigma reached 32 m after one course, so the pose limit is fine but not unbounded.
- **Wrap-arounds:** `t_obs` wraps after 49.7 days of simulation time and `seq` after 2³² packets. Neither is handled beyond the documented modulo; real missions would need an epoch field (v2).
- **No reliability layer:** no retransmission, acknowledgement or encryption. Integrity is protected by the CRC only, not authenticated. These are R5.2 transport decisions.
- **Snapshot assembly:** a snapshot split into several parts is only consistent if all parts arrive; receivers should group by (`session`, `t_obs`, `parts`).

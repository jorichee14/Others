# ROS 2 Wi-Fi Monitor — System Documentation

## 1. Overview

`ros2_wifi_monitor` is a ROS 2 package suite that continuously assesses the quality and capacity of a robot's wireless network link. It is built around three complementary monitoring nodes — informally, the three "radars" that each sweep a different dimension of the link:

- **Radar 1 — `wifi_monitor_node`**: passive radio-frequency and link-state monitor (RSSI, SNR, rate, retries, error counters).
- **Radar 2 — `iperf_runner_node`**: active throughput monitor that measures real achievable capacity by driving traffic across the link.
- **Radar 3 — `ping_monitor_node`**: continuous latency and packet-loss monitor.

Together they give a full picture of link health — signal quality, achievable bandwidth, and responsiveness — that can be time-joined to the robot's pose for coverage/connectivity mapping.

The package targets ROS 2 Humble, Iron, and Jazzy, is pure Python (no compiled dependencies beyond its own message package), and requires no root privileges — all three nodes read information that Linux already exposes to unprivileged users.

**Package layout**

| Package | Build type | Contents |
| --- | --- | --- |
| `wifi_monitor_msgs` | ament_cmake | `WifiLinkStatus.msg`, `IperfResult.msg`, `PingStat.msg` |
| `wifi_monitor` | ament_python | `wifi_monitor_node`, `iperf_runner_node`, `ping_monitor_node`, launch files, unit tests |

---

## 2. System Architecture

### 2.1 Design philosophy

The three nodes are intentionally independent processes rather than one monolithic monitor, because they have fundamentally different operating characteristics:

- **Passive vs. active** — `wifi_monitor_node` only *reads* existing driver/kernel state; it never injects traffic. `iperf_runner_node` *generates* traffic to measure capacity. `ping_monitor_node` generates a small amount of traffic (ICMP) but far less than iperf.
- **Saturating vs. non-saturating** — iperf tests can consume the entire link's bandwidth for their duration, which is unsafe to do continuously while the robot depends on the link. Ping and the passive monitor are cheap enough to run continuously during real operation.
- **Sampling cadence** — the passive monitor runs fastest (up to ~5 Hz, the practical ceiling for polling `iw`/`iwconfig`/`/proc`/`/sys`), ping runs at 1 Hz, and iperf runs either as periodic short bursts or as a continuous 1 Hz stream during a dedicated survey pass.

Running them as separate nodes means each can be started, stopped, or rate-limited independently — e.g., disabling iperf during normal operation while leaving the passive monitor and ping running continuously.

### 2.2 Data flow

Each node reads from OS-level sources (network interface, `/proc`, `/sys`, `iw`/`iwconfig` command output, `iperf3`/`ping` subprocesses), parses that output into a strongly-typed ROS 2 message, stamps it with a header (timestamp + `frame_id`), and publishes it on its own topic. `wifi_monitor_node` additionally publishes a standard ROS `diagnostics` array so link-quality problems surface through the normal ROS diagnostics/aggregator tooling, not just as raw message fields.

Downstream, the recommended pattern is to record all three topics (plus the robot's transform/odometry topics) to a bag on the robot's own disk, then time-join the wireless metrics to position offline to build coverage or dead-zone maps.

### 2.3 Shared conventions

All three nodes follow the same conventions so their outputs can be merged and compared consistently:

- **Explicit "unknown" rather than misleading defaults.** When a value cannot be determined (a field the driver doesn't report, a failed test, a lost ping), the corresponding float field is published as `NaN` rather than `0.0`, and integer fields that have no natural zero use `-1`. This keeps "unmeasured" visibly distinct from "measured as zero."
- **Every message carries a standard header** with a timestamp and a configurable `frame_id`, enabling straightforward time-alignment with other ROS data (TF, odometry).
- **Graceful degradation.** None of the nodes crash or block when a data source is unavailable (a driver that doesn't expose noise floor, a `station dump` that is denied, a server that is unreachable); they publish what they can and mark the rest as invalid/unknown.

---

## 3. Radar 1 — `wifi_monitor_node` (Passive RF / Link Monitor)

### 3.1 Purpose

Continuously samples the state of a wireless network interface without generating any traffic, publishing RSSI, SNR, negotiated rate, PHY details, retry/error counters, and interface traffic counters.

### 3.2 Data sources

The node merges several independent sources so that it degrades gracefully when a given driver is stingy with data:

| Source | Fields contributed |
| --- | --- |
| `iw dev <iface> link` (preferred) | ESSID, BSSID, frequency, bitrate, signal |
| `iwconfig <iface>` (legacy fallback) | The above, plus wireless error counters (invalid nwid/crypt/frag, excessive retries, missed beacon) |
| `/proc/net/wireless` | Link quality, signal, and noise floor |
| `/sys/class/net/<iface>/statistics/*` | RX/TX packets, bytes, errors, dropped, overruns, carrier errors, collisions |
| `iw dev <iface> station dump` | Per-association PHY/MCS/NSS/width, retries, failed frames, expected throughput, connected time, per-association byte/packet counts |
| `iw dev <iface> survey dump` | Noise floor and channel active/busy time (channel utilization) |

`station dump` and `survey dump` are read-only and typically require no elevated privileges, but if a driver denies them the node falls back to `NaN`/`-1` for those specific fields while continuing to publish everything else. Triggering a scan (`iw ... scan`) requires `CAP_NET_ADMIN` and is deliberately never used, since it would interfere with the existing association.

### 3.3 Signal quality and SNR handling

RSSI (`signal_dbm`) is always published when the driver reports it. SNR is computed as signal minus noise floor and is only populated when the driver genuinely reports a noise floor (`noise_valid = true`); many consumer/USB Wi-Fi drivers never report noise, in which case `noise_dbm` and `snr_db` are `NaN` and `snr_valid = false` — the node does not fabricate an SNR from an assumed noise floor. The noise value itself is taken from `iw survey dump` first, then `/proc/net/wireless`, discarding sentinel "not measured" values some USB adapters report. When no real noise floor is available, RSSI and link quality should be used as the signal indicators instead of SNR. Chipsets such as `ath9k_htc`, `mt76`, and Intel `iwlwifi` do report a measured noise floor and enable true SNR.

### 3.4 Disconnect / reconnect behavior

The node keeps publishing through an outage rather than stalling: while disassociated it emits `associated = false` with RF fields set to `NaN` so gaps are explicit in the recorded data rather than looking like missing samples, and it logs the disconnect and reconnect transitions. It never blocks or crashes if the interface disappears.

### 3.5 Message: `WifiLinkStatus`

The message is organized into logical groups:

- **Identity** — interface name, MAC address, administrative up/running state, association state.
- **Association** — ESSID, BSSID, mode, frequency/channel, negotiated bit rate, TX power.
- **Signal quality** — link quality (raw and ratio), RSSI, averaged RSSI, noise floor, SNR, and validity flags for noise/SNR.
- **PHY rate detail** — separate RX/TX bit rates, MCS index, spatial stream count, channel width, and PHY mode (HT/VHT/HE) for RX and TX, plus TX short guard interval.
- **Link reliability** — TX retries, TX failures, driver-expected throughput, connected duration, and per-association RX/TX byte and packet counts.
- **Channel survey** — channel active time, busy time, and busy ratio (channel utilization).
- **Wireless error counters** — invalid-nwid/crypt/frag counts, excessive retries, missed beacons, and other misc errors (from `iwconfig`/`proc`).
- **Interface traffic statistics** — RX/TX packets, bytes, errors, dropped, overrun, frame/carrier errors, and collisions (from `/sys`).

### 3.6 Key parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `interface` | (empty) | Interface to monitor; empty means auto-detect the first wireless interface. |
| `publish_rate_hz` | 5.0 | Sampling/publishing rate; suits a robot moving up to roughly 1 m/s and is the practical ceiling for this polling approach. |
| `frame_id` | `wifi` | Frame ID stamped in each message header. |
| `warn_signal_dbm` | -70.0 | RSSI threshold at/below which diagnostics report WARN. |
| `error_signal_dbm` | -80.0 | RSSI threshold at/below which diagnostics report ERROR. |

---

## 4. Radar 2 — `iperf_runner_node` (Active Throughput Monitor)

### 4.1 Purpose

`wifi_monitor_node` is passive and reports negotiated rate, not achieved throughput. `iperf_runner_node` closes that gap by periodically driving an `iperf3` test against a fixed server and publishing the measured throughput, retransmits, RTT (TCP) or jitter/loss (UDP).

### 4.2 Deployment topology

To ensure the measurement reflects the robot's own wireless link rather than some other bottleneck, the iperf3 server must sit on a machine wired (Ethernet) to the access point/router — never itself on Wi-Fi. That way the only wireless hop in the path is the robot's own connection to the AP.

### 4.3 Operating modes

Two distinct survey styles are supported, selected by the `continuous` parameter:

- **Periodic bursts (default).** A short test is run every `interval_s`. Each test pays a TCP connect/teardown cost, limiting back-to-back tests to roughly 0.25 Hz. This mode suits spot-checks during otherwise-normal operation, using a long interval so the link isn't saturated often.
- **Continuous 1 Hz survey mode.** A single iperf3 session is held open for the whole survey, and a result is published every second from its interval reports, eliminating per-test connection overhead. This is the densest throughput sampling the system offers and is intended for a dedicated survey pass (not for use while the robot depends on the link, since it saturates it continuously). Multiple parallel TCP streams can be used to fully fill the link. In this mode jitter and loss fields are not applicable and are published as `NaN`.

In continuous mode, RTT is additionally sampled once per second from the live connection's kernel TCP RTT (via the same underlying statistic `iperf` itself reports), so a single consistent source yields both throughput and RTT together at 1 Hz, without needing a separate ping. The idle control connection is excluded from this sampling so the reported RTT reflects the actual loaded data streams. This RTT sampling depends on the `iproute2` toolset being installed; if unavailable, RTT falls back to `NaN` in that mode.

### 4.4 Recommended survey practice

A multi-second test taken while the robot is moving smears the measurement over several metres as the channel conditions change mid-test. For the cleanest, most comparable capacity numbers, the recommended practice is a stop-and-measure approach: pause the robot at waypoints and run a short, full-parallel-stream test at each stop (these stops are identifiable offline as zero-velocity segments in the odometry). Continuous mode is the better choice when a dense throughput trace along a slow, continuous survey drive is preferred over discrete waypoint measurements.

### 4.5 Failsafes for a moving robot

The node is built to tolerate the link dropping and returning as the robot moves:

- **Link-down gate.** Before starting a test, the node checks the interface's carrier state. If Wi-Fi is not associated, it does not attempt to start iperf (which would otherwise hang); instead it publishes a failed result tagged as a link-down condition and polls at a configurable interval until the link returns, then resumes testing automatically. These link-down samples are useful for marking dead zones.
- **Connect timeout.** A bounded wait for the server to respond ensures that a case where Wi-Fi is up but the server is unreachable fails fast rather than stalling.
- **Failure backoff.** A failed test backs off to the reconnect-poll interval even in continuous survey mode, so an unreachable server is never hammered with repeated attempts.
- **Robust error handling.** All subprocess failure modes (timeout, launch failure, malformed output) are caught and reported as a failed result with an error description; the node itself never terminates because of a failed test.

### 4.6 Message: `IperfResult`

Published once per completed test, so results can be time-joined to pose. It records the test configuration (server address/port, protocol, direction, requested duration), an overall success flag with an error string on failure, throughput (goodput bitrate, total bytes, TCP retransmit count), TCP round-trip time statistics (mean/min/max, from the sender), and UDP reliability statistics (jitter, lost/total packets, loss percentage) — the TCP and UDP-specific groups are populated as applicable and left as `NaN`/zero otherwise.

---

## 5. Radar 3 — `ping_monitor_node` (Latency / Loss Monitor)

### 5.1 Purpose

iperf's live throughput stream does not carry round-trip time, so continuous latency and loss tracking comes from a separate, inexpensive source: ICMP ping. Because ping does not saturate the link, unlike iperf it can run continuously during real robot operation, not just during dedicated survey passes.

### 5.2 Behavior

The node pings a fixed target host at a steady rate and publishes, for every ping, the sequence number, whether a reply was received, and that reply's RTT — plus rolling-window statistics computed over recent samples: loss percentage across the window, and average/min/max RTT over the replies received in that window. The target only needs to answer ICMP echo requests; no ROS software or agent is required on the far end (only a firewall rule permitting ICMP if the target's default firewall blocks it).

### 5.3 Message: `PingStat`

Published per ping outcome, combining the instantaneous sample (target host, sequence number, reply/no-reply flag, this reply's RTT) with rolling-window aggregates (window size, loss percentage, and mean/min/max RTT over received replies in the window).

### 5.4 Relationship to `iperf_runner_node`'s RTT

Because continuous-mode `iperf_runner_node` now derives its own dense RTT from the loaded TCP connection, `ping_monitor_node`'s RTT measurement would be redundant during a dedicated survey pass that already runs continuous iperf — the two are complementary rather than always run together. `ping_monitor_node` is the appropriate choice when iperf is turned off (e.g., during real operation) but continuous RTT and loss visibility are still wanted without saturating the link.

---

## 6. Combined Operation — Full Survey Stack

A single launch file starts all three nodes together, sharing one server/target address, so passive link data, active throughput, and latency/loss are all captured on a unified time base. It supports two operating profiles, selected via boolean flags for whether iperf and ping run:

- **Dedicated survey pass** — passive monitoring plus continuous iperf (throughput + RTT via the kernel-socket sampling described in Section 4.3). This profile saturates the link for its duration and is meant for a deliberate survey run, not for use while the robot depends on the link.
- **Non-saturating live monitoring** — passive monitoring plus ping only, with iperf disabled. This is safe to run continuously during real operation since neither component saturates the link.

Because continuous iperf already reports RTT, the ping node defaults to off when the combined launch is used with iperf enabled, to avoid duplicating the RTT measurement; it should be enabled instead when iperf is disabled and RTT/loss visibility is still desired.

**Resulting topics when all three radars are active:**

| Topic | Typical rate | Content |
| --- | --- | --- |
| Wi-Fi status (passive) | Up to ~5 Hz | RSSI, MCS/rate, retries — continuous |
| Iperf result (active throughput) | ~1 Hz in continuous mode | Achievable throughput — continuous only during a dedicated survey pass |
| Ping stat (latency/loss) | 1 Hz | RTT and rolling packet loss — continuous, cheap |

Every message across all three topics is timestamped with a header, so the recommended workflow is to record them (together with the robot's transform and odometry topics) to a bag stored on the robot's own local disk — recording locally, rather than streaming the bag over the very Wi-Fi link being measured, avoids losing data precisely when the link degrades. The three streams can then be time-joined to position offline to produce coverage, throughput, and connectivity-quality maps of the robot's operating area.

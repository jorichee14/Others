# Wi-Fi / CSI measurement setup — working notes

Two mobile robots and a Nexmon CSI receiver measuring an ASUS RT-BE82U AP,
with the goal of pairing channel state against achievable throughput for
resource-allocation work in collaborative perception.

---

## 1. Hardware

| Role | Device | Interface / MAC | Notes |
| --- | --- | --- | --- |
| mobile_1 | Jetson + D-Link RTL8852BU | `wlx8876b9eae0ff` | iperf3 client, port 5201 |
| mobile_2 | Jetson + D-Link RTL8852BU | `wlx8876b9eae101` | iperf3 client, port 5203 |
| CSI receiver | Raspberry Pi, BCM43455c0 | `wlan0` | nexmon_csi, monitor mode |
| Server | wired laptop | `192.168.51.11` | `iperf3 -s` per port |
| AP | ASUS RT-BE82U | BSSID `ba:82:e2:21:55:75` | `ASUS_70_5G`, ch 44, 5220 MHz |

Driver: out-of-tree Realtek `8852bu` v1.19.21 (DKMS), **not** in-tree `rtw89`.

---

## 2. Required configuration

### 2.1 Disable HE on both robots — mandatory for CSI

The BCM43455c0 is an 802.11ac part. It **cannot demodulate 802.11ax (HE)
preambles**. While the robots negotiated `HE-MCS 11 HE-NSS 2`, the extractor
captured only Block Acks and beacons — never data frames.

```bash
echo "options 8852bu rtw_he_enable=0" | sudo tee /etc/modprobe.d/8852bu.conf
sudo modprobe -r 8852bu && sudo modprobe 8852bu
iw dev <iface> link          # must show VHT-MCS, not HE-MCS
```

To revert: `sudo rm /etc/modprobe.d/8852bu.conf` and reload the module.

**Cost:** ~185 Mbit/s (HE, 20 MHz) → ~140 Mbit/s (VHT, 20 MHz), about 25%.
Setting the AP to 80 MHz recovers this and more, and yields 256-slot CSI
instead of 64 — but requires re-baselining every measurement.

### 2.2 Check for traffic shapers

mobile_2 was capped at 10 Mbit/s egress by a leftover token bucket, which
looked exactly like a broken transmit chain (healthy PHY, zero errors,
throughput pinned at 9 Mbit/s regardless of channel conditions).

```bash
tc qdisc show dev <iface>                    # look for 'tbf'
sudo tc qdisc del dev <iface> root           # remove
# restore:
sudo tc qdisc add dev <iface> root tbf rate 10Mbit burst 2Kb latency 400ms
```

### 2.3 Container packages

These vanish on container rebuild — put them in the Dockerfile:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        iperf3 iproute2 iw wireless-tools chrony \
    && rm -rf /var/lib/apt/lists/*
```

`iperf3` for throughput, `iproute2` for `ss` (uplink RTT) and `tc`,
`iw`/`wireless-tools` for RSSI/MCS/survey, `chrony` for the NTP client.

`chronyd` runs on the **host**, not in the container. The container needs only
`chronyc`, and host networking so it can reach the daemon on 127.0.0.1:323.

### 2.4 AP settings for survey runs

- **Roaming assistant → Disable.** Default kicks clients below −70 dBm, which
  truncates exactly the low-signal region you want to measure.
- **Smart Connect / WiFi Agile Multiband / MLO → off**, so clients can't be
  steered mid-capture.
- **Control channel fixed at 44**, not Auto.

---

## 3. The comms workspace

Six original packages merged into two, in `comms_ws`:

```
comms_msgs/     WifiLinkStatus, IperfResult, PingStat, NtpStatus,
                CsiFrame, CsiStatus
comms/
  comms/wifi/   wifi_monitor_node, iperf_runner_node, ping_monitor_node
  comms/ntp/    ntp_client_node, ntp_server_node
  comms/csi/    csi_publisher, csi_monitor
  launch/       one per subsystem, unchanged in spirit
  config/       csi.yaml, ntp_monitor.yaml, rosbag_qos_override.yaml
  scripts/      ntp_bag_postprocess.py
```

### Modifications made to the iperf runner

- **Slotting removed.** The r2r `start_delay_s` scheme and `agent.launch.py`
  were dropped along with robot-to-robot measurement.
- **Bidirectional mode** alternates uplink/downlink sequentially (not
  `iperf3 --bidir`, because Wi-Fi is half-duplex and simultaneous two-way
  traffic contends with itself).
- **Split topics.** With `bidirectional:=true` the node publishes to
  `wifi/iperf_up` and `wifi/iperf_down` instead of `wifi/iperf`, routed on the
  message's own `reverse` field.
- **Segment alignment.** `bidir_period_s` is rounded to a whole number of
  report intervals; reports covering less than 60% of an interval are dropped
  so every published sample has the same width.
- **Swap cost is logged.** Measured at ~2.1 s per direction change, which is
  ~20% of samples lost at `bidir_period_s:=10`.

---

## 4. Measurement modes

### Continuous (moving survey)

One long `iperf3 -t 0 -i 1`, a sample per report interval.

```bash
ros2 launch comms survey.launch.py \
    server_address:=192.168.51.11 server_port:=5201 \
    interface:=wlx8876b9eae0ff \
    parallel:=4 continuous_interval_s:=1.0 \
    namespace:=mobile_1 run_ntp:=true
```

- 1 Hz, unbroken, no reconnects
- No `-O`, so nothing removes TCP slow-start
- Jitter and packet loss are always NaN (they only exist in the end summary)

### Periodic (stationary spots)

One process per test, one aggregate message per test.

```bash
ros2 launch comms survey.launch.py \
    server_address:=192.168.51.11 server_port:=5201 \
    interface:=wlx8876b9eae0ff \
    continuous:=false duration_s:=10.0 omit_s:=1.0 interval_s:=0.0 \
    parallel:=4 bidirectional:=true \
    namespace:=mobile_1 run_ntp:=true
```

- One number per ~11.2 s test, slow-start discarded via `-O`
- Cleanest labels — this is the mode for the dataset
- UDP (`protocol:=udp` with a real `udp_bitrate_mbps`) is the only way to get
  jitter and loss

### Idle baseline

```bash
ros2 launch comms survey.launch.py \
    server_address:=192.168.51.11 run_iperf:=false run_ping:=true \
    namespace:=mobile_1
```

Loaded minus idle latency gives a bufferbloat figure per location.

---

## 5. Latency: which number to trust

| Source | Valid? | What it measures |
| --- | --- | --- |
| `iperf_up` `rtt_ms_*` | yes | The bulk flow's own `tcpi_rtt`, loaded send queue |
| `iperf_down` `rtt_ms_*` | **no** | Frozen — the robot only sends ACKs, so `tcpi_rtt` never updates. One stale value per segment, NaN at teardown |
| `/wifi/ping` | yes | ICMP, direction-agnostic, survives segment swaps |

Measured example: `ss` RTT ~4.3 ms while ping showed ~47 ms mean (2.5–113 ms
range) on the same link at the same time. Both are real — iperf's packets sit
at the front of their own flow, ICMP queues behind everything iperf buffered
at the AP. **Ping is the number to report**, since it describes the link
rather than one privileged connection.

Use ping as the single latency source. Drop `rtt_ms_*` from downlink data or
document it as invalid.

---

## 6. Stream rates

| Stream | Rate | At 1 m/s |
| --- | --- | --- |
| CSI — Block Acks | ~116 Hz | 0.9 cm |
| CSI — data frames | ~8.5 Hz | 12 cm |
| `/wifi/status` | 5 Hz | 20 cm |
| `/wifi/iperf` continuous | 1 Hz | 1 m |
| `/wifi/ping` | 1 Hz | 1 m |
| `/wifi/iperf` periodic | ~0.09 Hz | 11 m |
| `/ntp/client/status` | 10 Hz | — |

Throughput is the bottleneck by two orders of magnitude, not CSI. Periodic
mode is stationary-only for this reason.

---

## 7. CSI

### 7.1 Frame types — the key finding

`frame_control` reports the **first byte of the PPDU**, not a parsed
frame-control field. For an aggregated A-MPDU that byte is a delimiter, so
data frames never appear as `0x88` (136). Observed values:

| Value | Meaning |
| --- | --- |
| 148 (0x94) | Block Ack — the bulk of frames |
| 128 (0x80) | Beacon |
| 224 (0xE0) | **Data frames**, aggregated |

**Use `seq`, not `frame_control`, to separate them:**

```python
data = df[df.seq != 65535]     # real 802.11 sequence numbers
bas  = df[df.seq == 65535]     # 0xFFFF placeholder
```

Verified: data-frame sequence numbers increment in strict steps of **16**
(1456, 1472, 1488, …) — one A-MPDU of 16 MPDUs per CSI record. That step is
independent confirmation of the aggregation explanation.

Do **not** set `frame_control_filter` — it keys on the same unreliable byte,
and `0x88` would filter everything out.

### 7.2 Trim is unreliable — keep `trim: false`

Four runs of `usable_slots()` on the same node returned four different bands:
`1..63` (60 slots), `32..62` (30), `28..29` (2), and one that kept the
constant artefacts while dropping real subcarriers.

The firmware artefact slots (28–35) carry magnitudes two orders above real
subcarriers (`-32640`, `10277`), which breaks the Otsu threshold. Gap-closing
also re-admits artefacts the variance test already flagged. Slot 29 varies
(`-32640` / `-8960` / `-9728`) so a variance test can't catch it at all.

Publish raw with full `subcarrier_index` and reject by magnitude offline.

### 7.3 Configuration

```yaml
    interface: wlan0
    channel: "44/20"          # must match the AP's control channel and width
    mac_filter: "88:76:b9:ea:e0:ff,88:76:b9:ea:e1:01"
    trim: false
```

MACs map to topics by list position: `/mobile1/csi`, `/mobile2/csi`.

---

## 8. Two-robot contention

Concurrent runs measure **contended share**, not per-link capacity — roughly
half each, with the sum near the solo figure. Measured: downlink recovered
from 65–84 Mbit/s to 154–166 when the other robot stopped.

Requirements:
- Separate server ports (5201 / 5203), separate `iperf3 -s` per port
- Both robots on VHT, or their CSI isn't comparable
- Both publishing `/ntp/client/status`, or the two bags can't be aligned

`ntp_client_node` publishes to the **absolute** topic `/ntp/client/status`, so
`namespace:=` does not separate them. Either drop the leading slash in the
node's `TOPIC` constant, or use separate `ROS_DOMAIN_ID` per robot (which also
keeps DDS discovery off the link being measured).

Note bidirectional mode makes contention messy: the two robots flip direction
independently, so up/up, up/down and down/down all get averaged together. For
a clean two-robot number, pin both to the same direction.

---

## 9. Parallel streams

Measured sweep (solo, near AP, `-t 10 -O 2`):

| `-P` | Mbit/s |
| --- | --- |
| 1 | 174 |
| 2 | 201 |
| 4 | 196 |
| 6 | 200 |
| 8 | 201 |

Knee at 2; everything above is flat within noise. Either 2 or 4 is
defensible — fix one for the whole campaign and report it:

> Throughput saturates at N parallel TCP streams on this hardware (174 Mbit/s
> at 1, 201 at 2, no gain to 8); N was used for all measurements.

Re-run under VHT, since this sweep was taken under HE.

---

## 10. Recording

```bash
ros2 bag record --qos-profile-overrides-path \
    ~/comms_ws/src/comms/config/rosbag_qos_override.yaml \
    -o spot03_loaded \
    /mobile_1/wifi/status /mobile_1/wifi/iperf_up /mobile_1/wifi/iperf_down \
    /mobile_1/wifi/ping /ntp/client/status \
    /mobile1/csi /csi_publisher/status \
    /tf /tf_static /odom
```

- The QoS override is required — `/ntp/client/status` is `TRANSIENT_LOCAL`
- Record to the robot's **local disk**, never over the link being measured
- One bag per pass per location, named for both
- Ctrl-C the bag before the launch, so the last samples are written

Post-processing:

```bash
python3 ~/comms_ws/src/comms/scripts/ntp_bag_postprocess.py \
    ./spot03_loaded /mobile_1/wifi/iperf_up /ntp/client/status
```

Needs `pip install rosbags numpy`. Outputs
`original_ns, corrected_ns, offset_s, delta_us, stepped` per message.

### Analysis filters

```python
df = df[df.success]                       # drop link-down samples
up = df[~df.reverse]                      # group by direction
data_csi = csi[csi.seq != 65535]          # data frames only
rtt = up[up.rtt_ms_mean.notna()]          # uplink RTT only
```

---

## 11. The core trade-off

**Resource allocation needs the link configured normally.** Aggregation on,
MCS 9, NSS 2 — because that's the link the allocation decision is about. This
caps data-frame CSI at ~8.5 Hz.

**Sensing needs 100 Hz+.** Published datasets get there by disabling
aggregation, forcing a single spatial stream, and injecting fixed-rate traffic
purely to probe the channel — they make no throughput claim.

Lowering `rtw_tx_ampdu_num` would raise the CSI rate but detune the link,
invalidating the throughput numbers being paired against. **Don't**, while
resource allocation is the goal.

At 8.5 Hz you get ~8 CSI samples per continuous throughput sample, or ~95 per
periodic test. There is more CSI than there are labels — density is not the
constraint.

### Using both frame types

- **Data frames** are the predictor: measured under the same modulation whose
  throughput you're predicting.
- **Block Acks** are the context: the dense 116 Hz stream shows whether the
  channel held steady across the 120 ms gaps between data samples. A rate
  prediction from one data-frame sample is only trustworthy if the BA stream
  says the channel didn't move around it.

They arrive interleaved on the same topic. Record both, split on `seq`.

---

## 12. NTP node: five defects fixed (13 Sep)

The `ntp_monitor` code carried over from the original package had five
parser defects, all silent:

| field | was | now |
| --- | --- | --- |
| `offset_seconds` | sign discarded: every offset positive | signed, negative = clock behind NTP |
| `delay_seconds` | parsed wrong column of `sources -v`, ValueError, stayed 0.0 | `chronyc ntpdata` "Peer delay" |
| `jitter_seconds` | same, stayed 0.0 | `chronyc sourcestats` Std Dev |
| `frequency_error_ppm` | sign discarded | signed, negative = runs slow |
| `reference_time` | publish time, changed every message | chrony's own `Ref time (UTC)` from `tracking`, 1 s resolution, changes only when a poll lands |

Added `last_offset_seconds` (measured, per poll), `skew_ppm`, `fit_samples`.

`chronyc ntpdata` needs the daemon's Unix socket. Unprivileged, it prints
nothing and delay would stay 0.0; the parser then falls back to twice the
`sources -v` error bound and adds a warning saying so. To get the real value:
`sudo usermod -aG _chrony $USER` and re-login. Server node brought to parity.
Tests in `test/test_ntp_parse.py` pin all of this against real chrony formats.
**Message definition changed** -- bags recorded before this need the old
`comms_msgs` to read. For those bags, `offset_seconds` is unsigned and
`delay`/`jitter` are zero.

The one-sample-per-run symptom was chrony's poll backoff, not the node:
set `minpoll 3 maxpoll 3` (8 s) in `chrony.conf` on each client.

## 12b. Per-client view: activity, not offsets (13 Sep)

`chronyc clients` prints packets, drops, the client's own poll interval
(`Int` is its log2) and seconds since last contact. It has **no offset or
jitter column** -- a server answers requests, it never computes a client's
clock error. The parser previously read `Int`/`IntL`/`Last` as
stratum/offset/jitter, and took the `501 Not authorised` status line for a
client called "501", which made the server node create the topic
`/ntp/clients/501/status` and die on an invalid topic name.

Now: status lines are skipped, `connected_clients` is -1 when the socket is
not readable, topic segments are sanitised (`192.168.25.31` ->
`ip_192_168_25_31`, `wicoms-robot1` -> `wicoms_robot1`), and the per-client
message carries the activity counts in `warnings` with the poll interval in
`poll_interval_seconds`. Use it to confirm both robots are polling at 8 s
from the server's side; it is not a second measurement of the offset.

`chronyc clients` and `chronyc ntpdata` both need socket access. Without it,
delay is an upper bound and the client list is empty:
`sudo usermod -aG _chrony $USER`, then re-login.

## 13. Open decisions

1. **20 MHz or 80 MHz.** 80 recovers the VHT bandwidth loss and gives 256-slot
   CSI. Decide before collecting, not during — it invalidates comparisons.
2. **Stationary spots or moving route.** Periodic mode gives the cleanest
   throughput labels but is stationary-only. Continuous works moving at 1 m
   per sample.
3. **`parallel` value** — 2 or 4, re-measured under VHT.
4. **NTP topic namespacing** — needed before both robots record to one domain.
5. **Bidirectional or two unidirectional passes.** Two passes give full
   density on each direction with no seams; bidirectional halves per-direction
   rate and costs 2.1 s per swap.

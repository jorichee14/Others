# ros2_wifi_monitor

A ROS 2 node that monitors a wireless (Wi‑Fi) interface and publishes link
quality and traffic statistics — **RSSI, SNR, link quality, bit rate, TX
power, frequency/channel**, the wireless error counters, and the RX/TX
packet/byte/error/drop counters — as a custom message plus standard ROS
diagnostics.

It was built from live output like this:

```
wlx8876b9eae0ff  IEEE 802.11  ESSID:"BML"
          Mode:Managed  Frequency:5.18 GHz  Access Point: 82:2A:A8:CB:D4:34
          Bit Rate=270 Mb/s   Tx-Power=12 dBm
          Link Quality=70/70  Signal level=-37 dBm
          Rx invalid nwid:0  Rx invalid crypt:0  Rx invalid frag:0
          Tx excessive retries:0  Invalid misc:0   Missed beacon:0
```

## Packages

| Package             | Build type   | Contents                                   |
| ------------------- | ------------ | ------------------------------------------ |
| `wifi_monitor_msgs` | ament_cmake  | `WifiLinkStatus.msg`, `IperfResult.msg`, `PingStat.msg` |
| `wifi_monitor`      | ament_python | `wifi_monitor_node`, `iperf_runner_node`, `ping_monitor_node`, launch files, tests |

Tested against ROS 2 (Humble / Iron / Jazzy). Python‑only node, no compiled
dependencies beyond the message package.

## Data sources

The node merges several sources so it degrades gracefully when a driver is
stingy with data:

| Source                                   | Fields                                    |
| ---------------------------------------- | ----------------------------------------- |
| `iw dev <iface> link` (preferred)        | ESSID, BSSID, frequency, bitrate, signal  |
| `iwconfig <iface>` (legacy fallback)     | the above **plus** the wireless error counters (nwid/crypt/frag, excessive retries, missed beacon) |
| `/proc/net/wireless`                     | link quality, signal **and noise floor**  |
| `/sys/class/net/<iface>/statistics/*`    | RX/TX packets, bytes, errors, dropped, overruns, carrier, collisions |

**SNR** is computed as `signal_dbm - noise_dbm` and is only populated when
the driver actually reports a noise floor (`noise_valid = true`). Many
consumer Wi‑Fi drivers do not report noise; in that case `snr_db` and
`noise_dbm` are published as `NaN`. Unknown float fields are always `NaN`
rather than a misleading `0.0`.

> **Noise floor availability.** The noise value is taken from
> `iw ... survey dump` first, then `/proc/net/wireless`. Some USB adapters
> report a sentinel of `-256` (or `0`) in `/proc/net/wireless`, meaning "not
> measured" — the node correctly discards that. If neither source yields a
> real noise floor, `snr_db` stays `NaN`; use the RSSI (`signal_dbm`) and
> link quality as your signal indicators instead.

### Richer PHY + reliability data (from `iw`)

Beyond RSSI, the node parses `iw dev <iface> link`, `station dump` and
`survey dump`:

| Field | Source | Example |
| ----- | ------ | ------- |
| `rx_bitrate_mbps` / `tx_bitrate_mbps` | link / station | 162.0 / 240.0 |
| `rx_mcs` / `tx_mcs` | link / station | 4 / 5 |
| `rx_nss` / `tx_nss` | link / station | 2 / 2 |
| `rx_width_mhz` / `tx_width_mhz` | link / station | 40 |
| `rx_phy_mode` / `tx_phy_mode` | link / station | VHT / HE / HT |
| `tx_short_gi` | link / station | true |
| `signal_avg_dbm` | station dump | -65 |
| `tx_retries` / `tx_failed` | station dump | 123 / 4 |
| `expected_mbps` | station dump | 114.7 |
| `connected_time_s` | station dump | 3600 |
| `sta_rx_bytes` / `sta_tx_bytes` (per-assoc) | link / station | 170807384 |
| `noise_dbm` / `snr_db` | survey dump / proc | -95 / 29 |
| `channel_active_ms` / `channel_busy_ms` / `channel_busy_ratio` | survey dump | 10000 / 2500 / 0.25 |

`station dump` and `survey dump` are read-only and usually need no root, but
if a driver denies them the node falls back to `NaN`/`-1` for those fields
and keeps publishing everything else. `iw ... scan` (triggering a scan) does
require `CAP_NET_ADMIN` and is intentionally **not** used.

## Message: `wifi_monitor_msgs/msg/WifiLinkStatus`

Key fields (see [the full definition](wifi_monitor_ws/src/wifi_monitor_msgs/msg/WifiLinkStatus.msg)):

```
std_msgs/Header header
string  interface          # wlx8876b9eae0ff
string  essid              # BML
string  bssid              # 82:2A:A8:CB:D4:34
float64 frequency_ghz      # 5.18
int32   channel            # 36
float64 bit_rate_mbps      # 270.0
float64 tx_power_dbm       # 12.0
int32   link_quality       # 70
int32   link_quality_max   # 70
float64 signal_dbm         # -37.0  (RSSI)
float64 noise_dbm          # NaN if not reported
float64 snr_db             # signal - noise, NaN if noise unknown
int64   missed_beacon
int64   tx_excessive_retries
uint64  rx_packets / tx_packets / rx_errors / tx_errors / rx_dropped ...
```

## Build

```bash
cd wifi_monitor_ws
rosdep install --from-paths src --ignore-src -r -y   # optional
colcon build
source install/setup.bash
```

## Run

```bash
# Auto-detect the first wireless interface:
ros2 run wifi_monitor wifi_monitor_node

# Or specify the interface and rate:
ros2 run wifi_monitor wifi_monitor_node --ros-args \
    -p interface:=wlx8876b9eae0ff -p publish_rate_hz:=5.0

# Via launch (all parameters overridable):
ros2 launch wifi_monitor wifi_monitor.launch.py interface:=wlx8876b9eae0ff
```

Inspect the output:

```bash
ros2 topic echo /wifi/status
ros2 topic echo /diagnostics
```

## Active throughput: `iperf_runner` node

`wifi_monitor` is passive (RSSI, MCS/rate, retries). To measure **actual
throughput / capacity**, the `iperf_runner` node periodically runs `iperf3`
against a fixed server and publishes an `IperfResult` on `/wifi/iperf`.

**Setup — single wireless hop.** Put the iperf3 server on a machine **wired
(Ethernet) to the router/AP**, not on Wi-Fi. Then the only wireless hop is
the robot's link, so the measurement reflects *the robot's* Wi-Fi capacity:

```
robot ──wifi──► AP ──ethernet──► laptop (iperf3 -s)
```

On the server (the wired laptop at `192.168.233.142`; Windows or Linux):

```bash
iperf3 -s                 # listens on 5201
# Windows: allow inbound TCP/UDP 5201 through Windows Firewall
```

On the robot (the server defaults to `192.168.233.142`):

```bash
# TCP capacity + RTT, a 2 s test every 30 s (uplink):
ros2 launch wifi_monitor iperf_runner.launch.py

# Downlink instead (server -> robot):
ros2 launch wifi_monitor iperf_runner.launch.py reverse:=true

# Both directions, alternating test by test (see "Bidirectional" below):
ros2 launch wifi_monitor iperf_runner.launch.py bidirectional:=true

# UDP loss/jitter, pushing 300 Mbit/s:
ros2 launch wifi_monitor iperf_runner.launch.py protocol:=udp udp_bitrate_mbps:=300

# Continuous 1 Hz survey mode (ONE long iperf, per-second results):
ros2 launch wifi_monitor iperf_runner.launch.py continuous:=true parallel:=4

# Different server:
ros2 launch wifi_monitor iperf_runner.launch.py server_address:=192.168.233.50
```

`IperfResult` carries `bitrate_mbps` (goodput), `retransmits`,
`rtt_ms_mean/min/max` (TCP), and `jitter_ms` / `lost_packets` /
`lost_percent` (UDP), plus `success`/`error`.

> ⚠️ iperf **saturates the link** for the test duration — use short periodic
> bursts (`interval_s` >> `duration_s`) during real operation, or a dedicated
> survey pass. Don't run it continuously while the robot depends on the link.

**Two survey styles:**

- **Periodic bursts** (default) — a short test every `interval_s`. Each test
  pays a TCP connect/teardown cost, so back-to-back tests only reach ~0.25 Hz.
  Good for spot-checks during normal operation with a long interval.
- **Continuous 1 Hz** (`continuous:=true`) — keeps a **single** iperf3 open and
  publishes a result **every second** from its interval reports, with no
  per-test connection overhead. This is the densest throughput sampling
  available and the right choice for a *dedicated survey pass* on a moving
  robot. It saturates the link the whole time, so don't use it during real
  operation. Use `parallel:=4` to fill the link. `jitter`/`loss` are `NaN`
  in this mode.

  **RTT in continuous mode (`rtt_via_ss`, default on).** Continuous mode fills
  `rtt_ms_mean/min/max` by sampling the live connection's kernel TCP RTT via
  `ss -ti` each second — the *same* `tcpi_rtt` iperf reports, but dense and
  measured on the very connection carrying the throughput. So you get 1 Hz
  throughput **and** RTT from one consistent source, no separate ping. The
  idle control socket (`app_limited`) is excluded so the average reflects the
  loaded data streams. Needs `iproute2` (`sudo apt install -y iproute2`); if
  `ss` is missing, RTT falls back to `NaN`.

> **Stop-and-measure is the most accurate way to map capacity.** A multi-second
> test taken while moving smears over several metres and the channel changes
> mid-test. For clean, comparable capacity numbers, pause the robot at
> waypoints and run a full `duration_s:=5 parallel:=4` test; the stops show up
> offline as zero-velocity segments in `/odom`. Use continuous mode when you
> want a dense throughput trace along a slow survey drive instead.

### Bidirectional (uplink + downlink) from one instance

`bidirectional:=true` makes a single `iperf_runner` instance **alternate
direction** — in periodic mode every other test is downlink (`-R`), in
continuous mode it swaps every `bidir_period_s` seconds (default 10, one
reconnect per swap). Nothing changes on the server side; `-R` only tells the
existing server to send instead of receive. Every published message records
its direction in the **`reverse` field** (`false` = uplink, `true` =
downlink), so the two directions split cleanly offline:

```bash
# Periodic: up at t=0, down at t=30, up at t=60, ...
ros2 launch wifi_monitor iperf_runner.launch.py bidirectional:=true

# Continuous survey: 10 s up, 10 s down, 10 s up, ... at ~1 Hz
ros2 launch wifi_monitor survey.launch.py bidirectional:=true

# Longer segments (20 s per direction):
ros2 launch wifi_monitor iperf_runner.launch.py \
    continuous:=true bidirectional:=true bidir_period_s:=20
```

**Why alternating and not iperf3 `--bidir` (both at once)?** Wi-Fi is
half-duplex — uplink and downlink share the same airtime. With `--bidir` the
two directions contend with *each other*, so neither number is that
direction's capacity; you'd measure an arbitrary split of one channel.
Alternating measures each direction at full channel capacity and halves
nothing. The cost is time resolution: each direction is sampled half as
often (a position sample every ~2·`bidir_period_s` along the path in
continuous mode — shrink `bidir_period_s` if you need denser coverage, at
one reconnect per swap).

Note: uplink and downlink genuinely differ on Wi-Fi (robot TX power/antennas
vs. AP TX power, different MCS per direction), which is exactly why measuring
both is worth it.

### Robot-to-robot (agent-to-agent) measurement

To also measure the link **between two robots**, reuse the same node: one
robot plays iperf3 server, the other runs a second `iperf_runner` instance
pointed at it. No new software is needed on either side beyond `iperf3`.

On robot B (the "server" robot) — use a **different port** than the laptop
server so the two measurements can never grab each other's server:

```bash
iperf3 -s -p 5202          # keep it running (e.g. under systemd or tmux)
```

On robot A, run the robot-to-robot instance **alongside** the robot-to-server
one, with its own node name, topic, and a half-interval offset:

```bash
# Instance 1: robot A <-> laptop, alternating up/down at t = 0, 30, 60, ...
ros2 launch wifi_monitor iperf_runner.launch.py \
    server_address:=192.168.233.142 bidirectional:=true

# Instance 2: robot A <-> robot B, alternating up/down at t = 15, 45, 75, ...
ros2 launch wifi_monitor iperf_runner.launch.py \
    server_address:=<robot_B_ip> server_port:=5202 \
    name:=iperf_runner_r2r topic:=wifi/iperf_r2r start_delay_s:=15 \
    bidirectional:=true
```

With `bidirectional:=true` on both, each instance alternates direction test
by test, so robot A collects all four series — up/down to the server and
A→B / B→A — from two client instances, and each sample's `reverse` field
says which direction it was. Note each *direction* is then sampled every
2×`interval_s`.

Each instance publishes on its own topic (`/wifi/iperf` vs
`/wifi/iperf_r2r`), and every message carries `server_address`/`server_port`,
so the two data sets stay cleanly separable in the bag:

```bash
ros2 bag record /wifi/status /wifi/iperf /wifi/iperf_r2r /wifi/ping /tf /odom
```

**Does it affect the other measurements? Yes, if they overlap — so don't let
them overlap.** All of this traffic shares the same Wi-Fi channel (airtime),
and iperf deliberately saturates it. Concretely:

* **Interleave, don't parallelize.** Two iperf tests running at the same
  moment split the airtime and each reports roughly half the true capacity.
  Give both instances the same `interval_s` and offset one by
  `start_delay_s:=interval_s/2` (as above); with `duration_s` of a few
  seconds and `interval_s` of 30, the tests never touch each other. The
  offset is best-effort (it assumes both instances start around the same
  time and drifts if tests fail), so keep `duration_s << interval_s` for
  margin, and sanity-check offline that the `header.stamp` windows of the
  two topics don't overlap.
* **Never run `continuous:=true` on both at once** — continuous mode
  saturates the link 100% of the time. Do robot-to-server and robot-to-robot
  as **separate survey passes**, or keep one continuous and the other off.
* **A robot-to-robot test also perturbs robot B.** The traffic occupies
  robot B's link too, so pause/offset any measurement robot B itself is
  running during robot A's r2r test slots (and vice versa). Same for two
  robots testing against the laptop simultaneously — same channel, same
  airtime, interleave them as well.
* **One `iperf3 -s` serves one client at a time.** A second client that
  connects mid-test is refused with "server busy". Distinct ports per
  measurement pair (5201 laptop, 5202 robot B, ...) avoid this entirely.

#### Full two-robot deployment (agent↔server + agent↔agent)

The complete recipe for measuring both robots against the server **and** the
robot-to-robot link, on one shared 30 s cycle with three non-overlapping test
slots (t = 0, 10, 20 s). Direction policy is **per instance**: here the
agent↔server instances run one direction only (add `reverse:=true` for
downlink, or `bidirectional:=true` for both), while the agent↔agent instance
runs bidirectional — one client on robot A then covers both A→B and B→A, so
robot B never needs an r2r client:

**Laptop** (wired, 192.168.233.142) — one server per client so a drifted
schedule can never hit "server busy":

```bash
iperf3 -s -p 5201 &        # serves robot A
iperf3 -s -p 5211 &        # serves robot B
```

**Robot B** — additionally serves robot A's r2r tests:

```bash
iperf3 -s -p 5202
```

**Robot A** — two client instances (plus the passive monitor):

```bash
# A -> server, uplink only, slot t = 0
# (reverse:=true for downlink instead, bidirectional:=true for both)
ros2 launch wifi_monitor iperf_runner.launch.py \
    server_address:=192.168.233.142 server_port:=5201

# A <-> B, both directions alternating, slot t = 10
ros2 launch wifi_monitor iperf_runner.launch.py \
    server_address:=<robot_B_ip> server_port:=5202 \
    name:=iperf_runner_r2r topic:=wifi/iperf_r2r \
    bidirectional:=true start_delay_s:=10
```

**Robot B** — one client instance, slot t = 20:

```bash
# B -> server, uplink only, slot t = 20
ros2 launch wifi_monitor iperf_runner.launch.py \
    server_address:=192.168.233.142 server_port:=5211 \
    start_delay_s:=20
```

That yields four series — A→server, B→server (every 30 s each), and A→B /
B→A (every 60 s each, since the r2r instance alternates). Record on **each**
robot locally:

```bash
# robot A:
ros2 bag record /wifi/status /wifi/iperf /wifi/iperf_r2r /tf /odom
# robot B:
ros2 bag record /wifi/status /wifi/iperf /tf /odom
```

Offline, split by topic (which pair), by `reverse` (which direction), and by
bag (which robot).

Practicalities:

* The slots are offsets from each *node's own start*, so start the three
  client instances within a few seconds of each other, and sync the robots'
  clocks (chrony/NTP) — you need that anyway to time-join stamps to pose
  across machines. Verify offline that the `header.stamp` windows of the
  three topics don't overlap; with 2–5 s tests in 10 s slots there is wide
  margin.
* Default `duration_s:=2.0` fits easily; don't push `duration_s` past ~8 s
  or the slots start touching.
* This periodic scheme runs during normal operation. For **continuous**
  survey passes there is no interleaving trick — continuous mode owns the
  channel — so do them one at a time: pass 1 A↔server, pass 2 A↔B, pass 3
  B↔server (`continuous:=true bidirectional:=true`, others off).

**Interpreting the number.** In infrastructure mode the robot-to-robot path
is `robot A ──wifi──► AP ──wifi──► robot B`: **two wireless hops on the same
channel**. The AP must receive every frame and retransmit it, so the
end-to-end throughput is at best about **half** the weaker robot's single-hop
capacity — a correct measurement of what robots actually get when they talk
to each other, but not a property of a single link. Comparing it with each
robot's own `/wifi/iperf` (single-hop) trace tells you which side of the
relay is the bottleneck. (Only a direct ad-hoc/mesh link between the robots
would measure a single robot-to-robot hop; that is a different network
setup, not an iperf option.)

Passive `/wifi/status` and `ping` are so cheap they are unaffected by any of
this — though a ping RTT sampled *during* someone's iperf burst will show the
queueing delay of the saturated link (which can itself be a useful signal:
that is latency-under-load).

### Failsafes for a moving robot (disconnect / reconnect)

Both nodes are built to survive the link dropping and coming back as the
robot moves:

**`wifi_monitor`** keeps publishing right through an outage — during a
disconnect it emits `associated=false` with RF fields as `NaN` (so gaps are
explicit in the data, not missing samples), and logs the disconnect and
reconnect edges. It never blocks or crashes if the interface goes away.

**`iperf_runner`** guards every test:
- **Link-down gate** — before launching iperf it checks the interface
  carrier. If the Wi-Fi is not associated it does *not* start a test (which
  would hang); it publishes a `success=false` result tagged `link down` and
  polls every `reconnect_poll_s` until the link returns, then resumes
  automatically. Those `link down` samples mark dead zones in your map.
- **`--connect-timeout`** (`connect_timeout_ms`, default 2000) bounds how
  long a test waits for the server, so a reachable-Wi-Fi/unreachable-server
  case fails fast instead of stalling.
- **Failure backoff** — a failed test backs off to `reconnect_poll_s` even in
  survey mode (`interval_s=0`), so a dead server is never hammered.
- All subprocess errors (timeout, launch failure, bad JSON) are caught and
  reported as `success=false` with an `error` string; the node never dies.

## Latency + loss: `ping_monitor` node

iperf's live throughput stream carries **no RTT** (and on iperf 3.9 there is
no per-interval JSON), so continuous latency/loss comes from a separate,
cheap source: `ping`. The `ping_monitor` node pings a fixed host and
publishes a `PingStat` on `/wifi/ping` with the per-ping RTT plus
**rolling-window loss and RTT stats**. Ping does **not** saturate the link,
so unlike iperf it can run continuously during real operation.

The target only needs to answer ICMP — **no ROS or software on the other
end**. On the Windows laptop, allow *ICMP Echo Request* through the firewall:

```powershell
New-NetFirewallRule -DisplayName "ICMPv4-In" -Protocol ICMPv4 -IcmpType 8 -Direction Inbound -Action Allow
```

Run it:
```bash
ros2 launch wifi_monitor ping_monitor.launch.py target:=192.168.233.142
ros2 topic echo /wifi/ping     # rtt_ms, loss_percent, rtt_ms_avg/min/max
```

## Full survey stack in one launch

`survey.launch.py` starts all three nodes at once (passive + continuous
throughput + latency/loss), sharing one server/target address:

```bash
# Dedicated survey pass -- throughput + RTT (iperf+ss), saturates the link:
ros2 launch wifi_monitor survey.launch.py server_address:=192.168.233.142

# Non-saturating monitoring during real operation (passive + ping, no iperf):
ros2 launch wifi_monitor survey.launch.py \
    server_address:=192.168.233.142 run_iperf:=false run_ping:=true
```

> Since continuous iperf now reports RTT itself (via `ss`), the ping node is
> **off by default** (`run_ping:=false`) — it would only duplicate the RTT.
> Turn it on when you drop iperf (`run_iperf:=false`) and still want RTT +
> loss without saturating the link.

This gives you, together:

| Topic | Rate | Content |
| ----- | ---- | ------- |
| `/wifi/status` | 5 Hz | RSSI, MCS/rate, retries (passive, continuous) |
| `/wifi/iperf`  | ~1 Hz | max achievable throughput (continuous; `run_iperf:=true`) |
| `/wifi/ping`   | 1 Hz | RTT + rolling packet loss (continuous, cheap) |

### Mapping results to position

Every message is stamped, so record locally on the robot and time-join to
pose offline:

```bash
ros2 bag record /wifi/status /wifi/iperf /wifi/ping /tf /odom
```

Recording **on the robot's local disk** (not streamed over the Wi-Fi you are
measuring) avoids losing data exactly when the link degrades.

## Parameters

| Parameter          | Default | Description                                    |
| ------------------ | ------- | ---------------------------------------------- |
| `interface`        | `""`    | Interface to monitor (empty = auto-detect).    |
| `publish_rate_hz`  | `5.0`   | Sampling / publishing rate. 5 Hz suits a robot up to ~1 m/s; ~5 Hz is the practical ceiling. |
| `frame_id`         | `wifi`  | `header.frame_id` stamped on each message.     |
| `warn_signal_dbm`  | `-70.0` | RSSI at/below which diagnostics report WARN.    |
| `error_signal_dbm` | `-80.0` | RSSI at/below which diagnostics report ERROR.   |

### When the driver reports no noise floor

Many USB adapters (e.g. Realtek `rtl88xx`/`rtl8852bu`) never report a noise
floor, so a real SNR cannot be computed. In that case the node makes the
situation explicit rather than guessing:

* `signal_dbm` (RSSI) is still published — use it as your signal indicator.
* `noise_dbm` and `snr_db` are published as **`NaN`**.
* `noise_valid = false` and **`snr_valid = false`**.

The node never fabricates an SNR from an assumed noise floor. For a
**measured** noise floor / true SNR, use an `ath9k_htc`, `mt76`, or Intel
`iwlwifi` adapter — those report it via `iw survey dump`, and the node then
sets `snr_valid = true` automatically.

## Notes

* `iw`/`iwconfig` read link state without root. The node never needs root.
* If neither `iw` nor `iwconfig` is installed, the node still publishes
  MAC/link-state and the `/sys` traffic counters, leaving RF fields as
  `NaN`. Install with `sudo apt install wireless-tools iw`.

## Tests

```bash
cd wifi_monitor_ws/src/wifi_monitor
python3 -m pytest test/          # pure-Python parser tests, no hardware
```

## License

MIT

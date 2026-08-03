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

See [`docs/architecture.md`](wifi_monitor_ws/src/wifi_monitor/docs/architecture.md)
for a block diagram of the three nodes, their data sources, topics and consumers.

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

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
| `wifi_monitor_msgs` | ament_cmake  | `WifiLinkStatus.msg` (custom message type) |
| `wifi_monitor`      | ament_python | `wifi_monitor_node`, launch file, tests    |

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
    -p interface:=wlx8876b9eae0ff -p publish_rate_hz:=2.0

# Via launch (all parameters overridable):
ros2 launch wifi_monitor wifi_monitor.launch.py interface:=wlx8876b9eae0ff
```

Inspect the output:

```bash
ros2 topic echo /wifi/status
ros2 topic echo /diagnostics
```

## Parameters

| Parameter          | Default | Description                                    |
| ------------------ | ------- | ---------------------------------------------- |
| `interface`        | `""`    | Interface to monitor (empty = auto-detect).    |
| `publish_rate_hz`  | `1.0`   | Sampling / publishing rate.                    |
| `frame_id`         | `wifi`  | `header.frame_id` stamped on each message.     |
| `warn_signal_dbm`  | `-70.0` | RSSI at/below which diagnostics report WARN.    |
| `error_signal_dbm` | `-80.0` | RSSI at/below which diagnostics report ERROR.   |
| `assumed_noise_dbm`| `NaN`   | Assumed noise floor to **estimate** SNR when the driver reports none (e.g. `-95`). `NaN` disables estimation. |

### Estimated SNR (for drivers without a noise floor)

Many USB adapters (e.g. Realtek `rtl88xx`/`rtl8852bu`) never report a noise
floor, so a *measured* SNR is impossible. Set `assumed_noise_dbm` to get an
**estimate** instead:

```bash
ros2 run wifi_monitor wifi_monitor_node --ros-args \
    -p interface:=wlx8876b9eae0ff -p assumed_noise_dbm:=-95.0
```

Then `snr_db = signal_dbm - assumed_noise_dbm`, published with
`snr_estimated = true` and `noise_valid = false` so it is never confused with
a measured value. Leave the parameter at `NaN` (default) to publish `snr_db`
as `NaN` whenever noise is unmeasured. Typical assumed floors: `-95` dBm at
5 GHz, `-90` dBm at 2.4 GHz. For a **measured** noise floor / true SNR, use
an `ath9k_htc`, `mt76`, or Intel `iwlwifi` adapter — those report it via
`iw survey dump`.

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

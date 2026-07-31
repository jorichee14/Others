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

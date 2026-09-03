# wifi_csi — ROS 2 CSI publisher for Nexmon

Publishes Nexmon WiFi CSI as ROS 2 topics, using exactly the commands verified
by hand on a Pi 4B / Ubuntu 22.04 with a BCM43455c0 running 7.45.189
(nexmon.org/csi: a975-1).

```
wifi_csi_msgs/   CsiFrame, CsiStatus
wifi_csi/        csi_publisher (arms + publishes), csi_monitor (CLI summary)
```

## Build

```bash
cd ~/ros2_ws/src
cp -r wifi_csi wifi_csi_msgs .
cd ~/ros2_ws
colcon build --packages-select wifi_csi_msgs wifi_csi
source install/setup.bash
```

## Run

`nexutil` needs `cap_net_admin`, already granted during the nexmon build, so the
node does not need root.

```bash
ros2 launch wifi_csi csi.launch.py
ros2 run wifi_csi csi_monitor
ros2 topic hz /csi_publisher/csi
```

Set the transmitter MAC in `config/csi.yaml` before collecting anything real.

## Topics

| Topic | Type | Rate |
|---|---|---|
| `~/csi` | `wifi_csi_msgs/CsiFrame` | one per received OFDM frame (~100–500 Hz) |
| `~/status` | `wifi_csi_msgs/CsiStatus` | 0.5 Hz |
| `/mobileN/csi` | `wifi_csi_msgs/CsiFrame` | that transmitter's share of `~/csi` |

`/mobileN` is one namespace per entry in `mac_filter`, numbered by position:
the first MAC in the list is `/mobile1`, the second `/mobile2`. The node logs
the mapping at startup. `~/csi` still carries every frame in arrival order.

`status` is deliberately NOT duplicated per mobile: it reports chip-level state
(chanspec, monitor mode, firmware, totals) which is shared, since there is one
radio. Per-mobile liveness is `ros2 topic hz /mobileN/csi`.

`CsiFrame` carries `subcarrier_index[]` alongside `csi_real[]`/`csi_imag[]`, so
indices stay comparable across nodes and captures even after trimming.

## Design notes

**QoS is best-effort, depth 100.** CSI arrives at hundreds of Hz and a late frame
is worthless; dropping beats queueing. Recording with `ros2 bag record` needs a
matching best-effort profile or it silently receives nothing.

**Reads the socket directly**, not `tcpdump`. The firmware broadcasts to UDP 5500,
so binding it needs no external process and no pipe to buffer.

**Constant slots are found by calibration, not hardcoded.** Some slots carry fixed
firmware fields rather than measurements — on the verified node, raw 127–131 held
`std == 0.0` and a single unique value across 6164 frames, at magnitudes up to
11498 against ~1850 for real subcarriers. A constant is only visible as constant
across many frames, so the node collects `calib_frames` (default 300) first and
publishes untrimmed until then. Their positions are not a pattern worth
generalising; the node measures them per boot.

**The DC null is the acceptance test.** After trimming you should get exactly 56
subcarriers for a 20 MHz transmission, with the removed null at the centre of the
span — on the verified node raw 160, exactly ±28 from both edges. If the null is
off-centre, the parse is misaligned.

**Arming retries.** Twice on cold boot the extractor reported `monitor=1` with the
correct chanspec and emitted nothing until `-s500` was reissued; the node retries
up to `arm_attempts` times and logs which attempt succeeded.

## Header layout

Established from captured data, not documentation:

| Offset | Size | Field |
|---|---|---|
| 0–1 | 2 | magic `0x1111` — **2 bytes, not 4** |
| 2 | 1 | RSSI (int8, dBm) |
| 3 | 1 | frame control |
| 4–9 | 6 | source MAC |
| 10–11 | 2 | sequence number |
| 12–13 | 2 | core / spatial stream |
| 14–15 | 2 | chanspec |
| 16–17 | 2 | chip version |
| 18+ | 4×N | subcarriers, int16 pairs in (imag, real) order |

A strict 4-byte magic check rejects every genuine packet.

`chanspec` low byte is the **centre** channel, not the control channel: `0xe02a`
carries 42 while nexutil reports 36/80. `decode_chanspec()` applies the sideband
correction.

**I/Q order matters for phase only.** Getting `imag_first` wrong yields `i*conj(z)`:
amplitude identical, phase *mirrored* (θ → π/2 − θ). Not recoverable by negating
phase downstream. Amplitude-only pipelines are unaffected.

## Gotchas

- **`mac_filter` is essential for real data.** Unfiltered, dozens of transmitters at
  different bandwidths average into a meaningless profile. It takes a
  comma-separated list; each MAC is passed as its own `-m` to `makecsiparams` and
  is also enforced in software, so a build whose `makecsiparams` only accepts one
  `-m` still filters correctly, just with more frames parsed and discarded.
- **An associated station is never silent, so frames arrive without you generating
  any.** Measured here: `frame_control` is overwhelmingly `0x94` (Block Ack), which
  a station emits whenever the AP sends *it* data — nothing to do with what the
  station itself originates. Stopping a ping on one device does not stop its CSI.
  The consequence is that per-mobile rates are uncontrolled and asymmetric: for
  anything comparing two channels over time, drive fixed-rate traffic on both.
- **`frame_control_filter` is a first-byte filter, not a MAC filter.** `0x88` is QoS
  Data; on a channel dominated by Block Acks (`0x94`) it rejects nearly everything.
- **Zero frames/s is often not a fault.** CSI is produced per received OFDM frame;
  an idle channel yields nothing. Check with an unfiltered `tcpdump -i wlan0`.
- **5 GHz only.** 2.4 GHz beacons are DSSS and produce no channel estimate.

## Several transmitters, one chip

One radio can serve several transmitters, and that is the normal case: list them
all in `mac_filter` and each gets a `/mobileN/csi` topic. The binding constraint
is the chanspec — arming tunes the chip to exactly one channel/bandwidth, so
every transmitter you want must share it. Check each with `iw dev <iface> info`
and set `channel` to match:

```
channel 149 (5745 MHz), width: 80 MHz     ->  channel: "149/80"
```

Running a second `csi_publisher` against the same radio does not work and is not
a way around this: arming is global chip state, so the second node's `_arm()`
reprograms the extractor out from under the first, and both would bind UDP 5500
where only one socket actually receives the datagrams.

## Multi-node

Genuinely separate nodes mean separate radios — in practice one Pi each. Give
each a distinct namespace and `frame_id`, and sync clocks: the Pi has no RTC and
without NTP the clock reverts to a stale on-disk value. Cross-node comparison is
meaningless without it.

```bash
ros2 launch wifi_csi csi.launch.py config:=/path/to/csi_b.yaml namespace:=robot_b
```

Vary `namespace`, not `node_name`: the params file keys on `/**/csi_publisher`,
which matches any namespace but only the literal node name `csi_publisher`.
Renaming the node makes it silently pick up none of its parameters.

#!/usr/bin/env python3
"""Write a small synthetic rosbag2-style MCAP with the dataset's NTP schema.

Used only to test extract_bag.py and ntp_analysis.py when the real bag is not
available. Mimics: /infra_1/ntp/status and /mobile_2/ntp/status (clients of
mobile_1), /infra_1/ntp/events, and a few header-bearing sensor topics so the
stamp audit has something to chew on.
"""
from __future__ import annotations

import math
import random
import numpy as np
import sys
from pathlib import Path

from mcap_ros2.writer import Writer

HEADER_DEF = """
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
"""

NTP_STATUS = """std_msgs/Header header
string role
string hostname
bool synchronized
string sync_source
string stratum_level
int32  stratum
float64 offset_seconds
float64 delay_seconds
float64 jitter_seconds
float64 root_delay
float64 root_dispersion
float64 frequency_error_ppm
int32  poll_interval_seconds
int32  reach_register
int32  reachability_percent
builtin_interfaces/Time reference_time
int32  connected_clients
string leap_indicator
string[] warnings
uint64  seq
float64 monotonic_seconds
bool    clock_stepped
float64 offset_delta_seconds
""" + HEADER_DEF

STRING = "string data\n"

WIFI_STATUS = """std_msgs/Header header
string   interface
string   mac_address
bool     up
bool     associated
string   essid
string   bssid
string   mode
float64  frequency_ghz
int32    channel
float64  bit_rate_mbps
float64  tx_power_dbm
int32    link_quality
int32    link_quality_max
float64  link_quality_ratio
float64  signal_dbm
float64  signal_avg_dbm
float64  noise_dbm
float64  snr_db
bool     noise_valid
bool     snr_valid
float64  rx_bitrate_mbps
float64  tx_bitrate_mbps
int32    rx_mcs
int32    tx_mcs
int32    rx_nss
int32    tx_nss
int32    rx_width_mhz
int32    tx_width_mhz
string   rx_phy_mode
string   tx_phy_mode
bool     tx_short_gi
int64    tx_retries
int64    tx_failed
float64  expected_mbps
int64    connected_time_s
int64    sta_rx_bytes
int64    sta_tx_bytes
int64    sta_rx_packets
int64    sta_tx_packets
float64  channel_active_ms
float64  channel_busy_ms
float64  channel_busy_ratio
int64    rx_invalid_nwid
int64    rx_invalid_crypt
int64    rx_invalid_frag
int64    tx_excessive_retries
int64    invalid_misc
int64    missed_beacon
uint64   rx_packets
uint64   rx_bytes
uint64   rx_errors
uint64   rx_dropped
uint64   rx_overruns
uint64   rx_frame_errors
uint64   tx_packets
uint64   tx_bytes
uint64   tx_errors
uint64   tx_dropped
uint64   tx_overruns
uint64   tx_carrier_errors
uint64   collisions
""" + HEADER_DEF

IPERF = """std_msgs/Header header
string   server_address
uint16   server_port
string   protocol
bool     reverse
float64  duration_s
bool     success
string   error
float64  bitrate_mbps
uint64   bytes
uint32   retransmits
float64  rtt_ms_mean
float64  rtt_ms_min
float64  rtt_ms_max
float64  jitter_ms
uint64   lost_packets
uint64   total_packets
float64  lost_percent
""" + HEADER_DEF


POSE = """std_msgs/Header header
geometry_msgs/Pose pose
""" + HEADER_DEF + """
================================================================================
MSG: geometry_msgs/Pose
geometry_msgs/Point position
geometry_msgs/Quaternion orientation
================================================================================
MSG: geometry_msgs/Point
float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
"""


CSI = """std_msgs/Header header
string     src_mac
int8       rssi
uint8      frame_control
uint16     seq
uint8      core
uint8      spatial_stream
uint16     chanspec
uint16     chip_version
uint16     channel
uint16     bandwidth_mhz
int32[]    subcarrier_index
float32[]  csi_real
float32[]  csi_imag
bool       trimmed
uint32     raw_slots
""" + HEADER_DEF


IMU = """std_msgs/Header header
geometry_msgs/Quaternion orientation
float64[9] orientation_covariance
geometry_msgs/Vector3 angular_velocity
float64[9] angular_velocity_covariance
geometry_msgs/Vector3 linear_acceleration
float64[9] linear_acceleration_covariance
""" + HEADER_DEF + """
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
"""


def stamp(ns: int) -> dict:
    return {"sec": ns // 1_000_000_000, "nanosec": ns % 1_000_000_000}


def main(out: Path) -> None:
    rng = random.Random(0)
    t0 = 1_787_899_802_217_921_000
    dur_s = 156.0
    with open(out, "wb") as f:
        w = Writer(f)
        ntp = w.register_msgdef("ntp_monitor_msgs/msg/NtpStatus", NTP_STATUS)
        ev = w.register_msgdef("std_msgs/msg/String", STRING)
        imu = w.register_msgdef("sensor_msgs/msg/Imu", IMU)

        # NTP clients, mimicking the real recording: the poll interval (256 s) is longer than the
        # run, so the daemon delivers only one real measurement; every status message before that
        # first poll carries a placeholder offset of exactly 0.0, and the messages after it repeat
        # the same measurement, slowly extrapolated by the daemon.
        POLL_S = 256
        clients = {
            # topic: (hostname, publish rate, first-poll time, offset at that poll, drift per second)
            "/infra_1/ntp/status": ("wicoms-robot2", 8.5, 0.0, 0.000228, -8e-9),
            "/mobile_2/ntp/status": ("ubuntu", 9.3, 46.0, 0.000673, -7e-7),
        }
        for topic, (host, hz, t_poll, off0, drift) in clients.items():
            n = int(dur_s * hz)
            prev = None
            for i in range(n):
                t = i / hz
                ns = t0 + int(t * 1e9)
                o = 0.0 if t < t_poll else off0 + drift * (t - t_poll)
                stepped = prev is not None and prev == 0.0 and o != 0.0
                msg = {
                    "header": {"stamp": stamp(ns), "frame_id": host},
                    "role": "client",
                    "hostname": host,
                    "synchronized": True,
                    "sync_source": "ubuntu",
                    "stratum_level": "5",
                    "stratum": 5,
                    "offset_seconds": o,
                    "delay_seconds": 0.0,
                    "jitter_seconds": 0.0,
                    "root_delay": 0.01,
                    "root_dispersion": 0.02,
                    "frequency_error_ppm": 12.3,
                    "poll_interval_seconds": POLL_S,
                    "reach_register": 0o377,
                    "reachability_percent": 100,
                    # the daemon updates reference_time only when it actually polls the server
                    "reference_time": stamp(t0 + int(max(t_poll, 0.0) * 1e9)),
                    "connected_clients": -1,
                    "leap_indicator": "none",
                    "warnings": [] if not stepped else ["first poll"],
                    "seq": i,
                    "monotonic_seconds": t,
                    "clock_stepped": stepped,
                    "offset_delta_seconds": 0.0 if prev is None else o - prev,
                }
                prev = o
                # recorder receives ~3 ms later; recorder clock == mobile_1 clock, so the
                # header (in the client's clock) is ahead by the client's offset
                log = ns + 3_000_000 + int(o * 1e9)
                w.write_message(topic, ntp, msg, log_time=log, publish_time=log - 1_000_000)

        for k in range(1):
            ns = t0 + int(4.2e9)
            w.write_message("/infra_1/ntp/events", ev, {"data": f"event {k}: reselected server mobile_1"}, log_time=ns, publish_time=ns)

        # header-bearing sensor topics on each node for the stamp audit
        for topic, host, hz, off, unset_stamp in [
            ("/mobile_1/ouster/imu", "mobile_1", 100.0, 0.0, False),
            ("/mobile_2/imu", "mobile_2", 100.0, 0.0007, False),
            ("/infra_1/imu_fake", "infra_1", 20.0, 0.0002, False),
            # header.stamp never filled in by this driver
            ("/mobile_1/zed/imu/data", "mobile_1", 60.0, 0.0, True),
        ]:
            n = int(dur_s * hz)
            for i in range(n):
                t = i / hz
                ns = 0 if unset_stamp else t0 + int(t * 1e9)
                msg = {
                    "header": {"stamp": stamp(ns), "frame_id": host},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    "orientation_covariance": [0.0] * 9,
                    "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "angular_velocity_covariance": [0.0] * 9,
                    "linear_acceleration": {"x": 0.0, "y": 0.0, "z": 9.81},
                    "linear_acceleration_covariance": [0.0] * 9,
                }
                log = t0 + int(t * 1e9) + 2_000_000 + int(off * 1e9) + rng.randint(0, 1_000_000)
                w.write_message(topic, imu, msg, log_time=log, publish_time=log - 500_000)

        # --- Wi-Fi link status ------------------------------------------------
        # mobile_1 runs two radios on one topic (the dual-radio mode for rho);
        # mobile_2 runs one. Radios are told apart by the `interface` field.
        wifi = w.register_msgdef("wifi_monitor_msgs/msg/WifiLinkStatus", WIFI_STATUS)
        iperf = w.register_msgdef("wifi_monitor_msgs/msg/IperfResult", IPERF)

        radios = [
            # topic, iface, mac, band GHz, channel, base RSSI, fade period s, phase, hz
            ("/mobile_1/wifi/status", "wlx8876b9eae0ff", "88:76:b9:ea:e0:ff", 5.18, 36, -47.0, 38.0, 0.0, 4.9),
            ("/mobile_1/wifi/status", "wlx00c0ca9a1b2c", "00:c0:ca:9a:1b:2c", 2.437, 6, -55.0, 41.0, 1.1, 4.9),
            ("/mobile_2/wifi/status", "wlp2s0", "d8:3a:dd:11:22:33", 5.18, 36, -58.0, 33.0, 2.0, 4.6),
        ]
        for topic, iface, mac, ghz, chan, rssi0, fade_T, phase, hz in radios:
            n = int(dur_s * hz)
            tx_pkts = rx_pkts = tx_ret = tx_fail = 0
            act_ms = busy_ms = 0.0
            for i in range(n):
                t = i / hz
                ns = t0 + int(t * 1e9)
                # slow fading plus a deep fade in the middle of the run
                rssi = rssi0 + 7.0 * math.sin(2 * math.pi * t / fade_T + phase) + rng.gauss(0, 1.2)
                if 70.0 < t < 88.0:
                    rssi -= 14.0 if ghz > 3 else 6.0        # the 5 GHz radio suffers more
                mcs = max(0, min(9, int((rssi + 82) / 4)))
                width = 40 if ghz > 3 else 20
                nss = 2 if ghz > 3 else 1
                tx_rate = 6.5 * (mcs + 1) * nss * (width / 20.0)
                # more retries when the link is weak
                p_ret = min(0.6, max(0.01, (-rssi - 45) / 60.0))
                d_pkts = int(rng.uniform(300, 900))
                tx_pkts += d_pkts
                rx_pkts += int(d_pkts * 1.6)
                tx_ret += int(d_pkts * p_ret)
                tx_fail += int(d_pkts * p_ret * 0.08)
                d_act = 1000.0 / hz
                act_ms += d_act
                busy_ms += d_act * min(0.95, 0.15 + p_ret)
                msg = {
                    "header": {"stamp": stamp(ns), "frame_id": iface},
                    "interface": iface, "mac_address": mac, "up": True, "associated": True,
                    "essid": "BML", "bssid": "82:2a:a8:cb:d4:34", "mode": "Managed",
                    "frequency_ghz": ghz, "channel": chan,
                    "bit_rate_mbps": tx_rate, "tx_power_dbm": 12.0,
                    "link_quality": int(max(0, min(70, 70 + (rssi + 40) * 1.4))), "link_quality_max": 70,
                    "link_quality_ratio": max(0.0, min(1.0, (70 + (rssi + 40) * 1.4) / 70)),
                    "signal_dbm": rssi, "signal_avg_dbm": rssi + rng.gauss(0, 0.4),
                    "noise_dbm": float("nan"), "snr_db": float("nan"),
                    "noise_valid": False, "snr_valid": False,
                    "rx_bitrate_mbps": tx_rate * 0.85, "tx_bitrate_mbps": tx_rate,
                    "rx_mcs": mcs, "tx_mcs": mcs, "rx_nss": nss, "tx_nss": nss,
                    "rx_width_mhz": width, "tx_width_mhz": width,
                    "rx_phy_mode": "VHT" if ghz > 3 else "HT", "tx_phy_mode": "VHT" if ghz > 3 else "HT",
                    "tx_short_gi": True,
                    "tx_retries": tx_ret, "tx_failed": tx_fail,
                    "expected_mbps": tx_rate * 0.6, "connected_time_s": int(600 + t),
                    "sta_rx_bytes": rx_pkts * 800, "sta_tx_bytes": tx_pkts * 700,
                    "sta_rx_packets": rx_pkts, "sta_tx_packets": tx_pkts,
                    "channel_active_ms": act_ms, "channel_busy_ms": busy_ms,
                    "channel_busy_ratio": busy_ms / max(act_ms, 1e-9),
                    "rx_invalid_nwid": 0, "rx_invalid_crypt": 0, "rx_invalid_frag": 0,
                    "tx_excessive_retries": tx_fail, "invalid_misc": 0, "missed_beacon": 0,
                    "rx_packets": rx_pkts, "rx_bytes": rx_pkts * 800, "rx_errors": 0, "rx_dropped": 0,
                    "rx_overruns": 0, "rx_frame_errors": 0,
                    "tx_packets": tx_pkts, "tx_bytes": tx_pkts * 700, "tx_errors": tx_fail,
                    "tx_dropped": 0, "tx_overruns": 0, "tx_carrier_errors": 0, "collisions": 0,
                }
                log = ns + 1_500_000 + rng.randint(0, 500_000)
                w.write_message(topic, wifi, msg, log_time=log, publish_time=log)

        # --- iperf: periodic bursts to the fixed server, plus robot-to-robot ----
        for topic, n_tests, base, r2r in [
            ("/mobile_1/wifi/iperf", 10, 210.0, False),
            ("/mobile_2/wifi/iperf", 4, 140.0, False),
            ("/mobile_1/wifi/iperf_r2r", 10, 95.0, True),
        ]:
            for k in range(n_tests):
                t = 6.0 + k * (dur_s - 12.0) / max(n_tests - 1, 1)
                ns = t0 + int(t * 1e9)
                deep = 70.0 < t < 88.0
                mbps = base * (0.35 if deep else 1.0) * (1 + rng.gauss(0, 0.06))
                msg = {
                    "header": {"stamp": stamp(ns), "frame_id": "wifi"},
                    "server_address": "192.168.1.50" if not r2r else "192.168.1.12",
                    "server_port": 5201, "protocol": "TCP", "reverse": bool(k % 2),
                    "duration_s": 2.0, "success": True, "error": "",
                    "bitrate_mbps": mbps, "bytes": int(mbps * 1e6 / 8 * 2),
                    "retransmits": int(abs(rng.gauss(0, 6)) * (4 if deep else 1)),
                    "rtt_ms_mean": (14.0 if deep else 4.2) + abs(rng.gauss(0, 0.8)),
                    "rtt_ms_min": 2.1, "rtt_ms_max": (40.0 if deep else 11.0),
                    "jitter_ms": float("nan"), "lost_packets": 0, "total_packets": 0,
                    "lost_percent": float("nan"),
                }
                log = ns + 2_000_000
                w.write_message(topic, iperf, msg, log_time=log, publish_time=log)


        # --- ground-truth poses, so link samples can be placed on the map ---
        pose = w.register_msgdef("geometry_msgs/msg/PoseStamped", POSE)
        for topic, phase, r0 in [("/mobile_1/global_pose", 0.0, 7.0),
                                 ("/mobile_2/global_pose", 2.1, 4.5)]:
            hz = 18.0
            n = int(dur_s * hz)
            for i in range(n):
                t = i / hz
                ns = t0 + int(t * 1e9)
                # a lap around the room, so RSSI has somewhere to vary
                a = 2 * math.pi * t / 78.0 + phase
                x = r0 * math.cos(a) + 1.5 * math.cos(3 * a)
                y = 0.6 * r0 * math.sin(a) - 4.0 + 1.0 * math.sin(2 * a)
                msg = {
                    "header": {"stamp": stamp(ns), "frame_id": "map"},
                    "pose": {"position": {"x": x, "y": y, "z": 0.35},
                             "orientation": {"x": 0.0, "y": 0.0,
                                             "z": math.sin(a / 2), "w": math.cos(a / 2)}},
                }
                log = ns + 1_000_000
                w.write_message(topic, pose, msg, log_time=log, publish_time=log)


        # --- Wi-Fi CSI ---------------------------------------------------------
        # A synthetic multipath channel so the derived metrics can be checked
        # against a known truth: a dominant (line-of-sight) tap plus scatterers.
        # Between t=70 s and t=88 s the dominant tap is attenuated and the
        # scatterers strengthened, which must show up as the K-factor falling
        # and the RMS delay spread rising.
        csi = w.register_msgdef("wifi_csi_msgs/msg/CsiFrame", CSI)
        RAW, BW_HZ = 256, 80e6
        keep = np.array([i for i in range(RAW)
                         if not (i < 6 or i > 249 or abs(i - 128) < 3)], dtype=np.int32)
        f_hz = (keep - RAW / 2) * (BW_HZ / RAW)
        for topic, mac, phase in [("/mobile1/csi", "82:2a:a8:cb:d4:34", 0.0),
                                  ("/mobile2/csi", "82:2a:a8:cb:d4:34", 1.7)]:
            hz = 171.0
            n = int(dur_s * hz)
            for i in range(n):
                t = i / hz
                ns = t0 + int(t * 1e9)
                nlos = 70.0 < t < 88.0
                # dominant tap, then a few scatterers at longer delays
                taps = [(1.0 if not nlos else 0.18, 0.0)]
                for k in range(1, 7):
                    d = (18e-9 if not nlos else 55e-9) * k
                    a = (0.22 if not nlos else 0.55) * math.exp(-k / 2.4)
                    taps.append((a * (1 + 0.25 * math.sin(3 * t + k + phase)), d))
                H = np.zeros(len(keep), dtype=np.complex128)
                for a, d in taps:
                    ph = 2 * math.pi * rng.random()
                    H += a * np.exp(-2j * math.pi * f_hz * d + 1j * ph)
                H += (rng.gauss(0, 0.02) + 1j * rng.gauss(0, 0.02))
                H *= 10 ** (rng.gauss(0, 0.8) / 20)       # AGC jitter, scale only
                msg = {
                    "header": {"stamp": stamp(ns), "frame_id": "wlan"},
                    "src_mac": mac, "rssi": int(-45 if not nlos else -62),
                    "frame_control": 0x88, "seq": i % 4096,
                    "core": 0, "spatial_stream": 0,
                    "chanspec": 0xe02a, "chip_version": 0x4366,
                    "channel": 149, "bandwidth_mhz": 80,
                    "subcarrier_index": keep.tolist(),
                    "csi_real": H.real.astype(np.float32).tolist(),
                    "csi_imag": H.imag.astype(np.float32).tolist(),
                    "trimmed": True, "raw_slots": RAW,
                }
                log = ns + 800_000
                w.write_message(topic, csi, msg, log_time=log, publish_time=log)

        w.finish()
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic.mcap"))

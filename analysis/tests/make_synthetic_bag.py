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

        # NTP clients: infra_1 (steady, ~0.3 ms) and mobile_2 (drifts, one step at t=60 s)
        clients = {
            "/infra_1/ntp/status": ("infra_1", 8.0, lambda t: 0.0003 + 0.0001 * math.sin(t / 20)),
            "/mobile_2/ntp/status": ("mobile_2", 8.7, lambda t: (0.004 - 0.00002 * t) if t < 60 else 0.0008),
        }
        for topic, (host, hz, off) in clients.items():
            n = int(dur_s * hz)
            prev = None
            for i in range(n):
                t = i / hz
                ns = t0 + int(t * 1e9)
                o = off(t) + rng.gauss(0, 0.00005)
                stepped = prev is not None and abs(o - prev) > 0.002
                msg = {
                    "header": {"stamp": stamp(ns), "frame_id": host},
                    "role": "client",
                    "hostname": host,
                    "synchronized": True,
                    "sync_source": "mobile_1",
                    "stratum_level": "3",
                    "stratum": 3,
                    "offset_seconds": o,
                    "delay_seconds": 0.0025 + abs(rng.gauss(0, 0.0005)),
                    "jitter_seconds": 0.0002 + abs(rng.gauss(0, 0.00005)),
                    "root_delay": 0.01,
                    "root_dispersion": 0.02,
                    "frequency_error_ppm": 12.3,
                    "poll_interval_seconds": 16,
                    "reach_register": 0o377,
                    "reachability_percent": 100,
                    "reference_time": stamp(ns - 5_000_000_000),
                    "connected_clients": -1,
                    "leap_indicator": "none",
                    "warnings": [] if not stepped else ["clock stepped"],
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

        for k in range(10):
            ns = t0 + int(k * 15.6e9)
            w.write_message("/infra_1/ntp/events", ev, {"data": f"event {k}: reselected server mobile_1"}, log_time=ns, publish_time=ns)

        # header-bearing sensor topics on each node for the stamp audit
        for topic, host, hz, off in [
            ("/mobile_1/ouster/imu", "mobile_1", 100.0, 0.0),
            ("/mobile_2/imu", "mobile_2", 100.0, 0.002),
            ("/infra_1/imu_fake", "infra_1", 20.0, 0.0003),
        ]:
            n = int(dur_s * hz)
            for i in range(n):
                t = i / hz
                ns = t0 + int(t * 1e9)
                msg = {
                    "header": {"stamp": stamp(ns), "frame_id": host},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    "orientation_covariance": [0.0] * 9,
                    "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "angular_velocity_covariance": [0.0] * 9,
                    "linear_acceleration": {"x": 0.0, "y": 0.0, "z": 9.81},
                    "linear_acceleration_covariance": [0.0] * 9,
                }
                log = ns + 2_000_000 + int(off * 1e9) + rng.randint(0, 1_000_000)
                w.write_message(topic, imu, msg, log_time=log, publish_time=log - 500_000)
        w.finish()
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic.mcap"))

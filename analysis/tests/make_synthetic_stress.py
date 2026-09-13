#!/usr/bin/env python3
"""A synthetic NTP stress run, to test ntp_stress.py before a real one exists.

    python3 tests/make_synthetic_stress.py out_dir

Writes out_dir/stress.mcap with the v2 NtpStatus layout (last_offset_seconds,
skew_ppm, fit_samples added, reference_time = chrony's Ref time) for two
clients, and beside it the CSVs that tools/ntp_side_log.sh would produce,
one per agent per phase, with the SoC temperature.

The physics being imitated: the crystal's frequency follows temperature at
about -0.3 ppm per degree; while the temperature moves, chrony's fit lags
and its skew rises; the measured offset at each 8 s poll is the wander since
the last poll (skew x 8 s) plus per-sample Wi-Fi noise; the steered System
time slews toward each new measurement. mobile_2 is on Wi-Fi (noisy samples),
infra_1 on Ethernet (clean). Phases: thermal (0-900 s: heat 45->72 C then hold)
and cool (900-1500 s).
"""
from __future__ import annotations

import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from mcap_ros2.writer import Writer

HEADER_DEF = """
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
"""

# The field list is copied verbatim from the DEPLOYED comms_msgs/msg/NtpStatus.msg
# (final_comms branch). A fixture that does not match the real message tests
# the fixture, not the pipeline.
NTP_STATUS_V2 = """std_msgs/Header header
string role
string hostname
bool synchronized
string sync_source
string stratum_level
int32  stratum
float64 offset_seconds
float64 last_offset_seconds
float64 delay_seconds
float64 jitter_seconds
float64 root_delay
float64 root_dispersion
float64 frequency_error_ppm
float64 skew_ppm
int32   fit_samples
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
float64 rms_offset_seconds       # tracking "RMS offset": typical recent measured offset
float64 residual_freq_ppm        # tracking "Residual freq": what the last fit could not explain
float64 update_interval_seconds  # tracking "Update interval": the actual poll spacing
float64 fit_span_seconds         # sourcestats "Span": time covered by the fit's samples
float64 temperature_c            # SoC temperature; the crystal's rate follows it (~0.1-1 ppm/degree)
float64 cpu_load_1min            # /proc/loadavg, so a stress phase is visible in the same message
""" + HEADER_DEF


def stamp(ns: int) -> dict:
    return {"sec": ns // 1_000_000_000, "nanosec": ns % 1_000_000_000}


def temperature(t: float) -> float:
    """45 C, heated to 72 C over 0-600 s, held, cooled from 900 s."""
    if t < 600:
        return 45.0 + 27.0 * (1 - math.exp(-t / 200.0))
    if t < 900:
        return 45.0 + 27.0 * (1 - math.exp(-600 / 200.0))
    return 45.0 + 27.0 * (1 - math.exp(-600 / 200.0)) * math.exp(-(t - 900) / 250.0)


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(3)
    t0 = 1_789_270_000_000_000_000                       # ns, UTC epoch
    dur_s, poll_s, hz = 1500.0, 8, 10.0
    agents = {
        # the topics a namespaced client node actually publishes
        "/mobile_2/ntp/client/status": ("mobile_2", -27.2, 120e-6, 0.0063),
        "/infra_1/ntp/client/status":  ("infra_1",  +10.1,  25e-6, 0.0004),
    }
    with open(out_dir / "stress.mcap", "wb") as f:
        w = Writer(f)
        ntp = w.register_msgdef("comms_msgs/msg/NtpStatus", NTP_STATUS_V2)
        for topic, (agent, f0, sigma, delay) in agents.items():
            sys_off, last_off, prev_off = 0.0, 0.0, 0.0
            fit_freq = f0                                  # chrony's estimate lags the truth
            ref_ns = t0
            rows = []
            n = int(dur_s * hz)
            for i in range(n):
                t = i / hz
                T = temperature(t)
                true_freq = f0 - 0.3 * (T - 45.0)          # ppm, follows temperature
                if i % int(poll_s * hz) == 0:              # a poll lands
                    # wander since last poll = uncorrected residual x interval, plus sample noise
                    resid = true_freq - fit_freq
                    last_off = resid * 1e-6 * poll_s + rng.gauss(0, sigma)
                    fit_freq += 0.35 * resid               # the fit catches up over a few polls
                    ref_ns = t0 + int(t * 1e9)
                    prev_off = sys_off
                skew = 0.2 + abs(true_freq - fit_freq) * 1.5 + (0.3 if agent == "mobile_2" else 0.0)
                sys_off += (last_off - sys_off) * 0.15 / hz * 8   # slews toward the measurement
                stepped = t < 1.0 and i == 0
                msg = {
                    "header": {"stamp": stamp(t0 + int(t * 1e9)), "frame_id": agent},
                    "role": "client", "hostname": "ubuntu" if agent == "mobile_2" else "wicoms-robot2",
                    "synchronized": True, "sync_source": "192.168.25.25",
                    "stratum_level": "4", "stratum": 4,
                    "offset_seconds": sys_off, "last_offset_seconds": last_off,
                    "delay_seconds": delay, "jitter_seconds": sigma,
                    "root_delay": 0.0125 + delay, "root_dispersion": 0.0009,
                    "frequency_error_ppm": fit_freq, "skew_ppm": skew,
                    "fit_samples": min(64, 4 + int(t / poll_s)),
                    "poll_interval_seconds": poll_s, "reach_register": 0o377, "reachability_percent": 100,
                    "reference_time": stamp(ref_ns), "connected_clients": -1,
                    "leap_indicator": "no_warning",
                    "warnings": ["delay_seconds is an upper bound (2x sources error term): "
                                 "chronyc ntpdata needs socket access"],
                    "seq": i, "monotonic_seconds": t, "clock_stepped": stepped,
                    "offset_delta_seconds": sys_off - prev_off,
                    "rms_offset_seconds": abs(last_off), "residual_freq_ppm": true_freq - fit_freq,
                    "update_interval_seconds": float(poll_s),
                    "fit_span_seconds": float(min(64, 4 + int(t / poll_s)) * poll_s),
                    # carried IN the message now, so a stress run needs no side log
                    "temperature_c": T,
                    "cpu_load_1min": 3.8 if t < 600 else 0.4,
                }
                log = t0 + int(t * 1e9) + 2_000_000
                w.write_message(topic, ntp, msg, log_time=log, publish_time=log - 500_000)
                if i % int(poll_s * hz) == 0:
                    rows.append((t, T, sys_off, last_off, fit_freq, skew, delay, poll_s, sigma))
            # the side logs, one per phase
            for phase, lo, hi in (("thermal", 0, 900), ("cool", 900, 1500)):
                p = out_dir / f"ntp_side_{agent}_{phase}_synthetic.csv"
                with open(p, "w") as fh:
                    fh.write("t_utc,agent,phase,temp_c,load1,system_offset_s,last_offset_s,rms_offset_s,"
                             "freq_ppm,resid_ppm,skew_ppm,root_delay_s,interval_s,np,span_s,stddev_s\n")
                    for (t, T, so, lo_, fq, sk, dl, ps, sg) in rows:
                        if not (lo <= t < hi):
                            continue
                        iso = datetime.fromtimestamp(t0 / 1e9 + t, tz=timezone.utc).isoformat()
                        load = 3.8 if (phase == "thermal" and t < 600) else 0.4
                        fh.write(f"{iso},{agent},{phase},{T:.2f},{load},{so:.9f},{lo_:.9f},{sg:.9f},"
                                 f"{fq:.3f},{0.0:.3f},{sk:.3f},{0.0125 + dl:.6f},{ps:.1f},"
                                 f"{min(64, 4 + int(t / ps))},{min(64, 4 + int(t / ps)) * ps},{sg:.9f}\n")
        w.finish()
    print(f"wrote {out_dir}/stress.mcap and the side logs")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_stress"))

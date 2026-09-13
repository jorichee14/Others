#!/usr/bin/env python3
"""Run every command the NTP node's parser depends on, as the current user,
and show what came back and what the parser made of it.

    python3 src/comms/scripts/check_ntp_commands.py

Run it on each robot as the user that launches the node. It needs the
workspace sourced (for comms.ntp) but not ROS running.
"""
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from comms.ntp.ntp_parser import get_ntp_metrics, get_clock_environment, _find_cpu_thermal_zone  # noqa: E402

COMMANDS = [
    (["chronyc", "tracking"],        "must print 13 lines"),
    (["chronyc", "sources", "-v"],   "the '^*' line is the selected server"),
    (["chronyc", "sourcestats"],     "NP, Span, Freq Skew, Std Dev"),
    (["chronyc", "ntpdata"],         "needs socket access; empty or an error means delay falls back to a bound"),
    (["chronyc", "clients"],         "server only; needs socket access"),
]

ok = True
print(f"user: {os.environ.get('USER', '?')}   chronyc: {shutil.which('chronyc') or 'NOT FOUND'}\n")
for cmd, note in COMMANDS:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        out = (r.stdout or r.stderr).strip()
        first = out.splitlines()[0] if out else "(no output)"
        status = "ok" if r.returncode == 0 and r.stdout.strip() else "EMPTY/ERROR"
    except FileNotFoundError:
        first, status = "(chronyc not installed)", "MISSING"
    except subprocess.TimeoutExpired:
        first, status = "(timed out)", "TIMEOUT"
    if status != "ok" and cmd[1] not in ("ntpdata", "clients"):
        ok = False
    print(f"  {' '.join(cmd):24s} {status:12s} {first[:70]}    # {note}")

zone = _find_cpu_thermal_zone()
print(f"\n  thermal zone: {zone}  exists: {os.path.exists(zone)}")

print("\nparsed:")
m = get_ntp_metrics(); m.update(get_clock_environment())
for k in ("sync_source", "stratum", "offset_seconds", "last_offset_seconds", "delay_seconds",
          "jitter_seconds", "frequency_error_ppm", "residual_freq_ppm", "skew_ppm",
          "poll_interval_seconds", "update_interval_seconds", "fit_samples", "fit_span_seconds",
          "reach_register", "temperature_c", "cpu_load_1min", "warnings"):
    print(f"  {k:26s} {m.get(k)}")

print("\nverdict:")
if m.get("backend") != "chronyc":
    print("  FAIL  chrony is not answering; the node will publish an unsynchronised message"); ok = False
if m.get("poll_interval_seconds") != 8:
    print(f"  FAIL  poll interval is {m.get('poll_interval_seconds')} s, not 8: chrony.conf change not applied"); ok = False
if m.get("delay_seconds", 0) == 0:
    print("  WARN  delay is 0: neither ntpdata nor the sources error term was readable")
elif any("upper bound" in w for w in m.get("warnings", [])):
    print("  note  delay is an upper bound (no socket access to ntpdata); fine, or add the user to _chrony")
if m.get("temperature_c") != m.get("temperature_c"):
    print("  WARN  temperature unreadable; check /sys/class/thermal/thermal_zone*/type")
if m.get("skew_ppm", 99) > 1.0:
    print(f"  WARN  skew {m.get('skew_ppm')} ppm: still warming up, wait before recording")
print("  PASS" if ok else "  see FAIL lines above")

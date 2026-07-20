#!/usr/bin/env python3
"""
wifi_signal_logger.py — record WiFi signal metrics (SNR, RSSI/"RSRP", link
quality, bitrate) on Ubuntu 22.04 / NVIDIA Jetson.

Note on RSRP: RSRP is a *cellular* (LTE/5G) metric and does not exist for WiFi.
The WiFi equivalent of "received signal power" is the RSSI / signal level in dBm,
which is what this tool records in the `signal_dbm` column. SNR is derived as
`signal_dbm - noise_dbm`, with the noise floor taken from the radio's survey data.

Data sources (all standard on Ubuntu 22.04, no Python dependencies):
  * `iw dev <iface> link`          -> SSID, BSSID, freq, signal, tx bitrate
  * `iw dev <iface> station dump`  -> signal, signal avg, tx/rx bitrate
  * `iw dev <iface> survey dump`   -> noise floor of the in-use channel (for SNR)
  * /proc/net/wireless             -> link quality, level, noise (fallback)
  * `iwconfig <iface>`             -> last-resort fallback if `iw` is missing

Only the Linux wireless stack is used; nothing here is Jetson-specific, but it is
tested against the Jetson's stock tooling.

Usage examples
--------------
  # Auto-detect the interface, sample every second, print + append to CSV:
  sudo python3 wifi_signal_logger.py

  # Explicit interface, 0.5 s interval, stop after 60 s, custom file:
  sudo python3 wifi_signal_logger.py -i wlan0 -t 0.5 -d 60 -o run1.csv

  # Quiet (CSV only, no console table):
  sudo python3 wifi_signal_logger.py --quiet

`sudo` (or CAP_NET_ADMIN) is recommended: some drivers only report `signal` /
survey `noise` to privileged callers. Without it the tool still runs and records
whatever is exposed, leaving unavailable fields blank.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

# Fields written to the CSV, in order.
CSV_FIELDS = [
    "timestamp_iso",
    "timestamp_unix",
    "iface",
    "ssid",
    "bssid",
    "freq_mhz",
    "channel",
    "signal_dbm",       # RSSI — the WiFi "RSRP" equivalent
    "signal_avg_dbm",
    "noise_dbm",
    "snr_db",           # signal_dbm - noise_dbm
    "link_quality",     # from /proc/net/wireless (e.g. "70/70")
    "tx_bitrate_mbps",
    "rx_bitrate_mbps",
]


def run(cmd: list[str], timeout: float = 5.0) -> str:
    """Run a command, returning stdout ('' on any failure)."""
    try:
        out = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        return out.stdout.decode("utf-8", "replace")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def freq_to_channel(freq_mhz):
    """Convert a WiFi centre frequency (MHz) to a channel number."""
    if freq_mhz is None:
        return None
    f = int(freq_mhz)
    if f == 2484:
        return 14
    if 2412 <= f <= 2472:
        return (f - 2412) // 5 + 1
    if 5000 <= f <= 5900:
        return (f - 5000) // 5
    if 5955 <= f <= 7115:  # 6 GHz / WiFi 6E
        return (f - 5955) // 5 + 1
    return None


def detect_interfaces() -> list[str]:
    """Return wireless interface names, best-effort."""
    ifaces = []
    if have("iw"):
        out = run(["iw", "dev"])
        ifaces = re.findall(r"Interface\s+(\S+)", out)
    if not ifaces:
        # Fallback: sysfs — any netdev with a wireless/ dir is a WiFi iface.
        try:
            for name in sorted(os.listdir("/sys/class/net")):
                if os.path.isdir(os.path.join("/sys/class/net", name, "wireless")):
                    ifaces.append(name)
        except OSError:
            pass
    return ifaces


def parse_bitrate(text: str):
    """Pull the leading number out of e.g. '866.7 MBit/s'."""
    m = re.search(r"([\d.]+)\s*MBit/s", text)
    return float(m.group(1)) if m else None


def sample_iw_link(iface: str) -> dict:
    out = run(["iw", "dev", iface, "link"])
    d = {}
    if not out or "Not connected" in out:
        return d
    m = re.search(r"Connected to ([0-9a-fA-F:]{17})", out)
    if m:
        d["bssid"] = m.group(1)
    m = re.search(r"SSID:\s*(.+)", out)
    if m:
        d["ssid"] = m.group(1).strip()
    m = re.search(r"freq:\s*(\d+)", out)
    if m:
        d["freq_mhz"] = int(m.group(1))
    m = re.search(r"signal:\s*(-?\d+)\s*dBm", out)
    if m:
        d["signal_dbm"] = int(m.group(1))
    m = re.search(r"tx bitrate:\s*([\d.]+\s*MBit/s)", out)
    if m:
        d["tx_bitrate_mbps"] = parse_bitrate(m.group(1))
    m = re.search(r"rx bitrate:\s*([\d.]+\s*MBit/s)", out)
    if m:
        d["rx_bitrate_mbps"] = parse_bitrate(m.group(1))
    return d


def sample_iw_station(iface: str) -> dict:
    out = run(["iw", "dev", iface, "station", "dump"])
    d = {}
    if not out:
        return d
    # 'signal:  -45 [-46, -50] dBm' -> take the first (combined) value.
    m = re.search(r"signal:\s*(-?\d+)", out)
    if m:
        d["signal_dbm"] = int(m.group(1))
    m = re.search(r"signal avg:\s*(-?\d+)", out)
    if m:
        d["signal_avg_dbm"] = int(m.group(1))
    m = re.search(r"tx bitrate:\s*([\d.]+\s*MBit/s)", out)
    if m:
        d["tx_bitrate_mbps"] = parse_bitrate(m.group(1))
    m = re.search(r"rx bitrate:\s*([\d.]+\s*MBit/s)", out)
    if m:
        d["rx_bitrate_mbps"] = parse_bitrate(m.group(1))
    return d


def sample_survey_noise(iface: str):
    """Noise floor (dBm) of the in-use channel from `iw survey dump`."""
    out = run(["iw", "dev", iface, "survey", "dump"])
    if not out:
        return None
    # Split into per-frequency blocks and find the one marked '[in use]'.
    for block in re.split(r"Survey data from", out):
        if "[in use]" in block:
            m = re.search(r"noise:\s*(-?\d+)\s*dBm", block)
            if m:
                return int(m.group(1))
    return None


def sample_proc_wireless(iface: str) -> dict:
    """Fallback metrics from /proc/net/wireless."""
    d = {}
    try:
        with open("/proc/net/wireless", "r") as fh:
            lines = fh.readlines()
    except OSError:
        return d
    for line in lines:
        if not line.strip().startswith(iface + ":"):
            continue
        # face: status  link  level  noise ...
        parts = line.replace(iface + ":", " ").split()
        # parts: [status, link, level, noise, ...] with trailing '.' on values.
        try:
            link = parts[1].rstrip(".")
            level = int(float(parts[2].rstrip(".")))
            noise = int(float(parts[3].rstrip(".")))
        except (IndexError, ValueError):
            return d
        d["link_quality"] = f"{link}/70"
        # level/noise are dBm when large-negative; ignore the -256 "invalid" sentinel.
        if level < 0:
            d.setdefault("signal_dbm", level)
        if noise < 0 and noise != -256:
            d["noise_dbm"] = noise
    return d


def sample_iwconfig(iface: str) -> dict:
    """Last-resort fallback via legacy wireless-tools."""
    out = run(["iwconfig", iface])
    d = {}
    if not out:
        return d
    m = re.search(r'ESSID:"([^"]*)"', out)
    if m:
        d["ssid"] = m.group(1)
    m = re.search(r"Access Point:\s*([0-9a-fA-F:]{17})", out)
    if m:
        d["bssid"] = m.group(1)
    m = re.search(r"Signal level=(-?\d+)\s*dBm", out)
    if m:
        d["signal_dbm"] = int(m.group(1))
    m = re.search(r"Noise level=(-?\d+)\s*dBm", out)
    if m:
        d["noise_dbm"] = int(m.group(1))
    m = re.search(r"Link Quality[=:]\s*(\d+/\d+)", out)
    if m:
        d["link_quality"] = m.group(1)
    m = re.search(r"Bit Rate[=:]\s*([\d.]+)\s*Mb/s", out)
    if m:
        d["tx_bitrate_mbps"] = float(m.group(1))
    return d


def collect(iface: str, use_iw: bool) -> dict:
    """Gather one sample, merging all available sources."""
    now = datetime.now(timezone.utc).astimezone()
    row = {k: "" for k in CSV_FIELDS}
    row["timestamp_iso"] = now.isoformat(timespec="milliseconds")
    row["timestamp_unix"] = f"{now.timestamp():.3f}"
    row["iface"] = iface

    data: dict = {}
    if use_iw:
        # station dump first (richest), then link fills any gaps.
        data.update(sample_iw_station(iface))
        for k, v in sample_iw_link(iface).items():
            data.setdefault(k, v)
        noise = sample_survey_noise(iface)
        if noise is not None:
            data["noise_dbm"] = noise

    # /proc always cheap; fills link_quality and noise if survey lacked it.
    for k, v in sample_proc_wireless(iface).items():
        data.setdefault(k, v)

    if "signal_dbm" not in data and not use_iw:
        for k, v in sample_iwconfig(iface).items():
            data.setdefault(k, v)

    # Derived fields.
    if "freq_mhz" in data:
        ch = freq_to_channel(data["freq_mhz"])
        if ch is not None:
            data["channel"] = ch
    if "signal_dbm" in data and "noise_dbm" in data:
        data["snr_db"] = data["signal_dbm"] - data["noise_dbm"]

    for k, v in data.items():
        if k in row and v is not None:
            row[k] = v
    return row


def fmt(v) -> str:
    return "--" if v == "" or v is None else str(v)


def print_row(row: dict, counter: int) -> None:
    if counter % 20 == 0:
        print(
            f"\n{'time':<12} {'ssid':<16} {'ch':>3} "
            f"{'sig':>5} {'noise':>6} {'snr':>5} {'tx':>7} {'rx':>7}",
            flush=True,
        )
    t = row["timestamp_iso"][11:23]
    ssid = fmt(row["ssid"])[:16]
    print(
        f"{t:<12} {ssid:<16} {fmt(row['channel']):>3} "
        f"{fmt(row['signal_dbm']):>5} {fmt(row['noise_dbm']):>6} "
        f"{fmt(row['snr_db']):>5} {fmt(row['tx_bitrate_mbps']):>7} "
        f"{fmt(row['rx_bitrate_mbps']):>7}",
        flush=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Record WiFi SNR / RSSI ('RSRP') / link metrics to CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--iface", help="wireless interface (auto-detected if omitted)")
    p.add_argument("-t", "--interval", type=float, default=1.0, help="seconds between samples")
    p.add_argument("-d", "--duration", type=float, default=0.0, help="stop after N seconds (0 = run until Ctrl+C)")
    p.add_argument("-o", "--output", default="wifi_signal_log.csv", help="CSV output path")
    p.add_argument("--overwrite", action="store_true", help="overwrite the CSV instead of appending")
    p.add_argument("--quiet", action="store_true", help="do not print the live table")
    args = p.parse_args()

    use_iw = have("iw")
    if not use_iw and not have("iwconfig"):
        print("error: neither `iw` nor `iwconfig` found. Install with: "
              "sudo apt install iw", file=sys.stderr)
        return 1

    iface = args.iface
    if not iface:
        found = detect_interfaces()
        if not found:
            print("error: no wireless interface found. Specify one with -i.",
                  file=sys.stderr)
            return 1
        iface = found[0]
        if len(found) > 1 and not args.quiet:
            print(f"note: multiple interfaces {found}; using '{iface}' "
                  f"(override with -i)", file=sys.stderr)

    if os.geteuid() != 0 and not args.quiet:
        print("note: not running as root; some drivers hide signal/noise "
              "from unprivileged callers.", file=sys.stderr)

    new_file = args.overwrite or not os.path.exists(args.output)
    mode = "w" if args.overwrite else "a"
    csv_fh = open(args.output, mode, newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()
        csv_fh.flush()

    if not args.quiet:
        print(f"logging '{iface}' every {args.interval}s -> {args.output} "
              f"({'until Ctrl+C' if args.duration <= 0 else f'{args.duration}s'})",
              flush=True)

    stop = {"flag": False}

    def _handler(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    start = time.monotonic()
    counter = 0
    try:
        while not stop["flag"]:
            tick = time.monotonic()
            row = collect(iface, use_iw)
            writer.writerow(row)
            csv_fh.flush()
            if not args.quiet:
                print_row(row, counter)
            counter += 1

            if args.duration > 0 and (time.monotonic() - start) >= args.duration:
                break

            # Sleep the remainder of the interval, staying responsive to Ctrl+C.
            elapsed = time.monotonic() - tick
            remaining = args.interval - elapsed
            while remaining > 0 and not stop["flag"]:
                time.sleep(min(0.1, remaining))
                remaining = args.interval - (time.monotonic() - tick)
    finally:
        csv_fh.close()

    if not args.quiet:
        print(f"\nwrote {counter} samples to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Pure parsing helpers for wireless interface data.

These functions have no ROS dependency so they can be unit-tested in
isolation. Each collector is defensive: a missing tool, a missing sysfs
file, or an unexpected line simply yields ``None``/absent keys rather than
raising, so the node keeps publishing even when the driver is stingy with
data.

Data sources
------------
* ``/proc/net/wireless``                     -> link quality, signal, noise
* ``iw dev <iface> link`` (preferred)        -> essid, bssid, freq, bitrate, signal
* ``iwconfig <iface>`` (legacy fallback)     -> same, plus wireless error counters
* ``/sys/class/net/<iface>/statistics/*``    -> RX/TX packet & error counters
* ``/sys/class/net/<iface>/{address,...}``   -> MAC address and link state
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from typing import Dict, Optional

SYS_NET = "/sys/class/net"
PROC_WIRELESS = "/proc/net/wireless"

# Interface flag bits from <linux/if.h>
IFF_UP = 0x1
IFF_RUNNING = 0x40


def _run(cmd: list) -> Optional[str]:
    """Run a command, returning stdout text or ``None`` on any failure."""
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe] + cmd[1:],
            capture_output=True,
            text=True,
            timeout=4.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return out.stdout or None
    return out.stdout


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path, "r") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def _read_str(path: str) -> Optional[str]:
    try:
        with open(path, "r") as fh:
            return fh.read().strip()
    except OSError:
        return None


def channel_from_freq_ghz(freq_ghz: Optional[float]) -> int:
    """Best-effort 802.11 channel number from a centre frequency in GHz."""
    if freq_ghz is None or math.isnan(freq_ghz):
        return -1
    mhz = freq_ghz * 1000.0
    if 2412.0 <= mhz <= 2472.0:
        return int(round((mhz - 2412.0) / 5.0)) + 1
    if abs(mhz - 2484.0) < 1.0:  # channel 14
        return 14
    if 5000.0 <= mhz <= 5900.0:
        return int(round((mhz - 5000.0) / 5.0))
    if 5955.0 <= mhz <= 7115.0:  # 6 GHz band
        return int(round((mhz - 5955.0) / 5.0)) + 1
    return -1


# --------------------------------------------------------------------------
# /sys/class/net
# --------------------------------------------------------------------------
def collect_sysfs(iface: str) -> Dict[str, object]:
    """MAC address, link state and traffic statistics from sysfs."""
    base = os.path.join(SYS_NET, iface)
    stats = os.path.join(base, "statistics")
    data: Dict[str, object] = {}

    mac = _read_str(os.path.join(base, "address"))
    if mac:
        data["mac_address"] = mac

    flags = _read_int(os.path.join(base, "flags"))
    if flags is not None:
        data["up"] = bool(flags & IFF_UP)
        data["running"] = bool(flags & IFF_RUNNING)

    # sysfs stat name -> message field name
    stat_map = {
        "rx_packets": "rx_packets",
        "rx_bytes": "rx_bytes",
        "rx_errors": "rx_errors",
        "rx_dropped": "rx_dropped",
        "rx_fifo_errors": "rx_overruns",
        "rx_frame_errors": "rx_frame_errors",
        "tx_packets": "tx_packets",
        "tx_bytes": "tx_bytes",
        "tx_errors": "tx_errors",
        "tx_dropped": "tx_dropped",
        "tx_fifo_errors": "tx_overruns",
        "tx_carrier_errors": "tx_carrier_errors",
        "collisions": "collisions",
    }
    for sys_name, field in stat_map.items():
        val = _read_int(os.path.join(stats, sys_name))
        if val is not None:
            data[field] = val
    return data


# --------------------------------------------------------------------------
# /proc/net/wireless
# --------------------------------------------------------------------------
def collect_proc_wireless(iface: str) -> Dict[str, object]:
    """Link quality, signal level and noise floor from /proc/net/wireless.

    Line format (values may carry a trailing '.')::

        wlx...: 0000   70.  -37.  -256        0      0      0 ...
                status link  level noise
    """
    data: Dict[str, object] = {}
    try:
        with open(PROC_WIRELESS, "r") as fh:
            lines = fh.readlines()
    except OSError:
        return data

    for line in lines:
        if not line.strip().startswith(iface + ":"):
            continue
        rest = line.split(":", 1)[1]
        parts = rest.split()
        if len(parts) < 4:
            break
        try:
            link = float(parts[1].rstrip("."))
            level = float(parts[2].rstrip("."))
            noise = float(parts[3].rstrip("."))
        except ValueError:
            break
        data["link_quality"] = int(link)
        data["signal_dbm"] = level
        # Drivers that cannot measure noise report a sentinel (commonly
        # -256 or 0). Treat those as "no data".
        if noise > -254.0 and noise != 0.0:
            data["noise_dbm"] = noise
            data["noise_valid"] = True
        break
    return data


# --------------------------------------------------------------------------
# iw dev <iface> link  (preferred, modern nl80211 tool)
# --------------------------------------------------------------------------
_IW_ESSID = re.compile(r"SSID:\s*(.+)")
_IW_BSSID = re.compile(r"Connected to\s+([0-9a-fA-F:]{17})")
_IW_FREQ = re.compile(r"freq:\s*(\d+)")
_IW_SIGNAL = re.compile(r"signal:\s*(-?\d+)\s*dBm")
_IW_RXBR = re.compile(r"rx bitrate:\s*([\d.]+)\s*MBit/s")
_IW_TXBR = re.compile(r"tx bitrate:\s*([\d.]+)\s*MBit/s")


def collect_iw_link(iface: str) -> Dict[str, object]:
    out = _run(["iw", "dev", iface, "link"])
    data: Dict[str, object] = {}
    if not out:
        return data
    if "Not connected" in out:
        data["associated"] = False
        return data

    data["associated"] = True
    m = _IW_BSSID.search(out)
    if m:
        data["bssid"] = m.group(1).upper()
    m = _IW_ESSID.search(out)
    if m:
        data["essid"] = m.group(1).strip()
    m = _IW_FREQ.search(out)
    if m:
        data["frequency_ghz"] = int(m.group(1)) / 1000.0
    m = _IW_SIGNAL.search(out)
    if m:
        data["signal_dbm"] = float(m.group(1))
    # Prefer the tx bitrate as the reported PHY rate; fall back to rx.
    m = _IW_TXBR.search(out) or _IW_RXBR.search(out)
    if m:
        data["bit_rate_mbps"] = float(m.group(1))
    return data


# --------------------------------------------------------------------------
# iwconfig <iface>  (legacy fallback, but the source of the error counters)
# --------------------------------------------------------------------------
_WC_ESSID = re.compile(r'ESSID:"([^"]*)"')
_WC_MODE = re.compile(r"Mode:(\S+)")
_WC_FREQ = re.compile(r"Frequency[:=]\s*([\d.]+)\s*GHz")
_WC_AP = re.compile(r"Access Point:\s*([0-9a-fA-F:]{17}|Not-Associated)")
_WC_BITRATE = re.compile(r"Bit Rate[:=]\s*([\d.]+)\s*Mb/s")
_WC_TXPOWER = re.compile(r"Tx-Power[:=]\s*(-?\d+)\s*dBm")
_WC_QUALITY = re.compile(r"Link Quality[:=]\s*(\d+)/(\d+)")
_WC_LEVEL = re.compile(r"Signal level[:=]\s*(-?\d+)\s*dBm")
_WC_NWID = re.compile(r"Rx invalid nwid[:=]\s*(\d+)")
_WC_CRYPT = re.compile(r"Rx invalid crypt[:=]\s*(\d+)")
_WC_FRAG = re.compile(r"Rx invalid frag[:=]\s*(\d+)")
_WC_RETRIES = re.compile(r"Tx excessive retries[:=]\s*(\d+)")
_WC_MISC = re.compile(r"Invalid misc[:=]\s*(\d+)")
_WC_BEACON = re.compile(r"Missed beacon[:=]\s*(\d+)")


def collect_iwconfig(iface: str) -> Dict[str, object]:
    out = _run(["iwconfig", iface])
    data: Dict[str, object] = {}
    if not out:
        return data

    m = _WC_ESSID.search(out)
    if m:
        data["essid"] = m.group(1)
    m = _WC_MODE.search(out)
    if m:
        data["mode"] = m.group(1)
    m = _WC_FREQ.search(out)
    if m:
        data["frequency_ghz"] = float(m.group(1))
    m = _WC_AP.search(out)
    if m:
        ap = m.group(1)
        if ap.lower() == "not-associated":
            data["associated"] = False
        else:
            data["bssid"] = ap.upper()
            data["associated"] = True
    m = _WC_BITRATE.search(out)
    if m:
        data["bit_rate_mbps"] = float(m.group(1))
    m = _WC_TXPOWER.search(out)
    if m:
        data["tx_power_dbm"] = float(m.group(1))
    m = _WC_QUALITY.search(out)
    if m:
        data["link_quality"] = int(m.group(1))
        data["link_quality_max"] = int(m.group(2))
    m = _WC_LEVEL.search(out)
    if m:
        data["signal_dbm"] = float(m.group(1))

    for key, rx in (
        ("rx_invalid_nwid", _WC_NWID),
        ("rx_invalid_crypt", _WC_CRYPT),
        ("rx_invalid_frag", _WC_FRAG),
        ("tx_excessive_retries", _WC_RETRIES),
        ("invalid_misc", _WC_MISC),
        ("missed_beacon", _WC_BEACON),
    ):
        m = rx.search(out)
        if m:
            data[key] = int(m.group(1))
    return data


def list_wireless_interfaces() -> list:
    """Return interface names that expose a ``wireless`` sysfs node."""
    found = []
    try:
        for name in sorted(os.listdir(SYS_NET)):
            if os.path.isdir(os.path.join(SYS_NET, name, "wireless")):
                found.append(name)
    except OSError:
        pass
    return found


def collect_all(iface: str) -> Dict[str, object]:
    """Merge every source into a single dict.

    Precedence (later overrides earlier for overlapping keys):
    sysfs -> /proc/net/wireless -> iwconfig -> iw. ``iw`` wins for the
    association fields because nl80211 is the most reliable source; the
    wireless error counters only come from ``iwconfig``; noise only comes
    from ``/proc/net/wireless``.
    """
    merged: Dict[str, object] = {}
    merged.update(collect_sysfs(iface))
    merged.update(collect_proc_wireless(iface))
    merged.update(collect_iwconfig(iface))
    merged.update(collect_iw_link(iface))

    # Derived fields.
    lq = merged.get("link_quality")
    lq_max = merged.get("link_quality_max")
    if isinstance(lq, int) and isinstance(lq_max, int) and lq_max > 0:
        merged["link_quality_ratio"] = lq / float(lq_max)

    sig = merged.get("signal_dbm")
    noise = merged.get("noise_dbm")
    if (
        merged.get("noise_valid")
        and isinstance(sig, float)
        and isinstance(noise, float)
    ):
        merged["snr_db"] = sig - noise

    if "frequency_ghz" in merged:
        merged["channel"] = channel_from_freq_ghz(
            float(merged["frequency_ghz"])
        )
    return merged

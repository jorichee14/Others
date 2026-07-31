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
# Shared bitrate-line parser
# --------------------------------------------------------------------------
# Handles all of:
#   240.0 MBit/s VHT-MCS 5 40MHz short GI VHT-NSS 2
#   162.0 MBit/s VHT-MCS 4 40MHz VHT-NSS 2
#   130.0 MBit/s MCS 15 40MHz short GI          (legacy HT, NSS implied)
#   1201.0 MBit/s HE-MCS 11 80MHz HE-NSS 2 HE-GI 0
_BR_MBPS = re.compile(r"([\d.]+)\s*MBit/s")
_BR_MCS = re.compile(r"(VHT|HE|HT)-MCS\s*(\d+)")
_BR_MCS_PLAIN = re.compile(r"(?<![-\w])MCS\s*(\d+)")  # legacy HT "MCS 15"
_BR_NSS = re.compile(r"(?:VHT|HE)-NSS\s*(\d+)")
_BR_WIDTH = re.compile(r"(\d+)\s*MHz")


def _parse_bitrate(line: str) -> Dict[str, object]:
    """Extract rate/MCS/NSS/width/GI/mode from an `iw` bitrate line."""
    d: Dict[str, object] = {}
    m = _BR_MBPS.search(line)
    if m:
        d["mbps"] = float(m.group(1))
    m = _BR_MCS.search(line)
    if m:
        d["phy_mode"] = m.group(1)
        d["mcs"] = int(m.group(2))
    else:
        m = _BR_MCS_PLAIN.search(line)
        if m:
            d["phy_mode"] = "HT"
            d["mcs"] = int(m.group(1))
    m = _BR_NSS.search(line)
    if m:
        d["nss"] = int(m.group(1))
    m = _BR_WIDTH.search(line)
    if m:
        d["width"] = int(m.group(1))
    d["short_gi"] = "short GI" in line
    return d


def _apply_bitrate(data: Dict[str, object], line: str, direction: str) -> None:
    """Store a parsed bitrate line under rx_*/tx_* keys."""
    br = _parse_bitrate(line)
    if "mbps" in br:
        data[f"{direction}_bitrate_mbps"] = br["mbps"]
    if "mcs" in br:
        data[f"{direction}_mcs"] = br["mcs"]
    if "nss" in br:
        data[f"{direction}_nss"] = br["nss"]
    if "width" in br:
        data[f"{direction}_width_mhz"] = br["width"]
    if "phy_mode" in br:
        data[f"{direction}_phy_mode"] = br["phy_mode"]
    if direction == "tx":
        data["tx_short_gi"] = bool(br.get("short_gi"))
        if "mbps" in br:
            data["bit_rate_mbps"] = br["mbps"]  # backward-compat field


# --------------------------------------------------------------------------
# iw dev <iface> link  (preferred, modern nl80211 tool)
# --------------------------------------------------------------------------
_IW_ESSID = re.compile(r"SSID:\s*(.+)")
_IW_BSSID = re.compile(r"Connected to\s+([0-9a-fA-F:]{17})")
_IW_FREQ = re.compile(r"freq:\s*(\d+)")
_IW_SIGNAL = re.compile(r"signal:\s*(-?\d+)\s*dBm")
_IW_RXLINE = re.compile(r"rx bitrate:\s*(.+)")
_IW_TXLINE = re.compile(r"tx bitrate:\s*(.+)")
_IW_RXBYTES = re.compile(r"RX:\s*(\d+)\s*bytes\s*\((\d+)\s*packets\)")
_IW_TXBYTES = re.compile(r"TX:\s*(\d+)\s*bytes\s*\((\d+)\s*packets\)")


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
    m = _IW_RXLINE.search(out)
    if m:
        _apply_bitrate(data, m.group(1), "rx")
    m = _IW_TXLINE.search(out)
    if m:
        _apply_bitrate(data, m.group(1), "tx")
    m = _IW_RXBYTES.search(out)
    if m:
        data["sta_rx_bytes"] = int(m.group(1))
        data["sta_rx_packets"] = int(m.group(2))
    m = _IW_TXBYTES.search(out)
    if m:
        data["sta_tx_bytes"] = int(m.group(1))
        data["sta_tx_packets"] = int(m.group(2))
    return data


# --------------------------------------------------------------------------
# iw dev <iface> station dump  (retries, failed, expected throughput, avg)
# --------------------------------------------------------------------------
_ST_RXBYTES = re.compile(r"rx bytes:\s*(\d+)")
_ST_RXPKTS = re.compile(r"rx packets:\s*(\d+)")
_ST_TXBYTES = re.compile(r"tx bytes:\s*(\d+)")
_ST_TXPKTS = re.compile(r"tx packets:\s*(\d+)")
_ST_RETRIES = re.compile(r"tx retries:\s*(\d+)")
_ST_FAILED = re.compile(r"tx failed:\s*(\d+)")
_ST_SIGNAL = re.compile(r"signal:\s*(-?\d+)")
_ST_SIGNAL_AVG = re.compile(r"signal avg:\s*(-?\d+)")
_ST_EXPECTED = re.compile(r"expected throughput:\s*([\d.]+)\s*[MG]bps")
_ST_CONNTIME = re.compile(r"connected time:\s*(\d+)\s*seconds")
_ST_RXLINE = re.compile(r"rx bitrate:\s*(.+)")
_ST_TXLINE = re.compile(r"tx bitrate:\s*(.+)")


def collect_iw_station(iface: str) -> Dict[str, object]:
    """Per-station link reliability. Needs no root on most kernels, but
    returns an empty dict if the driver denies the dump."""
    out = _run(["iw", "dev", iface, "station", "dump"])
    data: Dict[str, object] = {}
    if not out or "Station" not in out:
        return data

    for key, rx, cast in (
        ("sta_rx_bytes", _ST_RXBYTES, int),
        ("sta_rx_packets", _ST_RXPKTS, int),
        ("sta_tx_bytes", _ST_TXBYTES, int),
        ("sta_tx_packets", _ST_TXPKTS, int),
        ("tx_retries", _ST_RETRIES, int),
        ("tx_failed", _ST_FAILED, int),
        ("connected_time_s", _ST_CONNTIME, int),
    ):
        m = rx.search(out)
        if m:
            data[key] = cast(m.group(1))

    m = _ST_SIGNAL_AVG.search(out)
    if m:
        data["signal_avg_dbm"] = float(m.group(1))
    m = _ST_SIGNAL.search(out)
    if m:
        data["signal_dbm"] = float(m.group(1))
    m = _ST_EXPECTED.search(out)
    if m:
        val = float(m.group(1))
        # normalise Gbps -> Mbps if the unit says Gbps
        if "Gbps" in out[m.start():m.end() + 4]:
            val *= 1000.0
        data["expected_mbps"] = val
    m = _ST_RXLINE.search(out)
    if m:
        _apply_bitrate(data, m.group(1), "rx")
    m = _ST_TXLINE.search(out)
    if m:
        _apply_bitrate(data, m.group(1), "tx")
    return data


# --------------------------------------------------------------------------
# iw dev <iface> survey dump  (noise floor + channel busy time)
# --------------------------------------------------------------------------
_SV_INUSE = re.compile(r"frequency:\s*(\d+)\s*MHz\s*\[in use\]")
_SV_NOISE = re.compile(r"noise:\s*(-?\d+)\s*dBm")
_SV_ACTIVE = re.compile(r"channel active time:\s*(\d+)\s*ms")
_SV_BUSY = re.compile(r"channel busy time:\s*(\d+)\s*ms")


def collect_iw_survey(iface: str) -> Dict[str, object]:
    """Noise floor and channel busy/active time for the in-use channel."""
    out = _run(["iw", "dev", iface, "survey", "dump"])
    data: Dict[str, object] = {}
    if not out:
        return data

    # Each channel is a "Survey data from <iface>" block; pick the in-use one.
    for block in re.split(r"Survey data from", out):
        if "[in use]" not in block:
            continue
        m = _SV_NOISE.search(block)
        if m:
            noise = float(m.group(1))
            if noise > -254.0 and noise != 0.0:
                data["noise_dbm"] = noise
                data["noise_valid"] = True
        m = _SV_ACTIVE.search(block)
        if m:
            data["channel_active_ms"] = float(m.group(1))
        m = _SV_BUSY.search(block)
        if m:
            data["channel_busy_ms"] = float(m.group(1))
        break
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
    sysfs -> /proc/net/wireless -> iwconfig -> iw link -> iw station ->
    iw survey. nl80211 (``iw``) wins for association/PHY fields; the
    wireless error counters only come from ``iwconfig``; the noise floor
    comes from ``iw survey`` when available, else ``/proc/net/wireless``;
    retries/failed/expected-throughput only come from ``iw station``.
    """
    merged: Dict[str, object] = {}
    merged.update(collect_sysfs(iface))
    merged.update(collect_proc_wireless(iface))
    merged.update(collect_iwconfig(iface))
    merged.update(collect_iw_link(iface))
    merged.update(collect_iw_station(iface))
    merged.update(collect_iw_survey(iface))

    # Derived fields.
    lq = merged.get("link_quality")
    lq_max = merged.get("link_quality_max")
    if isinstance(lq, int) and isinstance(lq_max, int) and lq_max > 0:
        merged["link_quality_ratio"] = lq / float(lq_max)

    sig = merged.get("signal_dbm")
    noise = merged.get("noise_dbm")
    if (
        merged.get("noise_valid")
        and isinstance(sig, (int, float))
        and isinstance(noise, (int, float))
    ):
        merged["snr_db"] = float(sig) - float(noise)

    active = merged.get("channel_active_ms")
    busy = merged.get("channel_busy_ms")
    if isinstance(active, float) and isinstance(busy, float) and active > 0.0:
        merged["channel_busy_ratio"] = busy / active

    if "frequency_ghz" in merged:
        merged["channel"] = channel_from_freq_ghz(
            float(merged["frequency_ghz"])
        )
    return merged

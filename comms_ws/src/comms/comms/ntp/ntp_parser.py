"""
ntp_parser.py
Parses ntpq and chronyc output into NtpStatus.msg fields.
Supports: ntpq -p / -c rv  and  chronyc tracking / sources / clients
"""

from __future__ import annotations
import re
from datetime import datetime, timezone
import subprocess
from typing import Optional


def _run(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return result.stdout
    except Exception:
        return ""


def _reachability_percent(reach_octal: int) -> int:
    bits = bin(reach_octal).count("1")
    return int(bits / 8 * 100)


LEAP_MAP = {
    "0": "no_warning", "00": "no_warning",
    "1": "add_second", "01": "add_second",
    "2": "del_second", "10": "del_second",
    "3": "alarm",      "11": "alarm",
}


# ntpq parser

def _parse_ntpq() -> Optional[dict]:
    peers_out = _run(["ntpq", "-p", "-n"])
    rv_out    = _run(["ntpq", "-c", "rv"])
    if not peers_out and not rv_out:
        return None

    metrics: dict = {
        "backend": "ntpq", "synchronized": False, "sync_source": "",
        "stratum": 16, "offset_seconds": 0.0, "delay_seconds": 0.0,
        "jitter_seconds": 0.0, "root_delay": 0.0, "root_dispersion": 0.0,
        "frequency_error_ppm": 0.0, "poll_interval_seconds": 0,
        "reach_register": 0, "reachability_percent": 0,
        "leap_indicator": "alarm", "warnings": [],
    }

    for line in peers_out.splitlines():
        if not line.startswith("*"):
            continue
        parts = line[1:].split()
        if len(parts) < 10:
            continue
        try:
            metrics["sync_source"]           = parts[0]
            metrics["stratum"]               = int(parts[2])
            metrics["poll_interval_seconds"] = 2 ** int(parts[3])
            reach_octal                      = int(parts[5], 8)
            metrics["reach_register"]        = reach_octal
            metrics["reachability_percent"]  = _reachability_percent(reach_octal)
            metrics["delay_seconds"]         = float(parts[7]) / 1000.0
            metrics["offset_seconds"]        = float(parts[8]) / 1000.0
            metrics["jitter_seconds"]        = float(parts[9]) / 1000.0
            metrics["synchronized"]          = reach_octal > 0
        except (ValueError, IndexError):
            pass
        break

    rv = {}
    for token in rv_out.replace("\n", ",").split(","):
        token = token.strip()
        if "=" in token:
            k, _, v = token.partition("=")
            rv[k.strip()] = v.strip()

    for rk, mk, div in [("rootdelay","root_delay",1000.0),
                         ("rootdisp","root_dispersion",1000.0)]:
        if rk in rv:
            try: metrics[mk] = float(rv[rk]) / div
            except ValueError: pass
    if "frequency" in rv:
        try: metrics["frequency_error_ppm"] = float(rv["frequency"])
        except ValueError: pass
    if "leap" in rv:
        metrics["leap_indicator"] = LEAP_MAP.get(rv["leap"].strip(), "alarm")
    return metrics


# chronyc parser

def _parse_secs(s: str) -> float:
    """'0.000123 seconds slow of NTP time' -> -0.000123.

    Sign convention: NEGATIVE when the local clock is BEHIND (slow of) NTP,
    positive when ahead (fast of). That matches `Last offset`, which chrony
    prints already signed. The original dropped the word and returned every
    offset as positive, so a bag could not say which way to correct.
    """
    parts = s.replace("[", " ").replace("]", " ").split()
    if not parts:
        return 0.0
    tok = parts[0].rstrip("[]")
    # a bare '-123us' style token: number and unit glued together
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)([a-z]*)$", tok)
    if not m:
        return 0.0
    val = float(m.group(1))
    unit = (m.group(2) or (parts[1].lower() if len(parts) > 1 else "seconds"))
    if unit in ("ms", "milliseconds"):     val /= 1_000.0
    elif unit in ("us", "microseconds"):   val /= 1_000_000.0
    elif unit in ("ns", "nanoseconds"):    val /= 1_000_000_000.0
    if "slow" in s:
        val = -abs(val)
    elif "fast" in s:
        val = abs(val)
    return val


def _parse_span(tok: str) -> float:
    """sourcestats Span: '1034', '89m', '17h', '2d' -> seconds; 0.0 if unreadable."""
    m = re.match(r"^(\d+(?:\.\d+)?)([smhd]?)$", tok.strip())
    if not m:
        return 0.0
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def _parse_chronyc() -> Optional[dict]:
    tracking = _run(["chronyc", "tracking"])
    sources  = _run(["chronyc", "sources", "-v"])
    if not tracking:
        return None

    metrics: dict = {
        "backend": "chronyc", "synchronized": False, "sync_source": "",
        "stratum": 16, "offset_seconds": 0.0, "delay_seconds": 0.0,
        "jitter_seconds": 0.0, "root_delay": 0.0, "root_dispersion": 0.0,
        "frequency_error_ppm": 0.0, "poll_interval_seconds": 0,
        "reach_register": 0, "reachability_percent": 0,
        "leap_indicator": "alarm", "warnings": [],
    }

    kv: dict[str, str] = {}
    for line in tracking.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            kv[k.strip()] = v.strip()

    ref_id = kv.get("Reference ID", "")
    m = re.search(r"\(([^)]+)\)", ref_id)
    metrics["sync_source"] = m.group(1) if m else ref_id.split()[0] if ref_id else ""

    if "Stratum" in kv:
        try: metrics["stratum"] = int(kv["Stratum"])
        except ValueError: pass

    # Chrony's own record of when the last poll landed. This is the only
    # honest reference_time: the node's cache timer fires every second
    # whether or not a poll happened, so its clock cannot count polls.
    # Format: "Sun Sep 13 10:00:00 2026", 1 s resolution, UTC.
    if "Ref time (UTC)" in kv:
        try:
            dt = datetime.strptime(kv["Ref time (UTC)"].strip(),
                                   "%a %b %d %H:%M:%S %Y")
            metrics["ref_time_epoch"] = dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    metrics["offset_seconds"]  = _parse_secs(kv.get("System time",     "0 seconds"))
    metrics["root_delay"]      = _parse_secs(kv.get("Root delay",      "0 seconds"))
    metrics["root_dispersion"] = _parse_secs(kv.get("Root dispersion", "0 seconds"))

    if "Frequency" in kv:
        # "27.237 ppm slow" -> -27.237. Negative = clock runs slow.
        try:
            f = float(kv["Frequency"].split()[0])
            metrics["frequency_error_ppm"] = -abs(f) if "slow" in kv["Frequency"] else abs(f)
        except ValueError:
            pass
    # Last offset is the MEASURED offset at the most recent poll, signed by
    # chrony. System time is the steered estimate. Keep both: the steered
    # value is what applies to a sensor stamp now; the measured one is the
    # observation it was derived from.
    if "Last offset" in kv:
        metrics["last_offset_seconds"] = _parse_secs(kv["Last offset"])
    if "Skew" in kv:
        try: metrics["skew_ppm"] = float(kv["Skew"].split()[0])
        except ValueError: pass
    if "RMS offset" in kv:
        metrics["rms_offset_seconds"] = abs(_parse_secs(kv["RMS offset"]))
    if "Residual freq" in kv:
        try:
            r = float(kv["Residual freq"].split()[0])
            metrics["residual_freq_ppm"] = -abs(r) if "slow" in kv["Residual freq"] else r
        except ValueError:
            pass
    if "Update interval" in kv:
        metrics["update_interval_seconds"] = abs(_parse_secs(kv["Update interval"]))

    leap_str = kv.get("Leap status", "Not synchronised").lower()
    if "normal" in leap_str:
        metrics["leap_indicator"] = "no_warning"; metrics["synchronized"] = True
    elif "insert" in leap_str:
        metrics["leap_indicator"] = "add_second"; metrics["synchronized"] = True
    elif "delete" in leap_str:
        metrics["leap_indicator"] = "del_second"; metrics["synchronized"] = True
    else:
        metrics["leap_indicator"] = "alarm"
        metrics["warnings"].append(f"Leap status: {leap_str}")

    # `sources -v` carries poll and reach. Its "Last sample" column is
    # adjusted[measured] +/- error, NOT delay and jitter -- the original read
    # it as such, hit a ValueError on "-123us[", and left both at 0.0 for
    # every poll of every run.
    for line in sources.splitlines():
        if not line.startswith("^*"):
            continue
        parts = line[2:].split()
        if len(parts) < 5:
            continue
        try:
            reach_octal = int(parts[3], 8)
            metrics["reach_register"]        = reach_octal
            metrics["reachability_percent"]  = _reachability_percent(reach_octal)
            metrics["poll_interval_seconds"] = 2 ** int(parts[2])
            # "... -123us[ -156us] +/-   45ms": the +/- term is the error
            # bound, roughly half the round-trip delay plus dispersion. It
            # is readable without socket access, unlike ntpdata.
            if "+/-" in parts:
                metrics["error_bound_seconds"] = abs(
                    _parse_secs(parts[parts.index("+/-") + 1]))
        except (ValueError, IndexError):
            pass
        break

    # Delay comes from `ntpdata` ("Peer delay"); jitter (std dev of the
    # offset fit) and skew from `sourcestats`. Two extra calls at 1 Hz.
    ntpdata = _run(["chronyc", "ntpdata"])
    for line in ntpdata.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip(); v = v.strip()
        if k == "Peer delay":
            metrics["delay_seconds"] = abs(_parse_secs(v))
        elif k == "Peer dispersion":
            metrics["peer_dispersion_seconds"] = abs(_parse_secs(v))
    # ntpdata needs the daemon's Unix socket, which an ordinary user cannot
    # open; it then prints nothing and delay stays 0.0. Fall back to twice
    # the sources error bound, which brackets the delay from above, and say
    # so. (Or: `sudo usermod -aG _chrony $USER` and re-login.)
    if metrics["delay_seconds"] == 0.0 and metrics.get("error_bound_seconds"):
        metrics["delay_seconds"] = 2.0 * metrics["error_bound_seconds"]
        metrics["warnings"].append(
            "delay_seconds is an upper bound (2x sources error term): "
            "chronyc ntpdata needs socket access")
    stats = _run(["chronyc", "sourcestats"])
    # Name/IP  NP NR Span Frequency FreqSkew Offset StdDev
    for line in stats.splitlines():
        parts = line.split()
        if len(parts) >= 8 and metrics.get("sync_source") and parts[0] == metrics["sync_source"]:
            try:
                metrics["jitter_seconds"]  = abs(_parse_secs(parts[7]))
                metrics["fit_samples"]     = int(parts[1])
                metrics["skew_ppm"]        = float(parts[5])
                metrics["fit_span_seconds"] = _parse_span(parts[3])
            except (ValueError, IndexError):
                pass
            break
    return metrics


# per-client view (server node only)

def parse_chronyc_clients() -> list[dict]:
    """
    Parse chronyc clients -v.  Returns one dict per connected client:
      ip, stratum, offset_seconds, jitter_seconds
    Returns [] if chronyc unavailable or no clients connected.
    """
    out = _run(["chronyc", "clients", "-v"])
    clients = []
    for line in out.splitlines():
        parts = line.split()
        if not parts:
            continue
        first = parts[0]
        # skip header / separator lines
        if not (first[0].isdigit() or "." in first or ":" in first):
            continue
        try:
            clients.append({
                "ip":             first,
                "stratum":        int(parts[3])          if len(parts) > 3 else 0,
                "offset_seconds": float(parts[4]) / 1e9  if len(parts) > 4 else 0.0,
                "jitter_seconds": float(parts[5]) / 1e9  if len(parts) > 5 else 0.0,
            })
        except (ValueError, IndexError):
            continue
    return clients


# server client count

def get_connected_clients() -> int:
    out = _run(["chronyc", "clients"])
    # without socket access chronyc prints "501 Not authorised" and nothing
    # else; a 3-digit status line is an error, not a client
    if out and any(re.match(r"^\d{3} ", l) for l in out.splitlines()):
        return -1
    if out:
        count = sum(
            1 for line in out.splitlines()
            if line
            and not line.startswith("Hostname")
            and not line.startswith("=")
            and not line.startswith("Clients")
        )
        return max(count, 0)
    out = _run(["ntpq", "-c", "mrulist"])
    if out:
        count = sum(
            1 for line in out.splitlines()
            if line
            and not line.startswith("lstint")
            and not line.startswith("=")
        )
        return max(count, 0)
    return -1


# public API

def get_ntp_metrics() -> dict:
    """Query local NTP daemon and return a unified metrics dict."""
    metrics = _parse_chronyc() or _parse_ntpq()
    if metrics is None:
        return {
            "backend": "none", "synchronized": False, "sync_source": "",
            "stratum": 16, "stratum_level": "16",
            "offset_seconds": 0.0, "delay_seconds": 0.0, "jitter_seconds": 0.0,
            "root_delay": 0.0, "root_dispersion": 0.0,
            "frequency_error_ppm": 0.0, "poll_interval_seconds": 0,
            "reach_register": 0, "reachability_percent": 0,
            "leap_indicator": "alarm",
            "warnings": ["No NTP daemon found (tried chronyc and ntpq)"],
        }
    metrics["stratum_level"] = str(metrics["stratum"])
    return metrics


# clock environment: temperature and load

def _find_cpu_thermal_zone() -> str:
    """The first sysfs zone whose type contains 'cpu' (cpu-thermal on Pi and
    Orin, CPU-therm on Xavier); zone numbers differ between boards, types do not."""
    import glob
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try:
            with open(z + "/type") as f:
                if "cpu" in f.read().lower():
                    return z + "/temp"
        except OSError:
            continue
    return "/sys/class/thermal/thermal_zone0/temp"


_THERMAL_PATH = None


def get_clock_environment() -> dict:
    """SoC temperature [C] and 1-minute load; NaN when unreadable."""
    global _THERMAL_PATH
    if _THERMAL_PATH is None:
        _THERMAL_PATH = _find_cpu_thermal_zone()
    out = {"temperature_c": float("nan"), "cpu_load_1min": float("nan")}
    try:
        with open(_THERMAL_PATH) as f:
            out["temperature_c"] = int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        pass
    try:
        with open("/proc/loadavg") as f:
            out["cpu_load_1min"] = float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return out

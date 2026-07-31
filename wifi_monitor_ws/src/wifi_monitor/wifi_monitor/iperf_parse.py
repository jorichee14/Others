"""Parse iperf3 JSON (`iperf3 -c ... -J`) into a flat metrics dict.

Kept ROS-free so it can be unit-tested without a network or iperf binary.
Returns a dict with any subset of the following keys populated:

    success (bool), error (str),
    protocol ("TCP"|"UDP"), bitrate_mbps, bytes, retransmits,
    rtt_ms_mean, rtt_ms_min, rtt_ms_max,
    jitter_ms, lost_packets, total_packets, lost_percent
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional


def _mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


# --- Streaming (per-second) interval line parser ---------------------------
# Matches a live iperf3 interval report line, e.g.:
#   [  5]   1.00-2.00   sec  5.97 MBytes  50.0 Mbits/sec    0    269 KBytes
#   [SUM]   1.00-2.00   sec  13.1 MBytes   110 Mbits/sec    0
_IV_LINE = re.compile(
    r"\[\s*(SUM|\d+)\]\s+([\d.]+)-\s*([\d.]+)\s+sec\s+"
    r"([\d.]+)\s+([KMGT]?)Bytes\s+"
    r"([\d.]+)\s+([KMGT]?)bits/sec"
    r"(?:\s+(\d+))?"
)
_UNIT = {"G": 1000.0, "M": 1.0, "K": 1e-3, "": 1e-6}
# Transfer amounts are binary (KiB/MiB/GiB) in iperf3 output.
_BYTE_UNIT = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}


def parse_interval_line(line: str, parallel: int = 1) -> Optional[Dict[str, object]]:
    """Parse one live iperf3 interval line into {mbps, bytes, seconds[, retransmits]}.

    Returns None for non-interval lines (headers, connect banner, the final
    sender/receiver summary). With parallel>1, only the aggregate ``[SUM]``
    line is used; with a single stream, the per-stream line is used.
    """
    if "sender" in line or "receiver" in line:
        return None  # final summary rows, not a live interval
    m = _IV_LINE.search(line)
    if not m:
        return None
    ident = m.group(1)
    if parallel > 1 and ident != "SUM":
        return None          # wait for the SUM of all streams
    if parallel <= 1 and ident == "SUM":
        return None          # single stream: ignore any stray SUM
    try:
        start, end = float(m.group(2)), float(m.group(3))
        xfer = float(m.group(4))
        rate = float(m.group(6))
    except ValueError:
        return None
    xfer_unit = (m.group(5) or "").upper()
    rate_unit = (m.group(7) or "").upper()
    res: Dict[str, object] = {
        "mbps": rate * _UNIT.get(rate_unit, 1e-6),
        "bytes": int(xfer * _BYTE_UNIT.get(xfer_unit, 1)),
        "seconds": max(0.0, end - start),
    }
    if m.group(8) is not None:
        res["retransmits"] = int(m.group(8))
    return res


# --- ss -ti TCP RTT parser (for dense RTT in continuous mode) ---------------
# Reads the kernel's tcpi_rtt of the live iperf connection(s), the same value
# iperf itself reports -- but samplable every second without waiting for an
# end summary. The idle control connection is flagged 'app_limited' and is
# excluded so the average reflects the loaded data streams.
_SS_RTT = re.compile(r"\brtt:([\d.]+)/[\d.]+")
_SS_MINRTT = re.compile(r"\bminrtt:([\d.]+)")


def parse_ss_rtt(text: str) -> Optional[Dict[str, object]]:
    """Aggregate TCP RTT from `ss -ti` output into {mean, min, max} ms.

    Averages the data connections (excludes ``app_limited`` control sockets).
    Falls back to including all sockets if every one looks app-limited.
    Returns None if no RTT could be read.
    """
    def collect(skip_app_limited: bool):
        means: List[float] = []
        mins: List[float] = []
        for line in text.splitlines():
            if "rtt:" not in line:
                continue
            if skip_app_limited and "app_limited" in line:
                continue
            m = _SS_RTT.search(line)
            if not m:
                continue
            means.append(float(m.group(1)))
            mm = _SS_MINRTT.search(line)
            if mm:
                mins.append(float(mm.group(1)))
        return means, mins

    means, mins = collect(True)
    if not means:
        means, mins = collect(False)
    if not means:
        return None
    return {
        "rtt_ms_mean": sum(means) / len(means),
        "rtt_ms_min": min(mins) if mins else min(means),
        "rtt_ms_max": max(means),
    }


def parse_iperf_json(text: str) -> Dict[str, object]:
    """Convert iperf3 JSON text into a metrics dict.

    Never raises on malformed input: returns {'success': False, 'error': ...}.
    """
    try:
        d = json.loads(text)
    except (ValueError, TypeError) as exc:
        return {"success": False, "error": f"bad iperf3 JSON: {exc}"}

    # iperf3 reports a top-level "error" on failure (e.g. unreachable server).
    if isinstance(d, dict) and d.get("error"):
        return {"success": False, "error": str(d["error"])}

    end = d.get("end", {}) if isinstance(d, dict) else {}
    res: Dict[str, object] = {"success": True, "error": ""}

    if "sum_received" in end or "sum_sent" in end:
        # --- TCP ---
        res["protocol"] = "TCP"
        recv = end.get("sum_received", {})
        sent = end.get("sum_sent", {})
        # goodput is best read at the receiver
        bps = recv.get("bits_per_second", sent.get("bits_per_second", 0.0))
        res["bitrate_mbps"] = float(bps) / 1e6
        res["bytes"] = int(recv.get("bytes", sent.get("bytes", 0)))
        res["retransmits"] = int(sent.get("retransmits", 0) or 0)

        # Per-stream sender RTT (microseconds) -> ms, aggregated over streams.
        means, mins, maxs = [], [], []
        for s in end.get("streams", []):
            snd = s.get("sender", {})
            if "mean_rtt" in snd:
                means.append(snd.get("mean_rtt"))
                mins.append(snd.get("min_rtt"))
                maxs.append(snd.get("max_rtt"))
        mean_rtt = _mean(means)
        if mean_rtt is not None:
            res["rtt_ms_mean"] = mean_rtt / 1000.0
            res["rtt_ms_min"] = min(v for v in mins if v is not None) / 1000.0
            res["rtt_ms_max"] = max(v for v in maxs if v is not None) / 1000.0

    elif "sum" in end:
        # --- UDP ---
        res["protocol"] = "UDP"
        s = end.get("sum", {})
        res["bitrate_mbps"] = float(s.get("bits_per_second", 0.0)) / 1e6
        res["bytes"] = int(s.get("bytes", 0))
        if s.get("jitter_ms") is not None:
            res["jitter_ms"] = float(s["jitter_ms"])
        res["lost_packets"] = int(s.get("lost_packets", 0) or 0)
        res["total_packets"] = int(s.get("packets", 0) or 0)
        if s.get("lost_percent") is not None:
            res["lost_percent"] = float(s["lost_percent"])
    else:
        return {"success": False, "error": "iperf3 JSON missing end summary"}

    return res

"""Parse individual lines of streaming ``ping`` output.

Kept ROS-free for unit testing. Handles the common iputs/BusyBox reply
formats plus the loss indicators emitted by ``ping -O`` and unreachable
errors. Returns None for banner/summary/other lines.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

# 64 bytes from 192.168.233.142: icmp_seq=1 ttl=64 time=3.45 ms
_REPLY = re.compile(r"icmp_[rs]eq=(\d+).*?time=([\d.]+)\s*ms")
_SEQ = re.compile(r"icmp_[rs]eq=(\d+)")


def parse_ping_line(line: str) -> Optional[Dict[str, object]]:
    """Return {seq, reply, rtt_ms} for a ping line, or None if not a result.

    * a normal reply         -> {"seq": n, "reply": True,  "rtt_ms": float}
    * ``-O`` "no answer yet"  -> {"seq": n, "reply": False, "rtt_ms": None}
    * "Destination ... Unreachable" -> {"seq": n|-1, "reply": False, ...}
    """
    if "time=" in line and "icmp_" in line:
        m = _REPLY.search(line)
        if m:
            return {"seq": int(m.group(1)), "reply": True,
                    "rtt_ms": float(m.group(2))}

    if "no answer yet" in line or "Unreachable" in line:
        m = _SEQ.search(line)
        return {"seq": int(m.group(1)) if m else -1, "reply": False,
                "rtt_ms": None}

    return None

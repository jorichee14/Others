#!/usr/bin/env python3
"""UDP echo server for the burst_monitor node. No ROS, no dependencies.

Runs on the wired laptop (Windows, Linux or macOS -- stdlib only):

    python3 burst_echo_server.py            # listens on 0.0.0.0:5600
    python3 burst_echo_server.py --port 5601

Deliberately dumb: it reflects each datagram straight back and does nothing
else. No retransmission, no reordering, no buffering beyond the socket, so a
packet lost on the air stays lost and shows up as loss in the measurement
rather than being papered over. All timing happens on the robot's clock,
which removes any need for clock sync between the two machines.

Windows: allow inbound UDP on the port first, e.g.

    New-NetFirewallRule -DisplayName "burst-echo" -Direction Inbound `
        -Protocol UDP -LocalPort 5600 -Action Allow
"""

from __future__ import annotations

import argparse
import socket
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument(
        "--quiet", action="store_true",
        help="Suppress the periodic packet count.",
    )
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Generous buffers: a 40 kB burst is ~30 datagrams arriving back to back,
    # and a small receive buffer would drop them here rather than on the air,
    # which would look like wireless loss in the results.
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
    except OSError:
        pass
    sock.bind((args.host, args.port))

    print(f"burst echo server listening on {args.host}:{args.port}", flush=True)
    n = 0
    try:
        while True:
            data, addr = sock.recvfrom(4096)
            sock.sendto(data, addr)
            n += 1
            if not args.quiet and n % 10000 == 0:
                print(f"  echoed {n} datagrams", flush=True)
    except KeyboardInterrupt:
        print(f"\nstopped after {n} datagrams", flush=True)
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

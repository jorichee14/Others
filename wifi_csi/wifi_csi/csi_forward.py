#!/usr/bin/env python3
"""Forward nexmon CSI datagrams over a wired link, without ROS 2.

Reads the extractor's UDP dumps, filters them by source MAC, and re-sends each
datagram VERBATIM to a destination host. The payload stays in nexmon's wire
format, so the far end parses it with the unmodified csi_parser.parse_frame()
and nothing new has to be agreed between the two sides.

Deliberately stdlib-only. The source MAC lives at a fixed header offset, so
there is no reason to decode 256 subcarriers per frame just to forward the
bytes unchanged; skipping that keeps the hot loop cheap at 500 frames/s on a
Pi and lets this run on a machine with no ROS 2 and no numpy installed.

    ./csi_forward.py --dest 192.168.1.50:5500 \
        --mac-filter 88:76:b9:ea:e0:ff,88:76:b9:ea:e1:01

Do NOT run this at the same time as csi_publisher: both bind UDP 5500, and
with SO_REUSEADDR the kernel hands each datagram to only one of them, so the
two silently steal frames from each other.
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
import time

MAGIC = 0x1111
HEADER_LEN = 18
MAC_OFFSET = 4
DEFAULT_PORT = 5500


def parse_dest(text: str) -> tuple[str, int]:
    """'host:port' -> (host, port); bare 'host' keeps the default port."""
    host, _, port = text.rpartition(":")
    if not host:
        return text, DEFAULT_PORT
    return host, int(port)


def src_mac(payload: bytes) -> str:
    """Source MAC straight out of the header — same bytes parse_frame reads."""
    return payload[MAC_OFFSET:MAC_OFFSET + 6].hex(":")


def build_sender(bind_dev: str | None, src_ip: str | None) -> socket.socket:
    """Socket for the outbound leg.

    Which interface the frames leave by is normally decided by the routing
    table, so a destination on the wired subnet already goes out the wired
    link. --src-ip pins it without privileges by binding the wired address;
    --bind-dev forces it with SO_BINDTODEVICE, which needs CAP_NET_RAW and is
    only worth reaching for when several interfaces share a route.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
    if bind_dev:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE,
                         bind_dev.encode() + b"\0")
        except PermissionError:
            sys.exit(f"--bind-dev {bind_dev} needs root or CAP_NET_RAW; "
                     "use --src-ip <address of that interface> instead")
    if src_ip:
        s.bind((src_ip, 0))
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, metavar="HOST[:PORT]",
                    help="where to forward to")
    ap.add_argument("--listen-port", type=int, default=DEFAULT_PORT,
                    help="port the extractor dumps to (default: %(default)s)")
    ap.add_argument("--mac-filter", default="",
                    help="comma-separated source MACs to keep; empty = keep all")
    ap.add_argument("--split", action="store_true",
                    help="send the Nth MAC in --mac-filter to destination port "
                         "+N-1, mirroring the /mobileN topics")
    ap.add_argument("--bind-dev", metavar="IFACE",
                    help="force egress via this interface (needs root)")
    ap.add_argument("--src-ip", metavar="ADDR",
                    help="bind the sending socket to this local address")
    ap.add_argument("--stats-period", type=float, default=2.0,
                    help="seconds between stats lines, 0 to disable")
    args = ap.parse_args(argv)

    macs = [m.strip().lower() for m in args.mac_filter.split(",") if m.strip()]
    host, port = parse_dest(args.dest)
    # With --split each MAC gets its own destination port, so the receiver can
    # keep the streams apart without parsing anything.
    dests = {mac: (host, port + i) for i, mac in enumerate(macs)} if args.split else {}

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    rx.bind(("0.0.0.0", args.listen_port))
    rx.settimeout(0.5)
    tx = build_sender(args.bind_dev, args.src_ip)

    if dests:
        for mac, d in dests.items():
            print(f"{mac} -> {d[0]}:{d[1]}", file=sys.stderr)
    else:
        print(f"{', '.join(macs) or 'all sources'} -> {host}:{port}", file=sys.stderr)

    n_in = n_out = n_skip = 0
    window = time.monotonic()
    try:
        while True:
            try:
                data, _ = rx.recvfrom(4096)
            except socket.timeout:
                data = None
            except OSError:
                break

            if data is not None:
                n_in += 1
                # Same two cheap validity checks parse_frame makes, so a stray
                # datagram on this port is dropped rather than forwarded as if
                # it were CSI.
                if (len(data) < HEADER_LEN + 4
                        or struct.unpack_from("<H", data, 0)[0] != MAGIC):
                    n_skip += 1
                else:
                    mac = src_mac(data)
                    if macs and mac not in macs:
                        n_skip += 1
                    else:
                        tx.sendto(data, dests.get(mac, (host, port)))
                        n_out += 1

            if args.stats_period > 0:
                now = time.monotonic()
                dt = now - window
                if dt >= args.stats_period:
                    print(f"{n_out / dt:7.1f} fwd/s   in={n_in} out={n_out} "
                          f"skipped={n_skip}", file=sys.stderr)
                    n_in = n_out = n_skip = 0
                    window = now
    except KeyboardInterrupt:
        pass
    finally:
        rx.close()
        tx.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

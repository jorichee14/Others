#!/usr/bin/env python3
"""Send fixed-size bursts into an idle link and measure delivery-vs-deadline.

The capacity question ("how much does this link move in 10 s") and the
deadline question ("will a 40 kB feature map arrive in 50 ms") are different
measurements, and they diverge exactly where a collaborative-perception
scheduler has to make a decision. iperf3 answers the first. This node answers
the second.

Why not iperf3 with -n: a fresh iperf3 connection costs a TCP handshake plus
a control-channel exchange, hundreds of milliseconds on this link. For a
40 kB transfer that overhead is an order of magnitude larger than the thing
being measured. This node keeps one UDP socket open for the life of the run,
so a burst costs exactly the airtime it needs.

Why round-trip: one-way delay needs synchronised clocks AND a timestamping
receiver. The echo server (scripts/burst_echo_server.py) is deliberately
dumb -- it reflects datagrams and nothing else -- so all timing happens on
one clock here. Halve completion_ms for a rough one-way figure, but note
that uplink and downlink are not symmetric on Wi-Fi, so it is only rough.

Deliberately NOT saturating: bursts go into an otherwise idle link, which is
what a real feature-map transmission meets. Run this as its own pass, not
alongside iperf.

Parameters
----------
target : str          echo server host.
port : int            echo server UDP port. Default 5600.
burst_bytes : int     total payload per burst. Default 40960 (40 kB).
packet_bytes : int    per-datagram payload. Default 1400 (under a 1500 MTU
                      to avoid IP fragmentation, which would turn one lost
                      fragment into a lost datagram).
interval_s : float    seconds between bursts. Default 1.0.
deadline_ms : float   the deadline each burst is judged against. Default 50.
timeout_ms : float    give up waiting for stragglers. Should exceed
                      deadline_ms so late-but-arrived is distinguishable from
                      lost. Default 500.
interface : str       wireless iface for the link-down failsafe (""=auto).
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from comms_msgs.msg import BurstResult

from comms.wifi import wifi_parsers

NAN = float("nan")
MAGIC = 0xB0157

# seq (uint32) + burst id (uint32) + magic (uint32) -- enough to reject
# stragglers from a previous burst, which would otherwise be counted as
# this burst's packets and make completion_ms nonsense.
_HDR = struct.Struct("<III")


class BurstMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("burst_monitor")

        self.declare_parameter("target", "192.168.51.11")
        self.declare_parameter("port", 5600)
        self.declare_parameter("burst_bytes", 40960)
        self.declare_parameter("packet_bytes", 1400)
        self.declare_parameter("interval_s", 1.0)
        self.declare_parameter("deadline_ms", 50.0)
        self.declare_parameter("timeout_ms", 500.0)
        self.declare_parameter("interface", "")

        gp = self.get_parameter
        self._target = gp("target").get_parameter_value().string_value
        self._port = gp("port").get_parameter_value().integer_value
        self._burst_bytes = gp("burst_bytes").get_parameter_value().integer_value
        self._pkt_bytes = gp("packet_bytes").get_parameter_value().integer_value
        self._interval = gp("interval_s").get_parameter_value().double_value
        self._deadline = gp("deadline_ms").get_parameter_value().double_value
        self._timeout = gp("timeout_ms").get_parameter_value().double_value
        self._iface = gp("interface").get_parameter_value().string_value

        if self._pkt_bytes < _HDR.size:
            self._pkt_bytes = _HDR.size
        if self._pkt_bytes > 1472:
            self.get_logger().warn(
                f"packet_bytes {self._pkt_bytes} exceeds a 1500-byte MTU; IP "
                "will fragment and one lost fragment loses the whole "
                "datagram. 1400 is safe."
            )
        self._n_pkts = max(1, math.ceil(self._burst_bytes / self._pkt_bytes))

        if not self._iface:
            found = wifi_parsers.list_wireless_interfaces()
            self._iface = found[0] if found else ""

        qos = QoSProfile(
            depth=10,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._pub = self.create_publisher(BurstResult, "wifi/burst", qos)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self._sock.settimeout(0.05)

        self._burst_id = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f"burst_monitor -> {self._target}:{self._port}, "
            f"{self._burst_bytes}B in {self._n_pkts}x{self._pkt_bytes}B every "
            f"{self._interval:.1f}s, deadline {self._deadline:.0f}ms, iface "
            f"'{self._iface or '<none>'}' -> topic 'wifi/burst'."
        )

    # ----------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set() and rclpy.ok():
            if self._iface and wifi_parsers.link_up(self._iface) is False:
                msg = self._new_msg()
                msg.complete = False
                msg.met_deadline = False
                msg.loss_percent = 100.0
                self._pub.publish(msg)
                if self._stop.wait(max(1.0, self._interval)):
                    break
                continue

            self._pub.publish(self._one_burst())
            if self._stop.wait(max(0.0, self._interval)):
                break

    def _new_msg(self) -> BurstResult:
        msg = BurstResult()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "wifi"
        msg.target = self._target
        msg.port = int(self._port)
        msg.burst_bytes = int(self._burst_bytes)
        msg.packet_bytes = int(self._pkt_bytes)
        msg.packets_sent = int(self._n_pkts)
        msg.packets_returned = 0
        msg.loss_percent = NAN
        msg.completion_ms = NAN
        msg.first_reply_ms = NAN
        msg.spread_ms = NAN
        msg.deadline_ms = float(self._deadline)
        msg.met_deadline = False
        msg.complete = False
        msg.effective_mbps = NAN
        return msg

    def _drain_stale(self) -> None:
        """Discard anything still in flight from the previous burst."""
        self._sock.settimeout(0.0)
        try:
            while True:
                self._sock.recv(2048)
        except (BlockingIOError, socket.timeout, OSError):
            pass

    def _one_burst(self) -> BurstResult:
        msg = self._new_msg()
        self._burst_id = (self._burst_id + 1) & 0xFFFFFFFF
        bid = self._burst_id
        self._drain_stale()

        payload = bytes(self._pkt_bytes - _HDR.size)
        t0 = time.perf_counter()
        try:
            for seq in range(self._n_pkts):
                self._sock.sendto(
                    _HDR.pack(seq, bid, MAGIC) + payload,
                    (self._target, self._port),
                )
        except OSError as exc:
            self.get_logger().warn(f"burst send failed: {exc}")
            msg.loss_percent = 100.0
            return msg

        # Collect echoes until every packet is back or we stop waiting.
        seen = set()
        t_first = None
        t_last = None
        deadline_t = t0 + self._timeout / 1000.0
        self._sock.settimeout(0.01)
        while len(seen) < self._n_pkts and time.perf_counter() < deadline_t:
            try:
                data = self._sock.recv(2048)
            except (socket.timeout, BlockingIOError):
                continue
            except OSError:
                break
            if len(data) < _HDR.size:
                continue
            seq, rid, magic = _HDR.unpack_from(data, 0)
            if magic != MAGIC or rid != bid or seq >= self._n_pkts:
                continue          # straggler from an earlier burst
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last = now
            seen.add(seq)

        n = len(seen)
        msg.packets_returned = int(n)
        msg.loss_percent = 100.0 * (1.0 - n / self._n_pkts)
        msg.complete = n == self._n_pkts

        if t_last is not None:
            msg.completion_ms = (t_last - t0) * 1000.0
            msg.first_reply_ms = (t_first - t0) * 1000.0
            msg.spread_ms = msg.completion_ms - msg.first_reply_ms
            # Only a complete burst has a meaningful completion time: if
            # packets were lost, t_last is just when the last survivor
            # happened to arrive.
            msg.met_deadline = msg.complete and msg.completion_ms <= self._deadline
            if msg.complete and msg.completion_ms > 0:
                bits = self._burst_bytes * 8.0
                msg.effective_mbps = bits / (msg.completion_ms / 1000.0) / 1e6

        if not msg.complete:
            self.get_logger().warn(
                f"burst {n}/{self._n_pkts} returned "
                f"({msg.loss_percent:.0f}% loss)"
            )
        return msg

    # ----------------------------------------------------------------------
    def destroy_node(self) -> bool:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self._sock.close()
        except OSError:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BurstMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

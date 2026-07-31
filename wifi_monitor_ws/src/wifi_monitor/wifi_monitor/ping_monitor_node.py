#!/usr/bin/env python3
"""Continuously ping a fixed host and publish RTT + rolling loss.

Runs on the robot and pings a target (typically the iperf3 server / the
static wired laptop). Unlike iperf, ping is cheap -- it does not saturate the
link -- so it can run continuously during real operation to map latency and
packet loss against position. The target only needs to reply to ICMP; no ROS
or extra software is required there (on Windows, allow ICMP Echo through the
firewall).

Publishes a ``PingStat`` per ping (~``1/interval_s`` Hz) with the single-shot
RTT plus rolling-window loss and RTT statistics.

Parameters
----------
target : str        host to ping. Default 192.168.233.142.
interval_s : float  seconds between pings (ping -i). Default 1.0.
timeout_s : float   per-reply timeout (ping -W). Default 1.0.
window : int        rolling window (samples) for loss_percent / rtt stats.
                    Default 20.
interface : str     bind ping to this interface (ping -I); ""=default route.
"""

from __future__ import annotations

import collections
import select
import shutil
import subprocess
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from wifi_monitor_msgs.msg import PingStat

from wifi_monitor import ping_parse

NAN = float("nan")


class PingMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("ping_monitor")

        self.declare_parameter("target", "192.168.233.142")
        self.declare_parameter("interval_s", 1.0)
        self.declare_parameter("timeout_s", 1.0)
        self.declare_parameter("window", 20)
        self.declare_parameter("interface", "")

        gp = self.get_parameter
        self._target = gp("target").get_parameter_value().string_value
        self._interval = gp("interval_s").get_parameter_value().double_value
        self._timeout = gp("timeout_s").get_parameter_value().double_value
        self._window = max(1, gp("window").get_parameter_value().integer_value)
        self._iface = gp("interface").get_parameter_value().string_value

        qos = QoSProfile(
            depth=10,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._pub = self.create_publisher(PingStat, "wifi/ping", qos)

        if not self._target:
            self.get_logger().warn("'target' is empty; set it to ping.")
        if shutil.which("ping") is None:
            self.get_logger().error("ping not found on PATH.")

        self._recent = collections.deque(maxlen=self._window)  # (reply, rtt)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f"ping_monitor -> '{self._target}' every {self._interval:.1f}s "
            f"(window {self._window}) -> topic 'wifi/ping'."
        )

    # ----------------------------------------------------------------------
    def _build_cmd(self) -> list:
        cmd = ["ping", "-n", "-O", "-i", str(self._interval),
               "-W", str(self._timeout)]
        if self._iface:
            cmd += ["-I", self._iface]
        cmd += [self._target]
        return cmd

    def _loop(self) -> None:
        while not self._stop.is_set() and rclpy.ok():
            if not self._target or shutil.which("ping") is None:
                if self._stop.wait(3.0):
                    break
                continue
            self._run_ping()          # blocks until process exits or stop
            # ping exited (e.g. transient failure); pause briefly and restart.
            if self._stop.wait(1.0):
                break

    def _run_ping(self) -> None:
        try:
            proc = subprocess.Popen(
                self._build_cmd(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
        except OSError as exc:
            self.get_logger().error(f"ping launch failed: {exc}")
            return
        try:
            while not self._stop.is_set() and proc.poll() is None:
                rlist, _, _ = select.select([proc.stdout], [], [], 1.0)
                if not rlist:
                    continue
                line = proc.stdout.readline()
                if line == "":
                    break
                parsed = ping_parse.parse_ping_line(line)
                if parsed is not None:
                    self._publish(parsed)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def _publish(self, parsed: dict) -> None:
        reply = bool(parsed["reply"])
        rtt = parsed["rtt_ms"]
        self._recent.append((reply, rtt))

        received = [r for ok, r in self._recent if ok and r is not None]
        n = len(self._recent)
        lost = sum(1 for ok, _ in self._recent if not ok)

        msg = PingStat()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "wifi"
        msg.target = self._target
        seq = int(parsed["seq"])
        msg.icmp_seq = seq if seq >= 0 else 0
        msg.reply = reply
        msg.rtt_ms = float(rtt) if (reply and rtt is not None) else NAN
        msg.window = n
        msg.loss_percent = 100.0 * lost / n if n else NAN
        if received:
            msg.rtt_ms_avg = sum(received) / len(received)
            msg.rtt_ms_min = min(received)
            msg.rtt_ms_max = max(received)
        else:
            msg.rtt_ms_avg = NAN
            msg.rtt_ms_min = NAN
            msg.rtt_ms_max = NAN
        self._pub.publish(msg)

    # ----------------------------------------------------------------------
    def destroy_node(self) -> bool:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PingMonitorNode()
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

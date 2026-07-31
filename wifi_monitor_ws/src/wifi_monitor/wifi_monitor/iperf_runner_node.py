#!/usr/bin/env python3
"""Periodically run iperf3 against a fixed server and publish the result.

Designed for the "static wired laptop as iperf3 server" setup: the robot is
the iperf3 client, the laptop (plugged into the router via Ethernet) runs
``iperf3 -s``, so each measurement reflects the robot's Wi-Fi link capacity
over a single wireless hop.

Runs each test in a background thread so the ROS executor is never blocked
for the test duration. Publishes an ``IperfResult`` per test; stamp each with
``header.stamp`` so it time-joins to pose in a rosbag.

Failsafes for a moving robot (link drops and reconnects):
  * Before each test the node checks link carrier. If the Wi-Fi is not
    associated it does NOT launch iperf (which would hang or fail slowly);
    it publishes a ``success=false`` result tagged "link down" and polls at
    ``reconnect_poll_s`` until the link returns, then resumes automatically.
  * ``iperf3 --connect-timeout`` bounds how long a single test waits for the
    server, so a reachable-Wi-Fi-but-unreachable-server case fails fast.
  * A failed test backs off to ``reconnect_poll_s`` even in survey mode
    (interval 0), so a dead server is never hammered.
  * All subprocess errors are caught; the node never crashes on a bad test.

Parameters
----------
server_address : str   iperf3 server IP/host. Default 192.168.233.142.
server_port : int      default 5201.
interface : str        wireless iface for the link-state failsafe (""=auto).
protocol : str         "tcp" or "udp".
duration_s : float     per-test duration (iperf3 -t). Default 2.0.
interval_s : float     gap between tests. 0 => back-to-back (survey). Def 30.
reverse : bool         true => downlink (server -> robot, iperf3 -R).
udp_bitrate : str      target rate for UDP tests, e.g. "300M". Default "0".
omit_s : float         initial seconds to omit (iperf3 -O). Default 1.0.
connect_timeout_ms : int  iperf3 --connect-timeout. Default 2000.
reconnect_poll_s : float  poll/backoff period when link is down or a test
                          fails. Default 3.0.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from wifi_monitor_msgs.msg import IperfResult

from wifi_monitor import iperf_parse, wifi_parsers

NAN = float("nan")


class IperfRunnerNode(Node):
    def __init__(self) -> None:
        super().__init__("iperf_runner")

        self.declare_parameter("server_address", "192.168.233.142")
        self.declare_parameter("server_port", 5201)
        self.declare_parameter("interface", "")
        self.declare_parameter("protocol", "tcp")
        self.declare_parameter("duration_s", 2.0)
        self.declare_parameter("interval_s", 30.0)
        self.declare_parameter("reverse", False)
        self.declare_parameter("udp_bitrate", "0")
        self.declare_parameter("omit_s", 1.0)
        self.declare_parameter("connect_timeout_ms", 2000)
        self.declare_parameter("reconnect_poll_s", 3.0)

        gp = self.get_parameter
        self._server = gp("server_address").get_parameter_value().string_value
        self._port = gp("server_port").get_parameter_value().integer_value
        self._iface = gp("interface").get_parameter_value().string_value
        self._proto = (
            gp("protocol").get_parameter_value().string_value or "tcp"
        ).lower()
        self._duration = gp("duration_s").get_parameter_value().double_value
        self._interval = gp("interval_s").get_parameter_value().double_value
        self._reverse = gp("reverse").get_parameter_value().bool_value
        self._udp_bitrate = (
            gp("udp_bitrate").get_parameter_value().string_value or "0"
        )
        self._omit = gp("omit_s").get_parameter_value().double_value
        self._connect_timeout = (
            gp("connect_timeout_ms").get_parameter_value().integer_value
        )
        self._reconnect_poll = (
            gp("reconnect_poll_s").get_parameter_value().double_value
        )

        if not self._iface:
            found = wifi_parsers.list_wireless_interfaces()
            self._iface = found[0] if found else ""

        qos = QoSProfile(
            depth=10,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._pub = self.create_publisher(IperfResult, "wifi/iperf", qos)

        if not self._server:
            self.get_logger().warn(
                "'server_address' is empty; set it to the iperf3 server IP."
            )
        if shutil.which("iperf3") is None:
            self.get_logger().error(
                "iperf3 not found on PATH; install it (apt install iperf3)."
            )

        self._link_up = None  # tracks link state for transition logging
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f"iperf_runner -> {self._proto.upper()} to "
            f"'{self._server or '<unset>'}:{self._port}', "
            f"{self._duration:.0f}s every {self._interval:.0f}s, "
            f"{'downlink' if self._reverse else 'uplink'}, iface "
            f"'{self._iface or '<none>'}' -> topic 'wifi/iperf'."
        )

    # ----------------------------------------------------------------------
    def _loop(self) -> None:
        poll = max(0.5, self._reconnect_poll)
        while not self._stop.is_set() and rclpy.ok():
            if not self._server or shutil.which("iperf3") is None:
                if self._stop.wait(poll):
                    break
                continue

            # Failsafe: don't launch iperf while the Wi-Fi is disconnected.
            up = wifi_parsers.link_up(self._iface) if self._iface else None
            if up is False:
                if self._link_up is not False:
                    self.get_logger().warn(
                        f"link down on '{self._iface}'; pausing iperf, "
                        "will resume on reconnect."
                    )
                self._link_up = False
                self._pub.publish(
                    self._new_msg(success=False, error="link down (not associated)")
                )
                if self._stop.wait(poll):
                    break
                continue

            if self._link_up is False:
                self.get_logger().info(
                    f"link back up on '{self._iface}'; resuming iperf."
                )
            self._link_up = True

            msg = self._run_once()
            self._pub.publish(msg)

            # Normal cadence on success; back off on failure so a dead server
            # (even in survey mode, interval 0) is not hammered.
            wait = self._interval if msg.success else max(self._interval, poll)
            if self._stop.wait(max(0.0, wait)):
                break

    # ----------------------------------------------------------------------
    def _new_msg(self, success: bool = False, error: str = "") -> IperfResult:
        """Fresh IperfResult with header + config fields + NaN defaults."""
        msg = IperfResult()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "wifi"
        msg.server_address = self._server
        msg.server_port = int(self._port)
        msg.protocol = self._proto.upper()
        msg.reverse = bool(self._reverse)
        msg.duration_s = float(self._duration)
        msg.success = success
        msg.error = error
        msg.bitrate_mbps = NAN
        msg.rtt_ms_mean = NAN
        msg.rtt_ms_min = NAN
        msg.rtt_ms_max = NAN
        msg.jitter_ms = NAN
        msg.lost_percent = NAN
        return msg

    def _build_cmd(self) -> list:
        cmd = [
            "iperf3", "-c", self._server, "-p", str(self._port),
            "-t", str(self._duration), "-J",
        ]
        if self._connect_timeout > 0:
            cmd += ["--connect-timeout", str(self._connect_timeout)]
        if self._omit > 0.0:
            cmd += ["-O", str(self._omit)]
        if self._reverse:
            cmd += ["-R"]
        if self._proto == "udp":
            cmd += ["-u", "-b", self._udp_bitrate]
        return cmd

    def _run_once(self) -> IperfResult:
        msg = self._new_msg()
        timeout = self._duration + self._omit + 10.0
        try:
            out = subprocess.run(
                self._build_cmd(),
                capture_output=True, text=True, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            msg.success = False
            msg.error = "iperf3 timed out"
            self.get_logger().warn("iperf3 test timed out")
            return msg
        except OSError as exc:
            msg.success = False
            msg.error = f"iperf3 launch failed: {exc}"
            return msg

        parsed = iperf_parse.parse_iperf_json(out.stdout or "")
        if not parsed.get("success"):
            msg.success = False
            msg.error = str(
                parsed.get("error")
                or (out.stderr or "").strip()
                or f"iperf3 exit {out.returncode}"
            )
            self.get_logger().warn(f"iperf3 test failed: {msg.error}")
            return msg

        self._apply(msg, parsed)
        msg.success = True
        msg.error = ""
        arrow = "down" if self._reverse else "up"
        self.get_logger().info(
            f"iperf {msg.protocol} {arrow}: {msg.bitrate_mbps:.1f} Mbit/s"
            + (f", loss {msg.lost_percent:.1f}%" if msg.protocol == "UDP"
               else (f", rtt {msg.rtt_ms_mean:.1f} ms"
                     if not math.isnan(msg.rtt_ms_mean) else ""))
        )
        return msg

    @staticmethod
    def _apply(msg: IperfResult, parsed: dict) -> None:
        if "protocol" in parsed:
            msg.protocol = str(parsed["protocol"])
        for f in (
            "bitrate_mbps", "rtt_ms_mean", "rtt_ms_min", "rtt_ms_max",
            "jitter_ms", "lost_percent",
        ):
            if f in parsed:
                setattr(msg, f, float(parsed[f]))
        if "bytes" in parsed:
            msg.bytes = int(parsed["bytes"])
        if "retransmits" in parsed:
            msg.retransmits = int(parsed["retransmits"])
        if "lost_packets" in parsed:
            msg.lost_packets = int(parsed["lost_packets"])
        if "total_packets" in parsed:
            msg.total_packets = int(parsed["total_packets"])

    # ----------------------------------------------------------------------
    def destroy_node(self) -> bool:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IperfRunnerNode()
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

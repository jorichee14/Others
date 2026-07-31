#!/usr/bin/env python3
"""Periodically run iperf3 against a fixed server and publish the result.

Designed for the "static wired laptop as iperf3 server" setup: the robot is
the iperf3 client, the laptop (plugged into the router via Ethernet) runs
``iperf3 -s``, so each measurement reflects the robot's Wi-Fi link capacity
over a single wireless hop.

Runs each test in a background thread so the ROS executor is never blocked
for the test duration. Publishes an ``IperfResult`` per test; stamp each with
``header.stamp`` so it time-joins to pose in a rosbag.

Parameters
----------
server_address : str   iperf3 server IP/host (required; node warns if empty).
server_port : int      default 5201.
protocol : str         "tcp" or "udp".
duration_s : float     per-test duration (iperf3 -t). Default 2.0.
interval_s : float     gap between the END of one test and the START of the
                       next. 0 => back-to-back (survey mode). Default 30.0.
reverse : bool         true => measure downlink (server -> robot, iperf3 -R).
udp_bitrate : str      target rate for UDP tests, e.g. "300M". Default "0"
                       (iperf3 default 1 Mbit/s -> set this for UDP capacity).
omit_s : float         initial seconds to omit (iperf3 -O) to skip TCP
                       slow-start. Default 1.0.
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

from wifi_monitor import iperf_parse

NAN = float("nan")


class IperfRunnerNode(Node):
    def __init__(self) -> None:
        super().__init__("iperf_runner")

        self.declare_parameter("server_address", "")
        self.declare_parameter("server_port", 5201)
        self.declare_parameter("protocol", "tcp")
        self.declare_parameter("duration_s", 2.0)
        self.declare_parameter("interval_s", 30.0)
        self.declare_parameter("reverse", False)
        self.declare_parameter("udp_bitrate", "0")
        self.declare_parameter("omit_s", 1.0)

        gp = self.get_parameter
        self._server = gp("server_address").get_parameter_value().string_value
        self._port = gp("server_port").get_parameter_value().integer_value
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

        qos = QoSProfile(
            depth=10,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._pub = self.create_publisher(IperfResult, "wifi/iperf", qos)

        if not self._server:
            self.get_logger().warn(
                "'server_address' is empty; set it to the iperf3 server IP. "
                "No tests will run until it is provided."
            )
        if shutil.which("iperf3") is None:
            self.get_logger().error(
                "iperf3 not found on PATH; install it (apt install iperf3)."
            )

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f"iperf_runner -> {self._proto.upper()} to "
            f"'{self._server or '<unset>'}:{self._port}', "
            f"{self._duration:.0f}s every {self._interval:.0f}s, "
            f"{'downlink' if self._reverse else 'uplink'} -> topic 'wifi/iperf'."
        )

    # ----------------------------------------------------------------------
    def _loop(self) -> None:
        # Wait one interval before the first test if a gap is configured,
        # otherwise start immediately (survey mode).
        while not self._stop.is_set() and rclpy.ok():
            if self._server and shutil.which("iperf3") is not None:
                msg = self._run_once()
                self._pub.publish(msg)
            # Sleep the interval, but wake promptly on shutdown.
            wait = self._interval if self._server else 5.0
            if self._stop.wait(max(0.0, wait)):
                break

    # ----------------------------------------------------------------------
    def _build_cmd(self) -> list:
        cmd = [
            "iperf3", "-c", self._server, "-p", str(self._port),
            "-t", str(self._duration), "-J",
        ]
        if self._omit > 0.0:
            cmd += ["-O", str(self._omit)]
        if self._reverse:
            cmd += ["-R"]
        if self._proto == "udp":
            cmd += ["-u", "-b", self._udp_bitrate]
        return cmd

    def _run_once(self) -> IperfResult:
        msg = IperfResult()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "wifi"
        msg.server_address = self._server
        msg.server_port = int(self._port)
        msg.protocol = self._proto.upper()
        msg.reverse = bool(self._reverse)
        msg.duration_s = float(self._duration)

        # Unknown-value defaults.
        msg.bitrate_mbps = NAN
        msg.rtt_ms_mean = NAN
        msg.rtt_ms_min = NAN
        msg.rtt_ms_max = NAN
        msg.jitter_ms = NAN
        msg.lost_percent = NAN

        timeout = self._duration + self._omit + 10.0
        try:
            out = subprocess.run(
                self._build_cmd(),
                capture_output=True, text=True, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            msg.success = False
            msg.error = "iperf3 timed out"
            return msg
        except OSError as exc:
            msg.success = False
            msg.error = f"iperf3 launch failed: {exc}"
            return msg

        # iperf3 emits JSON on stdout even for most errors (-J).
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

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
bidirectional : bool   alternate direction on every test (periodic mode) or
                       every ``bidir_period_s`` segment (continuous mode), so
                       one instance measures uplink AND downlink. Each message
                       carries the direction in its ``reverse`` field. Chosen
                       over iperf3 --bidir because Wi-Fi is half-duplex:
                       simultaneous both-way traffic contends with itself and
                       neither number is a per-direction capacity. ``reverse``
                       picks the starting direction. Default False.
bidir_period_s : float continuous+bidirectional: seconds per direction before
                       swapping (each swap costs one reconnect). Default 10.
udp_bitrate_mbps : float  target rate for UDP tests in Mbit/s (0 = unlimited).
parallel : int         parallel TCP streams (iperf3 -P). Default 1; 4 fills a
                       Wi-Fi link far better and is recommended for capacity.
omit_s : float         initial seconds to omit (iperf3 -O). Default 1.0.
connect_timeout_ms : int  iperf3 --connect-timeout. Default 2000.
reconnect_poll_s : float  poll/backoff period when link is down or a test
                          fails. Default 3.0.
start_delay_s : float  delay before the first test. Use it to interleave two
                       instances (e.g. robot->server and robot->robot) so
                       their periodic tests never overlap: give both the same
                       interval_s and offset one by interval_s/2. Default 0.
continuous : bool      survey mode: keep ONE long iperf3 open and publish a
                       result every second from its interval reports (no
                       per-test connection overhead, true 1 Hz). Saturates the
                       link continuously, so it is a dedicated survey pass, not
                       for normal operation. In this mode duration_s / interval_s
                       / omit_s are ignored; jitter / loss are NaN.
rtt_via_ss : bool      in continuous mode, sample the live connection's TCP RTT
                       via `ss -ti` each second (same tcpi_rtt iperf reports) and
                       fill rtt_ms_mean/min/max -- dense RTT consistent with the
                       throughput, no ping. Default True; needs iproute2 (`ss`).
continuous_interval_s : float  reporting interval for continuous mode (iperf3
                       -i and the ss cadence). 1.0 = 1 Hz. 0.2-0.5 (2-5 Hz) is a
                       good moving-robot range; ~0.1 (10 Hz) works but is noisier.
"""

from __future__ import annotations

import math
import select
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
        # Alternate uplink/downlink from this one instance. Sequential (not
        # iperf3 --bidir) because Wi-Fi is half-duplex: simultaneous two-way
        # traffic contends with itself and measures neither direction cleanly.
        self.declare_parameter("bidirectional", False)
        # Continuous mode: seconds per direction before swapping.
        self.declare_parameter("bidir_period_s", 10.0)
        # UDP target rate in Mbit/s (0 = unlimited). Numeric so launch can
        # never mistype it; converted to iperf3's "<N>M" form internally.
        self.declare_parameter("udp_bitrate_mbps", 0.0)
        # Parallel TCP streams (iperf3 -P). >1 fills a Wi-Fi link much better
        # than a single stream; 4 is a good default for capacity tests.
        self.declare_parameter("parallel", 1)
        self.declare_parameter("omit_s", 1.0)
        self.declare_parameter("connect_timeout_ms", 2000)
        self.declare_parameter("reconnect_poll_s", 3.0)
        # Delay before the first test, so two instances (e.g. robot->server
        # and robot->robot) can interleave instead of saturating the same
        # airtime at the same moment.
        self.declare_parameter("start_delay_s", 0.0)
        # Continuous survey mode: keep ONE long iperf3 open and publish a
        # result every second from its interval reports (no per-test connection
        # overhead). Saturates the link continuously -> dedicated survey pass.
        self.declare_parameter("continuous", False)
        # In continuous mode, sample the live connection's TCP RTT via `ss`
        # each second and fill rtt_ms_* -- the same tcpi_rtt iperf reports, but
        # dense and consistent with the throughput (no ping needed). Ignored
        # if ss is unavailable or not in continuous mode.
        self.declare_parameter("rtt_via_ss", True)
        # Reporting interval for continuous mode (iperf3 -i, and the ss RTT
        # cadence). 1.0 = 1 Hz. Down to ~0.1 (10 Hz) is possible but noisier;
        # 0.2-0.5 (2-5 Hz) is a good range for a moving robot. Ignored outside
        # continuous mode.
        self.declare_parameter("continuous_interval_s", 1.0)

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
        self._bidir = gp("bidirectional").get_parameter_value().bool_value
        self._bidir_period = (
            gp("bidir_period_s").get_parameter_value().double_value
        )
        if self._bidir_period <= 0.0:
            self._bidir_period = 10.0
        # Direction of the *next* test; flips each round when bidirectional.
        self._cur_reverse = self._reverse
        self._udp_mbps = (
            gp("udp_bitrate_mbps").get_parameter_value().double_value
        )
        self._parallel = gp("parallel").get_parameter_value().integer_value
        self._omit = gp("omit_s").get_parameter_value().double_value
        self._connect_timeout = (
            gp("connect_timeout_ms").get_parameter_value().integer_value
        )
        self._reconnect_poll = (
            gp("reconnect_poll_s").get_parameter_value().double_value
        )
        self._start_delay = (
            gp("start_delay_s").get_parameter_value().double_value
        )
        self._continuous = gp("continuous").get_parameter_value().bool_value
        self._rtt_via_ss = gp("rtt_via_ss").get_parameter_value().bool_value
        self._cont_interval = (
            gp("continuous_interval_s").get_parameter_value().double_value
        )
        if self._cont_interval <= 0.0:
            self._cont_interval = 1.0
        self._ss_ok = shutil.which("ss") is not None
        if self._continuous and self._rtt_via_ss and not self._ss_ok:
            self.get_logger().warn(
                "rtt_via_ss set but 'ss' not found (apt install iproute2); "
                "continuous RTT will be NaN."
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
        mode = (
            f"continuous survey @ {1.0 / self._cont_interval:.1f} Hz"
            if self._continuous
            else f"{self._duration:.0f}s every {self._interval:.0f}s"
        )
        direction = (
            "bidirectional (alternating)"
            if self._bidir
            else ("downlink" if self._reverse else "uplink")
        )
        self.get_logger().info(
            f"iperf_runner -> {self._proto.upper()} to "
            f"'{self._server or '<unset>'}:{self._port}', {mode}, "
            f"{direction}, iface "
            f"'{self._iface or '<none>'}' -> topic '{self._pub.topic_name}'."
        )

    # ----------------------------------------------------------------------
    def _loop(self) -> None:
        poll = max(0.5, self._reconnect_poll)
        if self._start_delay > 0.0 and self._stop.wait(self._start_delay):
            return
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

            if self._continuous:
                # Blocks, publishing ~1 Hz until the iperf process exits
                # (link drop, error, or bidir segment end); then restart.
                healthy = self._run_continuous()
                if self._bidir:
                    self._cur_reverse = not self._cur_reverse
                # Quick swap between healthy bidir segments; otherwise back
                # off so a dead server is not hammered.
                wait = 0.2 if (self._bidir and healthy) else min(poll, 2.0)
                if self._stop.wait(wait):
                    break
                continue

            msg = self._run_once()
            self._pub.publish(msg)
            if self._bidir:
                self._cur_reverse = not self._cur_reverse

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
        msg.reverse = bool(self._cur_reverse)
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
        if self._parallel > 1:
            cmd += ["-P", str(self._parallel)]
        if self._cur_reverse:
            cmd += ["-R"]
        if self._proto == "udp":
            rate = f"{self._udp_mbps:g}M" if self._udp_mbps > 0 else "0"
            cmd += ["-u", "-b", rate]
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
            reason = str(parsed.get("error") or f"iperf3 exit {out.returncode}")
            stderr = (out.stderr or "").strip()
            if stderr:
                reason = f"{reason} | stderr: {stderr[:200]}"
            msg.success = False
            msg.error = reason
            self.get_logger().warn(f"iperf3 test failed: {reason}")
            return msg

        self._apply(msg, parsed)
        msg.success = True
        msg.error = ""
        arrow = "down" if self._cur_reverse else "up"
        self.get_logger().info(
            f"iperf {msg.protocol} {arrow}: {msg.bitrate_mbps:.1f} Mbit/s"
            + (f", loss {msg.lost_percent:.1f}%" if msg.protocol == "UDP"
               else (f", rtt {msg.rtt_ms_mean:.1f} ms"
                     if not math.isnan(msg.rtt_ms_mean) else ""))
        )
        return msg

    # ----------------------------------------------------------------------
    def _build_continuous_cmd(self) -> list:
        # -t 0 runs until we kill it; -i 1 emits a report every second.
        # Bidirectional: run bidir_period_s per segment instead, so the loop
        # restarts us with the direction flipped.
        seg = f"{self._bidir_period:g}" if self._bidir else "0"
        cmd = [
            "iperf3", "-c", self._server, "-p", str(self._port),
            "-t", seg, "-i", f"{self._cont_interval:g}", "--forceflush",
        ]
        if self._connect_timeout > 0:
            cmd += ["--connect-timeout", str(self._connect_timeout)]
        if self._parallel > 1:
            cmd += ["-P", str(self._parallel)]
        if self._cur_reverse:
            cmd += ["-R"]
        if self._proto == "udp":
            rate = f"{self._udp_mbps:g}M" if self._udp_mbps > 0 else "0"
            cmd += ["-u", "-b", rate]
        return cmd

    def _sample_ss_rtt(self):
        """Read the live iperf connection's TCP RTT via `ss` (dict or None)."""
        try:
            out = subprocess.run(
                ["ss", "-tin", "dst", f"{self._server}:{self._port}"],
                capture_output=True, text=True, timeout=2.0, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return iperf_parse.parse_ss_rtt(out.stdout or "")

    def _run_continuous(self) -> bool:
        """Run one long iperf3 and publish an IperfResult per interval line.

        Returns when the process exits or the node stops (True if at least
        one interval was published, so the caller can distinguish a healthy
        bidir segment end from a dead server). A 1 s select() watchdog means
        a mid-run link drop is noticed within a second so the loop can
        restart against the reconnected link.
        """
        try:
            proc = subprocess.Popen(
                self._build_continuous_cmd(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
        except OSError as exc:
            self._pub.publish(
                self._new_msg(success=False, error=f"iperf3 launch failed: {exc}")
            )
            return False

        arrow = "down" if self._cur_reverse else "up"
        seg = f" ({self._bidir_period:g}s segment)" if self._bidir else ""
        self.get_logger().info(
            f"continuous iperf {self._proto.upper()} {arrow} started{seg}."
        )
        published = False
        try:
            while not self._stop.is_set() and proc.poll() is None:
                rlist, _, _ = select.select([proc.stdout], [], [], 1.0)
                if not rlist:
                    # No output for 1 s: bail out if the link dropped.
                    if self._iface and wifi_parsers.link_up(self._iface) is False:
                        break
                    continue
                line = proc.stdout.readline()
                if line == "":
                    break  # EOF: process ended
                parsed = iperf_parse.parse_interval_line(line, self._parallel)
                if parsed is None:
                    continue
                msg = self._new_msg(success=True)
                msg.bitrate_mbps = float(parsed["mbps"])
                msg.bytes = int(parsed.get("bytes", 0))
                msg.duration_s = float(parsed.get("seconds", 1.0))
                if "retransmits" in parsed:
                    msg.retransmits = int(parsed["retransmits"])
                if self._rtt_via_ss and self._ss_ok:
                    rtt = self._sample_ss_rtt()
                    if rtt is not None:
                        msg.rtt_ms_mean = rtt["rtt_ms_mean"]
                        msg.rtt_ms_min = rtt["rtt_ms_min"]
                        msg.rtt_ms_max = rtt["rtt_ms_max"]
                self._pub.publish(msg)
                published = True
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        return published

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

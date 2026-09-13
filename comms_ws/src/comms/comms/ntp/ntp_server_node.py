#!/usr/bin/env python3
"""
ntp_server_node.py
──────────────────
ROS 2 Humble node — publishes NTP server metrics for rosbag / distributed
sensor network alignment.

Topics published
  <ns>/ntp/server/status      (comms_msgs/NtpStatus)   1 Hz default
  <ns>/ntp/clients/<ip>/status (comms_msgs/NtpStatus)   1 Hz, one per client
  <ns>/ntp/events             (std_msgs/String)               on clock-step only

Parameters
  publish_rate    Hz  (default 1.0 — server is reference, not sensor-aligned)
  hostname        override auto-detected hostname
  count_clients   bool, count connected NTP clients (default True)
"""

import re
import socket
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String

from comms_msgs.msg import NtpStatus
from comms.ntp.ntp_parser import (
    get_ntp_metrics,
    get_clock_environment,
    get_connected_clients,
    parse_chronyc_clients,
)


def _now(node: Node):
    return node.get_clock().now().to_msg()


def _topic_token(name: str) -> str:
    """One ROS topic segment from an address or hostname.

    A segment must match [A-Za-z_][A-Za-z0-9_]*, so dots, colons and hyphens
    have to go and a leading digit needs a prefix: 192.168.25.31 becomes
    ip_192_168_25_31, wicoms-robot1 becomes wicoms_robot1."""
    t = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return t if t[:1].isalpha() or t[:1] == "_" else "ip_" + t


class NtpServerNode(Node):

    TOPIC       = "ntp/server/status"
    EVENT_TOPIC = "ntp/events"

    def __init__(self):
        super().__init__("ntp_server_node")

        # parameters
        self.declare_parameter("publish_rate",  1.0)
        self.declare_parameter("hostname",      socket.gethostname())
        self.declare_parameter("count_clients", True)

        rate_hz: float        = self.get_parameter("publish_rate").value
        self._hostname: str   = self.get_parameter("hostname").value
        self._count_clients   = self.get_parameter("count_clients").value

        # QoS: RELIABLE + TRANSIENT_LOCAL — rosbag gets last state immediately
        # on subscription, and no messages are silently dropped on the network
        qos = QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._qos = qos   # saved so _publish_client_views can reuse it

        self._pub       = self.create_publisher(NtpStatus, self.TOPIC,      qos)
        self._event_pub = self.create_publisher(String,    self.EVENT_TOPIC, qos)

        # per-client publishers created dynamically as new clients are seen
        self._client_pubs: dict = {}   # ip -> Publisher

        # internal state
        self._cached_metrics: dict  = {}
        self._prev_offset:    float = 0.0
        self._prev_monotonic: float = 0.0
        self._seq:            int   = 0

        # Slow timer: daemon query at 1 Hz
        self._cache_timer = self.create_timer(1.0, self._refresh_cache)

        # Publish timer — default 1 Hz for server (change in yaml if needed)
        period = 1.0 / max(rate_hz, 0.01)
        self._pub_timer = self.create_timer(period, self._publish)

        self.get_logger().info(
            f"NTP server node | host={self._hostname} "
            f"publish={rate_hz} Hz cache=1 Hz "
            f"count_clients={self._count_clients} -> {self.TOPIC}"
        )

    # daemon cache

    def _refresh_cache(self) -> None:
        self._cached_metrics = get_ntp_metrics()
        self._cached_metrics.update(get_clock_environment())

    # per-client topics

    def _publish_client_views(self) -> None:
        """
        Query chronyc clients -v and publish one NtpStatus topic per connected
        client IP.  The server sees each robot's offset from its own reference,
        giving a second independent measurement for cross-validation in post.
        """
        for client in parse_chronyc_clients():
            ip    = client["ip"]
            topic = f"ntp/clients/{_topic_token(ip)}/status"

            if ip not in self._client_pubs:
                self._client_pubs[ip] = self.create_publisher(
                    NtpStatus, topic, self._qos
                )
                self.get_logger().info(f"New NTP client seen: {ip} -> {topic}")

            # `chronyc clients` reports ACTIVITY, not clock error: packets,
            # drops, the client's own poll interval and how long since it last
            # asked. A server cannot know a client's offset, so those fields
            # stay zero and the warning says why.
            msg = NtpStatus()
            msg.header.stamp    = _now(self)
            msg.header.frame_id = self._hostname
            msg.role            = "client_view_from_server"
            msg.hostname        = ip
            msg.poll_interval_seconds = int(client["poll_interval_seconds"])
            msg.connected_clients     = -1
            msg.leap_indicator        = "no_warning"
            msg.warnings              = [
                f"activity only: {client['ntp_packets']} packets, "
                f"{client['ntp_dropped']} dropped, last seen "
                f"{client['last_seen_seconds']} s ago; a server cannot measure "
                "a client's offset"]
            self._client_pubs[ip].publish(msg)

    # main publish callback

    def _publish(self) -> None:
        if not self._cached_metrics:
            return

        metrics  = self._cached_metrics
        now_mono = time.monotonic()

        elapsed      = now_mono - self._prev_monotonic if self._prev_monotonic else 0.0
        offset_delta = metrics["offset_seconds"] - self._prev_offset
        max_slew     = elapsed * 0.0005
        stepped      = elapsed > 0 and abs(offset_delta) > max_slew

        msg = NtpStatus()

        msg.header.stamp    = _now(self)
        msg.header.frame_id = self._hostname

        msg.role     = "server"
        msg.hostname = self._hostname

        msg.synchronized  = metrics["synchronized"]
        msg.sync_source   = metrics["sync_source"]
        msg.stratum_level = metrics["stratum_level"]
        msg.stratum       = metrics["stratum"]

        msg.offset_seconds      = metrics["offset_seconds"]
        msg.delay_seconds       = metrics["delay_seconds"]
        msg.jitter_seconds      = metrics["jitter_seconds"]
        msg.root_delay          = metrics["root_delay"]
        msg.root_dispersion     = metrics["root_dispersion"]
        msg.frequency_error_ppm = metrics["frequency_error_ppm"]
        msg.last_offset_seconds = float(metrics.get("last_offset_seconds", 0.0))
        msg.skew_ppm            = float(metrics.get("skew_ppm", 0.0))
        msg.fit_samples         = int(metrics.get("fit_samples", 0))
        msg.rms_offset_seconds      = float(metrics.get("rms_offset_seconds", 0.0))
        msg.residual_freq_ppm       = float(metrics.get("residual_freq_ppm", 0.0))
        msg.update_interval_seconds = float(metrics.get("update_interval_seconds", 0.0))
        msg.fit_span_seconds        = float(metrics.get("fit_span_seconds", 0.0))
        msg.temperature_c           = float(metrics.get("temperature_c", float("nan")))
        msg.cpu_load_1min           = float(metrics.get("cpu_load_1min", float("nan")))

        msg.poll_interval_seconds = metrics["poll_interval_seconds"]
        msg.reach_register        = metrics["reach_register"]
        msg.reachability_percent  = metrics["reachability_percent"]

        rt = metrics.get("ref_time_epoch")
        if rt:
            msg.reference_time.sec = int(rt)
            msg.reference_time.nanosec = int((rt - int(rt)) * 1e9)
        else:
            msg.reference_time = _now(self)
        msg.connected_clients = (
            get_connected_clients() if self._count_clients else -1
        )

        msg.leap_indicator = metrics["leap_indicator"]
        msg.warnings       = list(metrics["warnings"])

        if msg.stratum > 3:
            msg.warnings.append(f"High stratum ({msg.stratum}) — verify upstream")
        if msg.connected_clients == 0:
            msg.warnings.append("Server has 0 connected clients")

        # rosbag alignment fields
        msg.seq                  = self._seq
        msg.monotonic_seconds    = now_mono
        msg.clock_stepped        = stepped
        msg.offset_delta_seconds = offset_delta

        self._pub.publish(msg)

        # publish one topic per known client (server-side view)
        self._publish_client_views()

        if stepped:
            event = (
                f"STEP host={self._hostname} "
                f"delta={offset_delta*1000:.3f}ms "
                f"mono={now_mono:.3f}"
            )
            self._event_pub.publish(String(data=event))
            self.get_logger().error(event)

        self._prev_offset    = metrics["offset_seconds"]
        self._prev_monotonic = now_mono
        self._seq           += 1

        self.get_logger().info(
            f"stratum={msg.stratum} | "
            f"offset={msg.offset_seconds*1e3:.3f} ms | "
            f"jitter={msg.jitter_seconds*1e3:.3f} ms | "
            f"clients={msg.connected_clients} | "
            f"stepped={stepped}"
        )

        if not metrics["synchronized"]:
            self.get_logger().error(
                "Server NOT synchronised upstream — downstream clients degraded | "
                f"leap={msg.leap_indicator}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = NtpServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

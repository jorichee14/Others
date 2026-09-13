#!/usr/bin/env python3
"""
ntp_client_node.py
──────────────────
ROS 2 Humble node — publishes NTP client metrics for rosbag / distributed
sensor network alignment.

Topics published
  /ntp/client/status   (comms_msgs/NtpStatus)  10 Hz default
  /ntp/events          (std_msgs/String)              on clock-step only

Parameters
  publish_rate   Hz for the fast publish timer  (default 10.0)
  hostname       override auto-detected hostname
"""

import socket
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String

from comms_msgs.msg import NtpStatus
from comms.ntp.ntp_parser import get_ntp_metrics, get_clock_environment


def _now(node: Node):
    return node.get_clock().now().to_msg()


class NtpClientNode(Node):

    TOPIC = "/ntp/client/status"
    EVENT_TOPIC = "/ntp/events"

    def __init__(self):
        super().__init__("ntp_client_node")

        # parameters
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("hostname", socket.gethostname())

        rate_hz: float = self.get_parameter("publish_rate").value
        self._hostname: str = self.get_parameter("hostname").value

        # QoS: RELIABLE + TRANSIENT_LOCAL so rosbag never misses a message
        # and gets the last cached state immediately on subscription
        qos = QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._pub       = self.create_publisher(NtpStatus, self.TOPIC,       qos)
        self._event_pub = self.create_publisher(String,    self.EVENT_TOPIC,  qos)

        # internal state
        self._cached_metrics: dict  = {}
        self._cache_stamp           = None
        self._prev_offset:    float = 0.0
        self._prev_monotonic: float = 0.0
        self._seq:            int   = 0

        # Slow timer: shell out to chronyc/ntpq at 1 Hz (daemon poll rate)
        self._cache_timer = self.create_timer(1.0, self._refresh_cache)

        # Fast timer: publish cached data at sensor-aligned rate
        period = 1.0 / max(rate_hz, 0.01)
        self._pub_timer = self.create_timer(period, self._publish)

        self.get_logger().info(
            f"NTP client node | host={self._hostname} "
            f"publish={rate_hz} Hz cache=1 Hz -> {self.TOPIC}"
        )

    def _refresh_cache(self) -> None:
        """Hit the NTP daemon (slow -- 1 Hz). Never called at publish rate.

        The cache time is recorded so reference_time can carry it. The
        original stamped reference_time with the publish time, which made it
        change on every message and useless for counting daemon polls.
        """
        self._cached_metrics = get_ntp_metrics()
        self._cached_metrics.update(get_clock_environment())
        self._cache_stamp = _now(self)

    def _publish(self) -> None:
        if not self._cached_metrics:
            return   # wait for first cache fill (~1 s after start)

        metrics  = self._cached_metrics
        now_mono = time.monotonic()

        # clock-step detection
        # NTP slew cap is 500 ppm = 0.5 ms/s. Any offset change faster than
        # that between two publishes indicates a step (sudden correction).
        elapsed      = now_mono - self._prev_monotonic if self._prev_monotonic else 0.0
        offset_delta = metrics["offset_seconds"] - self._prev_offset
        max_slew     = elapsed * 0.0005
        stepped      = elapsed > 0 and abs(offset_delta) > max_slew

        msg = NtpStatus()

        # header — fresh stamp every publish even though metrics are cached
        msg.header.stamp    = _now(self)
        msg.header.frame_id = self._hostname

        msg.role     = "client"
        msg.hostname = self._hostname

        msg.synchronized  = metrics["synchronized"]
        msg.sync_source   = metrics["sync_source"]
        msg.stratum_level = metrics["stratum_level"]
        msg.stratum       = metrics["stratum"]

        # offset: NEGATIVE = local clock behind NTP. Steered estimate
        # (chrony "System time"). The measured per-poll value is in the
        # warnings-free extras below since the message has no field for it.
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

        # Chrony's own poll time when it parsed; else the query time, which
        # is NOT a poll time and must not be used to count polls.
        rt = metrics.get("ref_time_epoch")
        if rt:
            msg.reference_time.sec = int(rt)
            msg.reference_time.nanosec = int((rt - int(rt)) * 1e9)
        else:
            msg.reference_time = self._cache_stamp or _now(self)
        msg.connected_clients = -1

        msg.leap_indicator = metrics["leap_indicator"]
        msg.warnings       = list(metrics["warnings"])

        # rosbag alignment fields
        msg.seq                  = self._seq
        msg.monotonic_seconds    = now_mono
        msg.clock_stepped        = stepped
        msg.offset_delta_seconds = offset_delta

        self._pub.publish(msg)

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

        if not metrics["synchronized"]:
            self.get_logger().warn(
                f"Clock NOT synchronised | stratum={metrics['stratum']} "
                f"leap={metrics['leap_indicator']}"
            )
        else:
            self.get_logger().debug(
                f"offset={metrics['offset_seconds']*1000:.3f} ms  "
                f"jitter={metrics['jitter_seconds']*1000:.3f} ms  "
                f"src={metrics['sync_source']}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = NtpClientNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

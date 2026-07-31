#!/usr/bin/env python3
"""ROS 2 node that periodically publishes wireless link statistics.

Publishes a :class:`wifi_monitor_msgs/msg/WifiLinkStatus` at a fixed rate
containing RSSI, SNR, link quality, bit rate and the RX/TX traffic counters
for one wireless interface. A companion ``diagnostic_msgs/DiagnosticArray``
is also published so the data shows up in ``rqt_robot_monitor``.

Parameters
----------
interface : str
    Wireless interface to monitor (e.g. ``wlx8876b9eae0ff``). If empty, the
    first interface exposing ``/sys/class/net/<iface>/wireless`` is used.
publish_rate_hz : float
    Sampling / publishing rate. Default 1.0 Hz.
frame_id : str
    ``header.frame_id`` stamped on every message. Default ``wifi``.
warn_signal_dbm : float
    RSSI at or below which diagnostics report WARN. Default -70.
error_signal_dbm : float
    RSSI at or below which diagnostics report ERROR. Default -80.
"""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

from wifi_monitor_msgs.msg import WifiLinkStatus

from wifi_monitor import wifi_parsers

NAN = float("nan")


class WifiMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("wifi_monitor")

        self.declare_parameter("interface", "")
        self.declare_parameter("publish_rate_hz", 1.0)
        self.declare_parameter("frame_id", "wifi")
        self.declare_parameter("warn_signal_dbm", -70.0)
        self.declare_parameter("error_signal_dbm", -80.0)

        self._interface = (
            self.get_parameter("interface").get_parameter_value().string_value
        )
        rate = (
            self.get_parameter("publish_rate_hz")
            .get_parameter_value()
            .double_value
        )
        if rate <= 0.0:
            rate = 1.0
        self._frame_id = (
            self.get_parameter("frame_id").get_parameter_value().string_value
        )
        self._warn_dbm = (
            self.get_parameter("warn_signal_dbm")
            .get_parameter_value()
            .double_value
        )
        self._error_dbm = (
            self.get_parameter("error_signal_dbm")
            .get_parameter_value()
            .double_value
        )

        if not self._interface:
            found = wifi_parsers.list_wireless_interfaces()
            if found:
                self._interface = found[0]
                self.get_logger().info(
                    f"No 'interface' set; auto-selected '{self._interface}'."
                )
            else:
                self.get_logger().warn(
                    "No wireless interface found under /sys/class/net; "
                    "set the 'interface' parameter."
                )

        qos = QoSProfile(
            depth=10,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._pub = self.create_publisher(WifiLinkStatus, "wifi/status", qos)
        self._diag_pub = self.create_publisher(
            DiagnosticArray, "diagnostics", qos
        )

        self._timer = self.create_timer(1.0 / rate, self._on_timer)
        self.get_logger().info(
            f"wifi_monitor running on '{self._interface or '<none>'}' "
            f"at {rate:.1f} Hz -> topic 'wifi/status'."
        )

    # ----------------------------------------------------------------------
    def _on_timer(self) -> None:
        msg = WifiLinkStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.interface = self._interface

        # Float fields default to 0.0 in ROS; use NaN to mean "unknown".
        msg.frequency_ghz = NAN
        msg.bit_rate_mbps = NAN
        msg.tx_power_dbm = NAN
        msg.link_quality_ratio = NAN
        msg.signal_dbm = NAN
        msg.noise_dbm = NAN
        msg.snr_db = NAN
        msg.channel = -1

        if self._interface:
            data = wifi_parsers.collect_all(self._interface)
            self._apply(msg, data)

        self._pub.publish(msg)
        self._diag_pub.publish(self._to_diagnostics(msg))

    # ----------------------------------------------------------------------
    @staticmethod
    def _apply(msg: WifiLinkStatus, data: dict) -> None:
        """Copy parsed values onto the message, respecting field types."""
        str_fields = ("mac_address", "essid", "bssid", "mode")
        for f in str_fields:
            if f in data:
                setattr(msg, f, str(data[f]))

        bool_fields = ("up", "associated", "noise_valid")
        for f in bool_fields:
            if f in data:
                setattr(msg, f, bool(data[f]))

        float_fields = (
            "frequency_ghz",
            "bit_rate_mbps",
            "tx_power_dbm",
            "link_quality_ratio",
            "signal_dbm",
            "noise_dbm",
            "snr_db",
        )
        for f in float_fields:
            if f in data:
                setattr(msg, f, float(data[f]))

        int32_fields = ("channel", "link_quality", "link_quality_max")
        for f in int32_fields:
            if f in data:
                setattr(msg, f, int(data[f]))

        int64_fields = (
            "rx_invalid_nwid",
            "rx_invalid_crypt",
            "rx_invalid_frag",
            "tx_excessive_retries",
            "invalid_misc",
            "missed_beacon",
        )
        for f in int64_fields:
            if f in data:
                setattr(msg, f, int(data[f]))

        uint64_fields = (
            "rx_packets",
            "rx_bytes",
            "rx_errors",
            "rx_dropped",
            "rx_overruns",
            "rx_frame_errors",
            "tx_packets",
            "tx_bytes",
            "tx_errors",
            "tx_dropped",
            "tx_overruns",
            "tx_carrier_errors",
            "collisions",
        )
        for f in uint64_fields:
            if f in data:
                setattr(msg, f, int(data[f]))

    # ----------------------------------------------------------------------
    def _to_diagnostics(self, msg: WifiLinkStatus) -> DiagnosticArray:
        arr = DiagnosticArray()
        arr.header = msg.header
        st = DiagnosticStatus()
        st.name = f"wifi_monitor: {msg.interface or 'unknown'}"
        st.hardware_id = msg.mac_address

        sig = msg.signal_dbm
        if not msg.associated:
            st.level = DiagnosticStatus.ERROR
            st.message = "Not associated"
        elif isinstance(sig, float) and math.isnan(sig):
            st.level = DiagnosticStatus.WARN
            st.message = "Associated; signal unknown"
        elif sig <= self._error_dbm:
            st.level = DiagnosticStatus.ERROR
            st.message = f"Very weak signal ({sig:.0f} dBm)"
        elif sig <= self._warn_dbm:
            st.level = DiagnosticStatus.WARN
            st.message = f"Weak signal ({sig:.0f} dBm)"
        else:
            st.level = DiagnosticStatus.OK
            st.message = f"OK ({sig:.0f} dBm)"

        def kv(key: str, val) -> KeyValue:
            return KeyValue(key=key, value=str(val))

        st.values = [
            kv("essid", msg.essid),
            kv("bssid", msg.bssid),
            kv("frequency_ghz", f"{msg.frequency_ghz:.3f}"),
            kv("channel", msg.channel),
            kv("bit_rate_mbps", f"{msg.bit_rate_mbps:.1f}"),
            kv("tx_power_dbm", f"{msg.tx_power_dbm:.0f}"),
            kv("signal_dbm", f"{msg.signal_dbm:.0f}"),
            kv("noise_dbm", f"{msg.noise_dbm:.0f}"),
            kv("snr_db", f"{msg.snr_db:.1f}"),
            kv("link_quality", f"{msg.link_quality}/{msg.link_quality_max}"),
            kv("missed_beacon", msg.missed_beacon),
            kv("tx_excessive_retries", msg.tx_excessive_retries),
            kv("rx_packets", msg.rx_packets),
            kv("tx_packets", msg.tx_packets),
            kv("rx_errors", msg.rx_errors),
            kv("tx_errors", msg.tx_errors),
            kv("rx_dropped", msg.rx_dropped),
            kv("tx_dropped", msg.tx_dropped),
        ]
        arr.status = [st]
        return arr


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WifiMonitorNode()
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

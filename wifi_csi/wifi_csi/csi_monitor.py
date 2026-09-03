"""ROS 2 node: print a live summary of the CSI stream.

Subscribes rather than touching the radio, so it is safe to run on the Orin
against a remote node, or alongside a recording without disturbing it.

    ros2 run wifi_csi csi_monitor --ros-args -r __ns:=/csi_publisher
"""
from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from wifi_csi_msgs.msg import CsiFrame, CsiStatus

SHADES = " .:-=+*#%@"


class CsiMonitor(Node):
    def __init__(self) -> None:
        super().__init__("csi_monitor")
        self.declare_parameter("topic", "/csi_publisher/csi")
        self.declare_parameter("status_topic", "/csi_publisher/status")
        self.declare_parameter("window", 200)
        self.declare_parameter("period", 2.0)

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=100)
        self.create_subscription(CsiFrame, self.get_parameter("topic").value,
                                 self._on_frame, qos)
        self.create_subscription(CsiStatus, self.get_parameter("status_topic").value,
                                 self._on_status, 1)
        self.buf: list[np.ndarray] = []
        self.idx: np.ndarray | None = None
        self.window = int(self.get_parameter("window").value)
        self.last = None
        self.n = 0
        self.t0 = time.monotonic()
        self.create_timer(float(self.get_parameter("period").value), self._report)

    def _on_status(self, msg: CsiStatus) -> None:
        self.last = msg

    def _on_frame(self, msg: CsiFrame) -> None:
        self.n += 1
        z = np.asarray(msg.csi_real, np.float32) + 1j * np.asarray(msg.csi_imag, np.float32)
        if self.idx is None or len(msg.subcarrier_index) != self.idx.size:
            self.idx = np.asarray(msg.subcarrier_index, np.int32)
            self.buf = []
        self.buf.append(z)
        if len(self.buf) > self.window:
            self.buf.pop(0)

    @staticmethod
    def _spark(v: np.ndarray, width: int = 72) -> str:
        v = np.asarray(v, float)
        if v.size == 0:
            return ""
        v = v[np.linspace(0, v.size - 1, min(width, v.size)).astype(int)]
        lo, hi = v.min(), v.max()
        if hi <= lo:
            return SHADES[0] * v.size
        q = ((v - lo) / (hi - lo) * (len(SHADES) - 1)).clip(0, len(SHADES) - 1)
        return "".join(SHADES[int(round(x))] for x in q)

    def _report(self) -> None:
        dt = time.monotonic() - self.t0
        rate = self.n / dt if dt > 0 else 0.0
        self.t0, self.n = time.monotonic(), 0

        if self.last is not None:
            s = self.last
            fw_ok = "nexmon" in s.firmware.lower()
            self.get_logger().info(
                f"{s.interface} ch{s.channel}/{s.bandwidth_mhz} "
                f"monitor={int(s.monitor_mode)} fw={'nexmon' if fw_ok else 'STOCK?'} "
                f"filter={s.mac_filter or 'none'} "
                f"total={s.frames_total} dropped={s.dropped_total}")

        if not self.buf:
            self.get_logger().warning(
                f"{rate:.0f} frames/s — nothing to summarise. Either no OFDM "
                "traffic from the filtered source, or the extractor needs re-arming.")
            return

        X = np.stack(self.buf)
        amp = np.abs(X)
        mean = amp.mean(0)
        # Relative temporal variation is the number that matters for sensing:
        # a static room sits near 0.004, motion pushes it an order higher.
        motion = float((np.abs(amp - mean) / np.maximum(mean, 1e-9)).mean())
        self.get_logger().info(
            f"{rate:6.1f} Hz  n={X.shape[0]:4d}x{X.shape[1]:3d}  "
            f"sub {self.idx.min()}..{self.idx.max()}  "
            f"|CSI| {mean.min():.0f}-{mean.max():.0f}  motion={motion:.4f}")
        self.get_logger().info("  profile " + self._spark(mean))
        self.get_logger().info("  tempstd " + self._spark(amp.std(0)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CsiMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

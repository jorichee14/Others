"""ROS 2 node: arm the nexmon extractor and publish CSI from UDP 5500.

Uses the same commands verified by hand:
    makecsiparams -c <chan>/<bw> -C 1 -N 1 [-m <mac>]
    nexutil -I<iface> -s500 -b -l34 -v<params>
    nexutil -I<iface> -m1

Reads the firmware's UDP datagrams directly from a socket rather than shelling
out to tcpdump, so there is no pipe to buffer and no external process to babysit.
"""
from __future__ import annotations

import re
import socket
import subprocess
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from wifi_csi_msgs.msg import CsiFrame, CsiStatus

from .csi_parser import (DEFAULT_PORT, ParseError, decode_chanspec,
                         parse_frame, usable_slots)


class CsiPublisher(Node):
    def __init__(self) -> None:
        super().__init__("csi_publisher")

        self.declare_parameter("interface", "wlan0")
        self.declare_parameter("channel", "36/80")
        self.declare_parameter("mac_filter", "")       # empty = all sources
        self.declare_parameter("core_mask", 1)
        self.declare_parameter("nss_mask", 1)
        self.declare_parameter("frame_control_filter", "")  # e.g. "0x88"
        self.declare_parameter("port", DEFAULT_PORT)
        self.declare_parameter("arm_on_start", True)
        self.declare_parameter("arm_attempts", 5)
        self.declare_parameter("trim", True)           # drop constant fields + DC
        self.declare_parameter("calib_frames", 300)    # batch size for finding them
        self.declare_parameter("frame_id", "wlan0")
        self.declare_parameter("status_period", 2.0)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.iface = g("interface")
        self.chan = g("channel")
        self.mac = g("mac_filter")
        self.mac_list = [m.strip().lower() for m in self.mac.split(",") if m.strip()]
        self.trim = g("trim")
        self.frame_id = g("frame_id")
        self.port = int(g("port"))

        # Sensor-style QoS: best-effort and shallow. CSI arrives at hundreds of
        # Hz and a late frame is worthless, so dropping beats queueing.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=100,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.pub = self.create_publisher(CsiFrame, "~/csi", qos)
        self.pub_status = self.create_publisher(CsiStatus, "~/status", 1)
        # One namespace per filtered transmitter, numbered by position in
        # mac_filter, so each mobile can be subscribed or bagged on its own.
        # The combined ~/csi stream is still published for anything that wants
        # every frame in arrival order.
        self.pub_by_mac = {
            mac: self.create_publisher(CsiFrame, f"/mobile{i}/csi", qos)
            for i, mac in enumerate(self.mac_list, start=1)
        }

        self.n_frames = 0
        self.n_dropped = 0
        # Constant firmware fields can only be identified across many frames, so
        # the node calibrates on the first batch and publishes untrimmed until
        # then. _keep_for guards against the slot count changing mid-run, which
        # happens when transmitters of different bandwidth share the channel.
        self._calib: list = []
        self._calib_n = int(self.get_parameter("calib_frames").value)
        self._keep = None
        self._keep_for = 0
        self._window_start = time.monotonic()
        self._window_count = 0
        self._rate = 0.0
        self._firmware = self._read_firmware_banner()

        if "nexmon" not in self._firmware.lower():
            self.get_logger().error(
                f"firmware banner does not mention nexmon: {self._firmware!r} — "
                "the patched blob is probably not loaded (update-alternatives --set, "
                "then reload brcmfmac)")

        if g("arm_on_start"):
            self._arm(int(g("arm_attempts")))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.setblocking(False)

        # Poll fast enough to keep the socket drained at ~500 frames/s.
        self.create_timer(0.002, self._drain)
        self.create_timer(float(g("status_period")), self._publish_status)
        self.get_logger().info(
            f"listening on UDP {self.port}, iface={self.iface}, chan={self.chan}, "
            f"mac_filter={self.mac or '(none)'}")
        for i, mac in enumerate(self.mac_list, start=1):
            self.get_logger().info(f"  /mobile{i}/csi <- {mac}")

    # ---------------------------------------------------------------- arming
    def _sh(self, cmd: list[str]) -> str:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return (r.stdout or "") + (r.stderr or "")
        except (OSError, subprocess.SubprocessError) as exc:
            self.get_logger().warning(f"{' '.join(cmd)}: {exc}")
            return ""

    def _read_firmware_banner(self) -> str:
        """dmesg needs root on Ubuntu 22.04, so fall back to the alternatives
        symlink, which is world-readable and names the selected blob."""
        import os
        out = self._sh(["dmesg"])
        hits = re.findall(r"Firmware: BCM.*", out)
        if hits:
            return hits[-1]
        try:
            return "selected blob: " + os.path.realpath(
                "/lib/firmware/brcm/brcmfmac43455-sdio.bin")
        except OSError:
            return "unknown"

    def _make_params(self) -> str:
        argv = ["makecsiparams", "-c", self.chan,
                "-C", str(self.get_parameter("core_mask").value),
                "-N", str(self.get_parameter("nss_mask").value)]
        for mac in self.mac_list:
            argv += ["-m", mac]
        fc = self.get_parameter("frame_control_filter").value
        if fc:
            # -b filters on the frame's FIRST BYTE (frame control), not the MAC.
            # 0x88 = QoS Data. On a channel dominated by Block Acks (0x94) this
            # rejects almost everything, so leave it unset unless you mean it.
            argv += ["-b", fc]
        out = self._sh(argv).strip()
        if not out:
            raise RuntimeError("makecsiparams produced no output")
        return out.splitlines()[-1].strip()

    def _arm(self, attempts: int) -> None:
        """Configure the extractor, retrying until packets actually appear.

        The extractor can report monitor=1 with the right chanspec and still
        emit nothing if -s500 lands before the interface has settled after an
        SDIO probe. Observed twice on cold boot. Reissuing fixes it.
        """
        params = self._make_params()
        for attempt in range(1, attempts + 1):
            self._sh(["nexutil", f"-I{self.iface}", "-s500", "-b", "-l34",
                      f"-v{params}"])
            self._sh(["nexutil", f"-I{self.iface}", "-m1"])
            time.sleep(1.0)
            if self._probe_once():
                self.get_logger().info(f"extractor armed on attempt {attempt}")
                return
            time.sleep(2.0)
        self.get_logger().warning(
            f"armed but no CSI seen after {attempts} attempts — "
            "channel may simply be idle; check with an unfiltered tcpdump")

    def _probe_once(self, timeout: float = 2.0) -> bool:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.settimeout(timeout)
        try:
            s.bind(("0.0.0.0", self.port))
            s.recvfrom(4096)
            return True
        except (socket.timeout, OSError):
            return False
        finally:
            s.close()

    # ------------------------------------------------------------- publishing
    def _drain(self) -> None:
        for _ in range(256):          # bounded so one timer tick can't stall
            try:
                data, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                return
            except OSError:
                return
            try:
                fr = parse_frame(data)
            except ParseError:
                self.n_dropped += 1
                continue
            if self.mac_list and fr.src_mac not in self.mac_list:
                continue
            if self.trim and self._keep is None:
                self._collect_calibration(fr)
            msg = self._to_msg(fr)
            self.pub.publish(msg)
            per_mac = self.pub_by_mac.get(fr.src_mac)
            if per_mac is not None:
                per_mac.publish(msg)
            self.n_frames += 1
            self._window_count += 1

    def _to_msg(self, fr) -> CsiFrame:
        m = CsiFrame()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frame_id
        m.src_mac = fr.src_mac
        m.rssi = int(fr.rssi)
        m.frame_control = int(fr.frame_control)
        m.seq = int(fr.seq)
        m.core = int(fr.core)
        m.spatial_stream = int(fr.spatial_stream)
        m.chanspec = int(fr.chanspec)
        m.chip_version = int(fr.chip_version)
        ch, bw = decode_chanspec(fr.chanspec)
        m.channel, m.bandwidth_mhz = int(ch), int(bw)
        m.raw_slots = int(fr.raw_slots)

        if self.trim and self._keep is not None and fr.csi.size == self._keep_for:
            idx, vals = fr.select(self._keep)
            m.trimmed = True
        else:
            idx = np.arange(fr.csi.size)
            vals = fr.csi
            m.trimmed = False

        m.subcarrier_index = idx.astype(np.int32).tolist()
        m.csi_real = np.real(vals).astype(np.float32).tolist()
        m.csi_imag = np.imag(vals).astype(np.float32).tolist()
        return m

    def _collect_calibration(self, fr) -> None:
        if self._calib and fr.csi.size != self._calib[0].size:
            return                       # different bandwidth; ignore for calibration
        self._calib.append(fr.csi)
        if len(self._calib) < self._calib_n:
            return
        try:
            keep, const = usable_slots(self._calib)
        except ParseError as exc:
            self.get_logger().warning(f"calibration failed ({exc}); publishing untrimmed")
            self._calib, self._keep_for = [], 0
            self._keep = np.arange(self._calib[0].size) if self._calib else None
            return
        self._keep, self._keep_for = keep, self._calib[0].size
        self.get_logger().info(
            f"calibrated on {len(self._calib)} frames: publishing slots "
            f"{keep.min()}..{keep.max()} ({keep.size} subcarriers); "
            f"{const.size} constant/empty slots excluded")
        self._calib = []

    def _publish_status(self) -> None:
        now = time.monotonic()
        dt = now - self._window_start
        if dt > 0:
            self._rate = self._window_count / dt
        self._window_start, self._window_count = now, 0

        mon = self._sh(["nexutil", f"-I{self.iface}", "-m"])
        chanspec_txt = self._sh(["nexutil", f"-I{self.iface}", "-k"])
        cs = 0
        mo = re.search(r"0x([0-9a-fA-F]+)", chanspec_txt)
        if mo:
            cs = int(mo.group(1), 16)
        ch, bw = decode_chanspec(cs)

        s = CsiStatus()
        s.header.stamp = self.get_clock().now().to_msg()
        s.header.frame_id = self.frame_id
        s.interface = self.iface
        s.monitor_mode = "1" in mon.split(":")[-1]
        s.chanspec = cs
        s.channel, s.bandwidth_mhz = int(ch), int(bw)
        s.firmware = self._firmware
        s.frames_per_sec = float(self._rate)
        s.frames_total = int(self.n_frames)
        s.dropped_total = int(self.n_dropped)
        s.mac_filter = self.mac
        self.pub_status.publish(s)

        if self._rate == 0.0:
            self.get_logger().warning(
                "0 frames/s — no OFDM traffic on this channel from the filtered "
                "source, or the extractor needs re-arming")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CsiPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

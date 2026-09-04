#!/usr/bin/env python3
"""ROS 2 node: publish one image per CSI frame.

Subscribes to a `wifi_csi_msgs/CsiFrame` topic and publishes a rendered
`sensor_msgs/Image` (and a JPEG `CompressedImage`) for every frame, so the
channel can be watched live in `rqt_image_view` or RViz while the robot drives.

The image is one panel with three views of the same frame:

    waterfall     the last N frames, subcarrier against time. This is where a
                  blockage shows up first: the smooth banding of a clean channel
                  breaks into deep, fast-moving notches.
    amplitude     |H| across subcarriers for the current frame, in dB relative
                  to that frame's own median (which removes the gain control).
    delay profile the power delay profile, rolled onto its strongest tap, so the
                  x axis is excess delay past the direct path.

and a header line with the source MAC, RSSI, Rician K and RMS delay spread.

Colour follows the offline figures: diverging about 0 dB for amplitude, so
fades and peaks separate rather than both being "far from the middle".

BANDWIDTH -- READ THIS
    CSI arrives at ~170 Hz. A 640x480 bgr8 image is 920 kB, so publishing one
    per frame is ~150 MB/s. That is fine over shared memory on the robot and
    hopeless over Wi-Fi. The node prints the figure it is about to produce at
    startup. Use `publish_every_n` to decimate, drop `publish_raw` and keep the
    compressed topic, or shrink the image.

Usage
-----
    ros2 run ... or directly:
        python3 csi_image_node.py --ros-args \
            -p input_topic:=/mobile1/csi -p publish_every_n:=4

    # render one frame from synthetic data, no ROS needed, to check the layout:
        python3 csi_image_node.py --selftest /tmp/csi.png

Parameters
----------
    input_topic        CsiFrame topic to subscribe to     (/mobile1/csi)
    image_topic        base output topic; "" = <input>/image
    history            frames held in the waterfall       (256)
    width, height      output image size                  (640, 480)
    publish_every_n    publish one image per N frames      (1 = every frame)
    db_range           waterfall/plot range, +/- dB        (8.0)
    publish_raw        publish sensor_msgs/Image           (true)
    publish_compressed publish .../compressed as JPEG      (true)
    jpeg_quality       0-100                               (80)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csi_core import (  # noqa: E402
    amplitude_db, band_mask, colour_lut, delay_profile, effective_bandwidth_mhz, occupied_band,
    quantise, rician_k, rms_delay_spread, usable_subcarriers,
)

try:
    import cv2
except ImportError:                      # text and JPEG are optional extras
    cv2 = None

# Same palettes as the offline figures, so the live view and the paper agree.
AMP_DIVERGING = ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f5a173", "#e34948", "#8f1f1e"]
BG = (250, 250, 248)          # BGR
INK = (20, 20, 20)
MUTED = (110, 108, 100)
GRID = (228, 226, 220)
ACCENT = (214, 120, 42)       # BGR of #2a78d6


class CsiRenderer:
    """Turns a stream of CSI frames into one panel image per frame.

    Deliberately free of ROS and of matplotlib: it is a pure array operation so
    it can keep up at the capture rate and be tested without either."""

    def __init__(self, width=640, height=480, history=256, db_range=8.0):
        self.w, self.h, self.history, self.db = width, height, history, db_range
        self.lut = colour_lut(AMP_DIVERGING)
        self.buf: np.ndarray | None = None      # (history, n_sub) ring of amp_db
        self.n_seen = 0
        self.meta: dict = {}
        # layout: waterfall on top, the two per-frame plots side by side below
        self.y_split = int(height * 0.56)
        self.x_split = width // 2

    def push(self, H, idx, raw_slots, bandwidth_mhz, rssi=None, src_mac=""):
        """Add one frame. H complex (n_sub,), idx its raw subcarrier slots."""
        H = np.asarray(H).reshape(1, -1)
        idx = np.asarray(idx, int)
        # Which slots carry signal, decided from a running average rather than
        # one frame -- a single frame is too noisy to tell a guard band from a
        # deep fade. Everything downstream uses only those slots, and the
        # bandwidth they span rather than the one the capture was configured for.
        self.acc = getattr(self, "acc", None)
        pw = np.abs(H[0]) ** 2
        self.acc = pw if self.acc is None or self.acc.size != pw.size else 0.98 * self.acc + 0.02 * pw
        use = usable_subcarriers(self.acc.reshape(1, -1))
        if use.sum() < 8:
            use = np.ones_like(use)
        band_lo, band_span = occupied_band(idx, use)
        use &= band_mask(idx, band_lo, band_span)
        Hu, idxu = H[:, use], idx[use]
        eff_bw = effective_bandwidth_mhz(band_span, bandwidth_mhz, raw_slots)
        # The receiver's fixed per-subcarrier shape, tracked as a slow average
        # of log-amplitude (about 2000 frames, ~10 s), is divided out before
        # anything is drawn or measured: it is not the room, and at 20 dB deep
        # it would otherwise be all the waterfall shows.
        la = 20 * np.log10(np.abs(Hu[0]) + 1e-12)
        self.shape = getattr(self, "shape", None)
        if self.shape is None or self.shape.size != la.size:
            self.shape = la.copy()
        else:
            self.shape += (la - self.shape) / 2000.0
        Hu = Hu / (10 ** (self.shape / 20))[None, :]
        amp = amplitude_db(Hu)[0]
        if self.buf is None or self.buf.shape[1] != amp.size:
            self.buf = np.zeros((self.history, amp.size), dtype=np.float32)
            self.n_seen = 0
        self.buf[:-1] = self.buf[1:]
        self.buf[-1] = amp
        self.n_seen += 1

        bw_hz = max(eff_bw, 1.0) * 1e6
        P = delay_profile(Hu, idxu, band_span, band_lo)
        window = max(band_span // 4, 8)
        tau = rms_delay_spread(P, 1.0 / bw_hz, window)[0]
        self.meta = {
            "amp": amp, "idx": idxu,
            "pdp": P[0, :window], "tap_ns": 1e9 / bw_hz,
            "tau_ns": float(tau * 1e9) if np.isfinite(tau) else float("nan"),
            "k_db": float(10 * np.log10(max(rician_k(Hu)[0], 1e-3))),
            "rssi": rssi, "src_mac": src_mac,
            "bw": int(bandwidth_mhz), "eff_bw": eff_bw, "n_use": int(use.sum()),
            "shape_ptp_db": float(np.ptp(self.shape)),
        }

    # -- drawing helpers (vectorised; a per-column Python loop cannot keep up) --
    @staticmethod
    def _line(img, v, lo, hi, colour, base=None):
        """Draw a value series across the full width of `img`, y from lo..hi."""
        h, w = img.shape[:2]
        if v.size < 2 or not np.isfinite(v).any():
            return
        src = np.clip((np.linspace(0, v.size - 1, w)).astype(int), 0, v.size - 1)
        y = np.clip((hi - v[src]) / max(hi - lo, 1e-9) * (h - 1), 0, h - 1).astype(int)
        rows = np.arange(h)[:, None]
        if base is not None:
            y0 = int(np.clip((hi - base) / max(hi - lo, 1e-9) * (h - 1), 0, h - 1))
            fill = (rows >= np.minimum(y, y0)) & (rows <= np.maximum(y, y0))
            img[fill] = (np.asarray(colour) * 0.35 + img[fill] * 0.65).astype(np.uint8)
        seg = np.minimum(y[:-1], y[1:]), np.maximum(y[:-1], y[1:])
        mask = np.zeros((h, w), bool)
        mask[:, :-1] = (rows >= seg[0]) & (rows <= seg[1])
        mask[y, np.arange(w)] = True
        img[mask] = colour

    @staticmethod
    def _text(img, s, org, colour=INK, scale=0.38, thick=1):
        if cv2 is not None:
            cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)

    def render(self) -> np.ndarray:
        img = np.full((self.h, self.w, 3), BG, np.uint8)
        if not self.meta:
            self._text(img, "waiting for CSI frames...", (12, self.h // 2), MUTED, 0.5)
            return img
        m = self.meta
        top = 30                                     # header strip

        # ---- waterfall: subcarrier (y) against time (x), newest on the right --
        wf_h = self.y_split - top - 16
        if wf_h > 4 and self.buf is not None:
            q = quantise(self.buf.T, -self.db, self.db)      # (n_sub, history)
            rows = np.clip(np.linspace(0, q.shape[0] - 1, wf_h).astype(int), 0, q.shape[0] - 1)
            cols = np.clip(np.linspace(0, q.shape[1] - 1, self.w).astype(int), 0, q.shape[1] - 1)
            img[top:top + wf_h, :] = self.lut[q[np.ix_(rows, cols)]]
            # blank the part of the ring that has not been filled yet
            unseen = max(self.history - self.n_seen, 0)
            if unseen:
                img[top:top + wf_h, : int(self.w * unseen / self.history)] = BG
            # labels go on the light strips above and below, not on the
            # waterfall itself, where the pale end of the ramp swallows them
            self._text(img, f"waterfall: subcarrier (y) x last {min(self.n_seen, self.history)} "
                            f"frames (x, newest right)", (6, top - 6), MUTED, 0.32)

        y0 = self.y_split
        img[y0 - 8: y0 - 7, :] = GRID

        # ---- current amplitude across subcarriers ----------------------------
        pane = img[y0 + 14: self.h - 16, 4: self.x_split - 6]
        pane[:] = BG
        pane[pane.shape[0] // 2: pane.shape[0] // 2 + 1, :] = GRID
        self._line(pane, m["amp"], -self.db, self.db, ACCENT, base=0.0)
        self._text(img, f"|H| vs subcarrier  +/-{self.db:.0f} dB", (8, y0 + 8), MUTED, 0.34)

        # ---- power delay profile --------------------------------------------
        pdp = m["pdp"]
        pdp_db = 10 * np.log10(np.maximum(pdp / max(pdp.max(), 1e-30), 1e-4))
        pane = img[y0 + 14: self.h - 16, self.x_split + 4: self.w - 4]
        pane[:] = BG
        self._line(pane, pdp_db, -40.0, 0.0, ACCENT, base=-40.0)
        span_ns = len(pdp) * m["tap_ns"]
        self._text(img, f"delay profile  0 to {span_ns:.0f} ns  ({m['tap_ns']:.1f} ns per tap)",
                   (self.x_split + 8, y0 + 8), MUTED, 0.34)

        # ---- header ----------------------------------------------------------
        rssi = "" if m["rssi"] is None else f"  RSSI {int(m['rssi'])} dBm"
        self._text(img, f"{m['src_mac']}  {m['eff_bw']:.0f}/{m['bw']} MHz  "
                        f"{m['n_use']} sc  shape {m['shape_ptp_db']:.0f} dB out{rssi}", (6, 15), INK, 0.42)
        tau = "n/a" if not np.isfinite(m["tau_ns"]) else f"{m['tau_ns']:.0f} ns"
        # K at the floor means the moment estimator found no dominant path at
        # all; printing the floor value as if it were a measurement would be a lie
        k = "K none (diffuse)" if m["k_db"] <= -29.9 else f"K {m['k_db']:+.1f} dB"
        right = f"{k}   spread {tau}"
        x = self.w - 8 - int(len(right) * 7.4)
        self._text(img, right, (max(x, 8), 15), INK, 0.42)
        return img


# --------------------------------------------------------------------------- #
def run_node(argv):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import CompressedImage, Image
    from wifi_csi_msgs.msg import CsiFrame

    class CsiImageNode(Node):
        def __init__(self):
            super().__init__("csi_image")
            p = self.declare_parameters("", [
                ("input_topic", "/mobile1/csi"), ("image_topic", ""),
                ("history", 256), ("width", 640), ("height", 480),
                ("publish_every_n", 1), ("db_range", 8.0),
                ("publish_raw", True), ("publish_compressed", True),
                ("jpeg_quality", 80),
            ])
            g = {q.name: q.value for q in p}
            base = g["image_topic"] or (g["input_topic"].rstrip("/") + "/image")
            self.every = max(int(g["publish_every_n"]), 1)
            self.quality = int(g["jpeg_quality"])
            self.renderer = CsiRenderer(int(g["width"]), int(g["height"]),
                                        int(g["history"]), float(g["db_range"]))
            self.pub_raw = (self.create_publisher(Image, base, 1)
                            if g["publish_raw"] else None)
            self.pub_jpg = (self.create_publisher(CompressedImage, base + "/compressed", 1)
                            if g["publish_compressed"] and cv2 is not None else None)
            if g["publish_compressed"] and cv2 is None:
                self.get_logger().warn("cv2 not importable: no compressed topic, and no text "
                                       "in the image")
            # CSI is best-effort, high rate: matching that avoids the subscription
            # silently not connecting to a sensor-QoS publisher
            qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST)
            self.sub = self.create_subscription(CsiFrame, g["input_topic"], self.on_csi, qos)
            self.n = 0
            mb = int(g["width"]) * int(g["height"]) * 3 * 170 / self.every / 1e6
            self.get_logger().info(
                f"{g['input_topic']} -> {base}"
                + (" (+ /compressed)" if self.pub_jpg else "")
                + f", every {self.every} frame(s), {g['width']}x{g['height']}")
            if self.pub_raw:
                self.get_logger().info(
                    f"raw Image at ~170 Hz/{self.every} is about {mb:.0f} MB/s; if that is too "
                    f"much raise publish_every_n or set publish_raw:=false and use /compressed")

        def on_csi(self, msg: CsiFrame):
            self.n += 1
            H = np.asarray(msg.csi_real, np.float32) + 1j * np.asarray(msg.csi_imag, np.float32)
            if H.size == 0:
                return
            self.renderer.push(H, msg.subcarrier_index, msg.raw_slots or H.size,
                               msg.bandwidth_mhz, msg.rssi, msg.src_mac)
            if self.n % self.every:
                return
            img = self.renderer.render()
            if self.pub_raw is not None:
                im = Image()
                im.header = msg.header
                im.height, im.width = img.shape[0], img.shape[1]
                im.encoding = "bgr8"
                im.is_bigendian = 0
                im.step = img.shape[1] * 3
                im.data = img.tobytes()
                self.pub_raw.publish(im)
            if self.pub_jpg is not None:
                ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
                if ok:
                    c = CompressedImage()
                    c.header = msg.header
                    c.format = "jpeg"
                    c.data = enc.tobytes()
                    self.pub_jpg.publish(c)

    rclpy.init(args=argv)
    node = CsiImageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def selftest(path: Path, n_frames: int = 300):
    """Render the panel from a synthetic channel, so the layout can be checked
    without a robot. Frames 200+ switch to a blocked channel: the waterfall
    should visibly break up and K should fall."""
    rng = np.random.default_rng(0)
    raw, bw = 256, 80
    idx = np.array([i for i in range(raw) if not (i < 6 or i > 249 or abs(i - 128) < 3)])
    f = (idx - raw / 2) * (bw * 1e6 / raw)
    r = CsiRenderer()

    def regime(blocked):
        """Tap gains, delays, phases and Doppler for one propagation regime.

        The phase of each tap is drawn ONCE and then advances slowly, because a
        tap is a physical reflection: it persists across frames and only drifts
        as the robot moves. Redrawing it every frame would make each frame an
        independent sample, which is a model of receiver noise -- the waterfall
        would show static rather than a channel, and the frame-to-frame
        coherence that decides whether a real capture is usable would read zero
        on synthetic data that is supposed to pass."""
        a = [0.18 if blocked else 1.0] + [(0.55 if blocked else 0.22) * np.exp(-j / 2.4)
                                          for j in range(1, 7)]
        d = [0.0] + [(55e-9 if blocked else 18e-9) * j for j in range(1, 7)]
        ph0 = rng.uniform(0, 2 * np.pi, len(a))
        fd = rng.normal(0, 0.8, len(a))          # Hz, walking-pace Doppler
        return np.array(a), np.array(d), ph0, fd

    clear_r, blocked_r = regime(False), regime(True)
    k_clear = None
    for k in range(n_frames):
        blocked = k > 200
        a, d, ph0, fd = blocked_r if blocked else clear_r
        t = k / r.rate_hz if getattr(r, "rate_hz", 0) else k / 180.0
        ph = ph0 + 2 * np.pi * fd * t
        H = (a[None, :] * np.exp(1j * ph[None, :])
             * np.exp(-2j * np.pi * f[:, None] * d[None, :])).sum(axis=1)
        H = H + (rng.normal(0, 0.02, f.size) + 1j * rng.normal(0, 0.02, f.size))
        r.push(H, idx, raw, bw, rssi=-62 if blocked else -45, src_mac="82:2a:a8:cb:d4:34")
        if k == 200:
            k_clear = r.meta["k_db"]
    img = r.render()
    if cv2 is not None:
        cv2.imwrite(str(path), img)
    else:                                    # PNG without cv2, via matplotlib
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(str(path), img[:, :, ::-1])
    print(f"wrote {path}  ({img.shape[1]}x{img.shape[0]})")
    print(f"  K clear {k_clear:+.1f} dB -> blocked {r.meta['k_db']:+.1f} dB, "
          f"spread {r.meta['tau_ns']:.0f} ns")
    if not (k_clear is not None and k_clear > r.meta["k_db"] + 3):
        print("  WARNING: K did not fall when the path was blocked; the "
              "renderer or the synthetic channel is wrong.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--selftest", type=Path, default=None)
    args, rest = ap.parse_known_args()
    if args.selftest:
        selftest(args.selftest)
    else:
        run_node(sys.argv[1:])

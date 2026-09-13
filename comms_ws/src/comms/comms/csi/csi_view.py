#!/usr/bin/env python3
"""Collect CSI frames from the live topic and save plots. Headless-safe.

    ros2 run comms csi_view --n 500
    ros2 run comms csi_view --topic /mobile2/csi --n 1000 --out /tmp/csi
    ros2 run comms csi_view --n 500 --frames ba        # Block Acks only

Separates Block Acks from data frames on ``seq``: 65535 (0xFFFF) is the
placeholder the firmware writes for control frames, while data frames carry a
real 802.11 sequence number. ``frame_control`` is NOT used for this -- it
reports the first byte of the PPDU, which for an aggregated A-MPDU is a
delimiter rather than a frame-control field, so data frames show up as 0xE0
rather than 0x88.

Artefact rejection is by magnitude, not by variance. The firmware writes
several slots that are not subcarriers at all (constant values like -32640 +
10277j, two orders above the real ones), and at least one of them varies
between frames, so a variance test misses it. Any subcarrier whose median
amplitude exceeds ``--artefact-factor`` times the overall median is dropped
and marked on the plots.

Writes three PNGs:
  <out>_heatmap.png     amplitude, frames x subcarriers -- the overview
  <out>_profile.png     mean amplitude and phase per subcarrier
  <out>_timeseries.png  a few subcarriers over time -- shows channel dynamics
"""

from __future__ import annotations

import argparse
import sys

import matplotlib
matplotlib.use("Agg")            # before pyplot: no display on a robot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import rclpy                     # noqa: E402
from rclpy.node import Node      # noqa: E402
from rclpy.qos import (QoSHistoryPolicy, QoSProfile,  # noqa: E402
                       QoSReliabilityPolicy)

from comms_msgs.msg import CsiFrame  # noqa: E402

BA_SEQ = 65535


# --------------------------------------------------------------- analysis
def artefact_mask(amp: np.ndarray, factor: float = 10.0,
                  var_eps: float = 1e-6) -> np.ndarray:
    """True where a subcarrier is a firmware artefact rather than a channel.

    amp is (frames, subcarriers). TWO tests are needed, because neither alone
    catches all of them:

      * magnitude -- some artefact slots hold values two orders above the
        real subcarriers (-32640 + 10277j against a few hundred). Compared
        against the median of per-subcarrier medians, which the artefacts
        themselves cannot drag upward the way a mean would.

      * variance -- others hold small constants (111 - 80j) that sit BELOW
        the real subcarriers, so magnitude cannot see them. They are exactly
        constant across frames, which nothing real ever is.

    Using only variance would also miss the loud slot that varies between
    frames (-32640 / -8960 / -9728 in the observed data). Hence both.
    """
    med = np.median(amp, axis=0)
    ref = np.median(med[med > 0]) if np.any(med > 0) else 1.0
    loud = med > factor * ref
    # Relative std, so the test does not depend on absolute scale.
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_std = np.where(med > 0, amp.std(axis=0) / med, 0.0)
    frozen = rel_std < var_eps
    return loud | frozen | (med <= 0)


def summarise(amp: np.ndarray, bad: np.ndarray, label: str) -> str:
    good = amp[:, ~bad]
    if good.size == 0:
        return f"{label}: no usable subcarriers"
    return (f"{label}: {amp.shape[0]} frames, {amp.shape[1]} slots, "
            f"{(~bad).sum()} usable, mean amp {good.mean():.0f}, "
            f"per-frame std {good.std(axis=1).mean():.0f}")


# --------------------------------------------------------------- plotting
def plot_all(amp, phase, idx, bad, out, title):
    n_bad = int(bad.sum())

    # 1. heatmap -- the overview. Artefact columns are blanked rather than
    #    dropped so subcarrier positions stay readable against idx.
    fig, ax = plt.subplots(figsize=(11, 5))
    shown = np.where(bad[None, :], np.nan, amp)
    im = ax.imshow(shown, aspect="auto", origin="lower",
                   interpolation="nearest", cmap="viridis",
                   extent=[idx[0], idx[-1], 0, amp.shape[0]])
    ax.set_xlabel("subcarrier index (raw firmware slot)")
    ax.set_ylabel("frame")
    ax.set_title(f"{title} -- amplitude ({n_bad} artefact slots blanked)")
    fig.colorbar(im, ax=ax, label="|CSI|")
    fig.tight_layout()
    fig.savefig(f"{out}_heatmap.png", dpi=130)
    plt.close(fig)

    # 2. per-subcarrier profile: mean amplitude and mean phase
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    mean_amp = amp.mean(axis=0)
    a1.plot(idx[~bad], mean_amp[~bad], lw=1.2)
    if n_bad:
        a1.plot(idx[bad], mean_amp[bad], "rx", ms=6, label="artefact")
        a1.legend()
    a1.set_ylabel("mean |CSI|")
    a1.set_title(f"{title} -- per-subcarrier profile")
    a1.grid(alpha=0.3)

    a2.plot(idx[~bad], np.angle(np.mean(np.exp(1j * phase), axis=0))[~bad],
            lw=1.2, color="tab:orange")
    a2.set_xlabel("subcarrier index")
    a2.set_ylabel("mean phase (rad)")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}_profile.png", dpi=130)
    plt.close(fig)

    # 3. a few subcarriers over time -- is the channel static or moving?
    fig, ax = plt.subplots(figsize=(11, 4))
    usable = np.where(~bad)[0]
    for k in usable[np.linspace(0, len(usable) - 1, 5).astype(int)]:
        ax.plot(amp[:, k], lw=0.9, label=f"sc {idx[k]}")
    ax.set_xlabel("frame")
    ax.set_ylabel("|CSI|")
    ax.set_title(f"{title} -- selected subcarriers over time")
    ax.legend(ncol=5, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}_timeseries.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------- collector
class Collector(Node):
    def __init__(self, topic: str, want: int, kind: str) -> None:
        super().__init__("csi_view")
        self.want = want
        self.kind = kind
        self.rows: list = []
        self.idx = None
        self.n_seen = 0
        self.n_skipped = 0
        qos = QoSProfile(depth=200, history=QoSHistoryPolicy.KEEP_LAST,
                         reliability=QoSReliabilityPolicy.RELIABLE)
        self.create_subscription(CsiFrame, topic, self._cb, qos)
        self.get_logger().info(
            f"collecting {want} '{kind}' frames from {topic} ...")

    def _cb(self, msg: CsiFrame) -> None:
        if len(self.rows) >= self.want:
            return
        self.n_seen += 1
        is_ba = msg.seq == BA_SEQ
        if (self.kind == "ba" and not is_ba) or \
           (self.kind == "data" and is_ba):
            self.n_skipped += 1
            return
        real = np.asarray(msg.csi_real, dtype=np.float64)
        imag = np.asarray(msg.csi_imag, dtype=np.float64)
        if real.size == 0 or real.size != imag.size:
            return
        # Frame widths can change mid-run (a 20 MHz frame inside an 80 MHz
        # capture fills fewer slots). Mixing widths in one array would be
        # meaningless, so lock to the first one seen.
        if self.idx is None:
            self.idx = np.asarray(msg.subcarrier_index, dtype=np.int32)
        if real.size != self.idx.size:
            self.n_skipped += 1
            return
        self.rows.append(real + 1j * imag)
        n = len(self.rows)
        if n % 100 == 0:
            self.get_logger().info(f"  {n}/{self.want}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", default="/mobile1/csi")
    ap.add_argument("--n", type=int, default=500, help="frames to collect")
    ap.add_argument("--frames", choices=["ba", "data", "all"], default="all",
                    help="ba = Block Acks (seq 65535), data = everything else")
    ap.add_argument("--out", default="csi", help="output prefix")
    ap.add_argument("--artefact-factor", type=float, default=10.0,
                    help="magnitude test: drop slots this many times above "
                         "the median subcarrier")
    ap.add_argument("--var-eps", type=float, default=1e-6,
                    help="variance test: drop slots whose relative std is "
                         "below this (exactly-constant firmware slots)")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args(argv)

    rclpy.init()
    node = Collector(args.topic, args.n, args.frames)
    start = node.get_clock().now()
    try:
        while rclpy.ok() and len(node.rows) < args.n:
            rclpy.spin_once(node, timeout_sec=0.2)
            if (node.get_clock().now() - start).nanoseconds / 1e9 > args.timeout:
                node.get_logger().warn("timed out")
                break
    except KeyboardInterrupt:
        pass

    rows, idx = node.rows, node.idx
    seen, skipped = node.n_seen, node.n_skipped
    node.destroy_node()
    rclpy.shutdown()

    if not rows:
        print(f"no frames collected ({seen} seen, {skipped} filtered out). "
              "Is the publisher running, and does --frames match what is "
              "on the channel?", file=sys.stderr)
        return 1

    csi = np.vstack(rows)
    amp = np.abs(csi)
    phase = np.angle(csi)
    bad = artefact_mask(amp, args.artefact_factor, args.var_eps)

    label = {"ba": "Block Acks", "data": "data frames",
             "all": "all frames"}[args.frames]
    print(summarise(amp, bad, label))
    print(f"artefact slots: {idx[bad].tolist()}")
    print(f"({seen} frames seen, {skipped} filtered out)")

    plot_all(amp, phase, idx, bad, args.out, f"{args.topic} -- {label}")
    for suffix in ("heatmap", "profile", "timeseries"):
        print(f"wrote {args.out}_{suffix}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

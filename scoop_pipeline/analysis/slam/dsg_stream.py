#!/usr/bin/env python3
"""How big is Hydra's scene-graph stream? Every update on hydra/backend/dsg
is the whole scene graph (with its mesh when the config serializes it), so
its size grows with the map; this logs each update's size against bag time
and plots it.

    # inside Hydra's container, while the bag plays (rclpy, hydra_msgs):
    python3 dsg_stream.py log  [--topic /hydra/backend/dsg] [--out dsg_stream.csv]
    # anywhere (matplotlib):
    python3 dsg_stream.py plot dsg_stream.csv [--goodput 175] [--out dsg_stream.png]

log: one CSV row per update, written as it arrives: t (its header stamp, s,
bag time), t_recv (receipt on the node's clock: bag time under --clock), bytes
(the serialized message, what a link would carry). Stop it with Ctrl-C.

plot: the update size (MB) and the rate it asks of a link at real time
(Mbit/s: size x 8 over the bag time since the previous update), against bag
time since the first update, one panel each; --goodput draws a link's goodput
(Mbit/s) on the rate panel. Prints the size and rate at the start, middle and
end, and their maxima.
"""
import argparse
import csv
import os
import struct
import sys


def log(topic, out):
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from hydra_msgs.msg import DsgUpdate

    rclpy.init()
    node = Node("dsg_stream_logger", parameter_overrides=[Parameter("use_sim_time", value=True)])
    fh = open(out, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["t", "t_recv", "bytes"])
    n = [0]

    def cb(raw):
        # CDR: 4 bytes encapsulation, then the header's stamp (int32 sec, uint32 nanosec)
        sec, nsec = struct.unpack_from("<iI", raw, 4)
        now = node.get_clock().now().nanoseconds * 1e-9
        w.writerow(["%.6f" % (sec + nsec * 1e-9), "%.6f" % now, len(raw)])
        fh.flush()
        n[0] += 1
        node.get_logger().info("update %d: %.2f MB" % (n[0], len(raw) / 1e6))

    node.create_subscription(DsgUpdate, topic, cb, QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
                             raw=True)
    node.get_logger().info("logging %s to %s (Ctrl-C to stop)" % (topic, out))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        fh.close()
        print("%d updates in %s" % (n[0], out))


def load(path):
    rows = [r for r in csv.DictReader(open(path))]
    t = [float(r["t"]) for r in rows]
    b = [int(r["bytes"]) for r in rows]
    order = sorted(range(len(t)), key=t.__getitem__)
    return [t[i] for i in order], [b[i] for i in order]


def series(t, b):
    """(x s since the first update, size MB, rate Mbit/s at real time) per update (rate from the 2nd)."""
    x = [v - t[0] for v in t]
    mb = [v / 1e6 for v in b]
    rate = [None] + [b[i] * 8 / 1e6 / (t[i] - t[i - 1]) if t[i] > t[i - 1] else None
                     for i in range(1, len(t))]
    return x, mb, rate


def plot(path, out=None, goodput=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, b = load(path)
    if len(t) < 2:
        sys.exit("%s: fewer than 2 updates" % path)
    x, mb, rate = series(t, b)
    ink, ink2, grid, line, ref = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#52514e"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    xr = [x[i] for i in range(len(x)) if rate[i] is not None]
    rr = [r for r in rate if r is not None]
    for ax, y, xs, lab in ((a1, mb, x, "Update size (MB)"), (a2, rr, xr, "Rate at real time (Mbit/s)")):
        ax.plot(xs, y, color=line, lw=2)
        ax.set_ylabel(lab)
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_ylim(bottom=0)
    if goodput:
        a2.axhline(goodput, color=ref, lw=1, ls="--")
        a2.text(x[-1], goodput, "link goodput %g Mbit/s" % goodput, color=ink2, ha="right", va="bottom")
        a2.set_ylim(top=max(max(rr), goodput) * 1.15)
    a2.set_xlabel("Bag time since the first update (s)")
    fig.suptitle("Hydra scene-graph stream (%s)" % os.path.basename(path), color=ink, x=0.06, ha="left")
    fig.tight_layout()
    out = out or os.path.splitext(path)[0] + ".png"
    fig.savefig(out, dpi=150, facecolor="#fcfcfb")
    n = len(x)
    print("%d updates over %.0f s, every %.2f s on average" % (n, x[-1], x[-1] / (n - 1)))
    for name, i in (("start", min(n - 1, 1)), ("middle", n // 2), ("end", n - 1)):
        r = "%8.1f Mbit/s" % rate[i] if rate[i] is not None else "       - Mbit/s"
        print("  %-6s at %6.0f s: %8.2f MB, %s" % (name, x[i], mb[i], r))
    print("  max size %.2f MB, max rate %.1f Mbit/s" % (max(mb), max(rr)))
    print("  wrote %s" % out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log")
    lg.add_argument("--topic", default="/hydra/backend/dsg")
    lg.add_argument("--out", default="dsg_stream.csv")
    pl = sub.add_parser("plot")
    pl.add_argument("csv")
    pl.add_argument("--out", default=None)
    pl.add_argument("--goodput", type=float, default=None, help="a link's goodput to draw (Mbit/s)")
    a = ap.parse_args()
    if a.cmd == "log":
        log(a.topic, a.out)
    else:
        plot(a.csv, a.out, a.goodput)


if __name__ == "__main__":
    main()

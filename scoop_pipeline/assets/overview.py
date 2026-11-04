#!/usr/bin/env python3
"""The registry to look at: every object in its own colour, and a top-down
picture with each one's name, over the room.

    python3 scoop_pipeline/assets/overview.py <registry> [--map <room .ply/.pcd>]
        [--label chair,table] [--max-points 3000]

Writes into the registry:
  overview.ply        every object's points, one colour per object
  overview_top.png    top-down: the room (--map, grey) and the objects, each
                      named at its centre; an object "on" another drawn with
                      a thin edge; numbers in brackets: frames segmented /
                      seen (views.py)
and prints the objects to look at first: those that pooled several names
(consolidate merge) and those kept with a low recall (consolidate prune's
--keep-views rule).
"""
import argparse
import colorsys
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from consolidate import read_ply                                        # noqa: E402
from objects import write_ply                                           # noqa: E402


def colour(k):
    h = (k * 0.618033988749895) % 1.0
    return np.array(colorsys.hsv_to_rgb(h, 0.75, 0.95)) * 255


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("registry")
    ap.add_argument("--map", default=None)
    ap.add_argument("--label", default=None, help="only these labels (comma-separated)")
    ap.add_argument("--max-points", type=int, default=3000, help="per object, in the picture")
    a = ap.parse_args(argv)
    reg = os.path.expanduser(a.registry)
    objs = json.load(open(os.path.join(reg, "objects.json")))["objects"]
    if a.label:
        want = {w.strip() for w in a.label.split(",")}
        objs = [o for o in objs if o["label"] in want]
    P, C = [], []
    for k, o in enumerate(objs):
        xyz, _ = read_ply(os.path.join(reg, o["points"]))
        P.append(xyz)
        C.append(np.tile(colour(k), (len(xyz), 1)))
    write_ply(os.path.join(reg, "overview.ply"), np.vstack(P), np.vstack(C).astype(np.uint8))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
    if a.map:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "analysis"))
        from twin_compare import load_cloud
        M = load_cloud(a.map)
        M = M[np.random.default_rng(0).choice(len(M), min(len(M), 400000), replace=False)]
        ax.scatter(M[:, 0], M[:, 1], s=0.05, c="0.8", linewidths=0, rasterized=True)
    rng = np.random.default_rng(1)
    for k, (o, xyz) in enumerate(zip(objs, P)):
        s = xyz if len(xyz) <= a.max_points else xyz[rng.choice(len(xyz), a.max_points, replace=False)]
        ax.scatter(s[:, 0], s[:, 1], s=0.4, color=colour(k) / 255, linewidths=0, rasterized=True)
        rec = "%d/%d" % (o.get("segmented", 0), o.get("seen", 0)) if "seen" in o else ""
        ax.text(o["centroid"][0], o["centroid"][1], "%s\n[%s]" % (o["id"], rec), fontsize=4.5,
                ha="center", va="center", color="black" if "on" not in o else "0.35",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=colour(k) / 255,
                          lw=0.4 if "on" in o else 0.9, alpha=0.75))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, map)")
    ax.set_ylabel("y (m, map)")
    ax.set_title("%s: %d objects" % (reg, len(objs)), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(reg, "overview_top.png"))
    print("-> %s, %s" % (os.path.join(reg, "overview.ply"), os.path.join(reg, "overview_top.png")))
    several = [o for o in objs if len(o.get("labels", {})) > 2]
    low = [o for o in objs if "recall" in o and o["recall"] < 0.15]
    if several:
        print("\npooled three or more names (one thing, or two touching ones?):")
        for o in sorted(several, key=lambda o: -len(o["labels"])):
            print("  %-20s %s" % (o["id"], ", ".join("%s %.0f" % kv for kv in o["labels"].items())))
    if low:
        print("\nkept with a recall under 15 % (found in many frames, but rarely when in sight):")
        for o in sorted(low, key=lambda o: o["recall"]):
            print("  %-20s %3d / %3d frames (%2.0f %%), %6d pts, centre %s" % (
                o["id"], o["segmented"], o["seen"], 100 * o["recall"], o["n_points"], o["centroid"]))


if __name__ == "__main__":
    sys.exit(main())

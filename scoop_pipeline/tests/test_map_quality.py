#!/usr/bin/env python
"""Self-test for analysis/mapping/map_quality.py's layer split, on a
synthetic map at 2 cm with 0.5 cm noise: a floor; a 15 cm wall seen from
both sides (the path passes on both); a 20 cm wall seen from both sides
(one layer that thick would fail the planarity test); one surface mapped
twice 10 cm apart, seen from one side; and a smeared wall (4 cm sigma).
The walls' two faces must come out as two thin layers, the doubled surface
as doubled, the smear as one thick layer. Needs numpy and scipy.

    python tests/test_map_quality.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis", "mapping"))

import map_quality as mq                                               # noqa: E402

rng = np.random.default_rng(0)
G = 0.02


def grid(u0, u1, v0, v1):
    u, v = np.meshgrid(np.arange(u0, u1, G), np.arange(v0, v1, G))
    return u.ravel(), v.ravel()


def wall_x(x, y0, y1, sigma=0.005):            # plane x = const, z 0.1..2
    y, z = grid(y0, y1, 0.1, 2.0)
    return np.c_[x + rng.normal(0, sigma, len(y)), y, z]


def wall_y(y, x0, x1, sigma=0.005):
    x, z = grid(x0, x1, 0.1, 2.0)
    return np.c_[x, y + rng.normal(0, sigma, len(x)), z]


def main():
    x, y = grid(-6, 8, -2, 7)
    floor = np.c_[x, y, rng.normal(0, 0.005, len(x))]
    parts = {
        "faces15": np.vstack([wall_x(2.0, 0, 4), wall_x(2.15, 0, 4)]),
        "faces20": np.vstack([wall_x(5.0, 0, 4), wall_x(5.20, 0, 4)]),
        "doubled": np.vstack([wall_y(5.0, -3, 0), wall_y(5.10, -3, 0)]),
        "smeared": wall_x(-4.0, 0, 4, sigma=0.04),
    }
    P = np.vstack([floor] + list(parts.values()))
    # the path: along both sides of the two thick walls, and in front of the
    # doubled surface and the smeared wall only
    ys = np.arange(0, 4, 0.1)
    path = np.vstack([np.c_[np.full_like(ys, x0), ys, np.full_like(ys, 0.5)]
                      for x0 in (1.0, 3.5, 6.5, -3.0)])
    centres = []
    st, rej = mq.patch_stats(P, n_samples=4000, centres=centres, positions=path)
    C = np.array(centres)[:, :3]
    where = {
        "faces15": (np.abs(C[:, 0] - 2.075) < 0.12) & (C[:, 1] > 0.3) & (C[:, 1] < 3.7) & (C[:, 2] > 0.4),
        "faces20": (np.abs(C[:, 0] - 5.10) < 0.15) & (C[:, 1] > 0.3) & (C[:, 1] < 3.7) & (C[:, 2] > 0.4),
        "doubled": (np.abs(C[:, 1] - 5.05) < 0.12) & (C[:, 0] > -2.7) & (C[:, 0] < -0.3) & (C[:, 2] > 0.4),
        "smeared": (np.abs(C[:, 0] + 4.0) < 0.12) & (C[:, 1] > 0.3) & (C[:, 1] < 3.7) & (C[:, 2] > 0.4),
        "floor": (C[:, 2] < 0.05) & (C[:, 0] < -4.5),
    }
    expect = {"faces15": (mq.FACES, 15), "faces20": (mq.FACES, 20),
              "doubled": (mq.DOUBLED, 10), "smeared": (mq.ONE, None), "floor": (mq.ONE, None)}
    fails = []
    for name, m in where.items():
        if m.sum() < 5:
            fails.append(f"{name}: only {m.sum()} patches measured")
            continue
        kind, sep = expect[name]
        k = st[m, 4].astype(int)
        share = np.mean(k == kind)
        surf = np.median(st[m, 2]) * 100
        apart = np.median(st[m & (st[:, 4] == kind), 3]) * 100 if share else float("nan")
        msg = (f"{name:8s} {m.sum():4d} patches, {100 * share:3.0f} % {mq.KINDS[kind]}, "
               f"surface {surf:4.1f} cm" + (f", apart {apart:4.1f} cm" if sep else ""))
        ok = share > 0.9
        if sep:
            ok &= abs(apart - sep) < 1.5 and surf < 3.0
        elif name == "smeared":
            ok &= surf > 10.0
        else:
            ok &= surf < 3.0
        print(("  ok    " if ok else "  FAIL  ") + msg)
        if not ok:
            fails.append(name)
    # without the path the same two layers are only "side unknown"
    st2, _ = mq.patch_stats(P, n_samples=4000)
    if not np.all(st2[st[:, 4] == mq.FACES, 4] == mq.UNKNOWN):
        fails.append("without --traj two faces are not 'side unknown'")
    print("map quality ok" if not fails else "map quality FAILED: %s" % fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

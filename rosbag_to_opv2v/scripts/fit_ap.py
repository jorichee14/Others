#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Where the access point stands, solved from the radio instead of picked.

A pick in a viewer is a judgement about which blob in a point cloud is the
antenna, and a wrong one is indistinguishable from a right one until the
figure is drawn. The signal itself knows better: every frame of every agent
is a measurement of distance to the AP, and tens of thousands of them over
two trajectories pin a position down far harder than one click can.

    python3 scripts/fit_ap.py --root <split> [--field wifi.signal_dbm]

Fits the log-distance path loss

    rssi(d) = a - 10 n log10(d)

for the AP's x and y, with ``a`` (the 1 m reference level) and ``n`` (the
path loss exponent) solved exactly at each candidate position, so only two
unknowns are ever searched. Fits each agent alone as well as the pair, and
prints them side by side: trajectories that disagree have not seen enough
of the geometry between them to place it, and the script says so rather
than returning a confident number from one corridor.

Indoors this is a lower bound on honesty, not a survey -- walls attenuate
and corridors guide -- so treat a spread of a metre or so as the real
uncertainty, and check the answer against the cloud before using it.
"""

import argparse
import math
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def dig(block, path):
    """A dotted path into nested dicts, or None where it runs out."""
    for key in path.split("."):
        if not isinstance(block, dict) or key not in block:
            return None
        block = block[key]
    return block if isinstance(block, (int, float)) and not isinstance(block, bool) else None


def read_samples(root, field):
    """[(agent, x, y, rssi)] over every frame of every case in a split."""
    root = os.path.expanduser(root)
    out = {}
    cases = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    for case in cases:
        case_dir = os.path.join(root, case)
        for agent in sorted(d for d in os.listdir(case_dir)
                            if os.path.isdir(os.path.join(case_dir, d))):
            agent_dir = os.path.join(case_dir, agent)
            for name in sorted(f for f in os.listdir(agent_dir) if f.endswith(".yaml")):
                with open(os.path.join(agent_dir, name), encoding="utf-8") as handle:
                    params = yaml.safe_load(handle)
                value = dig(params.get("comms") or {}, field)
                pose = params.get("lidar_pose")
                if value is None or not pose:
                    continue
                out.setdefault(agent, []).append((float(pose[0]), float(pose[1]),
                                                  float(value)))
    return {agent: np.asarray(rows, dtype=np.float64) for agent, rows in out.items()}


def residual_at(x, y, samples, floor):
    """Best achievable residual with the AP here, a and n solved exactly.

    The model is linear in a and n once the distances are fixed, so they are
    never searched -- only the position is. That is what makes an exhaustive
    sweep of the floor plan affordable, and an exhaustive sweep is what makes
    the answer a global one rather than wherever a descent happened to land.
    """
    distance = np.hypot(samples[:, 0] - x, samples[:, 1] - y)
    u = -10.0 * np.log10(np.maximum(distance, floor))
    v = samples[:, 2]
    # Two parameters: the normal equations in closed form, not an SVD. The
    # sweep calls this once per candidate cell, tens of thousands of times.
    du, dv = u - u.mean(), v - v.mean()
    denominator = float(du @ du)
    n = float(du @ dv) / denominator if denominator > 1e-12 else 0.0
    a = float(v.mean() - n * u.mean())
    error = a + n * u - v
    return float(np.sqrt(np.mean(error ** 2))), (a, n)


def sweep(samples, x0, x1, y0, y1, step, floor, best=None):
    for x in np.arange(x0, x1 + step / 2, step):
        for y in np.arange(y0, y1 + step / 2, step):
            rmse, solution = residual_at(x, y, samples, floor)
            if best is None or rmse < best[0]:
                best = (rmse, float(x), float(y), solution)
    return best


def fit(samples, low, high, coarse, fine, floor):
    """Sweep the floor plan, then refine around the best cell."""
    best = sweep(samples, low[0], high[0], low[1], high[1], coarse, floor)
    return sweep(samples, best[1] - coarse, best[1] + coarse,
                 best[2] - coarse, best[2] + coarse, fine, floor, best)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="split dir holding case_*/")
    parser.add_argument("--field", default="wifi.signal_dbm",
                        help="the dotted comms path to fit (default: wifi.signal_dbm)")
    parser.add_argument("--pad", type=float, default=12.0,
                        help="how far outside the walked area the AP may be (m)")
    parser.add_argument("--coarse", type=float, default=0.25)
    parser.add_argument("--fine", type=float, default=0.02)
    parser.add_argument("--floor", type=float, default=0.5,
                        help="distances below this are clamped: log10 has no "
                             "opinion at zero and the near field is not the model")
    args = parser.parse_args()

    per_agent = read_samples(args.root, args.field)
    if not per_agent:
        print(f"no frames carry {args.field!r} under {args.root}", file=sys.stderr)
        return 2
    pooled = np.concatenate(list(per_agent.values()))
    low = pooled[:, :2].min(axis=0) - args.pad
    high = pooled[:, :2].max(axis=0) + args.pad
    print(f"{len(pooled):,} samples over {len(per_agent)} agents, "
          f"{args.field} from {pooled[:, 2].min():.0f} to {pooled[:, 2].max():.0f}")
    print(f"searching x {low[0]:.1f}..{high[0]:.1f}, y {low[1]:.1f}..{high[1]:.1f}\n")

    print(f"{'source':>12}  {'x':>8} {'y':>8}  {'rmse':>6}  {'a':>7} {'n':>5}  samples")
    answers = {}
    for name, rows in list(per_agent.items()) + [("both", pooled)]:
        rmse, x, y, (a, n) = fit(rows, low, high, args.coarse, args.fine, args.floor)
        answers[name] = (x, y)
        label = f"agent {name}" if name != "both" else "both"
        print(f"{label:>12}  {x:8.3f} {y:8.3f}  {rmse:6.2f}  {a:7.1f} {n:5.2f}  "
              f"{len(rows):,}")

    singles = [v for k, v in answers.items() if k != "both"]
    if len(singles) > 1:
        spread = max(math.hypot(p[0] - q[0], p[1] - q[1])
                     for p in singles for q in singles)
        print(f"\nthe agents disagree by {spread:.2f} m")
        if spread > 3.0:
            print("that is too much to call it placed -- one of them probably "
                  "never changed its distance to the AP enough to constrain it")
    x, y = answers["both"]
    print(f"\n  --ap {x:.3f},{y:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

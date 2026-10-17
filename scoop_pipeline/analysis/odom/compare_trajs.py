#!/usr/bin/env python3
"""
How a robot's trajectories of one pass differ: GLIM's, 02's refined and
02b's bundle-adjusted (or any others), pose by pose. No ground truth: this
says where and on which time scale they disagree, and what each one's
loop closure on the boards was.

  python3 analysis/odom/compare_trajs.py <pipeline config> \\
      [--traj NAME=FILE ...] [--plot out.png] [--out out.json]

Default trajectories, from odometry/<machine>/ of the config's pass:
  glim     traj_lidar.txt          (GLIM, exported by process_recording.py)
  02       traj_lidar_refined.txt  (02_refine_poses.py)
  ba       traj_lidar_ba.txt       (02b_bundle_adjust.py)
--traj NAME=FILE adds or replaces one.

For every pair (A, B), B interpolated at A's stamps:
  absolute   |p_A - p_B| and the rotation between them, as they are (both
             in the same frame: 02 starts from GLIM's, BA from 02's) and
             after the best rigid fit of B onto A (what is left is shape,
             not a global offset)
  relative   over 1 s, 10 s, 100 s: the difference of the motion each one
             says the robot made in that time,
               | (T_A(t)^-1 T_A(t+d)).t - (T_B(t)^-1 T_B(t+d)).t |
             small at 1 s and growing with d = they differ by slow drift;
             already large at 1 s = they differ in short-term motion
and per trajectory: path length, and the loop closure on the boards from
<traj>_loop_closure.json (GLIM: glim_loop_closure.json) when it is there.
--plot draws the top view and |p_A - p_B| over time.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "map_stages"))

from pipeline_common import load_pipeline, load_traj                         # noqa: E402

DELTAS = (1.0, 10.0, 100.0)
DEFAULT = (("glim", "traj_lidar.txt"), ("02", "traj_lidar_refined.txt"),
           ("ba", "traj_lidar_ba.txt"))


def interp(t, T, tq):
    """Poses of (t, T) at the stamps tq (inside its span)."""
    R = Slerp(t, Rotation.from_matrix(T[:, :3, :3]))(tq).as_matrix()
    p = np.stack([np.interp(tq, t, T[:, i, 3]) for i in range(3)], 1)
    out = np.tile(np.eye(4), (len(tq), 1, 1))
    out[:, :3, :3], out[:, :3, 3] = R, p
    return out


def rot_deg(R):
    return np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec(), axis=-1))


def rigid_fit(src, dst):
    """R, t minimizing |R src + t - dst| (Kabsch, no scale)."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, md - R @ ms


def stats(v):
    v = np.asarray(v)
    return dict(median=float(np.median(v)), p95=float(np.percentile(v, 95)),
                max=float(v.max())) if len(v) else None


def path_length(T):
    return float(np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1).sum())


def loop_closure(path, name):
    """The loop-closure report next to a trajectory, summarized per board."""
    d = os.path.dirname(path)
    for f in (os.path.splitext(path)[0] + "_loop_closure.json",
              os.path.join(d, "glim_loop_closure.json") if name == "glim" else ""):
        if f and os.path.isfile(f):
            r = json.load(open(f))
            out = {}
            for b, v in (r.get("boards") or {}).items():
                lc = v.get("loop_closure")
                if lc:
                    out[b] = dict(mm=lc["mm"], deg=lc["deg"], noise_mm=lc["noise_mm"],
                                  significant=lc["significant"])
            return out
    return None


def relative(TA, TB, t, d):
    """Translation difference of the motion over d s, at every stamp that
    has a stamp d later."""
    j = np.searchsorted(t, t + d)
    ok = j < len(t)
    ok &= np.abs(t[np.minimum(j, len(t) - 1)] - (t + d)) < 0.5 * max(d, 0.2)
    i, j = np.flatnonzero(ok), j[ok]

    def rel(T):
        return np.einsum("nji,nj->ni", T[i, :3, :3], T[j, :3, 3] - T[i, :3, 3])
    dt = np.linalg.norm(rel(TA) - rel(TB), axis=1)
    RA = np.einsum("nji,njk->nik", TA[i, :3, :3], TA[j, :3, :3])
    RB = np.einsum("nji,njk->nik", TB[i, :3, :3], TB[j, :3, :3])
    dr = rot_deg(np.einsum("nji,njk->nik", RA, RB))
    return dt, dr


def compare(a, b, ta, TA, tb, TB):
    lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
    m = (ta >= lo) & (ta <= hi)
    t, A = ta[m], TA[m]
    B = interp(tb, TB, t)
    dp = np.linalg.norm(A[:, :3, 3] - B[:, :3, 3], axis=1)
    dz = np.abs(A[:, 2, 3] - B[:, 2, 3])
    dR = rot_deg(np.einsum("nji,njk->nik", A[:, :3, :3], B[:, :3, :3]))
    R, tt = rigid_fit(B[:, :3, 3], A[:, :3, 3])
    dp_fit = np.linalg.norm(A[:, :3, 3] - (B[:, :3, 3] @ R.T + tt), axis=1)
    out = dict(pair=f"{a} vs {b}", poses=int(len(t)),
               position_cm=stats(dp * 100), vertical_cm=stats(dz * 100),
               rotation_deg=stats(dR), after_rigid_fit_cm=stats(dp_fit * 100),
               rigid_fit=dict(t_cm=(tt * 100).round(2).tolist(),
                              deg=float(rot_deg(R[None])[0])), relative={})
    for d in DELTAS:
        rt, rr = relative(A, B, t, d)
        out["relative"][f"{d:g}s"] = dict(translation_cm=stats(rt * 100), rotation_deg=stats(rr))
    return out, (t, dp)


def fmt(s, nd=1):
    return "-" if s is None else f"{s['median']:.{nd}f} / {s['p95']:.{nd}f} / {s['max']:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--traj", action="append", default=[], metavar="NAME=FILE")
    ap.add_argument("--plot")
    ap.add_argument("--out")
    a = ap.parse_args()

    P = load_pipeline(a.config)
    files = {}
    for name, f in DEFAULT:
        p = P.outp(f)
        if name == "glim" and not os.path.isfile(p):
            p = P.glim_traj()
        files[name] = p
    for kv in a.traj:
        name, _, f = kv.partition("=")
        if not f:
            raise SystemExit(f"--traj {kv}: write NAME=FILE")
        files[name] = os.path.expanduser(f)
    trajs = {}
    print("trajectories:")
    for name, f in files.items():
        if not os.path.isfile(f):
            print(f"  {name:6s} (none at {f})")
            continue
        t, T = load_traj(f)
        trajs[name] = (t, T, f)
        print(f"  {name:6s} {len(t)} poses, {t[-1] - t[0]:.0f} s, {path_length(T):.1f} m  {f}")
    if len(trajs) < 2:
        raise SystemExit("need at least two trajectories")

    print("\nloop closure on the boards (mm / deg, noise mm):")
    lcs = {}
    for name, (_, _, f) in trajs.items():
        lcs[name] = loop_closure(f, name)
        if not lcs[name]:
            print(f"  {name:6s} -")
        for b, v in (lcs[name] or {}).items():
            print(f"  {name:6s} {b:10s} {v['mm']:6.1f} / {v['deg']:.3f}  (noise {v['noise_mm']:.1f})"
                  f"  {'DRIFT' if v['significant'] else 'not significant'}")

    results, curves = [], {}
    for x, y in itertools.combinations(trajs, 2):
        r, curve = compare(x, y, trajs[x][0], trajs[x][1], trajs[y][0], trajs[y][1])
        results.append(r)
        curves[r["pair"]] = curve

    print("\nabsolute difference, median / p95 / max:")
    print(f"  {'pair':14s}{'position cm':>22s}{'vertical cm':>22s}{'rotation deg':>24s}"
          f"{'after rigid fit cm':>22s}")
    for r in results:
        print(f"  {r['pair']:14s}{fmt(r['position_cm']):>22s}{fmt(r['vertical_cm']):>22s}"
              f"{fmt(r['rotation_deg'], 3):>24s}{fmt(r['after_rigid_fit_cm']):>22s}")
    print("\nrelative difference: the motion over d s, translation cm, median / p95 / max:")
    print(f"  {'pair':14s}" + "".join(f"{f'{d:g} s':>22s}" for d in DELTAS))
    for r in results:
        print(f"  {r['pair']:14s}" + "".join(
            f"{fmt(r['relative'][f'{d:g}s']['translation_cm'], 2):>22s}" for d in DELTAS))
    print("  rotation deg, median / p95 / max:")
    for r in results:
        print(f"  {r['pair']:14s}" + "".join(
            f"{fmt(r['relative'][f'{d:g}s']['rotation_deg'], 3):>24s}" for d in DELTAS))

    out = a.out or os.path.join(os.path.dirname(trajs[next(iter(trajs))][2]), "traj_compare.json")
    with open(out, "w") as fh:
        json.dump(dict(files={k: v[2] for k, v in trajs.items()}, loop_closure=lcs,
                       pairs=results), fh, indent=1)
    print(f"\nreport: {out}")

    if a.plot:
        try:
            import matplotlib
        except ImportError:
            print("no plot: matplotlib is not installed (pip install matplotlib)")
            return
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        for name, (_, T, _) in trajs.items():
            ax[0].plot(T[:, 0, 3], T[:, 1, 3], lw=0.8, label=name)
        ax[0].set_aspect("equal")
        ax[0].set_xlabel("x (m)")
        ax[0].set_ylabel("y (m)")
        ax[0].set_title("top view")
        ax[0].legend()
        for pair, (t, dp) in curves.items():
            ax[1].plot(t - t[0], dp * 100, lw=0.8, label=pair)
        ax[1].set_xlabel("time (s)")
        ax[1].set_ylabel("position difference (cm)")
        ax[1].set_title("difference over the run")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(a.plot, dpi=130)
        print(f"plot: {a.plot}")


if __name__ == "__main__":
    main()

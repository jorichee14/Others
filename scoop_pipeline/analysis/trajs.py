#!/usr/bin/env python3
"""Trajectory checks, all in one module; run them through check.py:

  python3 check.py trajs  <config> [--traj NAME=FILE ...] [--ref NAME] [--plot P]
  python3 check.py time   <config> [--traj FILE] [--ref FILE]
  python3 check.py imu    <config> [--traj NAME=FILE ...]
  python3 check.py loop   <config> [--traj FILE] [--out REPORT]

  trajs  a pass's trajectories against each other: ATE / RPE against a
         reference, heading error vs turning (time offset / scale), loop
         closure from the reports beside them
  time   is GLIM's pose time its scans'? (every pass, run_pass.py's time_check)
  imu    is the IMU's stamping on the LiDAR's time? (|w| lined up)
  loop   a trajectory's loop closure on the boards (05 --check)

Each command's own help (check.py <command> -h) is the section's text below.
The stages import from here too: 02 and 02b loop_check(), run_pass.py
reference_traj() / odometry_file(), scoop/recording.py loop_run().
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STAGES = os.path.join(ROOT, "map_stages")
sys.path.insert(0, ROOT)
sys.path.insert(0, STAGES)

from pipeline_common import load_pipeline, load_traj                     # noqa: E402



# ============================================================================
# trajs: trajectories against each other (was analysis/odom/compare_trajs.py)
# ============================================================================
COMPARE_DOC = r'''
How a robot's trajectories of one pass differ: GLIM's, 02's bundle-adjusted
and 02b's scan-to-map (or any others), pose by pose. No ground truth: this
says where and on which time scale they disagree, and what each one's
loop closure on the boards was.

  python3 check.py trajs <pipeline config> \
      [--traj NAME=FILE ...] [--plot out.png] [--out out.json]

Default trajectories, from odometry/<machine>/ of the config's pass:
  glim     traj_lidar.txt          (GLIM, exported by process_recording.py)
  ba       traj_lidar_ba.txt       (02_bundle_adjust.py)
  s2m      traj_lidar_refined.txt  (02b_scan_to_map.py)
(check.py report is the same against the reference, per pass.)
--traj NAME=FILE adds or replaces one.

For every pair (A, B), B interpolated at A's stamps:
  absolute   |p_A - p_B| and the rotation between them, as they are (both
             in the same frame: 02 and 02b start from GLIM's) and
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

--shift NAME=SECONDS  adds NAME with its stamps moved (NAME+<ms>ms)
--ref NAME  every other trajectory against this one in the usual metrics
            (Sturm et al., IROS 2012), as a table and as LaTeX rows:
              ATE  RMSE of the position (and rotation) error after the best
                   rigid fit of the trajectory onto the reference
              RPE  RMSE of the error of the motion over 1 s, 10 s, 100 s
            and whether each one's heading error follows the turns (a time
            offset or a gyro scale / extrinsic error) or not (drift)
'''


DELTAS = (1.0, 10.0, 100.0)
DEFAULT = (("glim", "traj_lidar.txt"), ("ba", "traj_lidar_ba.txt"),
           ("s2m", "traj_lidar_refined.txt"))


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
                max=float(v.max()), rmse=float(np.sqrt(np.mean(v ** 2)))) if len(v) else None


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


def yaw(T):
    return np.unwrap(np.arctan2(T[:, 1, 0], T[:, 0, 0]))


def smooth(v, n):
    if n <= 1:
        return v
    k = np.ones(n) / n
    return np.convolve(np.pad(v, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def heading_vs_turning(ta, TA, tb, TB, hp_s=20.0):
    """Does the heading error of A against the reference B follow B's
    turning? Two regressions over the run:
      timing   e(t) against the yaw rate w(t): a pose stamped dt off is
               rotated by w dt, so e = dt w; e with its slow part (> hp_s)
               removed, so drift does not count
      scale    the change of e over 1 s against the turned angle over 1 s:
               a gyro scale or extrinsic tilt error makes it proportional
    -> dict(r_timing, dt_ms, r_scale, scale_pct)."""
    lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
    m = (ta >= lo) & (ta <= hi)
    t, A = ta[m], TA[m]
    B = interp(tb, TB, t)
    step = float(np.median(np.diff(t)))
    psi_b = yaw(B)
    e = yaw(A) - psi_b
    e -= np.median(e)
    # the same band-pass on both, so the slope is not biased by the filter
    ns, nh = max(1, int(round(0.3 / step))), max(1, int(round(hp_s / step)))
    w_raw = np.gradient(psi_b, t)
    w = smooth(w_raw, ns) - smooth(w_raw, nh)
    e_hp = smooth(e, ns) - smooth(e, nh)
    out = {}
    if np.std(w) > 1e-6:
        out["r_timing"] = float(np.corrcoef(e_hp, w)[0, 1])
        out["dt_ms"] = float(np.dot(e_hp - e_hp.mean(), w - w.mean()) / np.sum((w - w.mean()) ** 2) * 1000)
    k = max(1, int(round(1.0 / step)))
    de, dpsi = e[k:] - e[:-k], psi_b[k:] - psi_b[:-k]
    if np.std(dpsi) > 1e-6:
        out["r_scale"] = float(np.corrcoef(de, dpsi)[0, 1])
        out["scale_pct"] = float(np.dot(de - de.mean(), dpsi - dpsi.mean())
                                 / np.sum((dpsi - dpsi.mean()) ** 2) * 100)
    return out


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
    dR_fit = rot_deg(np.einsum("nji,njk->nik", A[:, :3, :3], R @ B[:, :3, :3]))
    out = dict(pair=f"{a} vs {b}", a=a, b=b, poses=int(len(t)),
               position_cm=stats(dp * 100), vertical_cm=stats(dz * 100),
               rotation_deg=stats(dR), after_rigid_fit_cm=stats(dp_fit * 100),
               after_rigid_fit_deg=stats(dR_fit),
               rigid_fit=dict(t_cm=(tt * 100).round(2).tolist(),
                              deg=float(rot_deg(R[None])[0])), relative={})
    for d in DELTAS:
        rt, rr = relative(A, B, t, d)
        out["relative"][f"{d:g}s"] = dict(translation_cm=stats(rt * 100), rotation_deg=stats(rr))
    return out, (t, dp)


def fmt(s, nd=1):
    return "-" if s is None else f"{s['median']:.{nd}f} / {s['p95']:.{nd}f} / {s['max']:.{nd}f}"


def main_compare(argv=None):
    ap = argparse.ArgumentParser(description=COMPARE_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--traj", action="append", default=[], metavar="NAME=FILE")
    ap.add_argument("--plot")
    ap.add_argument("--out")
    ap.add_argument("--ref", help="name of the reference trajectory (e.g. ba_glim)")
    ap.add_argument("--shift", action="append", default=[], metavar="NAME=SECONDS",
                    help="also compare NAME with its stamps moved by SECONDS, as NAME+<ms>ms "
                         "(e.g. glim=0.046: is its error a time offset?)")
    a = ap.parse_args(argv)

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
    for kv in a.shift:
        name, _, sec = kv.partition("=")
        if name not in trajs or not sec:
            raise SystemExit(f"--shift {kv}: write NAME=SECONDS with NAME one of {', '.join(trajs)}")
        t, T, f = trajs[name]
        new = f"{name}{float(sec) * 1000:+.0f}ms"
        trajs[new] = (t + float(sec), T, f)
        print(f"  {new:6s} {name} with its stamps moved by {float(sec) * 1000:+.1f} ms")
    if len(trajs) < 2:
        raise SystemExit("need at least two trajectories")

    print("\nloop closure on the boards (mm / deg, noise mm):")
    lcs = {}
    for name, (_, _, f) in trajs.items():
        lcs[name] = None if name.endswith("ms") and name not in files else loop_closure(f, name)
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

    if a.ref:
        if a.ref not in trajs:
            raise SystemExit(f"--ref {a.ref}: no such trajectory ({', '.join(trajs)})")
        print(f"\nagainst the reference '{a.ref}' (RMSE; ATE after the best rigid fit):")
        hdr = (f"  {'trajectory':10s}{'ATE cm':>8s}{'ATE deg':>9s}"
               + "".join(f"{f'RPE {d:g}s cm':>13s}{'deg':>7s}" for d in DELTAS)
               + f"{'path m':>9s}")
        print(hdr)
        rows = []
        for name in trajs:
            if name == a.ref:
                continue
            r = next(x for x in results if {x["a"], x["b"]} == {name, a.ref})
            rel = [(r["relative"][f"{d:g}s"]["translation_cm"]["rmse"],
                    r["relative"][f"{d:g}s"]["rotation_deg"]["rmse"]) for d in DELTAS]
            ate = (r["after_rigid_fit_cm"]["rmse"], r["after_rigid_fit_deg"]["rmse"])
            path = path_length(trajs[name][1])
            print(f"  {name:10s}{ate[0]:8.2f}{ate[1]:9.3f}"
                  + "".join(f"{x:13.2f}{y:7.3f}" for x, y in rel) + f"{path:9.1f}")
            rows.append(f"{name} & {ate[0]:.1f} & {ate[1]:.2f} & "
                        + " & ".join(f"{x:.1f} / {y:.2f}" for x, y in rel)
                        + f" & {path:.1f} \\\\")
        print(f"  ({a.ref}: path {path_length(trajs[a.ref][1]):.1f} m)")
        print(f"\nheading error against '{a.ref}' vs turning (correlation r; "
              f"|r| > 0.5 = follows the turns):")
        print(f"  {'trajectory':10s}{'r timing':>10s}{'time offset ms':>16s}{'r scale':>10s}"
              f"{'per turned angle':>18s}")
        turn = {}
        for name in trajs:
            if name == a.ref:
                continue
            h = heading_vs_turning(trajs[name][0], trajs[name][1], trajs[a.ref][0], trajs[a.ref][1])
            turn[name] = h
            print(f"  {name:10s}{h.get('r_timing', float('nan')):10.2f}{h.get('dt_ms', float('nan')):16.1f}"
                  f"{h.get('r_scale', float('nan')):10.2f}{h.get('scale_pct', float('nan')):17.2f}%")
        print("  timing: a pose stamped dt off turns e = w dt; scale: a gyro scale or extrinsic\n"
              "  tilt error grows with the angle turned. Neither (|r| small): the error is not\n"
              "  tied to turning (drift / registration).")
        print("  LaTeX rows (trajectory & ATE cm & ATE deg & RPE " +
              " & ".join(f"{d:g} s cm / deg" for d in DELTAS) + " & path m):")
        for ln in rows:
            print("    " + ln)

    out = a.out or os.path.join(os.path.dirname(trajs[next(iter(trajs))][2]), "traj_compare.json")
    with open(out, "w") as fh:
        json.dump(dict(files={k: v[2] for k, v in trajs.items()}, loop_closure=lcs,
                       pairs=results, heading_vs_turning=turn if a.ref else None), fh, indent=1)
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


# ============================================================================
# time: GLIM's pose time against the scans' (was map_stages/pose_time_check.py)
# ============================================================================
TIME_DOC = r'''Is GLIM's pose time its scans' time? Checked on every pass, against the
trajectory the scans themselves fix.

    python3 check.py time <pipeline config> [--traj FILE] [--ref FILE] [--out JSON]

--traj  GLIM's trajectory as exported (default odometry/<machine>/
        traj_lidar.txt: its poses moved by recording.yaml's
        glim.pose_time_offset, which glim.yaml beside it records)
--ref   default: a mapping pass, 02's bundle-adjusted trajectory
        (02_bundle_adjust.output); a run, 04's LiDAR track (every scan
        registered to the map)
--out   default odometry/<machine>/pose_time_check.json

GLIM labels each pose with the start of its 100 ms sweep, other LiDAR
odometry with the middle or the end, and how far the pose it computes sits
from the label is to be measured per sensor (github.com/MOLAorg/
mola_glim_wrapper); on mobile_2 it is 46 ms (20261015). The reference is
placed by the scans at their own stamps, so a GLIM pose dt off its stamp
turns by e = w dt against it at yaw rate w: the heading error regressed on
the turning gives dt,
its correlation r whether the error is timing at all (trajs' heading_vs_turning).

  on time   |dt| < 10 ms or r < 0.5 (20261015 with the correction: 3.9 ms,
            r 0.23; without: 45.8 ms, r 0.92)
  OFF       set glim.pose_time_offset to (the applied one + dt) for this
            machine in configs/recording.yaml, re-export GLIM
            (processing/glim_check.py), and run the pass again from 02

Exits 0 either way: 02's reference and 04's track are fixed by the scans
(references built from GLIM before and after the 46 ms correction agree to
0.5 cm), so only GLIM's own trajectory depends on the offset.
'''


TIMING_MS, TIMING_R = 10.0, 0.5           # OFF beyond both


def applied_offset(traj):
    """The pose_time_offset (s) GLIM's export applied: glim.yaml beside it."""
    meta = os.path.join(os.path.dirname(traj), "glim.yaml")
    if os.path.isfile(meta):
        for ln in open(meta):
            if ln.startswith("pose_time_offset_s:"):
                return float(ln.split(":", 1)[1].strip() or 0)
    return 0.0


def time_check(t, T, tr, Tr, applied_s=0.0):
    """Heading of (t, T) against the reference (tr, Tr) vs its turning ->
    dict(dt_ms, r_timing, r_scale, scale_pct, applied_offset_ms, on_time,
    offset_to_set_s)."""
    h = heading_vs_turning(t, T, tr, Tr)
    dt, r = h.get("dt_ms", 0.0), h.get("r_timing", 0.0)
    h.update(applied_offset_ms=applied_s * 1000,
             on_time=not (abs(dt) >= TIMING_MS and r >= TIMING_R),
             offset_to_set_s=round(applied_s + dt / 1000, 3))
    return h


def time_verdict(h):
    """The check's lines, as printed."""
    out = [f"GLIM's heading error against the reference follows the turning as "
           f"{h.get('dt_ms', float('nan')):+.1f} ms (r {h.get('r_timing', float('nan')):.2f}); "
           f"exported with pose_time_offset {h['applied_offset_ms']:.0f} ms"]
    if h["on_time"]:
        out.append("-> on time: GLIM's poses are its scans' (keep the offset)")
    else:
        out.append(f"-> OFF by {h['dt_ms']:+.0f} ms: set glim.pose_time_offset to "
                   f"{h['offset_to_set_s']:.3f} for this machine in configs/recording.yaml, "
                   "re-export GLIM (processing/glim_check.py) and run the pass again from 02")
    return out


def reference_traj(P):
    """02's output (a mapping pass) or 04's LiDAR track (a run)."""
    if P.reference:
        return P.lidar_track_traj()
    return P.outp((P.cfg.get("02_bundle_adjust") or {}).get("output") or "traj_lidar_ba.txt")


def odometry_file(P, name):
    """odometry/<machine>/<name>, whatever ROUTES would do with the name."""
    return os.path.join(os.path.dirname(P.outp("traj_lidar.txt")), name)


def main_time(argv=None):
    ap = argparse.ArgumentParser(description=TIME_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--traj")
    ap.add_argument("--ref")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    P = load_pipeline(a.config)
    traj = os.path.expanduser(a.traj) if a.traj else P.seed_traj()
    ref = os.path.expanduser(a.ref) if a.ref else reference_traj(P)
    out = a.out or odometry_file(P, "pose_time_check.json")
    for name, f in (("GLIM", traj), ("reference", ref)):
        if not os.path.isfile(f):           # nothing to check: not an error of the pass
            print(f"pose time not checked: no {name} trajectory at {f}")
            return
    t, T = load_traj(traj)
    tr, Tr = load_traj(ref)
    print(f"GLIM       {traj} ({len(t)} poses)\nreference  {ref} ({len(tr)} poses)")
    h = time_check(t, T, tr, Tr, applied_offset(traj))
    for ln in time_verdict(h):
        print(ln)
    h.update(traj=traj, reference=ref)
    with open(out, "w") as fh:
        json.dump(h, fh, indent=1)
    print(f"report: {out}")


# ============================================================================
# imu: the IMU's stamps against the LiDAR's time (was analysis/odom/imu_vs_traj.py)
# ============================================================================
IMU_DOC = r'''Are the IMU's stamps on the same time as the LiDAR's? The IMU's angular
rate against a trajectory's, lined up.

  python3 check.py imu <pipeline config> [--traj NAME=FILE ...]
      [--topic /mobile_2/ouster/imu]

Default trajectories, from odometry/<machine>/ of the config's pass: ref
(traj_lidar_ba_glim.txt: the scans made consistent, so on the LiDAR's time)
and glim (traj_lidar.txt). The IMU is the config's bag's <machine>'s
ouster/imu, or --topic.

The rigid body turns at one rate whatever the frame, so |w| from the IMU's
gyro and |w| from the trajectory (central differences of its rotations,
which carry no half-step delay) are one signal; the shift that lines them
up best (scoop/imu_sync.py line_up, the IMU-to-IMU check of the retimed
step) is how far the IMU's stamps run behind the trajectory's:

  against ref   ~0           the IMU and the LiDAR are on the same time:
                             the stamping is right
                ~ GLIM's     the IMU's stamps are off by that much: fix them
                offset       (or GLIM's imu_time_offset), and GLIM again
  against glim               the same comparison for GLIM's own poses

Reads the bag's IMU only, writes nothing.
'''


def traj_rate(t, T):
    """|w| at each interior pose, from the poses either side (central)."""
    dR = np.einsum("nji,njk->nik", T[:-2, :3, :3], T[2:, :3, :3])
    ang = np.linalg.norm(Rotation.from_matrix(dR).as_rotvec(), axis=1)
    return t[1:-1], ang / (t[2:] - t[:-2])


def main_imu(argv=None):
    from scoop import bag
    from scoop.imu_sync import angular_rates, line_up
    ap = argparse.ArgumentParser(description=IMU_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--traj", action="append", default=[], metavar="NAME=FILE")
    ap.add_argument("--topic")
    ap.add_argument("--max-shift", type=float, default=0.5, help="search +/- this (s)")
    a = ap.parse_args(argv)

    P = load_pipeline(a.config)
    files = {"ref": P.outp("traj_lidar_ba_glim.txt"), "glim": P.outp("traj_lidar.txt")}
    for kv in a.traj:
        name, _, f = kv.partition("=")
        files[name] = os.path.expanduser(f)
    topic = a.topic or f"/{P.machine}/ouster/imu"
    bag_path = os.path.expanduser(P.dataset["bag"])
    print(f"IMU {topic} in {bag_path}")
    ti, wi = angular_rates(bag.open_bag(bag_path), topic)
    if len(ti) < 100:
        raise SystemExit(f"{len(ti)} IMU messages on {topic}")
    ti = ti * 1e-9
    print(f"  {len(ti)} messages, {1 / np.median(np.diff(ti)):.0f} Hz")
    print(f"\n{'trajectory':12s}{'IMU behind it':>16s}{'correlation':>13s}{'scale':>8s}")
    for name, f in files.items():
        if not os.path.isfile(f):
            print(f"{name:12s}  (none at {f})")
            continue
        t, T = load_traj(f)
        tr, wr = traj_rate(t, T)
        s, c, sc, n, turn = line_up(tr, wr, ti, wi, a.max_shift)
        print(f"{name:12s}{s * 1000:13.1f} ms{c:13.3f}{sc:8.3f}")
    print("\n+ = the IMU's stamps are later than the trajectory's for the same motion.\n"
          "Against ref (on the LiDAR's time): ~0 = the stamping is right.")


# ============================================================================
# loop: loop closure on the boards (was map_stages/loop_check.py)
# ============================================================================
LOOP_DOC = r'''A trajectory's loop closure on the boards: stage 05 in its read-only check
mode (05_anchor.py --traj --check), summed up and set against GLIM's.

    python3 check.py loop <pipeline config> [--traj FILE] [--out REPORT]

--traj  the trajectory to check (default: dataset.traj, e.g. 02's
        traj_lidar_ba.txt in odometry/<machine>/)
--out   the report (default: <trajectory>_loop_closure.json next to it)

For every board seen at the start and again later -- start and end on the
same board -- where the trajectory puts it the first and the last time, in
mm / deg against the detections' own noise (DRIFT or not significant), and
the co-visible cross-checks. When GLIM's own check is there
(odometry/<machine>/glim_loop_closure.json, written after GLIM) the two are
printed side by side. Stages 02 and 02b run this on their results by themselves; the GLIM
step of process_recording.py runs it on GLIM's.
'''


def loop_run(config, traj, report, log=print):
    """05 --check on ``traj`` -> ``report``; the summary (:func:`summarize`),
    or None when the check failed."""
    code = subprocess.call([sys.executable, os.path.join(STAGES, "05_anchor.py"), str(config),
                            "--traj", str(traj), "--check", str(report)])
    if code != 0 or not os.path.isfile(report):
        log("    loop closure: the check failed (exit %s)" % code)
        return None
    return loop_summary(report)


def loop_summary(report):
    """The report's per-board loop closures and cross-checks, compact."""
    with open(report) as fh:
        r = json.load(fh)
    boards = {}
    for name, b in (r.get("boards") or {}).items():
        lc = b.get("loop_closure")
        boards[name] = ({"sections": b["sections"], "from_section": lc["from_section"],
                         "to_section": lc["to_section"], "over_s": lc["dt_s"],
                         "mm": lc["mm"], "deg": lc["deg"], "noise_mm": lc["noise_mm"],
                         "significant": lc["significant"]} if lc else
                        {"sections": b["sections"], "mm": None})
    return {"report": str(report), "trajectory": r.get("trajectory"),
            "anchor_board": r.get("anchor_board"), "boards": boards,
            "cross_checks": [{k: c[k] for k in ("a", "b", "disagreement_mm",
                                                "disagreement_deg", "significant")}
                             for c in r.get("cross_checks") or []]}


def loop_lines(summary, label="loop closure"):
    out = []
    for name, b in summary["boards"].items():
        if b.get("mm") is None:
            out.append("%s %s: seen in %d section(s), not revisited" % (label, name, b["sections"]))
        else:
            out.append("%s %s: s%d->s%d over %.0f s: %.1f mm / %.3f deg (noise %.1f mm) -> %s"
                       % (label, name, b["from_section"], b["to_section"], b["over_s"],
                          b["mm"], b["deg"], b["noise_mm"],
                          "DRIFT" if b["significant"] else "not significant"))
    return out


def loop_vs_glim(this, glim, label="refined"):
    """One line per board seen by both: GLIM's closure -> this one's."""
    out = []
    for name, b in this["boards"].items():
        g = glim["boards"].get(name) or {}
        if b.get("mm") is None or g.get("mm") is None:
            continue
        out.append("%-12s GLIM %6.1f mm / %.3f deg  ->  %s %6.1f mm / %.3f deg"
                   % (name, g["mm"], g["deg"], label, b["mm"], b["deg"]))
    return out


def loop_check(config, traj, report=None, log=print, label="refined"):
    """Run the check on ``traj``, print it, and set it against GLIM's when
    that is there (glim_loop_closure.json in the same folder)."""
    report = report or os.path.splitext(traj)[0] + "_loop_closure.json"
    log("\n=== loop closure of %s on the boards ===" % os.path.basename(traj))
    summ = loop_run(config, traj, report, log)
    if summ is None:
        return None
    for ln in loop_lines(summ, "loop closure"):
        log("  " + ln)
    glim = os.path.join(os.path.dirname(traj), "glim_loop_closure.json")
    if os.path.isfile(glim) and os.path.abspath(glim) != os.path.abspath(report):
        cmp_ = loop_vs_glim(summ, loop_summary(glim), label)
        if cmp_:
            log("  against GLIM:")
            for ln in cmp_:
                log("    " + ln)
    log("  report: %s" % report)
    return summ


def main_loop(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    opt = {}
    for flag in ("--traj", "--out"):
        if flag in args:
            i = args.index(flag)
            opt[flag] = args[i + 1]
            del args[i:i + 2]
    if not args or args[0] in ("-h", "--help"):
        print(LOOP_DOC)
        return
    P = load_pipeline(args[0])
    traj = os.path.expanduser(opt.get("--traj") or P.outp(P.dataset["traj"]))
    if not os.path.isfile(traj):
        sys.exit("no trajectory at %s" % traj)
    if loop_check(args[0], traj, opt.get("--out")) is None:
        sys.exit(1)

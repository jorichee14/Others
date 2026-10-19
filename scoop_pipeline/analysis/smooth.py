#!/usr/bin/env python3
"""How smooth is a trajectory? A robot's body moves smoothly at 10 Hz, so what
a trajectory wobbles about a smooth path within a fraction of a second is
estimation jitter, not motion. From the trajectories alone (no bag, no map):

  python3 check.py smooth <pipeline config> [--traj NAME=FILE ...]
      [--windows 0.5,1.0] [--still 0.02] [--out smooth.json]

Default trajectories (odometry/<machine>/), each left out when missing:
  GLIM   the BA's input (traj_lidar.txt, pose-time corrected)
  02b    scan-to-map (02b_scan_to_map.output, traj_lidar_refined.txt)
  BA     dataset.traj (traj_lidar_ba_glim.txt, the reference)
--traj NAME=FILE adds or replaces one (a bare name is in odometry/<machine>/).

Per trajectory, over the time all of them share:
  jitter, position   RMS of each position about a local quadratic fit of the
                     window around it (Savitzky-Golay, order 2): a fit that
                     follows any constant acceleration exactly, so what is
                     left is wobble faster than the window (mm)
  jitter, rotation   the same for the orientation: the poses of the window as
                     rotations from the centre pose, fitted the same way (deg)
  jerk, ang. accel   RMS of the third / second finite difference of position /
                     orientation (m/s^3, deg/s^2): the raw roughness, which
                     jitter inflates most
Each for the whole run and for the robot standing still (speed below --still
m/s and turning below 1 deg/s over 2 s, judged on BA's trajectory for all, so
a jittery one is not judged by its own wobble), where every wobble is jitter.
Lower is smoother. A white-noise wobble of sigma reads as 0.72 sigma at a
5-pose window and 0.89 sigma at 11; the 1 s window also takes in real motion
that is not quadratic over 1 s (sharp turns), so 0.5 s is the cleaner reading.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "map_stages"))

from pipeline_common import load_pipeline, load_traj                     # noqa: E402


def odd_window(seconds, dt):
    n = max(5, int(round(seconds / dt)) | 1)
    return n if n % 2 else n + 1


def pos_jitter(p, n):
    """(N,) distance of each position from its local quadratic fit (m)."""
    return np.linalg.norm(p - savgol_filter(p, n, 2, axis=0, mode="interp"), axis=1)


def rot_jitter(R, n):
    """(N,) angle of each orientation from its local quadratic fit (rad):
    the window's rotations from the centre pose, Log(R_i^T R_j), fitted with
    the Savitzky-Golay centre weights; the fit at the centre is the jitter."""
    h = n // 2
    w = savgol_coeffs(n, 2)                         # weights of the centre value
    rot = Rotation.from_matrix(R)
    N = len(R)
    i = np.arange(N)
    fit = np.zeros((N, 3))
    for k, wk in zip(range(-h, h + 1), w):
        j = np.clip(i + k, 0, N - 1)
        fit += wk * (rot[i].inv() * rot[j]).as_rotvec()
    ang = np.linalg.norm(fit, axis=1)
    ang[:h] = ang[-h:] = np.nan                     # windows cut by the ends
    return ang


def roughness(t, p, R):
    """RMS jerk (m/s^3) and angular acceleration (rad/s^2), finite differences."""
    v = np.gradient(p, t, axis=0)
    a = np.gradient(v, t, axis=0)
    j = np.gradient(a, t, axis=0)
    rot = Rotation.from_matrix(R)
    dt = np.diff(t)
    w = (rot[:-1].inv() * rot[1:]).as_rotvec() / dt[:, None]      # body rate
    tm = 0.5 * (t[1:] + t[:-1])
    al = np.gradient(w, tm, axis=0)
    return (float(np.sqrt(np.mean(np.sum(j ** 2, 1)))),
            float(np.sqrt(np.mean(np.sum(al ** 2, 1)))))


def rms(x, m=None):
    x = x if m is None else x[m]
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")


def still_times(t, T, still, smooth_s=2.0):
    """(t, still?) of a trajectory: speed below `still` m/s and turning below
    1 deg/s, both over smooth_s s, so a jittery trajectory is not judged by
    its own wobble; every trajectory is then judged on the same times."""
    dt = float(np.median(np.diff(t)))
    n = odd_window(smooth_s, dt)
    p, R = T[:, :3, 3], T[:, :3, :3]
    speed = np.linalg.norm(savgol_filter(p, n, 1, deriv=1, delta=dt, axis=0, mode="interp"), axis=1)
    yaw = np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    rate = np.abs(savgol_filter(yaw, n, 1, deriv=1, delta=dt, mode="interp"))
    return t, (speed < still) & (rate < np.radians(1.0)), speed


def measure(t, T, windows, still_at):
    dt = float(np.median(np.diff(t)))
    p, R = T[:, :3, 3], T[:, :3, :3]
    ts, sm, speed = still_at
    st = sm[np.clip(np.searchsorted(ts, t), 0, len(ts) - 1)]
    out = dict(poses=len(t), dt_s=dt, still_pct=float(st.mean() * 100),
               speed_ms=float(np.median(speed[~sm])) if (~sm).any() else 0.0)
    for w in windows:
        n = odd_window(w, dt)
        pj, rj = pos_jitter(p, n), rot_jitter(R, n)
        pj[: n // 2] = pj[-(n // 2):] = np.nan
        out[f"{w:g}s"] = dict(window_poses=n,
                              pos_mm=rms(pj) * 1e3, pos_still_mm=rms(pj, st) * 1e3,
                              rot_deg=np.degrees(rms(rj)), rot_still_deg=np.degrees(rms(rj, st)))
    jerk, alpha = roughness(t, p, R)
    out.update(jerk_ms3=jerk, ang_acc_degs2=float(np.degrees(alpha)))
    return out


def default_trajs(P):
    s2m = (P.cfg.get("02b_scan_to_map") or {}).get("output", "traj_lidar_refined.txt")
    try:
        glim = P.seed_traj()
    except (SystemExit, Exception):
        glim = None
    return {"GLIM": glim, "02b": P.outp(s2m), "BA": P.outp(P.dataset["traj"])}


def main_smooth(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--traj", action="append", default=[], metavar="NAME=FILE")
    ap.add_argument("--windows", default="0.5,1.0", help="s, comma-separated")
    ap.add_argument("--still", type=float, default=0.02, help="m/s: below this the robot stands still")
    ap.add_argument("--out", help="also write the numbers here (JSON)")
    a = ap.parse_args(argv)
    P = load_pipeline(a.config)
    files = default_trajs(P)
    for s in a.traj:
        name, f = s.split("=", 1)
        files[name] = P.outp(f)
    files = {k: v for k, v in files.items() if v and os.path.isfile(v)}
    if not files:
        raise SystemExit("no trajectory found")
    loaded = {k: load_traj(v) for k, v in files.items()}
    t0 = max(t[0] for t, _ in loaded.values())
    t1 = min(t[-1] for t, _ in loaded.values())
    windows = [float(x) for x in a.windows.split(",")]
    judge = "BA" if "BA" in loaded else next(iter(loaded))
    tj, Tj = loaded[judge]
    still_at = still_times(tj, Tj, a.still)
    res = {}
    for k, (t, T) in loaded.items():
        m = (t >= t0) & (t <= t1)
        print(f"  {k:<6} {files[k]}  ({m.sum()} poses)")
        res[k] = measure(t[m], T[m], windows, still_at)
    first = next(iter(res.values()))
    print(f"\nsmoothness over {t1 - t0:.0f} s shared; still {first['still_pct']:.0f} % of it "
          f"(by {judge}: speed < {a.still} m/s, turning < 1 deg/s); lower is smoother")
    head = "".join(f" | jitter {w:g} s: pos mm, rot deg (still)" for w in windows)
    print(f"\n{'':<6}{head} | jerk m/s^3 | ang. acc deg/s^2")
    for k, r in res.items():
        cells = "".join(f" | {r[f'{w:g}s']['pos_mm']:5.2f} ({r[f'{w:g}s']['pos_still_mm']:4.2f}), "
                        f"{r[f'{w:g}s']['rot_deg']:6.3f} ({r[f'{w:g}s']['rot_still_deg']:5.3f})"
                        for w in windows)
        print(f"{k:<6}{cells} | {r['jerk_ms3']:10.2f} | {r['ang_acc_degs2']:8.1f}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(files=files, windows_s=windows, still_ms=a.still, results=res), fh, indent=1)
        print(f"\n-> {a.out}")
    return res


if __name__ == "__main__":
    main_smooth()

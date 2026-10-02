#!/usr/bin/env python3
"""Where does a map's wall thickness come from? Tells apart wrong point
timing (deskew), a timing offset of the trajectory, and drift, from a few
hundred scans.

  python3 scoop_pipeline/analysis/mapping/scan_consistency.py \\
      scoop_pipeline/map_stages/pipeline_config.json [--traj FILE]

Uses the bag, trajectory (dataset.traj, or --traj), topic and 01_build_map
settings (time_tol, lidar_min/max, deskew_bins) of the pipeline config, and
measures wall thickness the way map_quality.py does (5-95 % spread of the
points of 0.25 m planar patches):

  1. SINGLE scans: the sensor's own noise floor. Nothing below it is
     reachable. (Timing errors bend a whole scan smoothly, which keeps it
     locally flat, so they do not show here -- they show in 2.)
  2. PAIRS of scans 0.1 s, 1 s, 10 s apart and revisits (same place, minutes
     apart), measured only where both hit the same wall; split by whether
     the robot was turning, and the turning pairs once more without deskew:
       turning pairs thick, straight thin   ->  timing (see 3) or deskew
       turning thinner WITHOUT deskew        ->  the point times are wrong
       thickens with the separation          ->  drift
       thin until the revisits               ->  loop closure / pass to pass
  3. the 1 s / 10 s / revisit pairs again with the trajectory's stamps
     shifted by each --offsets value:
       thinnest at an offset other than 0  ->  the pose given for t is the
       pose at t + that offset

Reads only the scans it needs (--anchors x 5), writes nothing.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "map_stages"))

from pipeline_common import (DEFAULT_TOPICS, deskew_to_world, load_pipeline,  # noqa: E402
                             load_traj, pc2_xyzt, traj_quats)

R = 0.25            # patch radius (m), as map_quality.py
MIN_PTS = 30
PLANARITY = 0.60
MIN_EACH = 10       # pair patches: points needed from each scan
WALL, FLOOR = 60.0, 30.0      # normal tilt from vertical (deg): wall > 60, floor < 30


def turn_rate_and_speed(tr_t, tr_T):
    """Per pose: rotation rate (deg/s, about any axis) and speed (m/s)."""
    dt = np.maximum(np.diff(tr_t), 1e-6)
    dR = np.einsum("nji,njk->nik", tr_T[:-1, :3, :3], tr_T[1:, :3, :3])
    ang = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    v = np.linalg.norm(np.diff(tr_T[:, :3, 3], axis=0), axis=1)
    rate, speed = ang / dt, v / dt
    pad = lambda a: np.concatenate([[a[0]], 0.5 * (a[1:] + a[:-1]), [a[-1]]])
    return pad(rate), pad(speed)


def place(scan, tr_t, tr_T, tr_q, deskew, nb, shift=0.0):
    """World points of one scan; the trajectory's stamps moved by `shift` s.
    deskew: every point through the pose at its own time, as the map stages
    do (pipeline_common.deskew_to_world); else the whole scan through the
    pose at the cloud stamp."""
    t, p, dt = scan
    return deskew_to_world(p, dt if deskew else None, t, tr_t, tr_T, tr_q, nb, shift)


def patches(P, centers, owner=None):
    """(thickness m, normal tilt deg) of every planar patch around `centers`;
    with `owner` (0/1 per point) only patches holding MIN_EACH of both."""
    tree = cKDTree(P)
    out, rough = [], 0
    for nb in tree.query_ball_point(centers, R):
        if len(nb) < MIN_PTS:
            continue
        if owner is not None:
            k = int(owner[nb].sum())
            if min(k, len(nb) - k) < MIN_EACH:
                continue
        Q = P[nb] - P[nb].mean(0)
        _, sv, vt = np.linalg.svd(Q, full_matrices=False)
        if sv[1] < 1e-9 or sv[2] / sv[1] > PLANARITY:
            rough += 1
            continue
        d = Q @ vt[2]
        out.append((np.percentile(d, 95) - np.percentile(d, 5),
                    np.degrees(np.arccos(min(1.0, abs(vt[2][2]))))))
    return np.array(out).reshape(-1, 2), rough


def read_scans(bag, topic, tr_t, need, tol, lo, hi, window):
    """{pose index: (stamp s, xyz float32, point times or None)} for the
    poses in `need`. Messages whose log time is more than `window` s from
    every needed stamp are not decoded (0 = decode all)."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    want = np.array(sorted(tr_t[list(need)]))
    out = {}
    with AnyReader([Path(bag)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as r:
        conns = [c for c in r.connections if c.topic == topic]
        if not conns:
            raise SystemExit(f"{topic} is not in {bag}")
        for conn, log_ns, raw in r.messages(connections=conns):
            if window > 0:
                i = np.searchsorted(want, log_ns * 1e-9)
                near = min(abs(want[min(i, len(want) - 1)] - log_ns * 1e-9),
                           abs(want[max(i - 1, 0)] - log_ns * 1e-9))
                if near > window:
                    continue
            msg = r.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            j = int(np.clip(np.searchsorted(tr_t, t), 1, len(tr_t) - 1))
            j = j if abs(tr_t[j] - t) < abs(tr_t[j - 1] - t) else j - 1
            if j not in need or abs(tr_t[j] - t) > tol or j in out:
                continue
            p, dt = pc2_xyzt(msg)
            d = np.linalg.norm(p, axis=1)
            keep = np.isfinite(p).all(1) & (d > lo) & (d < hi)
            out[j] = (t, p[keep].astype(np.float64),
                      None if dt is None else dt[keep])
            if len(out) == len(need):
                break
    return out


def cm(a):
    return f"{np.median(a) * 100:6.1f}" if len(a) else "     -"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="pipeline_config.json")
    ap.add_argument("--traj", help="trajectory (TUM); default dataset.traj")
    ap.add_argument("--anchors", type=int, default=40, help="scans sampled along the run")
    ap.add_argument("--patches", type=int, default=200, help="patches per scan / pair")
    ap.add_argument("--turn-dps", type=float, default=15.0, help="turning above this (deg/s)")
    ap.add_argument("--straight-dps", type=float, default=3.0,
                    help="straight below this (deg/s), while moving > 0.2 m/s")
    ap.add_argument("--revisit-s", type=float, default=60.0, help="revisit: at least this apart")
    ap.add_argument("--revisit-m", type=float, default=1.0, help="revisit: within this distance")
    ap.add_argument("--offsets", default="-0.1,-0.05,-0.02,0,0.02,0.05,0.1",
                    help="trajectory stamp shifts to try (s); write --offsets=-0.1,0,0.1")
    ap.add_argument("--log-window", type=float, default=2.0,
                    help="decode only clouds logged within this of a needed stamp (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    P = load_pipeline(a.config)
    s = P.cfg["01_build_map"]
    nb = max(2, int(s.get("deskew_bins", 100)))
    traj = a.traj or P.outp(P.dataset["traj"])
    tr_t, tr_T = load_traj(traj)
    tr_q = traj_quats(tr_T)
    topic = P.dataset.get("points_topic") or DEFAULT_TOPICS["points_topic"]
    rate, speed = turn_rate_and_speed(tr_t, tr_T)
    step = float(np.median(np.diff(tr_t)))
    N = len(tr_t)
    print(f"trajectory {traj}: {N} poses, {tr_t[-1] - tr_t[0]:.0f} s, "
          f"one per {step * 1000:.0f} ms")

    seps = {f"{sep:g}s": int(round(sep / step)) for sep in (0.1, 1.0, 10.0)}
    far = max(seps.values())
    anchors = np.unique(np.linspace(5, N - far - 6, a.anchors).astype(int))
    pairs = {k: [(j, j + d) for j in anchors] for k, d in seps.items()}
    pos = tr_T[:, :3, 3]
    rv = []
    for j in anchors:
        dist = np.linalg.norm(pos - pos[j], axis=1)
        ok = (np.abs(tr_t - tr_t[j]) > a.revisit_s) & (dist < a.revisit_m)
        if ok.any():
            rv.append((j, int(np.flatnonzero(ok)[np.argmin(dist[ok])])))
    pairs["revisit"] = rv
    need = set(int(j) for j in anchors) | {int(k) for v in pairs.values() for _, k in v}
    print(f"reading {len(need)} scans of {topic} ({len(rv)} revisits found within "
          f"{a.revisit_m} m and > {a.revisit_s:g} s apart)")
    scans = read_scans(P.dataset["bag"], topic, tr_t, need, float(s["time_tol"]),
                       float(s["lidar_min"]), float(s["lidar_max"]), a.log_window)
    if len(scans) < len(need) // 2:
        raise SystemExit(f"only {len(scans)} of {len(need)} scans found; if the "
                         "bag's log times are not near the stamps, pass --log-window 0")
    rng = np.random.default_rng(a.seed)

    def centers(Pw, n):
        return Pw[rng.choice(len(Pw), size=min(n, len(Pw)), replace=False)]

    def state(j):
        if rate[j] > a.turn_dps:
            return "turning"
        return "straight" if rate[j] < a.straight_dps and speed[j] > 0.2 else "other"

    # 1. single scans: the noise floor ----------------------------------------
    sw, sf = [], []
    for j in anchors:
        if j in scans:
            Pw = place(scans[j], tr_t, tr_T, tr_q, True, nb)
            st, _ = patches(Pw, centers(Pw, a.patches))
            sw += list(st[st[:, 1] > WALL, 0])
            sf += list(st[st[:, 1] < FLOOR, 0])
    print("\n1. SINGLE scans (the sensor's noise floor), median cm")
    print(f"   walls {cm(sw)}   floors {cm(sf)}   ({len(sw)} + {len(sf)} patches)")

    # 2. + 3. pairs ------------------------------------------------------------
    def pair_walls(j, k, deskew, off):
        A = place(scans[j], tr_t, tr_T, tr_q, deskew, nb, off)
        B = place(scans[k], tr_t, tr_T, tr_q, deskew, nb, off)
        dB, _ = cKDTree(B).query(A, distance_upper_bound=R / 2)
        both = A[np.isfinite(dB)]             # where the two see the same surface
        if len(both) < MIN_PTS:
            return None
        owner = np.r_[np.zeros(len(A), bool), np.ones(len(B), bool)]
        st, _ = patches(np.vstack([A, B]), centers(both, a.patches), owner)
        return list(st[st[:, 1] > WALL, 0])

    offsets = sorted(set(float(x) for x in a.offsets.split(",")) | {0.0})
    cols = ("all", "straight", "turning", "turn/rigid")
    table = {n: {c: [] for c in cols} for n in pairs}
    npair = {n: {c: 0 for c in cols} for n in pairs}
    timing = {(n, off): [] for n in pairs for off in offsets}
    for name, plist in pairs.items():
        for j, k in plist:
            if j not in scans or k not in scans:
                continue
            st = {state(j), state(k)}
            kind = "turning" if "turning" in st else ("straight" if st == {"straight"} else None)
            for off in (offsets if name != "0.1s" else [0.0]):
                w = pair_walls(j, k, True, off)
                if w is None:
                    continue
                timing[(name, off)] += w
                if off == 0.0:
                    table[name]["all"] += w
                    npair[name]["all"] += 1
                    if kind:
                        table[name][kind] += w
                        npair[name][kind] += 1
            if kind == "turning":
                w = pair_walls(j, k, False, 0.0)
                if w is not None:
                    table[name]["turn/rigid"] += w
                    npair[name]["turn/rigid"] += 1
    print("\n2. PAIRS: wall thickness where both scans see the same wall, median cm "
          "(pairs)")
    print("   apart    " + "".join(f"{c:>15s}" for c in cols))
    for n in pairs:
        print(f"   {n:8s} " + "".join(f"{cm(table[n][c]):>9s} ({npair[n][c]:3d})" for c in cols))
    print(f"   turning: above {a.turn_dps:g} deg/s at either scan; straight: both below "
          f"{a.straight_dps:g} deg/s and moving; turn/rigid: the turning pairs without deskew")

    names = [n for n in pairs if n != "0.1s"]
    print("\n3. TIMING: pairs with the trajectory's stamps shifted, median cm")
    print("   shift_ms " + "".join(f"{n:>10s}" for n in names))
    for off in offsets:
        print(f"   {off * 1000:+8.0f} " + "".join(f"{cm(timing[(n, off)]):>10s}" for n in names))

    # what it points at ----------------------------------------------------------
    med = lambda v: float(np.median(v)) if len(v) else np.nan
    print("\nreading:")
    print(f"   sensor noise floor {med(sw) * 100:.1f} cm (single scans)")
    base = med(table["0.1s"]["all"])           # neighbouring scans: local consistency
    # a timing offset: one stamp shift removes most of what the pairs add over
    # their neighbours. When there is one, the other checks run on wrong poses.
    for n in names:
        b = {off: med(timing[(n, off)]) for off in offsets}
        b = {k: v for k, v in b.items() if np.isfinite(v)}
        if not b or 0.0 not in b:
            continue
        best = min(b, key=b.get)
        m0, mb = b[0.0], b[best]
        if best != 0.0 and mb < 0.8 * m0 and (m0 - mb) > (mb - base):
            print(f"   TIMING: {n} pairs are thinnest with the trajectory's stamps shifted "
                  f"{best * 1000:+.0f} ms ({mb * 100:.1f} vs {m0 * 100:.1f} cm): the pose "
                  f"given for t is the pose at t {'+' if best > 0 else '-'} "
                  f"{abs(best) * 1000:.0f} ms.\n   The deskew and drift checks are skipped: "
                  "on time-shifted poses they would only echo this. Fix it, run again")
            return
    said = False
    for n in ("1s", "10s"):
        tu, sr, rg = (med(table[n][c]) for c in ("turning", "straight", "turn/rigid"))
        if np.isfinite(tu) and np.isfinite(sr) and tu > sr + 0.02 and \
                (not np.isfinite(rg) or tu > 0.8 * rg):
            print(f"   DESKEW: {n} pairs are {100 * (tu - sr):.1f} cm thicker turning than "
                  f"straight, and deskew does not thin them ({tu * 100:.1f} cm vs "
                  f"{rg * 100:.1f} cm without): the point times or the deskew are wrong")
            said = True
            break
    # drift on straight pairs (little rotation, so point timing barely matters),
    # all pairs when there are too few straight ones
    col = "straight" if all(npair[n]["straight"] >= 3 for n in pairs) else "all"
    w = {n: med(table[n][col]) for n in pairs}
    w = {n: v for n, v in w.items() if np.isfinite(v)}
    if "0.1s" in w and len(w) > 1:
        grow = {n: v - w["0.1s"] for n, v in w.items() if n != "0.1s"}
        worst = max(grow, key=grow.get)
        if grow[worst] > 0.03:
            print(f"   DRIFT: {col} pairs thicken with separation ({w['0.1s'] * 100:.1f} cm "
                  f"at 0.1 s, {w[worst] * 100:.1f} cm at {worst}): the poses drift apart"
                  + (" -- worst at the revisits: loop closure / pass to pass"
                     if worst == "revisit" else ""))
            said = True
    if not said:
        print("   no timing, deskew or drift signature between these scans: the wall "
              "thickness of the map comes from something they do not sample")


if __name__ == "__main__":
    main()

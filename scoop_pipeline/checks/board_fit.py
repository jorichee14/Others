#!/usr/bin/env python3
"""Why a depth track does not take a board sighting: at the depth frames with
one (04's cached board_fixes_<track>.json), each copy of the board's design
puts the robot somewhere; the frame is registered to the map from there (as
depth_track's from_board does) -- its status, points on the map (share),
how far ICP moved it -- and compared with where the track has it.

    python scoop_pipeline/checks/board_fit.py <pipeline config> [--track mobile_2_depth]
        [--from 0] [--to 1e9] [--samples 30]

--from/--to: seconds from the track's first pose.
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--track", default="mobile_2_depth")
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--to", dest="t_to", type=float, default=1e9)
    ap.add_argument("--samples", type=int, default=30)
    a = ap.parse_args()
    from pipeline_common import load_pipeline
    from scoop import tftree
    A = importlib.import_module("02_refine_poses")
    R04 = importlib.import_module("04_reference_traj")
    import depth_track as DT
    P = load_pipeline(a.config)
    c = dict(R04.DEFAULTS)
    c.update(P.cfg.get("04_reference") or {})
    tr = [t for t in c.get("tracks") or [] if t["name"] == a.track]
    if not tr:
        raise SystemExit("no track %r in 04_reference.tracks" % a.track)
    c = {k: v for k, v in c.items() if k != "tracks"}
    c.update(DT.DEFAULTS)
    c.update(tr[0])
    c.setdefault("nn_prior_beta", c["prior_beta"])
    od = P.reference_dir()
    cache = os.path.join(od, "board_fixes_%s.json" % a.track)
    if not os.path.exists(cache):
        raise SystemExit("no %s: run 04 --track %s first" % (cache, a.track))
    fixes = [(f["t"], f["board"], np.array(f["T"]), f["range"], f["reproj"])
             for f in json.load(open(cache))["fixes"]]
    t2, T2 = A.load_traj(os.path.join(od, "traj_%s.tum" % a.track))
    t0 = t2[0]
    fixes = [f for f in fixes if a.t_from <= f[0] - t0 <= a.t_to]
    if not fixes:
        raise SystemExit("no board sighting in %.0f-%.0f s" % (a.t_from, a.t_to))
    ft = np.array([f[0] for f in fixes])
    print("%d sightings in %.0f-%.0f s; where each copy puts the body (median):"
          % (len(fixes), ft.min() - t0, ft.max() - t0))
    for b in sorted({f[1] for f in fixes}):
        p = np.median([f[2][:3, 3] for f in fixes if f[1] == b], axis=0)
        r = np.median([f[3] for f in fixes if f[1] == b])
        e = np.median([f[4] for f in fixes if f[1] == b])
        print("  %-10s (%.2f, %.2f, %.2f)   range %.2f m, reproj %.2f px" % (b, *p, r, e))

    bag = os.path.expanduser(c.get("bag") or P.dataset["bag"])
    edges = tftree.tf_edges([bag], seconds=1.0)
    T_bd = DT.rel_pose(edges, c["body_frame"], c["depth_frame"])
    times, scans = DT.read_frames(bag, c, T_bd, ft.min() - c["board_tol_s"], ft.max() + c["board_tol_s"])
    fix_at = DT.fix_lookup(fixes, times, float(c["board_tol_s"]))
    idx = sorted(fix_at)
    idx = [idx[k] for k in np.linspace(0, len(idx) - 1, min(a.samples, len(idx))).astype(int)]
    ref_map = P.ref_file("anchored_map") if P.reference else P.outp("map_final_{tag}_anchored.pcd")
    ref = R04.load_reference(ref_map, c)
    print("\nthe frame registered from each copy's pose (from_board takes it when ok and "
          "moved <= %.2f m; ok needs >= %d points and share >= %.2f on the map)"
          % (float(c["board_reset_m"]), int(c["min_corr"]), float(c["track_min_fit"])))
    print("%8s %6s  %-10s %-15s %6s %6s %7s   %s" % ("t s", "pts", "copy", "status", "n_corr",
                                                   "share", "moved", "track: where, m off this copy"))
    stat = {}
    for j in idx:
        k = int(np.argmin(np.abs(t2 - times[j])))
        Tt = T2[k] if abs(t2[k] - times[j]) <= 0.2 else None
        for b, Tb in fix_at[j]:
            T, status, row, share = DT.step(A, ref, scans[j], Tb, c)
            take = status == "ok" and row[3] <= float(c["board_reset_m"])
            stat.setdefault(b, []).append((take, status, share, row[3]))
            print("%8.1f %6d  %-10s %-15s %6d %6.2f %7.2f   %s"
                  % (times[j] - t0, len(scans[j]), b, status + (" TAKEN" if take else ""), row[1],
                     share, row[3],
                     "(%.1f, %.1f), %.2f" % (Tt[0, 3], Tt[1, 3], np.linalg.norm(Tt[:3, 3] - Tb[:3, 3]))
                     if Tt is not None else "no pose"))
    print("\nper copy: taken / tried, statuses, median share, median ICP move")
    for b, v in sorted(stat.items()):
        sts = {}
        for x in v:
            sts[x[1]] = sts.get(x[1], 0) + 1
        print("  %-10s %d / %d   %s   share %.2f   moved %.2f m"
              % (b, sum(x[0] for x in v), len(v), ", ".join("%s %d" % kv for kv in sts.items()),
                 np.median([x[2] for x in v]), np.nanmedian([x[3] for x in v])))


if __name__ == "__main__":
    main()

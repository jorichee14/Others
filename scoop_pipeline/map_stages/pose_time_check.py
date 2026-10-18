#!/usr/bin/env python3
"""Is GLIM's pose time its scans' time? Checked on every pass, against the
trajectory the scans themselves fix.

    python3 map_stages/pose_time_check.py <pipeline config> [--traj FILE]
        [--ref FILE] [--out JSON]

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
the turning (analysis/odom/compare_trajs.py heading_vs_turning) gives dt,
its correlation r whether the error is timing at all.

  on time   |dt| < 10 ms or r < 0.5 (20261015 with the correction: 3.9 ms,
            r 0.23; without: 45.8 ms, r 0.92)
  OFF       set glim.pose_time_offset to (the applied one + dt) for this
            machine in configs/recording.yaml, re-export GLIM
            (processing/glim_check.py), and run the pass again from 02

Exits 0 either way: 02's reference and 04's track are fixed by the scans
(references built from GLIM before and after the 46 ms correction agree to
0.5 cm), so only GLIM's own trajectory depends on the offset.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "analysis", "odom"))

from pipeline_common import load_pipeline, load_traj                     # noqa: E402

TIMING_MS, TIMING_R = 10.0, 0.5           # OFF beyond both


def applied_offset(traj):
    """The pose_time_offset (s) GLIM's export applied: glim.yaml beside it."""
    meta = os.path.join(os.path.dirname(traj), "glim.yaml")
    if os.path.isfile(meta):
        for ln in open(meta):
            if ln.startswith("pose_time_offset_s:"):
                return float(ln.split(":", 1)[1].strip() or 0)
    return 0.0


def check(t, T, tr, Tr, applied_s=0.0):
    """Heading of (t, T) against the reference (tr, Tr) vs its turning ->
    dict(dt_ms, r_timing, r_scale, scale_pct, applied_offset_ms, on_time,
    offset_to_set_s)."""
    from compare_trajs import heading_vs_turning
    h = heading_vs_turning(t, T, tr, Tr)
    dt, r = h.get("dt_ms", 0.0), h.get("r_timing", 0.0)
    h.update(applied_offset_ms=applied_s * 1000,
             on_time=not (abs(dt) >= TIMING_MS and r >= TIMING_R),
             offset_to_set_s=round(applied_s + dt / 1000, 3))
    return h


def verdict(h):
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


def reference(P):
    """02's output (a mapping pass) or 04's LiDAR track (a run)."""
    if P.reference:
        return P.lidar_track_traj()
    return P.outp((P.cfg.get("02_bundle_adjust") or {}).get("output") or "traj_lidar_ba.txt")


def odometry_file(P, name):
    """odometry/<machine>/<name>, whatever ROUTES would do with the name."""
    return os.path.join(os.path.dirname(P.outp("traj_lidar.txt")), name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--traj")
    ap.add_argument("--ref")
    ap.add_argument("--out")
    a = ap.parse_args()
    P = load_pipeline(a.config)
    traj = os.path.expanduser(a.traj) if a.traj else P.seed_traj()
    ref = os.path.expanduser(a.ref) if a.ref else reference(P)
    out = a.out or odometry_file(P, "pose_time_check.json")
    for name, f in (("GLIM", traj), ("reference", ref)):
        if not os.path.isfile(f):           # nothing to check: not an error of the pass
            print(f"pose time not checked: no {name} trajectory at {f}")
            return
    t, T = load_traj(traj)
    tr, Tr = load_traj(ref)
    print(f"GLIM       {traj} ({len(t)} poses)\nreference  {ref} ({len(tr)} poses)")
    h = check(t, T, tr, Tr, applied_offset(traj))
    for ln in verdict(h):
        print(ln)
    h.update(traj=traj, reference=ref)
    with open(out, "w") as fh:
        json.dump(h, fh, indent=1)
    print(f"report: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Put recordings of several cameras on one clock from the motion they see (no flash needed).

    python motion_sync.py rec/                         # every cam_* video there
    python motion_sync.py rec/ --ref cam_10 --seconds 600
    python motion_sync.py rec/ --check-drift

One recording per camera, made at about the same time (offsets up to --max-lag
seconds). People moving in the room make the cameras that see them change at
the same moment; the offsets come from cross-correlating those changes (see
ed305/sync.py). Needs some movement in the first --seconds (people walking
around is plenty; an empty room is not).

Writes next to each video cam_N_times.csv (frame, pts_s, t_s): the same format
as flash_sync.py, so multiview_label.py and room_track.py use it as it is.
t_s is on the reference camera's clock. Prints per camera the offset and how
well the camera pairs agree (the residual, in milliseconds: a frame is 33 ms).
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ed305.sync import motion_signal, pair_offset, solve_offsets
from ed305.yolo_data import camera_of

EXT = (".mkv", ".mp4", ".avi", ".mov")


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rec", type=Path, help="folder with one cam_N video per camera")
    ap.add_argument("--ref", help="camera whose clock is the common clock (default: the first)")
    ap.add_argument("--seconds", type=float, default=600, help="seconds of video to compare (default 600)")
    ap.add_argument("--start", type=float, default=0.0, help="start this far into each file (s)")
    ap.add_argument("--max-lag", type=float, default=10.0, help="largest offset searched (s, default 10)")
    ap.add_argument("--min-conf", type=float, default=6.0, help="pairs below this confidence are not used")
    ap.add_argument("--tol", type=float, default=0.05, help="pairs disagreeing more (s) are dropped")
    ap.add_argument("--check-drift", action="store_true",
                    help="also fit the next --seconds of video and compare (reads twice as much)")
    ap.add_argument("--workers", type=int, default=4)
    return ap.parse_args()


def fit(names, sig, a, lo=None, hi=None):
    pairs = {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            (tx, mx), (ty, my) = sig[x], sig[y]
            if lo is not None:
                kx, ky = (tx >= lo) & (tx < hi), (ty >= lo) & (ty < hi)
                tx, mx, ty, my = tx[kx], mx[kx], ty[ky], my[ky]
            if len(tx) < 100 or len(ty) < 100:
                continue
            pairs[(x, y)] = pair_offset(tx, mx, ty, my, max_lag=a.max_lag)
    return solve_offsets(names, pairs, a.min_conf, a.tol, a.ref)


def main():
    a = parse()
    vids = {}
    for p in sorted(a.rec.iterdir()):
        if p.suffix.lower() in EXT and p.name.lower().startswith("cam"):
            if camera_of(p) in vids:
                raise SystemExit(f"two videos for {camera_of(p)}: {vids[camera_of(p)].name}, {p.name}; keep one per camera")
            vids[camera_of(p)] = p
    names = sorted(vids, key=lambda n: (len(n), n))
    if len(names) < 2:
        raise SystemExit(f"need 2+ cam_* videos in {a.rec}")
    a.ref = a.ref or names[0]
    span = a.seconds * (2 if a.check_drift else 1)
    print(f"{len(names)} cameras, reading {span:.0f} s of motion each ...")
    with ProcessPoolExecutor(a.workers) as ex:
        out = dict(zip(names, ex.map(motion_signal, [vids[n] for n in names], [span] * len(names),
                                     [a.start] * len(names))))
    sig = {n: (o[1], o[2]) for n, o in out.items()}
    off, res, used = fit(names, sig, a, a.start, a.start + a.seconds)
    per_cam = {n: [abs(r) * 1000 for k, r in res.items() if n in k] for n in names}
    print(f"\n{'camera':8s} {'offset s':>9s} {'pairs':>6s} {'residual ms':>12s}")
    for n in names:
        o = off[n]
        r = per_cam[n]
        print(f"{n:8s} {('%.3f' % o) if o is not None else 'NOT SYNCED':>9s} {len(r):6d} "
              f"{('%.0f' % np.median(r)) if r else '-':>12s}")
        if o is None:
            continue
        with open(vids[n].with_name(vids[n].stem + "_times.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "pts_s", "t_s"])
            for i, t in enumerate(out[n][0]):
                w.writerow([i, f"{t:.4f}", f"{t + o:.4f}"])
    bad = [n for n in names if off[n] is None]
    if bad:
        print(f"\nNOT SYNCED (no camera pair with enough shared motion): {bad}. More --seconds with people "
              f"walking, or a larger --max-lag; their times files were not written.")
    if a.check_drift:
        o2, res2, _ = fit(names, sig, a, a.start + a.seconds, a.start + 2 * a.seconds)
        print(f"\ndrift: offset from the next {a.seconds:.0f} s minus offset above, ms "
              f"(a few ms is noise; tens of ms or more = that camera's clock drifts or it dropped frames)")
        for n in names:
            pairs = sum(n in k for k in res2)
            if off[n] is not None and o2.get(n) is not None:
                print(f"  {n:8s} {1000 * (o2[n] - off[n]):+8.1f}   ({pairs} pairs)")
            else:
                print(f"  {n:8s} {'-':>8s}   (no reliable pair in the second window: too little motion, or the recording ends)")
    print(f"\ntimes files written next to the videos; common clock = {a.ref}'s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
How far the ZED's own tracking (its map frame, map_zed, from the bag's /tf)
drifts from the LiDAR trajectory over a run, both as the ZED left optical
frame in `map`.

  ZED:   T_map_mapzed @ T_mapzed_cam(t)        (map -> map_zed from 05:
                                               measured at the run's start)
  LiDAR: stage 04's traj_<name>_in_cam.tum (a run), or 02's refined
         trajectory @ T_lidar_camera (a mapping pass)

Both start together (map -> map_zed is fixed at the start board), so the gap
is the ZED tracking's drift. Anything built on the ZED's map_zed later in the
run -- e.g. the ChArUco 'pose_in_map' of an infra camera's extrinsic yaml --
carries that drift.

    python3 analysis/odom/zed_vs_lidar.py map_stages/pipeline_config_survey_1.json \
        [--at 1790230600.77] [--every 60]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "map_stages"))

from pipeline_common import (load_pipeline, load_traj, make_T, traj_quats,  # noqa: E402
                             interp_poses)
from pipeline_boards import read_tf_and_info                                 # noqa: E402


def ang_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def yaw_deg(R):
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--at", type=float, nargs="*", default=[],
                    help="stamps (s) to report as well, e.g. an extrinsic yaml's stamp")
    ap.add_argument("--every", type=float, default=60.0, help="table spacing (s)")
    ap.add_argument("--zed-map", default=None, help="default: 05_anchor.zed_map_frame")
    a = ap.parse_args()

    P = load_pipeline(a.config)
    s03 = P.cfg.get("05_anchor", {})
    zed_map = a.zed_map or s03.get("zed_map_frame", "map_zed")
    cam = s03.get("cam_frame", "zed_left_camera_optical_frame")
    T_lc = np.asarray(P.sensor.T_lidar_camera, float)

    if P.reference:
        traj = P.lidar_track_traj("_in_cam")
        tr_t, tr_T = load_traj(traj)
        frames = P.outp("boards_{tag}.json")
    else:
        traj = P.outp(P.dataset["traj"])
        tr_t, tr_T = load_traj(traj)
        tr_T = np.einsum("nij,jk->nik", np.asarray(tr_T), T_lc)
        frames = P.outp("anchor_frame.json")
    tr_t, tr_T = np.asarray(tr_t), np.asarray(tr_T)
    af = json.load(open(frames))
    if "T_map_mapzed" in af:
        T_map_mz = np.array(af["T_map_mapzed"], float)
    elif "map_to_mapzed_xyzquat" in af:
        v = af["map_to_mapzed_xyzquat"]
        T_map_mz = make_T(v[:3], v[3:])
    else:
        raise SystemExit("%s has no map -> map_zed (re-run 05)" % frames)
    print("LiDAR trajectory (camera frame, in map): %s" % traj)
    print("map -> %s: %s" % (zed_map, frames))
    print("reading /tf of %s ..." % P.dataset["bag"])
    _, tree, _ = read_tf_and_info(P.dataset["bag"], [], want_tf=True, verbose=False)

    tq = traj_quats(tr_T)
    stamps = np.arange(tr_t[0], tr_t[-1], 1.0)
    rows = []
    for t in list(stamps) + [float(x) for x in a.at]:
        Tz = tree.lookup(zed_map, cam, t)
        if Tz is None:
            continue
        R, p = interp_poses(tr_t, tr_T, tq, np.array([t]))
        T_l = np.eye(4)
        T_l[:3, :3], T_l[:3, 3] = R[0], p[0]
        T_z = T_map_mz @ Tz
        d = T_z[:3, 3] - T_l[:3, 3]
        rows.append((t, np.linalg.norm(d) * 100, np.linalg.norm(d[:2]) * 100, d[2] * 100,
                     ang_deg(T_l[:3, :3].T @ T_z[:3, :3]),
                     (yaw_deg(T_z[:3, :3]) - yaw_deg(T_l[:3, :3]) + 180) % 360 - 180))
    if not rows:
        raise SystemExit("no stamp where %s -> %s resolves in the bag's TF within the "
                         "trajectory; frames there: %s" % (zed_map, cam, tree.frames()))
    grid = [r for r in rows if r[0] in set(stamps)]
    t0 = tr_t[0]
    print("\nZED tracking (map_zed) vs LiDAR, ZED left optical frame in map:")
    print("%8s %10s %10s %8s %9s %9s" % ("t s", "gap cm", "horiz cm", "dz cm", "rot deg",
                                          "yaw deg"))
    last = -1e9
    for r in grid:
        if r[0] - last >= a.every - 1e-6 or r is grid[-1]:
            print("%8.0f %10.1f %10.1f %8.1f %9.2f %9.2f" % ((r[0] - t0,) + r[1:]))
            last = r[0]
    g = np.array([r[1] for r in grid])
    print("over the run: median %.1f cm, max %.1f cm (at t=%.0f s); rotation max %.2f deg"
          % (np.median(g), g.max(), grid[int(np.argmax(g))][0] - t0,
             max(r[4] for r in grid)))
    for r in rows[len(grid):]:
        print("at %.3f (t=%.1f s): gap %.1f cm (horizontal %.1f, dz %.1f), rotation %.2f deg, "
              "yaw %.2f deg" % (r[0], r[0] - t0, r[1], r[2], r[3], r[4], r[5]))


if __name__ == "__main__":
    main()

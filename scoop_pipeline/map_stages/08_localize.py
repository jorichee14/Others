#!/usr/bin/env python3
"""
STAGE 08 - put a run (a pass with no map of its own) into a reference pass's map.

  run bag + its GLIM trajectory (own world)  +  reference pass's denoised map
  ->  one rigid transform  reference world <- run world
  ->  traj_lidar_seed.txt  (the run's GLIM poses, in the reference world)

then 01a_refine_poses.py, seeded from it, registers every scan of the run to
the frozen reference map (dataset.reference_pass makes 01a do that), and 04
places the run's infra cameras in the reference pass's anchored map frame.

How the transform is found. GLIM starts every run in its own world (origin =
where the run started), so the run and the reference differ by an unknown
rigid motion. The run's scans (every `stride`-th, through its GLIM poses) are
accumulated into one cloud and registered to the reference map:

  1. global: FPFH features + RANSAC, `tries` times; a candidate that tilts
     the run (both GLIM worlds are gravity-aligned) by more than max_tilt_deg
     is dropped;
  2. each candidate polished by point-to-plane ICP, coarse to fine;
  3. the candidate with the most run points on the reference map (within
     `fit_m`) wins. A second, different candidate scoring nearly as well is
     reported: a symmetric building can fit two ways.

`init` [x, y, z, yaw_deg] skips step 1 and starts ICP from that transform --
for when the run covers too little of the map for the features to agree.

Config block "08_localize" (all optional):
  "run_traj":      <work>/<date>/<pass>/<machine>/glim/traj_lidar.txt
  "map":           the reference pass's denoised_<tag>.pcd
  "output":        "traj_lidar_seed.txt"  (dataset.traj must name it, for 01a)
  "stride":        5       # every Nth scan into the run cloud
  "voxel":         0.10    # ICP resolution (m)
  "feature_voxel": 0.30    # FPFH / RANSAC resolution (m)
  "tries":         5
  "max_tilt_deg":  5.0
  "fit_m":         0.10    # a run point within this of the map is "on" it
  "min_fitness":   0.30    # below this, stop and say so
  "init":          null    # [x, y, z, yaw_deg], reference world <- run world

  python3 08_localize.py pipeline_config_<run>.json
"""
import importlib
import json
import os
import sys

import numpy as np
import open3d as o3d

from pipeline_common import load_pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_01a = importlib.import_module("01a_refine_poses")
load_traj, write_traj, iter_scans = _01a.load_traj, _01a.write_traj, _01a.iter_scans
detect_points_topic, VoxelAvg = _01a.detect_points_topic, _01a.VoxelAvg

REG = o3d.pipelines.registration


def tilt_deg(R):
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


def yaw_deg(R):
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def T_from_init(v):
    x, y, z, yaw = map(float, v)
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = [x, y, z]
    return T


def cloud(pts, voxel, normals=True):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pc = pc.voxel_down_sample(voxel)
    if normals:
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5,
                                                                 max_nn=30))
    return pc


def icp(src, tgt, T0, voxel):
    T = T0
    for d in (10 * voxel, 5 * voxel, 2.5 * voxel, voxel):
        T = REG.registration_icp(src, tgt, d, T, REG.TransformationEstimationPointToPlane(),
                                 REG.ICPConvergenceCriteria(max_iteration=40)).transformation
    return T


def run_cloud(P, bag, topic, times, T, s01, stride):
    acc, n = VoxelAvg(0.05), 0
    for j, p in iter_scans(bag, topic, times, s01["time_tol"], s01["lidar_min"],
                           s01["lidar_max"]):
        if j % stride:
            continue
        acc.add(p @ T[j][:3, :3].T + T[j][:3, 3])
        n += 1
        if n % 500 == 0:
            print(f"    {n} scans", flush=True)
    pts = acc.points()
    print(f"    run cloud: {n} scans (every {stride}th) -> {len(pts)} points")
    return pts


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "pipeline_config.json"
    P = load_pipeline(cfg_path)
    s = dict(stride=5, voxel=0.10, feature_voxel=0.30, tries=5, max_tilt_deg=5.0,
             fit_m=0.10, min_fitness=0.30, init=None, output="traj_lidar_seed.txt")
    s.update(P.cfg.get("08_localize", {}))
    s01 = P.cfg["01_build_map"]
    bag = P.dataset["bag"]
    print(P.describe())

    run_traj = s.get("run_traj") or os.path.join(os.path.dirname(os.path.normpath(bag)),
                                                 P.machine, "glim", "traj_lidar.txt")
    run_traj = os.path.expanduser(run_traj)
    ref_map = os.path.expanduser(s.get("map") or P.ref_file("map"))
    for f, what in ((run_traj, "run trajectory (08_localize.run_traj)"),
                    (ref_map, "reference map (08_localize.map / dataset.reference_pass)")):
        if not os.path.exists(f):
            raise SystemExit(f"{what} not found: {f}")
    times, T_run = load_traj(run_traj)
    print(f"run trajectory: {run_traj}  ({len(times)} poses, "
          f"{times[-1] - times[0]:.1f} s)")
    print(f"reference map:  {ref_map}")
    topic = detect_points_topic(bag, P.cfg.get("01a_refine", {}).get("points_topic", ""),
                                len(times))

    print("\n[1] run cloud through its own poses")
    src_pts = run_cloud(P, bag, topic, times, T_run, s01, int(s["stride"]))
    print("[2] reference map")
    ref = o3d.io.read_point_cloud(ref_map)
    print(f"    {len(ref.points)} points")
    v, fv = float(s["voxel"]), float(s["feature_voxel"])
    tgt = cloud(np.asarray(ref.points), v)
    del ref
    src = cloud(src_pts, v)
    print(f"    ICP clouds @ {v} m: run {len(src.points)}, reference {len(tgt.points)}")

    def score(T):
        e = REG.evaluate_registration(src, tgt, float(s["fit_m"]), T)
        cov = REG.evaluate_registration(src, tgt, 3 * float(s["fit_m"]), T).fitness
        return e.fitness, e.inlier_rmse, cov

    cands = []
    if s.get("init"):
        print(f"\n[3] init from config {s['init']} (no global search)")
        T = icp(src, tgt, T_from_init(s["init"]), v)
        cands.append((T,) + score(T))
    else:
        print(f"\n[3] global registration: FPFH + RANSAC @ {fv} m, {s['tries']} tries")
        fs_src, fs_tgt = cloud(src_pts, fv), cloud(np.asarray(tgt.points), fv)
        feat = o3d.geometry.KDTreeSearchParamHybrid(radius=fv * 5, max_nn=100)
        f_src = REG.compute_fpfh_feature(fs_src, feat)
        f_tgt = REG.compute_fpfh_feature(fs_tgt, feat)
        for k in range(int(s["tries"])):
            o3d.utility.random.seed(k)
            g = REG.registration_ransac_based_on_feature_matching(
                fs_src, fs_tgt, f_src, f_tgt, True, 1.5 * fv,
                REG.TransformationEstimationPointToPoint(False), 3,
                [REG.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                 REG.CorrespondenceCheckerBasedOnDistance(1.5 * fv)],
                REG.RANSACConvergenceCriteria(1_000_000, 0.999))
            T0 = g.transformation
            if tilt_deg(T0[:3, :3]) > float(s["max_tilt_deg"]):
                print(f"    try {k}: tilt {tilt_deg(T0[:3, :3]):.1f} deg -> dropped")
                continue
            T = icp(src, tgt, T0, v)
            f, rmse, cov = score(T)
            print(f"    try {k}: on map {f * 100:5.1f} %  rmse {rmse * 100:.1f} cm  "
                  f"t={np.round(T[:3, 3], 2).tolist()} yaw {yaw_deg(T[:3, :3]):.1f} deg "
                  f"tilt {tilt_deg(T[:3, :3]):.2f} deg")
            if tilt_deg(T[:3, :3]) <= float(s["max_tilt_deg"]):
                cands.append((T, f, rmse, cov))
    if not cands:
        raise SystemExit("no usable alignment: every candidate tilted the run. Give "
                         "08_localize.init [x, y, z, yaw_deg] (run start in the reference map)")
    cands.sort(key=lambda c: -c[1])
    T, f, rmse, cov = cands[0]
    for T2, f2, *_ in cands[1:]:
        dt = np.linalg.norm(T2[:3, 3] - T[:3, 3])
        dyaw = abs((yaw_deg(T2[:3, :3]) - yaw_deg(T[:3, :3]) + 180) % 360 - 180)
        if (dt > 1.0 or dyaw > 10) and f2 > 0.8 * f:
            print(f"  ! another fit scores nearly as well ({f2 * 100:.1f} % on map) "
                  f"{dt:.1f} m / {dyaw:.0f} deg away: check the result, or pin it "
                  "with 08_localize.init")
            break

    print(f"\nreference world <- run world: t={np.round(T[:3, 3], 4).tolist()} m  "
          f"yaw {yaw_deg(T[:3, :3]):.2f} deg  tilt {tilt_deg(T[:3, :3]):.2f} deg")
    print(f"  run points on the reference map: {f * 100:.1f} % within "
          f"{s['fit_m'] * 100:.0f} cm (rmse {rmse * 100:.2f} cm), {cov * 100:.1f} % within "
          f"{3 * s['fit_m'] * 100:.0f} cm")
    if f < float(s["min_fitness"]):
        raise SystemExit(f"only {f * 100:.1f} % of the run lies on the reference map "
                         f"(min_fitness {s['min_fitness']}): wrong reference, a run "
                         "mostly outside the mapped area, or a wrong fit -- give "
                         "08_localize.init")
    if cov < 0.9:
        print(f"  note: {100 - cov * 100:.0f} % of the run is not near the reference map "
              "(outside it, or things that moved); 01a keeps those scans' seed poses")

    out = P.outp(s["output"])
    write_traj(out, times, np.einsum("ij,njk->nik", T, T_run))
    rec = {"run_traj": run_traj, "reference_map": ref_map,
           "T_ref_run": T.tolist(), "fitness": f, "inlier_rmse_m": rmse, "coverage": cov,
           "fit_m": s["fit_m"], "init": s.get("init")}
    jp = P.outp(f"localize_{P.tag}.json" if P.tag else "localize.json")
    json.dump(rec, open(jp, "w"), indent=2)
    print(f"\nwrote {out}\n      {jp}")
    if os.path.basename(P.dataset.get("traj", "")) != os.path.basename(out):
        print(f"  ! dataset.traj is {P.dataset.get('traj')!r}: set it to "
              f"\"{s['output']}\" so 01a starts from these poses")
    print(f"next: python3 01a_refine_poses.py {cfg_path}")


if __name__ == "__main__":
    main()

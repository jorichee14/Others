#!/usr/bin/env python
"""Self-test for 2Fast-2Lamaa in the odometry harness, no ROS: make_config's
fast2lamaa and fast2lamaa_loc on test_odometry's synthetic bag, score.py's
in-map error, analysis/slam/relative_pose_loc.py and slam/odometry/compare.py.

fast2lamaa must give the launch file both nodes' parameters with the LiDAR in
the IMU frame (translation, rotation vector) from the bag, the IMU as it is
(not inverted), the bag's clock, and the pose topic in the IMU frame.
fast2lamaa_loc must also write the map as gp_map loads it (binary PLY of
doubles, voxel-thinned, and its .info), start at the reference's IMU pose at
the first scan, mark its world as map, and refuse without a reference.
score must report the unaligned in-map error of a run in map: 0 for a
perfect one, the offset for one put 10 cm off (which the rigid fit hides).
relative_pose_loc must find 0 between two perfect robots and the offset of
one robot moved 0.2 m in map. compare must list the runs and the
robot-to-robot files.

    python scoop_pipeline/tests/test_fast2lamaa.py
"""
import ast
import json
import os
import shutil
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "slam", "odometry"))
sys.path.insert(0, os.path.join(ROOT, "analysis", "slam"))

import test_odometry as TO                                          # noqa: E402
import make_config as MK                                            # noqa: E402


def launch_params(run):
    """The ODOM and MAP dicts of the generated launch file, read without ROS."""
    tree = ast.parse(open(os.path.join(run, "config", "fast2lamaa.launch.py")).read())
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("ODOM", "MAP"):
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out["ODOM"], out["MAP"]


def read_ply(path):
    raw = open(path, "rb").read()
    head, _, body = raw.partition(b"end_header\n")
    n = int([l for l in head.decode().splitlines() if l.startswith("element vertex")][0].split()[-1])
    return np.frombuffer(body, "<f8").reshape(n, 3), head.decode()


def check_config(tmp, f):
    bag = os.path.join(tmp, "bag")
    TO.write_bag(bag)
    run = lambda name: os.path.join(tmp, "runs", name)                 # noqa: E731
    T_imu_lidar = np.linalg.inv(TO.T_of(*TO.T_SENSOR_IMU)) @ TO.T_of(*TO.T_SENSOR_LIDAR)

    r = MK.make("fast2lamaa", bag, "mobile_1", run("f2l"))
    odom, gmap = launch_params(run("f2l"))
    c = np.eye(4)
    c[:3, :3] = Rotation.from_rotvec([odom["calib_rx"], odom["calib_ry"], odom["calib_rz"]]).as_matrix()
    c[:3, 3] = [odom["calib_px"], odom["calib_py"], odom["calib_pz"]]
    if np.abs(c - T_imu_lidar).max() > 1e-9:
        f.append("fast2lamaa: calib\n%s\nwant\n%s" % (c, T_imu_lidar))
    if odom["invert_imu"] or gmap["invert_imu"] or not (odom["use_sim_time"] and gmap["use_sim_time"]):
        f.append("fast2lamaa: imu / clock %s %s" % (odom, gmap))
    if odom["min_range"] != 1.0 or gmap.get("localization_only"):
        f.append("fast2lamaa: min_range %s, localization %s" % (odom["min_range"], gmap.get("localization_only")))
    if (r["out_topic"], r["out_type"], r["tum_frame"], r["world"]) != \
            ("/scan_to_map_pose", "geometry_msgs/msg/TransformStamped", "os_imu_mobile1", "odom"):
        f.append("fast2lamaa: out %s %s, frame %s, world %s" % (r["out_topic"], r["out_type"], r["tum_frame"],
                                                               r["world"]))
    if "/mobile_1/ouster/imu" not in open(os.path.join(run("f2l"), "config", "fast2lamaa.launch.py")).read() \
            or r["relay"]["out"] != "/mobile_1/ouster/points":
        f.append("fast2lamaa: topics / relay %s" % r["relay"])

    # localization: a map (a 2 cm grid of a box room: thinned to 5 cm), the reference's first pose
    TO_ref = os.path.join(tmp, "ref.tum")
    tr, Tr = TO.ref_traj()
    TO.write_tum(TO_ref, tr, Tr)
    g = np.arange(-4, 4, 0.02)
    P = np.array([[x, y, 0.0] for x in g for y in g[::10]])
    mp = os.path.join(tmp, "map.ply")
    MK.write_ply(mp, P)
    refused = False
    try:
        MK.make("fast2lamaa_loc", bag, "mobile_1", run("loc_noref"), map_path=mp)
    except MK.Refused:
        refused = True
    if not refused:
        f.append("fast2lamaa_loc without a reference: not refused")
    r = MK.make("fast2lamaa_loc", bag, "mobile_1", run("loc"), map_path=mp, ref=TO_ref)
    odom, gmap = launch_params(run("loc"))
    Q, head = read_ply(os.path.join(run("loc"), "map", "map.ply"))
    if "property double x" not in head or not (0 < len(Q) < len(P)) or np.abs(Q[:, 2]).max() > 1e-9:
        f.append("fast2lamaa_loc: map %d points of %d, header %s" % (len(Q), len(P), head))
    if open(os.path.join(run("loc"), "map", "map.info")).read().split() != ["cell_size", "0.15"]:
        f.append("fast2lamaa_loc: map.info")
    Ti = np.eye(4)
    Ti[:3, :3] = Rotation.from_rotvec([gmap["init_pose_rx"], gmap["init_pose_ry"], gmap["init_pose_rz"]]).as_matrix()
    Ti[:3, 3] = [gmap["init_pose_x"], gmap["init_pose_y"], gmap["init_pose_z"]]
    want = Tr[0] @ np.linalg.inv(T_imu_lidar)                          # T_map_imu at the first scan
    if not gmap["localization_only"] or np.abs(Ti - want).max() > 1e-6 or r["world"] != "map" \
            or gmap["map_path"] != "/work/map/":
        f.append("fast2lamaa_loc: init\n%s\nwant\n%s, world %s, map_path %s" % (Ti, want, r["world"], gmap["map_path"]))
    if gmap["can_trust_init"] is not True or r["start_s"] != 0.0:
        f.append("fast2lamaa_loc: can_trust_init %s, start %s" % (gmap["can_trust_init"], r["start_s"]))
    # a reference that starts 2 s into the bag: the play starts there, the seed is its first pose
    late = os.path.join(tmp, "ref_late.tum")
    TO.write_tum(late, tr[20:], Tr[20:])
    r = MK.make("fast2lamaa_loc", bag, "mobile_1", run("loc_late"), map_path=mp, ref=late, coarse_init=True)
    odom, gmap = launch_params(run("loc_late"))
    Ti[:3, :3] = Rotation.from_rotvec([gmap["init_pose_rx"], gmap["init_pose_ry"], gmap["init_pose_rz"]]).as_matrix()
    Ti[:3, 3] = [gmap["init_pose_x"], gmap["init_pose_y"], gmap["init_pose_z"]]
    want = Tr[20] @ np.linalg.inv(T_imu_lidar)
    if abs(r["start_s"] - 2.0) > 1e-6 or np.abs(Ti - want).max() > 1e-6 or gmap["can_trust_init"] is not False:
        f.append("fast2lamaa_loc, late reference: start %s, can_trust_init %s, init\n%s\nwant\n%s"
                 % (r["start_s"], gmap["can_trust_init"], Ti, want))
    return bag, T_imu_lidar


def check_score(tmp, f, bag, T_imu_lidar):
    import score as SC
    tr, Tr = TO.ref_traj()
    ref = os.path.join(tmp, "ref.tum")
    run = os.path.join(tmp, "runs", "loc")
    E = np.einsum("nij,jk->nik", Tr, np.linalg.inv(T_imu_lidar))     # T_map_imu, perfect
    TO.write_tum(os.path.join(run, "trajectory.tum"), tr, E)
    a = SC.run_score(run, ref)
    if "error" in a or a["in_map"]["position_cm"]["rmse"] > 1e-3 or a["in_map"]["rotation_deg"]["rmse"] > 1e-3:
        f.append("score in map, perfect: %s" % a.get("in_map", a))
    off = TO.T_of([0.1, 0.0, 0.0], [0, 0, 0, 1])
    TO.write_tum(os.path.join(run, "trajectory.tum"), tr, np.einsum("ij,njk->nik", off, E))
    a = SC.run_score(run, ref)
    if abs(a["in_map"]["position_cm"]["median"] - 10.0) > 1e-3 or a["ate"]["position_cm"]["rmse"] > 1e-3:
        f.append("score in map, 10 cm off: in map %s, ATE %s" % (a["in_map"], a["ate"]))
    if "in map, no alignment" not in SC.report(a):
        f.append("score: the report has no in-map line")
    a = SC.run_score(os.path.join(tmp, "runs", "f2l"), ref) if os.path.exists(
        os.path.join(tmp, "runs", "f2l", "trajectory.tum")) else {}
    if "in_map" in a:
        f.append("score: in-map error for a run in odom")


def check_relative(f):
    import relative_pose_loc as RL
    mc = MK.mc
    tr, Tr = TO.ref_traj()
    Tr2 = np.einsum("ij,njk->nik", TO.T_of([0.0, 0.0, 0.0], Rotation.from_euler("z", 90, degrees=True).as_quat()), Tr)

    def interp(t, T):
        return mc.PoseInterp(np.round(t * 1e9).astype(np.int64), T[:, :3, 3],
                             Rotation.from_matrix(T[:, :3, :3]).as_quat(), max_gap_s=1.0)
    gts = [interp(tr, Tr), interp(tr, Tr2)]
    S, dur = RL.series(gts, gts, 1.0)
    rep = RL.RP._summary(S, dur, 20.0, (0.5, 2.0))
    if rep["available_pct"] < 99 or rep["translation_m"]["max"] > 1e-6 or rep["usable"]["pct_of_run"] < 99:
        f.append("relative_pose_loc, perfect: %s" % rep)
    moved = np.einsum("ij,njk->nik", TO.T_of([0.2, 0.0, 0.0], [0, 0, 0, 1]), Tr2)
    S, dur = RL.series([gts[0], interp(tr, moved)], gts, 1.0)
    rep = RL.RP._summary(S, dur, 20.0, (0.5, 2.0))
    if abs(rep["translation_m"]["median"] - 0.2) > 1e-6 or rep["rotation_deg"]["max"] > 1e-6:
        f.append("relative_pose_loc, robot 1 0.2 m off: %s" % rep)


def check_compare(tmp, f):
    import compare as CP
    d = os.path.join(tmp, "cmp")
    for name, method, world in (("a", "fast_lio", "odom"), ("b", "fast2lamaa_loc", "map")):
        r = os.path.join(d, name)
        os.makedirs(r)
        json.dump({"method": method, "machine": "mobile_1", "world": world}, open(os.path.join(r, "run.json"), "w"))
        json.dump({"method": "ran", "play": "done"}, open(os.path.join(r, "status.json"), "w"))
        json.dump({"accuracy": {"span_s": 100.0, "ate": {"position_cm": {"rmse": 2.5}, "rotation_deg": {"rmse": 0.3}},
                                "in_map": {"position_cm": {"rmse": 1.2}}},
                   "resources": {"method_cpu_pct": {"median": 80.0, "p95": 120.0}, "method_rss_mb_peak": 300.0},
                   "timing": {"latency_ms": {"p95": 40.0}, "kept_up": True}}, open(os.path.join(r, "summary.json"), "w"))
    json.dump({"available_pct": 100.0, "available_from_s": 0.0, "translation_m": {"median": 0.03, "p95": 0.05},
               "rotation_deg": {"median": 0.2, "p95": 0.4}, "usable": {"pct_of_run": 100.0}},
              open(os.path.join(d, "b", "relative_pose_loc.json"), "w"))
    odo, rel = CP.runs([d])
    rows = [CP.odo_row(r) for r in odo]
    if len(rows) != 2 or rows[0][5] != "" or rows[1][5] != "1.20" or rows[0][3] != "2.50" or len(rel) != 1:
        f.append("compare: rows %s, rel %s" % (rows, rel))
    if CP.rel_row(rel[0])[1] != "localized in map":
        f.append("compare: relative row %s" % CP.rel_row(rel[0]))


def main():
    tmp = tempfile.mkdtemp()
    f = []
    try:
        bag, T_il = check_config(tmp, f)
        check_score(tmp, f, bag, T_il)
        check_relative(f)
        check_compare(tmp, f)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if f:
        print("FAIL\n  " + "\n  ".join(f))
        sys.exit(1)
    print("fast2lamaa ok")


if __name__ == "__main__":
    main()

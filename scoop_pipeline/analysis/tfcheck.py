#!/usr/bin/env python3
"""Is a robot's TF in 10's poses bag right? The base link to the LiDAR, and
every frame where the trajectory puts it.

  python3 check.py tf <pipeline config> [--bag BAG] [--robot mobile_2]
      [--lidar os_lidar_mobile2] [--every 10]

Reads the bag's /tf_static and /tf (10's <tag>_best_poses by default, or the
finalized dataset bag) and the robot's 10_publish block:

  base -> LiDAR   the static edge chain from the base link to the LiDAR, as
                  xyz and roll / pitch / yaw, against the calibration: the
                  base -> camera optical chain of the bag and T_lidar_camera
                  (dataset.calib_json when set, else the bag's /tf_static at
                  load): what the tree says the LiDAR is, against what the
                  calibration says
  LiDAR in map    at every --every'th pose on /tf: map -> ... ~~ (the moving
                  edge) -> ... -> LiDAR, against the reference trajectory in
                  map (traj_<machine>_lidar.tum, made from BA's trajectory and
                  05's anchor, not from the tree)
  camera in map   the same for the camera optical frame against
                  traj_<machine>_lidar_in_cam.tum
  global_pose     the published /<robot>/global_pose against the tree's base
                  link (or the optical frame without a base_frame)

OK within 1 mm / 0.01 deg, ! otherwise; exit 1 on any !.
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "map_stages"))
sys.path.insert(0, os.path.dirname(HERE))

from pipeline_common import load_pipeline, load_traj                     # noqa: E402

MM, DEG = 1.0, 0.01
BAD = []


def say(ok, msg):
    print(("  OK  " if ok else "  !   ") + msg)
    if not ok:
        BAD.append(msg)


def err(A, B):
    """(mm, deg) between two poses."""
    d = np.linalg.inv(A) @ B
    return (float(np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000),
            float(np.degrees(np.linalg.norm(Rotation.from_matrix(d[:3, :3]).as_rotvec()))))


def describe(T):
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    return ("xyz [%.3f, %.3f, %.3f] m, roll / pitch / yaw [%.2f, %.2f, %.2f] deg, %.3f m away"
            % (*T[:3, 3], *rpy, np.linalg.norm(T[:3, 3])))


def moving_edge(frame, edges):
    """The /tf edge above `frame` (the one the trajectory moves), or None."""
    f = frame
    while f in edges:
        if not edges[f].static:
            return f
        f = edges[f].parent
    return None


def tf_poses(bag, child):
    """{t_ns: T_parent_child} of `child` on /tf."""
    from scoop import bag as sbag
    from scoop import tftree
    out = {}
    for _, _, pl, dec in sbag.open_bag(bag).iter_raw(["/tf"]):
        for tr in dec(pl).transforms:
            if tr.child_frame_id == child:
                o, q = tr.transform.translation, tr.transform.rotation
                t_ns = tr.header.stamp.sec * 10 ** 9 + tr.header.stamp.nanosec
                out[t_ns] = tftree.matrix([o.x, o.y, o.z], [q.x, q.y, q.z, q.w])
    return out


def pose_topic(bag, topic):
    from scoop import bag as sbag
    from scoop import tftree
    out = {}
    for _, _, pl, dec in sbag.open_bag(bag).iter_raw([topic]):
        m = dec(pl)
        o, q = m.pose.position, m.pose.orientation
        out[m.header.stamp.sec * 10 ** 9 + m.header.stamp.nanosec] = \
            tftree.matrix([o.x, o.y, o.z], [q.x, q.y, q.z, q.w])
    return out


def against(name, poses, ref_file, every):
    """Poses {t_ns: T_map_x} against a TUM file of the same stamps."""
    if not os.path.isfile(ref_file):
        say(False, f"{name}: no {ref_file}")
        return
    rt, RT = load_traj(ref_file)
    ts = sorted(poses)[::every]
    worst, n = (0.0, 0.0), 0
    for t_ns in ts:
        i = int(np.argmin(np.abs(rt - t_ns * 1e-9)))
        if abs(rt[i] - t_ns * 1e-9) > 1e-3:
            continue
        e = err(poses[t_ns], RT[i])
        worst = (max(worst[0], e[0]), max(worst[1], e[1]))
        n += 1
    say(n > 0 and worst[0] < MM and worst[1] < DEG,
        f"{name}: {n} poses against {os.path.basename(ref_file)}, worst {worst[0]:.2f} mm / {worst[1]:.4f} deg")


def main_tf(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--bag", help="default: 10's poses bag")
    ap.add_argument("--robot", help="default: the dataset's machine")
    ap.add_argument("--lidar", help="the LiDAR frame (default: the points topic's frame_id)")
    ap.add_argument("--every", type=int, default=10, help="every Nth pose on /tf")
    a = ap.parse_args(argv)
    BAD.clear()
    from scoop import tftree
    import pass_tf
    P = load_pipeline(a.config)
    s = P.cfg.get("10_publish") or {}
    name = a.robot or P.machine
    rb = next((r for r in P.robots() if r["name"] == name), None)
    if rb is None:
        raise SystemExit(f"no robot {name} in 10_publish")
    bag = a.bag or P.outp(s.get("output_bag") or "{tag}_best_poses")
    optical = rb.get("optical_frame", "camera_optical_frame")
    base = (rb.get("base_frame") or {}).get("name")
    lidar = a.lidar or pass_tf._frame_id([P.dataset["bag"]], P.points_topic)
    if not lidar:
        raise SystemExit("cannot tell the LiDAR frame: pass --lidar")
    print(f"TF of {name} in {bag}\n  base {base}, LiDAR {lidar}, camera {optical}")
    E = tftree.tf_edges([bag])
    frames = set(E) | {e.parent for e in E.values()}
    for f in [x for x in (base, lidar, optical) if x]:
        if f not in frames or tftree._root(f, E) != "map":
            say(False, f"{f} is not in the tree below map")
    if BAD:
        return 1
    top = base or optical
    mv = moving_edge(top, E)
    say(mv is not None, f"the trajectory moves {mv} ({E[mv].parent} ~~ {mv} on /tf)" if mv
        else f"no /tf edge above {top}")
    if mv is None:
        return 1

    print("\nbase -> LiDAR")
    if base:
        T_bl = tftree.static_between(base, lidar, E)
        print("        " + describe(T_bl))
        try:
            T_bc = tftree.static_between(base, optical, E)
            T_cal = T_bc @ np.linalg.inv(P.sensor.T_lidar_camera)
            src = "dataset.calib_json" if P.dataset.get("calib_json") else "the bag's /tf_static"
            e = err(T_bl, T_cal)
            say(e[0] < MM and e[1] < DEG, f"the tree's {base} -> {lidar} against base -> camera and "
                f"T_lidar_camera ({src}): {e[0]:.2f} mm / {e[1]:.4f} deg")
        except Exception as ex:                                      # noqa: BLE001
            print(f"        (no calibration to set it against: {type(ex).__name__}: {ex})")
        if mv != base:
            print(f"        {mv} -> {lidar}: " + describe(tftree.static_between(mv, lidar, E)))
    else:
        print("        no base_frame for this robot")

    print("\nframes in map along the run")
    T_map_origin = tftree.static_between("map", E[mv].parent, E)
    dyn = tf_poses(bag, mv)
    say(bool(dyn), f"{len(dyn)} poses of {mv} on /tf")
    if not dyn:
        return 1
    T_ml = {t: T_map_origin @ T @ tftree.static_between(mv, lidar, E) for t, T in dyn.items()}
    against("LiDAR in map", T_ml, P.lidar_track_traj(), a.every)
    T_mc = {t: T_map_origin @ T @ tftree.static_between(mv, optical, E) for t, T in dyn.items()}
    against("camera in map", T_mc, P.lidar_track_traj("_in_cam"), a.every)

    print("\npublished poses")
    gp = pose_topic(bag, f"/{name}/global_pose")
    T_top = tftree.static_between(mv, top, E)
    worst, n = (0.0, 0.0), 0
    for t_ns in sorted(gp)[::a.every]:
        if t_ns in dyn:
            e = err(gp[t_ns], T_map_origin @ dyn[t_ns] @ T_top)
            worst = (max(worst[0], e[0]), max(worst[1], e[1]))
            n += 1
    say(n > 0 and worst[0] < MM and worst[1] < DEG,
        f"/{name}/global_pose is {top} in map: {n} poses, worst {worst[0]:.2f} mm / {worst[1]:.4f} deg")
    print(f"\n{len(BAD)} problem(s)" if BAD else "\nthe TF is consistent: base -> LiDAR as calibrated, "
          "every frame where the trajectory puts it")
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main_tf())

#!/usr/bin/env python3
"""Fixed room cameras placed in the map from their ChArUco extrinsics to the
robot's colour camera, with the robot placed by its own LiDAR: no trajectory,
the robot standing still during each calibration.

    python3 scoop_pipeline/processing/place_cameras.py <calibration folder> [<more> ...] <out folder>
        --map <anchored map .pcd/.ply> --traj <the mapping pass's camera TUM, map frame>
        [--extrinsics <folder>] [--seconds 5] [--topic auto] [--min-range 1.0]
        [--static-tf configs/static_tf.yaml] [--lidar-frame os_lidar_mobile2]
        [--camera-frame camera_color_optical_frame]

    e.g. ED305 (cam_10, cam_15, cam_16 against mobile_2's RealSense):
    python3 scoop_pipeline/processing/place_cameras.py \\
        ~/workspaces/isaac_ros-dev/data/raw/20261103/calibration \\
        ~/workspaces/isaac_ros-dev/data/processed/20261103/mapping/frames/ed305_cameras \\
        --map .../mapping/map_final_mapping_20261103_anchored.pcd \\
        --traj .../odometry/reference_mapping_20261103/traj_mobile_2_lidar_in_cam.tum

    T_map_camN = T_map_lidar . T_lidar_colour . T_colour_camN

  T_colour_camN   the calibration: <extrinsics>/extrinsic_<a>__cam_N.yaml with
                  stage: locked and parent_frame --camera-frame (default:
                  the one folder named *extrinsics* in each calibration
                  folder). Several calibration folders, oldest first: a
                  camera calibrated in more than one is taken from the last;
  T_lidar_colour  the robot's LiDAR -> colour camera calibration (--static-tf,
                  --lidar-frame -> --camera-frame);
  T_map_lidar     where the robot stood: the calibration bag of cam_N (the
                  folder under <calibration folder> whose name ends in
                  cam<N>, cam_<N> or calib_<N>), its first --seconds of LiDAR points (the
                  robot still: one sharp cloud), in the LiDAR frame (an
                  os_sensor cloud moved to os_lidar by the bag's /tf_static),
                  levelled by the LiDAR's mean tilt in the mapping pass (the
                  robot on the same floor), then placed in the map by
                  map_stages/lidar_reloc's search (every heading x every
                  offset on the wall band, point-to-plane ICP from the best
                  few) and checked: share of the cloud on the map's surfaces
                  (fit) and that no distinct placement fits nearly as well
                  (unique: a rival must leave >= 2x and >= 5 points per 100
                  more of the cloud off the map).

<out folder>:
  cameras.yaml       transforms in configs/static_tf.yaml's format: map -> cam_N
                     (the optical frame of the room camera), with the quality
                     of each step under "quality"
  cameras.json       the same, with every robot pose and candidate
  cameras_on_map.ply every camera on the whole map, one colour each (frustum
                     + 1.5 m ray along its view; the robot's spot as a small frustum).
                     Again from cameras.json without relocalizing:
                     place_cameras.py show <out folder> --map <map>
  check_cam_<N>.ply  the robot's cloud (red) on the map round it (grey), the
                     robot (blue) and the room camera (green) as small frusta
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))
sys.path.insert(0, os.path.join(ROOT, "assets"))


def T_of(t, q):
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T


def tq(T):
    from scipy.spatial.transform import Rotation as R
    return [round(float(v), 6) for v in T[:3, 3]], [round(float(v), 6) for v in R.from_matrix(T[:3, :3]).as_quat()]


def level_from_traj(traj, T_lidar_cam):
    """The LiDAR's mean roll and pitch in the map over the mapping pass, as a
    rotation that levels a cloud in the LiDAR frame (yaw left at zero)."""
    from scipy.spatial.transform import Rotation as R
    a = np.loadtxt(os.path.expanduser(traj), comments="#", ndmin=2)
    R_cam_lidar = np.linalg.inv(T_lidar_cam)[:3, :3]
    eul = []
    for row in a[:: max(1, len(a) // 400)]:
        Rl = R.from_quat(row[4:8]).as_matrix() @ R_cam_lidar
        eul.append(R.from_matrix(Rl).as_euler("ZYX"))                  # yaw, pitch, roll
    eul = np.array(eul)
    pitch, roll = float(np.median(eul[:, 1])), float(np.median(eul[:, 2]))
    L = np.eye(4)
    L[:3, :3] = R.from_euler("ZYX", [0.0, pitch, roll]).as_matrix()
    return L, np.degrees(pitch), np.degrees(roll), float(np.degrees(np.std(eul[:, 1:], axis=0)).max())


def place_robot(scan_lidar, map_pts, L, c=None, log=print):
    """-> (T_map_lidar, fit, rmse, rival dict or None, candidates)."""
    import lidar_reloc as lr
    cc = dict(lr.DEFAULTS)
    cc.update(c or {})
    grid = lr.MapGrid(map_pts, cc["grid_m"])
    tgt = lr._cloud(map_pts, cc["voxel"])
    score = lr.Scorer(tgt)
    lvl = scan_lidar @ L[:3, :3].T
    cands, _ = lr.register(lvl, tgt, grid, cc, "robot", score)
    if not cands:
        raise SystemExit("no placement of the robot's cloud kept (all tilted)")
    T, fit, rmse = cands[0]
    rival = next(((Tb, fb, rb) for Tb, fb, rb in cands[1:] if lr.distinct(T, Tb)), None)
    riv = None
    if rival is not None:
        off, off_r = 1 - fit, 1 - rival[1]
        riv = {"fit": round(rival[1], 4), "t": np.round(rival[0][:3, 3], 3).tolist(),
               "unique": bool(off_r >= cc["rival_ratio"] * off and off_r - off >= cc["rival_gap"])}
    return T @ L, fit, rmse, riv, cands


def frustum(T, size=0.3, step=0.005):
    """Points along a small camera frustum (optical frame: z forward)."""
    c = np.zeros(3)
    k = np.array([[-0.6, -0.45, 1], [0.6, -0.45, 1], [0.6, 0.45, 1], [-0.6, 0.45, 1]]) * size
    segs = [(c, p) for p in k] + [(k[i], k[(i + 1) % 4]) for i in range(4)]
    P = np.vstack([a + (b - a) * np.linspace(0, 1, int(np.linalg.norm(b - a) / step) + 2)[:, None] for a, b in segs])
    return P @ T[:3, :3].T + T[:3, 3]


def find_bag(folder, n):
    """Bag folders of camera n: named ...cam<n>, ...cam_<n>, ...calib_<n> (zeros in front allowed)."""
    pat = re.compile(r"(?:cam|calib)_?0*%d$" % n)
    hits = [d for d in sorted(glob.glob(os.path.join(folder, "*"))) if os.path.isdir(d) and pat.search(os.path.basename(d))]
    bags = []
    for d in hits:                     # the folder itself, or a bag folder inside it
        if os.path.isfile(os.path.join(d, "metadata.yaml")) or glob.glob(os.path.join(d, "*.mcap")):
            bags.append(d)
        bags += [os.path.dirname(m) for m in glob.glob(os.path.join(d, "*", "metadata.yaml"))]
    return bags


def static_cloud(bag, topic, seconds, min_range, max_range=15.0, log=print):
    """(points in the cloud's frame, frame_id, scans used) of the first seconds."""
    from scoop.bag import iter_scans, open_bag
    r = open_bag(bag)
    tps = r.topics()
    if topic == "auto":
        cand = [t for t, i in tps.items() if "PointCloud2" in i.msgtype and "ouster" in t]
        if len(cand) != 1:
            raise SystemExit("%s: which points topic? %s (--topic)" % (bag, cand or list(tps)))
        topic = cand[0]
    frame = r.first_message(topic)[1].header.frame_id.lstrip("/")
    pts, n, t0 = [], 0, None
    for sc in iter_scans(r, topic, min_range=min_range, max_range=max_range):
        t0 = sc.stamp_ns if t0 is None else t0
        if sc.stamp_ns - t0 > seconds * 1e9:
            break
        pts.append(sc.xyz)
        n += 1
    if not pts:
        raise SystemExit("%s: no scans on %s" % (bag, topic))
    log("    %s: %s, frame %s, %d scans in the first %.1f s, %d points"
        % (os.path.basename(bag), topic, frame, n, seconds, sum(len(p) for p in pts)))
    return np.vstack(pts).astype(np.float64), frame, n, topic


def write_check(path, scan_map, map_pts, T_lidar, T_cam, radius=7.0):
    from objects import write_ply
    near = map_pts[np.linalg.norm(map_pts[:, :2] - T_lidar[:2, 3], axis=1) < radius]
    near = near[:: max(1, len(near) // 400000)]
    sc = scan_map[:: max(1, len(scan_map) // 300000)]
    fr, fc = frustum(T_lidar, 0.25), frustum(T_cam, 0.35)
    P = np.vstack([near, sc, fr, fc])
    C = np.vstack([np.tile([170, 170, 170], (len(near), 1)), np.tile([230, 40, 40], (len(sc), 1)),
                   np.tile([40, 80, 230], (len(fr), 1)), np.tile([30, 200, 60], (len(fc), 1))])
    write_ply(path, P, C.astype(np.uint8))


COLOURS = [(30, 200, 60), (230, 40, 40), (40, 80, 230), (240, 170, 0), (200, 40, 200), (0, 190, 200),
           (120, 60, 20), (255, 110, 160)]


def write_all(path, map_pts, cams, max_map=2000000):
    """Every camera on one map: the map grey (lighter higher up), each camera
    a frustum and a 1.5 m ray along its view in its own colour, the robot
    where it stood for that camera a smaller frustum in the same colour.
    cams: [(name, T_map_cam, T_map_lidar or None)] -> legend [(name, rgb)]."""
    from objects import write_ply
    m = map_pts[:: max(1, len(map_pts) // max_map)]
    z = m[:, 2]
    g = (110 + 110 * np.clip((z - np.percentile(z, 2)) / max(np.ptp(z), 1e-6), 0, 1)).astype(np.uint8)
    P, C, legend = [m], [np.c_[g, g, g]], []
    for k, (name, Tc, Tl) in enumerate(cams):
        col = COLOURS[k % len(COLOURS)]
        ray = Tc[:3, 3] + np.linspace(0, 1.5, 300)[:, None] * Tc[:3, 2]
        parts = [frustum(Tc, 0.5, 0.004), ray] + ([frustum(Tl, 0.3, 0.004)] if Tl is not None else [])
        for q in parts:
            P.append(q)
            C.append(np.tile(col, (len(q), 1)))
        legend.append((name, col))
    write_ply(path, np.vstack(P), np.vstack(C).astype(np.uint8))
    return legend


def show(argv):
    """place_cameras.py show <out folder> --map <map>: cameras_on_map.ply from
    the out folder's cameras.json, nothing relocalized again."""
    import open3d as o3d
    ap = argparse.ArgumentParser(prog="place_cameras.py show")
    ap.add_argument("out")
    ap.add_argument("--map", required=True)
    a = ap.parse_args(argv)
    out = os.path.expanduser(a.out)
    rows = json.load(open(os.path.join(out, "cameras.json")))["cameras"]
    cams = [(r["camera"], T_of(r["T_map_cam"]["t"], r["T_map_cam"]["q"]),
             T_of(r["T_map_lidar"]["t"], r["T_map_lidar"]["q"])) for r in rows]
    map_pts = np.asarray(o3d.io.read_point_cloud(os.path.expanduser(a.map)).points)
    report_all(out, map_pts, cams)


def report_all(out, map_pts, cams):
    f = os.path.join(out, "cameras_on_map.ply")
    legend = write_all(f, map_pts, cams)
    print("\nall cameras on one map -> %s" % f)
    for name, col in legend:
        print("  %-8s rgb %s  (big frustum + ray: the camera; small frustum: the robot)" % (name, col))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "show":
        return show(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("calibration", nargs="+", help="calibration folder(s), oldest first; then the out folder")
    ap.add_argument("--map", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--extrinsics", default=None)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--topic", default="auto")
    ap.add_argument("--min-range", type=float, default=1.0)
    ap.add_argument("--static-tf", default=os.path.join(ROOT, "configs", "static_tf.yaml"))
    ap.add_argument("--lidar-frame", default="os_lidar_mobile2")
    ap.add_argument("--camera-frame", default="camera_color_optical_frame")
    a = ap.parse_args(argv)
    import open3d as o3d
    from scoop import tftree
    if len(a.calibration) < 2:
        ap.error("give the calibration folder(s) and then the out folder")
    a.out = a.calibration[-1]
    cal_dirs = [os.path.expanduser(c) for c in a.calibration[:-1]]
    if a.extrinsics and len(cal_dirs) > 1:
        ap.error("--extrinsics with one calibration folder only (each finds its own *extrinsics* folder)")
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    cals = tftree.load_calibrations(yaml.safe_load(open(os.path.expanduser(a.static_tf))))
    lc = [c for c in cals if c.parent == a.lidar_frame and c.child == a.camera_frame]
    if not lc:
        raise SystemExit("%s has no %s -> %s" % (a.static_tf, a.lidar_frame, a.camera_frame))
    T_lidar_cam = T_of(lc[0].translation, lc[0].rotation_xyzw)
    L, pitch, roll, spread = level_from_traj(a.traj, T_lidar_cam)
    print("LiDAR tilt in the mapping pass: pitch %.2f deg, roll %.2f deg (spread %.2f deg)" % (pitch, roll, spread))
    map_pts = np.asarray(o3d.io.read_point_cloud(os.path.expanduser(a.map)).points)
    print("map: %d points" % len(map_pts))
    found = {}                                    # child -> (file, doc, calibration folder): the latest folder wins
    for cal in cal_dirs:
        if a.extrinsics:
            ext_dir = os.path.expanduser(a.extrinsics)
        else:
            cand = [d for d in sorted(glob.glob(os.path.join(cal, "*extrinsics*"))) if os.path.isdir(d)]
            if len(cand) != 1:
                raise SystemExit("%s: which extrinsics folder? %s (--extrinsics)" % (cal, cand or "none"))
            ext_dir = cand[0]
        print("%s: extrinsics in %s" % (cal, os.path.basename(ext_dir)))
        for f in sorted(glob.glob(os.path.join(ext_dir, "extrinsic_*__cam_*.yaml"))):
            if f.endswith("_prev.yaml"):
                continue
            e = yaml.safe_load(open(f))
            if e.get("stage") != "locked":
                print("  %s: stage %s, not locked: left out" % (os.path.basename(f), e.get("stage")))
                continue
            if e.get("parent_frame") != a.camera_frame:
                print("  %s: parent %s, not %s: left out" % (os.path.basename(f), e.get("parent_frame"), a.camera_frame))
                continue
            if e["child_frame"] in found:
                print("  %s: replaces the one in %s" % (e["child_frame"], found[e["child_frame"]][2]))
            found[e["child_frame"]] = (f, e, cal)
    exts = sorted(found.values(), key=lambda x: int(re.search(r"(\d+)$", x[1]["child_frame"]).group(1)))
    if not exts:
        raise SystemExit("no locked extrinsic to %s" % a.camera_frame)
    print("cameras: %s" % ", ".join(x[1]["child_frame"] for x in exts))
    rows, transforms = [], []
    for f, e, cal in exts:
        child = e["child_frame"]
        n = int(re.search(r"(\d+)$", child).group(1))
        print("\n%s (%s, %d placements, spread %.2f mm / %.3f deg)"
              % (child, os.path.basename(f), e["placements"], e["spread_mm"] or -1, e["spread_deg"] or -1))
        bags = find_bag(cal, n)
        if len(bags) != 1:
            print("  bag for %s: %s -- left out (one expected)" % (child, bags or "none"))
            continue
        scan, frame, nscan, topic = static_cloud(bags[0], a.topic, a.seconds, a.min_range)
        if "os_sensor" in frame:
            edges = tftree.tf_edges([bags[0]], seconds=1.0)
            lid = next((k for k in edges if "os_lidar" in k), None)
            if lid is None:
                raise SystemExit("%s: points in %s and no os_lidar in its /tf_static" % (bags[0], frame))
            T_ls = tftree.static_between(lid, frame, edges)
            scan = scan @ T_ls[:3, :3].T + T_ls[:3, 3]
            print("    moved from %s to %s (the bag's /tf_static)" % (frame, lid))
        elif "os_lidar" not in frame:
            raise SystemExit("%s: points in %s, neither os_lidar nor os_sensor" % (bags[0], frame))
        T_map_lidar, fit, rmse, riv, cands = place_robot(scan, map_pts, L)
        T_map_col = T_map_lidar @ T_lidar_cam
        T_col_cam = T_of(e["translation"], e["quaternion_xyzw"])
        T_map_cam = T_map_col @ T_col_cam
        t, q = tq(T_map_cam)
        ok = fit >= 0.5 and (riv is None or riv["unique"])
        print("  robot: on map %.1f %%, rmse %.1f cm, at %s%s -> %s at %s"
              % (100 * fit, 100 * rmse, np.round(T_map_lidar[:3, 3], 3).tolist(),
                 "" if riv is None else ("; rival %.1f %% at %s (%s)" % (100 * riv["fit"], riv["t"],
                                                                      "unique" if riv["unique"] else "NOT unique")),
                 child, t))
        if not ok:
            print("  ! not trusted: %s" % ("fit under 50 %" if fit < 0.5 else "another placement fits nearly as well"))
        scan_map = scan @ T_map_lidar[:3, :3].T + T_map_lidar[:3, 3]
        write_check(os.path.join(out, "check_%s.ply" % child), scan_map, map_pts, T_map_col, T_map_cam)
        quality = {"robot_fit": round(fit, 4), "robot_rmse_m": round(rmse, 4), "trusted": ok,
                   "rival": riv, "scans": nscan, "bag": os.path.basename(bags[0]), "topic": topic,
                   "extrinsic": os.path.basename(f), "calibration": cal, "extrinsic_spread_mm": e.get("spread_mm"),
                   "extrinsic_spread_deg": e.get("spread_deg"), "placements": e.get("placements")}
        transforms.append({"parent": "map", "child": child, "translation": t, "rotation_xyzw": q,
                           "quality": quality})
        rl, ql = tq(T_map_lidar)
        rows.append({"camera": child, "T_map_cam": {"t": t, "q": q}, "T_map_lidar": {"t": rl, "q": ql},
                     "quality": quality,
                     "candidates": [{"fit": round(fc, 4), "rmse": round(rc, 4), "t": np.round(Tc[:3, 3], 3).tolist()}
                                    for Tc, fc, rc in cands]})
    with open(os.path.join(out, "cameras.yaml"), "w") as fh:
        fh.write("# room cameras in the map: T_map_cam = T_map_lidar (LiDAR relocalized, robot still)\n"
                 "#   . T_lidar_colour (%s) . T_colour_cam (ChArUco)\n" % os.path.basename(a.static_tf))
        yaml.safe_dump({"transforms": transforms}, fh, sort_keys=False)
    json.dump({"level": {"pitch_deg": pitch, "roll_deg": roll}, "cameras": rows},
              open(os.path.join(out, "cameras.json"), "w"), indent=1)
    report_all(out, map_pts, [(r["camera"], T_of(r["T_map_cam"]["t"], r["T_map_cam"]["q"]),
                               T_of(r["T_map_lidar"]["t"], r["T_map_lidar"]["q"])) for r in rows])
    if len(rows) > 1:
        print("\nwhere the robot stood for each calibration:")
        for r in rows:
            print("  %-8s %s" % (r["camera"], r["T_map_lidar"]["t"]))
    print("-> %s (cameras.yaml, cameras.json, cameras_on_map.ply, check_cam_*.ply)" % out)


if __name__ == "__main__":
    sys.exit(main())

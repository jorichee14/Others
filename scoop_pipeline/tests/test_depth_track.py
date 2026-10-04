#!/usr/bin/env python
"""Self-test for map_stages/depth_track.py through 04_reference_traj.py --track:
a depth camera driving in a room, its odometry drifting (1.5 cm + 0.15 deg
per second, about an origin 5 m away), placed at the start by a session
anchor, and every depth frame registered to the room's map.

  * the depth images are ray-cast from the true camera_link poses (a room with
    a pillar and furniture), so the map explains every pixel;
  * the odometry seed drifts ~12 cm by 5.5 s; then tracking is lost for
    1.5 s (no odometry) and at 10 s it resets (1.2 m / 30 deg jump): the
    depth frames carry the track through both on the map; then 2.5 s without
    depth frames or odometry: the robot must be searched for and found again
    -- and again with every search made ambiguous, by tracking the candidate
    places on until one keeps fitting;
  * the track must come back to the truth (median 0.5 cm, p95 2 cm, 0.6 deg; where
    the camera sees one wall and the floor, the seed holds the direction
    along the wall, carried by the last registered frame: max 4 cm), its
    _in_cam file be the colour optical frame through the bag's /tf_static,
    and the summary say how far the odometry drifted.

Also fill_back (the track run backwards from a frame placed again) on its
own, and processing/vslam.py: its odometry recording -> TUM, and the launch file
it writes compiles.

    python scoop_pipeline/tests/test_depth_track.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))

from scoop import rosmsg, tftree                                    # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402

T0 = 1_790_000_000.0
# walls mid-cell of 04's 0.4 m plane grid, as a real map's thick walls are
# somewhere in their cells (a wall on a cell boundary has planes on one side only)
ROOM = (np.array([-3.0, -2.2, -0.2]), np.array([3.0, 2.2, 2.6]))
BOXES = [(np.array(a, float), np.array(b, float)) for a, b in (   # furniture: a room
    ((0.6, -1.0, -0.2), (1.0, -0.2, 2.6)),                          # that is not a plain box
    ((-2.6, 1.4, -0.2), (-1.8, 1.8, 0.6)),
    ((1.8, 1.0, -0.2), (2.2, 1.8, 1.4)),
    ((-1.0, -2.2, -0.2), (-0.2, -1.8, 1.0)),
    ((-3.0, -1.4, 0.6), (-2.6, -0.6, 1.8)))]
W, H, F = 160, 120, 120.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
R_LINK_OPT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # optical in camera_link


def rotz(deg, t=(0, 0, 0)):
    a = np.radians(deg)
    T = np.eye(4)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    T[:3, 3] = t
    return T


def Tm(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


T_LINK_DEPTH = Tm(R_LINK_OPT, (0, 0, 0))
T_LINK_COLOR = Tm(R_LINK_OPT, (0, 0.015, 0))


def truth(t):
    """camera_link in map: parked for the first second, then a loop."""
    s = max(t - 1.0, 0.0)
    return rotz(25 * s, (-1.2 + 1.0 * np.sin(0.25 * s), 0.6 * np.sin(0.4 * s), 0.5))


def odom(t):
    """The odometry: a wrong frame, drifting."""
    return rotz(0.15 * t, (0.015 * t, -0.008 * t, 0)) @ rotz(70, (5, -2, 0.1)) @ truth(t)


GAP = (5.5, 7.0)                       # tracking lost: no odometry
BLACKOUT = (12.0, 14.5)                # no depth frames and no odometry: found again by search


def odom_broken(t):
    """As cuVSLAM gives it: a reset at 10 s."""
    return (rotz(30, (1.2, -0.5, 0)) if t >= 10.0 else np.eye(4)) @ odom(t)


def raycast(T_map_depth):
    v, u = np.mgrid[0:H, 0:W]
    d = np.stack([(u - K[0, 2]) / F, (v - K[1, 2]) / F, np.ones_like(u, float)], -1)
    d = d.reshape(-1, 3) @ T_map_depth[:3, :3].T
    o = T_map_depth[:3, 3]
    lo, hi = ROOM
    with np.errstate(divide="ignore", invalid="ignore"):
        t_room = np.min(np.where(d > 0, (hi - o) / d, np.where(d < 0, (lo - o) / d, np.inf)),
                        axis=1)
        t = t_room
        for blo, bhi in BOXES:
            a, b = (blo - o) / d, (bhi - o) / d
            tn = np.max(np.minimum(a, b), axis=1)
            tf = np.min(np.maximum(a, b), axis=1)
            t = np.where((tn <= tf) & (tn > 0) & (tn < t), tn, t)
    z = (t * 1.0).reshape(H, W)                     # optical z of a ray with d_z = 1
    return np.round(np.nan_to_num(z, nan=0.0, posinf=0.0) * 1000).astype(np.uint16)


def room_points(step=0.03):
    pts = []
    for lo, hi in [ROOM] + BOXES:
        for ax in range(3):
            for side in (lo[ax], hi[ax]):
                o = [i for i in range(3) if i != ax]
                g = np.mgrid[lo[o[0]]:hi[o[0]]:step, lo[o[1]]:hi[o[1]]:step].reshape(2, -1).T
                p = np.zeros((len(g), 3))
                p[:, o[0]], p[:, o[1]], p[:, ax] = g[:, 0], g[:, 1], side
                pts.append(p)
    return np.vstack(pts)


def write_bag(path, times):
    T = rosmsg.msg_type
    with BagWriter(path) as w:
        st = []
        for child, M in (("camera_depth_optical_frame", T_LINK_DEPTH),
                         ("camera_color_optical_frame", T_LINK_COLOR)):
            t, q = tftree.to_tq(M)
            st.append(("camera_link", child, t, q))
        w.write("/tf_static", rosmsg.tf_message(st, int(T0 * 1e9)), int(T0 * 1e9))
        for t in times:
            if BLACKOUT[0] < t < BLACKOUT[1]:
                continue
            ns = int(round((T0 + t) * 1e9))
            img = raycast(truth(t) @ T_LINK_DEPTH)
            w.write("/mobile_2/depth/camera_info", T("sensor_msgs/msg/CameraInfo")(
                header=rosmsg.header(ns, "camera_depth_optical_frame"), height=H, width=W,
                distortion_model="plumb_bob", d=np.zeros(5), k=K.reshape(-1), r=np.eye(3).reshape(-1),
                p=np.zeros(12), binning_x=0, binning_y=0,
                roi=T("sensor_msgs/msg/RegionOfInterest")(x_offset=0, y_offset=0, height=0,
                                                          width=0, do_rectify=False)), ns)
            w.write("/mobile_2/depth/image_rect_raw", T("sensor_msgs/msg/Image")(
                header=rosmsg.header(ns, "camera_depth_optical_frame"), height=H, width=W,
                encoding="16UC1", is_bigendian=0, step=W * 2,
                data=img.view(np.uint8).reshape(-1).copy()), ns)


def write_tum(path, times, poses):
    with open(path, "w") as f:
        for t, M in zip(times, poses):
            p, q = tftree.to_tq(M)
            f.write("%.9f %s %s\n" % (T0 + t, " ".join(map(str, p)), " ".join(map(str, q))))


def main():
    import open3d as o3d
    tmp = tempfile.mkdtemp()
    f = []
    try:
        data = os.path.join(tmp, "data")
        work = os.path.join(data, "work", "20260101", "coop_1")
        proc = os.path.join(data, "processed", "20260101", "coop_1")
        ref = os.path.join(data, "processed", "20260101", "mapping_A", "mapping")
        for d in (work, os.path.join(proc, "frames"), ref):
            os.makedirs(d)
        times = np.arange(0.0, 22.0, 1 / 15)
        bag = os.path.join(work, "x_merged")
        write_bag(bag, times)
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(room_points()))
        o3d.io.write_point_cloud(os.path.join(ref, "map_final_mapping_A_20260101_anchored.pcd"), pc)
        vt = os.path.join(work, "mobile_2", "vslam")
        os.makedirs(vt)
        to = [t for t in times if not GAP[0] < t < GAP[1] and not BLACKOUT[0] < t < BLACKOUT[1]]
        write_tum(os.path.join(vt, "traj_vslam.txt"), to, [odom_broken(t) for t in to])
        os.makedirs(os.path.join(work, "mobile_1", "glim"))
        write_tum(os.path.join(work, "mobile_1", "glim", "traj_lidar.txt"), times[:2],
                  [np.eye(4)] * 2)
        T_cam = truth(0.5) @ T_LINK_COLOR
        p, q = tftree.to_tq(T_cam)
        json.dump({"cameras": {"realsense": {"mode": "static", "dwell_t0": T0 + 0.1,
                                             "dwell_t_end": T0 + 0.5,
                                             "map_to_cam": {"xyz": p, "qxyzw": q}}}},
                  open(os.path.join(proc, "frames", "session_anchor.json"), "w"))
        stages = os.path.join(tmp, "stages")
        os.makedirs(stages)
        cfg = os.path.join(stages, "pipeline_config_coop_1.json")
        json.dump({"dataset": {"bag": bag, "traj": None, "machine": "mobile_1",
                               "reference_pass": "mapping_A"},
                   "01_build_map": {},
                   "04_reference": {"name": "mobile_1_lidar", "out_dir": "reference_{tag}",
                                    "tracks": [{"name": "mobile_2_depth", "machine": "mobile_2",
                                                "range": [0.3, 6.0]}]}},
                  open(cfg, "w"))
        r = subprocess.run([sys.executable, os.path.join(ROOT, "map_stages", "04_reference_traj.py"),
                            cfg, "--track", "mobile_2_depth"], capture_output=True, text=True)
        print(r.stdout[-2500:])
        if r.returncode:
            f.append("04 --track: " + r.stdout[-1500:] + r.stderr[-2500:])
        else:
            from pipeline_common import load_traj
            od = os.path.join(proc, "odometry", "reference_coop_1_20260101")
            ts, Tb = load_traj(os.path.join(od, "traj_mobile_2_depth.tum"))
            _, Tc = load_traj(os.path.join(od, "traj_mobile_2_depth_in_cam.tum"))
            e_t, e_r, e_c = [], [], []
            for t, B, C in zip(ts, Tb, Tc):
                G = truth(t - T0)
                e_t.append(np.linalg.norm(B[:3, 3] - G[:3, 3]) * 100)
                e_r.append(np.degrees(np.arccos(np.clip((np.trace(B[:3, :3].T @ G[:3, :3]) - 1) / 2,
                                                        -1, 1))))
                e_c.append(np.linalg.norm(C[:3, 3] - (G @ T_LINK_COLOR)[:3, 3]) * 100)
            print("track vs truth: %.2f cm / %.3f deg median, %.2f cm / %.3f deg max over %d frames"
                  % (np.median(e_t), np.median(e_r), max(e_t), max(e_r), len(ts)))
            if len(ts) < 0.35 * len(times) or not (ts > T0 + 10.5).any():
                f.append("only %d of %d frames in the track" % (len(ts), len(times)))
            if np.median(e_t) > 0.5 or np.percentile(e_t, 95) > 2.0 or max(e_t) > 4.0 \
                    or max(e_r) > 0.6:
                f.append("track off: %.2f / %.2f / %.2f cm median / p95 / max, %.3f deg max"
                         % (np.median(e_t), np.percentile(e_t, 95), max(e_t), max(e_r)))
            if max(e_c) > 4.0:
                f.append("_in_cam off: %.2f cm" % max(e_c))
            s = json.load(open(os.path.join(od, "summary_mobile_2_depth.json")))
            if s["anchored_piece_seed_to_final_cm"]["at_end"] < 6:
                f.append("summary: odometry drift at the end %.1f cm (expected ~12 cm)"
                         % s["anchored_piece_seed_to_final_cm"]["at_end"])
            if s["odometry_pieces"] != 4 or not any(e["found"] for e in s["refind"]):
                f.append("summary: %d pieces, re-finds %s" % (s["odometry_pieces"], s["refind"]))
        # every search ambiguous (a rival must leave half the points more off): the robot
        # is found by tracking the candidate places on, the one that keeps fitting
        c = json.load(open(cfg))
        c["04_reference"]["tracks"][0]["rival_gap"] = 0.5
        json.dump(c, open(cfg, "w"))
        r = subprocess.run([sys.executable, os.path.join(ROOT, "map_stages", "04_reference_traj.py"),
                            cfg, "--track", "mobile_2_depth"], capture_output=True, text=True)
        print("\n".join(ln for ln in r.stdout.splitlines() if "re-find" in ln))
        if r.returncode:
            f.append("04 --track, hypotheses: " + r.stdout[-1500:] + r.stderr[-2500:])
        else:
            from pipeline_common import load_traj
            od = os.path.join(proc, "odometry", "reference_coop_1_20260101")
            ts, Tb = load_traj(os.path.join(od, "traj_mobile_2_depth.tum"))
            e = [np.linalg.norm(B[:3, 3] - truth(t - T0)[:3, 3]) * 100 for t, B in zip(ts, Tb)]
            s = json.load(open(os.path.join(od, "summary_mobile_2_depth.json")))
            hy = [ev for ev in s["refind"] if ev.get("hypotheses")]
            print("hypotheses: %d frames, %.2f cm median, %.2f cm max; %s"
                  % (len(ts), np.median(e), max(e), hy))
            if not hy or not (ts > T0 + 14.6).any() or max(e) > 4.0:
                f.append("hypotheses: %d frames, max %.2f cm, re-finds %s"
                         % (len(ts), max(e), s["refind"]))

        # fill_back: frames 5..19 unplaced, the frames after them placed: the
        # track run backwards fills them
        import importlib
        import depth_track as D
        A = importlib.import_module("02_refine_poses")
        R04 = importlib.import_module("04_reference_traj")
        cc = dict(R04.DEFAULTS)
        cc.update(D.DEFAULTS)
        cc.update(range=[0.3, 6.0], nn_prior_beta=cc["prior_beta"])
        pcr = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(room_points()))
        refm = A.Reference(np.asarray(pcr.voxel_down_sample(0.05).points), plane_voxel=0.4,
                           planarity=1.0)
        tb = np.arange(2.0, 6.0, 2 / 15)
        scans_b = []
        for t in tb:
            p = D.backproject(raycast(truth(t) @ T_LINK_DEPTH), K, 0.001, 0.3, 6.0, 4)
            q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p @ T_LINK_DEPTH[:3, :3].T))
            scans_b.append(np.asarray(q.voxel_down_sample(0.05).points))
        Tb_out = np.array([truth(t) for t in tb])
        Tb_out[5:20] = np.nan
        rows_b = [("ok", 0, 0, 0, 0)] * len(tb)
        nb = D.fill_back(A, refm, scans_b, tb, np.tile(np.eye(4), (len(tb), 1, 1)),
                         np.full(len(tb), -1), Tb_out, rows_b, cc)
        eb = [np.linalg.norm(Tb_out[i][:3, 3] - truth(tb[i])[:3, 3]) * 100 for i in range(5, 20)
              if np.isfinite(Tb_out[i]).all()]
        print("fill_back: %d frames placed, max %.2f cm" % (nb, max(eb) if eb else -1))
        if nb != 15 or max(eb) > 3.0:
            f.append("fill_back: %d of 15 frames, %s cm" % (nb, eb))

        # processing/vslam.py: the odometry recording -> TUM; the launch file it writes
        sys.path.insert(0, os.path.join(ROOT, "processing"))
        import vslam
        ob = os.path.join(tmp, "odometry")
        Tt = rosmsg.msg_type
        with BagWriter(ob) as w:
            for t in times[:30]:
                ns = int(round((T0 + t) * 1e9))
                p, q = tftree.to_tq(odom(t))
                w.write(vslam.ODOM, Tt("nav_msgs/msg/Odometry")(
                    header=rosmsg.header(ns, "odom_vslam"), child_frame_id="camera_link",
                    pose=Tt("geometry_msgs/msg/PoseWithCovariance")(
                        pose=Tt("geometry_msgs/msg/Pose")(
                            position=Tt("geometry_msgs/msg/Point")(x=p[0], y=p[1], z=p[2]),
                            orientation=Tt("geometry_msgs/msg/Quaternion")(
                                x=q[0], y=q[1], z=q[2], w=q[3])),
                        covariance=np.zeros(36)),
                    twist=Tt("geometry_msgs/msg/TwistWithCovariance")(
                        twist=Tt("geometry_msgs/msg/Twist")(
                            linear=Tt("geometry_msgs/msg/Vector3")(x=0.0, y=0.0, z=0.0),
                            angular=Tt("geometry_msgs/msg/Vector3")(x=0.0, y=0.0, z=0.0)),
                        covariance=np.zeros(36))), ns)
        sm = vslam.to_tum(ob, os.path.join(tmp, "traj_vslam.txt"), "camera_link")
        from pipeline_common import load_traj
        tv, Tv = load_traj(os.path.join(tmp, "traj_vslam.txt"))
        if sm["poses"] != 30 or sm["frame"] != "odom_vslam" or \
                max(np.linalg.norm(Tv[i][:3, 3] - odom(times[i])[:3, 3]) for i in range(30)) > 1e-5:
            f.append("vslam.to_tum: %s" % sm)
        c = dict(vslam.DEFAULTS)
        compile(vslam.LAUNCH % (vslam.node_params(c), vslam.remaps(c)), "vslam.launch.py", "exec")
        if dict(vslam.remaps(c))["visual_slam/image_0"] != "/mobile_2/infra1/image_rect_raw":
            f.append("vslam remaps: %s" % vslam.remaps(c))
    except Exception:                                                  # noqa: BLE001
        import traceback
        f.append(traceback.format_exc()[-2500:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "depth track ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Self-tests for mcap_convert.py's replica format with lidar depth. No ROS,
no real bags.

A synthetic recording: the camera slides sideways at 1 m/s past a plate 2 m
away in front of a wall 5 m away; a lidar 10 cm to its left (rotated, as an
Ouster is) scans at 10 Hz with per-point times. The depth rendered at each
image time is compared with the true depth.

    python tests/test_replica.py
"""
import logging
import os
import shutil
import sys
import tempfile
import traceback
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "datasets"))

import cv2                                                          # noqa: E402
import mcap_convert as mc                                           # noqa: E402
from scoop import rosmsg                                            # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402

FAILED = []
S = 1_000_000_000
T0 = 1_790_218_481 * S
W, H, FX, CX, CY = 160, 120, 100.0, 80.0, 60.0
SPEED = 1.0                                     # m/s along the camera's +x
PLATE = 0.4                                     # half size, at z = 2
CAM, LIDAR = "cam_optical", "os_lidar"
# lidar in the camera: 10 cm left, x forward / y left / z up (camera z / -x / -y)
R_CL = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
T_CL = np.eye(4)
T_CL[:3, :3], T_CL[:3, 3] = R_CL, [-0.10, 0.02, 0.0]
DURATION = 2.0


def cam_pose(t_ns):
    T = np.eye(4)
    T[0, 3] = SPEED * (t_ns - T0) * 1e-9
    return T


def true_depth(t_ns):
    """Depth the camera sees at t_ns: the plate at 2 m (fixed in the world at
    |x|, |y| < PLATE), else the wall at 5 m."""
    cx = cam_pose(t_ns)[0, 3]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x, y = (u - CX) / FX, (v - CY) / FX
    hit = (np.abs(cx + 2 * x) < PLATE) & (np.abs(2 * y) < PLATE)
    return np.where(hit, 2.0, 5.0)


def cast(origin, dirs):
    """World points hit by rays from origin: plate at z=2, else wall at z=5."""
    s2 = 2.0 / dirs[:, 2]
    p2 = origin + dirs * s2[:, None]
    hit = (np.abs(p2[:, 0]) < PLATE) & (np.abs(p2[:, 1]) < PLATE)
    p5 = origin + dirs * (5.0 / dirs[:, 2])[:, None]
    return np.where(hit[:, None], p2, p5)


CLOUD = np.dtype({"names": ["x", "y", "z", "t"], "formats": ["<f4", "<f4", "<f4", "<u4"],
                  "offsets": [0, 4, 8, 16], "itemsize": 32})


def scan(t_ns, rows=48, cols=200):
    """One organized cloud in the lidar frame: columns swept over 100 ms,
    each point where its ray hits the scene at the point's own time."""
    az = np.linspace(np.deg2rad(55), np.deg2rad(-55), cols)          # left to right
    el = np.linspace(np.deg2rad(35), np.deg2rad(-35), rows)
    a = np.zeros(rows * cols, CLOUD)
    for j, (phi, dt) in enumerate(zip(az, np.linspace(0, 0.099, cols))):
        T_wl = cam_pose(t_ns + int(dt * S)) @ T_CL
        d_l = np.stack([np.cos(el) * np.cos(phi), np.cos(el) * np.sin(phi), np.sin(el)], 1)
        d_w = d_l @ T_wl[:3, :3].T
        p_w = cast(T_wl[:3, 3], d_w)
        p_l = (p_w - T_wl[:3, 3]) @ T_wl[:3, :3]
        k = np.arange(rows) * cols + j
        a["x"][k], a["y"][k], a["z"][k] = p_l.T
        a["t"][k] = int(dt * S)
    return rosmsg.pointcloud2(a, t_ns, LIDAR, height=rows, width=cols)


def write_bag(path):
    T = rosmsg.msg_type
    recs = []
    info = T("sensor_msgs/msg/CameraInfo")(
        header=rosmsg.header(T0, CAM), height=H, width=W, distortion_model="plumb_bob",
        d=np.zeros(5), k=np.array([FX, 0, CX, 0, FX, CY, 0, 0, 1.0]),
        r=np.eye(3).reshape(-1), p=np.array([FX, 0, CX, 0, 0, FX, CY, 0, 0, 0, 1.0, 0]),
        binning_x=0, binning_y=0,
        roi=T("sensor_msgs/msg/RegionOfInterest")(x_offset=0, y_offset=0, height=0,
                                                  width=0, do_rectify=False))
    q = mc.T_to_xyzq(T_CL)
    recs.append((T0, "/tf_static", rosmsg.tf_message([(CAM, LIDAR, q[:3], q[3:])], T0)))
    for k in range(int(DURATION * 100) + 1):                     # poses, 100 Hz
        t = T0 + k * S // 100
        P = cam_pose(t)
        recs.append((t, "/pose", T("geometry_msgs/msg/PoseStamped")(
            header=rosmsg.header(t, "map"), pose=T("geometry_msgs/msg/Pose")(
                position=T("geometry_msgs/msg/Point")(x=P[0, 3], y=P[1, 3], z=P[2, 3]),
                orientation=T("geometry_msgs/msg/Quaternion")(x=0.0, y=0.0, z=0.0, w=1.0)))))
    for k in range(int(DURATION * 10)):                           # lidar, 10 Hz
        t = T0 + k * S // 10 + 3_000_000
        recs.append((t + 100_000_000, "/points", scan(t)))
    img = np.full((H, W, 3), 128, np.uint8)
    for k in range(int(DURATION * 15)):                           # images, 15 Hz
        t = T0 + 20_000_000 + k * S // 15
        info.header = rosmsg.header(t, CAM)
        recs.append((t, "/info", info))
        recs.append((t, "/image", T("sensor_msgs/msg/Image")(
            header=rosmsg.header(t, CAM), height=H, width=W, encoding="rgb8",
            is_bigendian=0, step=W * 3, data=img.reshape(-1))))
    w = BagWriter(path)
    for t, topic, msg in sorted(recs, key=lambda r: r[0]):
        w.write(topic, msg, t)
    w.close()


def check_frames(out, max_err=0.05):
    """Fraction of rendered pixels within max_err of the true depth, mean fill."""
    traj = np.loadtxt(os.path.join(out, "traj.txt")).reshape(-1, 4, 4)
    good = total = 0
    fills = []
    for i, T in enumerate(traj):
        t = T0 + int(round(T[0, 3] / SPEED * S))
        d = cv2.imread(os.path.join(out, "results", f"depth{i:06d}.png"),
                       cv2.IMREAD_UNCHANGED).astype(float) / 1000.0
        m = d > 0
        good += int((np.abs(d - true_depth(t))[m] < max_err).sum())
        total += int(m.sum())
        fills.append(m.mean())
    return good / max(total, 1), float(np.mean(fills)), len(traj)


def test_lidar_depth_matches_scene(tmp):
    bag = os.path.join(tmp, "bag")
    write_bag(bag)
    kw = dict(color="/image", info="/info", pose="/pose", pose_is_camera=True,
              lidar="/points", depth_min=0.2, depth_max=10.0)
    res = mc.to_replica(bag, os.path.join(tmp, "one"), **kw)
    assert res["depth_source"] == "lidar" and res["frames"] >= 25, res
    ok, fill1, n = check_frames(os.path.join(tmp, "one"))
    assert ok > 0.99, f"only {ok:.3f} of lidar depth pixels match the scene"
    assert fill1 > 0.15, fill1

    res = mc.to_replica(bag, os.path.join(tmp, "three"), lidar_scans=3, **kw)
    ok3, fill3, _ = check_frames(os.path.join(tmp, "three"))
    assert ok3 > 0.99, f"3 scans: only {ok3:.3f} match (motion not compensated?)"
    assert fill3 > fill1 * 1.3, (fill1, fill3)

    # without the hidden-point filter, wall seen by the lidar past the plate's
    # edge leaks into the camera's view of the plate
    mc.to_replica(bag, os.path.join(tmp, "nofilter"), lidar_scans=3, occlusion_px=0, **kw)
    ok0, _, _ = check_frames(os.path.join(tmp, "nofilter"))
    assert ok0 < ok3, (ok0, ok3)

    # traj is the camera pose; the config has the intrinsics
    traj = np.loadtxt(os.path.join(tmp, "one", "traj.txt")).reshape(-1, 4, 4)
    assert np.allclose(traj[:, :3, :3], np.eye(3)) and np.all(np.diff(traj[:, 0, 3]) > 0)
    cfg = open(os.path.join(tmp, "one", "splatam_data_config.yaml")).read()
    assert "fx: 100.000000" in cfg and "image_width: 160" in cfg


def test_missing_extrinsic_is_an_error(tmp):
    bag = os.path.join(tmp, "bag2")
    write_bag(bag)
    try:
        mc.to_replica(bag, os.path.join(tmp, "x"), color="/image", info="/info",
                      pose="/pose", pose_is_camera=True, lidar="/points",
                      cam_frame="other_frame")
    except mc.ConvertError as e:
        assert "--lidar-extrinsic" in str(e), e
    else:
        raise AssertionError("no error without a lidar-camera transform")


def test_project_depth_nearest_and_occlusion(tmp):
    K = mc.Intrinsics(100.0, 100.0, 10.0, 10.0, 21, 21, [0] * 4, None)
    pts = np.array([[0, 0, 5.0], [0, 0, 2.0], [0, 0, 3.0],        # same pixel
                    [0.02, 0, 2.0], [0.12, 0, 6.0], [1.0, 0, 1.0]])  # 1.0 off-image
    D = mc.project_depth(pts, K, occlusion_px=0)
    assert D[10, 10] == 2.0 and D[10, 11] == 2.0 and D[10, 12] == 6.0
    assert (D > 0).sum() == 3
    D = mc.project_depth(pts, K, occlusion_px=7)
    assert (D > 0).sum() == 2 and D[10, 11] == 2.0                  # the 6 m point is hidden


def test_cloud_points_padding_and_time(tmp):
    a = np.zeros(6, CLOUD)
    a["x"] = [1, 2, np.nan, 0, 4, 5]
    a["t"] = [0, 1000, 2000, 3000, 4000, 5000]
    msg = rosmsg.pointcloud2(a, T0, LIDAR, height=2, width=3)
    pad = 8                                                         # padded rows
    data = np.concatenate([np.r_[r, np.zeros(pad, np.uint8)]
                           for r in np.asarray(msg.data).reshape(2, -1)])
    m2 = SimpleNamespace(fields=msg.fields, is_bigendian=False, point_step=msg.point_step,
                         width=3, height=2, row_step=msg.row_step + pad, data=data)
    xyz, t = mc.cloud_points(m2)
    assert xyz[:, 0].tolist() == [1, 2, 4, 5], xyz                 # NaN and zero dropped
    assert np.allclose(t, [0, 1e-6, 4e-6, 5e-6])


def test_bag_files_in_recording_order(tmp):
    d = os.path.join(tmp, "split")
    os.makedirs(d)
    names = [f"split_{i}.mcap" for i in (0, 1, 2, 10, 11)]
    for n in names:
        open(os.path.join(d, n), "w").close()
    got = [os.path.basename(p) for p in mc.bag_files(d)]           # no metadata: natural
    assert got == names, got
    with open(os.path.join(d, "metadata.yaml"), "w") as f:
        f.write("rosbag2_bagfile_information:\n  relative_file_paths:\n"
                + "".join(f"  - {n}\n" for n in reversed(names)))
    got = [os.path.basename(p) for p in mc.bag_files(d)]
    assert got == list(reversed(names)), got


def main():
    logging.basicConfig(level=logging.WARNING)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        tmp = tempfile.mkdtemp()
        try:
            t(tmp)
            print(f"ok    {t.__name__}")
        except Exception:
            FAILED.append(t.__name__)
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(tests) - len(FAILED)}/{len(tests)} passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()

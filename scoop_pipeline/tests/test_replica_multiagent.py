#!/usr/bin/env python
"""Self-test for datasets/replica_multiagent.py on a synthetic two-robot bag.

Both robots slide along map x at 0.5 m/s, cameras facing a wall 3 m ahead.
  mobile_1  colour 160 x 120, f 100, at 15 Hz; depth in the colour frame
            (32FC1, metres, at half resolution: the ZED's depth_registered)
  mobile_2  colour 200 x 100, f 90, at 30 Hz; depth from its own imager
            (16UC1, mm, 120 x 90, f 70) 5 cm right of and 2 cm ahead of
            the colour camera (the RealSense)
mobile_2's poses start 1 s after mobile_1's and stop for 0.8 s in the
middle. Colour pixels carry their own column and row (R, G), so where the
common camera samples each image can be checked.

It must: put both robots on one camera (the field of view both cover);
take frame i of both at the same moment, starting when both have poses
and skipping the pose gap; give each frame the pose at its image stamp;
sample the colour where the common camera's ray falls in each image;
register both depths to 3 m (the RealSense's through its 2 cm offset);
leave nothing but the agent folders in the scene. Then the same bag with no
/tf_static (an older recording): the depth extrinsic given instead must
register mobile_2's depth the same.

    python scoop_pipeline/tests/test_replica_multiagent.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "datasets"))

import cv2                                                          # noqa: E402
import mcap_convert as mc                                           # noqa: E402
import replica_multiagent as RM                                     # noqa: E402
from scoop import rosmsg                                            # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402

S = 1_000_000_000
T0 = 1_790_218_481 * S
DUR = 6.0
WALL = 3.0
GAP = (3.0, 3.8)                     # mobile_2: no pose (s from T0)
CAMS = {"mobile_1": dict(W=160, H=120, f=100.0, rate=15, off=7_000_000, x0=0.0),
        "mobile_2": dict(W=200, H=100, f=90.0, rate=30, off=3_000_000, x0=2.0)}
DEPTH2 = dict(W=120, H=90, f=70.0)
T_CD2 = np.eye(4)
T_CD2[:3, 3] = [0.05, 0.0, 0.02]                      # depth imager in mobile_2's colour frame


def pose(m, t_ns):
    T = np.eye(4)
    T[0, 3] = CAMS[m]["x0"] + 0.5 * (t_ns - T0) / S
    return T


def info(W, H, f, frame, t):
    T = rosmsg.msg_type
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    return T("sensor_msgs/msg/CameraInfo")(
        header=rosmsg.header(t, frame), height=H, width=W, distortion_model="plumb_bob",
        d=np.zeros(5), k=np.array([f, 0, cx, 0, f, cy, 0, 0, 1.0]), r=np.eye(3).reshape(-1),
        p=np.array([f, 0, cx, 0, 0, f, cy, 0, 0, 0, 1.0, 0]), binning_x=0, binning_y=0,
        roi=T("sensor_msgs/msg/RegionOfInterest")(x_offset=0, y_offset=0, height=0, width=0,
                                                  do_rectify=False))


def image(arr, enc, frame, t):
    T = rosmsg.msg_type
    H, W = arr.shape[:2]
    ch = arr.shape[2] if arr.ndim == 3 else 1
    return T("sensor_msgs/msg/Image")(header=rosmsg.header(t, frame), height=H, width=W,
                                      encoding=enc, is_bigendian=0,
                                      step=W * ch * arr.dtype.itemsize,
                                      data=np.frombuffer(arr.tobytes(), np.uint8))


def write_bag(path, tf_static=True):
    T = rosmsg.msg_type
    recs = []
    q = mc.T_to_xyzq(T_CD2)
    if tf_static:
        recs.append((T0, "/tf_static", rosmsg.tf_message(
            [("m2_color", "m2_depth", q[:3], q[3:])], T0)))
    for m in CAMS:
        start = 1.0 if m == "mobile_2" else 0.0
        for k in range(int(DUR * 100) + 1):
            t = T0 + k * S // 100
            s = (t - T0) / S
            if s < start or (m == "mobile_2" and GAP[0] < s < GAP[1]):
                continue
            P = pose(m, t)
            recs.append((t, f"/{m}/global_pose", T("geometry_msgs/msg/PoseStamped")(
                header=rosmsg.header(t, "map"), pose=T("geometry_msgs/msg/Pose")(
                    position=T("geometry_msgs/msg/Point")(x=P[0, 3], y=P[1, 3], z=P[2, 3]),
                    orientation=T("geometry_msgs/msg/Quaternion")(x=0.0, y=0.0, z=0.0, w=1.0)))))
    for m, c in CAMS.items():
        W, H = c["W"], c["H"]
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        rgb = np.stack([np.round(u * 255 / (W - 1)), np.round(v * 255 / (H - 1)),
                        np.zeros_like(u)], 2).astype(np.uint8)
        cf = "m1_left" if m == "mobile_1" else "m2_color"
        sp = RM.AGENTS[m]
        for k in range(int(DUR * c["rate"])):
            t = T0 + c["off"] + k * S // c["rate"]
            recs.append((t, sp["info"], info(W, H, c["f"], cf, t)))
            recs.append((t, sp["color"], image(rgb, "rgb8", cf, t)))
            if m == "mobile_1":
                d = np.full((H // 2, W // 2), WALL, np.float32)
                recs.append((t, sp["depth_info"], info(W // 2, H // 2, c["f"] / 2, cf, t)))
                recs.append((t, sp["depth"], image(d, "32FC1", cf, t)))
            else:
                d = np.full((DEPTH2["H"], DEPTH2["W"]), round((WALL - 0.02) * 1000), np.uint16)
                recs.append((t, sp["depth_info"], info(DEPTH2["W"], DEPTH2["H"], DEPTH2["f"],
                                                       "m2_depth", t)))
                recs.append((t, sp["depth"], image(d, "16UC1", "m2_depth", t)))
    w = BagWriter(path)
    for t, topic, msg in sorted(recs, key=lambda r: r[0]):
        w.write(topic, msg, t)
    w.close()


def main():
    tmp = tempfile.mkdtemp()
    f = []
    try:
        # gap filling: a one-pixel hole takes its neighbour, an empty edge stays empty
        D = np.zeros((6, 6), np.float32)
        D[2:4, 2:4] = WALL
        D[2, 2] = 0
        F = RM.fill_gaps(D)
        if F[2, 2] != WALL or F[0, 0] != 0 or F.max() != WALL:
            f.append("fill_gaps: hole %s, empty corner %s, max %s (want %s, 0, %s)"
                     % (F[2, 2], F[0, 0], F.max(), WALL, WALL))
        bag = os.path.join(tmp, "bag")
        write_bag(bag)
        out = os.path.join(tmp, "scene")
        r = RM.convert(bag, out, rate=10.0, width=120)
        tgt = r["camera"]
        # one camera: the extent both cover (mobile_1 across, mobile_2 up and down)
        tx, ty = 79.5 / 100.0, 49.5 / 90.0
        if abs(tgt.fx - 59.5 / tx) > 1e-6 or tgt.height != int(2 * np.floor(59.5 / tx * ty)):
            f.append("common camera %s, want f %.3f, H %d" % (tgt, 59.5 / tx,
                                                                int(2 * np.floor(59.5 / tx * ty))))
        names = sorted(os.listdir(out))
        if names != ["agent_0_mobile_1", "agent_1_mobile_2"]:
            f.append("scene holds %s, want only the two agent folders" % names)
        ts = [np.loadtxt(os.path.join(out, n, "timestamps.txt")) for n in names]
        tr = [np.loadtxt(os.path.join(out, n, "traj.txt")) for n in names]
        n = len(ts[0])
        if len(ts[1]) != n or len(tr[0]) != n or len(tr[1]) != n:
            f.append("frame counts %s / poses %s differ" % ([len(x) for x in ts], [len(x) for x in tr]))
        for nm in names:
            got = len([x for x in os.listdir(os.path.join(out, nm, "results")) if x.startswith("frame")])
            if got != n:
                f.append("%s: %d images, %d stamps" % (nm, got, n))
        s0 = ts[0] - T0 / S
        if s0.min() < 1.0 - 0.05:
            f.append("first frame at %.2f s, before mobile_2 has poses (1 s)" % s0.min())
        if np.any((s0 > GAP[0] + 0.05) & (s0 < GAP[1] - 0.05)):
            f.append("frames inside mobile_2's pose gap: %s" % s0[(s0 > GAP[0]) & (s0 < GAP[1])])
        skew = np.abs(ts[0] - ts[1]).max()
        if skew > 0.04:
            f.append("agents' frames %.0f ms apart (one tick at 10 Hz, 15 / 30 Hz images)" % (skew * 1e3))
        if not 35 <= n <= 42:                                 # 5 s at 10 Hz less the 0.8 s gap
            f.append("%d frames, want ~40" % n)
        for m, t, P in zip(CAMS, ts, tr):
            err = max(abs(P[i].reshape(4, 4)[0, 3] - pose(m, int(round(t[i] * S)))[0, 3])
                      for i in range(len(t)))
            if err > 1e-3:
                f.append("%s: pose off its image stamp by %.1f mm" % (m, err * 1e3))
        # colour: the common camera's ray in each source image
        for m, nm in zip(CAMS, names):
            c = CAMS[m]
            img = cv2.imread(os.path.join(out, nm, "results", "frame000005.jpg"))[:, :, ::-1]
            uu = np.arange(tgt.width)
            us = c["f"] * (uu - tgt.cx) / tgt.fx + (c["W"] - 1) / 2.0
            want = us * 255 / (c["W"] - 1)
            row = img[int(tgt.cy), :, 0].astype(float)
            e = np.abs(row[2:-2] - want[2:-2]).max()
            if e > 8:
                f.append("%s: colour column off by %.0f levels (resampling)" % (m, e))
        # depth: 3 m in both colour cameras
        for nm in names:
            d = cv2.imread(os.path.join(out, nm, "results", "depth000005.png"),
                           cv2.IMREAD_UNCHANGED).astype(float) / RM.DEPTH_SCALE
            valid = d > 0
            if valid.mean() < 0.95:
                f.append("%s: %.0f %% of pixels with depth" % (nm, 100 * valid.mean()))
            if valid.any() and np.abs(np.median(d[valid]) - WALL) > 0.003:
                f.append("%s: wall at %.3f m, want %.3f" % (nm, np.median(d[valid]), WALL))
            if valid.any() and np.abs(d[valid] - WALL).max() > 0.01:
                f.append("%s: depth up to %.2f m off the wall (gap filling)" % (
                    nm, np.abs(d[valid] - WALL).max()))
        y = open(out + ".yaml").read()
        for k in ("input_path: %s" % out, "agent_ids: [0, 1]", "W: %d" % tgt.width,
                  "H: %d" % tgt.height, "depth_scale: 6553.5"):
            if k not in y:
                f.append("config lacks %r" % k)
        try:
            RM.convert(bag, out)
            f.append("a second run into the same scene was not refused")
        except mc.ConvertError:
            pass
        # an older bag: no /tf_static, the extrinsic given
        bag2 = os.path.join(tmp, "bag_old")
        write_bag(bag2, tf_static=False)
        out2 = os.path.join(tmp, "scene_old")
        RM.convert(bag2, out2, rate=10.0, width=120, max_frames=3,
                   depth_extrinsics={"mobile_2": list(mc.T_to_xyzq(T_CD2))})
        d = cv2.imread(os.path.join(out2, "agent_1_mobile_2", "results", "depth000001.png"),
                       cv2.IMREAD_UNCHANGED).astype(float) / RM.DEPTH_SCALE
        if not (d > 0).any() or abs(np.median(d[d > 0]) - WALL) > 0.003:
            f.append("no /tf_static, extrinsic given: wall at %s m, want %.3f"
                     % (np.median(d[d > 0]) if (d > 0).any() else None, WALL))
    except Exception as e:                                         # noqa: BLE001
        import traceback
        f.append(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "replica multiagent ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()

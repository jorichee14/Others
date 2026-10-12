#!/usr/bin/env python
"""Self-test for data_quality_validation/camera_lidar_sync.py on made-up
bags: a robot turning at random for 80 s; its camera (15 Hz) sees a texture
sliding by fx times the yaw, its LiDAR (10 Hz) a static room turning by
minus the yaw, stamped as an Ouster at the start of its 0.1 s sweep (the
points' own times 0-0.1 s after it), plus the robot's own mast 0.5 m away
and a blocked sector, both fixed to the sensor.

  synced  both stamped on time: every segment 0 s, PASS
  late    the LiDAR's stamps 0.3 s behind: every segment +0.30 s, FAIL;
          the plot written

    python scoop_pipeline/tests/test_camera_lidar_sync.py
"""
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "datasets"))
sys.path.insert(0, os.path.join(ROOT, "data_quality_validation"))
import camera_lidar_sync as CL                                      # noqa: E402
import mcap_convert as mc                                           # noqa: E402
import swarm_slam as SS                                             # noqa: E402
from scoop import rosmsg                                            # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402

T0 = 1_790_000_000_000_000_000
FX, W, H = 200.0, 240, 136
IMG, INFO, PTS = "/r/cam/image", "/r/cam/camera_info", "/r/lidar/points"
PT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("t", "<u4")])


def make_bag(path, lidar_late):
    rng = np.random.default_rng(3)
    t = np.arange(0, 80, 0.005)
    w = np.clip(np.convolve(rng.normal(0, 1, len(t)), np.ones(300) / 300, "same") * 5, -0.5, 0.5)
    yaw = np.cumsum(w) * 0.005
    # the room: a range per degree of azimuth, points at three heights
    rr = 4 + 3 * np.convolve(rng.random(400), np.ones(8) / 8, "same")[20:380]
    az = rng.uniform(-np.pi, np.pi, 4000)
    rng_az = np.interp(np.degrees(az) % 360, np.arange(360), rr)
    world = np.c_[rng_az * np.cos(az), rng_az * np.sin(az), rng.choice([-0.5, 0.0, 1.0], len(az))]
    ma = rng.uniform(2.0, 2.4, 300)                       # the mast, in the sensor frame
    mast = np.c_[0.5 * np.cos(ma), 0.5 * np.sin(ma), rng.uniform(-0.5, 1.0, 300)]
    pad = int(FX * (np.abs(yaw).max() + 0.1)) + 5
    tex = cv2.GaussianBlur(np.uint8(rng.integers(0, 255, (H, W + 2 * pad))), (5, 5), 1.5)
    tgt = mc.Intrinsics(FX, FX, W / 2, H / 2, W, H, [0.0] * 4, None)
    recs = []
    for k in range(0, len(t), 13):                       # ~15 Hz
        ts = T0 + int(round(t[k] * 1e9))
        x0 = int(round(pad - FX * yaw[k]))
        recs.append((ts, IMG, SS.image(np.ascontiguousarray(tex[:, x0:x0 + W]), "mono8", "cam", ts)))
        recs.append((ts, INFO, SS.camera_info(tgt, "cam", ts)))
    for k in range(0, len(t), 20):                       # 10 Hz
        c, s = np.cos(-yaw[k]), np.sin(-yaw[k])
        p = world @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
        p = p[np.abs(np.arctan2(p[:, 1], p[:, 0]) + 2.0) > 0.4]   # a blocked sector around -115 deg
        p = np.r_[p, mast]
        pts = np.zeros(len(p), PT)
        pts["x"], pts["y"], pts["z"] = p[:, 0], p[:, 1], p[:, 2]
        pts["t"] = rng.integers(0, 100_000_000, len(p))
        # the sweep 0.05 s either side of the pose's time, stamped at its start
        ts = T0 + int(round((t[k] - 0.05 + lidar_late) * 1e9))
        recs.append((ts, PTS, rosmsg.pointcloud2(pts, ts, "lidar")))
    w_ = BagWriter(path)
    for ts, topic, msg in sorted(recs, key=lambda r: (r[0], r[1])):
        w_.write(topic, msg, ts)
    w_.close()


def main():
    f = []
    tmp = tempfile.mkdtemp()
    try:
        for name, late, ok in (("synced", 0.0, True), ("late", 0.3, False)):
            bag = os.path.join(tmp, name)
            make_bag(bag, late)
            png = os.path.join(tmp, name + ".png")
            rep = CL.check(bag, IMG, INFO, PTS, segment=30.0, tol=0.05, max_shift=2.0, plot=png)
            seg = [x for x in rep["segments"] if x["counted"]]
            if len(seg) < 2 or any(abs(x["shift_s"] - late) > 0.03 or x["correlation"] < 0.9
                                   or abs(x["scale"] - 1) > 0.15 for x in seg):
                f.append("%s: segments %s, want %+.2f s" % (name, rep["segments"], late))
            if rep["pass"] != ok or ("PASS" if ok else "FAIL") not in CL.report(rep):
                f.append("%s: pass %s\n%s" % (name, rep["pass"], CL.report(rep)))
            if not os.path.exists(png):
                f.append("%s: no plot" % name)
    except Exception:                                              # noqa: BLE001
        import traceback
        f.append(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "camera lidar sync ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()

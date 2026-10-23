#!/usr/bin/env python
"""Self-test for checks/sync_quality.py (check.py sensors) on a made-up bag:
a robot turning at random for 90 s with an Ouster (10 Hz scans of a static
room, on time), an Ouster IMU (100 Hz, on time), a ZED IMU (200 Hz, stamped
80 ms late), a ZED RGB camera (15 Hz, on time) and a ZED depth camera (30 Hz,
stamped 50 ms late); the registered trajectory is the robot's true yaw at the
scan stamps.

  every stream found under the robot, the stamps' rates right, the lags
  recovered within 20 ms, the two late streams FAIL and the others PASS;
  without a trajectory the scans are the reference and the rest line up on
  them; the markdown and the figure written.

    python scoop_pipeline/tests/test_sync_quality.py
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
sys.path.insert(0, os.path.join(ROOT, "checks"))
import sync_quality as SQ                                           # noqa: E402
import mcap_convert as mc                                           # noqa: E402
import swarm_slam as SS                                             # noqa: E402
from scoop import rosmsg                                            # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402

T0 = 1_790_000_000_000_000_000
FX, W, H = 200.0, 240, 136
PT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("t", "<u4")])
LATE = {"depth": 0.05, "imu_zed": 0.08}


def make_bag(path, tum):
    rng = np.random.default_rng(5)
    t = np.arange(0, 90, 0.005)
    w = np.clip(np.convolve(rng.normal(0, 1, len(t)), np.ones(300) / 300, "same") * 5, -0.5, 0.5)
    yaw = np.cumsum(w) * 0.005
    rr = 4 + 3 * np.convolve(rng.random(400), np.ones(8) / 8, "same")[20:380]
    az = rng.uniform(-np.pi, np.pi, 4000)
    rng_az = np.interp(np.degrees(az) % 360, np.arange(360), rr)
    world = np.c_[rng_az * np.cos(az), rng_az * np.sin(az), rng.choice([-0.5, 0.0, 1.0], len(az))]
    pad = int(FX * (np.abs(yaw).max() + 0.1)) + 5
    tex = cv2.GaussianBlur(np.uint8(rng.integers(0, 255, (H, W + 2 * pad))), (5, 5), 1.5)
    # the depth camera sees the room's range per azimuth, in mm, rows alike
    cols = np.arange(W + 2 * pad)
    az_col = (cols - (W + 2 * pad) / 2) / FX
    # finer than the LiDAR's room: furniture-sized steps every quarter degree
    rr4 = np.repeat(rr, 4) * (1 + 0.15 * np.convolve(rng.random(1440), np.ones(6) / 6, "same"))
    dep = np.interp((np.degrees(az_col) % 360) * 4, np.arange(1440), rr4)
    dep = np.tile(dep, (H, 1)) * np.linspace(0.9, 1.1, H)[:, None] * 1000
    tgt = mc.Intrinsics(FX, FX, W / 2, H / 2, W, H, [0.0] * 4, None)
    recs = []
    for k in range(0, len(t), 13):                       # 15 Hz rgb
        ts = T0 + int(round(t[k] * 1e9))
        x0 = int(round(pad - FX * yaw[k]))
        recs.append((ts, "/r/zed/left/image_rect_color",
                     SS.image(np.ascontiguousarray(tex[:, x0:x0 + W]), "mono8", "cam", ts)))
        recs.append((ts, "/r/zed/left/camera_info", SS.camera_info(tgt, "cam", ts)))
    for k in range(0, len(t), 7):                        # ~30 Hz depth, stamped late
        ts = T0 + int(round((t[k] + LATE["depth"]) * 1e9))
        x0 = int(round(pad - FX * yaw[k]))
        d = np.ascontiguousarray(np.uint16(dep[:, x0:x0 + W]))
        d[rng.random(d.shape) < 0.05] = 0                # holes
        recs.append((ts, "/r/zed/depth/depth_registered", SS.image(d, "16UC1", "cam", ts)))
        recs.append((ts, "/r/zed/depth/camera_info", SS.camera_info(tgt, "cam", ts)))
    tum_lines = []
    for k in range(0, len(t), 20):                       # 10 Hz scans, on time
        c, s = np.cos(-yaw[k]), np.sin(-yaw[k])
        p = world @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
        pts = np.zeros(len(p), PT)
        pts["x"], pts["y"], pts["z"] = p[:, 0], p[:, 1], p[:, 2]
        pts["t"] = rng.integers(0, 100_000_000, len(p))
        ts = T0 + int(round(t[k] * 1e9))
        recs.append((ts, "/r/ouster/points", rosmsg.pointcloud2(pts, ts, "lidar")))
        tum_lines.append("%.9f 0 0 0 0 0 %.9f %.9f" % (ts * 1e-9, np.sin(yaw[k] / 2), np.cos(yaw[k] / 2)))
    for k in range(0, len(t), 2):                        # 100 Hz ouster imu, on time
        ts = T0 + int(round(t[k] * 1e9))
        recs.append((ts, "/r/ouster/imu", rosmsg.imu(ts, "imu", [0, 0, 9.81], [0, 0, w[k] + rng.normal(0, 0.01)])))
    for k in range(len(t)):                              # 200 Hz zed imu, late
        ts = T0 + int(round((t[k] + LATE["imu_zed"]) * 1e9))
        recs.append((ts, "/r/zed/imu/data", rosmsg.imu(ts, "zimu", [0, 9.81, 0], [w[k] * 0.7, 0, w[k] * 0.7])))
    w_ = BagWriter(path)
    for ts, topic, msg in sorted(recs, key=lambda r: (r[0], r[1])):
        w_.write(topic, msg, ts)
    w_.close()
    with open(tum, "w") as fh:
        fh.write("\n".join(tum_lines) + "\n")


def main():
    f = []
    tmp = tempfile.mkdtemp()
    try:
        bag, tum = os.path.join(tmp, "bag"), os.path.join(tmp, "traj_r_lidar.tum")
        make_bag(bag, tum)
        rep = SQ.assess(bag, trajs={"r": tum}, segment=30.0, max_shift=1.0, tol=0.03, log=lambda *_: None)
        R = rep["robots"].get("r")
        if R is None:
            f.append("robot r not found: %s" % list(rep["robots"]))
        else:
            want = {"lidar": 0.0, "rgb_zed": 0.0, "depth_zed": LATE["depth"], "imu_ouster": 0.0,
                    "imu_zed": LATE["imu_zed"]}
            if set(R["streams"]) != set(want):
                f.append("streams %s, want %s" % (sorted(R["streams"]), sorted(want)))
            for name, late in want.items():
                e = R["streams"].get(name)
                if e is None:
                    continue
                c = e["content"]
                if c.get("lag_s") is None or abs(c["lag_s"] - late) > 0.02 or c["correlation"] < 0.8:
                    f.append("%s: lag %s corr %s, want %+.2f s" % (name, c.get("lag_s"), c.get("correlation"), late))
                if e["pass"] != (late == 0.0):
                    f.append("%s: pass %s" % (name, e["pass"]))
                if "segment_lag_spread_s" in c and c["segment_lag_spread_s"] > 0.04:
                    f.append("%s: segment spread %.3f s" % (name, c["segment_lag_spread_s"]))
            rates = {n: R["streams"][n]["stamps"]["rate_hz"] for n in R["streams"]}
            for n, hz in (("lidar", 10), ("rgb_zed", 15.4), ("depth_zed", 28.6), ("imu_ouster", 100), ("imu_zed", 200)):
                if abs(rates[n] - hz) > 0.5:
                    f.append("%s: rate %.1f Hz, want %.1f" % (n, rates[n], hz))
            tf_ = R["streams"]["imu_ouster"]["to_frame"]
            if tf_["p95_abs_ms"] > 5.5:
                f.append("imu to frame p95 %.1f ms, want <= 5" % tf_["p95_abs_ms"])
            if rep["pass"]:
                f.append("the run must FAIL with two late streams")
        md = SQ.markdown(rep)
        if "| imu_zed |" not in md or "FAIL" not in md:
            f.append("markdown:\n%s" % md)
        png = os.path.join(tmp, "sq.png")
        SQ.draw(rep, png)
        if not os.path.exists(png):
            f.append("no figure")
        # without a trajectory: the scans are the reference
        rep2 = SQ.assess(bag, segment=30.0, max_shift=1.0, tol=0.03, log=lambda *_: None)
        R2 = rep2["robots"]["r"]
        if R2["reference"]["kind"] != "lidar_scans" or not R2["streams"]["lidar"]["pass"]:
            f.append("lidar_scans reference: %s" % R2["reference"])
        c = R2["streams"]["rgb_zed"]["content"]
        if c.get("lag_s") is None or abs(c["lag_s"]) > 0.03:
            f.append("rgb vs scans: lag %s" % c.get("lag_s"))
    except Exception:                                              # noqa: BLE001
        import traceback
        f.append(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "sync quality ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()

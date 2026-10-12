#!/usr/bin/env python
"""Self-test for datasets/euroc.py on a made-up bag of one robot (mobile_2's
topics): a stereo pair at 30 Hz for 2 s (one right image 5 ms late: no pair),
IMU at 200 Hz with one repeated stamp, camera_info with a rectified P, and
/tf_static IMU -> left and right cameras (baseline 5 cm).

It must write the EuRoC layout with --rate 10 Hz of pairs (left and right
under the same stamp), the IMU strictly increasing (gyro first), and the
parameters: intrinsics from P, T_BS the camera in the IMU frame, the IMU
rate and noise, the template's other files, logging on and the viewer off;
--imu-only with --imu-delay rewrites the IMU 0.5 s earlier, nothing else;
the ground truth also at a stereo pair between two global_poses.

    python scoop_pipeline/tests/test_euroc.py
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
import euroc as EU                                                  # noqa: E402
import mcap_convert as mc                                           # noqa: E402
import swarm_slam as SS                                             # noqa: E402
from scoop import rosmsg                                            # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402

T0 = 1_790_000_000_000_000_000
L_CAM, R_CAM, IMU_F = "camera_infra1_optical_frame", "camera_infra2_optical_frame", "camera_gyro_optical_frame"
IMU_MSG_F = "camera_imu_optical_frame"     # what the RealSense's united IMU messages name (in no TF tree)
COL = "camera_color_optical_frame"
T_IC = np.eye(4)                           # the colour camera in the IMU frame: 3 cm along y
T_IC[1, 3] = 0.03


def make_bag(path, imu_on_tf=False, tf_no_baseline=False):
    w = BagWriter(path)
    tgt = mc.Intrinsics(380.0, 381.0, 320.5, 240.5, 640, 480, [0.0] * 4, None)
    # IMU -> left: 1 cm along x, rotated 90 deg about z; left -> right: 5 cm along x
    T_il = np.eye(4)
    T_il[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    T_il[:3, 3] = [0.01, 0.0, 0.0]
    T_lr = np.eye(4)
    T_lr[0, 3] = 0.05
    T_tf = np.eye(4) if tf_no_baseline else T_lr          # as the RealSense driver: both at one place
    q_il, q_lr, q_ic = mc.T_to_xyzq(T_il), mc.T_to_xyzq(T_tf), mc.T_to_xyzq(T_IC)
    if imu_on_tf:
        w.write("/tf_static", rosmsg.tf_message([(L_CAM, R_CAM, q_lr[:3], q_lr[3:]),
                                                 (IMU_F, COL, q_ic[:3], q_ic[3:])], T0), T0)
        for k in range(3):
            w.write("/tf", rosmsg.tf_message([("odom", "base", [k, 0, 0], [0, 0, 0, 1]),
                                              (IMU_F, L_CAM, q_il[:3], q_il[3:])], T0 + k), T0 + k)
    else:
        w.write("/tf_static", rosmsg.tf_message([(IMU_F, L_CAM, q_il[:3], q_il[3:]),
                                                 (L_CAM, R_CAM, q_lr[:3], q_lr[3:]),
                                                 (IMU_F, COL, q_ic[:3], q_ic[3:])], T0), T0)
    recs = []
    for k in range(60):                                   # 30 Hz for 2 s
        t = T0 + k * 33_333_333
        img = np.full((480, 640), k, np.uint8)
        for topic, frame, dt in (("/mobile_2/infra1/image_rect_raw", L_CAM, 0),
                                 ("/mobile_2/infra2/image_rect_raw", R_CAM, 5_000_000 if k == 7 else 0)):
            recs.append((t + dt, topic, SS.image(img, "mono8", frame, t + dt)))
        for topic, frame in (("/mobile_2/infra1/camera_info", L_CAM), ("/mobile_2/infra2/camera_info", R_CAM)):
            ci = SS.camera_info(tgt, frame, t)
            if tf_no_baseline and frame == R_CAM:
                ci.p[3] = -380.0 * 0.05                     # rectified right: Tx = -fx b
            recs.append((t, topic, ci))
    for k in range(20):                                   # global_pose: the colour camera, 10 Hz, 1 m/s along x
        t = T0 + k * 100_000_000
        recs.append((t, "/mobile_2/color/camera_info", SS.camera_info(tgt, COL, t)))
        recs.append((t, "/mobile_2/global_pose", rosmsg.pose_stamped(t, "map", gt_cam(k)[:3, 3],
                                                                     mc.T_to_xyzq(gt_cam(k))[3:])))
    for k in range(400):                                  # 200 Hz, stamp 10 repeated
        t = T0 + (k if k != 11 else 10) * 5_000_000
        recs.append((t, "/mobile_2/imu", rosmsg.imu(t, IMU_MSG_F, [0.1, 0.2, 9.8], [0.01, 0.02, 0.03])))
    for t, topic, msg in sorted(recs, key=lambda r: (r[0], r[1])):
        w.write(topic, msg, t)
    w.close()
    return T_il, T_il @ T_lr


def gt_cam(k):
    """The colour camera in map at global_pose k: 0.1 m per 0.1 s along x, turned 90 deg about z."""
    T = np.eye(4)
    T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    T[:3, 3] = [0.1 * k, 2.0, 1.0]
    return T


def template(d):
    os.makedirs(os.path.join(d, "flags"))
    open(os.path.join(d, "PipelineParams.yaml"), "w").write("%YAML:1.0\nfrontend_type: 1\n")
    open(os.path.join(d, "BackendParams.yaml"), "w").write("%YAML:1.0\nautoInitialize: 1\n")
    open(os.path.join(d, "flags", "stereoVIOEuroc.flags"), "w").write("--log_output=false\n--visualize=true\n"
                                                                     "--backend_type=1\n")


def yaml_list(text, key):
    line = [ln for ln in text.splitlines() if ln.strip().startswith(key)][0]
    return [float(v) for v in line.split("[")[1].split("]")[0].split(",")]


def main():
    f = []
    tmp = tempfile.mkdtemp()
    try:
        bag = os.path.join(tmp, "bag")
        T_il, T_ir = make_bag(bag)
        # the IMU link only on /tf (as the ZED driver sends it), the camera chain on /tf_static
        bag2 = os.path.join(tmp, "bag_tf")
        make_bag(bag2, imu_on_tf=True)
        tpl = os.path.join(tmp, "D455")
        template(tpl)
        out = os.path.join(tmp, "euroc")
        EU.convert(bag, out, agents=["mobile_2"], rate=10.0, template=tpl)
        root = os.path.join(out, "mobile_2")
        c0 = open(os.path.join(root, "mav0", "cam0", "data.csv")).read().splitlines()[1:]
        c1 = open(os.path.join(root, "mav0", "cam1", "data.csv")).read().splitlines()[1:]
        if c0 != c1 or not 18 <= len(c0) <= 21:
            f.append("pairs: %d left, %d right (want the same ~20 at 10 Hz)" % (len(c0), len(c1)))
        ts = [int(x.split(",")[0]) for x in c0]
        if min(np.diff(ts)) < 0.9 * 1e8:
            f.append("pairs closer than --rate allows: %s" % np.diff(ts)[:5])
        if any(t == T0 + 7 * 33_333_333 for t in ts):
            f.append("the frame whose right image is 5 ms late was paired")
        img = cv2.imread(os.path.join(root, "mav0", "cam0", "data", c0[1].split(",")[1]), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 2:
            f.append("left image not a grey PNG")
        imu = open(os.path.join(root, "mav0", "imu0", "data.csv")).read().splitlines()[1:]
        it = [int(x.split(",")[0]) for x in imu]
        if len(imu) != 399 or any(b <= a for a, b in zip(it, it[1:])):
            f.append("imu: %d rows, increasing %s" % (len(imu), all(b > a for a, b in zip(it, it[1:]))))
        v = [float(x) for x in imu[0].split(",")[1:]]
        if not np.allclose(v, [0.01, 0.02, 0.03, 0.1, 0.2, 9.8]):
            f.append("imu row %s: want gyro then accel" % v)
        # --imu-only with --imu-delay: the same samples, every stamp 0.5 s earlier
        r = EU.convert(bag, out, agents=["mobile_2"], rate=10.0, template=tpl, imu_only=True, imu_delay=0.5)
        imu2 = open(os.path.join(root, "mav0", "imu0", "data.csv")).read().splitlines()[1:]
        it2 = [int(x.split(",")[0]) for x in imu2]
        if len(imu2) != 399 or [a - b for a, b in zip(it, it2)] != [500_000_000] * 399 \
                or [x.split(",")[1:] for x in imu2] != [x.split(",")[1:] for x in imu] or r["lines"]:
            f.append("--imu-only --imu-delay 0.5: %d rows, first %s (was %s)" % (len(imu2), it2[:1], it[:1]))
        EU.convert(bag, out, agents=["mobile_2"], rate=10.0, template=tpl, imu_only=True)
        if open(os.path.join(root, "mav0", "imu0", "data.csv")).read().splitlines()[1:] != imu:
            f.append("--imu-only without a delay did not restore the stamps (mobile_2 has none)")
        P = os.path.join(root, "params")
        lc, rc = open(os.path.join(P, "LeftCameraParams.yaml")).read(), open(os.path.join(P, "RightCameraParams.yaml")).read()
        if not np.allclose(yaml_list(lc, "intrinsics"), [380.0, 381.0, 320.5, 240.5]):
            f.append("intrinsics %s" % yaml_list(lc, "intrinsics"))
        for txt, T, n in ((lc, T_il, "left"), (rc, T_ir, "right")):
            got = np.array(yaml_list(txt, "data")).reshape(4, 4)
            if not np.allclose(got, T, atol=1e-6):
                f.append("%s T_BS\n%s\nwant\n%s" % (n, got, T))
        if "rate_hz: 10" not in lc:
            f.append("camera rate: %s" % [ln for ln in lc.splitlines() if "rate" in ln])
        ip = open(os.path.join(P, "ImuParams.yaml")).read()
        if "rate_hz: 200" not in ip or "gyroscope_noise_density: 0.000244" not in ip:
            f.append("imu params:\n" + ip)
        fl = open(os.path.join(P, "flags", "stereoVIOEuroc.flags")).read()
        if "--log_output=true" not in fl or "--visualize=false" not in fl or "--log_output=false" in fl \
                or "--visualize=true" in fl or "--backend_type=1" not in fl:
            f.append("flags:\n" + fl)
        if not os.path.exists(os.path.join(P, "BackendParams.yaml")):
            f.append("template files not copied")
        gt = [ln.split(",") for ln in open(os.path.join(root, "mav0", "state_groundtruth_estimate0",
                                                         "data.csv")).read().splitlines()[1:]]
        g5 = [float(v) for v in gt[5]]
        want = gt_cam(5) @ np.linalg.inv(T_IC)
        qw = mc.T_to_xyzq(want)
        if len(gt) != 20 or any(len(r) != 17 for r in gt) or int(gt[5][0]) != T0 + 5 * 100_000_000:
            f.append("ground truth: %d rows of %s columns" % (len(gt), {len(r) for r in gt}))
        elif not (np.allclose(g5[1:4], want[:3, 3], atol=1e-6)
                  and (np.allclose(g5[4:8], [qw[6], qw[3], qw[4], qw[5]], atol=1e-6)
                       or np.allclose(g5[4:8], [-qw[6], -qw[3], -qw[4], -qw[5]], atol=1e-6))
                  and np.allclose(g5[8:11], [1.0, 0.0, 0.0], atol=1e-6) and not any(g5[11:])):
            f.append("ground truth row 5 %s, want the IMU at %s moving 1 m/s along x" % (g5, want[:3, 3]))
        sy = open(os.path.join(root, "mav0", "state_groundtruth_estimate0", "sensor.yaml")).read()
        if not sy.startswith("%YAML:1.0") or "data: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0" not in sy:
            f.append("ground-truth sensor.yaml:\n" + sy)
        rep = open(os.path.join(out, "report.txt")).read()
        if "baseline 0.050 m" not in rep or "20 ground-truth poses" not in rep:
            f.append("report: " + rep)
        gtd = os.path.join(root, "mav0", "state_groundtruth_estimate0")
        shutil.rmtree(gtd)
        EU.convert(bag, out, agents=["mobile_2"], rate=10.0, template=tpl, gt_only=True)
        if len(open(os.path.join(gtd, "data.csv")).read().splitlines()) != 21:
            f.append("--gt-only did not rewrite the ground truth")
        # a stereo pair between two global_poses (Kimera-VIO wants a state within 10 ms of a frame)
        c0f = os.path.join(root, "mav0", "cam0", "data.csv")
        keep_c0 = open(c0f).read()
        open(c0f, "a").write("%d,%d.png\n" % (T0 + 550_000_000, T0 + 550_000_000))
        shutil.rmtree(gtd)
        EU.convert(bag, out, agents=["mobile_2"], rate=10.0, template=tpl, gt_only=True)
        open(c0f, "w").write(keep_c0)
        rows = {int(ln.split(",")[0]): [float(v) for v in ln.split(",")[1:]]
                for ln in open(os.path.join(gtd, "data.csv")).read().splitlines()[1:]}
        mid = rows.get(T0 + 550_000_000)
        a, b = rows.get(T0 + 500_000_000), rows.get(T0 + 600_000_000)
        if len(rows) != 21 or mid is None or a is None or b is None \
                or not np.allclose(mid[:3], (np.array(a[:3]) + b[:3]) / 2, atol=1e-6) \
                or not np.allclose(mid[7:10], [1.0, 0.0, 0.0], atol=1e-6):
            f.append("ground truth at a stereo pair between two poses: %s rows, %s" % (len(rows), mid))
        open(os.path.join(out, "report.txt"), "a").write("mobile_1: an earlier export\n")
        shutil.rmtree(os.path.join(out, "mobile_2"))
        EU.convert(bag, out, agents=["mobile_2"], rate=10.0, template=tpl)
        rep = open(os.path.join(out, "report.txt")).read().splitlines()
        if len(rep) != 2 or not rep[0].startswith("mobile_1: an earlier") or not rep[1].startswith("mobile_2:"):
            f.append("report after a second robot's export: %s" % rep)
        # /tf_static puts both cameras at one place; the right camera_info's P has the baseline
        bag3 = os.path.join(tmp, "bag_p")
        make_bag(bag3, tf_no_baseline=True)
        out3 = os.path.join(tmp, "euroc_p")
        EU.convert(bag3, out3, agents=["mobile_2"], rate=10.0, template=tpl)
        got = np.array(yaml_list(open(os.path.join(out3, "mobile_2", "params", "RightCameraParams.yaml")).read(),
                                 "data")).reshape(4, 4)
        if not np.allclose(got, T_ir, atol=1e-6) or "baseline 0.050 m (camera_info)" not in \
                open(os.path.join(out3, "report.txt")).read():
            f.append("baseline from camera_info P: right T_BS\n%s\nwant\n%s" % (got, T_ir))
        shutil.rmtree(os.path.join(out3, "mobile_2", "params"))
        r = EU.convert(bag3, out3, agents=["mobile_2"], rate=10.0, template=tpl, params_only=True)
        lp = os.path.join(out3, "mobile_2", "params", "LeftCameraParams.yaml")
        if not os.path.exists(lp) or "rate_hz: 10" not in open(lp).read():
            f.append("--params-only did not rewrite the parameters")
        out2 = os.path.join(tmp, "euroc_tf")
        EU.convert(bag2, out2, agents=["mobile_2"], rate=10.0, template=tpl)
        got = np.array(yaml_list(open(os.path.join(out2, "mobile_2", "params", "RightCameraParams.yaml")).read(),
                                 "data")).reshape(4, 4)
        if not np.allclose(got, T_ir, atol=1e-6):
            f.append("IMU link from /tf: right T_BS\n%s\nwant\n%s" % (got, T_ir))
        try:
            EU.convert(bag, out, agents=["mobile_2"], rate=10.0, template=tpl)
            f.append("an existing export was overwritten")
        except mc.ConvertError:
            pass
    except Exception:                                              # noqa: BLE001
        import traceback
        f.append(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "euroc ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()

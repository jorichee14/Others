#!/usr/bin/env python3
"""A multi-robot pass as Kimera-VIO input (github.com/MIT-SPARK/Kimera-VIO,
its offline stereoVIOEuroc): every robot's rectified stereo pair and IMU in
the EuRoC layout, with a Kimera-VIO parameter folder made from the recorded
calibration.

    python scoop_pipeline/datasets/euroc.py <finalized merged bag> [<out dir>]
        [--agents mobile_1 mobile_2] [--rate 20] [--template <Kimera-VIO params/D455>]
        [--gt-only | --params-only]

Writes <out> (default data/processed/<date>/<pass>/datasets/euroc/<pass>_<date>)/<machine>/:

  mav0/cam0/data.csv, data/<ns>.png   left image (grey), stamped as recorded
  mav0/cam1/data.csv, data/<ns>.png   right image of the same stereo pair
  mav0/imu0/data.csv                  ns, gyro x y z (rad/s), accel x y z (m/s^2)
  mav0/state_groundtruth_estimate0/   the IMU (Kimera-VIO's body) in map at every
                                      /<robot>/global_pose (the colour camera,
                                      moved through /tf_static) and, interpolated,
                                      at every stereo pair (autoInitialize: 0
                                      starts from the state within 10 ms of its
                                      first frame): ns, position, quaternion
                                      w x y z, velocity (finite differences),
                                      biases 0; Kimera-VIO's reader wants it, and
                                      logs it next to its estimate
  params/                             Kimera-VIO's parameters: --template's
                                      (D455: stereo + IMU, frontend_type 1),
                                      with LeftCameraParams.yaml and
                                      RightCameraParams.yaml (the rectified
                                      intrinsics of camera_info's P, no
                                      distortion; T_BS the camera in the IMU
                                      frame from /tf_static, the right one the
                                      left moved by the rectified baseline of
                                      the right camera_info's P where it has
                                      one: a driver's /tf_static may put both
                                      at one place), ImuParams.yaml
                                      (identity: the body is the IMU; its rate
                                      and the machine's noise) and the flags
                                      with logging on and the viewer off
  report.txt

--gt-only: (re)write only mav0/state_groundtruth_estimate0 of an existing export.
--imu-only: (re)write only mav0/imu0 of an existing export (after a new --imu-delay).
--imu-delay S: how far the IMU's stamps run behind the images' (s), taken off
  every IMU stamp. Default: measured in the bag against the machine's
  reference IMU (mobile_1: the Ouster's, scoop/imu_sync.py), so a bag whose
  ZED IMU is still late (20260924: the wrapper on the robot stamped it 2.12 s
  late) and one the retimed step already corrected both come out right; 0
  where there is no reference or the measurement is not sure.
  analysis/slam/imu_check.py checks the export against the images.
--params-only: (re)write only params/ of an existing export (rates from its data.csv files).

A stereo pair is a left and right image within 2 ms, at most --rate per
second; the IMU goes in whole. Frames come from each message's header; the
camera and IMU frames are the ones their messages name (the RealSense's united
IMU: the gyro's frame, as its messages name one no TF tree has). The IMU's link to the
cameras comes from /tf_static, or, where a driver publishes it on /tf (the
ZED's zed_imu_link), from the first /tf message of an edge touching the IMU
frame: the IMU is rigid on the camera.
"""
import argparse
import logging
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import mcap_convert as mc                                           # noqa: E402

log = logging.getLogger("euroc")

# per robot: the rectified stereo pair, its camera_infos, the IMU, and its noise
# (mobile_1: Kimera-VIO's Zed2i profile; mobile_2: isaac_ros_visual_slam's RealSense values)
MACHINES = {
    "mobile_1": {"left": "/mobile_1/zed/left/image_rect_color", "right": "/mobile_1/zed/right/image_rect_color",
                 "left_info": "/mobile_1/zed/left/camera_info", "right_info": "/mobile_1/zed/right/camera_info",
                 "imu": "/mobile_1/zed/imu/data",
                 # the IMU's delay is measured against the Ouster's IMU (20260924: 2.12 s; imu_check.py:
                 # gyro vs images +2.12 / +2.08 s, vs the ground truth +2.14 s over 26-200 and 200-440 s)
                 "imu_ref": "/mobile_1/ouster/imu",
                 "noise": {"gyroscope_noise_density": 3.0e-4, "gyroscope_random_walk": 1.0e-5,
                           "accelerometer_noise_density": 2.0e-3, "accelerometer_random_walk": 3.0e-3}},
    "mobile_2": {"left": "/mobile_2/infra1/image_rect_raw", "right": "/mobile_2/infra2/image_rect_raw",
                 "left_info": "/mobile_2/infra1/camera_info", "right_info": "/mobile_2/infra2/camera_info",
                 "imu": "/mobile_2/imu",
                 # the united IMU's messages name camera_imu_optical_frame, which the driver
                 # puts in no TF tree; it is the gyro's frame (as processing/vslam.py takes it)
                 "imu_frame": "camera_gyro_optical_frame",
                 "noise": {"gyroscope_noise_density": 2.44e-4, "gyroscope_random_walk": 1.9393e-5,
                           "accelerometer_noise_density": 1.862e-3, "accelerometer_random_walk": 3.0e-3}},
}
TEMPLATE = os.path.expanduser("~/workspaces/hydra_ws/src/kimera_vio/params/D455")
PAIR_TOL_NS = 2_000_000

CAMERA = """%%YAML:1.0
camera_id: %(name)s
T_BS:
  cols: 4
  rows: 4
  data: [%(T)s]
rate_hz: %(rate)d
resolution: [%(w)d, %(h)d]
camera_model: pinhole
intrinsics: [%(fx).6f, %(fy).6f, %(cx).6f, %(cy).6f]
distortion_model: radtan
distortion_coefficients: [0.0, 0.0, 0.0, 0.0]
"""

IMU = """%%YAML:1.0
imu_preintegration_type: 1
T_BS:
  cols: 4
  rows: 4
  data: [1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0]
rate_hz: %(rate)d
imu_bias_init_sigma: 1e-3
gyroscope_noise_density: %(gyroscope_noise_density)g
gyroscope_random_walk: %(gyroscope_random_walk)g
accelerometer_noise_density: %(accelerometer_noise_density)g
accelerometer_random_walk: %(accelerometer_random_walk)g
do_imu_rate_time_alignment: 0
time_alignment_window_size_s: 10.0
time_alignment_variance_threshold_scaling: 0.0
imu_integration_sigma: 1.0e-8
imu_time_shift: 0.0
n_gravity: [0.0, 0.0, -9.81]
"""


def default_out(bag):
    import replica_multiagent as RM
    d = RM.default_out(bag)
    return os.path.join(os.path.dirname(os.path.dirname(d)), "euroc", os.path.basename(d))


def imu_edges_from_tf(paths, imu_frame):
    """{(parent, child): T} of the first /tf transform of every edge touching
    imu_frame (the IMU is rigid on the camera; some drivers send it on /tf)."""
    out = {}
    try:
        for m in mc.read(paths, ["/tf"]):
            for t in m.ros_msg.transforms:
                k, T = mc._tf_edge(t)
                if imu_frame in k and k not in out:
                    out[k] = T
            if out:
                break
    except Exception as e:                                         # noqa: BLE001
        log.warning("  could not read /tf: %s" % e)
    return out


def write_gt(paths, edges, machine, imu_frame, folder, at_ns=()):
    """mav0/state_groundtruth_estimate0: the IMU in map from /<robot>/global_pose,
    at its own stamps and at at_ns (the stereo pairs) where it is no more than
    0.5 s from a pose either side."""
    import replica_multiagent as RM
    spec = RM.AGENTS[machine]
    info = mc.first_msg(paths, spec["info"])
    if info is None:
        raise mc.ConvertError(f"{machine}: no {spec['info']} (the colour camera global_pose is of)")
    col = info.header.frame_id.lstrip("/")
    T_ic = mc.lookup_static(edges, imu_frame, col)
    if T_ic is None:
        raise mc.ConvertError(f"{machine}: the IMU frame {imu_frame} and the colour frame {col} are not connected")
    ts0, pos, quat, _ = mc.load_poses(paths, spec["pose"])
    T_ci = np.linalg.inv(T_ic)
    poses = mc.PoseInterp(ts0, pos, quat) if len(ts0) > 1 else None
    ts0 = np.asarray(ts0, np.int64)
    extra = [int(t) for t in at_ns if poses is not None and poses.at(int(t)) is not None
             and poses.dt_to_nearest(int(t)) > 1e-3]               # not a pose's own stamp again
    ts = np.unique(np.concatenate([ts0, np.asarray(extra, np.int64)]))
    P = np.empty((len(ts), 3))
    Q = np.empty((len(ts), 4))
    for k in range(len(ts)):
        Tc = poses.at(int(ts[k])) if poses is not None else mc.xyzq_to_T(list(pos[0]) + list(quat[0]))
        T = Tc @ T_ci
        q = mc.T_to_xyzq(T)
        P[k], Q[k] = T[:3, 3], np.array(q[3:]) / np.linalg.norm(q[3:])
    V = np.gradient(P, (ts - ts[0]) * 1e-9, axis=0) if len(ts) > 1 else np.zeros_like(P)
    os.makedirs(folder)
    open(os.path.join(folder, "sensor.yaml"), "w").write(
        "%YAML:1.0\nsensor_type: visual-inertial\nT_BS:\n  cols: 4\n  rows: 4\n"
        "  data: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]\n")
    with open(os.path.join(folder, "data.csv"), "w") as fh:
        fh.write("#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],"
                 "v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1],b_w_RS_S_x [rad s^-1],"
                 "b_w_RS_S_y [rad s^-1],b_w_RS_S_z [rad s^-1],b_a_RS_S_x [m s^-2],b_a_RS_S_y [m s^-2],"
                 "b_a_RS_S_z [m s^-2]\n")
        for k in range(len(ts)):
            x, y, z, w = Q[k]
            fh.write("%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,0,0,0,0,0,0\n"
                     % (ts[k], P[k, 0], P[k, 1], P[k, 2], w, x, y, z, V[k, 0], V[k, 1], V[k, 2]))
    return len(ts)


def pair_stamps(root):
    """The stereo pairs' stamps (ns) of an export's mav0/cam0/data.csv."""
    f = os.path.join(root, "mav0", "cam0", "data.csv")
    if not os.path.exists(f):
        return []
    return [int(ln.split(",")[0]) for ln in open(f) if ln.strip() and not ln.startswith("#")]


def measured_delay(paths, spec, machine):
    """Seconds the machine's IMU runs behind its reference IMU in the bag (0
    without a reference, or when the two do not agree enough to say)."""
    if not spec.get("imu_ref"):
        return 0.0
    from scoop import bag as B, imu_sync as IS
    reader = B.open_bag(paths[0] if len(paths) == 1 else os.path.dirname(paths[0]))
    if spec["imu_ref"] not in reader.topics():
        log.warning("  %s: no %s to measure the IMU's delay against: none taken off" % (machine, spec["imu_ref"]))
        return 0.0
    r = IS.measure(reader, spec["imu"], spec["imu_ref"])
    if not r.ok:
        log.warning("  %s: IMU vs %s not sure (correlation %.2f): no delay taken off" % (
            machine, spec["imu_ref"], r.correlation))
        return 0.0
    log.info("  %s: %s %+.3f s behind %s (correlation %.2f)" % (machine, spec["imu"], r.shift_s, spec["imu_ref"],
                                                              r.correlation))
    return r.shift_s


def write_imu(paths, topic, path, delay_ns):
    """mav0/imu0/data.csv: every IMU message, its stamp less delay_ns, strictly
    increasing (a repeated or earlier stamp left out). (stamps written)"""
    t_imu = []
    with open(path, "w") as fh:
        fh.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                 "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for m in mc.read(paths, [topic]):
            msg = m.ros_msg
            t = mc.stamp_ns(msg) - delay_ns
            if t_imu and t <= t_imu[-1]:
                continue                                  # Kimera-VIO wants strictly increasing stamps
            a, w = msg.linear_acceleration, msg.angular_velocity
            fh.write("%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n" % (t, w.x, w.y, w.z, a.x, a.y, a.z))
            t_imu.append(t)
    return t_imu


def grey(msg):
    img = mc.decode_image(msg)
    if img.ndim == 2:
        return img
    return cv2.cvtColor(mc.to_bgr(img, msg.encoding), cv2.COLOR_BGR2GRAY)


def intrinsics(info):
    """(fx, fy, cx, cy) of a rectified image: camera_info's P."""
    P = np.asarray(info.p, float).reshape(3, 4)
    return P[0, 0], P[1, 1], P[0, 2], P[1, 2]


def flat(T):
    return ", ".join("%.9f" % v for v in np.asarray(T, float).reshape(-1))


def params(folder, template, rate_cam, rate_imu, infos, T_bs, noise):
    if not os.path.isdir(template):
        raise mc.ConvertError(f"no Kimera-VIO parameter template {template} (give --template)")
    shutil.copytree(template, folder)
    for side, name in (("left", "left_cam"), ("right", "right_cam")):
        info = infos[side]
        fx, fy, cx, cy = intrinsics(info)
        open(os.path.join(folder, "%sCameraParams.yaml" % side.capitalize()), "w").write(
            CAMERA % {"name": name, "T": flat(T_bs[side]), "rate": rate_cam, "w": info.width, "h": info.height,
                      "fx": fx, "fy": fy, "cx": cx, "cy": cy})
    open(os.path.join(folder, "ImuParams.yaml"), "w").write(IMU % dict(noise, rate=rate_imu))
    pp = os.path.join(folder, "PipelineParams.yaml")
    txt = open(pp).read()
    if "frontend_type: 1" not in txt:
        raise mc.ConvertError(f"{template}: not a stereo + IMU profile (frontend_type 1)")
    fl = os.path.join(folder, "flags", "stereoVIOEuroc.flags")
    lines = [ln for ln in open(fl).read().splitlines() if not ln.startswith(("--log_output", "--visualize="))]
    open(fl, "w").write("\n".join(["--log_output=true", "--visualize=false"] + lines) + "\n")


def rectified_baseline(info):
    """b (m) of a rectified right camera: its P has Tx = -fx b (0 when not given)."""
    P = np.asarray(info.p, float).reshape(3, 4)
    return -P[0, 3] / P[0, 0] if P[0, 0] else 0.0


def csv_rate(path):
    t = np.array([int(ln.split(",")[0]) for ln in open(path).read().splitlines()[1:] if ln.strip()])
    return int(round((len(t) - 1) * 1e9 / (t[-1] - t[0]))) if len(t) > 1 else 0


def convert_robot(paths, edges, machine, out, rate, template, gt_only=False, params_only=False,
                  imu_only=False, imu_delay=None):
    spec = MACHINES[machine]
    delay = delay_ns = None                               # measured only where the IMU is written
    infos = {"left": mc.first_msg(paths, spec["left_info"]), "right": mc.first_msg(paths, spec["right_info"])}
    for side, info in infos.items():
        if info is None:
            raise mc.ConvertError(f"{machine}: no {spec[side + '_info']}")
    imu0 = mc.first_msg(paths, spec["imu"])
    if imu0 is None:
        raise mc.ConvertError(f"{machine}: no IMU on {spec['imu']}")
    imu_frame = spec.get("imu_frame") or imu0.header.frame_id.lstrip("/")
    cams = [infos[s].header.frame_id.lstrip("/") for s in ("left", "right")]
    if any(mc.lookup_static(edges, imu_frame, f) is None for f in cams):
        edges = dict(edges)
        edges.update(imu_edges_from_tf(paths, imu_frame))
    T_bs = {}
    for side in ("left", "right"):
        f = infos[side].header.frame_id.lstrip("/")
        T = mc.lookup_static(edges, imu_frame, f)
        if T is None:
            near = sorted({x for k in edges for x in k if machine.split("_")[0] in x or "zed" in x
                           or "camera" in x or "imu" in x})
            raise mc.ConvertError(f"{machine}: the IMU frame {imu_frame} and the {side} camera frame {f} "
                                  f"are not connected in /tf_static or /tf (frames there: {near})")
        T_bs[side] = T
    # rectified stereo: the right camera is the left moved by the baseline along its x axis
    b_p = rectified_baseline(infos["right"])
    b_src = "/tf_static"
    if abs(b_p) > 1e-3:
        D = np.eye(4)
        D[0, 3] = b_p
        T_bs["right"] = T_bs["left"] @ D
        b_src = "camera_info"
    root = os.path.join(out, machine)
    if params_only:
        m0 = os.path.join(root, "mav0")
        if not os.path.isdir(m0):
            raise mc.ConvertError(f"--params-only: no {m0} (export it first)")
        shutil.rmtree(os.path.join(root, "params"), ignore_errors=True)
        params(os.path.join(root, "params"), template, csv_rate(os.path.join(m0, "cam0", "data.csv")),
               csv_rate(os.path.join(m0, "imu0", "data.csv")), infos, T_bs, spec["noise"])
        return "%s: parameters rewritten; baseline %.3f m (%s)" % (
            machine, np.linalg.norm(T_bs["left"][:3, 3] - T_bs["right"][:3, 3]), b_src)
    if gt_only:
        gt_dir = os.path.join(root, "mav0", "state_groundtruth_estimate0")
        if not os.path.isdir(os.path.join(root, "mav0")):
            raise mc.ConvertError(f"--gt-only: no {root}/mav0 (export it first)")
        shutil.rmtree(gt_dir, ignore_errors=True)
        return "%s: %d ground-truth poses (ground truth only)" % (
            machine, write_gt(paths, edges, machine, imu_frame, gt_dir, pair_stamps(root)))
    if imu_only or not (gt_only or params_only):
        delay = imu_delay if imu_delay is not None else measured_delay(paths, spec, machine)
        delay_ns = int(round(delay * 1e9))
    if imu_only:
        d = os.path.join(root, "mav0", "imu0")
        if not os.path.isdir(d):
            raise mc.ConvertError(f"--imu-only: no {d} (export it first)")
        n = len(write_imu(paths, spec["imu"], os.path.join(d, "data.csv"), delay_ns))
        return "%s: %d IMU samples rewritten, %.3f s taken off their stamps (IMU only)" % (machine, n, delay)
    if os.path.exists(root):
        raise mc.ConvertError(f"{root} exists: remove it first")
    dirs = {k: os.path.join(root, "mav0", k) for k in ("cam0", "cam1", "imu0")}
    for k in ("cam0", "cam1"):
        os.makedirs(os.path.join(dirs[k], "data"))
    os.makedirs(dirs["imu0"])
    csv = {k: open(os.path.join(dirs[k], "data.csv"), "w") for k in ("cam0", "cam1")}
    csv["cam0"].write("#timestamp [ns],filename\n")
    csv["cam1"].write("#timestamp [ns],filename\n")
    period = int(round(1e9 / rate))
    pend = {"l": [], "r": []}
    n_pair, first, last = 0, None, None
    for m in mc.read(paths, [spec["left"], spec["right"]]):
        topic, msg = m.channel.topic, m.ros_msg
        t = mc.stamp_ns(msg)
        pend["l" if topic == spec["left"] else "r"].append((t, msg))
        while pend["l"] and pend["r"]:
            (tl, ml), (tr, mr) = pend["l"][0], pend["r"][0]
            if tl - tr > PAIR_TOL_NS:
                pend["r"].pop(0)
            elif tr - tl > PAIR_TOL_NS:
                pend["l"].pop(0)
            else:
                pend["l"].pop(0)
                pend["r"].pop(0)
                if last is not None and tl - last < 0.9 * period:
                    continue
                for k, mm in (("cam0", ml), ("cam1", mr)):
                    cv2.imwrite(os.path.join(dirs[k], "data", "%d.png" % tl), grey(mm))
                    csv[k].write("%d,%d.png\n" % (tl, tl))
                first = tl if first is None else first
                last = tl
                n_pair += 1
                if n_pair % 1000 == 0:
                    log.info("  %s: %d stereo pairs" % (machine, n_pair))
    for f in csv.values():
        f.close()
    t_imu = write_imu(paths, spec["imu"], os.path.join(dirs["imu0"], "data.csv"), delay_ns)
    n_imu = len(t_imu)
    if n_pair < 2 or n_imu < 2:
        raise mc.ConvertError(f"{machine}: {n_pair} stereo pairs and {n_imu} IMU samples")
    rate_imu = int(round(1e9 / np.median(np.diff(t_imu))))
    rate_cam = int(round((n_pair - 1) * 1e9 / (last - first)))
    n_gt = write_gt(paths, edges, machine, imu_frame, os.path.join(root, "mav0", "state_groundtruth_estimate0"),
                    pair_stamps(root))
    params(os.path.join(root, "params"), template, rate_cam, rate_imu, infos, T_bs, spec["noise"])
    b = np.linalg.norm(T_bs["left"][:3, 3] - T_bs["right"][:3, 3])
    return ("%s: %d stereo pairs at %d Hz, %dx%d, fx %.1f; %d IMU samples at %d Hz (%s, %.3f s taken off); "
            "baseline %.3f m (%s); %d ground-truth poses"
            % (machine, n_pair, rate_cam, infos["left"].width, infos["left"].height, intrinsics(infos["left"])[0],
               n_imu, rate_imu, imu_frame, delay, b, b_src, n_gt))


def convert(bag, out=None, agents=("mobile_1", "mobile_2"), rate=20.0, template=TEMPLATE, gt_only=False,
            params_only=False, imu_only=False, imu_delay=None):
    paths = mc._paths(bag)
    out = os.path.abspath(os.path.expanduser(out or default_out(bag)))
    os.makedirs(out, exist_ok=True)
    edges = mc.build_tf_static(paths)
    rp = os.path.join(out, "report.txt")
    kept = [ln for ln in (open(rp).read().splitlines() if os.path.exists(rp) else [])
            if ln.split(":")[0] not in agents]           # robots exported before stay in the report
    lines = []
    for m in agents:
        line = convert_robot(paths, edges, m, out, rate, os.path.expanduser(template), gt_only, params_only,
                             imu_only, imu_delay)
        log.info(line)
        if gt_only or params_only or imu_only:
            continue                                      # the report keeps the export's own line
        lines.append(line)
        open(rp, "w").write("\n".join(sorted(kept + lines)) + "\n")
    return {"out": out, "lines": lines}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="the finalized merged bag")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--agents", nargs="+", default=["mobile_1", "mobile_2"], choices=sorted(MACHINES))
    ap.add_argument("--rate", type=float, default=20.0, help="most stereo pairs per second")
    ap.add_argument("--template", default=TEMPLATE, help="a Kimera-VIO stereo + IMU parameter folder")
    ap.add_argument("--gt-only", action="store_true", help="(re)write only the ground truth of an existing export")
    ap.add_argument("--params-only", action="store_true", help="(re)write only the parameters of an existing export")
    ap.add_argument("--imu-only", action="store_true", help="(re)write only the IMU of an existing export")
    ap.add_argument("--imu-delay", type=float, default=None,
                    help="seconds the IMU's stamps run behind the images' (default: measured against the "
                         "machine's reference IMU)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        convert(a.bag, a.out, a.agents, a.rate, a.template, a.gt_only, a.params_only, a.imu_only, a.imu_delay)
    except mc.ConvertError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()

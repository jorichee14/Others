#!/usr/bin/env python3
"""A multi-robot pass as Kimera-VIO input (github.com/MIT-SPARK/Kimera-VIO,
its offline stereoVIOEuroc): every robot's rectified stereo pair and IMU in
the EuRoC layout, with a Kimera-VIO parameter folder made from the recorded
calibration.

    python scoop_pipeline/datasets/euroc.py <finalized merged bag> [<out dir>]
        [--agents mobile_1 mobile_2] [--rate 20] [--template <Kimera-VIO params/D455>]

Writes <out> (default data/processed/<date>/<pass>/datasets/euroc/<pass>_<date>)/<machine>/:

  mav0/cam0/data.csv, data/<ns>.png   left image (grey), stamped as recorded
  mav0/cam1/data.csv, data/<ns>.png   right image of the same stereo pair
  mav0/imu0/data.csv                  ns, gyro x y z (rad/s), accel x y z (m/s^2)
  params/                             Kimera-VIO's parameters: --template's
                                      (D455: stereo + IMU, frontend_type 1),
                                      with LeftCameraParams.yaml and
                                      RightCameraParams.yaml (the rectified
                                      intrinsics of camera_info's P, no
                                      distortion; T_BS the camera in the IMU
                                      frame from /tf_static), ImuParams.yaml
                                      (identity: the body is the IMU; its rate
                                      and the machine's noise) and the flags
                                      with logging on and the viewer off
  report.txt

A stereo pair is a left and right image within 2 ms, at most --rate per
second; the IMU goes in whole. Frames come from each message's header; the
camera and IMU frames are the ones their messages name.
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
                 "noise": {"gyroscope_noise_density": 3.0e-4, "gyroscope_random_walk": 1.0e-5,
                           "accelerometer_noise_density": 2.0e-3, "accelerometer_random_walk": 3.0e-3}},
    "mobile_2": {"left": "/mobile_2/infra1/image_rect_raw", "right": "/mobile_2/infra2/image_rect_raw",
                 "left_info": "/mobile_2/infra1/camera_info", "right_info": "/mobile_2/infra2/camera_info",
                 "imu": "/mobile_2/imu",
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


def convert_robot(paths, edges, machine, out, rate, template):
    spec = MACHINES[machine]
    infos = {"left": mc.first_msg(paths, spec["left_info"]), "right": mc.first_msg(paths, spec["right_info"])}
    for side, info in infos.items():
        if info is None:
            raise mc.ConvertError(f"{machine}: no {spec[side + '_info']}")
    imu0 = mc.first_msg(paths, spec["imu"])
    if imu0 is None:
        raise mc.ConvertError(f"{machine}: no IMU on {spec['imu']}")
    imu_frame = imu0.header.frame_id.lstrip("/")
    T_bs = {}
    for side in ("left", "right"):
        f = infos[side].header.frame_id.lstrip("/")
        T = mc.lookup_static(edges, imu_frame, f)
        if T is None:
            raise mc.ConvertError(f"{machine}: the IMU frame {imu_frame} and the {side} camera frame {f} "
                                  "are not connected in /tf_static")
        T_bs[side] = T
    root = os.path.join(out, machine)
    if os.path.exists(root):
        raise mc.ConvertError(f"{root} exists: remove it first")
    dirs = {k: os.path.join(root, "mav0", k) for k in ("cam0", "cam1", "imu0")}
    for k in ("cam0", "cam1"):
        os.makedirs(os.path.join(dirs[k], "data"))
    os.makedirs(dirs["imu0"])
    csv = {k: open(os.path.join(dirs[k], "data.csv"), "w") for k in dirs}
    csv["cam0"].write("#timestamp [ns],filename\n")
    csv["cam1"].write("#timestamp [ns],filename\n")
    csv["imu0"].write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                      "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
    period = int(round(1e9 / rate))
    pend = {"l": [], "r": []}
    n_pair, n_imu, first, last, t_imu = 0, 0, None, None, []
    for m in mc.read(paths, [spec["left"], spec["right"], spec["imu"]]):
        topic, msg = m.channel.topic, m.ros_msg
        t = mc.stamp_ns(msg)
        if topic == spec["imu"]:
            if t_imu and t <= t_imu[-1]:
                continue                                  # Kimera-VIO wants strictly increasing stamps
            a, w = msg.linear_acceleration, msg.angular_velocity
            csv["imu0"].write("%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n" % (t, w.x, w.y, w.z, a.x, a.y, a.z))
            t_imu.append(t)
            n_imu += 1
            continue
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
    if n_pair < 2 or n_imu < 2:
        raise mc.ConvertError(f"{machine}: {n_pair} stereo pairs and {n_imu} IMU samples")
    rate_imu = int(round(1e9 / np.median(np.diff(t_imu))))
    rate_cam = int(round((n_pair - 1) * 1e9 / (last - first)))
    params(os.path.join(root, "params"), template, rate_cam, rate_imu, infos, T_bs, spec["noise"])
    b = np.linalg.norm(T_bs["left"][:3, 3] - T_bs["right"][:3, 3])
    return ("%s: %d stereo pairs at %d Hz, %dx%d, fx %.1f; %d IMU samples at %d Hz (%s); baseline %.3f m"
            % (machine, n_pair, rate_cam, infos["left"].width, infos["left"].height, intrinsics(infos["left"])[0],
               n_imu, rate_imu, imu_frame, b))


def convert(bag, out=None, agents=("mobile_1", "mobile_2"), rate=20.0, template=TEMPLATE):
    paths = mc._paths(bag)
    out = os.path.abspath(os.path.expanduser(out or default_out(bag)))
    os.makedirs(out, exist_ok=True)
    edges = mc.build_tf_static(paths)
    lines = []
    for m in agents:
        line = convert_robot(paths, edges, m, out, rate, os.path.expanduser(template))
        log.info(line)
        lines.append(line)
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    return {"out": out, "lines": lines}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="the finalized merged bag")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--agents", nargs="+", default=["mobile_1", "mobile_2"], choices=sorted(MACHINES))
    ap.add_argument("--rate", type=float, default=20.0, help="most stereo pairs per second")
    ap.add_argument("--template", default=TEMPLATE, help="a Kimera-VIO stereo + IMU parameter folder")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        convert(a.bag, a.out, a.agents, a.rate, a.template)
    except mc.ConvertError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()

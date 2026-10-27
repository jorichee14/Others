#!/usr/bin/env python3
"""One odometry run's configuration, from the bag itself: what to play, how to
start the method, what it answers on, and the method's config files.

    python scoop_pipeline/slam/odometry/make_config.py <method> <bag> <machine> <run dir>
        [--start S] [--duration S] [--imu-noise FILE] [--min-rate-frac 0.9]

method: kiss_icp, rko_lio, fast_lio, fast2lamaa, fast2lamaa_loc (LiDAR,
mobile_1 / mobile_2), openvins, rtabmap_rgbd, rtabmap_map, rtabmap_loc (the
D455, mobile_2). rtabmap_map maps the colour image + the depth reprojected
into it (--rtab-image infra: the left infrared image + depth) on the
robot's pose in the twin (--gt-topic, default /<machine>/global_pose: the
mapping pass), so its database is in the twin's frame; rtabmap_loc localizes
a run in it (--db) with RTAB-Map's own odometry between recognised places.
fast2lamaa runs 2Fast-2Lamaa's odometry + mapping (one global map);
fast2lamaa_loc localizes in the reference pass's anchored map (--map, default
the one stage 04 registered the robot to), started at the reference's first
pose (--ref, default stage 04's trajectory; the play starts where it does),
trusting that seed (--coarse-init to match the first scan coarse to fine
instead): its trajectory is in map. Writes <run dir>/run.json and
<run dir>/config/ (the method's files, as the container sees them under
/work/config); run_odometry.sh then runs it.

Read from the bag, never typed in: topics and frames (first message of each),
the LiDAR's beams (the cloud's height), T_imu_lidar and the camera-IMU
transforms (/tf_static; T_lidar_imu from the robot's GLIM config, which
processing/glim_config.py took from its own bag, where the pass bag does not
link the two), the cameras' intrinsics (camera_info), the IMU's rate.

LiDAR methods get their clouds through sweep_relay.py, at the sweep's end as
a live driver publishes them (the retimed bags replay them at its start).

Indoor changes, each named in run.json's "settings" beside its published
default:
  every LiDAR method leaves out points within 1.0 m (RKO-LIO's default;
  KISS-ICP's 0.0 keeps the robot and the operator, FAST-LIO2's 4.0 m blind
  zone is most of the room), so KISS-ICP and RKO-LIO differ in the IMU alone.
  KISS-ICP deskews (default off; the Ouster gives per-point times).
  Every method publishes its odometry only (debug clouds, maps, paths off):
  the cost logged is the odometry's.

Refusals (nothing written):
  openvins without --imu-noise (a YAML with accelerometer_noise_density,
  accelerometer_random_walk, gyroscope_noise_density, gyroscope_random_walk:
  the D455 IMU's Allan variance; a guessed noise model gives a result that
  completes and is wrong);
  rtabmap_rgbd when the depth image is not in the left infrared camera
  (other intrinsics, size, or a transform that is not the identity);
  a camera method on mobile_1 (the ZED: PLAN.md; 20261015 holds ~1 fps);
  any key topic whose rate in the bag is under --min-rate-frac of its nominal
  rate (the ZED's 1 fps would be caught here), or a frame /tf_static does not
  link.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "datasets"))
import mcap_convert as mc                                           # noqa: E402

LIDAR = {"mobile_1": {"points": "/mobile_1/ouster/points", "imu": "/mobile_1/ouster/imu"},
         "mobile_2": {"points": "/mobile_2/ouster/points", "imu": "/mobile_2/ouster/imu"}}
LIDAR_HZ = 10.0
D455 = {"mobile_2": {"left": "/mobile_2/infra1/image_rect_raw",
                     "left_info": "/mobile_2/infra1/camera_info",
                     "right": "/mobile_2/infra2/image_rect_raw",
                     "right_info": "/mobile_2/infra2/camera_info",
                     "depth": "/mobile_2/depth/image_rect_raw",
                     "depth_info": "/mobile_2/depth/camera_info",
                     "color": "/mobile_2/color/image_raw",
                     "color_info": "/mobile_2/color/camera_info",
                     "color_hz": 15.0,
                     "imu": "/mobile_2/imu",
                     "hz": 30.0}}
NOISE_KEYS = ("accelerometer_noise_density", "accelerometer_random_walk",
              "gyroscope_noise_density", "gyroscope_random_walk")
MIN_RANGE = 1.0
PC2, IMU, IMG = "sensor_msgs/msg/PointCloud2", "sensor_msgs/msg/Imu", "sensor_msgs/msg/Image"
ODOM = "nav_msgs/msg/Odometry"
TFS = "geometry_msgs/msg/TransformStamped"
CFG = "/work/config"


GLIM_CONFIG = os.path.join(ROOT, "configs", "glim", "%s", "config_sensors.json")


class Refused(RuntimeError):
    pass


# ------------------------------------------------------------------ the bag

def bag_metadata(bag):
    """(start ns, duration ns) from the rosbag2 folder's metadata.yaml."""
    p = os.path.join(bag, "metadata.yaml")
    if not os.path.isfile(p):
        raise Refused(f"{bag}: no metadata.yaml (want a rosbag2 folder: ros2 bag play takes it)")
    info = yaml.safe_load(open(p))["rosbag2_bagfile_information"]
    return int(info["starting_time"]["nanoseconds_since_epoch"]), int(info["duration"]["nanoseconds"])


def topic_rate(paths, topic):
    """(messages, rate Hz) from the MCAP summaries alone (statistics and chunk
    indexes: nothing decompressed), or (0, None). The span is that of the
    chunks holding the topic, which other topics' messages widen a little:
    the rate is approximate, slightly low, which a 90 % gate absorbs."""
    from mcap.reader import make_reader
    n, lo, hi = 0, None, None
    for f in paths:
        with open(f, "rb") as fh:
            s = make_reader(fh).get_summary()
        if s is None:
            continue
        ids = {i for i, c in s.channels.items() if c.topic == topic}
        if not ids:
            continue
        if s.statistics:
            n += sum(s.statistics.channel_message_counts.get(i, 0) for i in ids)
        for ci in s.chunk_indexes:
            if ids & set(ci.message_index_offsets):
                lo = ci.message_start_time if lo is None else min(lo, ci.message_start_time)
                hi = ci.message_end_time if hi is None else max(hi, ci.message_end_time)
    if not n or lo is None or hi <= lo:
        return n, None
    return n, n / ((hi - lo) * 1e-9)


def frame_of(paths, topic):
    m = mc.first_msg(paths, topic)
    if m is None:
        raise Refused(f"no messages on {topic}")
    return m, m.header.frame_id.lstrip("/")


def static(edges, src, dst):
    """T_src_dst (dst's pose in src: maps dst coordinates into src) from /tf_static."""
    T = np.eye(4) if src == dst else mc.lookup_static(edges, src, dst)
    if T is None:
        raise Refused(f"/tf_static does not link {src} and {dst}")
    return T


def lidar_imu(edges, machine, lidar, imu):
    """(T_lidar_imu, where it came from): /tf_static, or, where the pass bag
    does not link the two (mobile_2's Ouster in the 20261015 merged bags), the
    value processing/glim_config.py took from the robot's own bag for GLIM."""
    try:
        return static(edges, lidar, imu), "/tf_static"
    except Refused:
        pass
    path = GLIM_CONFIG % machine
    m = re.search(r'"T_lidar_imu"\s*:\s*\[([^\]]*)\]', open(path).read()) if os.path.exists(path) else None
    if m is None:
        raise Refused(f"/tf_static does not link {lidar} and {imu}, and {path} has no T_lidar_imu")
    v = [float(x) for x in m.group(1).replace("\n", " ").split(",")]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(v[3:7]).as_matrix()
    T[:3, 3] = v[:3]
    return T, os.path.relpath(path, ROOT) + " (not in the bag's /tf_static)"


def to_lidar(paths, edges, machine, frame):
    """(T_frame_lidar, where it came from): the robot's LiDAR in the frame a
    method reports, to put its trajectory in the reference's frame (stage 04's
    track is the LiDAR's). The IMU of the Ouster through lidar_imu(); a camera
    through the bag's /tf_static (mobile_2: the camera-LiDAR calibration)."""
    _, lidar = frame_of(paths, LIDAR[machine]["points"])
    if frame == lidar:
        return np.eye(4), "the LiDAR itself"
    _, imu = frame_of(paths, LIDAR[machine]["imu"])
    if frame == imu:
        T_li, src = lidar_imu(edges, machine, lidar, imu)
        return np.linalg.inv(T_li), src
    return static(edges, frame, lidar), "/tf_static"


def relay(points):
    """The sweep relay (sweep_relay.py) for a LiDAR method: the player's clouds
    renamed, handed on at the sweep's end."""
    return {"in": "/replay" + points, "out": points, "hold_s": 1.0 / LIDAR_HZ}


def quat_xyzw_xyz(T):
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    return [float(v) for v in q] + [float(v) for v in T[:3, 3]]


def gate(paths, topic, nominal, frac):
    n, hz = topic_rate(paths, topic)
    if hz is None:
        raise Refused(f"{topic}: not in the bag")
    if hz < frac * nominal:
        raise Refused(f"{topic}: {hz:.2f} Hz in the bag ({n} messages), under {frac:.0%} of its "
                      f"nominal {nominal:g} Hz: a method on it would be measuring the recording")
    return {"topic": topic, "messages": n, "hz": round(hz, 2), "nominal_hz": nominal}


# ------------------------------------------------------------------ methods

def kiss_icp(paths, edges, machine, a):
    t = LIDAR[machine]
    _, lidar = frame_of(paths, t["points"])
    cfg = {"kiss_icp_node": {"ros__parameters": {
        "data": {"deskew": True, "max_range": 100.0, "min_range": MIN_RANGE},
        "mapping": {"max_points_per_voxel": 20},
        "adaptive_threshold": {"initial_threshold": 2.0, "min_motion_th": 0.1}}}}
    return {
        "files": {"kiss_icp.yaml": yaml.safe_dump(cfg, sort_keys=False)},
        "launch": ("ros2 launch kiss_icp odometry.launch.py topic:=%s use_sim_time:=true "
                   "visualize:=false publish_odom_tf:=false config_file:=%s/kiss_icp.yaml"
                   % (t["points"], CFG)),
        "play": [t["points"]], "in": {t["points"]: PC2}, "key": t["points"],
        "out": "/kiss/odometry", "out_type": ODOM, "tum_frame": lidar, "relay": relay(t["points"]),
        "proc": r"kiss_icp_node", "stamp_tol_s": 0.0, "nominal_hz": LIDAR_HZ,
        "settings": {"data.deskew": [True, False, "the Ouster gives per-point times"],
                     "data.min_range": [MIN_RANGE, 0.0, "RKO-LIO's; leaves out the robot and operator"],
                     "publish_debug_clouds": [False, True, "odometry only"]}}


def rko_lio(paths, edges, machine, a):
    t = LIDAR[machine]
    _, lidar = frame_of(paths, t["points"])
    _, imu = frame_of(paths, t["imu"])
    T_li, src = lidar_imu(edges, machine, lidar, imu)
    cfg = {"lidar_topic": t["points"], "imu_topic": t["imu"],
           "base_frame": lidar, "lidar_frame": lidar, "imu_frame": imu,
           "extrinsic_lidar2base_quat_xyzw_xyz": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
           "extrinsic_imu2base_quat_xyzw_xyz": quat_xyzw_xyz(T_li),
           "publish_local_map": False, "publish_deskewed_scan": False}
    return {
        "files": {"rko_lio.yaml": yaml.safe_dump(cfg, sort_keys=False)},
        "launch": ("ros2 launch rko_lio odometry.launch.py config_file:=%s/rko_lio.yaml mode:=online "
                   "rviz:=false use_sim_time:=true autodetect:=false" % CFG),
        "play": [t["points"], t["imu"]], "in": {t["points"]: PC2, t["imu"]: IMU}, "key": t["points"],
        "out": "/rko_lio/odom", "out_type": ODOM, "tum_frame": lidar, "relay": relay(t["points"]),
        "proc": r"rko_lio.*online_node|/online_node", "stamp_tol_s": 0.0, "nominal_hz": LIDAR_HZ,
        "settings": {"base_frame": [lidar, None, "the pose is the LiDAR's"],
                     "extrinsic_imu2base": [src, "tf", "the IMU in the LiDAR frame"],
                     "publish_local_map": [False, True, "odometry only"],
                     "publish_deskewed_scan": [False, True, "odometry only"]}}


def fast_lio(paths, edges, machine, a):
    t = LIDAR[machine]
    pc, lidar = frame_of(paths, t["points"])
    _, imu = frame_of(paths, t["imu"])
    T_li, src = lidar_imu(edges, machine, lidar, imu)
    T = np.linalg.inv(T_li)                               # the LiDAR in the IMU frame
    cfg = {"/**": {"ros__parameters": {
        "feature_extract_enable": False, "point_filter_num": 3, "max_iteration": 3,
        "filter_size_surf": 0.5, "filter_size_map": 0.5, "cube_side_length": 1000.0,
        "runtime_pos_log_enable": False, "map_file_path": "/work/fast_lio.pcd",
        "common": {"lid_topic": t["points"], "imu_topic": t["imu"],
                   "time_sync_en": False, "time_offset_lidar_to_imu": 0.0},
        "preprocess": {"lidar_type": 3, "scan_line": int(pc.height), "timestamp_unit": 3,
                       "blind": MIN_RANGE},
        "mapping": {"acc_cov": 0.1, "gyr_cov": 0.1, "b_acc_cov": 0.0001, "b_gyr_cov": 0.0001,
                    "fov_degree": 360.0, "det_range": 150.0, "extrinsic_est_en": False,
                    "extrinsic_T": [float(v) for v in T[:3, 3]],
                    "extrinsic_R": [float(v) for v in T[:3, :3].reshape(-1)]},
        "publish": {"path_en": False, "scan_publish_en": False, "dense_publish_en": False,
                    "scan_bodyframe_pub_en": False},
        "pcd_save": {"pcd_save_en": False, "interval": -1}}}}
    return {
        "files": {"fast_lio.yaml": yaml.safe_dump(cfg, sort_keys=False)},
        "launch": ("ros2 launch fast_lio mapping.launch.py config_path:=%s config_file:=fast_lio.yaml "
                   "use_sim_time:=true rviz:=false" % CFG),
        "play": [t["points"], t["imu"]], "in": {t["points"]: PC2, t["imu"]: IMU}, "key": t["points"],
        "out": "/Odometry", "out_type": ODOM, "tum_frame": imu, "relay": relay(t["points"]),
        "proc": r"fastlio_mapping", "stamp_tol_s": 0.0, "nominal_hz": LIDAR_HZ,
        "settings": {"preprocess.blind": [MIN_RANGE, 4.0, "4 m is most of the room"],
                     "preprocess.scan_line": [int(pc.height), 64, "the cloud's height"],
                     "mapping.extrinsic_T/R": [src, "identity", "the LiDAR in the IMU frame"],
                     "publish.*": [False, True, "odometry only"],
                     "pcd_save.pcd_save_en": [False, True, "odometry only"]}}


def camera(machine, method):
    if machine not in D455:
        raise Refused(f"{method} on {machine}: only mobile_2's D455 is prepared (the ZED: PLAN.md; "
                      "the 20261015 recordings hold it at ~1 fps)")
    return D455[machine]


def kalibr_cam(T_cam_imu, info, topic, overlaps):
    k = list(info.k)
    d = [float(v) for v in list(info.d)[:4]] + [0.0] * max(0, 4 - len(info.d))
    rows = "\n".join("    - [%s]" % ", ".join("%.12g" % v for v in r) for r in T_cam_imu)
    return ("  T_cam_imu:\n%s\n  cam_overlaps: %s\n  camera_model: pinhole\n"
            "  distortion_coeffs: [%s]\n  distortion_model: radtan\n"
            "  intrinsics: [%.9g, %.9g, %.9g, %.9g]\n  resolution: [%d, %d]\n"
            "  rostopic: %s\n  timeshift_cam_imu: 0.0\n"
            % (rows, overlaps, ", ".join("%.12g" % v for v in d), k[0], k[4], k[2], k[5],
               info.width, info.height, topic))


# OpenVINS's config/rs_d455/estimator_config.yaml at v2.7, with stereo on (two
# infrared cameras here, its own example is the colour camera alone)
OV_ESTIMATOR = """%%YAML:1.0
verbosity: "INFO"
use_fej: true
integration: "rk4"
use_stereo: true
max_cameras: 2
calib_cam_extrinsics: true
calib_cam_intrinsics: true
calib_cam_timeoffset: true
calib_imu_intrinsics: false
calib_imu_g_sensitivity: false
max_clones: 11
max_slam: 50
max_slam_in_update: 25
max_msckf_in_update: 40
dt_slam_delay: 1
gravity_mag: 9.81
feat_rep_msckf: "GLOBAL_3D"
feat_rep_slam: "ANCHORED_MSCKF_INVERSE_DEPTH"
feat_rep_aruco: "ANCHORED_MSCKF_INVERSE_DEPTH"
try_zupt: false
zupt_chi2_multipler: 0
zupt_max_velocity: 0.1
zupt_noise_multiplier: 10
zupt_max_disparity: 0.5
zupt_only_at_beginning: false
init_window_time: 2.0
init_imu_thresh: 1.5
init_max_disparity: 10.0
init_max_features: 50
init_dyn_use: false
init_dyn_mle_opt_calib: false
init_dyn_mle_max_iter: 50
init_dyn_mle_max_time: 0.05
init_dyn_mle_max_threads: 6
init_dyn_num_pose: 6
init_dyn_min_deg: 10.0
init_dyn_inflation_ori: 10
init_dyn_inflation_vel: 100
init_dyn_inflation_bg: 10
init_dyn_inflation_ba: 100
init_dyn_min_rec_cond: 1e-12
init_dyn_bias_g: [ 0.0, 0.0, 0.0 ]
init_dyn_bias_a: [ 0.0, 0.0, 0.0 ]
record_timing_information: false
record_timing_filepath: "/work/ov_timing.txt"
save_total_state: false
filepath_est: "/work/ov_estimate.txt"
filepath_std: "/work/ov_estimate_std.txt"
filepath_gt: "/work/ov_groundtruth.txt"
use_klt: true
num_pts: 200
fast_threshold: 30
grid_x: 5
grid_y: 5
min_px_dist: 15
knn_ratio: 0.70
track_frequency: %(track_hz)g
downsample_cameras: false
num_opencv_threads: 4
histogram_method: "HISTOGRAM"
use_aruco: false
num_aruco: 1024
downsize_aruco: true
up_msckf_sigma_px: 1
up_msckf_chi2_multipler: 1
up_slam_sigma_px: 1
up_slam_chi2_multipler: 1
up_aruco_sigma_px: 1
up_aruco_chi2_multipler: 1
use_mask: false
relative_config_imu: "kalibr_imu_chain.yaml"
relative_config_imucam: "kalibr_imucam_chain.yaml"
"""
OV_IMU = """%%YAML:1.0

imu0:
  T_i_b:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
  accelerometer_noise_density: %(accelerometer_noise_density).12g
  accelerometer_random_walk: %(accelerometer_random_walk).12g
  gyroscope_noise_density: %(gyroscope_noise_density).12g
  gyroscope_random_walk: %(gyroscope_random_walk).12g
  rostopic: %(topic)s
  time_offset: 0.0
  update_rate: %(rate).1f
  model: "kalibr"
  Tw:
    - [ 1.0, 0.0, 0.0 ]
    - [ 0.0, 1.0, 0.0 ]
    - [ 0.0, 0.0, 1.0 ]
  R_IMUtoGYRO:
    - [ 1.0, 0.0, 0.0 ]
    - [ 0.0, 1.0, 0.0 ]
    - [ 0.0, 0.0, 1.0 ]
  Ta:
    - [ 1.0, 0.0, 0.0 ]
    - [ 0.0, 1.0, 0.0 ]
    - [ 0.0, 0.0, 1.0 ]
  R_IMUtoACC:
    - [ 1.0, 0.0, 0.0 ]
    - [ 0.0, 1.0, 0.0 ]
    - [ 0.0, 0.0, 1.0 ]
  Tg:
    - [ 0.0, 0.0, 0.0 ]
    - [ 0.0, 0.0, 0.0 ]
    - [ 0.0, 0.0, 0.0 ]
"""


def imu_noise(path):
    if not path:
        raise Refused("openvins needs --imu-noise: the D455 IMU's Allan variance (%s); "
                      "record the IMU standing still for 2-3 h first" % ", ".join(NOISE_KEYS))
    v = yaml.safe_load(open(path)) or {}
    missing = [k for k in NOISE_KEYS if not isinstance(v.get(k), (int, float))]
    if missing:
        raise Refused(f"{path}: no {', '.join(missing)}")
    return {k: float(v[k]) for k in NOISE_KEYS}


def openvins(paths, edges, machine, a):
    c = camera(machine, "openvins")
    noise = imu_noise(a.imu_noise)
    li, lf = frame_of(paths, c["left_info"])
    ri, rf = frame_of(paths, c["right_info"])
    _, imu = frame_of(paths, c["imu"])
    _, imu_hz = topic_rate(paths, c["imu"])
    if not imu_hz:
        raise Refused(f"{c['imu']}: no rate in the bag")
    imucam = ("%YAML:1.0\n\ncam0:\n" + kalibr_cam(static(edges, lf, imu), li, c["left"], [1])
              + "cam1:\n" + kalibr_cam(static(edges, rf, imu), ri, c["right"], [0]))
    return {
        "files": {"estimator_config.yaml": OV_ESTIMATOR % {"track_hz": c["hz"] + 1.0},
                  "kalibr_imu_chain.yaml": OV_IMU % dict(noise, topic=c["imu"], rate=imu_hz),
                  "kalibr_imucam_chain.yaml": imucam},
        "launch": ("ros2 launch ov_msckf subscribe.launch.py config_path:=%s/estimator_config.yaml "
                   "use_stereo:=true max_cameras:=2 rviz_enable:=false" % CFG),
        "play": [c["left"], c["right"], c["imu"]],
        "in": {c["left"]: IMG, c["right"]: IMG, c["imu"]: IMU}, "key": c["left"],
        "out": "/ov_msckf/poseimu", "out_type": "geometry_msgs/msg/PoseWithCovarianceStamped",
        "tum_frame": imu, "proc": r"run_subscribe_msckf",
        # the state's time is the image's plus the camera-IMU offset it calibrates online
        "stamp_tol_s": 0.5 / c["hz"], "nominal_hz": c["hz"],
        "settings": {"use_stereo, max_cameras": [[True, 2], [True, 1], "both infrared cameras (rs_d455 is colour alone)"],
                     "imu noise": [a.imu_noise, "rs_d455's inflated values", "this D455's Allan variance"],
                     "track_frequency": [c["hz"] + 1.0, 31.0, "the camera's rate + 1"]}}


def depth_in_left(paths, edges, c):
    """The left infrared frame, refusing a depth that is not in it."""
    li, lf = frame_of(paths, c["left_info"])
    di, df = frame_of(paths, c["depth_info"])
    if (li.width, li.height) != (di.width, di.height) or not np.allclose(li.k, di.k, atol=1e-3):
        raise Refused(f"the depth ({di.width}x{di.height}, K {list(di.k)}) is not in the left infrared "
                      f"camera ({li.width}x{li.height}, K {list(li.k)}): it would need registering")
    T = static(edges, lf, df)
    if np.abs(T[:3, 3]).max() > 1e-3 or Rotation.from_matrix(T[:3, :3]).magnitude() > 1e-3:
        raise Refused(f"{df} is not {lf}: {T[:3, 3]} m apart")
    return lf


def rtabmap_rgbd(paths, edges, machine, a):
    c = camera(machine, "rtabmap_rgbd")
    lf = depth_in_left(paths, edges, c)
    return {
        "files": {},
        "launch": ("ros2 run rtabmap_odom rgbd_odometry --ros-args -p frame_id:=%s -p publish_tf:=false "
                   "-p use_sim_time:=true -p approx_sync:=true -p approx_sync_max_interval:=0.005 "
                   "-p qos:=1 -r rgb/image:=%s -r depth/image:=%s -r rgb/camera_info:=%s"
                   % (lf, c["left"], c["depth"], c["left_info"])),
        "play": [c["left"], c["left_info"], c["depth"]],
        "in": {c["left"]: IMG, c["depth"]: IMG}, "key": c["left"],
        "out": "/odom", "out_type": ODOM, "tum_frame": lf,
        "proc": r"rgbd_odometry", "stamp_tol_s": 0.0, "nominal_hz": c["hz"],
        "settings": {"image": [c["left"], "colour", "the left infrared image: the depth is computed in it, "
                                                     "so no registration (the emitter was off: weak depth on bare walls)"],
                     "qos": [1, 0, "reliable: the default best-effort subscription loses frames below the node"],
                     "loop closure": ["none", None, "rgbd_odometry alone, no rtabmap node"]}}


RTAB_LAUNCH = '''# made by make_config.py: RTAB-Map on the bag's clock, %(mode)s
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

RTABMAP = %(rtabmap)r
ODOM = %(odom)r
IMAGES = %(images)r            # RTAB-Map's inputs
ODOM_IMAGES = %(odom_images)r  # rgbd_odometry's
REGISTER = %(register)r        # the depth reprojected into the colour camera (depth_image_proc), or none
RELAY = %(relay)r


def generate_launch_description():
    nodes = []
    if REGISTER:
        nodes.append(ComposableNodeContainer(
            name="register_container", namespace="", package="rclcpp_components", executable="component_container",
            composable_node_descriptions=[ComposableNode(
                package="depth_image_proc", plugin="depth_image_proc::RegisterNode", name="register",
                remappings=REGISTER, parameters=[{"use_sim_time": True}])],
            output="screen"))
    if RELAY:
        nodes.append(ExecuteProcess(cmd=["python3", "/work/config/gt_relay.py"] + RELAY, output="screen"))
    else:
        nodes.append(Node(package="rtabmap_odom", executable="rgbd_odometry", name="rgbd_odometry",
                          parameters=[ODOM], remappings=ODOM_IMAGES, output="screen"))
    nodes.append(Node(package="rtabmap_slam", executable="rtabmap", name="rtabmap", namespace="rtabmap",
                      parameters=[RTABMAP], remappings=IMAGES, arguments=%(args)r, output="screen"))
    return LaunchDescription(nodes)
'''
GT_RELAY = '''# made by make_config.py: a pose topic in map as the odometry RTAB-Map maps on
#   python3 gt_relay.py <PoseStamped topic> <its frame> <odom frame name>
# Publishes the pose as TF <odom frame> -> <its frame> (and on /gt_odom), on the
# message stamps: RTAB-Map looks the pose up at each image's stamp.
import sys

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_ros import TransformBroadcaster


def main():
    topic, frame, odom = sys.argv[1:4]
    rclpy.init()
    n = Node("gt_relay", parameter_overrides=[Parameter("use_sim_time", value=True)])
    tf, pub = TransformBroadcaster(n), n.create_publisher(Odometry, "/gt_odom", 100)

    def cb(m):
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = m.header.stamp, odom, frame
        p, q = m.pose.position, m.pose.orientation
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = p.x, p.y, p.z
        t.transform.rotation = q
        tf.sendTransform(t)
        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = m.header.stamp, odom, frame
        o.pose.pose = m.pose
        pub.publish(o)
    n.create_subscription(PoseStamped, topic, cb, 100)
    rclpy.spin(n)


if __name__ == "__main__":
    main()
'''
RTAB_MAP_FRAME = "map_rtab"


REGISTERED = "/depth_registered/image_rect"


def rtabmap_common(paths, edges, machine, a, method):
    """RTAB-Map's inputs: the colour image with the depth reprojected into it
    (--rtab-image color, the default: the infrared images change with the
    D455's projector, the colour one does not), or the left infrared image
    with its own depth. rgbd_odometry, in rtabmap_loc, keeps the infrared
    image + depth either way."""
    c = camera(machine, method)
    lf = depth_in_left(paths, edges, c)
    base = a.gt_frame or "%s_base_link" % machine
    static(edges, base, lf)                               # the robot's base reaches its camera in /tf_static
    infra = [("rgb/image", c["left"]), ("depth/image", c["depth"]), ("rgb/camera_info", c["left_info"])]
    common = {"frame_id": base, "map_frame_id": RTAB_MAP_FRAME, "subscribe_rgb": True, "subscribe_depth": True,
              "approx_sync": True, "approx_sync_max_interval": 0.005, "qos": 1, "use_sim_time": True,
              "wait_for_transform": 0.3, "database_path": "/work/rtabmap.db",
              "Rtabmap/DetectionRate": "2", "RGBD/LinearUpdate": "0.1", "RGBD/AngularUpdate": "0.1"}
    v = {"images": infra, "odom_images": infra, "register": [], "key": c["left"], "hz": c["hz"],
         "play": [c["left"], c["left_info"], c["depth"]], "image": c["left"]}
    if (a.rtab_image or "color") == "color":
        _, cf = frame_of(paths, c["color_info"])
        _, df = frame_of(paths, c["depth_info"])
        static(edges, df, cf)                             # the depth can be put in the colour camera
        v.update(images=[("rgb/image", c["color"]), ("depth/image", REGISTERED), ("rgb/camera_info", c["color_info"])],
                 register=[("depth/image_rect", c["depth"]), ("depth/camera_info", c["depth_info"]),
                           ("rgb/camera_info", c["color_info"]), ("depth_registered/image_rect", REGISTERED),
                           ("depth_registered/camera_info", "/depth_registered/camera_info")],
                 key=c["color"], hz=c["color_hz"], image=c["color"],
                 play=[c["color"], c["color_info"], c["depth"], c["depth_info"]])
        common["approx_sync_max_interval"] = 0.02         # the colour (15 Hz) and depth (30 Hz) are not exposed together
    return c, lf, base, v, common


def rtabmap_map(paths, edges, machine, a):
    c, lf, base, v, common = rtabmap_common(paths, edges, machine, a, "rtabmap_map")
    gt = a.gt_topic or "/%s/global_pose" % machine
    frame_of(paths, gt)                                   # refuses a bag without it
    rt = dict(common, **{"odom_frame_id": "twin_map", "Mem/IncrementalMemory": "true",
                         "Mem/InitWMWithAllNodes": "false"})
    return {
        "files": {"gt_relay.py": GT_RELAY,
                  "rtabmap.launch.py": RTAB_LAUNCH % {"mode": "mapping on known poses", "rtabmap": rt, "odom": {},
                                                      "images": v["images"], "odom_images": [],
                                                      "register": v["register"], "relay": [gt, base, "twin_map"],
                                                      "args": ["-d"]}},
        "launch": "ros2 launch %s/rtabmap.launch.py" % CFG,
        "play": v["play"] + [gt, "/tf_static"],
        "in": {v["key"]: IMG, c["depth"]: IMG}, "key": v["key"],
        "out": "/gt_odom", "out_type": ODOM, "tum_frame": base, "world": "map",
        "proc": r"rtabmap|gt_relay|component_container", "stamp_tol_s": 0.05, "nominal_hz": v["hz"], "stop_s": 180,
        "settings": {"odometry": [gt, "visual", "the robot's pose in the twin (stage 10): every map node sits at "
                                               "its twin pose, so the map is in the twin's frame"],
                     "frame_id": [base, "base_link", "the frame %s is in" % gt],
                     "image": [v["image"], "colour", "colour, the depth reprojected into it (depth_image_proc): "
                               "unchanged by the projector" if v["register"] else
                               "the left infrared image, the depth's own camera (changes with the projector)"],
                     "Rtabmap/DetectionRate": ["2", "1", "a node every 0.5 s at most"],
                     "database": ["/work/rtabmap.db", "~/.ros/rtabmap.db", "rtabmap_loc's --db"]}}


def rtabmap_loc(paths, edges, machine, a):
    c, lf, base, v, common = rtabmap_common(paths, edges, machine, a, "rtabmap_loc")
    if not a.db or not os.path.exists(a.db):
        raise Refused(f"rtabmap_loc: no database {a.db}: --db <the rtabmap_map run>/rtabmap.db")
    a.copy = {"rtabmap.db": os.path.abspath(a.db)}
    rt = dict(common, **{"odom_frame_id": "odom_vis", "Mem/IncrementalMemory": "false",
                         "Mem/InitWMWithAllNodes": "true", "RGBD/MaxOdomCacheSize": "0"})
    od = {"frame_id": base, "odom_frame_id": "odom_vis", "publish_tf": True, "approx_sync": True,
          "approx_sync_max_interval": 0.005, "qos": 1, "use_sim_time": True, "wait_for_transform": 0.3}
    return {
        "files": {"rtabmap.launch.py": RTAB_LAUNCH % {"mode": "localization in the rtabmap_map database",
                                                      "rtabmap": rt, "odom": od, "images": v["images"],
                                                      "odom_images": v["odom_images"], "register": v["register"],
                                                      "relay": [], "args": []}},
        "launch": "ros2 launch %s/rtabmap.launch.py" % CFG,
        "play": sorted(set(v["play"]) | {c["left"], c["left_info"], c["depth"]}) + ["/tf_static"],
        "in": {v["key"]: IMG, c["depth"]: IMG}, "key": v["key"],
        "out": "/rtabmap/localization_pose", "out_type": "geometry_msgs/msg/PoseWithCovarianceStamped",
        "tum_frame": base, "world": "map",
        "proc": r"rtabmap|rgbd_odometry|component_container", "stamp_tol_s": 0.05, "nominal_hz": v["hz"],
        "stop_s": 60,
        "settings": {"database": [os.path.abspath(a.db), None, "rtabmap_map's, on the mapping pass: in the twin"],
                     "odometry": ["rgbd_odometry", None, "RTAB-Map's own on the infrared image + depth, between "
                                  "localizations"],
                     "image": [v["image"], None, "what RTAB-Map recognises places in (rtabmap_map's must match)"],
                     "output": ["/rtabmap/localization_pose", None, "a pose in the map (= the twin) at each "
                                "recognised place: scored as it is, its coverage is how often that happens"],
                     "Mem/IncrementalMemory": ["false", "true", "localization: the map is not extended"],
                     "RGBD/MaxOdomCacheSize": ["0", "10", "a good localization is taken at once, not held for "
                                               "a second one to confirm it"]}}


F2L_LAUNCH = '''# made by make_config.py: 2Fast-2Lamaa's two nodes on the bag's clock
from launch import LaunchDescription
from launch.actions import Shutdown
from launch_ros.actions import Node

ODOM = %(odom)r
MAP = %(map)r


def generate_launch_description():
    return LaunchDescription([
        Node(package="ffastllamaa", executable="lidar_scan_odometry", name="lidar_scan_odometry",
             remappings=[("/imu/acc", %(imu)r), ("/imu/gyr", %(imu)r), ("/lidar_raw_points", %(points)r)],
             parameters=[ODOM], output="screen"),
        Node(package="ffastllamaa", executable="gp_map", name="gp_map",
             remappings=[("/points_input", "/lidar_scan_undistorted"), ("/pose_input", "/undistortion_pose")],
             parameters=[MAP], output="screen", on_exit=Shutdown()),
    ])
'''
F2L_VOXEL = 0.15                        # the Newer College OS1 launches' (an Ouster, handheld)


def write_ply(path, P):
    """P (Nx3) as a binary PLY of double x y z, as gp_map writes and loads its maps."""
    P = np.ascontiguousarray(P, dtype="<f8")
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty double x\n"
                 "property double y\nproperty double z\nend_header\n" % len(P)).encode())
        f.write(P.tobytes())


def voxel_mean(P, v):
    """One point per v-m voxel, the mean of the points in it."""
    k = np.floor(P / v).astype(np.int64)
    _, inv, n = np.unique(k, axis=0, return_inverse=True, return_counts=True)
    out = np.zeros((len(n), 3))
    np.add.at(out, inv.reshape(-1), P)
    return out / n[:, None]


def stage04_map(ref):
    """The map stage 04 registered the robot to (its summary next to the reference TUM), or None."""
    d, name = os.path.dirname(ref), os.path.basename(ref)[len("traj_"):-len(".tum")]
    s = os.path.join(d, "summary_%s.json" % name)
    if not os.path.exists(s):
        return None

    def find(o):
        if isinstance(o, dict):
            if isinstance(o.get("ref_map"), str):
                return o["ref_map"]
            for v in o.values():
                r = find(v)
                if r:
                    return r
        return None
    return find(json.load(open(s)))


def fast2lamaa(paths, edges, machine, a, localize=False):
    t = LIDAR[machine]
    _, lidar = frame_of(paths, t["points"])
    _, imu = frame_of(paths, t["imu"])
    T_li, src = lidar_imu(edges, machine, lidar, imu)
    T_il = np.linalg.inv(T_li)                            # the LiDAR in the IMU frame
    rv = Rotation.from_matrix(T_il[:3, :3]).as_rotvec()
    imu_par = {"acc_in_m_per_s2": True, "invert_imu": False, "use_sim_time": True}
    odom = dict(imu_par, **{
        "min_range": MIN_RANGE, "max_range": 100.0, "feature_voxel_size": F2L_VOXEL,
        "max_associations_per_type": 1000, "planar_only": True, "mode": "imu", "g": 9.81,
        "calib_px": float(T_il[0, 3]), "calib_py": float(T_il[1, 3]), "calib_pz": float(T_il[2, 3]),
        "calib_rx": float(rv[0]), "calib_ry": float(rv[1]), "calib_rz": float(rv[2]),
        "unsorted_pc": True, "point_cloud_scale": 1.0, "dense_pc_output": False})
    gmap = dict(imu_par, **{
        "voxel_size": F2L_VOXEL, "min_range": MIN_RANGE, "map_path": "/work/map/",
        "submap_length": -200.0, "using_submaps": False})
    settings = {"invert_imu": [False, True, "the Ouster's IMU reads as is (the default flips it)"],
                "calib_*": [src, None, "the LiDAR in the IMU frame"],
                "min_range": [MIN_RANGE, 1.0, "the harness's, every LiDAR method"],
                "voxel_size": [F2L_VOXEL, 0.3, "the Newer College OS1 launches' (Ouster, handheld)"],
                "unsorted_pc": [True, False, "the Ouster's cloud is in column order, not time order"],
                "use_imu_estimator": [False, False, "the pose at the scan is what is scored"],
                "rviz": ["off", "on", "odometry only"]}
    files = {}
    if not localize:
        gmap.update({"key_framing": False, "max_num_pts_for_registration": 2000,
                     "free_space_carving_radius": -50.0})
        settings["mode"] = ["odometry + mapping, one global map", None, "the paper's 'global map' variant"]
    else:
        ref = a.ref or default_ref(a.bag, machine)
        if not ref or not os.path.exists(ref):
            raise Refused(f"fast2lamaa_loc: no reference trajectory {ref} for the initial pose (give --ref)")
        mp = a.map or stage04_map(ref)
        if not mp or not os.path.exists(mp):
            raise Refused(f"fast2lamaa_loc: no map {mp}: give --map, the anchored map stage 04 "
                          f"registered {machine} to (map_final_<tag>_anchored.pcd of the reference pass)")
        P = mc.read_cloud(mp)
        P = voxel_mean(P[np.isfinite(P).all(1)], F2L_VOXEL / 3.0)
        a.map_points = P
        G = mc.PoseInterp.from_tum(ref, max_gap_s=1.0)
        # the play starts where the reference does, so the seed is the pose of the first scan played
        start = max(a.start, (int(G.ts[0]) - a.bag_start_ns) * 1e-9)
        t_init = a.bag_start_ns + int(round(start * 1e9))
        T_ml = G.at(t_init)
        if T_ml is None:
            raise Refused(f"{ref}: no pose at {t_init * 1e-9:.3f} s")
        T_mi = T_ml @ T_li                                 # the IMU in map at the first scan
        ri = Rotation.from_matrix(T_mi[:3, :3]).as_rotvec()
        gmap.update({"localization_only": True, "can_trust_init": not a.coarse_init,
                     "init_pose_x": float(T_mi[0, 3]), "init_pose_y": float(T_mi[1, 3]),
                     "init_pose_z": float(T_mi[2, 3]), "init_pose_rx": float(ri[0]),
                     "init_pose_ry": float(ri[1]), "init_pose_rz": float(ri[2]),
                     "key_framing": True, "key_framing_dist_thr": 1.5,
                     "key_framing_rot_thr": float(np.radians(15.0)), "key_framing_time_thr": 0.05,
                     "max_num_pts_for_registration": 1500, "free_space_carving_radius": -20.0})
        files["map/map.info"] = "cell_size %g\n" % F2L_VOXEL
        settings.update({
            "mode": ["localization only, in the reference pass's anchored map", None,
                     "%s: %d points after a %.2f m voxel mean" % (mp, len(P), F2L_VOXEL / 3.0)],
            "init_pose": ["the reference's pose at the first scan (%.3f s)" % (t_init * 1e-9), "origin",
                          "%s; live, the board start gives it (cm, 0.4-0.9 deg heading error there); the play "
                          "starts at %.1f s, where the reference does" % (os.path.basename(ref), start)],
            "can_trust_init": [not a.coarse_init, False,
                               "the seed is good: the first scan gets the normal registration (the coarse-to-fine "
                               "one, losses 10/5/2 m, slid site_2 mobile_2 17.6 m off a right seed)"
                               if not a.coarse_init else "--coarse-init: the first scan coarse to fine"],
            "key_framing": [True, False, "the OS1 localization launch's (0.05 s: every scan at 10 Hz)"]})
    files["fast2lamaa.launch.py"] = F2L_LAUNCH % {"odom": odom, "map": gmap, "imu": t["imu"],
                                                 "points": t["points"]}
    return {
        "files": files,
        "launch": "mkdir -p /work/map && ros2 launch %s/fast2lamaa.launch.py" % CFG,
        "play": [t["points"], t["imu"]], "in": {t["points"]: PC2, t["imu"]: IMU}, "key": t["points"],
        "out": "/scan_to_map_pose", "out_type": TFS, "tum_frame": imu, "relay": relay(t["points"]),
        "world": "map" if localize else "odom", "start_s": start if localize else None,
        # the pose is stamped at the scan's reference time, not its header's: matched within a period
        "proc": r"lidar_scan_odometry|gp_map", "stamp_tol_s": 1.0 / LIDAR_HZ, "nominal_hz": LIDAR_HZ,
        "settings": settings}


def fast2lamaa_loc(paths, edges, machine, a):
    return fast2lamaa(paths, edges, machine, a, localize=True)


def default_ref(bag, machine):
    sys.path.insert(0, os.path.join(ROOT, "datasets"))
    import swarm_slam_lidar as SL
    return SL.default_gt(bag, machine)


METHODS = {"kiss_icp": kiss_icp, "rko_lio": rko_lio, "fast_lio": fast_lio,
           "fast2lamaa": fast2lamaa, "fast2lamaa_loc": fast2lamaa_loc,
           "rtabmap_map": rtabmap_map, "rtabmap_loc": rtabmap_loc,
           "openvins": openvins, "rtabmap_rgbd": rtabmap_rgbd}


def make(method, bag, machine, run, start=0.0, duration=0.0, imu_noise=None, min_rate_frac=0.9,
         map_path=None, ref=None, coarse_init=False, gt_topic=None, gt_frame=None, db=None, rtab_image="color"):
    if method not in METHODS:
        raise Refused(f"{method}: one of {sorted(METHODS)}")
    if machine not in LIDAR:
        raise Refused(f"{machine}: one of {sorted(LIDAR)}")
    bag = os.path.abspath(os.path.expanduser(bag))
    t0, dur = bag_metadata(bag)
    paths = mc._paths(bag)
    edges = mc.build_tf_static(paths)
    a = argparse.Namespace(imu_noise=imu_noise, map=map_path, ref=ref, bag=bag, bag_start_ns=t0, start=start,
                           map_points=None, coarse_init=coarse_init, gt_topic=gt_topic, gt_frame=gt_frame,
                           db=db, copy={}, rtab_image=rtab_image)
    m = METHODS[method](paths, edges, machine, a)
    rate = gate(paths, m["key"], m["nominal_hz"], min_rate_frac)
    if m.get("start_s") is not None:                     # the method moved the start (fast2lamaa_loc)
        start = m["start_s"]
    if os.path.exists(os.path.join(run, "run.json")):
        raise Refused(f"{run} holds a run already: pick another folder")
    os.makedirs(os.path.join(run, "config"), exist_ok=True)
    for name, text in m["files"].items():
        path = os.path.join(run, name) if name.startswith("map/") else os.path.join(run, "config", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(text)
    if a.map_points is not None:                         # fast2lamaa_loc: the map gp_map loads
        write_ply(os.path.join(run, "map", "map.ply"), a.map_points)
    import shutil
    for dst, src in a.copy.items():                      # rtabmap_loc: the database, a copy per run
        shutil.copyfile(src, os.path.join(run, dst))
    rj = {"method": method, "machine": machine, "bag": bag,
          "bag_start_ns": t0, "bag_duration_s": dur * 1e-9,
          "start_s": start, "duration_s": duration,
          "play_topics": m["play"], "in_topics": m["in"], "key_topic": m["key"],
          "out_topic": m["out"], "out_type": m["out_type"], "tum_frame": m["tum_frame"],
          "proc_regex": m["proc"], "stamp_tol_s": m["stamp_tol_s"], "launch": m["launch"],
          "relay": m.get("relay"), "world": m.get("world", "odom"),
          "ref": os.path.abspath(ref) if ref else None, "stop_s": m.get("stop_s"),
          "key_rate": rate, "settings": m["settings"], "config_files": sorted(m["files"])}
    json.dump(rj, open(os.path.join(run, "run.json"), "w"), indent=1)
    return rj


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=sorted(METHODS))
    ap.add_argument("bag", help="the rosbag2 folder (the finalized merged pass bag)")
    ap.add_argument("machine", choices=sorted(LIDAR))
    ap.add_argument("run", help="the run folder to create")
    ap.add_argument("--start", type=float, default=0.0, help="s into the bag")
    ap.add_argument("--duration", type=float, default=0.0, help="s (0: to the end)")
    ap.add_argument("--imu-noise", default=None, help="openvins: the IMU's Allan variance (YAML)")
    ap.add_argument("--min-rate-frac", type=float, default=0.9)
    ap.add_argument("--map", default=None, help="fast2lamaa_loc: the anchored map to localize in (.pcd/.ply; "
                    "default: the one stage 04's summary names)")
    ap.add_argument("--ref", default=None, help="fast2lamaa_loc: the T_map_lidar TUM whose first pose starts it "
                    "(default: stage 04's for the robot)")
    ap.add_argument("--gt-topic", default=None, help="rtabmap_map: the robot's pose in map to map on "
                    "(default /<machine>/global_pose)")
    ap.add_argument("--gt-frame", default=None, help="rtabmap_*: the frame of that pose, RTAB-Map's frame_id "
                    "(default <machine>_base_link)")
    ap.add_argument("--db", default=None, help="rtabmap_loc: the rtabmap_map run's rtabmap.db")
    ap.add_argument("--rtab-image", default="color", choices=("color", "infra"),
                    help="rtabmap_*: the image RTAB-Map maps and recognises places in (both runs the same)")
    ap.add_argument("--coarse-init", action="store_true", help="fast2lamaa_loc: register the first scan coarse to "
                    "fine (for a rough seed) instead of trusting the seed")
    a = ap.parse_args()
    try:
        rj = make(a.method, a.bag, a.machine, a.run, a.start, a.duration, a.imu_noise, a.min_rate_frac,
                  a.map, a.ref, a.coarse_init, a.gt_topic, a.gt_frame, a.db, a.rtab_image)
    except (Refused, mc.ConvertError) as e:
        sys.exit(f"refused: {e}")
    print(json.dumps({k: rj[k] for k in ("method", "machine", "play_topics", "key_topic", "out_topic",
                                          "tum_frame", "key_rate")}, indent=1))


if __name__ == "__main__":
    main()

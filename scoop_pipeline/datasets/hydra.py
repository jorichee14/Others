#!/usr/bin/env python3
"""A multi-robot pass as input for Hydra (github.com/MIT-SPARK/Hydra, ROS 2,
hydra_ros), one robot per Hydra: made from the datasets/swarm_slam.py export
of the same pass, so both methods get the same frames, the same odometry and
the same start in map.

    python scoop_pipeline/datasets/hydra.py <swarm_slam export dir> [<out dir>]
        [--robots 0 1] [--poses odom|gt|kimera] [--kimera <euroc export>] [--no-mesh] [--configs-only]

Writes <out> (default: next to the export, datasets/hydra/<name>):

  bag_r<i>/       robot i alone, stamped as recorded (play it with --clock):
                    /camera/color/image_raw    bgr8, undistorted (as the export)
                    /camera/color/camera_info  its intrinsics, no distortion
                    /camera/depth/image_rect   16UC1 mm, in the colour camera
                    /tf                        r<i>/odom -> r<i>/base -> r<i>/camera
                  r<i>/camera is the colour camera's optical frame (Hydra's
                  sensor_frame); r<i>/base a body frame at the camera, x forward
                  and z up (REP 103; Hydra's robot_frame); r<i>/odom the
                  export's odometry frame, which starts in map at the session
                  anchor, so Hydra's map and odom frames start together
  hydra_r<i>.yaml the config of robot i: input as hydra_ros's zed2i.yaml with
                  this robot's id, frames and depth range; mapping and backend
                  as Hydra's kimera_multi_rgbd.yaml (3D freespace places, which
                  its loop closure needs, and rooms); the backend sending the
                  whole scene graph WITH its mesh at most once a second on
                  hydra/backend/dsg (--no-mesh: without the mesh)
  scoop_hydra.launch.yaml
                  semantic segmentation (semantic_inference, closed set) and
                  Hydra on the bag's topics, sim time; args robot_id, config
                  (a hydra_r<i>.yaml), log_path (where Hydra saves its scene
                  graph and trajectory when it stops; it stops by itself when
                  the bag ends, exit_after_clock), enable_lcd (Hydra's loop
                  closure detection, on by default: without it the optimized
                  trajectory is the odometry), dsg_topic (where the stream
                  goes; slam/run_hydra.sh gives robot i /r<i>/hydra/backend/dsg)
                  and visualize:=true
                  (hydra_visualizer + rviz on the scene-graph stream)
  gt_r<i>.tum     the colour camera in map (the export's), for the evaluation
  report.txt

--poses gt: the camera's pose in /tf is the ground truth (the export's
gt_r<i>.tum, /<robot>/global_pose) instead of the export's odometry, so
Hydra maps on the correct poses (a frame without one gets no /tf); the
output goes to datasets/hydra/<name>_gt by default.

--poses kimera --kimera <euroc export>: the camera's pose in /tf is
Kimera-VIO's odometry (slam/run_kimera.sh on datasets/euroc.py's export of
the same pass: out/traj_vio.csv, or out/stretch_<n>/ each), the IMU's pose
at its keyframes interpolated at every frame and moved to the colour camera
(their link from the two ground truths, which both come from global_pose);
a frame outside its runs gets no /tf. Kimera-VIO starts from the ground truth
in map, so its odometry frame is map. Its loop closures go to Hydra on
/hydra/external_loop_closures (pose_graph_tools_msgs/PoseGraph, keys the two
keyframe stamps, the query's pose in the match's base frame), half a second
after the query, each only when it agrees with Kimera-VIO's own odometry
between the two keyframes: translation within max(1.5 m, 10 % of the path
between) and rotation within 10 deg + 0.1 deg per metre of it. mobile_1 of
20260924: 48 of 62 closures put the end of the run on its start, 14 m away
(identical calibration boards), which Kimera-VIO's own outlier rejection kept
(they agree with each other). The report counts them, and how many the
ground truth says are right; the output goes to datasets/hydra/<name>_kimera.

--configs-only: rewrite the configs and the launch file of an existing
output, its bags untouched.

The semantic labels are not in the bag: Hydra's own segmentation node makes
them from the colour image while the bag plays, as hydra_ros's zed2i launch
does.
"""
import argparse
import logging
import os
import re
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import mcap_convert as mc                                           # noqa: E402
from scoop import bag as B, rosmsg                                  # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402

log = logging.getLogger("hydra")

# body (x forward, y left, z up) <- optical (x right, y down, z forward)
R_BODY_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
# depth range per robot (m): the paper's range caps (ZED 4.5 m, RealSense 5 m)
RANGE = {"mobile_1": (0.3, 4.5), "mobile_2": (0.3, 5.0)}
LC_TOPIC = "/hydra/external_loop_closures"
TOPICS = {"color/image_raw": "/camera/color/image_raw",
          "color/camera_info": "/camera/color/camera_info",
          "aligned_depth_to_color/image_raw": "/camera/depth/image_rect"}

# input and frames as hydra_ros/config/datasets/zed2i.yaml (MIT-SPARK/Hydra-ROS,
# BSD-2-Clause) with this robot's id, frames and range; the rest is Hydra's own
# RGB-D multi-robot config, hydra/config/datasets/kimera_multi_rgbd.yaml (BSD-2-Clause):
# 3D freespace places (GVD), which its loop closure detection searches, and rooms;
# and the backend's scene-graph stream
CONFIG = """\
map_frame: map
odom_frame: r%(i)d/odom
robot_frame: r%(i)d/base
sensor_frame: r%(i)d/camera
enable_zmq_interface: false
publish_backend_tf: true
robot_id: %(i)d
input:
  type: RosInput
  inputs:
    frontleft:
      receiver:
        type: ClosedSetImageReceiver
        queue_size: 30
      sensor:
        type: camera_info
        min_range: %(min_range)g
        max_range: %(max_range)g
        extrinsics:
          type: ros
map_window:
  type: spatial
  max_radius_m: 8.0
active_window:
  volumetric_map:
    voxels_per_side: 16
    voxel_size: 0.1
    truncation_distance: 0.3
  tsdf:
    interpolation_method:
      type: adaptive
      max_depth_difference_m: 0.5
    semantic_integrator:
      type: MLESemanticIntegrator
  mesh:
    min_weight: 0.5
frontend:
  pgmo:
    time_horizon: 15.0
    d_graph_resolution: 2.5
    mesh_resolution: 0.005
  objects:
    min_cluster_size: 20
    cluster_tolerance: 0.20
  surface_places:
    type: place_2d
    prefix: P
    pure_final_place_size: 1
    cluster_tolerance: .3
    min_cluster_size: 50
    max_cluster_size: 100000
    min_final_place_points: 10
    place_max_neighbor_z_diff: .5
    place_overlap_threshold: 0.0
  freespace_places:
    type: gvd
    filter_places: true
    min_places_component_size: 3
    filter_ground: false
    gvd:
      max_distance_m: 4.5
      min_distance_m: 0.2
      min_diff_m: 0.1
      voronoi_config:
        mode: L1_THEN_ANGLE
        min_distance_m: 0.30
        parent_l1_separation: 20
        parent_cos_angle_separation: 0.2
    graph:
      type: CompressionGraphExtractor
      compression_distance_m: 1.5
      min_node_distance_m: 0.4
      min_edge_distance_m: 0.25
      node_merge_distance_m: 0.7
      merge_policy: distance
    tsdf_interpolator:
      type: downsample
      ratio: 2
  use_frontiers: false
backend:
  type: BackendModule
  # the stream under test: the whole scene graph (and its mesh) at most once a second
  serialize_dsg_mesh: %(mesh)s
  min_dsg_separation_s: 1
  publish_mesh: false
  min_mesh_separation_s: 10
  add_places_to_deformation_graph: false
  optimize_on_lc: true
  max_external_loop_closure_time_difference: 1.0
  enable_node_merging: true
  update_functors:
    agents:
      type: UpdateAgentsFunctor
    objects:
      type: UpdateObjectsFunctor
    surface_places:
      type: Update2dPlacesFunctor
    places:
      type: UpdatePlacesFunctor
    frontiers:
      type: UpdateFrontiersFunctor
    rooms:
      type: UpdateRoomsFunctor
      room_finder:
        min_dilation_m: 0.5
        max_dilation_m: 1.2
        min_component_size: 10
        min_room_size: 10
        dilation_threshold_mode: PLATEAU
        min_lifetime_length_m: 0.1
        plateau_ratio: 0.15
        clustering_mode: NEIGHBORS
        dilation_diff_threshold_m: -1.0
    buildings:
      type: UpdateBuildingsFunctor
  pgmo:
    run_mode: FULL
    embed_trajectory_delta_t: 5.0
    num_interp_pts: 3
    interp_horizon: 10.0
    optimizer:
      type: KimeraRpgoOptimizer
      solver: LM
      gnc:
        inlier_probability: 0.9
        mu_step: 1.6
        max_iterations: 100
    add_initial_prior: true
    covariance:
      odom: 1.0e-02
      loop_close: 5.0e-02
      sg_loop_close: 1.0e-01
      prior: 1.0e-02
      mesh_mesh: 1.0e-02
      pose_mesh: 1.0e-02
      place_mesh: 1.0e-02
      place_edge: 10.0
      place_merge: 10.0
      object_merge: 10.0
"""

# hydra_ros/launch/datasets/zed2i.launch.yaml without the camera driver and the
# visualizer, on the bag's topics, in sim time
LAUNCH = """\
launch:
  - arg: {name: robot_id, default: '0'}
  - arg: {name: config, description: a hydra_r<i>.yaml of this export}
  - arg: {name: log_path, default: $(env HOME)/.hydra/scoop_r$(var robot_id)}
  - arg: {name: model_name, default: /root/hydra_ws/.models/ade20k-efficientvit_seg_l2.onnx}
  - arg: {name: model_config_path, default: $(find-pkg-share semantic_inference_ros)/config/models/ade20k-efficientvit_seg_l2.yaml}
  - arg: {name: labelspace_name, default: ade20k_mit}
  - arg: {name: visualize, default: 'false', description: hydra_visualizer + rviz on hydra/backend/dsg}
  - arg: {name: exit_after_clock, default: 'true', description: stop and save when the bag clock stops}
  - arg: {name: enable_lcd, default: 'true', description: scene-graph loop closure detection}
  - arg: {name: dsg_topic, default: /hydra/backend/dsg, description: where the scene-graph stream goes}
  - arg: {name: lcd_config, default: $(find-pkg-share hydra)/config/lcd/default.yaml}

  - node:
      pkg: semantic_inference_ros
      exec: closed_set_node
      namespace: frontleft
      name: segmentation
      param:
        - {name: use_sim_time, value: true}
      remap:
        - {from: color/image_raw, to: /camera/color/image_raw}
        - {from: color/camera_info, to: /camera/color/camera_info}
        - {from: depth/image_rect, to: /camera/depth/image_rect}
      args: >
        --config-utilities-file $(var model_config_path)
        --config-utilities-file $(find-pkg-share semantic_inference_ros)/config/label_groupings/$(var labelspace_name).yaml@output/recolor
        --config-utilities-yaml {output: {recolor: {colormap_path: $(find-pkg-share semantic_inference_ros)/config/distinct_150_colors.csv}}}
        --config-utilities-yaml {segmenter: {model: {model_file: $(var model_name)}}}
        --config-utilities-yaml {worker: {max_queue_size: 1, image_separation_s: 0.0}}

  - set_remap: {from: hydra/input/frontleft/rgb/image_raw, to: /camera/color/image_raw}
  - set_remap: {from: hydra/input/frontleft/rgb/camera_info, to: /camera/color/camera_info}
  - set_remap: {from: hydra/input/frontleft/depth_registered/image_rect, to: /camera/depth/image_rect}
  - set_remap: {from: hydra/input/frontleft/semantic/image_raw, to: /frontleft/semantic/image_raw}
  - set_remap: {from: /hydra/backend/dsg, to: $(var dsg_topic)}
  - node:
      pkg: hydra_ros
      exec: hydra_ros_node
      name: hydra
      param:
        - {name: use_sim_time, value: true}
      args: >-
        --config-utilities-file $(var config)
        --config-utilities-file $(find-pkg-share hydra)/config/label_spaces/$(var labelspace_name)_label_space.yaml
        --config-utilities-yaml {enable_lcd: $(var enable_lcd)}
        --config-utilities-file $(var lcd_config)
        --config-utilities-yaml {print_missing: false, show_config: true, verbosity: 2}
        --config-utilities-yaml {log_path: $(var log_path), exit_after_clock: $(var exit_after_clock)}
        --config-utilities-yaml {glog_level: 0, glog_verbosity: 0}

  - group:
      if: $(var visualize)
      children:
        - set_parameter: {name: use_sim_time, value: 'true'}
        - set_remap: {from: hydra_visualizer/dsg, to: $(var dsg_topic)}
        - include:
            file: $(find-pkg-share hydra_visualizer)/launch/streaming_visualizer.launch.yaml
            arg:
              - {name: visualizer_frame, value: map}
"""


def default_out(export):
    """.../datasets/swarm_slam/<name> -> .../datasets/hydra/<name>."""
    export = os.path.abspath(os.path.expanduser(export)).rstrip(os.sep)
    parent, name = os.path.split(export)
    return os.path.join(os.path.dirname(parent), "hydra", name)


def robots_of(export):
    """{i: machine} from the export's report.txt ("robots : r0 = mobile_1, ...")."""
    rep = os.path.join(export, "report.txt")
    if not os.path.exists(rep):
        return {}
    for ln in open(rep):
        if ln.startswith("robots"):
            return {int(i): m for i, m in re.findall(r"r(\d+) = (\S+?)(?:,|$)", ln.split(":", 1)[1].strip())}
    return {}


def body_T_camera():
    T = np.eye(4)
    T[:3, :3] = R_BODY_OPTICAL
    return T


def read_gt(path):
    """{t in us: T (4x4)} from a TUM file (t x y z qx qy qz qw, seconds)."""
    out = {}
    for ln in open(path):
        v = ln.split()
        if len(v) == 8 and not ln.startswith("#"):
            out[int(round(float(v[0]) * 1e6))] = mc.xyzq_to_T([float(x) for x in v[1:8]])
    return out


# loop closures Hydra gets from Kimera-VIO: the odometry between the two keyframes must agree,
# within what it drifts (mobile_1 20260924: its path 10 % short over the run)
GATE_M, GATE_FRAC, GATE_DEG, GATE_DEG_PER_M = 1.5, 0.10, 10.0, 0.1
# a closure the ground truth calls right (report only): relative pose within these
RIGHT_M, RIGHT_DEG = 0.5, 10.0
LC_DELAY_NS = 500_000_000            # sent half a second after the query keyframe


def _rows(path, int_cols=(0,)):
    """Rows of a Kimera-VIO csv (comment header) as floats, the int_cols (stamps,
    ns: a float64 has 53 bits, a stamp 61) exact as int."""
    out = []
    for ln in open(path):
        if ln.startswith("#") or not ln.strip():
            continue
        v = ln.strip().split(",")
        out.append([int(x) if k in int_cols else float(x) for k, x in enumerate(v)])
    return out


def _T(xyz, wxyz):
    return mc.xyzq_to_T(list(xyz) + [wxyz[1], wxyz[2], wxyz[3], wxyz[0]])


def _angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def kimera_runs(robot_dir):
    """Kimera-VIO's run folders of an export's robot: out/stretch_<n> each, or out."""
    out = os.path.join(robot_dir, "out")
    st = sorted((d for d in os.listdir(out) if d.startswith("stretch_")) if os.path.isdir(out) else [],
                key=lambda d: int(d.split("_")[1]))
    runs = [os.path.join(out, d) for d in st] or [out]
    return [r for r in runs if os.path.exists(os.path.join(r, "traj_vio.csv"))]


class KimeraOdometry:
    """Kimera-VIO's IMU in map (traj_vio.csv per run): at(t) interpolates within
    the run spanning t (keyframes up to max_gap_s apart: a robot standing still
    makes few), None outside every run."""

    def __init__(self, robot_dir, max_gap_s=5.0):
        self.runs = []
        for r in kimera_runs(robot_dir):
            rows = _rows(os.path.join(r, "traj_vio.csv"))
            if len(rows) < 2:
                continue
            ts = np.array([x[0] for x in rows], np.int64)
            pos = np.array([[x[1], x[2], x[3]] for x in rows])
            quat = np.array([[x[5], x[6], x[7], x[4]] for x in rows])       # xyzw
            self.runs.append((r, ts, mc.PoseInterp(ts, pos, quat, max_gap_s=max_gap_s)))
        if not self.runs:
            raise mc.ConvertError(f"no Kimera-VIO odometry (out/traj_vio.csv) under {robot_dir}")

    def at(self, t_ns):
        for _, ts, pi in self.runs:
            if ts[0] <= t_ns <= ts[-1]:
                return pi.at(int(t_ns))
        return None


def imu_T_colour(robot_dir, gt_tum):
    """The colour camera in the IMU frame: euroc.py's ground truth (the IMU in map)
    against the export's gt_r<i>.tum (the colour camera in map), both global_pose."""
    rows = _rows(os.path.join(robot_dir, "mav0", "state_groundtruth_estimate0", "data.csv"))
    imu = {int(round(r[0] / 1e3)): _T(r[1:4], r[4:8]) for r in rows}
    col = sorted(read_gt(gt_tum).items())
    Ts = [np.linalg.inv(imu[t]) @ T for t, T in col if t in imu][:200]
    if len(Ts) < 10:                     # other stamps (another camera's frames): interpolate the IMU
        ts = np.array([r[0] for r in rows], np.int64)
        pi = mc.PoseInterp(ts, np.array([r[1:4] for r in rows]), np.array([[r[5], r[6], r[7], r[4]] for r in rows]))
        Ts = [np.linalg.inv(Ti) @ T for t, T in col[::max(1, len(col) // 400)]
              for Ti in [pi.at(t * 1000)] if Ti is not None and pi.dt_to_nearest(t * 1000) < 0.05][:200]
    if not Ts:
        raise mc.ConvertError(f"{robot_dir}: its ground truth and {gt_tum} have no time in common")
    T = Ts[len(Ts) // 2].copy()
    spread = np.median([np.linalg.norm(x[:3, 3] - T[:3, 3]) for x in Ts])
    if spread > 0.02:
        raise mc.ConvertError(f"{robot_dir}: the IMU-to-camera link varies by {spread:.3f} m (median) between "
                              f"the ground truths")
    return T


def kimera_closures(robot_dir, T_ib, robot, gt_rows=None):
    """Kimera-VIO's loop closures of every run, each: stamps (match, query), the
    query's base pose in the match's (Hydra's to_T_from), whether the odometry
    agrees (passed), and whether the ground truth does (right, None without it)."""
    out = []
    for r in kimera_runs(robot_dir):
        f = os.path.join(r, "output_lcd_result.csv")
        if not os.path.exists(f):
            continue
        rows = _rows(os.path.join(r, "traj_vio.csv"))
        odo = {int(x[0]): _T(x[1:4], x[4:8]) for x in rows}
        ts = np.array(sorted(odo), np.int64)
        P = np.array([odo[t][:3, 3] for t in ts])
        for x in _rows(f, int_cols=(0, 1, 2, 3)):
            if len(x) < 13 or x[3] != 1:
                continue
            tq, tm = int(x[1]), int(x[2])
            lc = _T(x[6:9], x[9:13])                                  # imuMatch_T_imuQuery
            c = {"query_ns": tq, "match_ns": tm, "passed": False, "right": None}
            if tm in odo and tq in odo:
                rel = np.linalg.inv(odo[tm]) @ odo[tq]
                a, b = sorted((tm, tq))
                k = (ts >= a) & (ts <= b)
                path = float(np.sum(np.linalg.norm(np.diff(P[k], axis=0), axis=1))) if k.sum() > 1 else 0.0
                dt = float(np.linalg.norm(rel[:3, 3] - lc[:3, 3]))
                dr = _angle(rel[:3, :3].T @ lc[:3, :3])
                c.update(path_m=path, odo_diff_m=dt, odo_diff_deg=dr,
                         passed=dt <= max(GATE_M, GATE_FRAC * path) and dr <= GATE_DEG + GATE_DEG_PER_M * path)
            if gt_rows is not None:
                g = gt_rows
                def at(t):
                    i = int(np.clip(np.searchsorted(g[:, 0], t), 0, len(g) - 1))
                    p = [np.interp(t, g[:, 0], g[:, k]) for k in (1, 2, 3)]
                    return _T(p, g[i, 4:8]), abs(g[i, 0] - t) < 0.2e9
                Gm, okm = at(tm)
                Gq, okq = at(tq)
                if okm and okq:
                    rg = np.linalg.inv(Gm) @ Gq
                    c["right"] = bool(np.linalg.norm(rg[:3, 3] - lc[:3, 3]) < RIGHT_M
                                      and _angle(rg[:3, :3].T @ lc[:3, :3]) < RIGHT_DEG)
            e = np.linalg.inv(T_ib) @ lc @ T_ib                       # baseMatch_T_baseQuery
            c["edge"] = (tq, tm, robot, robot, e[:3, 3], mc.T_to_xyzq(e)[3:])
            out.append(c)
    return out


def convert(export, out=None, robots=None, mesh=True, configs_only=False, poses="odom", kimera=None):
    export = os.path.abspath(os.path.expanduser(export))
    src = os.path.join(export, "bag")
    if not os.path.isdir(src):
        raise mc.ConvertError(f"no {src}: run datasets/swarm_slam.py first")
    if poses not in ("odom", "gt", "kimera"):
        raise mc.ConvertError(f"poses {poses!r}: odom, gt or kimera")
    if poses == "kimera" and not kimera:
        raise mc.ConvertError("--poses kimera needs --kimera <datasets/euroc.py export with Kimera-VIO's out/>")
    suffix = {"gt": "_gt", "kimera": "_kimera"}.get(poses, "")
    out = os.path.abspath(os.path.expanduser(out or (default_out(export) + suffix)))
    reader = B.open_bag(src)
    schemas = B.topic_schemas(reader)
    have = sorted({int(m.group(1)) for t in schemas for m in [re.match(r"/r(\d+)/", t)] if m})
    robots = have if robots is None else [i for i in robots if i in have]
    if not robots:
        raise mc.ConvertError(f"{src} has no /r<i>/ topics for robots {robots} (have {have})")
    names = robots_of(export)
    os.makedirs(out, exist_ok=True)
    if configs_only:
        missing = [i for i in robots if not os.path.isdir(os.path.join(out, "bag_r%d" % i))]
        if missing:
            raise mc.ConvertError(f"--configs-only: no {out}/bag_r{missing[0]} (run without it first)")
        return write_configs(export, out, robots, names, mesh, None, poses)
    T_bc = body_T_camera()
    T_cb = np.linalg.inv(T_bc)
    kim, closures = {}, {}
    if poses == "kimera":
        for i in robots:
            m = names.get(i)
            rd = os.path.join(os.path.abspath(os.path.expanduser(kimera)), m or "")
            g = os.path.join(export, "gt_r%d.tum" % i)
            if not m or not os.path.isdir(rd):
                raise mc.ConvertError(f"r{i} ({m}): no {rd} in the Kimera-VIO export")
            T_ib = imu_T_colour(rd, g) @ T_cb                     # the IMU's base frame
            gtf = os.path.join(rd, "mav0", "state_groundtruth_estimate0", "data.csv")
            gt_rows = np.array(_rows(gtf), dtype=float) if os.path.exists(gtf) else None
            kim[i] = (KimeraOdometry(rd), T_ib)
            closures[i] = kimera_closures(rd, T_ib, i, gt_rows)
    for i in robots:
        if os.path.exists(os.path.join(out, "bag_r%d" % i)):
            raise mc.ConvertError(f"{out}/bag_r{i} exists: remove it first (or --configs-only)")
    gts = {}
    if poses == "gt":
        for i in robots:
            g = os.path.join(export, "gt_r%d.tum" % i)
            if not os.path.exists(g):
                raise mc.ConvertError(f"--poses gt: no {g}")
            gts[i] = read_gt(g)
    ts = rosmsg.typestore()
    q_bc = mc.T_to_xyzq(T_bc)
    writers = {i: BagWriter(os.path.join(out, "bag_r%d" % i)) for i in robots}
    route = {}
    for i in robots:
        for k, v in TOPICS.items():
            route["/r%d/%s" % (i, k)] = (i, v)
        route["/r%d/odom" % i] = (i, "/tf")
    counts = {i: {v: 0 for v in list(TOPICS.values()) + ["/tf", "no pose", LC_TOPIC]} for i in robots}
    # the loop closures that passed, each at its query + LC_DELAY_NS, in time with the rest
    pending = {i: sorted((c["query_ns"] + LC_DELAY_NS, c["edge"]) for c in closures.get(i, []) if c["passed"])
               for i in robots}

    def flush(i, upto):
        while pending[i] and pending[i][0][0] <= upto:
            t, e = pending[i].pop(0)
            writers[i].write(LC_TOPIC, rosmsg.pose_graph(t, "world", [e]), t)
            counts[i][LC_TOPIC] += 1

    for _, _, ch, msg in B.iter_mcap_records(reader, topics=list(route)):
        i, topic = route[ch.topic]
        w = writers[i]
        flush(i, msg.log_time)
        if topic != "/tf":
            w.write_raw(topic, schemas[ch.topic], msg.data, msg.log_time, msg.publish_time)
        else:
            o = ts.deserialize_cdr(msg.data, "nav_msgs/msg/Odometry")
            t = mc.stamp_ns(o)
            if poses == "gt":
                T_oc = gts[i].get(int(round(t / 1e3)))
                if T_oc is None:
                    counts[i]["no pose"] += 1
                    continue
            elif poses == "kimera":
                T_mi = kim[i][0].at(t)
                if T_mi is None:
                    counts[i]["no pose"] += 1
                    continue
                T_oc = T_mi @ kim[i][1] @ T_bc                         # the camera: base -> optical
            else:
                p = o.pose.pose
                T_oc = mc.xyzq_to_T([p.position.x, p.position.y, p.position.z,
                                     p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
            q_ob = mc.T_to_xyzq(T_oc @ T_cb)
            w.write("/tf", rosmsg.tf_message(
                [("r%d/odom" % i, "r%d/base" % i, q_ob[:3], q_ob[3:]),
                 ("r%d/base" % i, "r%d/camera" % i, q_bc[:3], q_bc[3:])], t), msg.log_time)
        counts[i][topic] += 1
    for i in robots:
        flush(i, np.iinfo(np.int64).max)
    for w in writers.values():
        w.close()
    for i in robots:
        if not counts[i]["/camera/color/image_raw"] or not counts[i]["/tf"]:
            raise mc.ConvertError(f"r{i}: no images or no odometry in {src}")
    return write_configs(export, out, robots, names, mesh, counts, poses, closures)


def write_configs(export, out, robots, names, mesh, counts, poses="odom", closures=None):
    """hydra_r<i>.yaml, the launch file, ground truth and report.txt (counts:
    the bags' message counts, None when the bags were not written now)."""
    src = os.path.join(export, "bag")
    lines = ["source              : %s" % src, "scene-graph mesh    : %s" % ("on" if mesh else "off"),
             "poses               : %s" % {"gt": "ground truth (gt_r<i>.tum, /<robot>/global_pose)",
                                             "kimera": "Kimera-VIO's odometry (traj_vio.csv)"}.get(
                                                 poses, "the export's odometry")]
    for i in robots:
        m = names.get(i, "r%d" % i)
        lo, hi = RANGE.get(m, (0.3, 5.0))
        open(os.path.join(out, "hydra_r%d.yaml" % i), "w").write(
            CONFIG % {"i": i, "min_range": lo, "max_range": hi, "mesh": "true" if mesh else "false"})
        gt = os.path.join(export, "gt_r%d.tum" % i)
        if os.path.exists(gt):
            shutil.copy(gt, os.path.join(out, "gt_r%d.tum" % i))
        if counts is None:
            lines.append("r%d %-16s: configs rewritten (bag_r%d kept); depth %g-%g m" % (i, m, i, lo, hi))
        else:
            c = counts[i]
            lines.append("r%d %-16s: %d colour, %d depth, %d camera_info, %d tf%s; depth %g-%g m; "
                         "frames r%d/odom -> r%d/base -> r%d/camera"
                         % (i, m, c["/camera/color/image_raw"], c["/camera/depth/image_rect"],
                            c["/camera/color/camera_info"], c["/tf"],
                            " (%d frames without a pose)" % c["no pose"] if c["no pose"] else "", lo, hi, i, i, i))
        cl = (closures or {}).get(i)
        if cl is not None:
            ok = [c for c in cl if c["passed"]]
            no = [c for c in cl if not c["passed"]]
            right = lambda xs: sum(1 for c in xs if c["right"])
            known = lambda xs: sum(1 for c in xs if c["right"] is not None)
            lines.append("r%d loop closures   : %d from Kimera-VIO; %d agree with its odometry -> %s "
                         "(%d of %d right by the ground truth), %d left out (%d of %d right)"
                         % (i, len(cl), len(ok), LC_TOPIC, right(ok), known(ok), len(no), right(no), known(no)))
    open(os.path.join(out, "scoop_hydra.launch.yaml"), "w").write(LAUNCH)
    report = "\n".join(lines) + "\n"
    open(os.path.join(out, "report.txt"), "w").write(report)
    log.info("\n" + report)
    return {"out": out, "robots": robots, "counts": counts}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", help="a datasets/swarm_slam.py export (has bag/ and report.txt)")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--robots", nargs="+", type=int, default=None, help="robot indices (default: all)")
    ap.add_argument("--poses", choices=["odom", "gt", "kimera"], default="odom",
                    help="the camera pose in /tf: the export's odometry, the ground truth (global_pose), or "
                         "Kimera-VIO's (--kimera)")
    ap.add_argument("--kimera", default=None,
                    help="--poses kimera: the datasets/euroc.py export Kimera-VIO ran on (<robot>/out/)")
    ap.add_argument("--no-mesh", action="store_true", help="send the scene graph without its mesh")
    ap.add_argument("--configs-only", action="store_true",
                    help="rewrite the configs and launch file of an existing output, not its bags")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        convert(a.export, a.out, a.robots, mesh=not a.no_mesh, configs_only=a.configs_only, poses=a.poses,
                kimera=a.kimera)
    except mc.ConvertError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()

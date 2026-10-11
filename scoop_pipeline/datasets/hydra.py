#!/usr/bin/env python3
"""A multi-robot pass as input for Hydra (github.com/MIT-SPARK/Hydra, ROS 2,
hydra_ros), one robot per Hydra: made from the datasets/swarm_slam.py export
of the same pass, so both methods get the same frames, the same odometry and
the same start in map.

    python scoop_pipeline/datasets/hydra.py <swarm_slam export dir> [<out dir>]
        [--robots 0 1] [--no-mesh] [--configs-only]

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
  hydra_r<i>.yaml the hydra_ros config of robot i: hydra_ros's zed2i.yaml with
                  this robot's id, frames and depth range, and the backend
                  sending the whole scene graph WITH its mesh at most once a
                  second on hydra/backend/dsg (--no-mesh: without the mesh)
  scoop_hydra.launch.yaml
                  semantic segmentation (semantic_inference, closed set) and
                  Hydra on the bag's topics, sim time; args robot_id, config
                  (a hydra_r<i>.yaml), log_path (where Hydra saves its scene
                  graph and trajectory when it stops) and visualize:=true
                  (hydra_visualizer + rviz on the scene-graph stream)
  gt_r<i>.tum     the colour camera in map (the export's), for the evaluation
  report.txt

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
TOPICS = {"color/image_raw": "/camera/color/image_raw",
          "color/camera_info": "/camera/color/camera_info",
          "aligned_depth_to_color/image_raw": "/camera/depth/image_rect"}

# hydra_ros/config/datasets/zed2i.yaml (MIT-SPARK/Hydra-ROS, BSD-2-Clause) with
# this robot's id, frames and range, and the backend's scene-graph stream
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
    semantic_integrator:
      type: MLESemanticIntegrator
frontend:
  type: GraphBuilder
  graph_connector:
    layers:
      - {parent_layer: MESH_PLACES}
  pgmo:
    time_horizon: 15.0
    d_graph_resolution: 2.5
    mesh_resolution: 0.005
  objects:
    min_cluster_size: 40
    cluster_tolerance: 0.25
    bounding_box_type: AABB
  surface_places:
    type: place_2d
    prefix: P
    pure_final_place_size: 3
    enable_place_purity: false
    impure_final_place_size: 1
    cluster_tolerance: 0.3
    min_cluster_size: 50
    max_cluster_size: 100000
    min_final_place_points: 10
    place_max_neighbor_z_diff: 0.5
    place_overlap_threshold: 0.0
backend:
  type: BackendModule
  # the stream under test: the whole scene graph (and its mesh) at most once a second
  serialize_dsg_mesh: %(mesh)s
  min_dsg_separation_s: 1
  publish_mesh: false
  min_mesh_separation_s: 10
  add_places_to_deformation_graph: false
  optimize_on_lc: true
  update_functors:
    agents: {type: UpdateAgentsFunctor}
    objects: {type: UpdateObjectsFunctor}
    surface_places: {type: Update2dPlacesFunctor, min_size: 3}
  pgmo:
    run_mode: FULL
    embed_trajectory_delta_t: 5.0
    num_interp_pts: 3
    interp_horizon: 10.0
    add_initial_prior: true
    optimizer:
      type: KimeraRpgoOptimizer
      solver: LM
      gnc:
        inlier_probability: 0.9
        mu_step: 1.6
        max_iterations: 100
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
  - node:
      pkg: hydra_ros
      exec: hydra_ros_node
      name: hydra
      param:
        - {name: use_sim_time, value: true}
      args: >-
        --config-utilities-file $(var config)
        --config-utilities-file $(find-pkg-share hydra)/config/label_spaces/$(var labelspace_name)_label_space.yaml
        --config-utilities-yaml {print_missing: false, show_config: true, verbosity: 2}
        --config-utilities-yaml {log_path: $(var log_path)}
        --config-utilities-yaml {glog_level: 0, glog_verbosity: 0}

  - group:
      if: $(var visualize)
      children:
        - set_remap: {from: hydra_visualizer/dsg, to: /hydra/backend/dsg}
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


def convert(export, out=None, robots=None, mesh=True, configs_only=False):
    export = os.path.abspath(os.path.expanduser(export))
    src = os.path.join(export, "bag")
    if not os.path.isdir(src):
        raise mc.ConvertError(f"no {src}: run datasets/swarm_slam.py first")
    out = os.path.abspath(os.path.expanduser(out or default_out(export)))
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
        return write_configs(export, out, robots, names, mesh, None)
    for i in robots:
        if os.path.exists(os.path.join(out, "bag_r%d" % i)):
            raise mc.ConvertError(f"{out}/bag_r{i} exists: remove it first (or --configs-only)")
    ts = rosmsg.typestore()
    T_bc = body_T_camera()
    T_cb = np.linalg.inv(T_bc)
    q_bc = mc.T_to_xyzq(T_bc)
    writers = {i: BagWriter(os.path.join(out, "bag_r%d" % i)) for i in robots}
    route = {}
    for i in robots:
        for k, v in TOPICS.items():
            route["/r%d/%s" % (i, k)] = (i, v)
        route["/r%d/odom" % i] = (i, "/tf")
    counts = {i: {v: 0 for v in list(TOPICS.values()) + ["/tf"]} for i in robots}
    for _, _, ch, msg in B.iter_mcap_records(reader, topics=list(route)):
        i, topic = route[ch.topic]
        w = writers[i]
        if topic != "/tf":
            w.write_raw(topic, schemas[ch.topic], msg.data, msg.log_time, msg.publish_time)
        else:
            o = ts.deserialize_cdr(msg.data, "nav_msgs/msg/Odometry")
            p = o.pose.pose
            T_oc = mc.xyzq_to_T([p.position.x, p.position.y, p.position.z,
                                 p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
            q_ob = mc.T_to_xyzq(T_oc @ T_cb)
            t = mc.stamp_ns(o)
            w.write("/tf", rosmsg.tf_message(
                [("r%d/odom" % i, "r%d/base" % i, q_ob[:3], q_ob[3:]),
                 ("r%d/base" % i, "r%d/camera" % i, q_bc[:3], q_bc[3:])], t), msg.log_time)
        counts[i][topic] += 1
    for w in writers.values():
        w.close()
    for i in robots:
        if not counts[i]["/camera/color/image_raw"] or not counts[i]["/tf"]:
            raise mc.ConvertError(f"r{i}: no images or no odometry in {src}")
    return write_configs(export, out, robots, names, mesh, counts)


def write_configs(export, out, robots, names, mesh, counts):
    """hydra_r<i>.yaml, the launch file, ground truth and report.txt (counts:
    the bags' message counts, None when the bags were not written now)."""
    src = os.path.join(export, "bag")
    lines = ["source              : %s" % src, "scene-graph mesh    : %s" % ("on" if mesh else "off")]
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
            lines.append("r%d %-16s: %d colour, %d depth, %d camera_info, %d tf; depth %g-%g m; "
                         "frames r%d/odom -> r%d/base -> r%d/camera"
                         % (i, m, c["/camera/color/image_raw"], c["/camera/depth/image_rect"],
                            c["/camera/color/camera_info"], c["/tf"], lo, hi, i, i, i))
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
    ap.add_argument("--no-mesh", action="store_true", help="send the scene graph without its mesh")
    ap.add_argument("--configs-only", action="store_true",
                    help="rewrite the configs and launch file of an existing output, not its bags")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        convert(a.export, a.out, a.robots, mesh=not a.no_mesh, configs_only=a.configs_only)
    except mc.ConvertError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()

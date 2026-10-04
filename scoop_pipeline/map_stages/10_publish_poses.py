#!/usr/bin/env python3
"""
STAGE 10 - publish the best pose per robot as a ROS 2 mcap bag.

For every robot in the "10_publish" block the chosen stage-04 trajectory (a
mapping pass: its refined trajectory through 05's T_N_world, written by
mapping_track() under the same names)
(TUM, camera OPTICAL frame in `map`) is written AS the camera optical frame:

  /<robot>/global_pose   geometry_msgs/PoseStamped, frame_id "map"
                         pose of the camera OPTICAL frame (x right, y down,
                         z forward) in the anchored map frame shared by all
                         robots: zed_left_camera_optical_frame for mobile_1,
                         camera_color_optical_frame for mobile_2
  /<robot>/local_pose    geometry_msgs/PoseStamped: with the complete TF, the
                         pose of the frame that carries the robot's sensors
                         (the root of its tree: zed_camera_link -> ZED,
                         os_sensor, radars) in the robot's origin frame
                         (map_zed), i.e. the TF edge map_zed ~~ zed_camera_link;
                         mobile_2 (a robot with "local_frame", e.g. camera_link):
                         the TF edge map_realsense ~~ camera_link; without it,
                         the optical frame relative to its first pose,
                         frame_id "<robot>/start"
  /tf, /tf_static        the complete TF tree ("full_tf": true, the default):
                         the dataset bag's own TF with configs/static_tf.yaml
                         and the pipeline outputs put in (map_stages/pass_tf.py:
                         map at the root, each mobile platform under its origin
                         frame (map_zed, map_realsense), mobile_1 on this
                         trajectory, boards, infra cameras and infra1_link). Play it next to
                         the dataset bag with that bag's TF moved aside:
                           ros2 bag play <merged> --clock \
                               --remap /tf:=/tf_recorded /tf_static:=/tf_static_recorded
                           ros2 bag play <this bag>
                         A merge done after the map stages carries the same tree
                         itself. "full_tf": false: only
                         map -> <robot>/<optical_frame> on /tf ("write_tf").

The optical frame is the frame the images and depth are expressed in, so a
consumer can project through global_pose directly with the camera intrinsics.
No body-frame conversion is applied. cam_extrinsic_xyzquat is no longer read
here; set "publish_body_frame": true on a robot to get the old behaviour.

Timestamps are the original bag stamps, so the output can be played next to
the raw bag.

    python3 10_publish_poses.py [pipeline_config.json]      # writes the bag
    python3 10_publish_poses.py pipeline_config.json --dry  # counts only

No ROS install needed: the bag is written by scoop.bagwrite (MCAP +
metadata.yaml, as `ros2 bag record` leaves it), like the merge. With full_tf
the dataset bag's /tf is read through, which takes a few minutes on a big bag.

CONFIG
"10_publish": {
  "output_bag": "map_stages_20260828_outputs/coop2_best_poses",   <- a directory; mcap inside
  "map_frame": "map",
  "write_tf": true,
  "rate_hz": 0,                      <- 0 = every pose in the file; else decimate
  "robots": [
    { "name": "mobile_1",
      "traj": "map_stages_20260828_outputs/reference_coop2_all/traj_mobile_1_lidar_in_cam.tum",
      "optical_frame": "zed_left_camera_optical_frame" },
    { "name": "mobile_2",
      "traj": "map_stages_20260828_outputs/reference_coop2_all/traj_mobile_2_rs_odom_icp_boards.tum",
      "optical_frame": "camera_color_optical_frame" }
    <- the trajectories are the optical frame already (stage 04 state); they
       are published unchanged. Only if a consumer needs the odometry BODY
       frame instead, add to that robot:
         "publish_body_frame": true,
         "cam_extrinsic_xyzquat": [...],   (T_body_cam, as in stage 04)
         "body_frame": "zed_camera_link"
  ]
}
"""
import json
import os
import shutil
import sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def Rt(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


def inv(T):
    R = T[:3, :3]; o = np.eye(4); o[:3, :3] = R.T; o[:3, 3] = -R.T @ T[:3, 3]; return o


def make_T_xyzq(v):
    return Rt(Rot.from_quat(v[3:7]).as_matrix(), np.asarray(v[0:3], float))


def read_tum(path):
    A = np.loadtxt(path)
    if A.ndim == 1:
        A = A[None]
    A = A[np.argsort(A[:, 0])]
    ts = A[:, 0]
    Ts = np.array([Rt(Rot.from_quat(r[4:8]).as_matrix(), r[1:4]) for r in A])
    return ts, Ts


def decimate(ts, Ts, rate_hz):
    if not rate_hz or rate_hz <= 0:
        return ts, Ts
    keep, t_last = [], -1e18
    for i, t in enumerate(ts):
        if t - t_last >= (1.0 / rate_hz) * 0.9:
            keep.append(i); t_last = t
    return ts[keep], Ts[keep]


def robot_poses(rb, rate_hz):
    """-> ts, T_map_target, T_start_target (the same poses relative to the first).

    Stage 04 writes the trajectory of the camera OPTICAL frame in `map` and
    that is what gets published: T_map_target = T_map_cam, no transform.

    Only with "publish_body_frame": true is the pose moved to the odometry
    body frame through X = cam_extrinsic_xyzquat (T_body_cam):
        T_map_body = T_map_cam @ inv(X)
    """
    ts, Tc = read_tum(rb["traj"])
    ts, Tc = decimate(ts, Tc, rate_hz)

    if rb.get("publish_body_frame"):
        if not rb.get("cam_extrinsic_xyzquat"):
            raise SystemExit("%s: publish_body_frame needs cam_extrinsic_xyzquat"
                             % rb["name"])
        X = make_T_xyzq(rb["cam_extrinsic_xyzquat"])
        Tb = np.array([T @ inv(X) for T in Tc])
        frame = rb.get("body_frame", "base")
        what = "moved to the odometry BODY frame through inv(cam_extrinsic)"
    else:
        Tb = Tc
        frame = rb.get("optical_frame", rb.get("body_frame", "camera_optical_frame"))
        what = "published AS the camera optical frame (no transform applied)"
        if rb.get("cam_extrinsic_xyzquat") and not rb.get("_warned"):
            print("  (%s: cam_extrinsic_xyzquat is ignored - poses are the optical "
                  "frame; set publish_body_frame: true to apply it)" % rb["name"])
    rb["_frame"] = frame

    T0 = Tb[0]
    Tl = np.array([inv(T0) @ T for T in Tb])
    path = float(np.sum(np.linalg.norm(np.diff(Tb[:, :3, 3], axis=0), axis=1)))
    print("  %s: %d poses, %.1f s, path %.1f m -> frame '%s' (%s); start in map "
          "xyz=%s" % (rb["name"], len(ts), ts[-1] - ts[0], path, frame, what,
                      np.round(T0[:3, 3], 3).tolist()))
    return ts, Tb, Tl


def mapping_track(P):
    """A mapping pass has no 04: its robot's track in map, as 04 writes a run's
    (traj_<name>.tum and traj_<name>_in_cam.tum in odometry/reference_<tag>/),
    is the refined trajectory (dataset.traj, GLIM's world) through 05's
    T_N_world. Written when missing or older than its inputs; -> the _in_cam
    path, or None without those inputs."""
    import importlib
    import json
    from pipeline_common import load_traj
    src, af = P.outp(P.dataset.get("traj") or ""), P.anchor_frame()
    if not (P.dataset.get("traj") and os.path.exists(src) and os.path.exists(af)):
        return None
    p_l, p_c = P.lidar_track_traj(), P.lidar_track_traj("_in_cam")
    if os.path.exists(p_c) and os.path.getmtime(p_c) >= max(os.path.getmtime(src),
                                                             os.path.getmtime(af)):
        return p_c
    T_N = np.array(json.load(open(af))["T_N_world"], float)
    times, T = load_traj(src)
    T_map = np.einsum("ij,njk->nik", T_N, T)
    W = importlib.import_module("02_refine_poses")
    os.makedirs(os.path.dirname(p_c), exist_ok=True)
    W.write_traj(p_l, times, T_map)
    W.write_traj(p_c, times, np.einsum("nij,jk->nik", T_map, P.sensor.T_lidar_camera))
    print("  %s: %s through T_N_world (%d poses)" % (p_c, src, len(times)))
    return p_c


def full_tf(P, s):
    """The complete tree against the dataset bag's TF: configs/static_tf.yaml
    (the calibrations the bag does not have yet), then the pipeline outputs
    (pass_tf.plan). -> (static [(parent, child, t, q)], drop f(parent, child)
    or None, /tf [(t_ns, parent, child, t, q)]). A bag that has the tree
    already gets it put in again from the outputs as they are now."""
    import yaml
    from scoop import tftree
    import pass_tf
    bag = P.dataset["bag"]
    print("TF of %s ..." % bag)
    edges = tftree.tf_edges([bag])
    st = []
    cal_path = s.get("static_tf") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "static_tf.yaml")
    with open(os.path.expanduser(cal_path)) as fh:
        cals = tftree.load_calibrations(yaml.safe_load(fh))
    frames = set(edges) | {e.parent for e in edges.values()}
    for c in cals:
        if c.parent in frames and c.child in frames and \
                tftree._root(c.parent, edges) == tftree._root(c.child, edges):
            continue                                    # the merge put it in already
        add, notes = tftree.attach([c], edges)
        for p, ch, t, q in add:
            edges[ch] = tftree.Edge(p, ch, True, tftree.matrix(t, q))
            frames |= {p, ch}
        st += add
        for n in notes:
            print("    %s  (configs/static_tf.yaml)" % n)
    drop, more, extra, after = pass_tf.plan(P, edges, [bag])
    print(tftree.format_tree(after))
    return (st + more, tftree.edge_matcher(["%s->%s" % d for d in drop]) if drop else None,
            extra)


def main():
    cfg_path = next((a for a in sys.argv[1:] if not a.startswith("--")), "pipeline_config.json")
    dry = "--dry" in sys.argv
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pipeline_common import load_pipeline
    P = load_pipeline(cfg_path)
    s = P.cfg.get("10_publish")
    if s is None:
        raise SystemExit("add a '10_publish' block to %s (sample in this file)" % cfg_path)
    s = dict(s)
    # processed layout: the bag into bags/ (<tag>_best_poses); a bare traj
    # name is a stage-04 output in odometry/reference_<tag>/
    s["output_bag"] = P.outp(s.get("output_bag") or "{tag}_best_poses")
    print(P.describe())
    map_frame = s.get("map_frame", "map")
    rate_hz = float(s.get("rate_hz", 0) or 0)
    write_tf = bool(s.get("write_tf", True))
    if not P.reference:
        mapping_track(P)
    robots = []
    for rb in s["robots"]:
        if not rb.get("enabled", True):
            print("  %s: switched off in the config" % rb["name"])
            continue
        rb = dict(rb)
        if not os.path.dirname(rb["traj"]):
            rb["traj"] = os.path.join(P.reference_dir(), rb["traj"])
        rb["traj"] = os.path.expanduser(rb["traj"])
        if not os.path.exists(rb["traj"]):
            raise SystemExit("%s: trajectory %s not found" % (rb["name"], rb["traj"]))
        robots.append((rb, *robot_poses(rb, rate_hz)))
    if dry:
        print("dry run: %d robots, %d messages" % (
            len(robots), sum(len(r[1]) * (3 if write_tf else 2) for r in robots)))
        return

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scoop import rosmsg
    from scoop.bag import TopicSchema
    from scoop.bagwrite import BagWriter

    out = s["output_bag"]
    if os.path.isdir(out):
        shutil.rmtree(out)
    topics = []
    for rb, ts, Tb, Tl in robots:
        for nm in ("global_pose", "local_pose"):
            topics.append("/%s/%s" % (rb["name"], nm))
    full = bool(s.get("full_tf", True))
    if full and not os.path.exists(P.dataset["bag"]):
        print("  (no dataset bag %s: map -> <robot>/<optical_frame> only)" % P.dataset["bag"])
        full = False
    write_tf = write_tf and not full

    parts = full_tf(P, s) if full else None
    # local_pose of each robot the tree moves: the TF edge origin ~~ sensor root
    # (the robot's "local_frame"; the dataset's machine: the first one the plan moves)
    by_root = {}
    for t_ns, p, c, t, q in (parts[2] if parts else []):
        by_root.setdefault(c, {})[t_ns] = (p, c, t, q)
    first = parts[2][0][2] if parts and parts[2] else None
    local = {rb["name"]: by_root.get(rb.get("local_frame") or
                                     (first if rb["name"] == P.machine else None), {})
             for rb, *_ in robots}

    # interleave all robots by time so the bag plays in order
    events = []
    for rb, ts, Tb, Tl in robots:
        for i, t in enumerate(ts):
            events.append((t, rb, Tb[i], Tl[i]))
    events.sort(key=lambda e: e[0])

    def messages():
        for t, rb, Tm, Tloc in events:
            t_ns = int(round(t * 1e9))
            qm = Rot.from_matrix(Tm[:3, :3]).as_quat()
            ql = Rot.from_matrix(Tloc[:3, :3]).as_quat()
            yield t_ns, "/%s/global_pose" % rb["name"], \
                rosmsg.pose_stamped(t_ns, map_frame, Tm[:3, 3], qm)
            if t_ns in local[rb["name"]]:
                p, c, t_, q_ = local[rb["name"]][t_ns]
                yield t_ns, "/%s/local_pose" % rb["name"], rosmsg.pose_stamped(t_ns, p, t_, q_)
            else:
                yield t_ns, "/%s/local_pose" % rb["name"], \
                    rosmsg.pose_stamped(t_ns, "%s/start" % rb["name"], Tloc[:3, 3], ql)
            if write_tf:
                yield t_ns, "/tf", rosmsg.tf_message(
                    [(map_frame, "%s/%s" % (rb["name"], rb["_frame"]), Tm[:3, 3], qm)], t_ns)

    if full:
        from scoop import merge
        st, drop_tf, extra = parts
        ps = "geometry_msgs/msg/PoseStamped"
        sch = TopicSchema(ps, "ros2msg", rosmsg.msgdef(ps).encode(), "cdr")
        seq = {}

        def records():
            for t_ns, tp, m in messages():
                seq[tp] = seq.get(tp, -1) + 1
                yield t_ns, tp, sch, t_ns, seq[tp], {}, rosmsg.serialize(m)
        res = merge.merge_bags([P.dataset["bag"]], out, topics=["/tf", "/tf_static"],
                               static_tf=st, drop_tf=drop_tf, extra_tf=extra,
                               extra_records=records())
        n = sum(res.counts.values())
        topics += ["/tf_static"]
    else:
        n = 0
        with BagWriter(out) as w:
            for t_ns, tp, m in messages():
                w.write(tp, m, t_ns)
                n += 1
    print("wrote %s: %d messages on %s%s" % (out, n, ", ".join(topics),
                                            ", /tf" if write_tf or full else ""))
    print("  global_pose: camera optical frame in '%s' (unless publish_body_frame)" % map_frame)
    for name, lp in local.items():
        if lp:
            p, c = next(iter(lp.values()))[:2]
            print("  local_pose of %s: %s in %s (the TF edge)" % (name, c, p))
        else:
            print("  local_pose of %s: the optical frame relative to its first pose "
                  "(frame '%s/start')" % (name, name))


if __name__ == "__main__":
    main()

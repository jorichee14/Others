#!/usr/bin/env python3
"""cuVSLAM localizing in a map it saved on another pass, scored as the other
methods are (slam/odometry/score.py): does a map improve cuVSLAM's odometry?

    python scoop_pipeline/processing/vslam_loc.py map <mapping config> [--machine mobile_2] [--bag B]
        [--ref-topic /mobile_2/global_pose [--ref-frame F]] [--imu] [--rate R] [--redo]
    python scoop_pipeline/processing/vslam_loc.py localize <coop config> --map-from <mapping config>
        [--machine mobile_2] [--imu] [--rate R] [--redo]

map: cuVSLAM (SLAM on) over the mapping pass's stereo pair (the pass the
twin's anchored map was built from, so the camera map and the LiDAR map come
from the same drive; a mapping pass's mobile_2 was parked) (processing/vslam.py's
preset for the robot); once the bag has played, its keyframes (GetAllPoses)
and the map (the FilePath service named save_map) are written, through
run_vslam.sh's after-play hook. Then the map's frame is put in the twin's: the
keyframes, in the LiDAR frame (the bag's /tf_static), fitted onto the mapping
pass's track of the robot in the twin: --ref-topic /<machine>/global_pose
(stage 10's, in the finalized bag; --ref-frame, the camera optical frame by
default), or stage 04's traj_<machine>_lidar.tum (the robot's "tracks" entry
in the mapping config's 04_reference), by the same pose fit score.py uses.
  -> work/<date>/<mapping pass>/<machine>/vslam_map/: map/, keyframes.tum
     (T_cmap_base), alignment.json (T_twin_cmap, the fit's residual), odometry/

localize: cuVSLAM over the coop pass with that map, the map's frame put in
the twin by the map poses within --align-radius m (5) of the run's path only
(a long drive's map bends; around one site it can still sit right); 3 s after the robot's
first image and its reference both exist, LocalizeInMap is called with the
reference's pose there, put in the map's frame (run_vslam.sh's during-play
hook, on /clock), and retried every 2 s until it succeeds. From then on the
pose in the twin is T_twin_cmap @ T_cmap_odom (/tf, recorded) @ T_odom_base
(the odometry): written as score/trajectory.tum with a run.json saying it is
in map, and scored against the coop pass's stage 04 track (ATE, in map with
no fit, RPE, drift; with VSLAM_MONITOR=1 also CPU, GPU and latency).
  -> work/<date>/<coop pass>/<machine>/vslam_loc/: odometry/, localize.json,
     score/ (run.json, trajectory.tum, accuracy.json, summary.json)

Run with VSLAM_MONITOR=1 in front for the cost columns. The service names
and request fields are looked up in the running node (isaac_ros_visual_slam
3.2: FilePath, GetAllPoses, LocalizeInMap); a missing one stops the run with
what the node offers.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))
sys.path.insert(0, os.path.join(ROOT, "slam", "odometry"))
sys.path.insert(0, os.path.join(ROOT, "datasets"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))

import vslam as VS                                                  # noqa: E402

MAP_FRAME, ODOM_FRAME = "map_vslam", "odom_vslam"                   # vslam.node_params
WARM_S = 3.0

SRV = '''# written by scoop_pipeline/processing/vslam_loc.py: cuVSLAM's map services,
# called from processing/run_vslam.sh's hooks, in the container
#   python3 vslam_srv.py after <dir>    keyframes (GetAllPoses) -> keyframes.tum, map -> <dir>/map
#   python3 vslam_srv.py during <dir>   at during_play.json's time on /clock, LocalizeInMap
import json
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from rosidl_runtime_py.utilities import get_service

PKG = "isaac_ros_visual_slam_interfaces/srv/"


def find(node, kind, wait_s=60.0):
    t = time.time()
    while time.time() - t < wait_s:
        names = sorted(n for n, ts in node.get_service_names_and_types() if PKG + kind in ts)
        if names:
            return names
        rclpy.spin_once(node, timeout_sec=0.5)
    offered = sorted(n for n, ts in node.get_service_names_and_types() if "visual_slam" in n)
    sys.exit("no %s service after %.0f s; the node offers %s" % (kind, wait_s, offered))


def call(node, name, kind, fill, timeout_s=600.0):
    srv = get_service(PKG + kind)
    cli = node.create_client(srv, name)
    if not cli.wait_for_service(timeout_sec=30.0):
        sys.exit("%s is not answering" % name)
    req = srv.Request()
    fill(req, req.get_fields_and_field_types())
    print("call %s (%s): %s" % (name, kind, req), flush=True)
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout_s)
    if not fut.done():
        sys.exit("%s: no answer in %.0f s" % (name, timeout_s))
    return fut.result()


def field(fields, want):
    for k, ty in fields.items():
        if ty == want:
            return k
    sys.exit("no %s field in %s" % (want, fields))


def ok(res):
    f = res.get_fields_and_field_types()
    return bool(getattr(res, "success")) if "success" in f else True


def after(node, d):
    cfg = json.load(open(os.path.join(d, "after_play.json")))
    out = {}
    names = find(node, "GetAllPoses")

    def fill_poses(req, fields):
        for k, ty in fields.items():
            if ty in ("int32", "int64", "uint32", "uint64"):
                setattr(req, k, 1000000)
    res = call(node, names[0], "GetAllPoses", fill_poses)
    seq = [k for k, ty in res.get_fields_and_field_types().items() if "PoseStamped" in ty]
    poses = getattr(res, seq[0]) if seq else []
    with open(os.path.join(d, "keyframes.tum"), "w") as fh:
        for p in poses:
            t = p.header.stamp.sec + p.header.stamp.nanosec * 1e-9
            q, o = p.pose.position, p.pose.orientation
            fh.write("%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\\n" % (t, q.x, q.y, q.z, o.x, o.y, o.z, o.w))
    out["keyframes"] = len(poses)
    out["keyframes_frame"] = poses[0].header.frame_id if poses else None
    print("keyframes: %d (%s)" % (len(poses), out["keyframes_frame"]), flush=True)
    saves = [n for n in find(node, "FilePath") if "save" in n]
    if not saves:
        sys.exit("no save service among %s" % find(node, "FilePath"))
    path = os.path.join(d, cfg.get("map", "map"))
    if os.path.exists(path):
        sys.exit("%s exists: the map is saved into a new folder" % path)

    def fill_path(req, fields):
        setattr(req, field(fields, "string"), path)
    res = call(node, saves[0], "FilePath", fill_path, timeout_s=float(cfg.get("timeout_s", 600)))
    out.update({"save_service": saves[0], "map": path, "saved": ok(res), "answer": str(res)})
    print("map saved: %s (%s)" % (out["saved"], res), flush=True)
    json.dump(out, open(os.path.join(d, "after_result.json"), "w"), indent=1)
    if not out["saved"]:
        sys.exit(1)


def during(node, d):
    cfg = json.load(open(os.path.join(d, "during_play.json")))
    now = {"t": 0.0}
    node.create_subscription(Clock, "/clock", lambda m: now.update(t=m.clock.sec + m.clock.nanosec * 1e-9), 10)
    print("waiting for %.3f s on /clock" % cfg["at_s"], flush=True)
    while now["t"] < cfg["at_s"]:
        rclpy.spin_once(node, timeout_sec=0.2)
    name = find(node, "LocalizeInMap")[0]
    path = os.path.join(d, cfg.get("map", "map"))
    h = cfg["hint"]
    tries = []

    def fill(req, fields):
        setattr(req, field(fields, "string"), path)
        pose = getattr(req, field(fields, "geometry_msgs/Pose"))
        pose.position.x, pose.position.y, pose.position.z = h[:3]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = h[3:7]
    for k in range(int(cfg.get("tries", 15))):
        t_call = now["t"]
        res = call(node, name, "LocalizeInMap", fill, timeout_s=60.0)
        tries.append({"clock_s": t_call, "success": ok(res), "answer": str(res)})
        print("try %d at %.3f s: %s" % (k + 1, t_call, res), flush=True)
        json.dump({"service": name, "map": path, "hint": h, "tries": tries},
                  open(os.path.join(d, "localize.json"), "w"), indent=1)
        if ok(res):
            return
        t_next = now["t"] + float(cfg.get("every_s", 2.0))
        while now["t"] < t_next:
            rclpy.spin_once(node, timeout_sec=0.2)
    sys.exit("LocalizeInMap did not succeed in %d tries" % len(tries))


def main():
    mode, d = sys.argv[1], os.path.abspath(sys.argv[2])
    rclpy.init()
    node = Node("vslam_loc_" + mode)
    try:
        (after if mode == "after" else during)(node, d)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
'''


def T_of(v):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(v[3:7]).as_matrix()
    T[:3, 3] = v[:3]
    return T


def vec_of(T):
    from scipy.spatial.transform import Rotation
    return [float(x) for x in T[:3, 3]] + [float(x) for x in Rotation.from_matrix(T[:3, :3]).as_quat()]


def load_tum(path):
    a = np.loadtxt(path, ndmin=2)
    return a[:, 0], np.array([T_of(r[1:8]) for r in a]) if len(a) else np.zeros((0, 4, 4))


def write_tum(path, t, T):
    with open(path, "w") as fh:
        for ti, Ti in zip(t, T):
            fh.write("%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n" % ((ti,) + tuple(vec_of(Ti))))


def T_base_lidar(bag, machine, base):
    import make_config as MK
    paths = MK.mc._paths(bag)
    T, src = MK.to_lidar(paths, MK.mc.build_tf_static(paths), machine, base)
    return T, src


def align(t, T_cmap_base, ref_t, ref_T, T_bl):
    """T_twin_cmap putting the keyframes (in the LiDAR frame) on the reference
    (T_twin_lidar), by score.py's pose fit; and the residual (m) per keyframe."""
    import score as SC
    import trajs
    m = (t >= ref_t[0]) & (t <= ref_t[-1])
    if m.sum() < 3:
        raise SystemExit("only %d keyframes inside the reference's span" % m.sum())
    E = T_cmap_base[m] @ T_bl
    R = trajs.interp(ref_t, ref_T, t[m])
    Rf, tf = SC.pose_fit(E, R)
    A = np.eye(4)
    A[:3, :3], A[:3, 3] = Rf, tf
    res = np.linalg.norm((A @ E)[:, :3, 3] - R[:, :3, 3], axis=1)
    return A, res, int(m.sum())


def compose(t_odom, T_odom_base, tf, t_from):
    """The poses in the map, T_map_odom (latest at or before each stamp, from
    `tf`: [(t, T_map_odom)] sorted) @ T_odom_base, for stamps from t_from on."""
    if not tf:
        return np.zeros(0), np.zeros((0, 4, 4))
    tt = np.array([x[0] for x in tf])
    keep, out = [], []
    for ti, Ti in zip(t_odom, T_odom_base):
        if ti < t_from:
            continue
        k = int(np.searchsorted(tt, ti, side="right")) - 1
        if k < 0:
            continue
        keep.append(ti)
        out.append(tf[k][1] @ Ti)
    return np.array(keep), np.array(out).reshape(-1, 4, 4)


def read_map_odom(rec):
    """[(t s, T_map_odom)] from the recording's /tf, sorted."""
    from scoop import bag as sbag, tftree
    r = sbag.open_bag(rec)
    if "/tf" not in r.topics():
        raise SystemExit("%s has no /tf (map -> odom)" % rec)
    out = []
    for _, _, payload, decode in r.iter_raw(["/tf"]):
        for tr in decode(payload).transforms:
            if tr.header.frame_id.lstrip("/") == MAP_FRAME and tr.child_frame_id.lstrip("/") == ODOM_FRAME:
                q, p = tr.transform.rotation, tr.transform.translation
                out.append((tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9,
                            tftree.matrix([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])))
    out.sort(key=lambda x: x[0])
    return out


def out_for(P, machine, name):
    """work/<date>/<pass>/<machine>/<name>, beside the robot's other outputs (a
    pass bag that is already in its robot's folder, as the mapping pass's, gets
    no second <machine>/)."""
    d = os.path.dirname(VS.out_dir(P, machine))
    if os.path.basename(os.path.dirname(d)) == machine:
        d = os.path.dirname(d)
    return os.path.join(d, name)


def preset(machine, imu):
    c = dict(VS.MACHINES[machine])
    c["enable_imu_fusion"] = bool(imu)
    c["params"] = dict(c.get("params") or {}, enable_localization_n_mapping=True)
    return c


def stage(part, c, bag):
    """The launch file, to_mono.py when needed, vslam_srv.py, in `part`."""
    os.makedirs(part)
    enc = VS.image_encoding(bag, c["images"][0])
    convert = [] if enc in VS.CUVSLAM_ENCODINGS else [(t, VS.mono_topic(t)) for t in c["images"]]
    if convert:
        open(os.path.join(part, "to_mono.py"), "w").write(VS.TO_MONO)
    launch = os.path.join(part, "vslam.launch.py")
    open(launch, "w").write(VS.LAUNCH % (VS.node_params(c), VS.remaps(c, convert), convert))
    open(os.path.join(part, "vslam_srv.py"), "w").write(SRV)
    return launch


def play(part, launch, bag, c, rate):
    topics = list(c["images"]) + list(c["camera_infos"]) + [c["imu"], "/tf_static"]
    code = subprocess.call([VS.RUN, bag, os.path.join(part, "odometry"), launch, str(rate)] + topics)
    if code != 0 or not os.path.isfile(os.path.join(part, "odometry", "metadata.yaml")):
        raise SystemExit("cuVSLAM failed (exit %d); what there is stays in %s" % (code, part))


def fresh(out, redo):
    if os.path.isdir(out):
        if not redo:
            return False
        prev = out + "_prev"
        if os.path.isdir(prev):
            shutil.rmtree(prev)
        shutil.move(out, prev)
        print("the run before: kept as %s" % prev)
    return True


def cmd_map(a):
    from pipeline_common import load_pipeline
    import swarm_slam_lidar as SL
    P = load_pipeline(a.config)
    bag = os.path.abspath(os.path.expanduser(a.bag)) if a.bag else os.path.expanduser(P.dataset["bag"])
    c = preset(a.machine, a.imu)
    out = out_for(P, a.machine, "vslam_map")
    if a.ref_topic:
        ref = ref_from_topic(bag, a.machine, a.ref_topic, a.ref_frame,
                             os.path.join(os.path.dirname(out), "ref_%s.tum" % a.ref_topic.strip("/").replace("/", "_")))
    else:
        ref = a.ref or SL.default_gt(bag, a.machine)
    if not ref or not os.path.exists(ref):
        raise SystemExit("no reference track of %s in this pass (%s): give --ref-topic (stage 10's "
                         "/%s/global_pose in the finalized bag) or run stage 04 with its \"tracks\" entry"
                         % (a.machine, ref, a.machine))
    print("bag: %s\nout: %s\nreference: %s" % (bag, out, ref))
    if fresh(out, a.redo):
        VS.check(bag, c)
        part = os.path.join(os.path.dirname(out), ".partial", "vslam_map")
        if os.path.isdir(part):
            shutil.rmtree(part)
        launch = stage(part, c, bag)
        json.dump({"map": "map"}, open(os.path.join(part, "after_play.json"), "w"))
        play(part, launch, bag, c, a.rate)
        if not os.path.isdir(os.path.join(part, "map")):
            raise SystemExit("no map saved in %s" % part)
        shutil.move(part, out)
    else:
        print("[skip] the map is there (--redo builds it again)")
    t, K = load_tum(os.path.join(out, "keyframes.tum"))
    src = "keyframes (GetAllPoses)"
    if len(t) < 3 or np.all(t <= 0):
        print("! keyframes without stamps: aligning with the odometry instead (it drifts: the fit is looser)")
        VS.to_tum(os.path.join(out, "odometry"), os.path.join(out, "odometry.tum"), c["base_frame"])
        t, K = load_tum(os.path.join(out, "odometry.tum"))
        src = "odometry (map = odom at the start)"
    T_bl, bl_src = T_base_lidar(bag, a.machine, c["base_frame"])
    rt, rT = load_tum(ref)
    A, res, n = align(t, K, rt, rT, T_bl)
    rep = {"T_twin_cmap": A.tolist(), "from": src, "poses": n, "reference": ref, "base_to_lidar": bl_src,
           "residual_cm": {"rmse": round(float(np.sqrt((res ** 2).mean()) * 100), 2),
                           "median": round(float(np.median(res) * 100), 2),
                           "max": round(float(res.max() * 100), 2)}}
    json.dump(rep, open(os.path.join(out, "alignment.json"), "w"), indent=1)
    print("map of %s in the twin: %d %s fitted onto %s, residual %.2f cm rmse (median %.2f, max %.2f)"
          % (a.machine, n, src, os.path.basename(ref), rep["residual_cm"]["rmse"], rep["residual_cm"]["median"],
             rep["residual_cm"]["max"]))


def ref_from_topic(bag, machine, topic, frame, path):
    """A pose topic of the robot in map (stage 10's /<robot>/global_pose: its
    `frame`, the camera optical frame unless the pass set a base_frame) as
    T_map_lidar TUM at `path`, through the bag's /tf_static."""
    from scoop import tftree
    import make_config as MK
    poses, parent, _ = tftree.read_pose_topic([bag], topic)
    if parent and parent.lstrip("/") != "map":
        print("! %s is in %s, not map" % (topic, parent))
    paths = MK.mc._paths(bag)
    T_fl, src = MK.to_lidar(paths, MK.mc.build_tf_static(paths), machine, frame)
    poses.sort(key=lambda x: x[0])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_tum(path, np.array([p[0] for p in poses]) * 1e-9, np.array([p[1] @ T_fl for p in poses]))
    print("reference: %s (%d poses of %s in %s) -> the LiDAR by %s: %s" % (topic, len(poses), frame, parent, src, path))
    return path


def align_near(mdir, aj, T_bl, run_T, radius, out):
    """T_twin_cmap from the map's poses within `radius` m of this run's path
    only (where they were, by the mapping pass's reference): a map that bends
    over a long drive can still sit right around one site."""
    from scipy.spatial import cKDTree
    import trajs
    t, K = load_tum(os.path.join(mdir, "keyframes.tum"))
    mt, mT = load_tum(aj["reference"])
    m = (t >= mt[0]) & (t <= mt[-1])
    t, K = t[m], K[m]
    d, _ = cKDTree(run_T[::5, :3, 3]).query(trajs.interp(mt, mT, t)[:, :3, 3])
    near = d <= radius
    if near.sum() < 30:
        raise SystemExit("only %d map poses within %g m of this run's path: --align-radius 0 for the "
                         "global fit, or a larger radius" % (near.sum(), radius))
    A, res, n = align(t[near], K[near], mt, mT, T_bl)
    rep = {"T_twin_cmap": A.tolist(), "radius_m": radius, "poses": n,
           "residual_cm": {"rmse": round(float(np.sqrt((res ** 2).mean()) * 100), 2),
                           "median": round(float(np.median(res) * 100), 2), "max": round(float(res.max() * 100), 2)},
           "global_residual_cm": aj["residual_cm"]}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out + "_alignment.json", "w"), indent=1)
    print("map aligned around this run: %d map poses within %g m of its path, residual %.2f cm rmse "
          "(median %.2f, max %.2f; the global fit: %.2f cm rmse)"
          % (n, radius, rep["residual_cm"]["rmse"], rep["residual_cm"]["median"], rep["residual_cm"]["max"],
             aj["residual_cm"]["rmse"]))
    return A


def first_stamp(bag, topic):
    import make_config as MK
    m = MK.mc.first_msg(MK.mc._paths(bag), topic)
    if m is None:
        raise SystemExit("no %s in %s" % (topic, bag))
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def cmd_localize(a):
    from pipeline_common import load_pipeline
    import swarm_slam_lidar as SL
    import score as SC
    P, Pm = load_pipeline(a.config), load_pipeline(a.map_from)
    bag = os.path.expanduser(P.dataset["bag"])
    c = preset(a.machine, a.imu)
    out = out_for(P, a.machine, "vslam_loc")
    mdir = out_for(Pm, a.machine, "vslam_map")
    if not os.path.isfile(os.path.join(mdir, "alignment.json")):
        raise SystemExit("no map at %s: vslam_loc.py map %s first" % (mdir, a.map_from))
    aj = json.load(open(os.path.join(mdir, "alignment.json")))
    A = np.array(aj["T_twin_cmap"])
    ref = a.ref or SL.default_gt(bag, a.machine)
    if not ref or not os.path.exists(ref):
        raise SystemExit("no reference track of %s in this pass (%s)" % (a.machine, ref))
    T_bl, _ = T_base_lidar(bag, a.machine, c["base_frame"])
    rt, rT = load_tum(ref)
    import trajs
    if a.align_radius > 0:
        A = align_near(mdir, aj, T_bl, rT, a.align_radius, out)
    t_call = max(first_stamp(bag, c["images"][0]), rt[0]) + WARM_S
    hint = np.linalg.inv(A) @ trajs.interp(rt, rT, np.array([t_call]))[0] @ np.linalg.inv(T_bl)
    print("bag: %s\nout: %s\nmap: %s\nlocalize at %.3f s, from the reference's pose there" % (bag, out, mdir, t_call))
    if fresh(out, a.redo):
        VS.check(bag, c)
        part = os.path.join(os.path.dirname(out), ".partial", "vslam_loc")
        if os.path.isdir(part):
            shutil.rmtree(part)
        launch = stage(part, c, bag)
        shutil.copytree(os.path.join(mdir, "map"), os.path.join(part, "map"))
        open(os.path.join(part, "record_extra.txt"), "w").write("/tf\n")
        json.dump({"at_s": t_call, "hint": vec_of(hint), "map": "map", "tries": 15, "every_s": 2.0},
                  open(os.path.join(part, "during_play.json"), "w"), indent=1)
        play(part, launch, bag, c, a.rate)
        shutil.rmtree(os.path.join(part, "map"))                  # a copy of the mapping pass's
        shutil.move(part, out)
    else:
        print("[skip] the run is there (--redo runs it again)")
    lj = os.path.join(out, "localize.json")
    loc = json.load(open(lj)) if os.path.exists(lj) else {"tries": []}
    won = [x for x in loc["tries"] if x["success"]]
    if not won:
        raise SystemExit("cuVSLAM never localized in the map (%d tries; %s)" % (len(loc["tries"]), lj))
    t_ok = won[0]["clock_s"]
    rec = os.path.join(out, "odometry")
    VS.to_tum(rec, os.path.join(out, "odometry.tum"), c["base_frame"])
    to, To = load_tum(os.path.join(out, "odometry.tum"))
    tm, Tm = compose(to, To, read_map_odom(rec), t_ok)
    sd = os.path.join(out, "score")
    os.makedirs(sd, exist_ok=True)
    write_tum(os.path.join(sd, "trajectory.tum"), tm, np.einsum("ij,njk->nik", A, Tm))
    rj = {"method": "cuvslam_loc", "machine": a.machine, "bag": os.path.abspath(bag), "tum_frame": c["base_frame"],
          "world": "map", "ref": os.path.abspath(ref), "key_topic": c["images"][0],
          "play_topics": list(c["images"]) + list(c["camera_infos"]) + [c["imu"]],
          "settings": {"map": [mdir, None, "cuVSLAM's own, saved on %s" % a.map_from],
                       "localized_at_s": [t_ok, None, "%d tries" % len(loc["tries"])]}}
    json.dump(rj, open(os.path.join(sd, "run.json"), "w"), indent=1)
    mon = os.path.join(rec, "monitor", "summary.json")
    if os.path.exists(mon):
        shutil.copy(mon, os.path.join(sd, "summary.json"))
    json.dump({"method": "ran", "play": "done"}, open(os.path.join(sd, "status.json"), "w"))
    acc = SC.run_score(sd, ref)
    print("cuVSLAM localized in its %s map from %.1f s (%d tries); %d poses\n%s"
          % (os.path.basename(os.path.dirname(os.path.dirname(mdir))), t_ok - rt[0], len(loc["tries"]), len(tm),
             SC.report(acc)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("map", "localize"))
    ap.add_argument("config")
    ap.add_argument("--map-from", default=None, help="localize: the mapping pass's config (where the map is)")
    ap.add_argument("--machine", default="mobile_2", choices=sorted(VS.MACHINES))
    ap.add_argument("--ref", default=None, help="the pass's T_map_lidar TUM of the robot (default: stage 04's)")
    ap.add_argument("--ref-topic", default=None, help="map: the robot's pose topic in map in the pass bag instead "
                    "(stage 10's /<robot>/global_pose)")
    ap.add_argument("--ref-frame", default="camera_color_optical_frame",
                    help="--ref-topic's frame (stage 10: the camera optical frame, or the pass's base_frame)")
    ap.add_argument("--imu", action="store_true", help="IMU fusion on (the measured odometry had it off)")
    ap.add_argument("--bag", default=None, help="map: the pass bag, where it is not at the config's path (an "
                    "external drive: the container must see it at the same path); the outputs stay where the "
                    "config puts them")
    ap.add_argument("--align-radius", type=float, default=5.0, help="localize: fit the map to the twin with the "
                    "map poses within this many m of the run's path only (0: the map's global fit)")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    if a.step == "localize" and not a.map_from:
        ap.error("localize needs --map-from <mapping config>")
    (cmd_map if a.step == "map" else cmd_localize)(a)


if __name__ == "__main__":
    main()

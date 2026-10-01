#!/usr/bin/env python3
"""
Shared helpers for the whole map pipeline. ONE place for:
  * SE3 math (quaternion <-> rotation, T composition)
  * trajectory / extrinsic / cameras-yaml IO
  * the single config loader that reads pipeline_config.json AND resolves the
    sensor calibration (intrinsics, T_lidar_camera, ROS topics) from the
    calibration.json it points at -- so no stage script hardcodes any of it.

CHANGES vs the previous revision
  * R_to_q: the old form divided by 4*w, which blows up as w -> 0 (rotations
    near 180 deg). With several boards facing different directions that WILL
    happen and it silently produced NaN in avg_T and in every exported
    quaternion. Replaced with the branch-on-largest-diagonal (Shepperd) form.
  * slerp + pose_at_interp: nearest-neighbour trajectory lookup costs up to
    time_tol of lever arm (0.1 s of walking is tens of mm -- the same size as
    the drift being measured). Interpolate instead.
  * Pipeline.stage(): resolving extrinsic_yaml unconditionally raised KeyError
    for cameras that do not have one (e.g. source: "board").
"""
import os
import json
from fnmatch import fnmatch
import numpy as np


# ------------------------------- SE3 math -------------------------------
def qR(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def make_T(t, q):
    T = np.eye(4); T[:3, :3] = qR(q); T[:3, 3] = np.asarray(t, float); return T


def R_to_q(R):
    """Rotation matrix -> quaternion [x, y, z, w], numerically safe everywhere.

    Branches on the largest diagonal term so the divisor is never near zero.
    The naive 1/(4w) form is singular at 180 deg rotations."""
    R = np.asarray(R, float)
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    q /= n
    return q if q[3] >= 0 else -q          # canonical hemisphere


def ang_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))


def slerp(q0, q1, f):
    """Shortest-path quaternion slerp, xyzw, both unit-norm."""
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1; d = -d
    d = min(d, 1.0)
    if d > 0.9995:
        q = q0 + f * (q1 - q0)
        return q / np.linalg.norm(q)
    th0 = np.arccos(d); th = th0 * f
    q2 = q1 - q0 * d; q2 /= np.linalg.norm(q2)
    return q0 * np.cos(th) + q2 * np.sin(th)


# ------------------------- trajectory / extrinsics ----------------------
def load_traj(path):
    """TUM: t tx ty tz qx qy qz qw = T_world_lidar. Returns sorted (t[], T[])."""
    t, T = [], []
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        v = [float(x) for x in ln.split()]
        t.append(v[0]); T.append(make_T(v[1:4], v[4:8]))
    t = np.array(t); T = np.array(T); o = np.argsort(t)
    return t[o], T[o]


def pose_at(tr_t, tr_T, stamp):
    """Nearest-in-time pose. Returns (T, gap_seconds). Kept for compatibility;
    prefer pose_at_interp."""
    i = int(np.searchsorted(tr_t, stamp)); i = min(max(i, 0), len(tr_t) - 1)
    if i > 0 and (stamp - tr_t[i - 1]) < (tr_t[i] - stamp):
        i -= 1
    return tr_T[i], abs(float(tr_t[i] - stamp))


def pose_at_interp(tr_t, tr_T, stamp):
    """Linear + slerp interpolated trajectory pose.

    Returns (T, gap_seconds) where gap is the distance to the NEAREST sample --
    so an existing `gap <= time_tol` test keeps the same meaning, it just no
    longer carries the lever-arm error that nearest-neighbour did."""
    n = len(tr_t)
    if n == 0:
        raise SystemExit("empty trajectory")
    if n == 1:
        return tr_T[0], abs(float(tr_t[0] - stamp))
    i = int(np.searchsorted(tr_t, stamp))
    if i <= 0:
        return tr_T[0], abs(float(tr_t[0] - stamp))
    if i >= n:
        return tr_T[-1], abs(float(tr_t[-1] - stamp))
    t0, t1 = float(tr_t[i - 1]), float(tr_t[i])
    gap = min(stamp - t0, t1 - stamp)
    f = 0.0 if t1 == t0 else (stamp - t0) / (t1 - t0)
    T0, T1 = tr_T[i - 1], tr_T[i]
    p = T0[:3, 3] * (1.0 - f) + T1[:3, 3] * f
    q = slerp(R_to_q(T0[:3, :3]), R_to_q(T1[:3, :3]), f)
    T = np.eye(4); T[:3, :3] = qR(q); T[:3, 3] = p
    return T, abs(float(gap))


def parse_extrinsic_yaml(path):
    """Read a per-camera calibration YAML (flat 'key: [..]' / 'key: scalar')."""
    d = {}
    for ln in open(path):
        if ":" not in ln:
            continue
        k, v = ln.split(":", 1); k = k.strip(); v = v.strip()
        if v.startswith("["):
            d[k] = [float(x) for x in v.strip("[]").split(",")]
        elif v:
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
    ext = make_T(d["translation"], d["quaternion_xyzw"]) if "translation" in d else None
    pim = (make_T(d["pose_in_map_translation"], d["pose_in_map_quaternion_xyzw"])
           if "pose_in_map_translation" in d else None)
    stamp = float(d["stamp"]) / 1e9 if "stamp" in d else None
    return {"ext": ext, "pose_in_map_zed": pim, "stamp": stamp, "raw": d}


# --------------------------- cameras-yaml IO ----------------------------
def _fmt_list(vals):
    return "[" + ", ".join("%.10g" % x for x in vals) + "]"


def dump_cameras_yaml(path, map_frame, calib_used, cameras):
    lines = ["# Corrected camera poses, recomputed from the LiDAR trajectory and",
             "# mapped into the board-origin 'map' frame via anchor_frame.json.",
             "map_frame: %s" % map_frame,
             "lidar_camera_extrinsic_xyz_qxyzw: %s" % _fmt_list(calib_used),
             "cameras:"]
    for c in cameras:
        lines.append("  - name: %s" % c["name"])
        lines.append("    parent_frame: %s" % c.get("parent_frame", map_frame))
        lines.append("    child_frame: %s" % c["child_frame"])
        lines.append("    stamp: %.9f" % c["stamp"])
        lines.append("    traj_gap_s: %.6f" % c["traj_gap_s"])
        lines.append("    translation: %s" % _fmt_list(c["translation"]))
        lines.append("    quaternion_xyzw: %s" % _fmt_list(c["quaternion_xyzw"]))
        if "zed_pose_in_map_translation" in c:
            lines.append("    zed_pose_in_map_translation: %s"
                         % _fmt_list(c["zed_pose_in_map_translation"]))
            lines.append("    raw_delta_cm: %.3f" % c["raw_delta_cm"])
            lines.append("    raw_delta_deg: %.3f" % c["raw_delta_deg"])
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def load_cameras_yaml(path):
    map_frame = "map"; cams = []; cur = None
    for ln in open(path):
        s = ln.rstrip("\n")
        if s.startswith("map_frame:"):
            map_frame = s.split(":", 1)[1].strip()
        elif s.strip().startswith("- name:"):
            if cur:
                cams.append(cur)
            cur = {"name": s.split(":", 1)[1].strip()}
        elif cur is not None and ":" in s:
            k, v = s.strip().split(":", 1); k = k.strip(); v = v.strip()
            if v.startswith("["):
                cur[k] = [float(x) for x in v.strip("[]").split(",")]
            elif v:
                try:
                    cur[k] = float(v)
                except ValueError:
                    cur[k] = v
    if cur:
        cams.append(cur)
    return map_frame, cams


# --------------------------- config + calibration -----------------------
DEFAULT_TOPICS = {"points_topic": "/mobile_1/ouster/points",
                  "image_topic": "/mobile_1/zed/left/image_rect_color",
                  "camera_info_topic": "/mobile_1/zed/left/camera_info"}


class Sensor:
    """Everything sensor-specific: intrinsics, image size, T_lidar_camera (pose
    of the camera optical frame in the LiDAR frame) and the ROS topics.

    Sensor.from_bag() reads them from the recording itself -- the camera's
    CameraInfo and the bag's /tf_static -- so no calibration file is needed.
    Sensor(calib) takes a calibration.json dict instead."""
    def __init__(self, calib):
        cam = calib["camera"]; res = calib["results"]; meta = calib.get("meta", {})
        self.fx, self.fy, self.cx, self.cy = cam["intrinsics"]
        dist = cam.get("distortion_coeffs", []) or []
        self.dist = np.array(dist, float) if dist else np.zeros(5)
        self.width = cam.get("width"); self.height = cam.get("height")
        self.T_lidar_camera_7 = list(res["T_lidar_camera"])
        self.T_lidar_camera = make_T(self.T_lidar_camera_7[0:3], self.T_lidar_camera_7[3:7])
        self.points_topic = meta.get("points_topic", DEFAULT_TOPICS["points_topic"])
        self.image_topic = meta.get("image_topic", DEFAULT_TOPICS["image_topic"])
        self.camera_info_topic = meta.get("camera_info_topic",
                                          DEFAULT_TOPICS["camera_info_topic"])
        self.source = "calibration.json"

    @classmethod
    def from_bag(cls, bag, points_topic=None, image_topic=None, camera_info_topic=None):
        """Intrinsics and size from the first CameraInfo (the rectified P when
        set, else K), T_lidar_camera chained through /tf_static from the frame
        the clouds are stamped in to the frame the CameraInfo is stamped in."""
        from pathlib import Path
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
        pt = points_topic or DEFAULT_TOPICS["points_topic"]
        it = image_topic or DEFAULT_TOPICS["image_topic"]
        ct = camera_info_topic or DEFAULT_TOPICS["camera_info_topic"]
        first, edges, T = {}, {}, None
        with AnyReader([Path(bag)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as r:
            have = {c.topic for c in r.connections}
            missing = [t for t in (pt, ct, "/tf_static") if t not in have]
            if missing:
                raise SystemExit("%s: no %s in the bag (needed for the intrinsics and "
                                 "T_lidar_camera); topics: %s"
                                 % (bag, ", ".join(missing), sorted(have)))
            conns = [c for c in r.connections if c.topic in (pt, ct, "/tf_static")]
            for conn, _, raw in r.messages(connections=conns):
                if conn.topic == "/tf_static":
                    for tr in r.deserialize(raw, conn.msgtype).transforms:
                        t = tr.transform.translation; q = tr.transform.rotation
                        edges[(tr.header.frame_id.lstrip("/"), tr.child_frame_id.lstrip("/"))] = \
                            make_T([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
                elif conn.topic not in first:
                    first[conn.topic] = r.deserialize(raw, conn.msgtype)
                if len(first) == 2 and edges:
                    lidar = first[pt].header.frame_id.lstrip("/")
                    camera = first[ct].header.frame_id.lstrip("/")
                    T = lookup_static(edges, lidar, camera)
                    if T is not None:
                        break
        if len(first) < 2:
            raise SystemExit("%s: no messages on %s" % (bag, " / ".join(
                t for t in (pt, ct) if t not in first)))
        if T is None:
            raise SystemExit(
                "%s: /tf_static does not link %s (the %s frame) to %s (the camera); "
                "frames there: %s. Add the calibration to the bag (tf_edit.py --add-file) "
                "or set dataset.calib_json" % (bag, lidar, pt, camera,
                                               sorted({f for e in edges for f in e})))
        info = first[ct]
        P = np.asarray(info.p, float).reshape(3, 4)
        M = P[:3, :3] if P[0, 0] else np.asarray(info.k, float).reshape(3, 3)
        d = list(np.asarray(info.d, float))
        q = R_to_q(T[:3, :3])
        s = cls({"camera": {"intrinsics": [M[0, 0], M[1, 1], M[0, 2], M[1, 2]],
                            "distortion_coeffs": d, "width": int(info.width),
                            "height": int(info.height)},
                 "results": {"T_lidar_camera": list(T[:3, 3]) + list(q)},
                 "meta": {"points_topic": pt, "image_topic": it, "camera_info_topic": ct}})
        s.source = "bag (%s, /tf_static %s -> %s)" % (ct, lidar, camera)
        return s

    def describe(self):
        size = "%dx%d " % (self.width, self.height) if self.width else ""
        return ("sensor from %s: %sfx=%.3f fy=%.3f cx=%.3f cy=%.3f, T_lidar_camera "
                "t=%s q=%s" % (self.source, size, self.fx, self.fy, self.cx, self.cy,
                               np.round(self.T_lidar_camera_7[:3], 6).tolist(),
                               np.round(self.T_lidar_camera_7[3:], 6).tolist()))

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], float)


def lookup_static(edges, src, dst):
    """T_src_dst through {(parent, child): T_parent_child}, edges usable both
    ways (breadth first), or None when the frames are not connected."""
    from collections import deque
    if src == dst:
        return np.eye(4)
    adj = {}
    for (p, c), T in edges.items():
        adj.setdefault(p, []).append((c, T))
        adj.setdefault(c, []).append((p, np.linalg.inv(T)))
    todo, seen = deque([(src, np.eye(4))]), {src}
    while todo:
        node, acc = todo.popleft()
        for nxt, T in adj.get(node, []):
            if nxt in seen:
                continue
            if nxt == dst:
                return acc @ T
            seen.add(nxt)
            todo.append((nxt, acc @ T))
    return None


# ------------------------------ output layout -----------------------------
# Outputs go to data/processed/<date>/<pass>/, mirroring data/raw and
# data/work, split by what they are. A bare file name is placed by the first
# pattern it matches; {machine} is the robot whose LiDAR the map stages use.
#   mapping/            clouds: merged, static, denoised, colored, map_final, cut, ...
#   odometry/<machine>/ trajectories: traj_lidar_refined*.txt, *.tum
#   odometry/           reference tracks (08) of every robot
#   bags/               bags written by the stages (09's best-poses bag)
#   frames/             anchors, camera poses, TF scripts: anchor_frame.json, ...
#   comms/              wifi / iperf / ntp results (no stage writes here yet)
ROUTES = (
    (("traj*", "*.tum"), "odometry/{machine}"),
    (("reference_*",), "odometry"),
    (("*.pcd", "*.ply", "objects*"), "mapping"),
    (("*.mcap", "*_poses", "*_bag"), "bags"),
    (("*",), "frames"),
)


def processed_dir_for(path):
    """data/{raw,work,processed}/<date>/<pass>/... -> data/processed/<date>/<pass>.
    The path is not resolved, so a data folder reached through a symlink stays
    where it is."""
    parts = os.path.abspath(os.path.expanduser(path)).split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in ("raw", "work", "processed") and len(parts) > i + 3:
            return os.sep.join(parts[:i] + ["processed", parts[i + 1], parts[i + 2]])
    raise SystemExit("%s is not below data/{raw,work}/<date>/<pass>/, so the "
                     "processed folder cannot be derived from it; set "
                     "dataset.processed (or dataset.out_dir for one flat folder)" % path)


def topic_machine(topic):
    """'/mobile_1/ouster/points' -> 'mobile_1'."""
    return topic.strip("/").split("/")[0]


class Pipeline:
    def __init__(self, cfg, cfg_dir):
        self.cfg = cfg
        self.cfg_dir = cfg_dir
        d = self.dataset = dict(cfg["dataset"])
        for k in ("bag", "traj", "calib_json", "out_dir", "processed"):
            if isinstance(d.get(k), str) and d[k]:
                d[k] = os.path.expanduser(d[k])
        self.machine = d.get("machine") or topic_machine(
            d.get("points_topic") or DEFAULT_TOPICS["points_topic"])
        if d.get("out_dir"):                 # the old layout: one flat folder
            self.out_dir, self.routed = d["out_dir"], False
        else:
            self.out_dir = d.get("processed") or processed_dir_for(d["bag"])
            self.routed = True
        os.makedirs(self.out_dir, exist_ok=True)
        self._sensor = None

    @property
    def sensor(self):
        """From dataset.calib_json when it is set, else read from dataset.bag
        (camera_info + /tf_static); dataset.points_topic / image_topic /
        camera_info_topic override the default topics. Resolved on first use."""
        if self._sensor is None:
            d = self.dataset
            if d.get("calib_json"):
                self._sensor = Sensor(json.load(open(d["calib_json"])))
            else:
                self._sensor = Sensor.from_bag(d["bag"], d.get("points_topic"),
                                               d.get("image_topic"),
                                               d.get("camera_info_topic"))
            print(self._sensor.describe())
        return self._sensor

    def folder_for(self, name):
        """The folder a bare output name belongs in (see ROUTES)."""
        if not self.routed:
            return self.out_dir
        for pats, sub in ROUTES:
            if any(fnmatch(name, p) for p in pats):
                return os.path.join(self.out_dir, sub.format(machine=self.machine))
        return self.out_dir

    def outp(self, name):
        """A bare file name -> its place in the layout (folder created); a path
        with a folder in it (absolute, ~/..., or relative) is used as given."""
        if os.path.isabs(name) or os.path.dirname(name):
            return os.path.expanduser(name)
        d = self.folder_for(name)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, name)

    def describe(self):
        if not self.routed:
            return "outputs -> %s/" % self.out_dir
        return ("outputs -> %s/  (mapping/, odometry/%s/, frames/, bags/)"
                % (self.out_dir, self.machine))

    def stage(self, key):
        """Return this stage's dict with any *file* fields resolved against the layout."""
        s = dict(self.cfg[key])
        for k in ("input", "output", "frame_out", "anchor_frame", "script_out"):
            if k in s and isinstance(s[k], str):
                s[k] = self.outp(s[k])
        if "cameras" in s:
            out = []
            for c in s["cameras"]:
                c = dict(c)
                if isinstance(c.get("extrinsic_yaml"), str):
                    c["extrinsic_yaml"] = self.outp(c["extrinsic_yaml"])
                out.append(c)
            s["cameras"] = out
        return s


def load_pipeline(path="pipeline_config.json"):
    cfg = json.load(open(path))
    return Pipeline(cfg, os.path.dirname(os.path.abspath(path)))

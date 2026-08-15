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
class Sensor:
    """Everything sensor-specific, resolved once from calibration.json."""
    def __init__(self, calib):
        cam = calib["camera"]; res = calib["results"]; meta = calib.get("meta", {})
        self.fx, self.fy, self.cx, self.cy = cam["intrinsics"]
        dist = cam.get("distortion_coeffs", []) or []
        self.dist = np.array(dist, float) if dist else np.zeros(5)
        self.T_lidar_camera_7 = res["T_lidar_camera"]
        self.T_lidar_camera = make_T(self.T_lidar_camera_7[0:3], self.T_lidar_camera_7[3:7])
        self.points_topic = meta.get("points_topic", "/ouster/points")
        self.image_topic = meta.get("image_topic", "/zed/zed_node/left/image_rect_color")
        self.camera_info_topic = meta.get("camera_info_topic",
                                          "/zed/zed_node/left/camera_info")

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], float)


class Pipeline:
    def __init__(self, cfg, cfg_dir):
        self.cfg = cfg
        self.cfg_dir = cfg_dir
        self.dataset = cfg["dataset"]
        self.out_dir = self.dataset["out_dir"]
        self.sensor = Sensor(json.load(open(self.dataset["calib_json"])))
        os.makedirs(self.out_dir, exist_ok=True)

    def outp(self, name):
        """Resolve a bare filename against out_dir (absolute/with-dir passes through)."""
        if os.path.isabs(name) or os.path.dirname(name):
            return name
        return os.path.join(self.out_dir, name)

    def stage(self, key):
        """Return this stage's dict with any *file* fields resolved against out_dir."""
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

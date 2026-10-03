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


# ---------------------- point times, deskew --------------------------
# ROS PointField datatype enum -> numpy
_PF_DT = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
          5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}
# per-point time field, in the order drivers commonly name it
_T_NAMES = ("t", "time", "timestamp", "time_offset", "point_time", "ts")
_T_INFO = {}          # detected once, then reused (and printed once)


def pc2_xyzt(msg):
    """(N,3) xyz plus (N,) per-point time in SECONDS RELATIVE to the frame
    stamp, or None when the cloud carries no usable time field.

    Handles both conventions: relative offsets (Ouster 't', uint32 ns) and
    absolute per-point stamps (float64 s). The scale is detected from the data
    rather than trusted from config -- a wrong guess is silent, it just
    produces a differently-shaped smear."""
    off = {f.name: (f.offset, f.datatype) for f in msg.fields}
    raw = np.frombuffer(bytes(msg.data), np.uint8).reshape(-1, msg.point_step)

    def col(name):
        o, dt = off[name]
        t = _PF_DT[dt]
        w = np.dtype(t).itemsize
        return raw[:, o:o + w].copy().view(t).ravel()

    xyz = np.stack([col("x").astype(np.float32),
                    col("y").astype(np.float32),
                    col("z").astype(np.float32)], 1)

    name = next((n for n in _T_NAMES if n in off), None)
    if name is None:
        if "scale" not in _T_INFO:
            _T_INFO.update(scale=0.0, absolute=False, field=None)
            print(f"    [deskew] no per-point time field (fields: "
                  f"{sorted(off)}) -> deskew DISABLED")
        return xyz, None

    tt = col(name).astype(np.float64)
    if "scale" not in _T_INFO:
        # a LiDAR sweep is 50-200 ms; pick the scale that lands the span there
        span = float(tt.max() - tt.min())
        sc = 0.0
        if span > 0:
            for cand in (1e-9, 1e-6, 1e-3, 1.0):
                if 0.005 < span * cand < 1.0:
                    sc = cand
                    break
        _T_INFO.update(scale=sc, field=name,
                       absolute=bool(sc and tt.min() * sc > 1e6))
        if sc:
            print(f"    [deskew] time field '{name}', scale {sc:g}, "
                  f"{'absolute' if _T_INFO['absolute'] else 'relative'}, "
                  f"sweep {span * sc * 1e3:.1f} ms")
        else:
            print(f"    [deskew] time field '{name}' has an unusable span "
                  f"({span:g}) -> deskew DISABLED")
    if not _T_INFO["scale"]:
        return xyz, None
    tt = tt * _T_INFO["scale"]
    if _T_INFO["absolute"]:
        tt = tt - (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
    return xyz, tt.astype(np.float64)


def pc2_xyz(msg):
    """Geometry only, for callers that do not need the point times."""
    return pc2_xyzt(msg)[0]


def traj_quats(tr_T):
    """(M,4) xyzw quaternions for the trajectory rotations, computed once."""
    R = tr_T[:, :3, :3]
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    q = np.zeros((len(R), 4))
    b0 = tr > 0
    b1 = (~b0) & (m00 >= m11) & (m00 >= m22)
    b2 = (~b0) & (~b1) & (m11 >= m22)
    b3 = ~(b0 | b1 | b2)
    if b0.any():
        s = np.sqrt(np.maximum(1e-12, tr[b0] + 1.0)) * 2
        q[b0] = np.stack([(R[b0, 2, 1] - R[b0, 1, 2]) / s,
                          (R[b0, 0, 2] - R[b0, 2, 0]) / s,
                          (R[b0, 1, 0] - R[b0, 0, 1]) / s, 0.25 * s], 1)
    for b, (i, j, k) in ((b1, (0, 1, 2)), (b2, (1, 2, 0)), (b3, (2, 0, 1))):
        if not b.any():
            continue
        s = np.sqrt(np.maximum(1e-12,
                               1.0 + R[b, i, i] - R[b, j, j] - R[b, k, k])) * 2
        qq = np.empty((int(b.sum()), 4))
        qq[:, i] = 0.25 * s
        qq[:, j] = (R[b, j, i] + R[b, i, j]) / s
        qq[:, k] = (R[b, k, i] + R[b, i, k]) / s
        qq[:, 3] = (R[b, k, j] - R[b, j, k]) / s
        q[b] = qq
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def quats_to_R(q):
    """(K,4) xyzw -> (K,3,3)."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)
    ], 1).reshape(-1, 3, 3)


def interp_poses(tr_t, tr_T, tr_q, times):
    """Trajectory poses at arbitrary `times`: (K,3,3) rotations, (K,3)
    translations. Slerp on rotation, linear on translation, clamped at the
    ends. Vectorised over K, so a whole scan's bins cost one call."""
    j = np.clip(np.searchsorted(tr_t, times), 1, len(tr_t) - 1)
    t0, t1 = tr_t[j - 1], tr_t[j]
    u = np.clip((times - t0) / np.maximum(t1 - t0, 1e-9), 0.0, 1.0)
    q0 = tr_q[j - 1].copy()
    q1 = tr_q[j].copy()
    d = np.sum(q0 * q1, 1)
    q1[d < 0] *= -1.0                        # shortest arc
    d = np.abs(d)
    q = q0 + u[:, None] * (q1 - q0)          # lerp; exact enough when parallel
    slow = d <= 0.9995
    if slow.any():
        th = np.arccos(np.clip(d[slow], -1.0, 1.0))[:, None]
        us = u[slow][:, None]
        q[slow] = (np.sin((1 - us) * th) * q0[slow]
                   + np.sin(us * th) * q1[slow]) / np.sin(th)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    p = tr_T[j - 1][:, :3, 3] + u[:, None] * (tr_T[j][:, :3, 3]
                                              - tr_T[j - 1][:, :3, 3])
    return quats_to_R(q), p

def deskew_bins(dt, t, tr_t, tr_T, tr_q, nb=100, shift=0.0):
    """The one deskew of the pipeline: a sweep cut into `nb` time slices,
    each point through the trajectory pose at its slice's centre.

    dt (N,) are the points' times after the cloud stamp t (s); the
    trajectory's stamps may be moved by `shift` s. -> (Rb (B,3,3), tb (B,3),
    idx (N,)): point i's world position is Rb[idx[i]] @ p_i + tb[idx[i]].
    100 slices over 100 ms leave 1 ms, about a millimetre at walking speed.
    A single-shot cloud (no spread in dt) gets one pose."""
    ts = tr_t + shift if shift else tr_t
    lo, hi = float(dt.min()), float(dt.max())
    if hi - lo < 1e-4:
        Rb, tb = interp_poses(ts, tr_T, tr_q, np.array([t + lo]))
        return Rb, tb, np.zeros(len(dt), np.int64)
    edges = np.linspace(lo, hi, nb + 1)
    Rb, tb = interp_poses(ts, tr_T, tr_q, t + 0.5 * (edges[:-1] + edges[1:]))
    idx = np.clip(((dt - lo) / (hi - lo) * nb).astype(np.int64), 0, nb - 1)
    return Rb, tb, idx


def deskew_to_world(p, dt, t, tr_t, tr_T, tr_q, nb=100, shift=0.0):
    """One scan's points in the world: deskewed (deskew_bins) when the point
    times dt are given, else the whole scan through the pose at its stamp."""
    if dt is None:
        ts = tr_t + shift if shift else tr_t
        Rb, tb = interp_poses(ts, tr_T, tr_q, np.array([t]))
        return p @ Rb[0].T + tb[0]
    Rb, tb, idx = deskew_bins(dt, t, tr_t, tr_T, tr_q, nb, shift)
    return np.einsum("nij,nj->ni", Rb[idx], p) + tb[idx]


def deskew_to_pose(p, dt, t, j, tr_t, tr_T, tr_q, nb=100):
    """One scan's points into the sensor frame of trajectory pose j: each
    point through the pose at its own time and back through pose j,
        p_j = T_j^-1 T(t + dt) p
    (deskew_bins) -- what a rigid registration of the scan then solves for."""
    Rb, tb, idx = deskew_bins(dt, t, tr_t, tr_T, tr_q, nb)
    Rj, tj = tr_T[j][:3, :3], tr_T[j][:3, 3]
    Rr = np.einsum("ji,bjk->bik", Rj, Rb)          # Rj^T Rb
    tr = (tb - tj) @ Rj                             # Rj^T (tb - tj), as rows
    return np.einsum("nij,nj->ni", Rr[idx], p) + tr[idx]


def nearest_idx(tr_t, t):
    """Index of the trajectory stamp nearest t."""
    i = int(np.searchsorted(tr_t, t))
    if i <= 0:
        return 0
    if i >= len(tr_t):
        return len(tr_t) - 1
    return i if (tr_t[i] - t) < (t - tr_t[i - 1]) else i - 1


def iter_scans(bag, topic, tr_t, tol, lo, hi, tr_T=None, deskew=False, nb=100,
               frame="pose", min_points=100, verbose=True):
    """Every scan of `topic` with a trajectory pose within `tol` s of its
    stamp: yields (pose index j, points), points in range (lo, hi) m.

    frame "pose": in the sensor frame of pose j (deskew_to_pose with deskew,
                  else as recorded) -- for registration
    frame "world": in the world through the trajectory (deskew_to_world: each
                  point at its own time with deskew, else the scan through the
                  pose at its stamp) -- for building clouds
    Both need tr_T for deskew; "world" always does."""
    from pathlib import Path
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    if frame not in ("pose", "world"):
        raise ValueError(frame)
    tr_q = traj_quats(tr_T) if tr_T is not None else None
    if (deskew or frame == "world") and tr_q is None:
        raise ValueError("iter_scans: deskew / frame='world' need tr_T")
    told = [False]
    with AnyReader([Path(bag)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as r:
        conns = [c for c in r.connections if c.topic == topic and "PointCloud2" in c.msgtype]
        if not conns:
            have = sorted({c.topic for c in r.connections})
            raise SystemExit(f"topic {topic!r} not in bag; available: {have}")
        for conn, _, raw in r.messages(connections=conns):
            msg = r.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            j = nearest_idx(tr_t, t)
            if abs(tr_t[j] - t) > tol:
                continue
            p, dt = pc2_xyzt(msg)
            p = p.astype(np.float64)
            d = np.linalg.norm(p, axis=1)
            keep = np.isfinite(p).all(1) & (d > lo) & (d < hi)
            if keep.sum() < min_points:
                continue
            p = p[keep]
            dt = dt[keep] if (deskew and dt is not None) else None
            if verbose and not told[0] and deskew:
                told[0] = True
                print(f"    [deskew] {'ON, %d bins' % nb if dt is not None else 'no point times: OFF'}"
                      f" (each point through the trajectory at its own time)")
            if frame == "world":
                yield j, deskew_to_world(p, dt, t, tr_t, tr_T, tr_q, nb)
            elif dt is not None:
                yield j, deskew_to_pose(p, dt, t, j, tr_t, tr_T, tr_q, nb)
            else:
                yield j, p


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
        lines.append("    source: %s" % c.get("source", "?"))
        lines.append("    stamp: %.9f" % c["stamp"])
        if "traj_gap_s" in c:                       # extrinsic_yaml cameras
            lines.append("    traj_gap_s: %.6f" % c["traj_gap_s"])
        for k in ("board", "n_detections", "spread_std_mm", "spread_max_mm",
                  "spread_max_deg", "mean_reproj_px"):  # board cameras
            if k in c:
                lines.append("    %s: %s" % (k, c[k]))
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
#   odometry/           reference tracks (04) of every robot
#   bags/               bags written by the stages (10's best-poses bag)
#   frames/             anchors, camera poses, TF scripts: anchor_frame.json, ...
#   comms/              wifi / iperf / ntp results (no stage writes here yet)
#   datasets/<machine>/ <format>_<depth>_<pass>_<date> (datasets/mcap_convert.py)
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
        # names the outputs: <pass>_<date> from data/processed/<date>/<pass>,
        # or dataset.name; none for the old flat folder
        date, pas = os.path.normpath(self.out_dir).split(os.sep)[-2:]
        self.tag = d.get("name") or ("%s_%s" % (pas, date) if self.routed else "")
        os.makedirs(self.out_dir, exist_ok=True)
        # files recorded during the session, next to its raw data:
        # data/raw/<date>/<pass>/ (infra_cameras/ holds the camera extrinsic yamls)
        self.raw_dir = (os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.normpath(self.out_dir)))), "raw", date, pas) if self.routed else None)
        # a run localized in another pass's map (08_localize): dataset.reference_pass
        # is that pass ("mapping_A", same date, or "<date>/<pass>")
        self.reference = d.get("reference_pass") or None
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

    def pcd(self, base):
        """The name of one of the stages' clouds: 'denoised' ->
        denoised_<tag>.pcd (denoised.pcd when there is no tag)."""
        return "%s_%s.pcd" % (base, self.tag) if self.tag else base + ".pcd"

    def outp(self, name):
        """A bare file name -> its place in the layout (folder created); a path
        with a folder in it (absolute, ~/..., or relative) is used as given.
        {tag} in a name becomes the pass tag (map_final_{tag}.pcd)."""
        if "{tag}" in name:
            name = name.replace("{tag}", self.tag) if self.tag else \
                name.replace("_{tag}", "").replace("{tag}", "")
        if os.path.isabs(name) or os.path.dirname(name):
            return os.path.expanduser(name)
        d = self.folder_for(name)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, name)

    def ref_file(self, kind):
        """A file of the reference pass (dataset.reference_pass), in its
        processed folder: 'map' -> mapping/denoised_<tag>.pcd (GLIM frame),
        'anchored_map' -> mapping/map_final_<tag>_anchored.pcd (05's output,
        the map a run is localized in), 'anchor' -> frames/anchor_frame.json
        (its board-anchored map frame)."""
        if not self.reference:
            raise SystemExit("dataset.reference_pass is not set")
        root = os.path.dirname(os.path.dirname(os.path.normpath(self.out_dir)))
        ref = self.reference.strip("/")
        if "/" not in ref:
            ref = os.path.join(os.path.basename(os.path.dirname(
                os.path.normpath(self.out_dir))), ref)
        date, pas = ref.split("/")[-2:]
        d = os.path.join(root, date, pas)
        if kind == "map":
            return os.path.join(d, "mapping", "denoised_%s_%s.pcd" % (pas, date))
        if kind == "anchored_map":
            return os.path.join(d, "mapping", "map_final_%s_%s_anchored.pcd" % (pas, date))
        if kind == "anchor":
            return os.path.join(d, "frames", "anchor_frame.json")
        raise ValueError(kind)

    def anchor_frame(self, name="anchor_frame.json"):
        """This pass's anchor_frame.json, or the reference pass's for a run
        (dataset.reference_pass) that has none of its own."""
        p = self.outp(name)
        if self.reference and not os.path.exists(p):
            return self.ref_file("anchor")
        return p

    def glim_traj(self):
        """The run's own GLIM trajectory: <work>/<date>/<pass>/<machine>/glim/
        traj_lidar.txt, next to the merged bag (process_recording.py writes it)."""
        return os.path.join(os.path.dirname(os.path.normpath(self.dataset["bag"])),
                            self.machine, "glim", "traj_lidar.txt")

    def seed_traj(self):
        """The trajectory a mapping pass starts from (01 --seed, 02's seed):
        02_refine.seed when set (a bare name is in odometry/<machine>/), else
        odometry/<machine>/traj_lidar.txt when there, else GLIM's own file."""
        name = (self.cfg.get("02_refine") or {}).get("seed")
        if name:
            return self.outp(name)
        local = self.outp("traj_lidar.txt")
        return local if os.path.exists(local) else self.glim_traj()

    def reference_dir(self):
        """Stage 04's output folder: odometry/reference_<tag>/."""
        return self.outp(self.cfg.get("04_reference", {}).get("out_dir", "reference_{tag}"))

    def lidar_track_traj(self, suffix=""):
        """Stage 04's LiDAR trajectory (T_map_lidar, TUM): traj_<name>.tum in
        reference_dir(), name = 04_reference.name or <machine>_lidar; suffix
        "_in_cam" for the camera optical frame."""
        name = (self.cfg.get("04_reference") or {}).get("name") or "%s_lidar" % self.machine
        return os.path.join(self.reference_dir(), "traj_%s%s.tum" % (name, suffix))

    def infra_yaml(self, name, camera=None):
        """An infra camera's extrinsic yaml from the mapping session: a bare
        name is looked up in data/raw/<date>/<pass>/infra_cameras/<camera>/
        (the camera's own folder, with its recordings), then infra_cameras/,
        then frames/ (where older runs kept it); a path is used as given. When
        it is in none, the infra_cameras/<camera>/ path is returned, so the
        message says where to put it."""
        if os.path.isabs(name) or os.path.dirname(name):
            return os.path.expanduser(name)
        cands = []
        if self.raw_dir:
            ic = os.path.join(self.raw_dir, "infra_cameras")
            cands += ([os.path.join(ic, camera, name)] if camera else []) + [os.path.join(ic, name)]
        cands.append(os.path.join(self.folder_for(name), name))
        return next((c for c in cands if os.path.exists(c)), cands[0])

    def describe(self):
        if not self.routed:
            return "outputs -> %s/" % self.out_dir
        return ("outputs -> %s/  (mapping/, odometry/%s/, frames/, bags/), "
                "named *_%s" % (self.out_dir, self.machine, self.tag))

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
                    c["extrinsic_yaml"] = self.infra_yaml(c["extrinsic_yaml"], c.get("name"))
                out.append(c)
            s["cameras"] = out
        return s


def _merge(base, over):
    """`over` on top of `base`: dicts merged key by key; lists of named dicts
    (tracks, sensors, robots, cameras) merged by "name" -- an entry overrides
    the base entry of that name, {"name": n, "enabled": false} switches it
    off, a new name is added; anything else is replaced."""
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = _merge(base[k], v) if k in base else v
        return out
    named = (isinstance(base, list) and isinstance(over, list) and over
             and all(isinstance(x, dict) and "name" in x for x in base + over))
    if named:
        out = [dict(x) for x in base]
        idx = {x["name"]: i for i, x in enumerate(out)}
        for x in over:
            if x["name"] in idx:
                out[idx[x["name"]]] = _merge(out[idx[x["name"]]], x)
            else:
                out.append(dict(x))
        return out
    return over


# stage blocks under the numbers they had before the stages were put in run order
OLD_KEYS = {"01a_refine": "02_refine", "06_init": "03_init", "08_reference": "04_reference",
            "03_anchor": "05_anchor", "02_cut": "06_cut", "04_build_cameras": "07_build_cameras",
            "05_emit_tfs": "08_emit_tfs", "07_build": "09_build", "09_publish": "10_publish"}


def load_config(path):
    """A pipeline config; "extends": "<other config>" (relative to this one)
    starts from that one, so a run's config holds only what differs. A block
    under its old stage number (OLD_KEYS) is read as the new one."""
    cfg = json.load(open(path))
    for k in [k for k in cfg if k in OLD_KEYS]:
        if OLD_KEYS[k] in cfg:
            raise SystemExit("%s has both %s and %s (its old name)" % (path, OLD_KEYS[k], k))
        cfg[OLD_KEYS[k]] = cfg.pop(k)
    parent = cfg.pop("extends", None)
    if parent:
        parent = os.path.join(os.path.dirname(os.path.abspath(path)),
                              os.path.expanduser(parent))
        cfg = _merge(load_config(parent), cfg)
    return cfg


def load_pipeline(path="pipeline_config.json"):
    cfg = load_config(path)
    return Pipeline(cfg, os.path.dirname(os.path.abspath(path)))

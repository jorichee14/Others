#!/usr/bin/env python3
"""
mcap_convert.py — convert a ROS 2 / MCAP recording into SLAM dataset formats.

Output formats
==============
replica   RGB-D sequence for SplaTAM / NICE-SLAM / Point-SLAM / GradSLAM
            <out>/results/frame000000.jpg    colour (JPEG)
            <out>/results/depth000000.png    uint16 depth, metres = pixel / png_depth_scale
                                             (6553.5 as Replica: 0..65535 = 0..10 m)
            <out>/traj.txt                   camera-to-world 4x4 per frame (row-major,
                                             OpenCV axes: x right, y down, z forward)
            <out>/splatam_data_config.yaml   drop-in SplaTAM data config
            <out>/report.txt                 sanity statistics
          Depth comes from the LiDAR by default (--depth-source lidar): the scans
          nearest each image are moved into the camera at the image time (through
          the poses, point by point) and projected; 0 = no point. --depth-source map
          renders the built map (--map) instead, dense, as surfels; --depth-source
          zed uses the ZED depth image. --pose-tum takes the poses from a TUM file
          (the refined trajectory, of the LiDAR frame) instead of a pose topic.

mcd       MCD dataset layout (mcdviral.github.io), sensors described by a config
            <out>/<seq>_merged.bag           ROS1 bag, all sensors, MCD topic names
            <out>/groundtruth/pose_inW.csv   num t x y z qx qy qz qw (body pose in world)
            <out>/gt_tum.txt                 same poses, TUM format (evo)
            <out>/calibration.yaml           per sensor: topic, intrinsics, T_body_sensor
            <out>/camera.yaml imu.yaml lidar.yaml   MCD/CLIC style, extrinsics sensor -> IMU

mcgs      Stereo sequence for MCGS-SLAM (github.com/mcgs-slam/mcgs-slam)
            <out>/left/<t>.jpg <out>/right/<t>.jpg   file name = timestamp in seconds
            <out>/calib.yml                          intrinsic, baseline, T_cami_cam0
            <out>/gt_poses.txt                       left camera in world, TUM format

Command line
============
    python3 mcap_convert.py <bag dir | .mcap> <out dir> --format replica --pose-is-camera
    python3 mcap_convert.py <bag dir | .mcap> <out dir> --format replica --pose-is-camera \\
        --lidar-scans 5                                   # denser lidar depth
    python3 mcap_convert.py <bag dir | .mcap> <out dir> --format replica \\
        --pose-tum traj_lidar_refined.txt --depth-source map --map denoised_<tag>.pcd
    python3 mcap_convert.py <bag dir | .mcap> <out dir> --format mcd [--config my.yaml]
    python3 mcap_convert.py <bag dir | .mcap> <out dir> --format mcd \\
        --pose-tum traj_lidar_refined.txt [--depth-source lidar|map --map denoised_<tag>.pcd]
    python3 mcap_convert.py <bag dir | .mcap> <out dir> --format mcgs
    python3 mcap_convert.py <bag dir | .mcap> --inspect        # intrinsics, frames, topics

Python
======
    import mcap_convert as mc
    mc.to_replica("bag/", "out/rep", pose_is_camera=True, every=2)
    mc.to_replica("bag/", "out/rep", pose_is_camera=True, depth_source="zed")
    mc.to_mcd("bag/", "out/mcd", config="my_sensors.yaml")
    mc.to_mcgs("bag/", "out/mcgs", max_frames=500)

    # building blocks
    paths = mc.bag_files("bag/")
    for t_left, left_msg, t_right, right_msg in mc.iter_synced(paths, "/left", "/right", 10e6):
        img = mc.to_bgr(mc.decode_image(left_msg), left_msg.encoding)
    poses = mc.PoseInterp.from_topic(paths, "/mobile_1/global_pose")
    T_world_cam = poses.at(t_left)
    xyz, t = mc.cloud_points(cloud_msg)              # PointCloud2 -> Nx3, point times
    depth = mc.project_depth(xyz_in_camera, K)       # -> HxW metres, 0 = none

Every converter returns a dict summary and raises mc.ConvertError on bad input.

Requirements
============
    pip install mcap mcap-ros2-support numpy opencv-python scipy pyyaml
    pip install rosbags          # only for the mcd format (writes ROS1 bags, no ROS needed)
No ROS installation is needed: message definitions are read from the MCAP itself.
Do not save this file as mcap.py — it would hide the 'mcap' package it imports.
"""

import argparse
import glob
import heapq
import logging
import os
import re
import sys
from collections import deque, namedtuple

import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

if os.path.splitext(os.path.basename(__file__))[0] in ("mcap", "mcap_ros2"):
    raise ImportError(f"rename this file: '{os.path.basename(__file__)}' hides the real "
                      "'mcap' package it imports (e.g. mv mcap.py mcap_convert.py, "
                      "and delete __pycache__)")
try:
    from mcap_ros2.reader import read_ros2_messages
except ImportError as _e:
    raise ImportError(f"could not import the MCAP reader ({_e}); "
                      "pip install mcap mcap-ros2-support") from _e

__all__ = [
    # converters
    "to_replica", "to_mcd", "to_mcgs", "inspect_bag", "FORMATS", "ConvertError",
    # reading
    "bag_files", "read", "first_msg", "stamp_ns", "iter_synced",
    # images
    "decode_image", "to_bgr", "depth_to_metres",
    # lidar depth
    "cloud_points", "project_depth", "iter_lidar_depth",
    # geometry / calibration
    "mat_from_pos_quat", "xyzq_to_T", "T_to_xyzq", "Intrinsics", "intrinsics_from_info",
    "stereo_baseline",
    # tf
    "build_tf_static", "lookup_static", "tf_fixed_edges", "camera_extrinsic",
    # poses
    "load_poses", "PoseInterp", "tum_line", "write_tum",
    # naming
    "pass_of", "dataset_dir", "topic_machine",
    # mcd building blocks
    "DEFAULT_MCD_CONFIG", "load_mcd_config", "resolve_extrinsic",
]

log = logging.getLogger("mcap_convert")


class ConvertError(RuntimeError):
    """Bad input: missing topic, missing calibration, nothing to write, ..."""


# =========================================================================== #
# 1. Reading the recording
# =========================================================================== #

def _natural_key(path):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", os.path.basename(path))]


def bag_files(path):
    """A .mcap file, or a rosbag2 directory -> its .mcap paths in recording order
    (metadata.yaml's order; without one, _2 before _10, not the plain sort)."""
    if os.path.isdir(path):
        files = []
        meta = os.path.join(path, "metadata.yaml")
        if os.path.exists(meta):
            import yaml
            with open(meta) as f:
                info = (yaml.safe_load(f) or {}).get("rosbag2_bagfile_information") or {}
            files = [os.path.join(path, p) for p in info.get("relative_file_paths") or []
                     if p.endswith(".mcap") and os.path.exists(os.path.join(path, p))]
        if not files:
            files = sorted(glob.glob(os.path.join(path, "*.mcap")), key=_natural_key)
        if not files:
            raise ConvertError(f"no .mcap files in {path}")
        return files
    if not os.path.exists(path):
        raise ConvertError(f"not found: {path}")
    return [path]


def _paths(bag):
    """Accept a path or an already expanded list of .mcap files."""
    return list(bag) if isinstance(bag, (list, tuple)) else bag_files(bag)


def read(paths, topics):
    """Yield decoded messages (mcap_ros2 objects: .ros_msg, .channel.topic, ...)
    on `topics`, file by file in time order."""
    for f in paths:
        yield from read_ros2_messages(f, topics=topics)


def first_msg(paths, topic):
    """First message on `topic` (the ROS message itself) or None."""
    for m in read(paths, [topic]):
        return m.ros_msg
    return None


def stamp_ns(msg):
    """Header stamp of a message in integer nanoseconds."""
    s = msg.header.stamp
    return int(s.sec) * 1_000_000_000 + int(s.nanosec)


def iter_synced(paths, topic_a, topic_b, tol_ns):
    """Pair two streams by header stamp. Yields (t_a, msg_a, t_b, msg_b) for
    messages whose stamps differ by at most `tol_ns`; unmatched ones are dropped."""
    abuf, bbuf = deque(), deque()
    for m in read(paths, [topic_a, topic_b]):
        msg = m.ros_msg
        (abuf if m.channel.topic == topic_a else bbuf).append((stamp_ns(msg), msg))
        while abuf and bbuf:
            ta, ma = abuf[0]
            tb, mb = bbuf[0]
            if ta - tb > tol_ns:
                bbuf.popleft()
            elif tb - ta > tol_ns:
                abuf.popleft()
            else:
                abuf.popleft()
                bbuf.popleft()
                yield ta, ma, tb, mb


# =========================================================================== #
# 2. Images
# =========================================================================== #

ENCODINGS = {
    "mono8": (np.uint8, 1), "8UC1": (np.uint8, 1),
    "bgr8": (np.uint8, 3), "rgb8": (np.uint8, 3), "8UC3": (np.uint8, 3),
    "bgra8": (np.uint8, 4), "rgba8": (np.uint8, 4), "8UC4": (np.uint8, 4),
    "mono16": (np.uint16, 1), "16UC1": (np.uint16, 1),
    "32FC1": (np.float32, 1),
}


def decode_image(msg):
    """sensor_msgs/Image -> HxW or HxWxC numpy array (no colour conversion)."""
    enc = msg.encoding
    if enc not in ENCODINGS:
        raise ConvertError(f"unsupported image encoding: {enc}")
    dtype, ch = ENCODINGS[enc]
    if msg.is_bigendian and np.dtype(dtype).itemsize > 1:
        dtype = np.dtype(dtype).newbyteorder(">")
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    arr = raw.view(dtype).reshape(msg.height, msg.step // np.dtype(dtype).itemsize)
    arr = arr[:, : msg.width * ch]
    return arr.reshape(msg.height, msg.width, ch) if ch > 1 else arr


def to_bgr(img, encoding):
    """Any 8-bit colour/mono image -> 3-channel BGR (OpenCV order)."""
    if encoding == "rgb8":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if encoding in ("bgra8", "8UC4"):
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if encoding in ("mono8", "8UC1"):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img  # bgr8 / 8UC3


def depth_to_metres(img, encoding, input_scale=0.001, dmin=0.0, dmax=np.inf):
    """Depth image -> (float32 metres with invalid = 0, valid mask).
    Integer depth (16UC1/mono16) is multiplied by `input_scale` (0.001 = mm)."""
    d = img.astype(np.float32)
    if encoding in ("16UC1", "mono16"):
        d *= input_scale
    valid = np.isfinite(d) & (d >= dmin) & (d <= dmax)
    return np.where(valid, d, 0.0).astype(np.float32), valid


# =========================================================================== #
# 3. Geometry and calibration
# =========================================================================== #

def mat_from_pos_quat(p, q_xyzw):
    """Position + quaternion (x, y, z, w) -> 4x4 matrix."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = p
    return T


def xyzq_to_T(v):
    """[x, y, z, qx, qy, qz, qw] -> 4x4 matrix."""
    v = np.asarray(v, dtype=float)
    if v.shape != (7,):
        raise ConvertError(f"extrinsic must be [x, y, z, qx, qy, qz, qw], got {v.tolist()}")
    return mat_from_pos_quat(v[:3], v[3:7])


def T_to_xyzq(T):
    """4x4 matrix -> [x, y, z, qx, qy, qz, qw]."""
    return list(T[:3, 3]) + list(Rotation.from_matrix(T[:3, :3]).as_quat())


Intrinsics = namedtuple("Intrinsics", "fx fy cx cy width height dist P")
Intrinsics.k4 = property(lambda s: (s.fx, s.fy, s.cx, s.cy))
Intrinsics.__doc__ = ("Pinhole intrinsics from a CameraInfo. Uses the projection "
                      "matrix P (the rectified camera) when set, else K. dist = first "
                      "4 distortion coefficients (k1 k2 p1 p2), P = 3x4 array.")


def intrinsics_from_info(info):
    """sensor_msgs/CameraInfo -> Intrinsics."""
    P = np.asarray(info.p, dtype=float).reshape(3, 4)
    K = np.asarray(info.k, dtype=float).reshape(3, 3)
    M = P[:3, :3] if P[0, 0] != 0 else K
    d = np.asarray(info.d, dtype=float)
    return Intrinsics(M[0, 0], M[1, 1], M[0, 2], M[1, 2], int(info.width),
                      int(info.height), (list(d[:4]) + [0.0] * 4)[:4], P)


def stereo_baseline(right_info):
    """Baseline in metres from a rectified right CameraInfo (P[0,3] = -fx * B),
    or None when P carries no baseline."""
    P = np.asarray(right_info.p, dtype=float).reshape(3, 4)
    B = -P[0, 3] / P[0, 0] if P[0, 0] else 0.0
    return float(B) if B > 0 else None


# =========================================================================== #
# 4. Transforms (/tf_static, fixed /tf links)
# =========================================================================== #

def _tf_edge(t):
    tr, ro = t.transform.translation, t.transform.rotation
    return ((t.header.frame_id.lstrip("/"), t.child_frame_id.lstrip("/")),
            mat_from_pos_quat([tr.x, tr.y, tr.z], [ro.x, ro.y, ro.z, ro.w]))


def build_tf_static(paths, topic="/tf_static"):
    """{(parent, child): T_parent_child} from every /tf_static message."""
    edges = {}
    try:
        for m in read(paths, [topic]):
            for t in m.ros_msg.transforms:
                k, T = _tf_edge(t)
                edges[k] = T
    except Exception as e:  # topic absent
        log.warning(f"  could not read {topic}: {e}")
    return edges


def lookup_static(edges, src, dst):
    """T_src_dst by breadth-first search over `edges` (usable in both
    directions), or None if the frames are not connected."""
    if src == dst:
        return np.eye(4)
    adj = {}
    for (p, c), T in edges.items():
        adj.setdefault(p, []).append((c, T))
        adj.setdefault(c, []).append((p, np.linalg.inv(T)))
    q, seen = deque([(src, np.eye(4))]), {src}
    while q:
        node, acc = q.popleft()
        for nxt, T in adj.get(node, []):
            if nxt in seen:
                continue
            acc2 = acc @ T
            if nxt == dst:
                return acc2
            seen.add(nxt)
            q.append((nxt, acc2))
    return None


_TF_FIXED_CACHE = {}


def tf_fixed_edges(paths, topic="/tf", tol_m=1e-3, tol_deg=0.1):
    """Links on /tf that never move over the recording (some drivers, e.g. the
    ZED IMU, publish fixed links there instead of /tf_static). Cached per bag."""
    key = (tuple(paths), topic, tol_m, tol_deg)
    if key in _TF_FIXED_CACHE:
        return _TF_FIXED_CACHE[key]
    seen = {}
    try:
        for m in read(paths, [topic]):
            for t in m.ros_msg.transforms:
                k, T = _tf_edge(t)
                if k not in seen:
                    seen[k] = [T, True]
                elif seen[k][1]:
                    T0 = seen[k][0]
                    dt = np.linalg.norm(T[:3, 3] - T0[:3, 3])
                    dr = np.linalg.norm(Rotation.from_matrix(T0[:3, :3].T @ T[:3, :3]).as_rotvec())
                    if dt > tol_m or np.rad2deg(dr) > tol_deg:
                        seen[k][1] = False
    except Exception as e:
        log.warning(f"  could not read {topic}: {e}")
    _TF_FIXED_CACHE[key] = {k: v[0] for k, v in seen.items() if v[1]}
    return _TF_FIXED_CACHE[key]


def camera_extrinsic(edges, cam_frame, body_frame=None, extrinsic=None,
                     pose_is_camera=False):
    """T_body_cam: pose of the camera optical frame in the frame the pose topic
    describes. Give exactly one of
        pose_is_camera=True   the pose already is the camera pose (identity)
        body_frame="..."      chain body_frame -> cam_frame through `edges`
        extrinsic=[x y z qx qy qz qw] or a 4x4 matrix
    """
    given = sum(x is not None and x is not False for x in (body_frame, extrinsic)) \
        + bool(pose_is_camera)
    if given != 1:
        raise ConvertError(
            "need exactly one camera extrinsic source: pose_is_camera, body_frame "
            "or extrinsic.\n  The pose topic describes some body frame; the output "
            "needs the CAMERA optical pose,\n  and without that transform the "
            "trajectory is wrong in a way that still looks plausible.")
    if pose_is_camera:
        log.info("  pose topic is the camera optical pose (no extrinsic)")
        return np.eye(4)
    if extrinsic is not None:
        T = np.asarray(extrinsic, float)
        T = T if T.shape == (4, 4) else xyzq_to_T(T)
        log.info(f"  extrinsic given: t={np.round(T[:3, 3], 4).tolist()} m")
        for guess in sorted({f for e in edges for f in e}):      # cross-check
            T_alt = lookup_static(edges, guess, cam_frame)
            if T_alt is None:
                continue
            dt = float(np.linalg.norm(T_alt[:3, 3] - T[:3, 3]))
            dr = float(np.linalg.norm(Rotation.from_matrix(
                T_alt[:3, :3].T @ T[:3, :3]).as_rotvec()))
            if dt < 0.02 and dr < np.deg2rad(2):
                log.info(f"  cross-check: matches /tf_static '{guess}' -> {cam_frame} "
                         f"to {dt*100:.1f} cm / {np.rad2deg(dr):.2f} deg")
                break
        else:
            log.info("  cross-check: no frame in /tf_static reproduces this extrinsic "
                     "to 2 cm / 2 deg — using the given value")
        return T
    log.info(f"  extrinsic: {body_frame} -> {cam_frame}")
    T = lookup_static(edges, body_frame.lstrip("/"), cam_frame)
    if T is None:
        raise ConvertError(f"could not chain {body_frame} -> {cam_frame} through "
                           "/tf_static; see inspect_bag() for the frame names")
    if abs(np.trace(T[:3, :3]) - 3.0) < 1e-6:
        log.warning("  that transform has no rotation; a body->optical extrinsic "
                    "normally does. Check you picked a *_optical_frame.")
    log.info(f"  t_body_cam = {np.round(T[:3, 3], 4).tolist()} m")
    return T


# =========================================================================== #
# 5. Poses
# =========================================================================== #

def load_poses(paths, topic):
    """PoseStamped / Odometry topic -> (stamps_ns, positions Nx3, quats Nx4 xyzw,
    frame names seen). Sorted, duplicate stamps dropped."""
    ts, pos, quat, frames = [], [], [], set()
    for m in read(paths, [topic]):
        msg = m.ros_msg
        pose = msg.pose.pose if hasattr(msg.pose, "pose") else msg.pose
        ts.append(stamp_ns(msg))
        pos.append([pose.position.x, pose.position.y, pose.position.z])
        o = pose.orientation
        quat.append([o.x, o.y, o.z, o.w])
        frames.add(msg.header.frame_id.lstrip("/"))
        if hasattr(msg, "child_frame_id"):
            frames.add(msg.child_frame_id.lstrip("/"))
    if not ts:
        raise ConvertError(f"no messages on {topic}")
    ts = np.asarray(ts, dtype=np.int64)
    order = np.argsort(ts)
    ts, pos, quat = ts[order], np.asarray(pos)[order], np.asarray(quat)[order]
    keep = np.concatenate([[True], np.diff(ts) > 0])
    return ts[keep], pos[keep], quat[keep], frames


class PoseInterp:
    """Interpolated pose stream: linear position, slerp rotation.
    `at(t_ns)` returns a 4x4 or None outside the stream / inside a dropout
    longer than `max_gap_s`."""

    def __init__(self, ts, pos, quat, max_gap_s=0.5):
        self.ts = ts
        self.t0 = ts[0]
        self.rel = (ts - ts[0]).astype(np.float64) * 1e-9
        self.pos = pos
        self.slerp = Slerp(self.rel, Rotation.from_quat(quat))
        self.max_gap = float(np.max(np.diff(self.rel))) if len(ts) > 1 else 0.0
        self.max_gap_s = max_gap_s
        self.frames = set()

    @classmethod
    def from_topic(cls, paths, topic, **kw):
        ts, pos, quat, frames = load_poses(paths, topic)
        obj = cls(ts, pos, quat, **kw)
        obj.frames = frames
        return obj

    @classmethod
    def from_tum(cls, path, **kw):
        """A TUM file: 't x y z qx qy qz qw' per line (seconds), e.g. GLIM's
        traj_lidar.txt or 01a's traj_lidar_refined.txt."""
        a = np.loadtxt(os.path.expanduser(path), comments="#", ndmin=2)
        if a.shape[1] < 8 or not len(a):
            raise ConvertError(f"{path}: not a TUM trajectory (t x y z qx qy qz qw)")
        ts = np.round(a[:, 0] * 1e9).astype(np.int64)
        o = np.argsort(ts)
        ts, a = ts[o], a[o]
        keep = np.concatenate([[True], np.diff(ts) > 0])
        return cls(ts[keep], a[keep, 1:4], a[keep, 4:8], **kw)

    def __len__(self):
        return len(self.ts)

    def dt_to_nearest(self, t_ns):
        """Seconds from t_ns to the closest real pose sample."""
        t = (t_ns - self.t0) * 1e-9
        i = int(np.searchsorted(self.rel, t))
        best = np.inf
        if i > 0:
            best = min(best, abs(t - self.rel[i - 1]))
        if i < len(self.rel):
            best = min(best, abs(self.rel[i] - t))
        return float(best)

    def at(self, t_ns, max_extrap_s=0.0):
        t = (t_ns - self.t0) * 1e-9
        if t < self.rel[0] - max_extrap_s or t > self.rel[-1] + max_extrap_s:
            return None
        t = float(np.clip(t, self.rel[0], self.rel[-1]))
        i = int(np.searchsorted(self.rel, t))
        lo, hi = max(i - 1, 0), min(i, len(self.rel) - 1)
        if self.rel[hi] - self.rel[lo] > self.max_gap_s:
            return None
        T = np.eye(4)
        T[:3, :3] = self.slerp(t).as_matrix()
        T[:3, 3] = [np.interp(t, self.rel, self.pos[:, k]) for k in range(3)]
        return T


class FramePoses:
    """A pose stream of one frame re-expressed for another rigidly attached to
    it: at(t) = base.at(t) @ T (T = pose of the new frame in the old one)."""

    def __init__(self, base, T):
        self.base, self.T = base, T
        self.ts, self.max_gap, self.frames = base.ts, base.max_gap, base.frames

    def __len__(self):
        return len(self.base)

    def dt_to_nearest(self, t_ns):
        return self.base.dt_to_nearest(t_ns)

    def at(self, t_ns, max_extrap_s=0.0):
        T = self.base.at(t_ns, max_extrap_s)
        return None if T is None else T @ self.T


def tum_line(t_ns, T):
    """'t x y z qx qy qz qw' (seconds, metres) for one pose."""
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    return (f"{t_ns*1e-9:.6f} {T[0,3]:.9f} {T[1,3]:.9f} {T[2,3]:.9f} "
            f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}")


def write_tum(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _pose_source(paths, pose_bag, topic, pose_tum=None):
    if pose_tum:
        log.info(f"  poses from {pose_tum}")
        return PoseInterp.from_tum(pose_tum)
    pose_paths = _paths(pose_bag) if pose_bag else paths
    if pose_bag:
        log.info(f"  poses from {pose_bag}")
    return PoseInterp.from_topic(pose_paths, topic)


# =========================================================================== #
# 5b. Naming: where a dataset goes
# =========================================================================== #

def pass_of(bag):
    """(date, pass) of a bag below data/{raw,work,processed}/<date>/<pass>/...,
    or None. The tag <pass>_<date> names the pass's outputs (as the map
    stages name their clouds)."""
    p = bag[0] if isinstance(bag, (list, tuple)) else bag
    parts = os.path.abspath(os.path.expanduser(p)).split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in ("raw", "work", "processed") and len(parts) > i + 3:
            return parts[i + 1], parts[i + 2]
    return None


def topic_machine(topic):
    """'/mobile_1/zed/left/image_rect_color' -> 'mobile_1'."""
    return topic.strip("/").split("/")[0]


def dataset_dir(bag, fmt, depth=None, machine="mobile_1"):
    """data/processed/<date>/<pass>/datasets/<machine>/<fmt>[_<depth>]_<pass>_<date>
    for a bag below data/{raw,work}/<date>/<pass>/ -- e.g.
    .../datasets/mobile_1/replica_map_mapping_A_20260924."""
    dp = pass_of(bag)
    if dp is None:
        raise ConvertError(f"{bag} is not below data/{{raw,work}}/<date>/<pass>/: "
                           "give the output folder")
    date, pas = dp
    p = os.path.abspath(os.path.expanduser(bag[0] if isinstance(bag, (list, tuple)) else bag))
    root = p[:p.rindex(os.sep + date + os.sep + pas)]
    root = root[:root.rindex(os.sep)]                       # .../data
    name = "_".join(x for x in (fmt, depth, pas, date) if x)
    return os.path.join(root, "processed", date, pas, "datasets", machine, name)


# =========================================================================== #
# 6. LiDAR depth
# =========================================================================== #

_PF_DTYPE = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def cloud_points(msg):
    """sensor_msgs/PointCloud2 -> (Nx3 float64 xyz in the cloud's frame, N point
    times in seconds after the cloud stamp, or None). NaN and all-zero points
    (no return) are dropped. The times come from a 't' field (ouster_ros: uint32
    ns after the stamp; a float field is taken as seconds)."""
    fields = {f.name: f for f in msg.fields}
    if not all(k in fields for k in "xyz"):
        raise ConvertError(f"point cloud has no x/y/z fields: {sorted(fields)}")
    names = [k for k in ("x", "y", "z", "t") if k in fields]
    order = ">" if msg.is_bigendian else "<"
    try:
        dt = np.dtype({"names": names,
                       "formats": [order + _PF_DTYPE[fields[k].datatype] for k in names],
                       "offsets": [fields[k].offset for k in names],
                       "itemsize": msg.point_step})
    except KeyError as e:
        raise ConvertError(f"point cloud field type {e} not supported") from e
    w, h, ps = msg.width, msg.height, msg.point_step
    raw = np.frombuffer(bytes(msg.data), np.uint8)
    if msg.row_step != w * ps:                         # padded rows
        raw = raw[: h * msg.row_step].reshape(h, msg.row_step)[:, : w * ps]
    a = np.ascontiguousarray(raw).reshape(-1)[: w * h * ps].view(dt)
    xyz = np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float64)
    keep = np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 0)
    t = None
    if "t" in names:
        t = a["t"][keep].astype(np.float64)
        if dt["t"].kind in "iu":
            t *= 1e-9
    return xyz[keep], t


def project_depth(pts, K, dmin=0.0, dmax=np.inf, occlusion_px=7, occlusion_margin=0.1):
    """Points in the camera optical frame (x right, y down, z forward, metres)
    -> float32 HxW depth in metres at K's resolution, 0 = no point. The nearest
    point wins each pixel.
    occlusion_px > 1: a point is dropped when another point within that window
    is nearer by more than `occlusion_margin` (fraction of its depth) — the
    background the lidar sees through the gaps between a nearer object's points,
    or past its edge from the lidar's own viewpoint, which the camera cannot see."""
    W, H = int(K.width), int(K.height)
    pts = np.asarray(pts, np.float64).reshape(-1, 3)
    z = pts[:, 2]
    ok = np.isfinite(pts).all(axis=1) & (z > max(dmin, 1e-3)) & (z <= dmax)
    p, z = pts[ok], z[ok]
    u = np.round(K.fx * p[:, 0] / z + K.cx).astype(np.int64)
    v = np.round(K.fy * p[:, 1] / z + K.cy).astype(np.int64)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx, z = v[ok] * W + u[ok], z[ok]
    D = np.zeros(H * W, np.float32)
    if len(z):
        o = np.lexsort((z, idx))                        # by pixel, nearest first
        first = np.r_[True, idx[o][1:] != idx[o][:-1]]
        D[idx[o][first]] = z[o][first]
    D = D.reshape(H, W)
    if occlusion_px > 1 and len(z):
        k = int(occlusion_px) | 1
        near = cv2.erode(np.where(D > 0, D, np.float32(1e9)), np.ones((k, k), np.uint8))
        D[D > near * (1.0 + occlusion_margin)] = 0
    return D


_Scan = namedtuple("_Scan", "stamp center xyz t")


def _scan_in_camera(scan, t_img, cam_pose, bins=32):
    """One scan's points (camera frame at the point's own time) -> the camera
    frame at t_img, through the camera poses: p = T_wc(t_img)^-1 T_wc(t_point) p.
    Point times are binned (`bins` per scan); points without a pose are dropped."""
    T_img = cam_pose(t_img)
    if T_img is None:
        return None
    inv = np.linalg.inv(T_img)
    if scan.t is None or len(scan.t) == 0 or scan.t.max() <= scan.t.min():
        T = cam_pose(scan.center)
        if T is None:
            return None
        M = inv @ T
        return scan.xyz @ M[:3, :3].T + M[:3, 3]
    lo, hi = float(scan.t.min()), float(scan.t.max())
    b = np.minimum(((scan.t - lo) / (hi - lo) * bins).astype(np.int64), bins - 1)
    out, keep = np.empty(scan.xyz.shape), np.zeros(len(b), bool)
    for i in np.unique(b):
        T = cam_pose(scan.stamp + int((lo + (i + 0.5) * (hi - lo) / bins) * 1e9))
        if T is None:
            continue
        sel = b == i
        M = inv @ T
        out[sel] = scan.xyz[sel] @ M[:3, :3].T + M[:3, 3]
        keep[sel] = True
    return out[keep]


def iter_lidar_depth(paths, color, lidar, T_cam_lidar, cam_pose, K, *, n_scans=1,
                     max_dt_s=0.1, dmin=0.0, dmax=np.inf, occlusion_px=7,
                     occlusion_margin=0.1, keep_s=3.0):
    """Colour images with depth rendered from the lidar.

    Yields (t_ns, colour msg, depth) for every image on `color`; depth() returns
    (float32 metres HxW, valid mask), or None when no scan is within `max_dt_s`
    of the image or no point lands in it. It renders the `n_scans` scans nearest
    the image, each point moved to the image time through `cam_pose(t_ns)`
    (camera-to-world 4x4 or None), so more scans give denser depth without
    smearing as long as the poses are good.
    T_cam_lidar: pose of the cloud's frame in the camera optical frame."""
    R, tr = T_cam_lidar[:3, :3], T_cam_lidar[:3, 3]
    need_after = n_scans // 2 + 1                  # scans past an image before it is done
    max_dt, keep = int(max_dt_s * 1e9), int(keep_s * 1e9)
    scans, pending = deque(), deque()

    def ready(t):
        return sum(s.center > t for s in scans) >= need_after

    def emit(t, msg):
        near = sorted(scans, key=lambda s: abs(s.center - t))[:n_scans]
        ok = bool(near) and abs(near[0].center - t) <= max_dt

        def depth():
            if not ok:
                return None
            pts = [p for p in (_scan_in_camera(s, t, cam_pose) for s in near)
                   if p is not None and len(p)]
            if not pts:
                return None
            D = project_depth(np.concatenate(pts), K, dmin, dmax, occlusion_px,
                              occlusion_margin)
            return (D, D > 0) if (D > 0).any() else None
        return t, msg, depth

    for m in read(paths, [color, lidar]):
        msg = m.ros_msg
        if m.channel.topic == lidar:
            xyz, pt = cloud_points(msg)
            st = stamp_ns(msg)
            center = st + int(np.median(pt) * 1e9) if pt is not None and len(pt) else st
            scans.append(_Scan(st, center, (xyz @ R.T + tr).astype(np.float32),
                               None if pt is None else pt.astype(np.float32)))
            while pending and ready(pending[0][0]):
                yield emit(*pending.popleft())
            horizon = (pending[0][0] if pending else scans[-1].center) - keep
            while len(scans) > n_scans + need_after and scans[0].center < horizon:
                scans.popleft()
        else:
            t = stamp_ns(msg)
            if not pending and ready(t):
                yield emit(t, msg)
            else:
                pending.append((t, msg))
    while pending:
        yield emit(*pending.popleft())


def _zed_depth(paths, color, depth, tol_ns, input_scale, dmin, dmax):
    """Colour images paired with the depth image: (t_ns, colour msg, depth)."""
    for ct, cmsg, _, dmsg in iter_synced(paths, color, depth, tol_ns):
        yield ct, cmsg, (lambda d=dmsg: depth_to_metres(decode_image(d), d.encoding,
                                                         input_scale, dmin, dmax))


def _lidar_extrinsic(paths, edges, cam_frame, lidar, extrinsic=None):
    """T_cam_lidar: the cloud's frame in the camera optical frame."""
    m = first_msg(paths, lidar)
    if m is None:
        raise ConvertError(f"no messages on {lidar}")
    frame = m.header.frame_id.lstrip("/")
    if extrinsic is not None:
        T = np.asarray(extrinsic, float)
        T = T if T.shape == (4, 4) else xyzq_to_T(T)
        how = "given"
    else:
        T = lookup_static(edges, cam_frame, frame)
        how = "/tf_static"
        if T is None:
            fixed = dict(edges)
            fixed.update(tf_fixed_edges(paths))
            T, how = lookup_static(fixed, cam_frame, frame), "/tf_static + fixed /tf links"
        if T is None:
            raise ConvertError(
                f"no transform {cam_frame} -> {frame} (the {lidar} frame) in /tf_static.\n"
                "  Add the lidar-camera calibration to the bag (tf_edit.py --add) or "
                "pass --lidar-extrinsic")
    log.info(f"  lidar frame {frame}; T_cam_lidar ({how}): "
             f"t={np.round(T[:3, 3], 4).tolist()} m")
    return T


# =========================================================================== #
# 6b. Depth rendered from a map
# =========================================================================== #

def _array_module(gpu=True):
    """cupy when asked for and working, else numpy."""
    if not gpu:
        return np
    # CuPy's CUB kernels compile against the system CUDA headers, which may be
    # another CUDA major than the wheel (see 01_build_map); its own kernels do not
    os.environ.setdefault("CUPY_ACCELERATORS", "")
    try:
        import cupy as cp
        a = cp.arange(1 << 16, dtype=cp.int64)
        float(cp.sort(a[::-1])[:3].sum() + (cp.eye(3, dtype=cp.float32) @ cp.ones(
            3, dtype=cp.float32)).sum() + cp.flatnonzero(a < 5).size)
        return cp
    except Exception as e:
        log.info(f"  map depth on the CPU (no usable GPU: {type(e).__name__})")
        return np


_PLY_T = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1",
          "uint8": "u1", "char": "i1", "int8": "i1", "ushort": "u2", "uint16": "u2",
          "short": "i2", "int16": "i2", "uint": "u4", "uint32": "u4", "int": "i4",
          "int32": "i4"}
_PCD_T = {("F", 4): "f4", ("F", 8): "f8", ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4",
          ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4"}


def read_cloud(path):
    """xyz (float64 Nx3) of a .ply or .pcd, ascii or binary, without open3d."""
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    low = path.lower()
    if low.endswith(".ply"):
        end = head.find(b"end_header")
        if end < 0:
            raise ConvertError(f"{path}: no end_header, not a PLY")
        off = head.index(b"\n", end) + 1
        lines = head[:end].decode("ascii", "replace").splitlines()
        fmt = next(l.split()[1] for l in lines if l.startswith("format"))
        n, props, in_vertex = 0, [], False
        for l in lines:
            w = l.split()
            if w[:1] == ["element"]:
                in_vertex = w[1] == "vertex"
                n = int(w[2]) if in_vertex else n
            elif w[:1] == ["property"] and in_vertex:
                if w[1] == "list":
                    raise ConvertError(f"{path}: list properties in the vertex element")
                props.append((w[2], _PLY_T[w[1]]))
        names = [q[0] for q in props]
        if fmt == "ascii":
            a = np.loadtxt(path, skiprows=head[:off].count(b"\n"), max_rows=n, ndmin=2)
            return a[:, [names.index(k) for k in "xyz"]].astype(np.float64)
        e = "<" if fmt == "binary_little_endian" else ">"
        a = np.fromfile(path, dtype=np.dtype([(k, e + t) for k, t in props]), count=n,
                        offset=off)
    elif low.endswith(".pcd"):
        hdr, off = {}, 0
        for line in head.split(b"\n"):
            off += len(line) + 1
            w = line.decode("ascii", "replace").split()
            if w and not w[0].startswith("#"):
                hdr[w[0].upper()] = w[1:]
            if w[:1] == ["DATA"]:
                break
        names = hdr["FIELDS"]
        cnt = [int(c) for c in hdr.get("COUNT", ["1"] * len(names))]
        n = int(hdr["POINTS"][0])
        if hdr["DATA"][0] == "ascii":
            a = np.loadtxt(path, skiprows=head[:off].count(b"\n"), max_rows=n, ndmin=2)
            col = np.cumsum([0] + cnt)
            return a[:, [col[names.index(k)] for k in "xyz"]].astype(np.float64)
        if hdr["DATA"][0] != "binary":
            raise ConvertError(f"{path}: PCD DATA {hdr['DATA'][0]} needs open3d")
        dt = [(k, "<" + _PCD_T[(t, int(z))], (c,) if c > 1 else ())
              for k, t, z, c in zip(names, hdr["TYPE"], hdr["SIZE"], cnt)]
        a = np.fromfile(path, dtype=np.dtype(dt), count=n, offset=off)
    else:
        raise ConvertError(f"{path}: expected .ply or .pcd")
    return np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)


def _voxel_centroids(P, voxel):
    k = np.floor(P / voxel).astype(np.int64) + (1 << 20)
    key = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return np.stack([np.bincount(inv, P[:, i]) for i in range(3)], 1) / cnt[:, None]


def _pca_normals(P, k=12, chunk=200_000):
    tree = cKDTree(P)
    N = np.empty_like(P)
    for a in range(0, len(P), chunk):
        _, idx = tree.query(P[a:a + chunk], k=min(k, len(P)))
        Q = P[idx] - P[idx].mean(1, keepdims=True)
        N[a:a + chunk] = np.linalg.eigh(np.einsum("nki,nkj->nij", Q, Q))[1][:, :, 0]
    return N


def load_map(path, voxel=0.02):
    """A map cloud -> (points, normals) float32 Nx3, downsampled to `voxel`
    (centroids) with normals from the local neighbourhood. open3d when
    installed (fast), else numpy/scipy."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise ConvertError(f"map not found: {path}")
    try:
        import open3d as o3d
    except ImportError:
        o3d = None
    if o3d is not None:
        pc = o3d.io.read_point_cloud(path)
        n0 = len(pc.points)
        if not n0:
            raise ConvertError(f"{path}: no points")
        if voxel > 0:
            pc = pc.voxel_down_sample(voxel)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(4 * voxel, 0.05), max_nn=16))
        P, N = np.asarray(pc.points), np.asarray(pc.normals)
    else:
        P = read_cloud(path)
        n0 = len(P)
        if voxel > 0:
            P = _voxel_centroids(P, voxel)
        N = _pca_normals(P)
    log.info(f"  map {path}: {n0} points -> {len(P)} at {voxel} m, normals estimated")
    return P.astype(np.float32), N.astype(np.float32)


class MapDepth:
    """Depth images rendered from a point map as surfels: every map point is a
    disc of radius splat x voxel on its local plane (its normal), a pixel's
    depth is where its ray meets the nearest disc. That is exact on slanted
    surfaces (a floor seen at a grazing angle), closes the gaps between
    points, and the nearest disc hides what is behind it.

    The map must be in the world frame of the poses used (the trajectory it
    was built from). Things that moved are not in a map with them removed."""

    def __init__(self, pts, nrm, K, voxel, dmin=0.2, dmax=10.0, splat=1.0, max_px=24,
                 gpu=True, cell=2.0):
        self.xp = xp = _array_module(gpu)
        self.K, self.dmin, self.dmax = K, max(float(dmin), 1e-3), float(dmax)
        self.rho = float(splat) * float(voxel)
        self.max_px = int(max_px)
        self.budget = 15_000_000 if xp is not np else 6_000_000   # candidates per batch
        if xp is np:                    # cull by coarse cells; the GPU just transforms all
            k = np.floor(pts / cell).astype(np.int64) + (1 << 20)
            key = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
            o = np.argsort(key, kind="stable")
            ks = key[o]
            self.start = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
            self.count = np.diff(np.r_[self.start, len(ks)])
            self.P, self.N = pts[o], nrm[o]
            lo, hi = (np.minimum.reduceat(self.P, self.start),
                      np.maximum.reduceat(self.P, self.start))
            self.cen, self.rad = 0.5 * (lo + hi), 0.5 * np.linalg.norm(hi - lo, axis=1)
        else:
            self.P, self.N = xp.asarray(pts, xp.float32), xp.asarray(nrm, xp.float32)

    def _visible_cells(self, R, t):
        K = self.K
        C = (self.cen - t) @ R
        r = self.rad
        ok = (C[:, 2] > -r) & (C[:, 2] - r < self.dmax + self.rho)
        for a, b, c in ((K.cx / K.fx, 0, 1), ((K.width - K.cx) / K.fx, 0, -1),
                        (K.cy / K.fy, 1, 1), ((K.height - K.cy) / K.fy, 1, -1)):
            n = np.zeros(3)
            n[b], n[2] = c, a                      # inward normal of a side plane
            ok &= (C @ n) / np.linalg.norm(n) > -r
        bi = np.flatnonzero(ok)
        if not len(bi):
            return None
        return np.concatenate([np.arange(s, s + c) for s, c in
                               zip(self.start[bi], self.count[bi])])

    def render(self, T_wc):
        """HxW float32 depth in metres (0 = no surface) for camera-to-world T_wc."""
        xp, K, rho = self.xp, self.K, self.rho
        W, H = int(K.width), int(K.height)
        R, t = np.asarray(T_wc[:3, :3], np.float64), np.asarray(T_wc[:3, 3], np.float64)
        if xp is np:
            idx = self._visible_cells(R, t)
            if idx is None:
                return np.zeros((H, W), np.float32)
            P, N = self.P[idx], self.N[idx]
        else:
            P, N = self.P, self.N
        Rd, td = xp.asarray(R, xp.float32), xp.asarray(t, xp.float32)
        fx, fy, cx, cy = (float(v) for v in K.k4)
        keys = []
        step = 4_000_000
        for a in range(0, len(P), step):
            Pc = (P[a:a + step] - td) @ Rd                # world -> camera
            z = Pc[:, 2]
            m = (z > self.dmin) & (z < self.dmax + rho)
            Pc, Nc, z = Pc[m], N[a:a + step][m] @ Rd, z[m]
            u0, v0 = fx * Pc[:, 0] / z + cx, fy * Pc[:, 1] / z + cy
            r = xp.clip(xp.ceil(fx * rho / z), 1, self.max_px).astype(xp.int32)
            m = (u0 > -r - 1) & (u0 < W + r) & (v0 > -r - 1) & (v0 < H + r)
            Pc, Nc, u0, v0, r = Pc[m], Nc[m], xp.rint(u0[m]), xp.rint(v0[m]), r[m]
            npd = (Nc * Pc).sum(1)
            for rv in [int(x) for x in _host(xp.unique(r))]:
                sel = xp.flatnonzero(r == rv)
                g = xp.arange(-rv, rv + 1, dtype=xp.float32)
                du, dv = (x.reshape(-1) for x in xp.meshgrid(g, g))
                rows = max(1, self.budget // len(du))
                for b in range(0, len(sel), rows):
                    s_ = sel[b:b + rows]
                    uc, vc = u0[s_, None] + du, v0[s_, None] + dv
                    rx, ry = (uc - cx) / fx, (vc - cy) / fy
                    p, n = Pc[s_], Nc[s_]
                    nd = n[:, 0:1] * rx + n[:, 1:2] * ry + n[:, 2:3]
                    zh = npd[s_, None] / xp.where(xp.abs(nd) > 1e-6, nd, xp.nan)
                    d2 = ((rx * zh - p[:, 0:1]) ** 2 + (ry * zh - p[:, 1:2]) ** 2
                          + (zh - p[:, 2:3]) ** 2)
                    ok = ((uc >= 0) & (uc < W) & (vc >= 0) & (vc < H) & (d2 <= rho * rho)
                          & (zh > self.dmin) & (zh <= self.dmax))
                    pix = (vc[ok].astype(xp.int64) * W + uc[ok].astype(xp.int64))
                    keys.append((pix << 32) | (zh[ok] * 1e4).astype(xp.int64))
        D = xp.zeros(H * W, xp.float32)
        if keys:
            k = xp.sort(xp.concatenate(keys))             # by pixel, nearest first
            pix = k >> 32
            first = xp.concatenate([xp.ones(1, bool), pix[1:] != pix[:-1]])
            D[pix[first]] = (k[first] & 0xFFFFFFFF).astype(xp.float32) * 1e-4
        return _host(D.reshape(H, W))


def _host(a):
    """A cupy array -> numpy (numpy arrays pass through)."""
    return a.get() if hasattr(a, "get") and not isinstance(a, np.ndarray) else np.asarray(a)


def iter_map_depth(paths, color, cam_pose, renderer):
    """Colour images with depth rendered from the map at the camera pose of
    each: (t_ns, colour msg, depth()) like iter_lidar_depth."""
    for m in read(paths, [color]):
        msg = m.ros_msg
        t = stamp_ns(msg)

        def depth(t=t):
            T = cam_pose(t)
            if T is None:
                return None
            D = renderer.render(T)
            return (D, D > 0) if (D > 0).any() else None
        yield t, msg, depth


# =========================================================================== #
# 7. Converter: Replica (SplaTAM, NICE-SLAM, Point-SLAM, GradSLAM)
# =========================================================================== #

def to_replica(bag, out, *,
               color="/mobile_1/zed/left/image_rect_color",
               depth="/mobile_1/zed/depth/depth_registered",
               info="/mobile_1/zed/left/camera_info",
               pose="/mobile_1/global_pose",
               pose_bag=None, pose_tum=None,
               pose_is_camera=False, body_frame=None, extrinsic=None, cam_frame=None,
               depth_source="lidar", lidar="/mobile_1/ouster/points", lidar_scans=1,
               lidar_max_dt_ms=100.0, lidar_extrinsic=None,
               occlusion_px=7, occlusion_margin=0.1,
               map=None, map_voxel=0.02, splat=1.0, max_splat_px=24, gpu=True,
               sync_tol_ms=10.0, max_pose_dt_ms=0.0,
               depth_scale=6553.5, depth_input_scale=0.001,
               depth_min=0.2, depth_max=10.0,
               every=1, max_frames=0, jpeg_quality=95):
    """Replica-style RGB-D sequence.

    bag, out           recording (.mcap or bag dir, or list of .mcap) / output dir
    color, info        colour image and its CameraInfo
    pose, pose_bag     pose topic (PoseStamped/Odometry), optionally from another bag
    pose_tum           poses from a TUM file instead (e.g. traj_lidar_refined.txt);
                       body_frame then defaults to the frame_id of `lidar` (the
                       frame GLIM / 01a trajectories are of)
    pose_is_camera / body_frame / extrinsic
                       how to get the camera pose from the poses (exactly one, see
                       camera_extrinsic()); cam_frame defaults to the colour frame_id
    depth_source       "lidar": depth projected from `lidar` (PointCloud2) into the
                       colour camera; "map": rendered from the map cloud `map`
                       (MapDepth; give the trajectory the map was built from as
                       pose_tum); "zed": the `depth` image (aligned to colour)
    map, map_voxel     [map] the map (.pcd/.ply) and the voxel it is rendered at
    splat, max_splat_px
                       [map] surfel radius in voxels, and its cap in pixels
    gpu                [map] render on the GPU when cupy works
    lidar_scans        scans rendered into each frame (nearest in time; >1 = denser,
                       each moved to the image time through the poses)
    lidar_max_dt_ms    drop a frame when no scan is this close to it
    lidar_extrinsic    T_cam_lidar [x y z qx qy qz qw] (cloud frame in the camera
                       optical frame); default: chained from /tf_static
    occlusion_px, occlusion_margin
                       hidden-point filter of project_depth() (occlusion_px 0 = off)
    sync_tol_ms        [zed] max colour/depth stamp difference for a pair
    max_pose_dt_ms     keep a frame only within this of a real pose sample (0 = off)
    depth_scale        PNG units per metre: 6553.5 as Replica (uint16 0..65535 =
                       0..10 m, 0.15 mm steps); 1000 = mm (up to 65.5 m)
    depth_input_scale  [zed] metres per unit of integer input depth (0.001 = mm)
    depth_min/max      metres; outside -> 0 (invalid)
    every, max_frames  keep every Nth image / stop after N frames (0 = all)
    jpeg_quality       colour JPEG quality
    Returns a dict with frames, candidates, dropped counts, intrinsics, path length.
    """
    if depth_source not in ("lidar", "map", "zed"):
        raise ConvertError(f"depth_source must be lidar, map or zed, got {depth_source!r}")
    if depth_max * depth_scale > 65535.5:
        log.warning(f"  depth_max {depth_max} m does not fit a uint16 PNG at {depth_scale} "
                    f"units/m (max {65535 / depth_scale:.2f} m): farther depth is clipped "
                    f"to that. Lower --depth-scale (1000 = mm) or --depth-max")
    if depth_source == "map" and not map:
        raise ConvertError("depth_source map needs the map cloud (--map)")
    paths = _paths(bag)
    res_dir = os.path.join(out, "results")
    os.makedirs(res_dir, exist_ok=True)

    log.info("pass 1: intrinsics / poses / tf_static")
    ci = first_msg(paths, info)
    if ci is None:
        raise ConvertError(f"no messages on {info}")
    K = intrinsics_from_info(ci)
    if any(abs(v) > 1e-6 for v in np.asarray(ci.d, float)):
        log.warning("  camera_info has non-zero distortion — these images are NOT "
                    "rectified. Undistort them before using this sequence.")
    log.info(f"  {K.width}x{K.height}  fx={K.fx:.3f} fy={K.fy:.3f} "
             f"cx={K.cx:.3f} cy={K.cy:.3f}")

    interp = _pose_source(paths, pose_bag, pose, pose_tum)
    log.info(f"  {len(interp)} poses, frames seen: {sorted(interp.frames)}, "
             f"max gap {interp.max_gap*1e3:.0f} ms")
    if pose_bag:   # separate pose bag: the classic place for a clock mismatch
        img_t = first_msg(paths, color)
        if img_t is not None:
            img_t, lo, hi = stamp_ns(img_t), interp.ts[0], interp.ts[-1]
            log.info(f"  pose span {(hi-lo)*1e-9:.1f} s; first colour stamp is "
                     f"{(img_t-lo)*1e-9:+.1f} s from the pose start")
            if not (lo - 5e9 <= img_t <= hi + 5e9):
                raise ConvertError("the pose stream and the images do not overlap in "
                                   "time — different clocks or the wrong pose bag")

    edges = build_tf_static(paths)
    if edges:
        log.info(f"  tf_static frames: {sorted({f for e in edges for f in e})}")
    if cam_frame is None:
        m = first_msg(paths, color)
        cam_frame = m.header.frame_id.lstrip("/") if m is not None else None
    log.info(f"  colour image frame_id: {cam_frame}")
    if pose_tum and body_frame is None and extrinsic is None and not pose_is_camera:
        m = first_msg(paths, lidar)
        if m is None:
            raise ConvertError(f"pose_tum: say which frame its poses are of (body_frame); "
                               f"no messages on {lidar} to take it from")
        body_frame = m.header.frame_id.lstrip("/")
        log.info(f"  {pose_tum} taken as poses of {body_frame} (the {lidar} frame)")
    T_bc = camera_extrinsic(edges, cam_frame, body_frame, extrinsic, pose_is_camera)

    def cam_pose(t_ns):
        T = interp.at(t_ns)
        return None if T is None else T @ T_bc

    if depth_source == "map":
        if not pose_tum:
            log.warning("  depth from a map: the poses must be in the map's frame -- give "
                        "the trajectory the map was built from (pose_tum)")
        pts, nrm = load_map(map, map_voxel)
        renderer = MapDepth(pts, nrm, K, map_voxel, depth_min, depth_max, splat,
                            max_splat_px, gpu)
        del pts, nrm
        frames = iter_map_depth(paths, color, cam_pose, renderer)
        src = (f"map {map} at {map_voxel} m, surfels of {splat} voxel"
               f"{' on the GPU' if renderer.xp is not np else ''}")
    elif depth_source == "lidar":
        T_cl = _lidar_extrinsic(paths, edges, cam_frame, lidar, lidar_extrinsic)
        frames = iter_lidar_depth(paths, color, lidar, T_cl, cam_pose, K,
                                  n_scans=max(1, int(lidar_scans)),
                                  max_dt_s=lidar_max_dt_ms * 1e-3, dmin=depth_min,
                                  dmax=depth_max, occlusion_px=occlusion_px,
                                  occlusion_margin=occlusion_margin)
        src = (f"lidar {lidar}, {max(1, int(lidar_scans))} scan(s)/frame, occlusion "
               + (f"{occlusion_px} px / {occlusion_margin*100:.0f}%" if occlusion_px > 1
                  else "off"))
    else:
        frames = _zed_depth(paths, color, depth, int(sync_tol_ms * 1e6),
                            depth_input_scale, depth_min, depth_max)
        src = f"zed {depth}"
    log.info(f"  depth: {src}")

    log.info("pass 2: writing frames")
    gate = max_pose_dt_ms * 1e-3
    traj, valid_frac, pose_dt = [], [], []
    n = skipped_pose = skipped_dt = skipped_depth = 0
    q = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    for ct, cmsg, get_depth in frames:
        n += 1
        if (n - 1) % every:
            continue
        pdt = interp.dt_to_nearest(ct)
        if gate > 0 and pdt > gate:
            skipped_dt += 1
            continue
        T = interp.at(ct)
        if T is None:
            skipped_pose += 1
            continue
        got = get_depth()
        if got is None:
            skipped_depth += 1
            continue
        d, valid = got
        pose_dt.append(pdt)
        valid_frac.append(float(valid.mean()))
        png = np.clip(np.round(d * depth_scale), 0, 65535).astype(np.uint16)
        bgr = to_bgr(decode_image(cmsg), cmsg.encoding)
        if png.shape[:2] != bgr.shape[:2]:
            png = cv2.resize(png, (bgr.shape[1], bgr.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        i = len(traj)
        cv2.imwrite(os.path.join(res_dir, f"frame{i:06d}.jpg"), bgr, q)
        cv2.imwrite(os.path.join(res_dir, f"depth{i:06d}.png"), png)
        traj.append(" ".join(f"{v:.9f}" for v in (T @ T_bc).reshape(-1)))
        if len(traj) % 500 == 0:
            log.info(f"  {len(traj)} frames")
        if max_frames and len(traj) >= max_frames:
            break
    if not traj:
        what = (f"colour images with {depth_source} depth" if depth_source != "zed"
                else "synchronised colour/depth pairs")
        raise ConvertError(f"no {what} with a valid pose")

    with open(os.path.join(out, "traj.txt"), "w") as f:
        f.write("\n".join(traj) + "\n")
    with open(os.path.join(out, "splatam_data_config.yaml"), "w") as f:
        f.write("# SplaTAM data config for this sequence.\n"
                "# The layout is byte-compatible with Replica, so dataset_name stays \"replica\".\n"
                "dataset_name: \"replica\"\ncamera_params:\n"
                f"  image_height: {K.height}\n  image_width: {K.width}\n"
                f"  fx: {K.fx:.6f}\n  fy: {K.fy:.6f}\n  cx: {K.cx:.6f}\n  cy: {K.cy:.6f}\n"
                f"  png_depth_scale: {depth_scale:.1f}\n  crop_edge: 0\n")

    xyz = np.array([[float(x) for x in l.split()][3::4][:3] for l in traj])
    d0 = np.linalg.norm(np.diff(xyz, axis=0), axis=1) if len(xyz) > 1 else np.zeros(1)
    pd = np.asarray(pose_dt) * 1e3 if pose_dt else np.zeros(1)
    report = (
        f"frames written      : {len(traj)}\n"
        + (f"colour images       : {n}\n" if depth_source != "zed"
           else f"pairs found         : {n}\n") +
        f"dropped (no pose)   : {skipped_pose}\n"
        f"dropped (pose dt)   : {skipped_dt}"
        + (f"  (gate {max_pose_dt_ms:.0f} ms)\n" if gate > 0 else "  (gate off)\n")
        + (f"dropped (no lidar) : {skipped_depth}  (no scan within "
           f"{lidar_max_dt_ms:.0f} ms, or no point in view)\n"
           if depth_source == "lidar" else
           f"dropped (no map)   : {skipped_depth}  (nothing of the map in view)\n"
           if depth_source == "map" else "") +
        f"depth source        : {src}\n"
        f"pose dt of kept     : median {np.median(pd):.1f} ms, "
        f"p95 {np.percentile(pd, 95):.1f} ms, max {pd.max():.1f} ms\n"
        f"resolution          : {K.width}x{K.height}\n"
        f"intrinsics          : fx={K.fx:.3f} fy={K.fy:.3f} cx={K.cx:.3f} cy={K.cy:.3f}\n"
        f"png_depth_scale     : {depth_scale}\n"
        f"valid depth px      : mean {np.mean(valid_frac)*100:.1f}%, "
        f"min {np.min(valid_frac)*100:.1f}%\n"
        f"pose stream max gap : {interp.max_gap*1e3:.0f} ms\n"
        f"inter-frame motion  : mean {d0.mean()*100:.1f} cm, max {d0.max()*100:.1f} cm\n"
        f"total path length   : {d0.sum():.2f} m\n")
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write(report)
    log.info("\n" + report)
    return {"format": "replica", "out": out, "frames": len(traj), "candidates": n,
            "dropped_no_pose": skipped_pose, "dropped_pose_dt": skipped_dt,
            "dropped_no_depth": skipped_depth, "depth_source": depth_source,
            "valid_depth_mean": float(np.mean(valid_frac)),
            "intrinsics": K.k4, "resolution": (K.width, K.height),
            "path_length_m": float(d0.sum())}


# =========================================================================== #
# 8. Converter: MCD (mcdviral.github.io), config driven
# =========================================================================== #

DEFAULT_MCD_CONFIG = """\
# MCD-style config for mobile_1 (ZED + Ouster)
#
#   python3 mcap_convert.py <bag dir> out/coop2_mobile_1 --format mcd
#
# Extrinsics are [x, y, z, qx, qy, qz, qw] (metres, Hamilton quaternion).
#   T_body_sensor : pose of the sensor in the body frame   (doc "body -> sensor")
#   T_sensor_body : pose of the body in the sensor frame   (doc "sensor -> body"), inverted here
#   same_as       : reuse another sensor's extrinsic
#   stereo_right_of: <left sensor>, baseline from this sensor's `info` (rectified P)
#   tf_frame      : look it up in /tf_static (or a fixed /tf link) from the body frame;
#                   "auto" = the frame_id stamped on the messages
#   frame         : the frame the given extrinsic refers to; if the messages are stamped
#                   in another frame, the link between them is chained from /tf_static
# Sensor types: image (encoding rgb8|mono8), depth (input_scale, min, max), imu, pointcloud
# A depth sensor's `source`: topic (its `topic`, e.g. the ZED depth), lidar (the LiDAR
# projected into `camera`, `scans` per frame) or map (rendered from `map` at `voxel` m);
# lidar/map depth is made for every image of `camera` (default: its same_as).
# pose: `topic` (PoseStamped/Odometry), or `tum` (a TUM file, e.g. 01a's
# traj_lidar_refined.txt) of the frame `tum_frame`, chained to `frame` via /tf_static.

# sequence: <name>        # bags are <sequence>_merged.bag; default: <pass>_<date> of the
#                          # bag (data/work/<date>/<pass>/...), else the output folder's name
compress: false            # true = bz2 like the released MCD bags (slower)
merged: true               # one <seq>_merged.bag like MCD; false = one bag per "bag:" name

pose:
  topic: /mobile_1/global_pose               # world (map) frame, per dataset notes
  frame: zed_left_camera_optical_frame       # global_pose IS the ZED left optical pose
  sample_at: ouster                          # MCD: one GT row per lidar scan (else rate_hz: 10)

yaml:                        # camera.yaml / imu.yaml / lidar.yaml (MCD style, sensor -> IMU)
  imu: ouster_imu            # the IMU the extrinsics are relative to (factory-calibrated, see /mobile_1/ouster/metadata)
  camera: zed_left
  lidar_timestamp_end: false # true if your Ouster stamps clouds at the END of the sweep

sensors:
  zed_left:
    type: image
    topic: /mobile_1/zed/left/image_rect_color
    info: /mobile_1/zed/left/camera_info
    out_topic: /zed/color/image_raw
    bag: zed
    is_body: true

  zed_depth:                                 # registered to the left RGB
    type: depth
    source: lidar                            # topic = the ZED depth below; lidar; map
    topic: /mobile_1/zed/depth/depth_registered
    info: /mobile_1/zed/depth/camera_info
    out_topic: /zed/depth/image_raw          # 16UC1 millimetres
    bag: zed
    same_as: zed_left
    min: 0.2
    max: 10.0

  zed_imu:
    type: imu
    topic: /mobile_1/zed/imu/data
    out_topic: /zed/imu
    bag: zed
    tf_frame: auto          # frame_id of the IMU messages, looked up in /tf_static or /tf

  ouster:
    type: pointcloud
    topic: /mobile_1/ouster/points
    out_topic: /os_cloud_node/points         # MCD topic name
    bag: ouster
    # notes: os_lidar -> zed_left_camera_optical_frame
    T_sensor_body: [-0.074928, -0.066971, -0.091627, -0.497829, -0.498035, 0.501789, 0.502329]
    frame: os_lidar

  ouster_imu:
    type: imu
    topic: /mobile_1/ouster/imu
    out_topic: /os_cloud_node/imu            # MCD topic name
    bag: ouster
    tf_frame: auto          # os_imu, reached via os_lidar (ouster extrinsic above)
"""

MSGTYPE = {"image": "sensor_msgs/msg/Image", "depth": "sensor_msgs/msg/Image",
           "imu": "sensor_msgs/msg/Imu", "pointcloud": "sensor_msgs/msg/PointCloud2"}


def load_mcd_config(config=None):
    """None -> built-in mobile_1 config; str -> yaml file path; dict -> as is."""
    import yaml
    if config is None:
        cfg, src = yaml.safe_load(DEFAULT_MCD_CONFIG), "built-in mobile_1"
    elif isinstance(config, dict):
        cfg, src = config, "dict"
    else:
        with open(config) as f:
            cfg, src = yaml.safe_load(f), config
    pose_cfg, sensors = cfg.get("pose") or {}, cfg.get("sensors") or {}
    if not ("topic" in pose_cfg or "tum" in pose_cfg) or "frame" not in pose_cfg or not sensors:
        raise ConvertError("config needs pose.topic (or pose.tum), pose.frame and at "
                           "least one sensor")
    for name, s in sensors.items():
        made = s.get("type") == "depth" and s.get("source") in ("lidar", "map")
        for k in ("type", "out_topic") + (() if made else ("topic",)):
            if k not in s:
                raise ConvertError(f"sensor '{name}' needs '{k}'")
        if s["type"] not in MSGTYPE:
            raise ConvertError(f"sensor '{name}': unknown type '{s['type']}' "
                               "(image | depth | imu | pointcloud)")
    log.info(f"config: {src}")
    return cfg


def resolve_extrinsic(name, s, body_frame, edges, done, paths):
    """T_body_sensor for one config sensor entry -> (4x4 or None, how).
    `done` holds already resolved sensors (for same_as / stereo_right_of);
    `edges` is extended in place so later sensors can chain through this one."""
    if s.get("is_body"):
        T, how = np.eye(4), "is the body frame"
    elif "T_body_sensor" in s:
        T, how = xyzq_to_T(s["T_body_sensor"]), "config T_body_sensor"
    elif "T_sensor_body" in s:
        T, how = np.linalg.inv(xyzq_to_T(s["T_sensor_body"])), "config T_sensor_body (inverted)"
    elif "same_as" in s:
        ref = s["same_as"]
        if done.get(ref) is None:
            return None, f"same_as '{ref}', which has no extrinsic"
        T, how = done[ref], f"same as {ref}"
    elif "stereo_right_of" in s:
        left = s["stereo_right_of"]
        if done.get(left) is None:
            return None, f"stereo_right_of '{left}', which has no extrinsic"
        info = first_msg(paths, s["info"])
        if info is None:
            return None, f"no messages on {s['info']} (needed for the baseline)"
        B = stereo_baseline(info)
        if B is None:
            return None, f"{s['info']} has no baseline in P"
        T_lr = np.eye(4)
        T_lr[0, 3] = B
        T, how = done[left] @ T_lr, f"{left} + stereo baseline {B*100:.2f} cm"
    elif "tf_frame" in s:
        frame = s["tf_frame"]
        if frame == "auto":
            msg = first_msg(paths, s["topic"])
            if msg is None:
                return None, f"no messages on {s['topic']}"
            frame = msg.header.frame_id.lstrip("/")
        T = lookup_static(edges, body_frame, frame)
        how = f"tf {body_frame} -> {frame}"
        if T is None:
            for (p, c), Td in tf_fixed_edges(paths).items():
                if frame in (p, c):
                    edges[(p, c)] = Td
                    how += f" (via fixed /tf link {p} -> {c})"
            T = lookup_static(edges, body_frame, frame)
        if T is None:
            return None, (f"'{frame}' not reachable from '{body_frame}' in /tf_static "
                          "or as a fixed link on /tf")
    else:
        return None, "no extrinsic given"

    decl = s.get("frame")
    if decl and any(k in s for k in ("T_body_sensor", "T_sensor_body")):
        edges[(body_frame, decl)] = T      # frames under it become reachable
    if decl:
        msg = first_msg(paths, s["topic"])
        got = msg.header.frame_id.lstrip("/") if msg is not None else decl
        if got != decl:
            T_fix = lookup_static(edges, decl, got)
            if T_fix is None:
                log.warning(f"  {name}: extrinsic is for '{decl}' but messages are in "
                            f"'{got}' and /tf_static does not link them — used as is")
            else:
                T = T @ T_fix
                how += f", chained {decl} -> {got}"
    return T, how


class _Ros1Writer:
    """ROS 2 messages -> ROS 1 messages for the MCD bags (rosbags typestore)."""

    def __init__(self):
        try:
            from rosbags.rosbag1 import Writer
            from rosbags.typesys import Stores, get_typestore
        except ImportError as e:
            raise ConvertError("the mcd format needs: pip install rosbags") from e
        self.Writer = Writer
        self.ts = get_typestore(Stores.ROS1_NOETIC)
        T = self.ts.types
        self.Header, self.Time = T["std_msgs/msg/Header"], T["builtin_interfaces/msg/Time"]
        self.Image, self.Imu = T["sensor_msgs/msg/Image"], T["sensor_msgs/msg/Imu"]
        self.PC2, self.PField = T["sensor_msgs/msg/PointCloud2"], T["sensor_msgs/msg/PointField"]
        self.Vec3, self.Quat = T["geometry_msgs/msg/Vector3"], T["geometry_msgs/msg/Quaternion"]
        self.seqno = {}

    def header(self, t_ns, frame, topic):
        self.seqno[topic] = self.seqno.get(topic, -1) + 1
        return self.Header(seq=self.seqno[topic],
                           stamp=self.Time(sec=t_ns // 1_000_000_000,
                                           nanosec=t_ns % 1_000_000_000),
                           frame_id=frame)

    def image(self, t_ns, arr, enc, frame, topic):
        arr = np.ascontiguousarray(arr)
        ch = arr.shape[2] if arr.ndim == 3 else 1
        return self.Image(header=self.header(t_ns, frame, topic), height=arr.shape[0],
                          width=arr.shape[1], encoding=enc, is_bigendian=0,
                          step=arr.shape[1] * arr.itemsize * ch,
                          data=arr.view(np.uint8).reshape(-1))

    def convert(self, s, name, msg, t):
        """One ROS 2 message of config sensor `s` -> serialised ROS 1 bytes."""
        typ, topic, frame = s["type"], s["out_topic"], s.get("out_frame", name)
        if typ == "image":
            bgr = to_bgr(decode_image(msg), msg.encoding)
            if s.get("encoding", "rgb8") == "mono8":
                out = self.image(t, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), "mono8", frame, topic)
            else:
                out = self.image(t, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "rgb8", frame, topic)
        elif typ == "depth":
            d, _ = depth_to_metres(decode_image(msg), msg.encoding, s.get("input_scale", 0.001),
                                   s.get("min", 0.1), s.get("max", 20.0))
            out = self.image(t, np.clip(d * 1000.0, 0, 65535).astype(np.uint16),
                             "16UC1", frame, topic)
        elif typ == "imu":
            out = self.Imu(
                header=self.header(t, frame, topic),
                orientation=self.Quat(x=msg.orientation.x, y=msg.orientation.y,
                                      z=msg.orientation.z, w=msg.orientation.w),
                orientation_covariance=np.asarray(msg.orientation_covariance, np.float64),
                angular_velocity=self.Vec3(x=msg.angular_velocity.x, y=msg.angular_velocity.y,
                                           z=msg.angular_velocity.z),
                angular_velocity_covariance=np.asarray(msg.angular_velocity_covariance, np.float64),
                linear_acceleration=self.Vec3(x=msg.linear_acceleration.x,
                                              y=msg.linear_acceleration.y,
                                              z=msg.linear_acceleration.z),
                linear_acceleration_covariance=np.asarray(msg.linear_acceleration_covariance,
                                                          np.float64))
        else:  # pointcloud
            out = self.PC2(
                header=self.header(t, frame, topic), height=msg.height, width=msg.width,
                fields=[self.PField(name=f.name, offset=f.offset, datatype=f.datatype,
                                    count=f.count) for f in msg.fields],
                is_bigendian=bool(msg.is_bigendian), point_step=msg.point_step,
                row_step=msg.row_step, data=np.frombuffer(bytes(msg.data), np.uint8),
                is_dense=bool(msg.is_dense))
        return self.ts.serialize_ros1(out, MSGTYPE[typ])


def _write_mcd_bags(paths, out, seq, cfg, sensors, generated=None):
    """generated: {sensor: iterator of (t_ns, colour msg, depth())} for depth
    made from the lidar or the map, merged in by time."""
    generated = generated or {}
    rw = _Ros1Writer()
    bags = {}
    for name, s in sensors.items():
        key = "merged" if cfg.get("merged", True) else s.get("bag", name)
        bags.setdefault(key, []).append(name)
    counts, stamps, files = {}, {}, []
    for suffix, names in bags.items():
        p = os.path.join(out, f"{seq}_{suffix}.bag")
        if os.path.exists(p):
            os.remove(p)
        w = rw.Writer(p)
        if cfg.get("compress"):
            w.set_compression(rw.Writer.CompressionFormat.BZ2)
        w.open()
        by_topic, made = {}, {}
        for name in names:
            s = sensors[name]
            conn = w.add_connection(s["out_topic"], MSGTYPE[s["type"]], typestore=rw.ts)
            if name in generated:
                made[name] = conn
            else:
                by_topic.setdefault(s["topic"], []).append((name, conn))
            counts[name], stamps[name] = 0, []

        def recorded():
            if by_topic:
                for m in read(paths, list(by_topic)):
                    yield stamp_ns(m.ros_msg), 0, m

        def rendered(name):
            for t, _, depth in generated[name]:
                got = depth()
                if got is not None:
                    yield t, 1, (name, got[0])

        for t, kind, item in heapq.merge(recorded(), *(rendered(n) for n in made),
                                         key=lambda x: x[0]):
            if kind == 0:
                for name, conn in by_topic[item.channel.topic]:
                    w.write(conn, t, rw.convert(sensors[name], name, item.ros_msg, t))
                    counts[name] += 1
                    stamps[name].append(t)
            else:
                name, D = item
                s = sensors[name]
                img = rw.image(t, np.clip(np.round(D * 1000.0), 0, 65535).astype(np.uint16),
                               "16UC1", s.get("out_frame", name), s["out_topic"])
                w.write(made[name], t, rw.ts.serialize_ros1(img, MSGTYPE["depth"]))
                counts[name] += 1
                stamps[name].append(t)
                if counts[name] % 1000 == 0:
                    log.info(f"  {name}: {counts[name]} depth frames")
        w.close()
        files.append(p)
        log.info(f"  {p}: " + ", ".join(f"{n} {counts[n]}" for n in names))
        for n in names:
            if counts[n] == 0:
                log.warning(f"  no messages on {sensors[n]['topic']} ({n})")
    return files, counts, stamps


def _write_mcd_groundtruth(out, pose_cfg, interp, stamps):
    at = pose_cfg.get("sample_at")
    if at:
        sample = list(enumerate(stamps.get(at, [])))
        if not sample:
            raise ConvertError(f"pose.sample_at '{at}' has no messages")
    else:
        step = int(1e9 / float(pose_cfg.get("rate_hz", 10.0)))
        sample = list(enumerate(range(int(interp.ts[0]), int(interp.ts[-1]) + 1, step)))
    rows, tum = [], []
    for num, t in sample:
        Tb = interp.at(t)
        if Tb is None:
            continue
        q = Rotation.from_matrix(Tb[:3, :3]).as_quat()
        x, y, z = Tb[:3, 3]
        rows.append(f"{num}\t{t*1e-9:.6f}\t{x:.12f}\t{y:.12f}\t{z:.12f}\t"
                    f"{q[0]:.12f}\t{q[1]:.12f}\t{q[2]:.12f}\t{q[3]:.12f}")
        tum.append(tum_line(t, Tb))
    gt_dir = os.path.join(out, "groundtruth")
    os.makedirs(gt_dir, exist_ok=True)
    with open(os.path.join(gt_dir, "pose_inW.csv"), "w") as f:
        f.write("num\tt\tx\ty\tz\tqx\tqy\tqz\tqw\n" + "\n".join(rows) + "\n")
    write_tum(os.path.join(out, "gt_tum.txt"), tum)
    log.info(f"  groundtruth/pose_inW.csv + gt_tum.txt: {len(rows)} poses")
    return len(rows)


def _write_mcd_calibration(out, seq, body_frame, sensors, done, how, paths):
    def mat(Tm, ind):
        if Tm is None:
            return f"{ind}T: null   # MISSING: add it to the config\n"
        return f"{ind}T:\n" + "".join(
            f"{ind}  - [{', '.join(f'{v:.9f}' for v in r)}]\n" for r in Tm)
    lines = [f"# MCD-style calibration for {seq}, written by mcap_convert.py\n",
             f"# body frame = {body_frame} (the frame pose_inW.csv describes)\n",
             "# T is T_body_sensor: p_body = R * p_sensor + t\n", "body:\n"]
    for name, s in sensors.items():
        lines.append(f"  {name}:\n    rostopic: {s['out_topic']}\n")
        if s["type"] in ("image", "depth") and s.get("info"):
            info = first_msg(paths, s["info"])
            if info is not None:
                K = intrinsics_from_info(info)
                d = [0.0] * 4 if s.get("rectified", True) else K.dist
                lines.append(
                    f"    resolution: [{K.width}, {K.height}]\n"
                    "    camera_model: pinhole\n"
                    f"    intrinsics: [{', '.join(f'{v:.6f}' for v in K.k4)}]\n"
                    "    distortion_model: radtan\n"
                    f"    distortion_coeffs: [{', '.join(f'{v:.9f}' for v in d)}]\n")
        lines.append(f"    # extrinsic: {how[name]}\n")
        lines.append(mat(done[name], "    "))
    with open(os.path.join(out, "calibration.yaml"), "w") as f:
        f.write("".join(lines))
    log.info("  calibration.yaml")


def _write_mcd_sensor_yamls(out, cfg, sensors, done, paths):
    """camera.yaml / imu.yaml / lidar.yaml (OpenCV %YAML:1.0, MCD/CLIC style).
    Extrinsics are sensor -> IMU, relative to the IMU chosen by `yaml.imu`."""
    ycfg = cfg.get("yaml") or {}

    def pick(kind, preferred):
        for n in [preferred] + [n for n, s in sensors.items() if s["type"] == kind]:
            if n in sensors and sensors[n]["type"] == kind and done.get(n) is not None:
                return n
        return None

    imu = pick("imu", ycfg.get("imu"))
    if imu is None:
        log.warning("  camera/imu/lidar yamls skipped: no IMU with a known extrinsic")
        return []
    if ycfg.get("imu") and imu != ycfg["imu"]:
        log.warning(f"  yamls: '{ycfg['imu']}' has no extrinsic, using '{imu}' as the IMU")
    T_imu_body = np.linalg.inv(done[imu])

    def block(key, Tm, comment, ind=""):
        T = T_imu_body @ Tm
        T = np.where(np.abs(T) < 1e-12, 0.0, T)
        rows = [", ".join(f"{v:.12g}" for v in r) for r in T[:3, :3]]
        pad = ind + " " * 11
        return (f"{ind}{key}:  # {comment}\n"
                f"{ind}    Trans: [{', '.join(f'{v:.12g}' for v in T[:3, 3])}]\n"
                f"{ind}    Rot: [{rows[0]},\n{pad}{rows[1]},\n{pad}{rows[2]}]\n")

    written = []
    cam = pick("image", ycfg.get("camera"))
    if cam and sensors[cam].get("info"):
        info = first_msg(paths, sensors[cam]["info"])
        if info is not None:
            K = intrinsics_from_info(info)
            d = [0.0] * 4 if sensors[cam].get("rectified", True) else K.dist
            with open(os.path.join(out, "camera.yaml"), "w") as f:
                f.write("%YAML:1.0\n\n"
                        f"image_topic: {sensors[cam]['out_topic']}\n\n"
                        f"image_width: {K.width}\nimage_height: {K.height}\n"
                        f"cam_fx: {float(K.fx)!r}\ncam_fy: {float(K.fy)!r}\n"
                        f"cam_cx: {float(K.cx)!r}\ncam_cy: {float(K.cy)!r}\n"
                        + "".join(f"cam_d{i}: {float(v)!r}\n" for i, v in enumerate(d))
                        + "cam_d4: 0.0\n\n"
                        + block("CameraExtrinsics", done[cam], f"camera to imu ({imu}_T_{cam})")
                        + "\nimg_time_offset: 0.0\n")
            written.append("camera.yaml")
            log.info(f"  camera.yaml ({cam})")

    with open(os.path.join(out, "imu.yaml"), "w") as f:
        f.write("%YAML:1.0\n\n"
                f"imu_topic: {sensors[imu]['out_topic']}\n\n"
                "imu_excite_threshold: 0.5\ngravity_mag: 9.81\n"
                "if_normalized: false\nif_use_init_bg: false\n\n"
                + block("IMUExtrinsics", done[imu], f"imu to imu ({imu} is the body frame)"))
    written.append("imu.yaml")
    log.info(f"  imu.yaml ({imu})")

    lidars = [n for n, s in sensors.items() if s["type"] == "pointcloud" and done.get(n) is not None]
    if lidars:
        parts = ["%YAML:1.0\n\n"]
        for i, n in enumerate(lidars):
            parts.append(f"lidar{i}:\n    topic: {sensors[n]['out_topic']}\n"
                         "    is_livox: false\n"
                         + block("Extrinsics", done[n], f"lidar to imu ({imu}_T_{n})", "    "))
        parts.append(f"num_lidars: {len(lidars)}\nlidar_timestamp_end: "
                     f"{str(bool(ycfg.get('lidar_timestamp_end', False))).lower()}\n")
        with open(os.path.join(out, "lidar.yaml"), "w") as f:
            f.write("".join(parts))
        written.append("lidar.yaml")
        log.info(f"  lidar.yaml ({', '.join(lidars)})")
    return written


def to_mcd(bag, out, config=None, *, pose_tum=None, tum_frame=None, depth_source=None,
           map=None, map_voxel=None, lidar=None, lidar_scans=None, splat=1.0, gpu=True):
    """MCD dataset layout.

    bag, out   recording (.mcap or bag dir, or list of .mcap) / output dir
    config     None = built-in mobile_1 config (DEFAULT_MCD_CONFIG), a yaml path,
               or a dict with the same structure. Top-level keys:
                 sequence, compress, merged,
                 pose{topic | tum + tum_frame, frame, sample_at|rate_hz, bag},
                 yaml{imu, camera, lidar_timestamp_end}, sensors{name: {...}}
    The keyword arguments override the config: pose_tum / tum_frame the pose
    source, depth_source (topic | lidar | map) / map / map_voxel / lidar /
    lidar_scans every depth sensor.
    Returns a dict with bags, message counts, ground-truth rows, missing extrinsics.
    """
    paths = _paths(bag)
    cfg = load_mcd_config(config)
    pose_cfg = dict(cfg["pose"])
    sensors = {k: dict(v) for k, v in cfg["sensors"].items()}
    dp = pass_of(bag) or pass_of(paths[0])
    seq = (cfg.get("sequence") or (f"{dp[1]}_{dp[0]}" if dp else None)
           or os.path.basename(os.path.normpath(out)))
    body_frame = pose_cfg["frame"]
    os.makedirs(out, exist_ok=True)

    def lidar_topic():
        t = lidar or next((v["topic"] for v in sensors.values()
                           if v["type"] == "pointcloud"), None)
        if t is None:
            raise ConvertError("no pointcloud sensor in the config and no lidar topic given")
        return t

    log.info("pass 1: poses / tf_static / extrinsics")
    edges = build_tf_static(paths)
    if edges:
        log.info(f"  tf_static frames: {sorted({f for e in edges for f in e})}")
    tum = pose_tum or pose_cfg.get("tum")
    if tum:
        base = PoseInterp.from_tum(tum)
        tf = tum_frame or pose_cfg.get("tum_frame")
        if not tf:
            m = first_msg(paths, lidar_topic())
            tf = m.header.frame_id.lstrip("/") if m is not None else None
        T = lookup_static(edges, tf, body_frame) if tf else None
        if T is None:
            raise ConvertError(f"{tum}: no /tf_static link from its frame {tf!r} to the body "
                               f"frame {body_frame}; give pose.tum_frame / --tum-frame")
        interp = FramePoses(base, T)
        log.info(f"  {len(base)} poses from {tum} ({tf}), chained to {body_frame}: "
                 f"t={np.round(T[:3, 3], 4).tolist()} m")
    else:
        interp = _pose_source(paths, pose_cfg.get("bag"), pose_cfg["topic"])
        log.info(f"  {len(interp)} poses on {pose_cfg['topic']} (body = {body_frame}), "
                 f"max gap {interp.max_gap*1e3:.0f} ms")
    # depth made from the lidar or the map is made for every image of its
    # camera: it takes the camera's topic, intrinsics and (if it has none) extrinsic
    made = {}
    for name, s in sensors.items():
        src = depth_source or s.get("source", "topic")
        if s["type"] != "depth" or src in ("topic", "zed"):
            continue
        if src not in ("lidar", "map"):
            raise ConvertError(f"sensor '{name}': source {src!r} (topic | lidar | map)")
        cam = s.get("camera") or s.get("same_as")
        if cam not in sensors or sensors[cam]["type"] != "image" or not sensors[cam].get("info"):
            raise ConvertError(f"depth sensor '{name}' from {src} needs camera: <an image "
                               "sensor with info>")
        s["topic"], s["info"] = sensors[cam]["topic"], sensors[cam]["info"]
        made[name] = (src, cam)
    done, how = {}, {}
    for name, s in sensors.items():
        done[name], how[name] = resolve_extrinsic(name, s, body_frame, edges, done, paths)
        if done[name] is None and name in made and done.get(made[name][1]) is not None:
            done[name], how[name] = done[made[name][1]], f"same as {made[name][1]}"
        flag = "" if done[name] is not None else "  <-- MISSING, T left empty"
        log.info(f"  {name:<14} {how[name]}{flag}")

    generated, renderers = {}, {}
    for name, (src, cam) in made.items():
        s = sensors[name]
        if done.get(cam) is None:
            raise ConvertError(f"depth sensor '{name}': camera '{cam}' has no extrinsic")
        info, img = first_msg(paths, s["info"]), first_msg(paths, s["topic"])
        if info is None or img is None:
            raise ConvertError(f"depth sensor '{name}': no messages on {s['topic']} / {s['info']}")
        K = intrinsics_from_info(info)
        dmin, dmax = float(s.get("min", 0.2)), float(s.get("max", 10.0))
        T_bc = done[cam]

        def cam_pose(t_ns, T_bc=T_bc):
            T = interp.at(t_ns)
            return None if T is None else T @ T_bc
        if src == "lidar":
            lt = lidar or s.get("lidar") or lidar_topic()
            T_cl = _lidar_extrinsic(paths, edges, img.header.frame_id.lstrip("/"), lt)
            n = int(lidar_scans or s.get("scans", 1))
            generated[name] = iter_lidar_depth(paths, s["topic"], lt, T_cl, cam_pose, K,
                                               n_scans=n, dmin=dmin, dmax=dmax)
            how[name] += f"; depth from {lt}, {n} scan(s) per {cam} image"
        else:
            mp = map or s.get("map")
            if not mp:
                raise ConvertError(f"depth sensor '{name}' from the map needs map (--map)")
            vx = float(map_voxel or s.get("voxel", 0.02))
            key = (mp, vx, K.k4, K.width, K.height, dmin, dmax)
            if key not in renderers:
                pts, nrm = load_map(mp, vx)
                renderers[key] = MapDepth(pts, nrm, K, vx, dmin, dmax, splat, gpu=gpu)
            generated[name] = iter_map_depth(paths, s["topic"], cam_pose, renderers[key])
            how[name] += f"; depth rendered from {mp} at {vx} m for every {cam} image"
        log.info(f"  {name:<14} {how[name]}")

    log.info("pass 2: writing bags")
    files, counts, stamps = _write_mcd_bags(paths, out, seq, cfg, sensors, generated)
    n_gt = _write_mcd_groundtruth(out, pose_cfg, interp, stamps)
    _write_mcd_calibration(out, seq, body_frame, sensors, done, how, paths)
    yamls = _write_mcd_sensor_yamls(out, cfg, sensors, done, paths)
    missing = [n for n in sensors if done[n] is None]
    if missing:
        log.warning(f"\n  extrinsic missing for: {', '.join(missing)} — their T is null "
                    "in calibration.yaml; the bags are written anyway")
    return {"format": "mcd", "out": out, "bags": files, "messages": counts,
            "groundtruth_rows": n_gt, "yamls": yamls, "missing_extrinsics": missing,
            "extrinsics": done}


# =========================================================================== #
# 9. Converter: MCGS-SLAM (stereo, Waymo-example layout)
# =========================================================================== #

def to_mcgs(bag, out, *,
            left="/mobile_1/zed/left/image_rect_color",
            right="/mobile_1/zed/right/image_rect_color",
            left_info="/mobile_1/zed/left/camera_info",
            right_info="/mobile_1/zed/right/camera_info",
            pose="/mobile_1/global_pose", pose_bag=None,
            sync_tol_ms=10.0, every=1, max_frames=0, jpeg_quality=95):
    """Rectified stereo sequence for MCGS-SLAM.

    bag, out              recording / output dir
    left, right           rectified left / right image topics (left = cam0)
    left_info, right_info their CameraInfo; the baseline comes from the right P
                          (cross-checked against /tf_static when the frames are there)
    pose, pose_bag        pose topic of the LEFT camera in world (written to gt_poses.txt)
    sync_tol_ms           max left/right stamp difference (capped at 19 ms: MCGS asserts < 20)
    every, max_frames     keep every Nth pair / stop after N pairs (0 = all)
    jpeg_quality          JPEG quality
    Returns a dict with pairs, ground-truth rows, baseline and intrinsics.
    """
    paths = _paths(bag)
    log.info("pass 1: intrinsics / baseline / poses")
    li, ri = first_msg(paths, left_info), first_msg(paths, right_info)
    if li is None:
        raise ConvertError(f"no messages on {left_info}")
    if ri is None or first_msg(paths, right) is None:
        raise ConvertError(
            f"no messages on {right} / {right_info}.\n"
            "MCGS-SLAM needs the right image too: record the ZED wrapper's right rectified\n"
            "image and its camera_info, or pass other topics.")
    KL, KR = intrinsics_from_info(li), intrinsics_from_info(ri)
    if (KL.width, KL.height) != (KR.width, KR.height):
        raise ConvertError(f"left {KL.width}x{KL.height} and right "
                           f"{KR.width}x{KR.height} resolutions differ")
    edges = build_tf_static(paths)
    lf, rf = li.header.frame_id.lstrip("/"), ri.header.frame_id.lstrip("/")
    T_lr = lookup_static(edges, lf, rf) if lf != rf else None
    B, src = stereo_baseline(ri), "right camera_info P"
    if B is None:
        if T_lr is None:
            raise ConvertError("no stereo baseline: right camera_info P has none and "
                               f"/tf_static does not link {lf} -> {rf}")
        B, src = float(np.linalg.norm(T_lr[:3, 3])), f"/tf_static {lf} -> {rf}"
    log.info(f"  {KL.width}x{KL.height}  left fx={KL.fx:.3f}  right fx={KR.fx:.3f}")
    log.info(f"  baseline {B*100:.2f} cm ({src})")
    if T_lr is not None and src.startswith("right"):
        log.info(f"  cross-check /tf_static {lf} -> {rf}: "
                 f"{np.linalg.norm(T_lr[:3, 3])*100:.2f} cm")
    distort = any(abs(v) > 1e-6 for v in KL.dist + KR.dist)
    if distort:
        log.info("  note: non-zero distortion; written to calib.yml so MCGS undistorts")

    interp = _pose_source(paths, pose_bag, pose)
    log.info(f"  {len(interp)} poses on {pose} (taken as the LEFT camera pose)")

    ldir, rdir = os.path.join(out, "left"), os.path.join(out, "right")
    for d in (ldir, rdir):
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, "*.jpg")):
            os.remove(f)

    log.info("pass 2: writing stereo pairs")
    q = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    gt, n, kept = [], 0, 0
    tol = int(min(sync_tol_ms, 19.0) * 1e6)
    for lt, lmsg, _, rmsg in iter_synced(paths, left, right, tol):
        n += 1
        if (n - 1) % every:
            continue
        name = f"{lt*1e-9:.6f}.jpg"                      # MCGS parses the number
        cv2.imwrite(os.path.join(ldir, name), to_bgr(decode_image(lmsg), lmsg.encoding), q)
        cv2.imwrite(os.path.join(rdir, name), to_bgr(decode_image(rmsg), rmsg.encoding), q)
        kept += 1
        T = interp.at(lt)
        if T is not None:
            gt.append(tum_line(lt, T))
        if kept % 500 == 0:
            log.info(f"  {kept} pairs")
        if max_frames and kept >= max_frames:
            break
    if not kept:
        raise ConvertError("no synchronised left/right pairs (check sync_tol_ms)")
    write_tum(os.path.join(out, "gt_poses.txt"), gt)

    def row(K):
        vals = list(K.k4) + (list(K.dist) if distort else [])
        return "[" + ", ".join(f"{float(v)!r}" for v in vals) + "]"
    with open(os.path.join(out, "calib.yml"), "w") as f:
        f.write("intrinsic:  [\n"
                f"    {row(KL)},  # left\n"
                f"    {row(KR)},  # right\n"
                "]\n\n"
                "camera:     'pinhole'\n\n"
                "# T_cam1_cam0 [tx, ty, tz, qx, qy, qz, qw]: rectified right camera,\n"
                f"# {B*100:.2f} cm along the left camera's +x\n"
                f"baseline:   [{-B:.6f}, 0, 0, 0, 0, 0, 1]\n\n"
                "T_cami_cam0:  [[0, 0, 0, 0, 0, 0, 1],\n"
                f"               [{-B:.6f}, 0, 0, 0, 0, 0, 1]]\n\n"
                "timescale:    1\n")
    log.info(f"\n  {kept} stereo pairs -> {ldir}, {rdir}\n"
             f"  gt_poses.txt: {len(gt)} poses (TUM, left camera in world)\n  calib.yml")
    log.info(f"\nrun:\n  python demo.py --calib {os.path.join(out, 'calib.yml')} \\\n"
             f"      --imagedir {ldir} {rdir} --stride 1 --output output/"
             f"{os.path.basename(os.path.normpath(out))}")
    return {"format": "mcgs", "out": out, "pairs": kept, "groundtruth_rows": len(gt),
            "baseline_m": B, "baseline_source": src,
            "intrinsics_left": KL.k4, "intrinsics_right": KR.k4,
            "resolution": (KL.width, KL.height)}


# =========================================================================== #
# 10. Inspection and registry
# =========================================================================== #

def inspect_bag(bag, info="/mobile_1/zed/left/camera_info",
                color="/mobile_1/zed/left/image_rect_color", pose=None):
    """Print and return topics, intrinsics, colour frame_id, /tf_static frames and
    the frames seen on the pose topic — what you need to pick the options above."""
    paths = _paths(bag)
    from mcap.reader import make_reader
    topics = {}
    for p in paths:
        with open(p, "rb") as f:
            summary = make_reader(f).get_summary()
        for cid, ch in summary.channels.items():
            sch = summary.schemas.get(ch.schema_id)
            n = summary.statistics.channel_message_counts.get(cid, 0) if summary.statistics else 0
            name = sch.name if sch else "?"
            topics[ch.topic] = (name, topics.get(ch.topic, (name, 0))[1] + n)
    log.info("topics:")
    for t in sorted(topics):
        log.info(f"  {t:<50} {topics[t][0]:<40} {topics[t][1]}")
    res = {"topics": topics}
    ci = first_msg(paths, info)
    if ci is not None:
        K = intrinsics_from_info(ci)
        res["intrinsics"] = K
        log.info(f"{info}: {K.width}x{K.height} fx={K.fx:.3f} fy={K.fy:.3f} "
                 f"cx={K.cx:.3f} cy={K.cy:.3f}")
    m = first_msg(paths, color)
    res["color_frame"] = m.header.frame_id.lstrip("/") if m is not None else None
    log.info(f"{color} frame_id: {res['color_frame']}")
    edges = build_tf_static(paths)
    res["tf_static_frames"] = sorted({f for e in edges for f in e})
    log.info(f"/tf_static frames: {res['tf_static_frames']}")
    if pose:
        _, _, _, frames = load_poses(paths, pose)
        res["pose_frames"] = sorted(frames)
        log.info(f"{pose} frames: {res['pose_frames']}")
    return res


FORMATS = {
    "replica": to_replica,   # SplaTAM / NICE-SLAM / Point-SLAM / GradSLAM RGB-D
    "mcd": to_mcd,           # MCD dataset: ROS1 bag + ground truth + calibration yamls
    "mcgs": to_mcgs,         # MCGS-SLAM stereo sequence
}


# =========================================================================== #
# 11. Command line
# =========================================================================== #

def _cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help=".mcap file or rosbag2 directory")
    ap.add_argument("out", nargs="?",
                    help="output directory (default: data/processed/<date>/<pass>/datasets/"
                         "<machine>/<format>[_<depth>]_<pass>_<date>, from the bag's place)")
    ap.add_argument("--format", choices=sorted(FORMATS), default="replica")
    ap.add_argument("--inspect", "--list-frames", dest="inspect", action="store_true",
                    help="print topics, intrinsics and frame names, then exit")
    ap.add_argument("--quiet", action="store_true", help="only warnings and errors")

    g = ap.add_argument_group("topics (replica, mcgs)")
    g.add_argument("--color", default="/mobile_1/zed/left/image_rect_color",
                   help="colour image; the LEFT image for mcgs")
    g.add_argument("--info", default="/mobile_1/zed/left/camera_info",
                   help="colour / left CameraInfo")
    g.add_argument("--depth", default="/mobile_1/zed/depth/depth_registered",
                   help="[replica, --depth-source zed] depth aligned to --color")
    g.add_argument("--right", default="/mobile_1/zed/right/image_rect_color",
                   help="[mcgs] rectified right image")
    g.add_argument("--right-info", default="/mobile_1/zed/right/camera_info",
                   help="[mcgs] right CameraInfo (baseline)")
    g.add_argument("--pose", default="/mobile_1/global_pose", help="pose topic")
    g.add_argument("--pose-bag", default=None,
                   help="read --pose from this bag instead (same clock as the images)")
    g.add_argument("--pose-tum", default=None,
                   help="poses from a TUM file instead (e.g. traj_lidar_refined.txt); "
                        "replica, mcd")
    g.add_argument("--tum-frame", default=None,
                   help="the frame the --pose-tum poses are of (default: the frame of "
                        "--lidar, which GLIM / 01a trajectories are)")

    g = ap.add_argument_group("camera pose (replica: exactly one)")
    g.add_argument("--pose-is-camera", action="store_true",
                   help="the pose topic already is the camera optical pose")
    g.add_argument("--body-frame", default=None,
                   help="frame the pose describes; body->camera chained from /tf_static")
    g.add_argument("--extrinsic", nargs=7, type=float, default=None,
                   metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
                   help="T_body_cam (camera optical frame in the body frame)")
    g.add_argument("--cam-frame", default=None,
                   help="camera optical frame (default: frame_id of --color)")

    g = ap.add_argument_group("depth (replica; mcd: overrides the config's depth sensors)")
    g.add_argument("--depth-source", choices=["lidar", "map", "zed"], default=None,
                   help="lidar: project --lidar into the colour camera (replica default); "
                        "map: render the --map cloud; zed: the --depth image")
    g.add_argument("--map", default=None,
                   help="[map] the map cloud (.pcd/.ply), in the frame of --pose-tum")
    g.add_argument("--map-voxel", type=float, default=0.02,
                   help="[map] voxel the map is rendered at (m)")
    g.add_argument("--splat", type=float, default=1.0,
                   help="[map] surfel radius in voxels (bigger closes more holes)")
    g.add_argument("--cpu", action="store_true", help="[map] render on the CPU")
    g.add_argument("--lidar", default="/mobile_1/ouster/points", help="PointCloud2 topic")
    g.add_argument("--lidar-scans", type=int, default=1,
                   help="scans rendered into each frame, nearest in time (more = denser)")
    g.add_argument("--lidar-max-dt-ms", type=float, default=100.0,
                   help="drop a frame when no scan is this close to it")
    g.add_argument("--lidar-extrinsic", nargs=7, type=float, default=None,
                   metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
                   help="T_cam_lidar: the cloud frame in the camera optical frame "
                        "(default: from /tf_static)")
    g.add_argument("--occlusion-px", type=int, default=7,
                   help="hidden-point filter window in pixels (0 = off)")
    g.add_argument("--occlusion-margin", type=float, default=0.1,
                   help="drop a point this fraction farther than the nearest in the window")

    g = ap.add_argument_group("frames")
    g.add_argument("--sync-tol-ms", type=float, default=10.0,
                   help="max stamp difference for colour/depth or left/right pairs")
    g.add_argument("--max-pose-dt-ms", type=float, default=0.0,
                   help="[replica] keep frames within this of a real pose sample (0 = off)")
    g.add_argument("--every", type=int, default=1, help="keep every Nth image/pair")
    g.add_argument("--max-frames", type=int, default=0, help="stop after N (0 = all)")
    g.add_argument("--jpeg-quality", type=int, default=95)

    g = ap.add_argument_group("depth (replica)")
    g.add_argument("--depth-scale", type=float, default=6553.5,
                   help="PNG units per metre: 6553.5 as Replica (0-10 m), 1000 = mm")
    g.add_argument("--depth-input-scale", type=float, default=0.001,
                   help="[zed] metres per unit of integer input depth")
    g.add_argument("--depth-min", type=float, default=0.2)
    g.add_argument("--depth-max", type=float, default=10.0)

    g = ap.add_argument_group("mcd")
    g.add_argument("--config", default=None,
                   help="sensor config yaml (default: built-in mobile_1, DEFAULT_MCD_CONFIG)")
    a = ap.parse_args()

    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO,
                        format="%(message)s")
    try:
        if a.inspect:
            inspect_bag(a.bag, info=a.info, color=a.color, pose=a.pose)
            return
        if not a.out:
            depth = None
            if a.format == "replica":
                depth = a.depth_source or "lidar"
            elif a.format == "mcd":
                srcs = [v.get("source", "topic") for v in
                        (load_mcd_config(a.config)["sensors"] or {}).values()
                        if v.get("type") == "depth"]
                depth = a.depth_source or (srcs[0] if srcs else None)
                depth = {"topic": "zed"}.get(depth, depth)
            a.out = dataset_dir(a.bag, a.format, depth, topic_machine(a.color))
            log.info(f"output: {a.out}")
        if a.format == "replica":
            to_replica(a.bag, a.out, color=a.color, depth=a.depth, info=a.info,
                       pose=a.pose, pose_bag=a.pose_bag, pose_tum=a.pose_tum,
                       pose_is_camera=a.pose_is_camera,
                       body_frame=a.body_frame or a.tum_frame, extrinsic=a.extrinsic,
                       cam_frame=a.cam_frame, depth_source=a.depth_source or "lidar",
                       map=a.map, map_voxel=a.map_voxel, splat=a.splat, gpu=not a.cpu,
                       lidar=a.lidar,
                       lidar_scans=a.lidar_scans, lidar_max_dt_ms=a.lidar_max_dt_ms,
                       lidar_extrinsic=a.lidar_extrinsic, occlusion_px=a.occlusion_px,
                       occlusion_margin=a.occlusion_margin,
                       sync_tol_ms=a.sync_tol_ms, max_pose_dt_ms=a.max_pose_dt_ms,
                       depth_scale=a.depth_scale, depth_input_scale=a.depth_input_scale,
                       depth_min=a.depth_min, depth_max=a.depth_max, every=a.every,
                       max_frames=a.max_frames, jpeg_quality=a.jpeg_quality)
        elif a.format == "mcd":
            to_mcd(a.bag, a.out, config=a.config, pose_tum=a.pose_tum, tum_frame=a.tum_frame,
                   depth_source=a.depth_source, map=a.map,
                   map_voxel=a.map_voxel if a.depth_source == "map" or a.map else None,
                   lidar=None if a.lidar == "/mobile_1/ouster/points" else a.lidar,
                   lidar_scans=a.lidar_scans if a.lidar_scans != 1 else None,
                   splat=a.splat, gpu=not a.cpu)
        else:
            to_mcgs(a.bag, a.out, left=a.color, right=a.right, left_info=a.info,
                    right_info=a.right_info, pose=a.pose, pose_bag=a.pose_bag,
                    sync_tol_ms=a.sync_tol_ms, every=a.every, max_frames=a.max_frames,
                    jpeg_quality=a.jpeg_quality)
    except ConvertError as e:
        sys.exit(f"\nerror: {e}")


if __name__ == "__main__":
    _cli()

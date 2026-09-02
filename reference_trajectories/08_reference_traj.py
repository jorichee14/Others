#!/usr/bin/env python3
"""
STAGE 08 - per-sensor reference trajectories in `map` for a coop bag.

Stage 06 measured where each camera STARTED (the opening dwell on its board).
This stage turns whole trajectories into the map frame and corrects them with
the absolute information each sensor can see.

MOBILE_1 WORKFLOW (Ouster + ZED on one rigid body) - run these two tracks:

  1. lidar_icp   every Ouster scan is registered to the FROZEN anchored map by
                 point-to-plane ICP. The session-start pose seeds scan 0
                 (through T_lidar_camera); afterwards each scan is seeded by
                 the lidar's OWN previous poses (constant velocity), so after
                 scan 0 nothing from the ZED enters the track. Boards are never
                 used either: the track is fiducial-free and odometry-free.
                 A scan whose ICP moved too far from its seed is retried from
                 the other seed and with wide gates before it is declared
                 unregistered. Per-scan quality goes to quality_<name>.csv/png.
                 The track ALSO writes the anchored ZED odometry at the same
                 stamps and prints the gap between the two, plus the per-step
                 ZED-vs-lidar disagreement that separates a ZED jump (one big
                 step) from drift (a run of small ones).

  2. arms        with "cloud_source": "lidar": three corrected trajectories of
                 the ZED optical frame from ONE estimator (same nodes, same
                 odometry factors, same solver):
                   A_icp     ZED odom + lidar point-to-plane map factors
                             (geometry only, no boards)
                   B_boards  ZED odom + board factors + session-anchor prior
                             (started from the anchored odometry: this is
                             "board sightings correct the ZED odom")
                   C_joint   everything: lidar + boards + odom
                 and the ablation table whose off-diagonal cells are
                 independent checks (A never saw a board, B never saw the map).
                 The lidar clouds and the chained-ICP poses come from track 1
                 (no second ICP pass); only the CLOUDS are re-registered inside
                 the graph, the chained POSES are initialisation-only (their
                 errors are correlated, feeding them in as measurements lets
                 the chain's drift out-vote the boards - measured).

  Then the stage prints one comparison table for the rig: lidar chained ICP,
  anchored ZED odom, A, B, C - all in the ZED left optical frame, all at the
  lidar stamps - and writes paths.png + compare_<rig>.csv.

OTHER TRACK TYPES (unchanged)
  arms with "cloud_source": "depth" (default) - any depth camera (mobile_2
              RealSense, or the ZED depth): a chained depth-ICP pass seeds the
              same three-arm graph. cam_extrinsic_xyzquat REQUIRED.
  rgbd_icp    depth-only chained ICP to the map (no boards, no graph).
  cam_boards  odometry + boards pose graph only (no depth, no lidar).

TRANSFORMS (the whole stage lives or dies on these - see README_08_mobile1.md)
  X   = cam_extrinsic_xyzquat = T_child_cam: pose of the camera OPTICAL frame
        in the odometry CHILD frame (the frame printed as child_frame_id).
        T_map_cam(t) = T_map_odom @ T_odom_child(t) @ X
  A   = session anchor (stage 06) = T_map_cam at the dwell -> T_map_odom =
        A @ inv(T_odom_child(t_dwell) @ X)
  T_lidar_camera (calibration.json) is used as the pose of the camera in the
        lidar frame: T_map_lidar = T_map_cam @ inv(T_lidar_camera). The
        cross-check at the end scores BOTH conventions against the board
        sightings and says which one is centimetres.
  Odometry increments live in the child frame and are conjugated into the
        state frame: lidar state -> T_cl = X @ inv(T_lidar_camera);
        camera state -> X.

BOARD SIGHTINGS
  Detected with the pipeline's own Board.detect + frame_fix (same convention as
  stages 03/06). Instances of a shared design are resolved by MAP position:
  the best available trajectory predicts where the sighted board is in map,
  and the nearest surveyed instance within instance_radius claims it.

CONFIG ("08_reference" stage block; see the sample at the bottom of this file
and pipeline_config_08_mobile1.json)
  python3 08_reference_traj.py [pipeline_config.json]
"""
import os
import sys
import json
import math
import time
import numpy as np

from pipeline_common import load_pipeline, R_to_q
from pipeline_boards import Board, read_bag, pick_intrinsics

# --------------------------------------------------------------------------- #
# SE(3) helpers (validated: exact right-Jacobian inverse, not the small-angle
# form - pose-graph rotation residuals are large at iteration 0 and the
# approximation stalls Gauss-Newton)
# --------------------------------------------------------------------------- #
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree
from scipy import sparse, ndimage
from scipy.sparse.linalg import spsolve


def Rt(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


def inv(T):
    R = T[:3, :3]; o = np.eye(4); o[:3, :3] = R.T; o[:3, 3] = -R.T @ T[:3, 3]; return o


def hat(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def log_R(R):
    return Rot.from_matrix(R).as_rotvec()


def exp_r(w):
    return Rot.from_rotvec(w).as_matrix()


def jr_inv(w):
    th = float(np.linalg.norm(w)); W = hat(w)
    if th < 1e-6:
        return np.eye(3) + 0.5 * W
    a = 1.0 / th ** 2 - (1.0 + math.cos(th)) / (2.0 * th * math.sin(th))
    return np.eye(3) + 0.5 * W + a * (W @ W)


def apply(T, P):
    return np.asarray(P) @ T[:3, :3].T + T[:3, 3]


def compose_all(Ts, T_right):
    """(N,4,4) @ (4,4) for every pose - e.g. lidar poses -> camera poses."""
    return np.asarray(Ts) @ np.tile(T_right, (len(Ts), 1, 1))


def interp_traj(ts, Ts, tq):
    from scipy.spatial.transform import Slerp
    tq = np.clip(tq, ts[0], ts[-1])
    i = np.clip(np.searchsorted(ts, tq) - 1, 0, len(ts) - 2)
    d = ts[i + 1] - ts[i]
    a = np.where(d > 0, (tq - ts[i]) / np.where(d > 0, d, 1), 0.0)
    sl = Slerp(ts, Rot.from_matrix(Ts[:, :3, :3]))
    out = np.tile(np.eye(4), (len(tq), 1, 1))
    out[:, :3, :3] = sl(tq).as_matrix()
    out[:, :3, 3] = Ts[i, :3, 3] * (1 - a)[:, None] + Ts[i + 1, :3, 3] * a[:, None]
    return out


def write_tum(path, ts, Ts):
    with open(path, "w") as f:
        for t, T in zip(ts, Ts):
            q = R_to_q(T[:3, :3])
            f.write("%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n"
                    % (t, T[0, 3], T[1, 3], T[2, 3], q[0], q[1], q[2], q[3]))
    print("  wrote %s (%d poses)" % (path, len(ts)))


def make_T_xyzq(v):
    return Rt(Rot.from_quat(v[3:7]).as_matrix(), np.asarray(v[0:3], float))


def rig_of(track_name):
    """'mobile_1_lidar' -> 'mobile_1': tracks of one rigid body share it."""
    p = track_name.split("_")
    return "_".join(p[:2]) if len(p) >= 2 else track_name


def traj_gap(Ta, Tb):
    """Per-sample translation (m) and rotation (rad) gap of two pose arrays
    given at the SAME stamps and expressed in the SAME body frame."""
    dt = np.linalg.norm(Ta[:, :3, 3] - Tb[:, :3, 3], axis=1)
    dr = np.array([np.linalg.norm(log_R(A[:3, :3].T @ B[:3, :3]))
                   for A, B in zip(Ta, Tb)])
    return dt, dr


def report_gap(label, ts, dt, dr, path_len=None):
    print("  %s: translation median %.1f cm  p95 %.1f cm  max %.1f cm | "
          "rotation median %.2f deg  max %.2f deg  (%d stamps)"
          % (label, np.median(dt) * 100, np.percentile(dt, 95) * 100,
             dt.max() * 100, math.degrees(np.median(dr)),
             math.degrees(dr.max()), len(dt)))
    qs = np.linspace(0, len(ts) - 1, 6).astype(int)
    print("     over time: " + "  ".join(
        "t=%.0fs %.0fcm" % (ts[i] - ts[0], dt[i] * 100) for i in qs))
    if path_len is not None and path_len > 1.0:
        print("     end gap %.1f cm over %.1f m of path = %.2f%% of distance "
              "travelled" % (dt[-1] * 100, path_len, 100 * dt[-1] / path_len))


def path_length(Ts):
    return float(np.sum(np.linalg.norm(np.diff(Ts[:, :3, 3], axis=0), axis=1)))


# --------------------------------------------------------------------------- #
# bag readers (odometry + point clouds; images go through pipeline_boards)
# --------------------------------------------------------------------------- #
def iter_topic(path, topic, stride=1, limit=None):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    try:
        from rosidl_runtime_py.utilities import get_message
    except ImportError:
        from rosidl_runtime_py.utility import get_message
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
           rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    if topic not in types:
        raise KeyError("%s not in bag; have %s..." % (topic, sorted(types)[:8]))
    cls = get_message(types[topic])
    f = rosbag2_py.StorageFilter(); f.topics = [topic]; r.set_filter(f)
    i = n = 0
    while r.has_next():
        _, data, t_bag = r.read_next()
        if i % stride:
            i += 1; continue
        i += 1
        m = deserialize_message(data, cls)
        h = getattr(m, "header", None)
        t = (h.stamp.sec + h.stamp.nanosec * 1e-9) if h is not None else t_bag * 1e-9
        yield t, m
        n += 1
        if limit and n >= limit:
            break


def topic_frame(bag, topic):
    """header.frame_id of the first message - the frame the data is IN."""
    for _, m in iter_topic(bag, topic, limit=1):
        h = getattr(m, "header", None)
        return h.frame_id if h is not None else "?"
    return "?"


def read_odom(bag, topic):
    ts, Ts, child, parent = [], [], None, None
    for t, m in iter_topic(bag, topic):
        p = m.pose.pose.position; o = m.pose.pose.orientation
        if child is None:
            child = getattr(m, "child_frame_id", "") or "?"
            parent = m.header.frame_id or "?"
        ts.append(t)
        Ts.append(Rt(Rot.from_quat([o.x, o.y, o.z, o.w]).as_matrix(),
                     np.array([p.x, p.y, p.z])))
    if not ts:
        raise SystemExit("no odometry on %s" % topic)
    ts = np.array(ts); Ts = np.array(Ts)
    print("  odom %s: %d poses, %.1f s, path %.1f m, frame '%s' -> "
          "child_frame_id '%s'"
          % (topic, len(ts), ts[-1] - ts[0], path_length(Ts), parent, child))
    print("  (cam_extrinsic_xyzquat must be T_%s_<optical>: "
          "ros2 run tf2_ros tf2_echo %s <optical frame>)" % (child, child))
    return ts, Ts, child


_DT = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
       5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def pc2_xyzt(msg):
    """-> (xyz float32 (N,3), t_rel seconds (N,) or None). Handles row padding."""
    n = msg.width * msg.height
    raw = np.frombuffer(msg.data, np.uint8)
    if msg.row_step != msg.width * msg.point_step and msg.height > 1:
        raw = raw.reshape(msg.height, msg.row_step)[:, :msg.width * msg.point_step]
        raw = raw.reshape(-1)
    buf = raw[:n * msg.point_step].reshape(n, msg.point_step)
    off = {f.name: (f.offset, _DT[f.datatype]) for f in msg.fields}

    def col(name):
        o, dt = off[name]
        return buf[:, o:o + np.dtype(dt).itemsize].copy().view(dt).ravel()

    xyz = np.column_stack([col("x"), col("y"), col("z")]).astype(np.float32)
    ok = np.isfinite(xyz).all(1) & (np.abs(xyz) < 1e4).all(1)
    tr = None
    for name in ("t", "time", "timestamp", "time_offset"):
        if name in off:
            tv = col(name).astype(np.float64)[ok]
            if tv.max() > 1e6:          # nanoseconds
                tv = tv * 1e-9
            tr = tv - tv.min()
            break
    return xyz[ok], tr


def voxel_centroid(P, v):
    """Average of the points in each voxel (NOT the voxel centre - centres cost
    2.4 cm of quantisation, measured)."""
    q = np.floor(P / v).astype(np.int64)
    q -= q.min(0)
    key = (q[:, 0] << 40) | (q[:, 1] << 20) | q[:, 2]
    o = np.argsort(key, kind="stable"); key = key[o]; Ps = P[o]
    br = np.r_[0, np.flatnonzero(np.diff(key)) + 1, len(key)]
    # float64 accumulator: a float32 cumsum over 1e5+ points loses millimetres
    # to rounding by the last blocks (measured: 6 mm on a 2.6 m wall)
    cs = np.vstack([np.zeros(3), np.cumsum(Ps.astype(np.float64), 0)])
    return ((cs[br[1:]] - cs[br[:-1]]) / np.diff(br)[:, None]).astype(np.float32)


def deskew(P, trel, dT_scan, bins=32):
    """Constant-velocity deskew to the scan midpoint. dT_scan = lidar motion
    over the scan (from odometry); each point is moved by the fractional
    motion Exp((f - 0.5) * Log(dT_scan)) for its time fraction f."""
    if trel is None or trel.max() <= 0:
        return P
    w = log_R(dT_scan[:3, :3]); v = dT_scan[:3, 3]
    if np.linalg.norm(w) < 1e-5 and np.linalg.norm(v) < 1e-4:
        return P
    f = trel / trel.max()
    out = P.copy()
    for b in range(bins):
        m = (f >= b / bins) & (f < (b + 1) / bins) if b < bins - 1 else (f >= b / bins)
        if not m.any():
            continue
        a = (b + 0.5) / bins - 0.5
        out[m] = P[m] @ exp_r(a * w).T + a * v
    return out


def depth_to_cloud(z_m, K, rmin=0.4, rmax=3.5,
                   edge_jump=0.05, voxel=0.05, max_pts=8000):
    """Depth image in METRES -> gated, edge-rejected, voxel-centroid cloud.
    Flying pixels at occlusion boundaries are the #1 ICP bias source for
    stereo depth: they always lie BETWEEN two surfaces."""
    z = np.array(z_m, np.float32, copy=True)
    z[(z < rmin) | (z > rmax)] = 0
    zz = np.where(z > 0, z, np.nan)
    hi = ndimage.maximum_filter(np.nan_to_num(zz, nan=-1e3), size=3)
    lo = ndimage.minimum_filter(np.nan_to_num(zz, nan=1e3), size=3)
    z[(hi - lo) > edge_jump] = 0
    v, u = np.nonzero(z)
    if len(u) < 500:
        return np.zeros((0, 3), np.float32)
    zc = z[v, u]
    P = np.column_stack([(u - K[0, 2]) * zc / K[0, 0],
                         (v - K[1, 2]) * zc / K[1, 1], zc])
    P = voxel_centroid(P.astype(np.float32), voxel)
    if len(P) > max_pts:
        P = P[np.linspace(0, len(P) - 1, max_pts).astype(int)]
    return P


def subsample(P, n):
    P = np.asarray(P)
    if len(P) > n:
        P = P[np.linspace(0, len(P) - 1, n).astype(int)]
    return P


def img16(m):
    a = np.frombuffer(m.data, np.uint8)
    return a.view(np.uint16).reshape(m.height, m.step // 2)[:, :m.width]


def img_depth(m):
    """Depth image -> metres. D455 publishes 16UC1 millimetres; ZED publishes
    32FC1 metres (with NaN/inf for invalid)."""
    a = np.frombuffer(m.data, np.uint8)
    if m.encoding in ("16UC1", "mono16"):
        return a.view(np.uint16).reshape(m.height, m.step // 2)[:, :m.width] \
            .astype(np.float32) * 0.001
    if m.encoding == "32FC1":
        z = a.view(np.float32).reshape(m.height, m.step // 4)[:, :m.width].copy()
        z[~np.isfinite(z)] = 0
        return z
    raise SystemExit("unsupported depth encoding %r" % m.encoding)


def read_map_xyz(path):
    import open3d as o3d
    P = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    if len(P) == 0:
        raise SystemExit("no points in %s" % path)
    return P


# --------------------------------------------------------------------------- #
# frozen-map reference: KD-tree + local planes (soft planarity weight, matched
# by voxel membership - both validated the hard way on the mapping sessions)
# --------------------------------------------------------------------------- #
class Reference:
    def __init__(self, P, voxel=0.05, plane_voxel=0.4, planarity=1.0, min_pts=12):
        self.P = voxel_centroid(np.asarray(P, float), voxel).astype(float)
        self.pv = plane_voxel
        self.origin = self.P.min(0)
        q = self._vox(self.P)
        key = (q[:, 0] << 40) | (q[:, 1] << 20) | q[:, 2]
        o = np.argsort(key, kind="stable"); ks = key[o]; Ps = self.P[o]
        br = np.r_[0, np.flatnonzero(np.diff(ks)) + 1, len(ks)]
        C, N, W, kk = [], [], [], []
        for a, b in zip(br[:-1], br[1:]):
            if b - a < min_pts:
                continue
            X = Ps[a:b]; c = X.mean(0)
            ev, V = np.linalg.eigh((X - c).T @ (X - c) / (b - a))
            C.append(c); N.append(V[:, 0]); kk.append(ks[a])
            W.append(1.0 - min((ev[0] / max(ev[1], 1e-12)) / planarity, 1.0))
        self.C = np.array(C); self.N = np.array(N); self.W = np.array(W)
        self.idx = {int(k): i for i, k in enumerate(kk)}
        print("  reference map: %d pts, %d plane cells" % (len(self.P), len(self.C)))

    def _vox(self, P):
        return np.clip(np.floor((P - self.origin) / self.pv).astype(np.int64),
                       0, (1 << 20) - 1)

    def plane_of(self, Q):
        q = self._vox(Q)
        key = (q[:, 0] << 40) | (q[:, 1] << 20) | q[:, 2]
        j = np.array([self.idx.get(int(k), -1) for k in key])
        m = j >= 0
        return self.C[j[m]], self.N[j[m]], self.W[j[m]], m


def icp_frame(P_body, T_init, ref, gates=(0.4, 0.2, 0.1), iters=5,
              huber=0.05, beta=0.02):
    """Point-to-plane ICP of one scan against the frozen map.
    Convention: t <- t + dt (world), R <- R exp(dphi) (body); Jacobian
    [n, cross(p, n @ R)] verified against numerical differentiation."""
    T = T_init.copy(); nu = 0; rms = np.nan; eig = np.zeros(6)
    L = 1.0 / max(float(np.median(np.linalg.norm(P_body, axis=1))), 1e-3)
    S = np.diag([1, 1, 1, L, L, L])
    for gate in gates:
        for _ in range(iters):
            Q = apply(T, P_body)
            c, n, w, m = ref.plane_of(Q)
            if m.sum() < 100:
                break
            p = P_body[m]; R = T[:3, :3]
            r = np.einsum("ij,ij->i", Q[m] - c, n)
            keep = np.abs(r) < gate
            if keep.sum() < 100:
                break
            c, n, w, p, r = c[keep], n[keep], w[keep], p[keep], r[keep]
            ww = w * np.minimum(1.0, huber / np.maximum(np.abs(r), 1e-9))
            J = np.hstack([n, np.cross(p, n @ R)])
            Jw = J * ww[:, None]
            H = J.T @ Jw; g = Jw.T @ r
            d = -np.linalg.solve(H + beta * np.trace(H) / 6.0 * np.eye(6), g)
            T = Rt(R @ exp_r(d[3:]), T[:3, 3] + d[:3])
            nu = int(keep.sum())
            rms = float(np.sqrt(np.mean(ww * r * r) / max(ww.mean(), 1e-9)))
            ev = np.linalg.eigvalsh(S @ H @ S)
            eig = ev / max(ev.max(), 1e-12)
            if np.linalg.norm(d[:3]) < 1e-4 and np.linalg.norm(d[3:]) < 1e-5:
                break
    return T, nu, rms, int((eig > 0.02).sum())


# --------------------------------------------------------------------------- #
# pose graph: odometry relative factors + absolute board-pose factors + a
# session-anchor prior. Jacobians validated numerically (incl. jr_inv).
# --------------------------------------------------------------------------- #
BOARD_HUBER = 5.0         # whitened-sigma knee for board factors
BOARD_OUTLIER = 25.0      # whitened sigma still gross AFTER convergence
ICP_SIGMA = 0.02          # m; the map's own surface noise
ICP_HUBER = 0.05
GAUGE_W = 1e-2


def solve_graph(node_t, T_init, Z_rel, sig_rel, abs_meas, clouds=None, ref=None,
                use_icp=False, use_board=True, icp_pts=400, iters=25,
                verbose=True, edge_scale=None, _second_pass=False):
    """One graph, selectable factor sets (this is what makes the A/B/C arms an
    ablation instead of three pipelines):
      odometry relative factors        always
      board/anchor absolute factors    use_board
      point-to-plane map factors       use_icp - RE-LINEARISED each iteration.
        Never chained-ICP poses as priors: their correlated drift out-weighed
        the boards in validation and the joint arm lost to boards-only.
    clouds: {node_index: (M,3) cloud in the STATE frame}.
    edge_scale: per-edge multiplier on the odometry sigma (1 = trust as
    configured; 1e3 = a free joint, used where the odometry is known to have
    jumped)."""
    n = len(node_t); Ts = T_init.copy()
    dt = np.maximum(np.diff(node_t), 1e-3)
    st, sr = sig_rel
    es = np.ones(n - 1) if edge_scale is None else np.asarray(edge_scale, float)
    sub = {}
    if use_icp:
        for k, v in (clouds or {}).items():
            P, sg = v if isinstance(v, tuple) else (v, ICP_SIGMA)
            sub[k] = (np.asarray(subsample(P, icp_pts), float), float(sg))
    lam, best_cost, Ts_best = 1e-8, np.inf, Ts.copy()
    n_flat, step_prev = 0, np.inf
    n_rejected_same, last_rejected = 0, np.nan
    for it in range(iters):
        I_, J_, V_, r_ = [], [], [], []

        def add(rows, cols, vals, res):
            base = len(r_); r_.extend(res)
            I_.extend((np.asarray(rows) + base).tolist())
            J_.extend(cols); V_.extend(vals)

        add(np.arange(6), list(range(6)), [GAUGE_W] * 6, list(np.zeros(6)))
        for k in range(n - 1):
            Ti, Tj, Zm = Ts[k], Ts[k + 1], Z_rel[k]
            Ri, Rj = Ti[:3, :3], Tj[:3, :3]
            d = Tj[:3, 3] - Ti[:3, 3]
            rt = Ri.T @ d - Zm[:3, 3]
            rr = log_R(Zm[:3, :3].T @ Ri.T @ Rj)
            Ji = jr_inv(rr)
            wt = 1.0 / (st * es[k] * dt[k] / 0.1)
            wr = 1.0 / (sr * es[k] * dt[k] / 0.1)
            B = np.vstack([
                np.hstack([-Ri.T, hat(Ri.T @ d), Ri.T, np.zeros((3, 3))]) * wt,
                np.hstack([np.zeros((3, 3)), -Ji @ Rj.T @ Ri,
                           np.zeros((3, 3)), Ji]) * wr])
            cols = list(range(6 * k, 6 * k + 6)) + \
                list(range(6 * (k + 1), 6 * (k + 1) + 6))
            rows, cc = np.meshgrid(np.arange(6), np.arange(12), indexing="ij")
            add(rows.ravel(), [cols[c] for c in cc.ravel()], B.ravel(),
                list(np.r_[rt * wt, rr * wr]))
        if use_board:
            for k, Tm, at, ar in abs_meas:
                Rm = Tm[:3, :3]
                res = np.r_[Rm.T @ (Ts[k][:3, 3] - Tm[:3, 3]),
                            log_R(Rm.T @ Ts[k][:3, :3])]
                Jb = np.block([[Rm.T, np.zeros((3, 3))],
                               [np.zeros((3, 3)), jr_inv(res[3:])]])
                W = np.diag([1 / at] * 3 + [1 / ar] * 3)
                # Huber on the whitened residual: sightings are no longer
                # pre-filtered against a drifting prediction, so one bad
                # detection must not dominate. It CANNOT be a hard gate here -
                # at iteration 0 every legitimate factor on a drifted track is
                # hundreds of sigma out, and that is exactly the signal.
                nz = float(np.linalg.norm(W @ res))
                hw = 1.0 if nz <= BOARD_HUBER else math.sqrt(BOARD_HUBER / nz)
                B = hw * (W @ Jb)
                rows, cc = np.meshgrid(np.arange(6), np.arange(6), indexing="ij")
                add(rows.ravel(), [6 * k + c for c in cc.ravel()], B.ravel(),
                    list(hw * (W @ res)))
        if use_icp:
            for k, (P, sg) in sub.items():
                Q = apply(Ts[k], P)
                c, nn, w, m = ref.plane_of(Q)
                if m.sum() < 30:
                    continue
                p = P[m]; R = Ts[k][:3, :3]
                r = np.einsum("ij,ij->i", Q[m] - c, nn)
                keep = np.abs(r) < 0.1
                if keep.sum() < 30:
                    continue
                c, nn, w, p, r = c[keep], nn[keep], w[keep], p[keep], r[keep]
                ww = w * np.minimum(1.0, ICP_HUBER / np.maximum(np.abs(r), 1e-9)) \
                    / sg
                Jp = np.hstack([nn, np.cross(p, nn @ R)]) * ww[:, None]
                rows, cc = np.meshgrid(np.arange(len(r)), np.arange(6),
                                       indexing="ij")
                add(rows.ravel(), [6 * k + cix for cix in cc.ravel()],
                    Jp.ravel(), list(r * ww))
        A = sparse.csr_matrix((V_, (I_, J_)), shape=(len(r_), 6 * n))
        rv = np.array(r_)
        cost = float(rv @ rv)
        # Levenberg-Marquardt. Point-to-plane correspondences are re-picked
        # every iteration, so a Gauss-Newton step can land somewhere WORSE and
        # the solver oscillates instead of converging. Reject such a step,
        # damp harder, retry.
        # tolerance: with correspondences re-picked each iteration the cost is
        # not a strictly comparable objective, so small rises are noise. Only a
        # clearly worse step is a real divergence.
        if it and cost > best_cost * 1.05:
            Ts = Ts_best.copy(); lam = max(lam * 10.0, 1e-4)
            if verbose:
                print("    it%2d cost %.1f REJECTED (worse than %.1f), "
                      "lambda -> %.1e" % (it, cost, best_cost, lam))
            n_rejected_same = n_rejected_same + 1 \
                if abs(cost - last_rejected) < 1e-6 * max(cost, 1) else 1
            last_rejected = cost
            if lam > 1e8 or n_rejected_same >= 3:
                if verbose:
                    print("    (stopping: the step keeps landing on the same "
                          "rejected cost - the linearisation is stuck)")
                break
            continue
        if it and abs(cost - best_cost) < 1e-4 * best_cost and step_prev < 1e-3:
            n_flat += 1
            if n_flat >= 3:
                if verbose:
                    print("    it%2d cost %.1f converged (cost flat)" % (it, cost))
                break
        else:
            n_flat = 0
        best_cost, Ts_best = cost, Ts.copy()
        lam = max(lam * 0.3, 1e-10)
        Hn = (A.T @ A).tocsc()
        d = Hn.diagonal()
        Hn = Hn + sparse.diags(np.maximum(d, 1e-9) * lam) \
            + sparse.identity(6 * n, format="csc") * 1e-9
        dx = spsolve(Hn, -(A.T @ rv))
        step = 0.0
        for k in range(n):
            dk = dx[6 * k:6 * k + 6]
            Ts[k] = Rt(Ts[k][:3, :3] @ exp_r(dk[3:]), Ts[k][:3, 3] + dk[:3])
            step = max(step, float(np.linalg.norm(dk[:3])))
        if verbose:
            print("    it%2d cost %.1f max step %.2f mm" % (it, cost, step * 1000))
        step_prev = step
        if step < 1e-5:
            break
    if np.isfinite(best_cost):
        Ts = Ts_best
    # AFTER convergence a factor still grossly out is a mis-detection, not
    # drift: the rest of the graph has already been pulled to the truth.
    if use_board and abs_meas and not _second_pass:
        keep, drop = [], []
        for f in abs_meas:
            k, Tm, at, ar = f
            Rm = Tm[:3, :3]
            res = np.r_[Rm.T @ (Ts[k][:3, 3] - Tm[:3, 3]),
                        log_R(Rm.T @ Ts[k][:3, :3])]
            nz = float(np.linalg.norm(
                np.diag([1 / at] * 3 + [1 / ar] * 3) @ res))
            (drop if nz > BOARD_OUTLIER else keep).append((f, nz))
        if drop and len(keep) >= 2:
            print("    rejected %d/%d board factor(s) still >%.0f sigma after "
                  "convergence (worst %.0f sigma) - re-solving"
                  % (len(drop), len(abs_meas), BOARD_OUTLIER,
                     max(n for _, n in drop)))
            return solve_graph(node_t, T_init, Z_rel, sig_rel,
                               [f for f, _ in keep], clouds, ref, use_icp,
                               use_board, icp_pts, iters, verbose,
                               edge_scale=es, _second_pass=True)
    return Ts


def report_factor_coverage(node_t, ks):
    """Where in time do the absolute factors sit? A track whose board factors
    all fall inside the opening dwell is anchored at t=0 and pure odometry
    afterwards - the correction-vs-odometry number then looks small only
    because the boards never reach the part that drifts."""
    if not ks:
        print("  ! NO absolute factors: this track is anchored odometry only")
        return
    t0, t1 = node_t[0], node_t[-1]
    tf = np.sort(node_t[np.array(sorted(set(ks)))]) - t0
    span = max(t1 - t0, 1e-9)
    gaps = np.diff(np.r_[0.0, tf, span])
    print("  board factors at t = %.1f .. %.1f s of %.1f s (%.0f%% of the run "
          "covered); largest un-anchored stretch %.1f s"
          % (tf[0], tf[-1], span, 100 * (tf[-1] - tf[0]) / span, gaps.max()))
    if tf[-1] < 0.25 * span:
        print("  !! every board factor is in the first quarter of the run: "
              "this track is anchored at the start and UNCONSTRAINED after it. "
              "Its 'correction vs odometry' understates the real error - the "
              "boards cannot see the stretch that drifts.")


def eval_board_resid(Ts, res_nodes, bmap):
    """Predicted vs surveyed board position at every sighting. Independent
    accuracy check for an arm that never used boards."""
    e = [np.linalg.norm((Ts[k] @ T_cb)[:3, 3] - bmap[b][0][:3, 3])
         for k, b, T_cb in res_nodes]
    return np.array(e) if e else np.array([np.nan])


def eval_map_rms(Ts, clouds, ref, cap=300):
    """Point-to-plane rms at the given poses (evaluated, not optimised).
    Independent check for an arm that never used the map."""
    out = []
    for k, v in clouds.items():
        P = v[0] if isinstance(v, tuple) else v
        Q = apply(Ts[k], np.asarray(subsample(P, cap), float))
        c, nn, w, m = ref.plane_of(Q)
        if m.sum() < 30:
            continue
        r = np.einsum("ij,ij->i", Q[m] - c, nn)
        r = r[np.abs(r) < 0.3]
        if len(r) > 20:
            out.append(float(np.sqrt(np.mean(r ** 2))))
    return np.array(out) if out else np.array([np.nan])


# --------------------------------------------------------------------------- #
def hand_eye(A_list, B_list):
    """Solve X in B X = X A (Park-Martin): A = camera motions, B = odometry-child
    motions, X = T_child_cam.

    Translation is only observable perpendicular to the rotation axes actually
    exercised: a yaw-only trajectory (every indoor corridor run) leaves t along
    the vertical NULL, and unguarded least-squares runs away with it - measured
    7.2 m of phantom t_z on the coop bag. Null directions (singular value
    < 10% of max) are projected out and t is set to 0 along them; the true
    offset there is a few cm on any real rig, so 0 is the honest choice.
    Returns (X, null_axes) - null_axes rows are the unobservable directions in
    the child frame (empty when fully observable)."""
    a = np.array([log_R(A[:3, :3]) for A in A_list])
    b = np.array([log_R(B[:3, :3]) for B in B_list])
    # Rotation-axis diversity gate: if every exercised rotation shares one axis
    # (strict yaw-only motion), R_X itself is free about that axis and the
    # solve returns garbage. Real indoor runs carry a little pitch/roll wobble
    # which weakly pins it - quantify instead of hoping.
    sv_a = np.linalg.svd(a, compute_uv=False)
    axis_div = float(sv_a[1] / max(sv_a[0], 1e-12))
    U, _, Vt = np.linalg.svd(b.T @ a)
    Rx = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    M, r = [], []
    for A, B in zip(A_list, B_list):
        M.append(B[:3, :3] - np.eye(3)); r.append(Rx @ A[:3, 3] - B[:3, 3])
    M = np.vstack(M); r = np.concatenate(r)
    Um, sv, Vm = np.linalg.svd(M, full_matrices=False)
    keep = sv > 0.1 * sv[0]
    tx = (Vm[keep].T * (1.0 / sv[keep])) @ (Um[:, keep].T @ r)
    return Rt(Rx, tx), Vm[~keep], axis_div


def estimate_cam_extrinsic(ot, oT, cam_ts, cam_Ts, dt=0.5, min_rot=0.05):
    """T_child_cam from odometry vs an independent camera-frame trajectory
    (here: the lidar-ICP track composed with T_lidar_camera).
    Returns (X, median residual, null_axes) or (None, None, None)."""
    t0, t1 = max(ot[0], cam_ts[0]), min(ot[-1], cam_ts[-1])
    tq = np.arange(t0, t1, dt)
    if len(tq) < 20:
        return None, None, None
    To = interp_traj(ot, oT, tq)
    Tc = interp_traj(cam_ts, cam_Ts, tq)
    A, B = [], []
    for i in range(len(tq) - 1):
        Ai = inv(Tc[i]) @ Tc[i + 1]
        if np.linalg.norm(log_R(Ai[:3, :3])) < min_rot:
            continue
        A.append(Ai); B.append(inv(To[i]) @ To[i + 1])
    if len(A) < 30:
        return None, None, None
    Xh, null_axes, axis_div = hand_eye(A, B)
    if axis_div < 0.03:
        return None, None, None       # rotation itself underdetermined: refuse
    res = [np.linalg.norm((inv(Xh) @ Bm @ Xh)[:3, 3] - Am[:3, 3])
           for Am, Bm in zip(A, B)]
    return Xh, float(np.median(res)), null_axes


def detect_boards_along(track, s, P, bmap, af, bag):
    """Board sightings over the whole image stream -> [(t, board_name,
    T_cam_board)], instance-resolved later against the anchored trajectory."""
    board_cfgs = P.cfg.get("boards", {})
    axes = af.get("board_axes", "opencv"); borig = af.get("board_origin", "corner")
    wanted = track.get("boards") or sorted(bmap)
    designs = sorted({bmap[b][1].get("design", b) for b in wanted if b in bmap})
    dets = {}
    for dgn in designs:
        if dgn not in board_cfgs:
            print("  ! design '%s' not in boards registry - skipped" % dgn)
            continue
        b = Board(dgn, board_cfgs[dgn])
        dets[dgn] = (b, b.frame_fix(axes, borig))
    imgs, infos, _, _ = read_bag(bag, image_topics=[track["image_topic"]],
                                 info_topics=[track.get("camera_info_topic")],
                                 want_tf=False,
                                 stride=int(track.get("img_stride", 2)),
                                 max_images=int(track.get("max_images", 0)))
    frames = imgs.get(track["image_topic"], [])
    K, D, from_bag = pick_intrinsics(infos, track.get("camera_info_topic"),
                                     track.get("rectified", False),
                                     track.get("K"), track.get("dist"))
    print("  detect: %d frames, intrinsics %s fx=%.1f, designs %s"
          % (len(frames), "bag" if from_bag else "config", K[0, 0], designs))
    out = []
    for st, gray in frames:
        for dgn, (b, fix) in dets.items():
            d = b.detect(gray, K, D)
            if d is None:
                continue
            out.append((st, dgn, d.T @ fix))
    print("  %d raw sightings" % len(out))
    return out


def resolve_instances(sights, Ts_est, node_t, bmap, wanted, radius=2.0,
                      pred_label="anchored odometry"):
    """Name each sighting's board.

    The position test exists ONLY to tell instances of a SHARED design apart.
    A design with a single surveyed instance has nothing to disambiguate, so it
    is accepted regardless of how far the prediction lands. Gating it on the
    prediction is self-defeating: the worse a track drifts, the fewer
    corrections survive, so the drift is never removed. Measured on the coop
    bag: 308 of 372 sightings discarded that way, leaving only the opening
    dwell and a meaningless 4 cm 'correction'.

    The prediction error is not noise either - for a single-instance board it
    IS a measurement of how far the predicting trajectory (pred_label) sits
    from the survey, so it is reported.
    -> [(node_idx, board_name, T_map_b_pred, T_cb)]"""
    out, dropped, n_noded, n_amb, pred_err = [], 0, 0, 0, []
    for t, dgn, T_cb in sights:
        k = int(np.argmin(np.abs(node_t - t)))
        if abs(node_t[k] - t) > 0.05:
            n_noded += 1; dropped += 1; continue
        T_map_b = Ts_est[k] @ T_cb
        cands = [n for n in wanted
                 if n in bmap and bmap[n][1].get("design", n) == dgn]
        if not cands:
            dropped += 1; continue
        d = {n: float(np.linalg.norm(bmap[n][0][:3, 3] - T_map_b[:3, 3]))
             for n in cands}
        order = sorted(cands, key=lambda n: d[n])
        best = order[0]
        if len(cands) > 1:
            # ambiguous design: nearest must be close AND clearly nearest
            if d[best] > radius or (d[order[1]] - d[best]) < radius:
                n_amb += 1; dropped += 1; continue
        pred_err.append(d[best])
        out.append((k, best, T_map_b, T_cb))
    print("  %d sightings resolved, %d dropped (%d no node within 50 ms, "
          "%d ambiguous between instances of one design; predicted from the %s)"
          % (len(out), dropped, n_noded, n_amb, pred_label))
    if pred_err:
        pe = np.array(pred_err)
        print("  prediction error at sightings: median %.2f m, max %.2f m "
              "(how far the %s sat from the survey)"
              % (np.median(pe), pe.max(), pred_label))
        # PER BOARD. A single board sitting metres out while the others are
        # centimetres is not drift - drift moves every board together. It means
        # that board's sightings are being attributed to the wrong physical
        # target, or its surveyed pose is wrong.
        # WITH TIME. Per-board error alone cannot separate the two causes:
        # a board seen only late will show a large error simply because the
        # odometry has drifted by then. Only a board seen over the SAME time
        # span as a good one, yet metres out, indicts identity or survey.
        pert = {}
        for (k, b, _, _), e in zip(out, pred_err):
            pert.setdefault(b, []).append((e, node_t[k] - node_t[0]))
        for b in sorted(pert):
            v = np.array([x[0] for x in pert[b]])
            tv = np.array([x[1] for x in pert[b]])
            print("      %-12s n=%4d  err med %7.2f m (min %6.2f max %6.2f)  "
                  "seen t=%.0f..%.0f s"
                  % (b, len(v), np.median(v), v.min(), v.max(), tv.min(), tv.max()))
        if len(pert) > 1:
            meds = {b: float(np.median([x[0] for x in v])) for b, v in pert.items()}
            spans = {b: (min(x[1] for x in v), max(x[1] for x in v))
                     for b, v in pert.items()}
            lo_b = min(meds, key=meds.get); hi_b = max(meds, key=meds.get)
            if meds[hi_b] > 1.0 and meds[hi_b] > 5 * max(meds[lo_b], 0.01):
                # do the two boards' viewing windows overlap?
                a0, a1 = spans[lo_b]; b0, b1 = spans[hi_b]
                overlap = min(a1, b1) - max(a0, b0)
                if overlap > 1.0:
                    print("      !! '%s' is %.1f m out while '%s' is %.2f m, and "
                          "they were seen at OVERLAPPING times (%.0f s of "
                          "overlap). Drift would move both together, so this is "
                          "a board-identity or survey problem."
                          % (hi_b, meds[hi_b], lo_b, meds[lo_b], overlap))
                else:
                    print("      ('%s' %.1f m out vs '%s' %.2f m, but their "
                          "viewing windows do NOT overlap - consistent with "
                          "odometry drift between them; not evidence of a "
                          "board problem on its own)"
                          % (hi_b, meds[hi_b], lo_b, meds[lo_b]))
    return out


def board_factors(res, bmap):
    """Resolved sightings -> absolute factors on the camera pose.
    Measured camera pose from the SURVEYED board: T_map_cam = T_map_board @
    inv(T_cam_board). Sigma from the survey's own spread, floored at 1 cm."""
    abs_meas, res_nodes = [], []
    for k, bname, T_map_b_pred, T_cb in res:
        Tb, rec = bmap[bname]
        T_meas = Tb @ inv(T_cb)
        sig_t = math.hypot(float(rec.get("std_mm", 10)) * 1e-3, 0.010)
        lc = rec.get("loop_closure") or {}
        sig_t = max(sig_t, float(lc.get("mm", 0)) * 1e-3)
        sig_r = math.radians(max(float(lc.get("deg", 0.3)), 1.0))
        abs_meas.append((k, T_meas, sig_t, sig_r))
        res_nodes.append((k, bname, T_cb))
    return abs_meas, res_nodes


def decimate_idx(ts, rate):
    """Indices of the first sample at or after every 1/rate mark."""
    marks = np.arange(ts[0], ts[-1] + 1e-9, 1.0 / rate)
    return np.unique(np.clip(np.searchsorted(ts, marks), 0, len(ts) - 1))



# --------------------------------------------------------------------------- #
# mobile_1 building blocks (pure functions of arrays, so they are testable
# without a bag): the chained lidar ICP and the three-arm graph
# --------------------------------------------------------------------------- #
def se3_scale(D, s):
    """Fraction s of the rigid motion D (constant-velocity extrapolation)."""
    return Rt(exp_r(s * log_R(D[:3, :3])), s * D[:3, 3])


def chain_icp(scans, ot, oT, T_map_origin, T_cl, REF, track, log_every=100,
              default_seed="lidar"):
    """Chained scan-to-map ICP for ANY range sensor (Ouster scans or D455/ZED
    depth clouds). `scans` yields (t, xyz (N,3) in the SENSOR frame,
    per-point time offsets or None). State = T_map_sensor; T_cl = the
    odometry child -> sensor transform.

    Seeding (track["seed"]):
      "lidar" (default) the lidar's OWN previous two ICP poses, extrapolated
                        at constant velocity. Nothing from the ZED after
                        scan 0, so the track cannot inherit an odometry jump.
      "odom"            previous ICP pose advanced by the odometry increment,
                        conjugated from the child frame into the lidar frame.
    Scan 0 is always seeded from the session anchor through the odometry pose
    at that stamp: T_map_odom @ T_odom_child(t) @ T_child_lidar.

    Recovery: an ICP result that moved more than max_shift/max_rot from its
    seed is NOT accepted, but the seed is not blindly kept either (that is how
    a chain gets poisoned): the scan is retried from the other seed, then
    from the constant-velocity seed with wide gates. Only if all fail is the
    constant-velocity seed kept and the scan marked unregistered (nobs 0).

    -> ts, Ts, RMS, NOBS, clouds (subsampled, LIDAR frame), n_rejected, Q
       Q = per-scan quality rows (t, rms, nobs, n_matched, shift_from_seed,
           odom_step_disagreement_m, odom_step_disagreement_rad, status)"""
    rate = float(track.get("rate_hz", 5.0))
    rmin = float(track.get("range_min", 0.7))
    rmax = float(track.get("range_max", 15.0))
    vox = float(track.get("scan_voxel", 0.10))
    keep_pts = int(track.get("keep_cloud_pts", 3000))
    min_pts = int(track.get("min_pts", 2000))
    max_shift = float(track.get("max_shift", 0.5))
    max_rot = math.radians(float(track.get("max_rot_deg", 5.0)))
    use_deskew = bool(track.get("deskew", True))
    seed_mode = track.get("seed", default_seed)
    beta = float(track.get("prior_beta", 0.02))     # damping toward the seed
    min_obs = int(track.get("min_obs", 3))           # observable DOF to accept
    cv_agree = bool(track.get("cv_must_agree_with_odom", seed_mode == "odom"))
    gates = tuple(track.get("gates", (0.4, 0.2, 0.1)))
    wide = tuple(track.get("wide_gates", (1.0, 0.5, 0.25, 0.1)))
    keep_dt = (1.0 / rate) * 0.9 if rate > 0 else 0.0   # 0 = every scan
    ts, Ts, RMS, NOBS, cl, Q = [], [], [], [], [], []
    n_rej = 0
    t_last = -1e18; T_prev = None; T_prev2 = None; t_prev = t_prev2 = None
    T_ol_prev = None; t0w = time.time()
    for t, xyz, trel in scans:
        if t - t_last < keep_dt:
            continue
        rng = np.linalg.norm(xyz, axis=1)
        sel = (rng > rmin) & (rng < rmax)
        Pb, tsel = xyz[sel], (None if trel is None else trel[sel])
        if len(Pb) < min_pts:
            continue
        T_ol = interp_traj(ot, oT, np.array([t]))[0]
        # the two candidate seeds
        T_seed_odom = T_seed_cv = None
        if T_prev is None:
            T_seed_odom = T_map_origin @ T_ol @ T_cl
        else:
            T_seed_odom = T_prev @ (inv(T_cl) @ inv(T_ol_prev) @ T_ol @ T_cl)
            if T_prev2 is not None and t_prev > t_prev2:
                D = inv(T_prev2) @ T_prev                # motion over last step
                # bounded extrapolation: an uneven stamp gap must not scale a
                # small wobble into a metre, and never more than 10 deg
                ratio = min((t - t_prev) / (t_prev - t_prev2), 1.5)
                wD = np.linalg.norm(log_R(D[:3, :3])) * ratio
                if wD > math.radians(10.0):
                    ratio *= math.radians(10.0) / wD
                T_seed_cv = T_prev @ se3_scale(D, ratio)
            else:
                T_seed_cv = T_prev.copy()
        if T_prev is None or seed_mode == "odom":
            order = [("odom", T_seed_odom), ("cv", T_seed_cv)]
        else:
            order = [("cv", T_seed_cv), ("odom", T_seed_odom)]
        order = [(n_, S) for n_, S in order if S is not None]
        # deskew with the motion of the primary seed over the scan
        if use_deskew and tsel is not None:
            span = float(tsel.max())
            if order[0][0] == "odom" or T_prev2 is None:
                T0, T1 = interp_traj(ot, oT, np.array([t, t + span]))
                dT_l = inv(T_cl) @ inv(T0) @ T1 @ T_cl
            else:
                dT_l = se3_scale(inv(T_prev2) @ T_prev, span / (t_prev - t_prev2))
            Pb = deskew(Pb.astype(float), tsel, dT_l)
        Pb = voxel_centroid(np.asarray(Pb, float), vox).astype(float)
        # register: primary seed, then the other seed, then wide gates
        status, T_i, nu, rms, nobs = "fail", None, 0, np.nan, 0
        attempts = [(n_, S, gates) for n_, S in order] + \
                   [(n_ + "+wide", S, wide) for n_, S in order]
        any_match = False
        for n_, S, gs in attempts:
            T_try, nu, rms, nobs = icp_frame(Pb, S, REF, gates=gs, beta=beta)
            any_match = any_match or nobs > 0
            d = float(np.linalg.norm(T_try[:3, 3] - S[:3, 3]))
            a = float(np.linalg.norm(log_R(S[:3, :3].T @ T_try[:3, :3])))
            lim = 2.0 if gs is wide else 1.0
            if d > max_shift * lim or a > max_rot * lim or nobs < min_obs:
                continue
            if cv_agree and not n_.startswith("odom") and T_prev is not None:
                # the odometry is the trusted seed: a result reached from the
                # constant-velocity guess must not contradict it. Without this
                # one wild cv guess (measured: 120 deg) gets accepted and
                # poisons every frame after it.
                do = float(np.linalg.norm(T_try[:3, 3] - T_seed_odom[:3, 3]))
                ao = float(np.linalg.norm(log_R(T_seed_odom[:3, :3].T @ T_try[:3, :3])))
                if do > 2 * max_shift or ao > 2 * max_rot:
                    continue
            status, T_i = n_, T_try
            break
        if T_i is None:
            status = "fail:nomatch" if not any_match else "fail:far"
        seed_used = order[0][1]
        if T_i is None:
            # unregistered: carry the constant-velocity (or anchor) seed
            T_i, rms, nobs = seed_used, np.nan, 0
            n_rej += 1
        shift = float(np.linalg.norm(T_i[:3, 3] - seed_used[:3, 3]))
        # odometry step vs lidar step: the ZED's per-step disagreement with
        # the map-registered motion. A jump in the ZED shows up here as ONE
        # large row; a scale/drift problem as a run of small ones.
        if T_prev is not None and T_seed_odom is not None:
            Dd = inv(T_seed_odom) @ T_i
            od_t, od_r = float(np.linalg.norm(Dd[:3, 3])), float(np.linalg.norm(log_R(Dd[:3, :3])))
        else:
            od_t = od_r = 0.0
        Q.append((t, rms, nobs, nu, shift, od_t, od_r, status))
        T_prev2, t_prev2 = T_prev, t_prev
        T_prev, t_prev, T_ol_prev = T_i, t, T_ol
        t_last = t
        ts.append(t); Ts.append(T_i); RMS.append(rms); NOBS.append(nobs)
        cl.append(subsample(Pb, keep_pts).astype(np.float32))
        if log_every and len(ts) % log_every == 0:
            print("  %5d scans  rms %5.2f cm  obs %d/6  seed %-8s %5.1fs"
                  % (len(ts), (rms if np.isfinite(rms) else 0) * 100,
                     nobs, status, time.time() - t0w), flush=True)
    return np.array(ts), np.array(Ts), RMS, NOBS, cl, n_rej, Q


chain_lidar = chain_icp        # name kept for the tests


def verify_odom_frames(ts, Ts, Ts_cam, ot, oT, T_cl, X, Q, ochild):
    """Two data checks that the odometry and the ICP track are in ONE frame
    and that the odometry's error is a break, not a frame mismatch.

    1. Replace ONLY the flagged odometry steps (odom step vs ICP step > 5 cm
       or 2 deg) by the ICP's own increments and re-integrate the odometry.
       A frame error is proportional to motion and lives in EVERY step, so
       the re-integrated chain would still diverge; a tracking break lives in
       the flagged steps only, so the chain would then follow the ICP.
    2. Hand-eye: solve T_child_cam from the odometry vs the ICP track over
       the clean window before the first flagged step and compare with the
       configured cam_extrinsic_xyzquat."""
    od = np.array([q[5] for q in Q]); odr = np.array([q[6] for q in Q])
    flagged = (od > 0.05) | (odr > math.radians(2.0))
    T_ol = interp_traj(ot, oT, ts)
    T_rep = [Ts[0]]
    for k in range(1, len(ts)):
        if flagged[k]:
            Z = inv(Ts[k - 1]) @ Ts[k]                       # ICP increment
        else:
            Z = inv(T_cl) @ inv(T_ol[k - 1]) @ T_ol[k] @ T_cl  # odom increment
        T_rep.append(T_rep[-1] @ Z)
    T_rep = np.array(T_rep)
    dt_, dr_ = traj_gap(T_rep, Ts)
    print("  == frame check 1: odometry re-integrated with its %d flagged "
          "step(s) replaced by the ICP increments ==" % int(flagged.sum()))
    print("     gap to the ICP track: median %.1f cm, max %.1f cm, rotation "
          "max %.1f deg over %.1f m of path"
          % (np.median(dt_) * 100, dt_.max() * 100, math.degrees(dr_.max()),
             path_length(Ts)))
    print("     -> %s" % (
        "the un-flagged odometry follows the ICP track: frames agree, the "
        "whole error is in the flagged steps (a tracking break)"
        if dt_.max() < 0.30 else
        "the odometry STILL diverges with the flagged steps removed: either a "
        "frame/extrinsic error (check cam_extrinsic_xyzquat) or a slow scale "
        "drift spread over all steps"))
    tf = ts[np.flatnonzero(flagged)] if flagged.any() else np.array([ts[-1]])
    t_end = tf[0] - 0.5
    sel = ts < t_end
    print("  == frame check 2: hand-eye T_%s_cam from odometry vs ICP over the "
          "clean window t < %.1f s (%d stamps) ==" % (ochild, t_end - ts[0], sel.sum()))
    if sel.sum() < 40:
        print("     (window too short - skipped)"); return
    Xh, res, null_axes = estimate_cam_extrinsic(ot, oT, ts[sel], Ts_cam[sel],
                                                dt=0.5, min_rot=0.05)
    if Xh is None:
        print("     (refused: the rotations in this window share one axis, the "
              "extrinsic rotation is not observable from this motion)"); return
    dR = math.degrees(np.linalg.norm(log_R(X[:3, :3].T @ Xh[:3, :3])))
    dtv = Xh[:3, 3] - X[:3, 3]
    print("     hand-eye: t=%s rpy=%s deg (residual %.1f mm/step)"
          % (np.round(Xh[:3, 3], 3).tolist(),
             np.round(Rot.from_matrix(Xh[:3, :3]).as_euler("xyz", degrees=True), 1).tolist(),
             res * 1000))
    print("     configured: t=%s rpy=%s deg"
          % (np.round(X[:3, 3], 3).tolist(),
             np.round(Rot.from_matrix(X[:3, :3]).as_euler("xyz", degrees=True), 1).tolist()))
    print("     rotation difference %.1f deg, translation difference %.1f cm%s"
          % (dR, np.linalg.norm(dtv) * 100,
             " (translation along %d unobservable axis/axes set to 0 by the "
             "solver - compare rotation only)" % len(null_axes)
             if null_axes is not None and len(null_axes) else ""))
    print("     -> %s" % ("configured extrinsic CONFIRMED by the data"
                          if dR < 5.0 else
                          "the data prefers a different extrinsic rotation: "
                          "cam_extrinsic_xyzquat is suspect"))


def build_submaps(frames, ot, oT, T_cd, window_s, voxel, max_pts, stride=1):
    """Local SLAM-style accumulation: every depth frame within +-window_s/2
    of a centre frame is moved into the CENTRE frame with the odometry's
    relative motion (conjugated into the depth frame by T_cd) and stacked.
    A 3 s submap taken while the robot turns has seen several directions,
    so registering it to the map constrains axes a single 87-deg frame
    cannot. frames: [(t, P_depth)] -> yields (t_centre, P_submap, None),
    the same shape chain_icp consumes."""
    ts_ = np.array([f[0] for f in frames])
    T_d = compose_all(interp_traj(ot, oT, ts_), T_cd)      # T_odom_depth
    half = window_s / 2.0
    n_fr, n_pt = [], []
    for i in range(0, len(frames), max(1, int(stride))):
        js = np.flatnonzero(np.abs(ts_ - ts_[i]) <= half)
        Ti_inv = inv(T_d[i])
        P = np.vstack([apply(Ti_inv @ T_d[j], frames[j][1]) for j in js])
        P = voxel_centroid(np.asarray(P, np.float32), voxel)
        n_fr.append(len(js)); n_pt.append(len(P))
        yield ts_[i], subsample(P, max_pts).astype(np.float32), None
    if n_fr:
        print("  submaps: %d, %.1f frames and %.0f points each on average "
              "(window %.1f s)" % (len(n_fr), np.mean(n_fr), np.mean(n_pt),
                                   window_s))


def report_chain_quality(ts, Q, outd, name, seed_mode="lidar"):
    """Per-scan CSV + a three-panel PNG (plane rms, observability, ZED step
    disagreement) and a summary of where the chain was weak."""
    t0 = ts[0]
    csv = os.path.join(outd, "quality_%s.csv" % name)
    with open(csv, "w") as f:
        f.write("t,t_rel,rms_cm,nobs,n_matched,shift_from_seed_cm,"
                "odom_step_disagree_cm,odom_step_disagree_deg,seed_status\n")
        for (t, rms, nobs, nu, sh, od_t, od_r, st) in Q:
            f.write("%.6f,%.3f,%.2f,%d,%d,%.2f,%.2f,%.3f,%s\n"
                    % (t, t - t0, (rms if np.isfinite(rms) else -1) * 100, nobs,
                       nu, sh * 100, od_t * 100, math.degrees(od_r), st))
    print("  wrote %s" % csv)
    st = [q[7] for q in Q[1:]]                 # scan 0 is always odom-seeded
    prim = "cv" if seed_mode == "lidar" else "odom"
    othr = "odom" if prim == "cv" else "cv"
    n_fail = sum(1 for s_ in st if s_.startswith("fail"))
    print("  seed statistics: primary (%s) %d, other seed (%s) %d, wide-gate "
          "recovery %d, unregistered %d of %d scans (%d found no map "
          "correspondences at all, %d moved too far from every seed)"
          % (prim, st.count(prim), othr, st.count(othr),
             sum(1 for s_ in st if s_.endswith("+wide")), n_fail, len(st),
             st.count("fail:nomatch"), st.count("fail:far")))
    if n_fail:
        tf = [q[0] - t0 for q in Q if q[7].startswith("fail")]
        print("  !! unregistered scans at t = %s s: poses there are "
              "constant-velocity extrapolation, not measurements"
              % np.round(tf[:12], 1).tolist())
    od = np.array([q[5] for q in Q]); odr = np.array([q[6] for q in Q])
    big = np.flatnonzero((od > 0.05) | (odr > math.radians(2.0)))
    print("  odometry step vs ICP step: median %.1f mm, p95 %.1f mm, "
          "largest %.1f cm / %.2f deg at t=%.1f s; %d step(s) > 5 cm or 2 deg"
          % (np.median(od) * 1000, np.percentile(od, 95) * 1000,
             od.max() * 100, math.degrees(odr.max()), ts[int(np.argmax(od))] - t0,
             len(big)))
    if len(big):
        print("     at t = %s s" % np.round(ts[big[:15]] - t0, 1).tolist())
        print("     (one big step = an odometry jump; a run of them = the "
              "odometry losing scale/tracking over that stretch - or, for a "
              "narrow-FOV depth chain, the chain sliding: check the rms and "
              "observable-DOF panels at those stamps)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rms = np.array([q[1] for q in Q]) * 100
        nobs = np.array([q[2] for q in Q])
        fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        ax[0].plot(ts - t0, rms, lw=0.8); ax[0].set_ylabel("plane rms [cm]")
        ax[0].set_title("%s: scan-to-map registration quality" % name)
        ax[1].plot(ts - t0, nobs, lw=0.8, drawstyle="steps-post")
        ax[1].set_ylabel("observable DOF /6"); ax[1].set_ylim(-0.2, 6.5)
        ax[2].plot(ts - t0, od * 100, lw=0.8, label="translation [cm]")
        ax[2].plot(ts - t0, np.degrees(odr), lw=0.8, label="rotation [deg]")
        ax[2].set_ylabel("odom step - ICP step"); ax[2].legend(fontsize=8)
        ax[2].set_xlabel("t [s]")
        for q in Q:
            if q[7].startswith("fail"):
                for a_ in ax:
                    a_.axvline(q[0] - t0, color="r", lw=0.6, alpha=0.5)
        for a_ in ax:
            a_.grid(alpha=.3)
        png = os.path.join(outd, "quality_%s.png" % name)
        plt.tight_layout(); plt.savefig(png, dpi=120); plt.close()
        print("  wrote %s" % png)
    except Exception as e:
        print("  (quality plot failed: %s: %s)" % (type(e).__name__, e))



def report_drift_corrections(node_t, Ts, To_n, X, To_anch, sight_nodes, gap_s=2.0):
    """What the pose graph did between board sightings. Sightings are grouped
    (gap > gap_s starts a new group). For every odometry-only stretch between
    two groups, the pose the ODOMETRY would have carried from the last
    corrected pose of the previous group is compared with the corrected pose
    at re-acquisition: that difference is the drift the odometry accumulated
    over the stretch, and the graph distributed exactly that correction back
    over the stretch's nodes. Also printed per stretch: how far the corrected
    trajectory moved from the anchored odometry (the applied correction)."""
    ks = sorted(set(sight_nodes))
    if not ks:
        return
    groups, cur = [], [ks[0]]
    for k in ks[1:]:
        if node_t[k] - node_t[cur[-1]] > gap_s:
            groups.append(cur); cur = [k]
        else:
            cur.append(k)
    groups.append(cur)
    t0 = node_t[0]
    print("  drift corrected at each board re-acquisition (%d sighting "
          "group(s)):" % len(groups))
    prev_end = 0                      # node 0 carries the session-anchor prior
    for g in groups:
        s = g[0]
        if s <= prev_end:
            prev_end = max(prev_end, g[-1]); continue
        Z = inv(X) @ inv(To_n[prev_end]) @ To_n[s] @ X
        T_pred = Ts[prev_end] @ Z
        d = inv(T_pred) @ Ts[s]
        dt_ = float(np.linalg.norm(d[:3, 3])); dr_ = float(np.linalg.norm(log_R(d[:3, :3])))
        seg = slice(prev_end, s + 1)
        corr = np.linalg.norm(Ts[seg, :3, 3] - To_anch[seg, :3, 3], axis=1)
        plen = path_length(To_anch[seg]) if s - prev_end > 1 else 0.0
        print("     t=%6.1f..%6.1f s (%5.1f s, %5.1f m of path, %4d nodes): "
              "odometry drift at re-acquisition %6.1f cm / %5.1f deg -> "
              "distributed over the stretch; applied correction vs anchored "
              "odometry median %5.1f cm, max %5.1f cm"
              % (node_t[prev_end] - t0, node_t[s] - t0, node_t[s] - node_t[prev_end],
                 plen, s - prev_end, dt_ * 100, math.degrees(dr_),
                 np.median(corr) * 100, corr.max() * 100))
        prev_end = g[-1]
    if prev_end < len(node_t) - 1:
        seg = slice(prev_end, len(node_t))
        corr = np.linalg.norm(Ts[seg, :3, 3] - To_anch[seg, :3, 3], axis=1)
        print("     t=%6.1f..%6.1f s after the last sighting: odometry only, "
              "no re-acquisition to measure the drift (correction vs anchored "
              "odometry median %.1f cm)"
              % (node_t[prev_end] - t0, node_t[-1] - t0, np.median(corr) * 100))


def run_arms(name, reg_t, reg_T, cl_l, sights, ot, oT, X, T_map_origin, bmap,
             wanted, track, REF, anchor_sig_t, src="depth", outd=None,
             verbose=True, cloud_sets=None):
    """The three-arm graph for one camera. State = the camera OPTICAL frame.
      reg_t/reg_T  chained-ICP camera poses (initialisation only)
      cl_l         one cloud per reg_t, already in the CAMERA frame
      cloud_sets   optional [(stamps, clouds, sigma_m, label)] - SEVERAL
                   range sensors feeding map factors into one graph (lidar
                   clouds at 2 cm plus ZED depth clouds at 5 cm, say); when
                   given, cl_l is ignored
      sights       [(t, design, T_cam_board)] from detect_boards_along
      X            T_child_cam; T_map_origin = T_map_odom
      track["arms_run"]  subset of A_icp / B_boards / B_breaks / C_joint
    -> dict(node_t, arms{...}, odom_only, chained, res_nodes, clouds,
            abs_meas)"""
    if cloud_sets is None:
        cloud_sets = [(reg_t, cl_l, ICP_SIGMA, src)]
    # nodes FIRST (registration stamps of every cloud set + exact sighting
    # stamps), then resolve against a trajectory at those nodes - board
    # factors land on their own stamps, never a neighbour 50 ms away
    st_extra = np.array(sorted({round(t, 6) for t, _, _ in sights}))
    node_t = np.unique(np.round(np.concatenate(
        [np.asarray(ts_, float) for ts_, _, _, _ in cloud_sets] + [st_extra]), 6))
    idx_of = {round(t, 6): i for i, t in enumerate(node_t)}
    clouds, by_set = {}, {}
    for ts_, cls_, sg_, lbl_ in cloud_sets:
        n_dup = 0
        by_set[lbl_] = {}
        for t, c in zip(np.round(ts_, 6), cls_):
            k = idx_of[round(t, 6)]
            by_set[lbl_][k] = (c, float(sg_))
            if k in clouds:          # two sensors on one stamp: concatenate
                P0, s0 = clouds[k]
                clouds[k] = (np.vstack([P0, c]), min(s0, sg_)); n_dup += 1
            else:
                clouds[k] = (c, float(sg_))
        print("  map factors: %d %s clouds at sigma %.0f mm%s"
              % (len(cls_), lbl_, sg_ * 1000,
                 " (%d shared a stamp with another set)" % n_dup if n_dup else ""))
    set_labels = [lbl_ for _, _, _, lbl_ in cloud_sets]
    To_n = interp_traj(ot, oT, node_t)
    # odometry increments, conjugated from the child frame into the camera
    Z_rel = np.array([inv(X) @ inv(To_n[i]) @ To_n[i + 1] @ X
                      for i in range(len(node_t) - 1)])
    T_init = interp_traj(reg_t, reg_T, node_t)
    To_anch = np.array([T_map_origin @ To_n[i] @ X for i in range(len(node_t))])
    # Odometry jumps. Each odometry increment is compared with the increment
    # of the chained (map-registered) trajectory over the same edge. An edge
    # that disagrees by more than odom_jump_m / odom_jump_deg is a ZED
    # tracking break, not drift: its factor is kept but with the sigma
    # multiplied by 1e3 (a free joint), so the graph does not spread a 2 m
    # jump over the neighbouring seconds. Only done for lidar clouds - a
    # depth chain is not reliable enough to indict the odometry.
    edge_scale = np.ones(len(node_t) - 1)
    if "lidar" in src and bool(track.get("odom_jump_check", True)):
        jm = float(track.get("odom_jump_m", 0.05))
        jr = math.radians(float(track.get("odom_jump_deg", 2.0)))
        dd = [inv(inv(T_init[i]) @ T_init[i + 1]) @ Z_rel[i]
              for i in range(len(node_t) - 1)]
        bad = [i for i, D in enumerate(dd)
               if np.linalg.norm(D[:3, 3]) > jm
               or np.linalg.norm(log_R(D[:3, :3])) > jr]
        for i in bad:
            edge_scale[i] = 1e3
        if bad:
            print("  %d odometry edge(s) freed (ZED step vs lidar step > %.0f cm "
                  "or %.0f deg) at t = %s s"
                  % (len(bad), jm * 100, math.degrees(jr),
                     np.round(node_t[bad][:12] - node_t[0], 1).tolist()))
            print("     (arm B has no absolute information across a freed "
                  "edge except the boards: its segments after a jump are "
                  "placed by the boards seen there, or not at all)")
    # instance resolution: predict where the sighted board is in map. With
    # lidar clouds the chained lidar track is far better than the drifting
    # odometry for this (and its prediction error is then the lidar-vs-survey
    # agreement, an independent number)
    if "lidar" in src:
        pred_T, pred_label = T_init, "chained lidar track"
    else:
        pred_T, pred_label = To_anch, "anchored odometry"
    res = resolve_instances(sights, pred_T, node_t, bmap, wanted,
                            float(track.get("instance_radius", 2.0)), pred_label)
    abs_meas, res_nodes = board_factors(res, bmap)
    anchor_prior = (0, To_anch[0], max(anchor_sig_t, 0.005), math.radians(1.0))
    print("  %d nodes, %d board factors, %d clouds"
          % (len(node_t), len(abs_meas), len(clouds)))
    report_factor_coverage(node_t, [k for k, _, _, _ in abs_meas])
    sig_rel = (float(track.get("odom_sigma_t", 0.003)),
               float(track.get("odom_sigma_r", 0.001)))
    # Initialisation of each arm:
    #   A  chained ICP poses (the geometry-only answer, smoothed by the
    #      odometry). If geometry alone cannot relocalise, that IS arm A's
    #      honest answer.
    #   B  the anchored odometry ("boards correct the odom") - so B never
    #      touches the map or the lidar, and its map rms is an independent
    #      check. boards_init="icp" starts it from the chained poses instead
    #      (only if the odometry is so far out that the graph cannot pull it
    #      back).
    #   C  depth clouds: from B - depth ICP can only REFINE; given a
    #      trajectory metres out it locks onto wrong-but-similar geometry and
    #      reports a small residual while staying wrong (measured: 1.4 cm rms
    #      while ~11 m out).
    #      lidar clouds: from A - a 360-deg lidar chained to the map is the
    #      reference itself; starting C from B would carry B's un-anchored
    #      stretches (metres, beyond the 10 cm plane gate) into a graph that
    #      then cannot recover them.
    b_init = track.get("boards_init", "odom")
    j_init = track.get("joint_init", "A_icp" if "lidar" in src else "B_boards")
    # A from the chained poses only when the chain is a lidar: a narrow-FOV
    # depth chain that slid is a worse start than the odometry it was seeded
    # from (measured: 27 m off), and map factors cannot relocalise it.
    icp_init = track.get("icp_init", "chained" if "lidar" in src else "odom")
    arms_run = track.get("arms_run") or ["A_icp", "B_boards", "B_breaks", "C_joint"]
    ARMS = {}
    n_breaks = int((edge_scale > 1).sum())
    # B_boards is the pure "odometry + boards" pose graph: NOTHING from the
    # lidar enters it, not even the break stamps - the ZED plus the boards is
    # all it has, and its result is what that pair alone can deliver.
    # B_breaks is the same graph with the lidar-detected break edges freed:
    # the difference between the two is exactly the value of knowing WHERE
    # the odometry broke. A_icp and C_joint use the freed edges (their map
    # factors already sit on the lidar).
    # (name, map factors, boards, edge scale, cloud sets used or None=all,
    #  initialisation). Per-source arms exist so "depth + boards" or
    # "lidar + boards" can be asked for on a rig that has both sensors;
    # a depth-only arm never receives the lidar-derived break stamps.
    ones = np.ones_like(edge_scale)
    arms = [("A_icp", True, False, edge_scale, None, "icp_init"),
            ("B_boards", False, True, ones, None, "odom")]
    if n_breaks:
        arms.append(("B_breaks", False, True, edge_scale, None, "odom"))
    for lbl_ in set_labels:
        if len(set_labels) > 1 or lbl_ != src:
            es_ = edge_scale if lbl_ == "lidar" else ones
            arms.append(("A_%s" % lbl_, True, False, es_, [lbl_],
                         "chained" if lbl_ == "lidar" else "odom"))
            arms.append(("C_%s" % lbl_, True, True, es_, [lbl_],
                         "A_%s" % lbl_ if lbl_ == "lidar" else "B_boards"))
    arms.append(("C_joint", True, True, edge_scale, None, "joint_init"))
    arms = [a_ for a_ in arms if a_[0] in arms_run]
    ARM_SETS = {}
    for arm, ui, ub, es, sets, init in arms:
        print("  == arm %s ==" % arm)
        am = abs_meas + [anchor_prior] if ub else []
        cl_arm = clouds if sets is None else \
            {k: v for lbl_ in sets for k, v in by_set.get(lbl_, {}).items()}
        ARM_SETS[arm] = "+".join(set_labels if sets is None else sets) if ui else ""
        if arm == "A_icp":
            start = T_init if icp_init == "chained" else To_anch
            print("     (initialised from the %s)"
                  % ("chained ICP poses" if icp_init == "chained"
                     else "anchored odometry"))
        elif arm.startswith("A_") or arm.startswith("C_") and arm != "C_joint":
            src_init = init
            start = (T_init if src_init == "chained" else
                     To_anch if src_init == "odom" else ARMS.get(src_init, To_anch))
            print("     (%s clouds; initialised from %s)"
                  % ("+".join(sets), "the chained ICP poses" if src_init == "chained"
                     else "the anchored odometry" if src_init == "odom"
                     else "arm " + src_init if src_init in ARMS
                     else "the anchored odometry"))
        elif arm.startswith("B_"):
            start = To_anch if b_init == "odom" else T_init
            print("     (initialised from the %s%s)"
                  % ("anchored odometry" if b_init == "odom"
                     else "chained ICP poses",
                     "; %d break edge(s) freed" % n_breaks
                     if arm == "B_breaks" else "; no lidar information"))
        else:
            start = ARMS.get(j_init, T_init)
            print("     (initialised from %s)"
                  % (("arm " + j_init) if j_init in ARMS
                     else "the chained ICP poses"))
        Ts = solve_graph(node_t, start.copy(), Z_rel, sig_rel, am, cl_arm, REF,
                         use_icp=ui, use_board=ub,
                         icp_pts=int(track.get("icp_pts", 400)),
                         iters=int(track.get("gn_iters", 12)), verbose=verbose,
                         edge_scale=es)
        ARMS[arm] = Ts
        if outd:
            write_tum(os.path.join(outd, "traj_%s_%s.tum" % (name, arm)),
                      node_t, Ts)
        if arm.startswith("B_"):
            report_drift_corrections(node_t, Ts, To_n, X, To_anch,
                                     [k for k, _, _ in res_nodes],
                                     float(track.get("sighting_group_gap_s", 2.0)))
    if outd:
        write_tum(os.path.join(outd, "traj_%s_odom_only.tum" % name),
                  node_t, To_anch)
    print("  == evaluation (off-diagonal cells are independent; map factors "
          "of A and C use %s clouds, the state is the %s optical frame) =="
          % (src, name))
    print("  %-10s %22s %22s %20s %20s"
          % ("arm", "board resid (cm)", "map rms (cm)", "vs C (cm)",
             "vs odom (cm)"))
    C_ref = ARMS.get("C_joint", list(ARMS.values())[-1])
    for arm, Ts in ARMS.items():
        br = eval_board_resid(Ts, res_nodes, bmap) * 100
        mr = eval_map_rms(Ts, clouds, REF) * 100
        dv = np.linalg.norm(Ts[:, :3, 3] - C_ref[:, :3, 3], axis=1) * 100
        do = np.linalg.norm(Ts[:, :3, 3] - To_anch[:, :3, 3], axis=1) * 100
        print("  %-10s %10.1f med %6.1f p95 %9.2f med %5.2f p95 "
              "%8.1f med %6.1f max %8.1f med %6.1f max"
              % (arm, np.nanmedian(br), np.nanpercentile(br, 95),
                 np.nanmedian(mr), np.nanpercentile(mr, 95),
                 np.median(dv), dv.max(), np.median(do), do.max()))
    print("  arm A board resid and arm B map rms are the honest cells "
          "(neither arm saw that data). C should match or beat both. "
          "'vs odom' is the correction each arm applied to the anchored "
          "odometry - the measured drift it removed.")
    return dict(node_t=node_t, arms=ARMS, odom_only=To_anch, chained=T_init,
                res_nodes=res_nodes, clouds=clouds, abs_meas=abs_meas,
                edge_scale=edge_scale, arm_clouds=ARM_SETS)


# --------------------------------------------------------------------------- #
def collect_methods(results, rig):
    """Every trajectory of ONE rig, in the camera optical frame:
    ([(label, ts, Ts, colour, linestyle)], has_reference). The first entry is
    the rig's geometry-only reference: the lidar ICP track if the rig has one,
    else the chained depth ICP of its arms track. Lidar tracks are drawn as
    the camera frame (Ts_cam) so all curves of a rig are the same point."""
    cols = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple",
            "tab:brown", "tab:pink", "tab:cyan"]
    ls = ["-", (0, (6, 3)), (0, (2, 2)), (0, (1, 3)), (0, (5, 1, 1, 1))]
    rs = {k: r for k, r in results.items() if rig_of(k) == rig}
    ref, odom, rest, i = [], [], [], 0
    for nm, r in rs.items():
        if r["kind"] == "lidar_icp":
            ref.append((nm + " lidar ICP", r["ts"], r.get("Ts_cam", r["Ts"]),
                        "k", "-"))
            if r.get("odom_only_cam") is not None:
                odom.append((nm + " odom only", r["ts"], r["odom_only_cam"],
                             "0.45", (0, (1, 2))))
            continue
        if r.get("chained_label") and r.get("chained") is not None:
            if r.get("chain_ok", True):
                ref.append(("%s %s" % (nm, r["chained_label"]), r["ts"],
                            r["chained"], "k", "-"))
            elif "A_icp" in (r.get("arms") or {}):
                # the chain lost most frames: the geometry-only reference of
                # this rig is arm A (odometry + map factors)
                ref.append(("%s A_icp (reference: depth chain failed)" % nm,
                            r["ts"], r["arms"]["A_icp"], "k", "-"))
        if r.get("odom_only") is not None:
            odom.append((nm + " odom only", r["ts"], r["odom_only"],
                         "0.45", (0, (1, 2))))
        arms = r.get("arms") or {nm: r["Ts"]}
        for an, aT in sorted(arms.items()):
            if an == "A_icp" and ref and "reference: depth chain failed" in ref[-1][0]:
                continue
            lbl = ("%s %s" % (nm, an)) if r.get("arms") else nm
            # say which clouds the map factors came from: "mobile_1_zed A_icp"
            # is ZED odometry + OUSTER clouds, not ZED depth
            ac = (r.get("arm_clouds") or {}).get(an)
            if ac:
                lbl += " [%s clouds]" % ac
            rest.append((lbl, r["ts"], aT, cols[i % len(cols)], ls[i % len(ls)]))
            i += 1
    # one reference, one odom-only curve (they are the same odometry); any
    # further geometry-only chain (e.g. the ZED depth chain of a rig that
    # also has a lidar) is drawn as an ordinary method against the reference
    for nm, r in rs.items():
        for lbl, ts, Ts in (r.get("extra_chains") or []):
            ref.append(("%s %s" % (nm, lbl), ts, Ts, "k", "-"))
    extra = [(lbl, ts, Ts, cols[(i + j) % len(cols)], ls[(i + j) % len(ls)])
             for j, (lbl, ts, Ts, _, _) in enumerate(ref[1:])]
    ref = ref[:1]; odom = odom[:1]
    return ref + odom + extra + rest, bool(ref)


def rigs_of(results):
    return sorted({rig_of(k) for k in results})


def save_paths_png(results, ref, bmap, outd, T_lc=None):
    """One figure PER RIG (paths_<rig>.png) with everything done so far:
      top-left   all trajectories over the map (overlay)
      top-right  each track's distance from the rig's geometry-only reference
                 (lidar ICP, or the chained depth ICP when there is no lidar)
      below      one panel PER METHOD over the map, the reference in light
                 grey behind it, so overlapping curves cannot hide each other
    Re-saved after every track, so the image exists even if a later track
    fails."""
    for rig in rigs_of(results):
        methods, has_ref = collect_methods(results, rig)
        if methods:
            _paths_figure(results, rig, methods, has_ref, ref, bmap, outd, T_lc)


def _paths_figure(results, rig, methods, has_ref, ref, bmap, outd, T_lc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lid = next((r for k, r in results.items()
                if r["kind"] == "lidar_icp" and rig_of(k) == rig), None)
    n_small = len(methods)
    ncol = 3
    nrow_small = int(math.ceil(n_small / ncol))
    fig = plt.figure(figsize=(18, 7.5 + 5.2 * nrow_small))
    gs = fig.add_gridspec(1 + nrow_small, ncol,
                          height_ratios=[1.6] + [1.0] * nrow_small)
    ax0 = fig.add_subplot(gs[0, :2]); ax1 = fig.add_subplot(gs[0, 2])

    def draw_map(ax):
        if ref is not None and len(ref.P):
            P = ref.P[::max(1, len(ref.P) // 60000)]
            ax.scatter(P[:, 0], P[:, 1], s=0.15, c="0.86", zorder=0,
                       linewidths=0)
        for nm, b in sorted(bmap.items()):
            p = b[0][:3, 3]
            ax.plot(p[0], p[1], "*", ms=13, mfc="gold", mec="k", mew=0.6, zorder=6)
            ax.annotate(nm, (p[0], p[1]), fontsize=7, zorder=7,
                        xytext=(5, 5), textcoords="offset points")
        ax.set_aspect("equal"); ax.grid(alpha=.3)

    draw_map(ax0)
    for nm, ts, Ts, c, l in methods:
        ax0.plot(Ts[:, 0, 3], Ts[:, 1, 3], ls=l, color=c,
                 lw=2.2 if c == "k" else 1.5, label=nm, zorder=4, alpha=0.9)
        ax0.plot(Ts[0, 0, 3], Ts[0, 1, 3], "o", color=c, ms=6, mec="k", mew=0.6,
                 zorder=5)
    # where the LIDAR track puts each board it saw (duplicate-board test)
    if lid is not None and T_lc is not None:
        for r in results.values():
            for k, bn, T_cb in (r.get("res_nodes") or []):
                t = r["ts"][k]
                if lid["ts"][0] <= t <= lid["ts"][-1]:
                    Tm = interp_traj(lid["ts"], lid["Ts"], np.array([t]))[0] \
                        @ T_lc @ T_cb
                    ax0.plot(Tm[0, 3], Tm[1, 3], "x", color="tab:purple", ms=4,
                             mew=0.8, zorder=8)
        ax0.plot([], [], "x", color="tab:purple", ms=6,
                 label="board position implied by the lidar track")
    ax0.set_title("%s: all methods, map frame, camera optical point (o = start)"
                  % rig)
    ax0.set_xlabel("x [m]"); ax0.set_ylabel("y [m]")
    ax0.legend(fontsize=7, loc="best")

    gaps = {}
    ref_label = methods[0][0] if has_ref else None
    if has_ref:
        ref_ts, ref_T = methods[0][1], methods[0][2]     # first entry = reference
        for nm, ts, Ts, c, l in methods[1:]:
            tq = ref_ts[(ref_ts >= ts[0]) & (ref_ts <= ts[-1])]
            if len(tq) < 5:
                continue
            d = np.linalg.norm(interp_traj(ts, Ts, tq)[:, :3, 3]
                               - interp_traj(ref_ts, ref_T, tq)[:, :3, 3], axis=1)
            gaps[nm] = (tq, d)
            ax1.plot(tq - ref_ts[0], d, ls=l, color=c, lw=1.4, label=nm)
        for r in results.values():
            for k, bn, _ in (r.get("res_nodes") or []):
                ax1.axvline(r["ts"][k] - ref_ts[0], color="gold", lw=0.4,
                            alpha=0.35, zorder=0)
        ax1.plot([], [], color="gold", lw=2, label="board sighting")
        ax1.set_xlabel("t [s]"); ax1.set_ylabel("distance from %s [m]" % ref_label)
        ax1.set_title("agreement with the geometry-only reference over time")
        ax1.set_yscale("symlog", linthresh=0.1)
        ax1.grid(alpha=.3, which="both"); ax1.legend(fontsize=7)
    else:
        ax1.axis("off"); ax1.text(.5, .5, "no geometry-only reference yet",
                                  ha="center")

    for i, (nm, ts, Ts, c, l) in enumerate(methods):
        ax = fig.add_subplot(gs[1 + i // ncol, i % ncol])
        draw_map(ax)
        if has_ref and i > 0:
            L = methods[0][2]
            ax.plot(L[:, 0, 3], L[:, 1, 3], "-", color="0.55", lw=1.0, zorder=3,
                    label=ref_label)
        ax.plot(Ts[:, 0, 3], Ts[:, 1, 3], "-", color=c, lw=1.6, zorder=4, label=nm)
        ax.plot(Ts[0, 0, 3], Ts[0, 1, 3], "o", color=c, ms=7, mec="k", mew=0.6,
                zorder=5)
        ax.plot(Ts[-1, 0, 3], Ts[-1, 1, 3], "s", color=c, ms=7, mec="k", mew=0.6,
                zorder=5)
        ttl = nm
        if nm in gaps:
            d = gaps[nm][1]
            ttl += "\nvs reference: median %.1f cm, p95 %.1f cm, max %.1f cm" % (
                np.median(d) * 100, np.percentile(d, 95) * 100, d.max() * 100)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.legend(fontsize=7, loc="best")
    png = os.path.join(outd, "paths_%s.png" % rig)
    plt.tight_layout(); plt.savefig(png, dpi=110); plt.close()
    print("  wrote %s (%d methods: o = start, square = end)" % (png, len(methods)))


def compare_rig(results, T_lc, outd):
    """One table per rig: every trajectory of that rigid body, in the CAMERA
    optical frame, sampled at the stamps of the rig's geometry-only reference
    (lidar ICP, or the chained depth ICP without a lidar). Pairwise median
    translation gap in cm, plus p95/max against the reference, plus a CSV of
    the gaps over time. Nothing here is ground truth: reference-vs-boards
    agreement is the accuracy statement, reference-vs-odom is the odometry
    drift, arms-vs-arms says what each information source changed."""
    for rig in rigs_of(results):
        methods, has_ref = collect_methods(results, rig)
        if not has_ref or len(methods) < 2:
            continue
        tracks = [(nm, ts, Ts) for nm, ts, Ts, _, _ in methods]
        t0 = max(ts[0] for _, ts, _ in tracks)
        t1 = min(ts[-1] for _, ts, _ in tracks)
        rts = tracks[0][1]
        tq = rts[(rts >= t0) & (rts <= t1)]
        if len(tq) < 10:
            print("\n== %s: tracks do not overlap in time, no comparison" % rig)
            continue
        S = [(nm, interp_traj(ts, Ts, tq)) for nm, ts, Ts in tracks]
        print("\n== %s: all trajectories in the camera optical frame at %d "
              "stamps of '%s' over %.0f s ==" % (rig, len(tq), S[0][0], tq[-1] - tq[0]))
        print("  vs the reference '%s':" % S[0][0])
        gaps = {}
        for nm, T in S[1:]:
            dt, dr = traj_gap(S[0][1], T)
            gaps[nm] = dt
            print("    %-32s median %6.1f cm  p95 %6.1f cm  max %6.1f cm | "
                  "rot median %.2f deg" % (nm, np.median(dt) * 100,
                                           np.percentile(dt, 95) * 100,
                                           dt.max() * 100,
                                           math.degrees(np.median(dr))))
        w = max(len(nm) for nm, _ in S)
        print("  pairwise median translation gap [cm]:")
        print("  %*s " % (w, "") + " ".join("%8s" % ("[%d]" % j) for j in range(len(S))))
        for i, (ni, Ti) in enumerate(S):
            row = ["%8.1f" % (np.median(traj_gap(Ti, Tj)[0]) * 100) for _, Tj in S]
            print("  %*s " % (w, ni) + " ".join(row) + "   [%d]" % i)
        csv = os.path.join(outd, "compare_%s.csv" % rig)
        with open(csv, "w") as f:
            f.write("t," + ",".join("gap_cm:" + nm.replace(",", " ") for nm in gaps) + "\n")
            for i, t in enumerate(tq):
                f.write("%.6f," % t + ",".join("%.2f" % (gaps[nm][i] * 100)
                                               for nm in gaps) + "\n")
        print("  wrote %s (gap to the reference over time)" % csv)


# --------------------------------------------------------------------------- #
def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "pipeline_config.json"
    P = load_pipeline(cfg_path)
    s = P.cfg.get("08_reference")
    if s is None:
        raise SystemExit("add an '08_reference' block to %s (sample at the "
                         "bottom of this file)" % cfg_path)
    bag = s["bag"]
    outd = s.get("out_dir", "reference_out")
    os.makedirs(outd, exist_ok=True)

    sa = json.load(open(s["session_anchor"]))
    af = json.load(open(s["anchor_frame"]))
    bmap = {}
    for name, rec in (af.get("boards") or {}).items():
        T = np.eye(4)
        T[:3, :3] = Rot.from_quat(rec["qxyzw"]).as_matrix()
        T[:3, 3] = rec["xyz"]
        bmap[name] = (T, rec)
    cams = sa.get("cameras", {})
    print("session anchors: %s" % {k: np.round(
        np.array(v["map_to_cam"]["xyz"]), 3).tolist() for k, v in cams.items()
        if "map_to_cam" in v})

    T_lc = getattr(P.sensor, "T_lidar_camera", None)
    T_cam_lidar = None
    if T_lc is not None:
        T_lc = np.asarray(T_lc, float)
        if s.get("invert_T_lidar_camera"):
            # calibration.json stores the OTHER convention (lidar pts -> camera)
            T_lc = inv(T_lc)
            print("(T_lidar_camera inverted per config)")
        print("T_lidar_camera used as the pose of the CAMERA in the LIDAR "
              "frame: t=%s rpy=%s deg"
              % (np.round(T_lc[:3, 3], 4).tolist(),
                 np.round(Rot.from_matrix(T_lc[:3, :3])
                          .as_euler("xyz", degrees=True), 2).tolist()))
        T_cam_lidar = inv(T_lc)
    else:
        print("! calibration.json has no T_lidar_camera: lidar tracks and the "
              "cross-check are unavailable")

    def arec_of(track):
        return cams[track["anchor_cam"]]

    def anchor_T(cam_name):
        rec = cams[cam_name]["map_to_cam"]
        return Rt(Rot.from_quat(rec["qxyzw"]).as_matrix(), np.array(rec["xyz"]))

    REF = None
    results = {}

    def get_ref():
        nonlocal REF
        if REF is None:
            REF = Reference(read_map_xyz(s["ref_map"]),
                            voxel=float(s.get("target_voxel", 0.05)),
                            plane_voxel=float(s.get("plane_voxel", 0.4)))
        return REF

    for track in s["tracks"]:
        name = track["name"]; kind = track["type"]
        if not track.get("enabled", True):
            print("\n== %s (%s): disabled in config, skipped ==" % (name, kind))
            continue
        print("\n== %s (%s) ==" % (name, kind))
        ot, oT, ochild = read_odom(bag, track["odom_topic"])
        # odometry child frame -> the camera optical frame the anchor refers to
        X = make_T_xyzq(track["cam_extrinsic_xyzquat"]) \
            if track.get("cam_extrinsic_xyzquat") else None
        if X is None:
            # Try hand-eye against a finished lidar track of the SAME rig -
            # its poses composed with T_lidar_camera are an independent
            # camera-frame trajectory, so T_child_cam is observable from
            # relative motions alone.
            rig = rig_of(name)
            lid = next((r for k, r in results.items()
                        if r["kind"] == "lidar_icp" and rig_of(k) == rig), None)
            if lid is not None and T_lc is not None:
                camT = compose_all(lid["Ts"], T_lc)
                X, he_res, null_axes = estimate_cam_extrinsic(ot, oT,
                                                              lid["ts"], camT)
                if X is None:
                    print("  (hand-eye refused: this run's rotations share one "
                          "axis, so the extrinsic is underdetermined - supply "
                          "cam_extrinsic_xyzquat)")
                if X is not None:
                    print("  hand-eye T_%s_cam from the lidar track: "
                          "t=%s rpy=%s deg (residual %.1f mm/step)"
                          % (ochild, np.round(X[:3, 3], 4).tolist(),
                             np.round(Rot.from_matrix(X[:3, :3])
                                      .as_euler("xyz", degrees=True), 2).tolist(),
                             he_res * 1000))
                    for ax in (null_axes if null_axes is not None else []):
                        print("  (t along child-frame axis %s unobservable from "
                              "this run's rotations - set to 0; supply "
                              "cam_extrinsic_xyzquat for the few-cm truth)"
                              % np.round(ax, 2).tolist())
        if X is None and track.get("depth_extrinsic_xyzquat") \
                and track.get("child_is_camera_link", True):
            # RealSense: `camera_link` IS the depth (left IR) body frame, so
            # T_child_color = R_optical @ inv(T_color_depth). Only valid if the
            # odometry child really is camera_link (Isaac VSLAM base_frame).
            R_opt = Rot.from_quat([-0.5, 0.5, -0.5, 0.5]).as_matrix()
            X = Rt(R_opt, np.zeros(3)) @ inv(
                make_T_xyzq(track["depth_extrinsic_xyzquat"]))
            print("  cam_extrinsic_xyzquat derived from depth_extrinsic_xyzquat "
                  "ASSUMING odometry child '%s' is the RealSense camera_link "
                  "(depth body frame): xyzquat=%s. If the child is a base_link "
                  "with a mount offset this is WRONG - run tf2_echo and set "
                  "cam_extrinsic_xyzquat explicitly."
                  % (ochild, np.round(np.r_[X[:3, 3], Rot.from_matrix(X[:3, :3])
                                                       .as_quat()], 5).tolist()))
        if X is None:
            X = np.eye(4)
            print("  !! no cam_extrinsic_xyzquat and no lidar track to "
                  "hand-eye against: assuming odometry child frame '%s' IS "
                  "the optical frame. SLAM odometry children are usually "
                  "BODY frames (x forward) - if so this is a ~90 deg error "
                  "that bends the whole track by metres. Get the truth with:\n"
                  "     ros2 run tf2_ros tf2_echo %s <optical frame>\n"
                  "  and put it in cam_extrinsic_xyzquat." % (ochild, ochild))
        else:
            print("  X = T_%s_cam: t=%s rpy=%s deg"
                  % (ochild, np.round(X[:3, 3], 4).tolist(),
                     np.round(Rot.from_matrix(X[:3, :3])
                              .as_euler("xyz", degrees=True), 2).tolist()))
        A = anchor_T(track["anchor_cam"])          # T_map_cam at session start
        # odom pose at the anchor's own timestamp, not blindly index 0
        t_anchor = cams[track["anchor_cam"]].get("dwell_t_end") or ot[0]
        T_o0 = interp_traj(ot, oT, np.array([min(max(t_anchor, ot[0]), ot[-1])]))[0]
        T_map_origin = A @ inv(T_o0 @ X)
        print("  anchored at t=%.2f s: map->odom-origin xyz=%s"
              % (t_anchor - ot[0], np.round(T_map_origin[:3, 3], 3).tolist()))

        if kind == "lidar_icp":
            if T_lc is None:
                raise SystemExit("calibration.json has no T_lidar_camera; the "
                                 "lidar track cannot be placed")
            REF = get_ref()
            pts_frame = topic_frame(bag, track["points_topic"])
            print("  points %s are in frame '%s' - T_lidar_camera in "
                  "calibration.json must be for THIS frame (Ouster os_lidar "
                  "and os_sensor differ by a 180 deg yaw and ~36 mm)"
                  % (track["points_topic"], pts_frame))
            T_cl = X @ T_cam_lidar          # T_child_lidar
            scans = ((t,) + pc2_xyzt(m)
                     for t, m in iter_topic(bag, track["points_topic"]))
            print("  seed mode '%s' (scan 0 from the session anchor; then %s)"
                  % (track.get("seed", "lidar"),
                     "the lidar's own previous poses, constant velocity"
                     if track.get("seed", "lidar") == "lidar"
                     else "the ZED odometry increment"))
            ts, Ts, RMS, NOBS, cl, n_rej, Q = chain_lidar(
                scans, ot, oT, T_map_origin, T_cl, REF, track)
            if len(ts) == 0:
                raise SystemExit("no usable scans on %s" % track["points_topic"])
            print("  %d scans (%d unregistered) | plane rms median %.2f cm "
                  "p95 %.2f cm | rank-deficient %.1f%%"
                  % (len(ts), n_rej, np.nanmedian(RMS) * 100,
                     np.nanpercentile(RMS, 95) * 100,
                     100 * np.mean(np.array(NOBS) < 6)))
            report_chain_quality(ts, Q, outd, name, track.get("seed", "lidar"))
            write_tum(os.path.join(outd, "traj_%s.tum" % name), ts, Ts)
            # the same track as the CAMERA optical frame (through
            # T_lidar_camera) so every trajectory of this rig can be compared
            # in one body frame; the scans likewise: P_cam = T_cam_lidar P
            Ts_cam = compose_all(Ts, T_lc)
            write_tum(os.path.join(outd, "traj_%s_in_cam.tum" % name), ts, Ts_cam)
            cl_cam = [apply(T_cam_lidar, c).astype(np.float32) for c in cl]
            # ---- lidar ICP vs anchored odometry: the odometry drift, measured
            # against the map. Same stamps, same body frame (lidar):
            #   T_map_lidar(odom) = T_map_odom @ T_odom_child(t) @ T_child_lidar
            To_l = compose_all(np.tile(T_map_origin, (len(ts), 1, 1))
                               @ interp_traj(ot, oT, ts), T_cl)
            To_cam = compose_all(To_l, T_lc)
            write_tum(os.path.join(outd, "traj_%s_odom_only.tum" % name),
                      ts, To_l)
            print("  == lidar ICP vs anchored odometry (lidar frame, %d "
                  "stamps) ==" % len(ts))
            dtr, drr = traj_gap(Ts, To_l)
            report_gap("odom - lidar", ts, dtr, drr, path_length(Ts))
            try:
                verify_odom_frames(ts, Ts, Ts_cam, ot, oT, T_cl, X, Q, ochild)
            except Exception as e:
                print("  (frame check failed: %s: %s)" % (type(e).__name__, e))
            print("  (both start at the session anchor, so the gap at t=0 is "
                  "the anchor-vs-map agreement; smooth growth is odometry "
                  "drift; a step is a ZED tracking break - the per-step line "
                  "above names its stamp; the chain itself is seeded from its "
                  "own poses and only an 'unregistered' scan can move it)")
            results[name] = dict(kind=kind, ts=ts, Ts=Ts, Ts_cam=Ts_cam,
                                 odom_only=To_l, odom_only_cam=To_cam,
                                 clouds_cam=cl_cam,
                                 frame="lidar", rms=float(np.nanmedian(RMS)))

        elif kind == "rgbd_icp":
            REF = get_ref()
            # anchor refers to the color optical frame; depth lives in the
            # depth/infra1 frame - a ~1.5 cm baseline plus a small rotation on
            # the D455. Leaving it identity puts that error on every frame.
            Xd = make_T_xyzq(track["depth_extrinsic_xyzquat"]) \
                if track.get("depth_extrinsic_xyzquat") else np.eye(4)
            if not track.get("depth_extrinsic_xyzquat"):
                print("  ! no depth_extrinsic_xyzquat (color optical -> depth "
                      "frame): assuming identity, ~1.5 cm systematic on D455")
            Kd = None
            for _, ci in iter_topic(bag, track["depth_info_topic"], limit=1):
                Kd = np.array(ci.k).reshape(3, 3)
            print("  depth intrinsics fx=%.1f" % Kd[0, 0])
            rate = float(track.get("rate_hz", 10.0))
            keep_dt = 1.0 / rate
            beta = float(track.get("prior_beta", 0.10))   # frustum: lean on seed
            ts, Ts, RMS, NOBS = [], [], [], []
            n_rej = 0
            t_last = -1e18; T_prev = None; T_ol_prev = None; t0w = time.time()
            for t, m in iter_topic(bag, track["depth_topic"]):
                if t - t_last < keep_dt:
                    continue
                Pb = depth_to_cloud(img_depth(m), Kd,
                                    rmin=float(track.get("range_min", 0.4)),
                                    rmax=float(track.get("range_max", 3.5)))
                if len(Pb) < 500:
                    continue
                T_ol = interp_traj(ot, oT, np.array([t]))[0]
                if T_prev is None:
                    T_seed = T_map_origin @ T_ol @ X @ Xd
                else:
                    Xdc = X @ Xd            # odom child -> depth frame
                    T_seed = T_prev @ (inv(Xdc) @ inv(T_ol_prev) @ T_ol @ Xdc)
                T_i, nu, rms, nobs = icp_frame(np.asarray(Pb, float), T_seed,
                                               REF, beta=beta)
                d = float(np.linalg.norm(T_i[:3, 3] - T_seed[:3, 3]))
                a = float(np.linalg.norm(log_R(T_seed[:3, :3].T @ T_i[:3, :3])))
                if d > float(track.get("max_shift", 0.3)) \
                        or a > math.radians(float(track.get("max_rot_deg", 5.0))):
                    T_i, rms, nobs = T_seed, np.nan, 0
                    n_rej += 1
                T_prev, T_ol_prev = T_i, T_ol
                t_last = t
                ts.append(t); Ts.append(T_i); RMS.append(rms); NOBS.append(nobs)
                if len(ts) % 200 == 0:
                    print("  %5d frames  rms %5.2f cm  obs %d/6  %5.1fs"
                          % (len(ts), (rms if np.isfinite(rms) else 0) * 100,
                             nobs, time.time() - t0w), flush=True)
            Ts = np.array(Ts); ts = np.array(ts)
            rd = 100 * np.mean(np.array(NOBS) < 6)
            print("  %d frames (%d ICP results rejected -> seed kept) | plane "
                  "rms median %.2f cm p95 %.2f cm | rank-deficient %.1f%%"
                  % (len(ts), n_rej, np.nanmedian(RMS) * 100,
                     np.nanpercentile(RMS, 95) * 100, rd))
            if rd > 50:
                print("  ! most frames are rank-deficient (corridors): those "
                      "poses lean on the odometry seed along the unobservable "
                      "axis. This track is the geometry-only arm - the boards "
                      "track is what bounds it there.")
            write_tum(os.path.join(outd, "traj_%s.tum" % name), ts, Ts)
            results[name] = dict(kind=kind, ts=ts, Ts=Ts, frame="depth",
                                 rms=float(np.nanmedian(RMS)))

        elif kind == "arms":
            # Three corrected trajectories of the CAMERA optical frame from
            # ONE estimator (run_arms):
            #   A_icp    odometry + relinearised map factors (geometry only)
            #   B_boards odometry + board factors + session-anchor prior
            #   C_joint  everything
            # Map factors come from EITHER this camera's depth ("depth") OR
            # the rig's lidar track ("lidar": mobile_1 - clouds and chained
            # poses reused from the lidar_icp track, transformed into the
            # camera frame through T_lidar_camera; no second ICP pass).
            if not track.get("cam_extrinsic_xyzquat"):
                print("  ! SKIPPING this track: cam_extrinsic_xyzquat is "
                      "REQUIRED (odometry child '%s' -> optical). Identity "
                      "bent this bag by 8-9.6 m. Get it:\n     ros2 run "
                      "tf2_ros tf2_echo %s <optical frame>" % (ochild, ochild))
                continue
            REF = get_ref()
            src = track.get("cloud_source", "depth")
            srcs = src.split("+")
            cloud_sets, extra_chains, chain_ok = [], [], True
            if "lidar" in srcs:
                lt = track.get("lidar_track") or next(
                    (k for k, r in results.items()
                     if r["kind"] == "lidar_icp" and rig_of(k) == rig_of(name)),
                    None)
                if lt not in results or results[lt]["kind"] != "lidar_icp":
                    print("  ! SKIPPING: cloud_source 'lidar' needs a finished "
                          "lidar_icp track of this rig earlier in 'tracks' "
                          "(lidar_track=%r, have %s)" % (lt, sorted(results)))
                    continue
                lid = results[lt]
                # chained lidar poses as CAMERA poses (T_map_cam = T_map_lidar
                # @ T_lidar_camera); clouds already in the camera frame
                reg_t, reg_T, cl_l = lid["ts"], lid["Ts_cam"], lid["clouds_cam"]
                cloud_sets.append((reg_t, cl_l,
                                   float(track.get("icp_sigma_lidar", 0.02)),
                                   "lidar"))
                print("  map factors from lidar track '%s': %d clouds "
                      "(camera frame), chained poses composed with "
                      "T_lidar_camera as initialisation" % (lt, len(cl_l)))
            if "depth" in srcs:
                Xd = make_T_xyzq(track["depth_extrinsic_xyzquat"]) \
                    if track.get("depth_extrinsic_xyzquat") else np.eye(4)
                if not track.get("depth_extrinsic_xyzquat"):
                    print("  (no depth_extrinsic_xyzquat: assuming depth is "
                          "registered to the anchored optical frame - true for "
                          "ZED depth_registered, ~1.5 cm off for raw D455 depth)")
                Kd = None
                for _, ci in iter_topic(bag, track["depth_info_topic"], limit=1):
                    Kd = np.array(ci.k).reshape(3, 3)
                print("  chained depth-ICP pass (depth fx=%.1f, seed '%s'): the "
                      "geometry-only estimate of this rig - a %s frustum is "
                      "far more degenerate than a lidar, so read the quality "
                      "report before trusting corridor stretches"
                      % (Kd[0, 0], track.get("seed", "odom"),
                         "narrow" if Kd[0, 0] > 300 else "wide"))
                rmin = float(track.get("range_min", 0.4))
                rmax = float(track.get("range_max", 3.5))
                T_cd = X @ Xd                       # odom child -> DEPTH frame

                # every depth frame at the requested rate, as a cloud in the
                # DEPTH frame (kept in memory: ~70 MB for a 150 s run)
                frate = float(track.get("rate_hz", 10.0))
                fdt = (1.0 / frate) * 0.9 if frate > 0 else 0.0
                frames, t_last_f = [], -1e18
                for t, m in iter_topic(bag, track["depth_topic"]):
                    if t - t_last_f < fdt:
                        continue
                    Pd = depth_to_cloud(img_depth(m), Kd, rmin=rmin, rmax=rmax)
                    if len(Pd) >= 500:
                        frames.append((t, np.asarray(Pd, np.float32)))
                        t_last_f = t
                print("  %d depth frames" % len(frames))
                win = float(track.get("submap_window_s", 3.0))
                # state = the DEPTH optical frame (the clouds' own frame)
                dtrack = dict(track)
                dtrack.setdefault("prior_beta", 0.10)
                dtrack.setdefault("min_obs", 1)
                dtrack.setdefault("max_shift", 0.3)
                dtrack.setdefault("scan_voxel", 0.05)
                dtrack.setdefault("min_pts", 500)
                dtrack["rate_hz"] = 0                                   # already decimated
                dtrack["range_min"] = 0.0; dtrack["range_max"] = 1e9   # gated already
                if win > 0:
                    print("  submap accumulation: frames within +-%.1f s stitched "
                          "into the centre frame with the odometry, then "
                          "registered as one cloud" % (win / 2))
                    scans_d = build_submaps(frames, ot, oT, T_cd, win,
                                            float(dtrack["scan_voxel"]),
                                            int(track.get("submap_max_pts", 20000)),
                                            int(track.get("submap_stride", 1)))
                else:
                    scans_d = ((t, P, None) for t, P in frames)
                d_t, d_T, RMS, NOBS, d_cl, n_rej, Q = chain_icp(
                    scans_d, ot, oT, T_map_origin, T_cd, REF, dtrack,
                    log_every=200, default_seed="odom")
                if len(d_t) == 0:
                    print("  ! SKIPPING: no usable depth frames"); continue
                rd = 100 * np.mean(np.array(NOBS) < 6)
                print("  %d frames (%d unregistered) | plane rms median %.2f cm "
                      "p95 %.2f cm | rank-deficient %.1f%%"
                      % (len(d_t), n_rej, np.nanmedian(RMS) * 100,
                         np.nanpercentile(RMS, 95) * 100, rd))
                if rd > 50:
                    print("  ! most frames are rank-deficient (corridors, flat "
                          "walls): along the unobservable axis those poses are "
                          "the odometry seed, damped, not a measurement")
                report_chain_quality(d_t, Q, outd, name + "_depth",
                                     track.get("seed", "odom"))
                chain_ok = n_rej < 0.1 * len(d_t)
                if not chain_ok and "lidar" not in srcs:
                    print("  !! the depth chain lost %d%% of its frames: it is "
                          "NOT usable as this rig's reference. Arm A (odometry "
                          "+ map factors, started from the odometry) takes that "
                          "role in the tables and the figure."
                          % (100 * n_rej // max(len(d_t), 1)))
                # depth-frame poses -> COLOR optical frame (the state of the
                # graph and of the anchor): T_map_color = T_map_depth @ inv(Xd);
                # clouds likewise: P_color = Xd P_depth
                dep_T = compose_all(d_T, inv(Xd))
                dep_cl = [apply(Xd, c).astype(np.float32) for c in d_cl]
                cloud_sets.append((d_t, dep_cl,
                                   float(track.get("icp_sigma_depth", 0.05)),
                                   "depth"))
                write_tum(os.path.join(outd, "traj_%s_depth_icp.tum" % name),
                          d_t, dep_T)
                To_c = compose_all(np.tile(T_map_origin, (len(d_t), 1, 1))
                                   @ interp_traj(ot, oT, d_t), X)
                print("  == depth ICP vs anchored odometry (color frame, %d "
                      "stamps) ==" % len(d_t))
                dtr, drr = traj_gap(dep_T, To_c)
                report_gap("odom - depth ICP", d_t, dtr, drr, path_length(dep_T))
                if "lidar" in srcs:
                    # the lidar chain stays the initialisation and reference;
                    # the depth chain is its own case in the figure and table
                    extra_chains.append(("depth %sICP chained"
                                         % ("submap " if win > 0 else ""),
                                         d_t, dep_T))
                else:
                    reg_t, reg_T, cl_l = d_t, dep_T, dep_cl
            sights = detect_boards_along(track, s, P, bmap, af, bag)
            arec = arec_of(track)
            g = run_arms(name, reg_t, reg_T, cl_l, sights, ot, oT, X,
                         T_map_origin, bmap, track.get("boards") or sorted(bmap),
                         track, REF, float(arec.get("std_mm", 10)) * 1e-3,
                         src=src, outd=outd, cloud_sets=cloud_sets)
            final = g["arms"].get("C_joint", list(g["arms"].values())[-1])
            results[name] = dict(kind=kind, ts=g["node_t"], Ts=final, frame="cam",
                                 arms=g["arms"], res_nodes=g["res_nodes"],
                                 bmap=bmap, odom_only=g["odom_only"],
                                 chained=g["chained"], cloud_source=src,
                                 chained_label=(None if "lidar" in srcs
                                                else "depth ICP chained"),
                                 chain_ok=(True if "lidar" in srcs else chain_ok),
                                 extra_chains=extra_chains,
                                 arm_clouds=g["arm_clouds"])

        elif kind == "cam_boards":
            rate = float(track.get("rate_hz", 10.0))
            keep = decimate_idx(ot, rate)
            node_t = ot[keep]
            To = oT[keep]
            T_init = np.array([T_map_origin @ To[i] @ X for i in range(len(keep))])
            Z_rel = np.array([inv(X) @ inv(To[i]) @ To[i + 1] @ X
                              for i in range(len(keep) - 1)])
            sights = detect_boards_along(track, s, P, bmap, af, bag)
            res = resolve_instances(sights, T_init, node_t, bmap,
                                    track.get("boards") or sorted(bmap),
                                    float(track.get("instance_radius", 2.0)))
            abs_meas, res_nodes = board_factors(res, bmap)
            # session-anchor prior on the first node
            arec = cams[track["anchor_cam"]]
            abs_meas.append((0, T_init[0],
                             max(arec.get("std_mm", 10) * 1e-3, 0.005),
                             math.radians(1.0)))
            print("  graph: %d nodes, %d board factors" % (len(node_t), len(res)))
            report_factor_coverage(node_t, [k for k, _, _, _ in res])
            Ts = solve_graph(node_t, T_init, Z_rel,
                             (float(track.get("odom_sigma_t", 0.003)),
                              float(track.get("odom_sigma_r", 0.001))),
                             abs_meas, use_icp=False, use_board=True)
            corr = np.linalg.norm(Ts[:, :3, 3] - T_init[:, :3, 3], axis=1)
            print("  correction vs anchored odometry: median %.1f cm, max %.1f cm "
                  "(this IS the measured drift the boards removed)"
                  % (np.median(corr) * 100, corr.max() * 100))
            write_tum(os.path.join(outd, "traj_%s.tum" % name), node_t, Ts)
            write_tum(os.path.join(outd, "traj_%s_odom_only.tum" % name),
                      node_t, T_init)
            results[name] = dict(kind=kind, ts=node_t, Ts=Ts, frame="cam",
                                 n_boards=len(res), res_nodes=res_nodes,
                                 bmap=bmap, odom_only=T_init)
        else:
            print("  ! unknown type '%s' - skipped" % kind)
        try:
            save_paths_png(results, REF, bmap, outd, T_lc)
        except Exception as e:
            print("  (path plot failed: %s: %s)" % (type(e).__name__, e))

    # -------- cross-check: two independent tracks of one rigid body -------- #
    lid_name = next((k for k, r in results.items() if r["kind"] == "lidar_icp"),
                    None)
    lid = results.get(lid_name)
    zed = next((v for k, v in results.items()
                if v["kind"] in ("cam_boards", "arms")
                and lid_name is not None and rig_of(k) == rig_of(lid_name)),
               None)
    if lid is not None and zed is not None and T_lc is not None:
        tq = lid["ts"][(lid["ts"] >= zed["ts"][0]) & (lid["ts"] <= zed["ts"][-1])]
        if len(tq) > 10:
            print("\n== cross-check %s: lidar-ICP track vs camera track "
                  "(same rigid body through T_lidar_camera) ==" % rig_of(lid_name))
            Tl = interp_traj(lid["ts"], lid["Ts"], tq)
            Tlc_all = compose_all(Tl, T_lc)[:, :3, 3]
            for an, aT in sorted((zed.get("arms") or {}).items()):
                ga = np.linalg.norm(
                    Tlc_all - interp_traj(zed["ts"], aT, tq)[:, :3, 3], axis=1)
                print("  arm %-10s vs lidar: median %7.1f cm  p95 %7.1f cm"
                      % (an, np.median(ga) * 100, np.percentile(ga, 95) * 100))
            Tz = interp_traj(zed["ts"], zed["Ts"], tq)
            gap = np.linalg.norm(Tlc_all - Tz[:, :3, 3], axis=1)
            print("  translation gap (final camera track): median %.1f cm  "
                  "p95 %.1f cm  max %.1f cm over %d stamps"
                  % (np.median(gap) * 100, np.percentile(gap, 95) * 100,
                     gap.max() * 100, len(tq)))
            # A constant gap is an extrinsic error; a growing one is drift in
            # whichever track has no absolute reference over that stretch.
            qs = np.linspace(0, len(tq) - 1, 6).astype(int)
            print("  gap over time:  " + "  ".join(
                "t=%.0fs %.0fcm" % (tq[i] - tq[0], gap[i] * 100) for i in qs))
            rel = (gap.max() - gap.min()) / max(gap.max(), 1e-9)
            if gap.max() < 0.10:
                verdict = ("AGREE to within %.0f cm everywhere: the two "
                           "estimates of this body are consistent"
                           % (gap.max() * 100))
            elif rel < 0.25:
                verdict = ("CONSTANT offset (%.0f%% variation): suspect "
                           "T_lidar_camera, not the trajectories" % (100 * rel))
            else:
                verdict = ("GROWING (%.0f cm -> %.0f cm): one track is "
                           "drifting; the one WITHOUT absolute information "
                           "over that stretch is the suspect - check the "
                           "board-factor coverage line above"
                           % (gap.min() * 100, gap.max() * 100))
            print("  -> %s" % verdict)
            # DECISIVE: score the lidar track on the camera track's board
            # sightings. The lidar never used a board and the boards never saw
            # the map, so this says which of (map+lidar) and (boards) is the
            # odd one out - and it tests the T_lidar_camera CONVENTION at the
            # same time, by trying both compositions.
            rn = zed.get("res_nodes"); zbm = zed.get("bmap")
            if rn and zbm:
                st = np.array([zed["ts"][k] for k, _, _ in rn])
                ok = (st >= lid["ts"][0]) & (st <= lid["ts"][-1])
                if ok.sum() > 5:
                    Tl_at = interp_traj(lid["ts"], lid["Ts"], st[ok])
                    for lbl, X_lc in (("T_lidar_camera", T_lc),
                                      ("inv(T_lidar_camera)", inv(T_lc))):
                        e = [np.linalg.norm(
                                (Tl_at[i] @ X_lc @ rn[j][2])[:3, 3]
                                - zbm[rn[j][1]][0][:3, 3])
                             for i, j in enumerate(np.flatnonzero(ok))]
                        print("  lidar track vs the SAME board sightings, "
                              "composed with %-19s median %8.2f m"
                              % (lbl, float(np.median(e))))
                    print("  -> if BOTH are metres, the map and the board survey "
                          "are not in one frame (stage 03), not a trajectory "
                          "problem; if one is centimetres, that is the correct "
                          "extrinsic convention and the other track is at fault "
                          "(set invert_T_lidar_camera if the inverse wins)")
                    # WHERE does the lidar track put the board it is seeing?
                    # T_map_board = T_map_cam(lidar) @ T_cb, per sighting.
                    #   tight cluster, far from the survey -> there is a real
                    #     physical board there and it is NOT the surveyed one
                    #     (duplicate print, or a wrong surveyed pose)
                    #   scattered -> the lidar track and the sightings are
                    #     mutually inconsistent, i.e. the lidar track is wrong
                    # The sighting RANGE settles it independently: a ChArUco
                    # board is only detectable within ~1-2 m, so a board seen
                    # at 1.5 m cannot be one the trajectory places 12 m away.
                    print("  where the LIDAR track places each board it saw:")
                    for bn in sorted({rn[j][1] for j in np.flatnonzero(ok)}):
                        js = [j for j in np.flatnonzero(ok) if rn[j][1] == bn]
                        Pb_ = np.array([(Tl_at[i] @ T_lc @ rn[j][2])[:3, 3]
                                        for i, j in enumerate(np.flatnonzero(ok))
                                        if rn[j][1] == bn])
                        rng_seen = np.array([np.linalg.norm(rn[j][2][:3, 3])
                                             for j in js])
                        ctr = np.median(Pb_, axis=0)
                        spread = float(np.median(np.linalg.norm(Pb_ - ctr, axis=1)))
                        surv = zbm[bn][0][:3, 3]
                        print("      %-12s n=%4d  seen at %.2f m  |  lidar puts "
                              "it at %s (scatter %.2f m)"
                              % (bn, len(js), float(np.median(rng_seen)),
                                 np.round(ctr, 2).tolist(), spread))
                        print("      %-12s survey says %s -> %.2f m away"
                              % ("", np.round(surv, 2).tolist(),
                                 float(np.linalg.norm(ctr - surv))))
                        if spread < 0.5 and np.linalg.norm(ctr - surv) > 1.0:
                            print("      %-12s !! TIGHT cluster %.1f m from the "
                                  "surveyed pose: a real board is there and it "
                                  "is NOT the surveyed one - duplicate print of "
                                  "this design, or that board's survey is wrong"
                                  % ("", float(np.linalg.norm(ctr - surv))))
                        elif spread > 1.0:
                            print("      %-12s !! SCATTERED (%.1f m): the lidar "
                                  "track and these sightings are mutually "
                                  "inconsistent - the lidar track is the suspect"
                                  % ("", spread))
            print("  (lidar track plane rms %.2f cm against the map, board-free "
                  "- a track registering that well is not the one that moved "
                  "metres)" % (lid.get("rms", float("nan")) * 100))
            print("  this needs no ground truth: two independent estimates of "
                  "one body. A constant offset = extrinsic error; growth "
                  "between board sightings = ZED odometry drift the boards "
                  "could not reach.")
    try:
        if T_lc is not None:
            compare_rig(results, T_lc, outd)
    except Exception as e:
        print("\n(comparison table failed: %s: %s)" % (type(e).__name__, e))
    print("\ndone -> %s" % outd)


SAMPLE_CONFIG = r"""
"08_reference": {
  "bag": "/path/to/mirc_dataset_coop2_20260828_merged",
  "session_anchor": "map_stages_20260828_outputs/session_anchor.json",
  "anchor_frame": "map_stages_20260828_outputs/anchor_frame.json",
  "ref_map": "map_stages_20260828_outputs/map_final_20260828_nc_anchored.pcd",
      <- MUST be the ANCHORED cloud (stage 03 output). denoised.pcd lives in
         the pre-anchor GLIM frame and would put a fixed R_align+offset error
         on every trajectory here.
  "out_dir": "map_stages_20260828_outputs/reference_coop2_mobile1",
  "target_voxel": 0.05, "plane_voxel": 0.4,
  "invert_T_lidar_camera": false,
      <- flip ONLY if the cross-check reports inv(T_lidar_camera) as the
         centimetre composition
  "tracks": [
    { "name": "mobile_1_lidar", "type": "lidar_icp",
      "points_topic": "/mobile_1/ouster/points",
      "odom_topic": "/mobile_1/zed/odom",
      "anchor_cam": "zed",
      "cam_extrinsic_xyzquat": [-0.010, 0.060, 0.015, -0.5, 0.5, -0.5, 0.5],
      "seed": "lidar",        <- "lidar": own previous poses (default);
                                 "odom": ZED increment (the ZED is only a
                                 fallback seed either way)
      "rate_hz": 10.0,        <- 0 = every scan
      "range_min": 0.7, "range_max": 10.0, "scan_voxel": 0.10,
      "deskew": true, "gates": [0.4, 0.2, 0.1],
      "wide_gates": [1.0, 0.5, 0.25, 0.1],
      "max_shift": 0.5, "max_rot_deg": 5.0,
      "keep_cloud_pts": 3000 },

    { "name": "mobile_1_zed", "type": "arms",
      "cloud_source": "lidar", "lidar_track": "mobile_1_lidar",
      "odom_topic": "/mobile_1/zed/odom",
      "anchor_cam": "zed",
      "cam_extrinsic_xyzquat": [-0.010, 0.060, 0.015, -0.5, 0.5, -0.5, 0.5],
      "image_topic": "/mobile_1/zed/left/image_rect_color",
      "camera_info_topic": "/mobile_1/zed/left/camera_info", "rectified": true,
      "boards": ["anchor", "anchor_b", "rs_anchor"],
      "img_stride": 2, "max_images": 0, "instance_radius": 2.0,
      "odom_sigma_t": 0.003, "odom_sigma_r": 0.001,
      "boards_init": "odom", "joint_init": "A_icp",
      "odom_jump_check": true, "odom_jump_m": 0.05, "odom_jump_deg": 2.0,
          <- break stamps are used by A, C and the extra B_breaks arm only;
             B_boards never sees anything from the lidar
      "sighting_group_gap_s": 2.0,
      "icp_pts": 400, "gn_iters": 25 }
  ]
}
"""

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
STAGE 08 - per-sensor reference trajectories in `map` for a coop bag.

Stage 06 measured where each camera STARTED (the opening dwell on its board).
This stage turns whole trajectories into the map frame and corrects them with
the absolute information each sensor can see:

  arms        track (any depth camera: mobile_1 ZED, mobile_2 RealSense):
              three corrected trajectories from ONE estimator sharing nodes,
              odometry factors and solver - arm A (odometry + map
              point-to-plane factors, geometry only), arm B (odometry + board
              factors + session-anchor prior), arm C (everything). Ends with
              the ablation table whose off-diagonal cells are independent
              checks. Supersedes running rgbd_icp + cam_boards separately for
              the same agent. cam_extrinsic_xyzquat REQUIRED.

              The chained ICP pass that precedes the graph registers every
              cloud to the map with the odometry-propagated seed, exactly like
              the lidar track. Only its POSES are initialisation-only: each was
              seeded from the previous, so their errors are correlated and
              feeding them in as fixed measurements lets the chain's own drift
              count as evidence (measured: it out-voted the boards). The graph
              re-registers the CLOUDS against the map at every iteration
              instead, so the depth data constrains the solution more
              thoroughly, not less.

  rgbd_icp    track (depth only, no boards): each depth frame is deprojected to
              a point cloud (range-gated, e.g. 0.4-3.5 m where D455 noise stays
              under the map's own floor, flying pixels rejected at edges)
              and registered to the frozen map exactly like the lidar track -
              geometry only, no boards. The 87-deg frustum is far more
              degenerate than a 360-deg lidar (facing a flat wall constrains
              one direction), so the damping leans harder on the odometry seed
              and the rank-deficiency rate is reported; read it before
              trusting corridor stretches.

  lidar_icp   track (mobile_1 Ouster): every scan is registered to the FROZEN
              reference map by point-to-plane ICP. The session-start pose seeds
              scan 0 (through T_lidar_camera); after that each scan is seeded
              by the previous one advanced by odometry. The map itself is the
              absolute reference - boards are not used, so this track is
              fiducial-free by construction.

  cam_boards  track (boards only, no depth): the SLAM/odometry
              chain is anchored at the session-start pose and corrected by a
              pose graph whenever a board is sighted along the run. Between
              sightings the odometry carries the pose; at a sighting the board
              pulls it back to the survey. Without the graph the odometry
              drift accumulates unbounded (measured on this bag: ~1 m over
              147 s); with it the error is pulled to the board sigma at every
              sighting.

Every track outputs a TUM trajectory in `map` plus per-sample quality, and the
stage cross-checks mobile_1's lidar track against its camera track through
T_lidar_camera - two independent estimates of one rigid body, so their gap is
an honest accuracy statement that needs no ground truth.

BOARD SIGHTINGS
  Detected with the pipeline's own Board.detect + frame_fix (same convention as
  stages 03/06). Instances of a shared design are resolved by MAP position:
  the anchored trajectory predicts where the sighted board is in map, and the
  nearest surveyed instance within instance_radius claims it - no marker-id
  guessing, works mid-run.

CONFIG ("08_reference" stage block; see the sample at the bottom of this file)
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


def read_odom(bag, topic):
    ts, Ts, child = [], [], None
    for t, m in iter_topic(bag, topic):
        p = m.pose.pose.position; o = m.pose.pose.orientation
        if child is None:
            child = getattr(m, "child_frame_id", "") or "?"
        ts.append(t)
        Ts.append(Rt(Rot.from_quat([o.x, o.y, o.z, o.w]).as_matrix(),
                     np.array([p.x, p.y, p.z])))
    if not ts:
        raise SystemExit("no odometry on %s" % topic)
    ts = np.array(ts); Ts = np.array(Ts)
    print("  odom %s: %d poses, %.1f s, path %.1f m, child_frame_id '%s'"
          % (topic, len(ts), ts[-1] - ts[0],
             float(np.sum(np.linalg.norm(np.diff(Ts[:, :3, 3], axis=0), axis=1))),
             child))
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
                verbose=True, _second_pass=False):
    """One graph, selectable factor sets (this is what makes the A/B/C arms an
    ablation instead of three pipelines):
      odometry relative factors        always
      board/anchor absolute factors    use_board
      point-to-plane map factors       use_icp - RE-LINEARISED each iteration.
        Never chained-ICP poses as priors: their correlated drift out-weighed
        the boards in validation and the joint arm lost to boards-only.
    clouds: {node_index: (M,3) cloud in the STATE frame}."""
    n = len(node_t); Ts = T_init.copy()
    dt = np.maximum(np.diff(node_t), 1e-3)
    st, sr = sig_rel
    sub = {}
    if use_icp:
        for k, P in (clouds or {}).items():
            if len(P) > icp_pts:
                P = P[np.linspace(0, len(P) - 1, icp_pts).astype(int)]
            sub[k] = np.asarray(P, float)
    lam, best_cost, Ts_best = 1e-8, np.inf, Ts.copy()
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
            wt = 1.0 / (st * dt[k] / 0.1); wr = 1.0 / (sr * dt[k] / 0.1)
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
            for k, P in sub.items():
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
                    / ICP_SIGMA
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
            Ts = Ts_best.copy(); lam *= 10.0
            if verbose:
                print("    it%2d cost %.1f REJECTED (worse than %.1f), "
                      "lambda -> %.1e" % (it, cost, best_cost, lam))
            if lam > 1e8:
                break
            continue
        best_cost, Ts_best = cost, Ts.copy()
        lam = max(lam * 0.1, 1e-10)
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
                               _second_pass=True)
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
    for k, P in clouds.items():
        if len(P) > cap:
            P = P[np.linspace(0, len(P) - 1, cap).astype(int)]
        Q = apply(Ts[k], np.asarray(P, float))
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
    (here: the lidar-ICP track composed with T_lidar_camera)."""
    t0, t1 = max(ot[0], cam_ts[0]), min(ot[-1], cam_ts[-1])
    tq = np.arange(t0, t1, dt)
    if len(tq) < 20:
        return None, None
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


def resolve_instances(sights, Ts_est, node_t, bmap, wanted, radius=2.0):
    """Name each sighting's board.

    The position test exists ONLY to tell instances of a SHARED design apart.
    A design with a single surveyed instance has nothing to disambiguate, so it
    is accepted regardless of how far the prediction lands. Gating it on the
    prediction is self-defeating: the worse a track drifts, the fewer
    corrections survive, so the drift is never removed. Measured on the coop
    bag: 308 of 372 sightings discarded that way, leaving only the opening
    dwell and a meaningless 4 cm 'correction'.

    The prediction error is not noise either - for a single-instance board it
    IS a drift measurement, so it is reported.
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
          "%d ambiguous between instances of one design)"
          % (len(out), dropped, n_noded, n_amb))
    if pred_err:
        pe = np.array(pred_err)
        print("  prediction error at sightings: median %.2f m, max %.2f m "
              "(how far the pre-graph trajectory sat from the survey)"
              % (np.median(pe), pe.max()))
        # PER BOARD. A single board sitting metres out while the others are
        # centimetres is not drift - drift moves every board together. It means
        # that board's sightings are being attributed to the wrong physical
        # target, or its surveyed pose is wrong.
        per = {}
        for (_, b, _, _), e in zip(out, pred_err):
            per.setdefault(b, []).append(e)
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


def avg_T(Ts):
    """Average of SE(3) samples: translation by median (outlier-tolerant),
    rotation by the SVD/Frobenius mean with a determinant correction."""
    Ts = np.asarray(Ts)
    t = np.median(Ts[:, :3, 3], axis=0)
    U, _, Vt = np.linalg.svd(Ts[:, :3, :3].sum(axis=0))
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return Rt(R, t)


def resurvey_boards(results, lid, T_lc, af, outd):
    """Re-survey every board the LIDAR-equipped agent saw, in THIS bag.

    A board that moved between the mapping session and this one makes its
    surveyed pose stale, and every pose derived from it - including other
    agents' session anchors - inherits the error. But the lidar track is an
    independent, map-locked reference, so a board it sees can be re-surveyed
    here: T_map_board = T_map_cam(lidar) @ T_cb, averaged over the sightings.
    Writes anchor_frame_resurveyed.json; boards not seen are copied through
    unchanged."""
    est = {}
    for r in results.values():
        for k, bn, T_cb in (r.get("res_nodes") or []):
            t = r["ts"][k]
            if not (lid["ts"][0] <= t <= lid["ts"][-1]):
                continue
            Tm = interp_traj(lid["ts"], lid["Ts"], np.array([t]))[0] @ T_lc @ T_cb
            est.setdefault(bn, []).append(Tm)
    if not est:
        return None
    out = json.loads(json.dumps(af))
    print("\n== in-session re-survey from the lidar track ==")
    moved = []
    for bn, Ts in sorted(est.items()):
        if bn not in out.get("boards", {}):
            continue
        T = avg_T(Ts)
        sc = float(np.median(np.linalg.norm(
            np.asarray(Ts)[:, :3, 3] - T[:3, 3], axis=1)))
        old_xyz = np.array(out["boards"][bn]["xyz"], float)
        d = float(np.linalg.norm(T[:3, 3] - old_xyz))
        print("  %-12s n=%4d  scatter %.3f m  |  survey %s -> in-session %s "
              "(%.2f m)" % (bn, len(Ts), sc, np.round(old_xyz, 3).tolist(),
                            np.round(T[:3, 3], 3).tolist(), d))
        if sc > 0.10:
            print("      scatter too large to trust - not updating this board")
            continue
        out["boards"][bn]["xyz"] = [round(float(v), 6) for v in T[:3, 3]]
        out["boards"][bn]["qxyzw"] = [round(float(v), 6) for v in R_to_q(T[:3, :3])]
        out["boards"][bn]["resurveyed"] = {
            "source": "lidar track, in-session", "n_views": len(Ts),
            "scatter_m": round(sc, 4), "moved_from_survey_m": round(d, 4)}
        if d > 0.10:
            moved.append((bn, d))
    out["resurvey_note"] = ("boards re-derived from the lidar track in this "
                            "bag; supersedes the mapping-session survey for "
                            "boards that moved")
    p = os.path.join(outd, "anchor_frame_resurveyed.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print("  wrote %s" % p)
    for bn, d in moved:
        print("  !! '%s' moved %.2f m since the survey. EVERY pose derived "
              "from it is stale - re-run stage 06 with the re-surveyed file "
              "so the session anchors of agents that use this board are "
              "recomputed, then re-run this stage." % (bn, d))
    return p


def save_paths_png(results, ref, bmap, outd, T_lc=None):
    """Two panels: the trajectories over the map in XY, and each camera
    track's distance from the lidar track over time. The first shows WHERE a
    track goes wrong, the second WHEN - a table of medians shows neither."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(17, 7.5),
                           gridspec_kw={"width_ratios": [1.35, 1]})

    if ref is not None and len(ref.P):
        P = ref.P[::max(1, len(ref.P) // 60000)]
        ax[0].scatter(P[:, 0], P[:, 1], s=0.15, c="0.86", zorder=0,
                      linewidths=0, label="reference map")
    for nm, b in sorted(bmap.items()):
        p = b[0][:3, 3]
        ax[0].plot(p[0], p[1], "*", ms=17, mfc="gold", mec="k", mew=0.8, zorder=6)
        ax[0].annotate(nm, (p[0], p[1]), fontsize=8, zorder=7,
                       xytext=(6, 6), textcoords="offset points")

    lid = next((r for r in results.values() if r["kind"] == "lidar_icp"), None)
    curves = []
    for nm, r in results.items():
        if r["kind"] == "lidar_icp":
            curves.append((nm, r["ts"], r["Ts"], "k", 2.4, "-"))
        else:
            for i, (an, aT) in enumerate(sorted((r.get("arms") or
                                                 {nm: r["Ts"]}).items())):
                # distinct dash patterns: arms that AGREE overlap exactly, and
                # a solid curve hidden under another teaches nothing
                curves.append(("%s %s" % (nm, an), r["ts"], aT,
                               ["tab:red", "tab:green", "tab:blue",
                                "tab:orange"][i % 4], 1.6,
                               ["-", (0, (6, 3)), (0, (2, 2)), (0, (1, 3))][i % 4]))
    for nm, ts, Ts, c, lw, *rest in curves:
        ls = rest[0] if rest else "-"
        ax[0].plot(Ts[:, 0, 3], Ts[:, 1, 3], ls=ls, color=c, lw=lw, label=nm,
                   zorder=4, alpha=0.9)
        ax[0].plot(Ts[0, 0, 3], Ts[0, 1, 3], "o", color=c, ms=7, mec="k",
                   mew=0.7, zorder=5)

    # where the LIDAR track puts each board it saw - the duplicate-board test,
    # drawn: a cross far from its gold star means the seen board is not that one
    if lid is not None and T_lc is not None:
        for r in results.values():
            for k, bn, T_cb in (r.get("res_nodes") or []):
                t = r["ts"][k]
                if not (lid["ts"][0] <= t <= lid["ts"][-1]):
                    continue
                Tm = interp_traj(lid["ts"], lid["Ts"],
                                 np.array([t]))[0] @ T_lc @ T_cb
                ax[0].plot(Tm[0, 3], Tm[1, 3], "x", color="tab:purple", ms=4,
                           mew=0.8, zorder=8)
        ax[0].plot([], [], "x", color="tab:purple", ms=6,
                   label="board position implied by the lidar track")
    ax[0].set_aspect("equal"); ax[0].grid(alpha=.3)
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
    ax[0].set_title("trajectories in the map frame (o = session start)")
    ax[0].legend(fontsize=7, loc="best")

    if lid is not None:
        for nm, ts, Ts, c, lw, *rest in curves:
            if c == "k":
                continue
            tq = ts[(ts >= lid["ts"][0]) & (ts <= lid["ts"][-1])]
            if len(tq) < 5:
                continue
            Tl = interp_traj(lid["ts"], lid["Ts"], tq) @ np.tile(
                T_lc if T_lc is not None else np.eye(4), (len(tq), 1, 1))
            d = np.linalg.norm(interp_traj(ts, Ts, tq)[:, :3, 3]
                               - Tl[:, :3, 3], axis=1)
            ax[1].plot(tq - lid["ts"][0], d, ls=(rest[0] if rest else "-"),
                       color=c, lw=1.6, label=nm)
        for r in results.values():
            for k, bn, _ in (r.get("res_nodes") or []):
                ax[1].axvline(r["ts"][k] - lid["ts"][0], color="gold",
                              lw=0.4, alpha=0.35, zorder=0)
        ax[1].plot([], [], color="gold", lw=2, label="board sighting")
        ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("distance from lidar track [m]")
        ax[1].set_title("agreement with the lidar track over time")
        ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
    else:
        ax[1].axis("off")
        ax[1].text(.5, .5, "no lidar track to compare against", ha="center")
    png = os.path.join(outd, "paths.png")
    plt.tight_layout(); plt.savefig(png, dpi=130); plt.close()
    print("\nwrote %s" % png)


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

    T_lidar_cam_ref = np.asarray(getattr(P.sensor, "T_lidar_camera", None)) \
        if getattr(P.sensor, "T_lidar_camera", None) is not None else None
    if T_lidar_cam_ref is not None and T_lidar_cam_ref.shape != (4, 4):
        T_lidar_cam_ref = make_T_xyzq(np.asarray(T_lidar_cam_ref).ravel())
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

    def arec_of(track):
        return cams[track["anchor_cam"]]

    def anchor_T(cam_name):
        rec = cams[cam_name]["map_to_cam"]
        return Rt(Rot.from_quat(rec["qxyzw"]).as_matrix(), np.array(rec["xyz"]))

    REF = None
    results = {}

    for track in s["tracks"]:
        name = track["name"]; kind = track["type"]
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
            rig = name.split("_")[0] + "_" + name.split("_")[1]
            lid = next((r for k, r in results.items()
                        if r["kind"] == "lidar_icp" and k.startswith(rig)), None)
            if lid is not None:
                camT = lid["Ts"] @ np.tile(P.sensor.T_lidar_camera,
                                           (len(lid["Ts"]), 1, 1))
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
        if X is None:
            X = np.eye(4)
            print("  !! no cam_extrinsic_xyzquat and no lidar track to "
                  "hand-eye against: assuming odometry child frame '%s' IS "
                  "the optical frame. SLAM odometry children are usually "
                  "BODY frames (x forward) - if so this is a ~90 deg error "
                  "that bends the whole track by metres. Get the truth with:\n"
                  "     ros2 run tf2_ros tf2_echo %s <optical frame>\n"
                  "  and put it in cam_extrinsic_xyzquat." % (ochild, ochild))
        A = anchor_T(track["anchor_cam"])          # T_map_cam at session start
        # odom pose at the anchor's own timestamp, not blindly index 0
        t_anchor = cams[track["anchor_cam"]].get("dwell_t_end") or ot[0]
        T_o0 = interp_traj(ot, oT, np.array([min(max(t_anchor, ot[0]), ot[-1])]))[0]
        T_map_origin = A @ inv(T_o0 @ X)
        print("  anchored: map->odom-origin xyz=%s"
              % np.round(T_map_origin[:3, 3], 3).tolist())

        if kind == "lidar_icp":
            if REF is None:
                REF = Reference(read_map_xyz(s["ref_map"]),
                                voxel=float(s.get("target_voxel", 0.05)),
                                plane_voxel=float(s.get("plane_voxel", 0.4)))
            T_cam_lidar = inv(P.sensor.T_lidar_camera)
            rate = float(track.get("rate_hz", 5.0))
            rmin = float(track.get("range_min", 0.7))
            rmax = float(track.get("range_max", 15.0))
            vox = float(track.get("scan_voxel", 0.10))
            keep_dt = 1.0 / rate
            ts, Ts, RMS, NOBS = [], [], [], []
            n_rej = 0
            t_last = -1e18; T_prev = None; t0w = time.time()
            use_deskew = bool(track.get("deskew", True))
            T_cl = X @ T_cam_lidar
            for t, m in iter_topic(bag, track["points_topic"]):
                if t - t_last < keep_dt:
                    continue
                xyz, trel = pc2_xyzt(m)
                rng = np.linalg.norm(xyz, axis=1)
                sel = (rng > rmin) & (rng < rmax)
                Pb, tsel = xyz[sel], (None if trel is None else trel[sel])
                if len(Pb) < 2000:
                    continue
                if use_deskew and tsel is not None:
                    span = float(tsel.max())
                    T0, T1 = interp_traj(ot, oT, np.array([t, t + span]))
                    dT_l = inv(T_cl) @ inv(T0) @ T1 @ T_cl
                    Pb = deskew(Pb.astype(float), tsel, dT_l)
                Pb = voxel_centroid(np.asarray(Pb, float), vox).astype(float)
                # seed: previous solution advanced by odometry (scan 0: anchor)
                T_ol = interp_traj(ot, oT, np.array([t]))[0]
                if T_prev is None:
                    T_seed = T_map_origin @ T_ol @ X @ T_cam_lidar
                else:
                    # the odometry increment lives in the odom CHILD frame;
                    # the state is the lidar frame, so it must be conjugated
                    # by T_cl. Skipping this misdirects every step by the
                    # body-vs-optical rotation and walks the track off.
                    T_seed = T_prev @ (inv(T_cl) @ inv(T_ol_prev) @ T_ol @ T_cl)
                T_i, nu, rms, nobs = icp_frame(Pb, T_seed, REF)
                # one bad ICP basin must not poison the chain: a correction
                # beyond max_shift/max_rot keeps the odometry-propagated seed
                # (same guard as 01a, which never chains for exactly this reason)
                d = float(np.linalg.norm(T_i[:3, 3] - T_seed[:3, 3]))
                a = float(np.linalg.norm(log_R(T_seed[:3, :3].T @ T_i[:3, :3])))
                if d > float(track.get("max_shift", 0.5)) \
                        or a > math.radians(float(track.get("max_rot_deg", 5.0))):
                    T_i, rms, nobs = T_seed, np.nan, 0
                    n_rej += 1
                T_prev, T_ol_prev = T_i, T_ol
                t_last = t
                ts.append(t); Ts.append(T_i); RMS.append(rms); NOBS.append(nobs)
                if len(ts) % 100 == 0:
                    print("  %5d scans  rms %5.2f cm  obs %d/6  %5.1fs"
                          % (len(ts), rms * 100, nobs, time.time() - t0w),
                          flush=True)
            Ts = np.array(Ts); ts = np.array(ts)
            print("  %d scans (%d ICP results rejected -> seed kept) | plane "
                  "rms median %.2f cm p95 %.2f cm | rank-deficient %.1f%%"
                  % (len(ts), n_rej, np.nanmedian(RMS) * 100,
                     np.nanpercentile(RMS, 95) * 100,
                     100 * np.mean(np.array(NOBS) < 6)))
            write_tum(os.path.join(outd, "traj_%s.tum" % name), ts, Ts)
            results[name] = dict(kind=kind, ts=ts, Ts=Ts,
                                 frame="lidar", rms=float(np.nanmedian(RMS)))

        elif kind == "rgbd_icp":
            if REF is None:
                REF = Reference(read_map_xyz(s["ref_map"]),
                                voxel=float(s.get("target_voxel", 0.05)),
                                plane_voxel=float(s.get("plane_voxel", 0.4)))
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
                          % (len(ts), rms * 100, nobs, time.time() - t0w),
                          flush=True)
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
            # mobile_2's three corrected trajectories from ONE estimator:
            #   A_icp    odometry + relinearised map factors (geometry only)
            #   B_boards odometry + board factors + session-anchor prior
            #   C_joint  everything
            # State = the COLOR optical frame; depth clouds pre-transformed
            # through the color->depth extrinsic so every factor agrees.
            if not track.get("cam_extrinsic_xyzquat"):
                print("  ! SKIPPING this track: cam_extrinsic_xyzquat is "
                      "REQUIRED (odometry child '%s' -> optical). Identity "
                      "bent this bag by 8-9.6 m. Get it:\n     ros2 run "
                      "tf2_ros tf2_echo %s <optical frame>" % (ochild, ochild))
                continue
            if REF is None:
                REF = Reference(read_map_xyz(s["ref_map"]),
                                voxel=float(s.get("target_voxel", 0.05)),
                                plane_voxel=float(s.get("plane_voxel", 0.4)))
            Xd = make_T_xyzq(track["depth_extrinsic_xyzquat"]) \
                if track.get("depth_extrinsic_xyzquat") else np.eye(4)
            if not track.get("depth_extrinsic_xyzquat"):
                print("  (no depth_extrinsic_xyzquat: assuming depth is "
                      "registered to the anchored optical frame - true for ZED "
                      "depth_registered, ~1.5 cm off for raw D455 depth)")
            Kd = None
            for _, ci in iter_topic(bag, track["depth_info_topic"], limit=1):
                Kd = np.array(ci.k).reshape(3, 3)
            rate = float(track.get("rate_hz", 10.0))
            keep_dt = 1.0 / rate
            beta = float(track.get("prior_beta", 0.10))
            print("  chained ICP pass (initialisation; depth fx=%.1f)" % Kd[0, 0])
            reg_t, reg_T, cl_l, RMS, NOBS = [], [], [], [], []
            n_rej = 0
            t_last = -1e18; T_prev = None; T_ol_prev = None; t0w = time.time()
            for t, m in iter_topic(bag, track["depth_topic"]):
                if t - t_last < keep_dt:
                    continue
                Pd = depth_to_cloud(img_depth(m), Kd,
                                    rmin=float(track.get("range_min", 0.4)),
                                    rmax=float(track.get("range_max", 3.5)))
                if len(Pd) < 500:
                    continue
                Pc = apply(Xd, Pd).astype(np.float32)   # cloud in COLOR frame
                T_ol = interp_traj(ot, oT, np.array([t]))[0]
                if T_prev is None:
                    T_seed = T_map_origin @ T_ol @ X
                else:
                    T_seed = T_prev @ (inv(X) @ inv(T_ol_prev) @ T_ol @ X)
                T_i, nu, rms, nobs = icp_frame(np.asarray(Pc, float), T_seed,
                                               REF, beta=beta)
                d = float(np.linalg.norm(T_i[:3, 3] - T_seed[:3, 3]))
                a = float(np.linalg.norm(log_R(T_seed[:3, :3].T @ T_i[:3, :3])))
                if d > float(track.get("max_shift", 0.3)) \
                        or a > math.radians(float(track.get("max_rot_deg", 5.0))):
                    T_i, rms, nobs = T_seed, np.nan, 0
                    n_rej += 1
                T_prev, T_ol_prev = T_i, T_ol
                t_last = t
                reg_t.append(t); reg_T.append(T_i); cl_l.append(Pc)
                RMS.append(rms); NOBS.append(nobs)
                if len(reg_t) % 200 == 0:
                    print("  %5d frames  rms %5.2f cm  obs %d/6  %5.1fs"
                          % (len(reg_t), (rms if np.isfinite(rms) else 0) * 100,
                             nobs, time.time() - t0w), flush=True)
            reg_t = np.array(reg_t); reg_T = np.array(reg_T)
            print("  %d frames (%d rejected -> seed kept) | rms median %.2f cm "
                  "| rank-deficient %.1f%%"
                  % (len(reg_t), n_rej, np.nanmedian(RMS) * 100,
                     100 * np.mean(np.array(NOBS) < 6)))
            sights = detect_boards_along(track, s, P, bmap, af, bag)
            # nodes FIRST (registration stamps + exact sighting stamps), then
            # resolve against the anchored odometry at those nodes - board
            # factors land on their own stamps, never a neighbour 50 ms away
            st_extra = np.array(sorted({round(t, 6) for t, _, _ in sights}))
            node_t = np.unique(np.round(np.r_[reg_t, st_extra], 6))
            idx_of = {round(t, 6): i for i, t in enumerate(node_t)}
            clouds = {idx_of[round(t, 6)]: c
                      for t, c in zip(np.round(reg_t, 6), cl_l)}
            To_n = interp_traj(ot, oT, node_t)
            Z_rel = np.array([inv(X) @ inv(To_n[i]) @ To_n[i + 1] @ X
                              for i in range(len(node_t) - 1)])
            T_init = interp_traj(reg_t, reg_T, node_t)
            To_anch = np.array([T_map_origin @ To_n[i] @ X
                                for i in range(len(node_t))])
            res = resolve_instances(sights, To_anch, node_t, bmap,
                                    track.get("boards") or sorted(bmap),
                                    float(track.get("instance_radius", 2.0)))
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
            anchor_prior = (0, T_init[0],
                            max(arec_of(track)["std_mm"] * 1e-3
                                if "std_mm" in arec_of(track) else 0.01, 0.005),
                            math.radians(1.0))
            print("  %d nodes, %d board factors, %d clouds"
                  % (len(node_t), len(abs_meas), len(clouds)))
            report_factor_coverage(node_t, [k for k, _, _, _ in abs_meas])
            sig_rel = (float(track.get("odom_sigma_t", 0.003)),
                       float(track.get("odom_sigma_r", 0.001)))
            ARMS = {}
            for arm, (ui, ub) in [("A_icp", (True, False)),
                                  ("B_boards", (False, True)),
                                  ("C_joint", (True, True))]:
                print("  == arm %s ==" % arm)
                am = abs_meas + [anchor_prior] if ub else []
                start = T_init
                if arm == "C_joint" and "B_boards" in ARMS:
                    # Start the joint arm from the BOARD-corrected solution.
                    # Scan-to-map ICP can only REFINE: given a trajectory that
                    # is metres out, it happily locks onto wrong-but-similar
                    # geometry and reports a small residual while staying
                    # wrong (measured here: 1.4 cm rms while ~11 m out). The
                    # boards are what remove a gross error; the map is what
                    # sharpens it afterwards. Arm A deliberately keeps the
                    # chained start - if geometry alone cannot relocalise,
                    # that IS arm A's honest answer.
                    start = ARMS["B_boards"]
                    print("     (initialised from arm B: ICP refines, it "
                          "cannot relocalise a metre-scale error)")
                Ts = solve_graph(node_t, start.copy(), Z_rel, sig_rel, am,
                                 clouds, REF, use_icp=ui, use_board=ub,
                                 icp_pts=int(track.get("icp_pts", 400)),
                                 iters=int(track.get("gn_iters", 12)))
                ARMS[arm] = Ts
                write_tum(os.path.join(outd, "traj_%s_%s.tum" % (name, arm)),
                          node_t, Ts)
            write_tum(os.path.join(outd, "traj_%s_odom_only.tum" % name),
                      node_t, To_anch)
            print("  == evaluation (off-diagonal cells are independent) ==")
            print("  %-10s %22s %22s %20s"
                  % ("arm", "board resid (cm)", "map rms (cm)", "vs C (cm)"))
            for arm, Ts in ARMS.items():
                br = eval_board_resid(Ts, res_nodes, bmap) * 100
                mr = eval_map_rms(Ts, clouds, REF) * 100
                dv = np.linalg.norm(Ts[:, :3, 3]
                                    - ARMS["C_joint"][:, :3, 3], axis=1) * 100
                print("  %-10s %10.1f med %6.1f p95 %9.2f med %5.2f p95 "
                      "%8.1f med %6.1f max"
                      % (arm, np.nanmedian(br), np.nanpercentile(br, 95),
                         np.nanmedian(mr), np.nanpercentile(mr, 95),
                         np.median(dv), dv.max()))
            print("  arm A board resid and arm B map rms are the honest cells "
                  "(neither arm saw that data). C should match or beat both.")
            results[name] = dict(kind=kind, ts=node_t, Ts=ARMS["C_joint"],
                                 frame="cam", arms=ARMS, res_nodes=res_nodes,
                                 bmap=bmap)

        elif kind == "cam_boards":
            rate = float(track.get("rate_hz", 10.0))
            keep = np.r_[0, np.flatnonzero(np.diff(ot) >= 0)[
                np.searchsorted(np.cumsum(np.diff(ot)),
                                np.arange(1.0 / rate, ot[-1] - ot[0], 1.0 / rate))]]
            keep = np.unique(np.clip(keep, 0, len(ot) - 1))
            node_t = ot[keep]
            To = oT[keep]
            T_init = np.array([T_map_origin @ To[i] @ X for i in range(len(keep))])
            Z_rel = np.array([inv(X) @ inv(To[i]) @ To[i + 1] @ X
                              for i in range(len(keep) - 1)])
            sights = detect_boards_along(track, s, P, bmap, af, bag)
            res = resolve_instances(sights, T_init, node_t, bmap,
                                    track.get("boards") or sorted(bmap),
                                    float(track.get("instance_radius", 2.0)))
            abs_meas = []
            for k, bname, T_map_b_pred, T_cb in res:
                Tb, rec = bmap[bname]
                # measured camera pose from the SURVEYED board
                T_meas = Tb @ inv(T_cb)
                sig_t = math.hypot(float(rec.get("std_mm", 10)) * 1e-3, 0.010)
                lc = rec.get("loop_closure") or {}
                sig_t = max(sig_t, float(lc.get("mm", 0)) * 1e-3)
                sig_r = math.radians(max(float(lc.get("deg", 0.3)), 1.0))
                abs_meas.append((k, T_meas, sig_t, sig_r))
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
                                 n_boards=len(res),
                                 res_nodes=[(k, b, T_cb) for k, b, _, T_cb in res],
                                 bmap=bmap)
        else:
            print("  ! unknown type '%s' - skipped" % kind)

    # -------- cross-check: two independent tracks of one rigid body -------- #
    lid = next((r for r in results.values() if r["kind"] == "lidar_icp"), None)
    zed = next((v for k, v in results.items()
                if v["kind"] in ("cam_boards", "arms") and "1" in k), None)
    if lid is not None and zed is not None:
        T_lc = P.sensor.T_lidar_camera
        tq = lid["ts"][(lid["ts"] >= zed["ts"][0]) & (lid["ts"] <= zed["ts"][-1])]
        if len(tq) > 10:
            Tl = interp_traj(lid["ts"], lid["Ts"], tq)
            Tlc_all = (Tl @ np.tile(T_lc, (len(tq), 1, 1)))[:, :3, 3]
            for an, aT in sorted((zed.get("arms") or {}).items()):
                ga = np.linalg.norm(
                    Tlc_all - interp_traj(zed["ts"], aT, tq)[:, :3, 3], axis=1)
                print("  arm %-10s vs lidar: median %7.1f cm  p95 %7.1f cm"
                      % (an, np.median(ga) * 100, np.percentile(ga, 95) * 100))
            Tz = interp_traj(zed["ts"], zed["Ts"], tq)
            gap = np.linalg.norm(Tlc_all - Tz[:, :3, 3], axis=1)
            print("\n== cross-check mobile_1: lidar-ICP track vs ZED-board track "
                  "(same rigid body through T_lidar_camera) ==")
            print("  translation gap: median %.1f cm  p95 %.1f cm  max %.1f cm "
                  "over %d stamps" % (np.median(gap) * 100,
                                      np.percentile(gap, 95) * 100,
                                      gap.max() * 100, len(tq)))
            # A constant gap is an extrinsic error; a growing one is drift in
            # whichever track has no absolute reference over that stretch.
            qs = np.linspace(0, len(tq) - 1, 6).astype(int)
            print("  gap over time:  " + "  ".join(
                "t=%.0fs %.0fcm" % (tq[i] - tq[0], gap[i] * 100) for i in qs))
            rel = (gap.max() - gap.min()) / max(gap.max(), 1e-9)
            print("  -> %s" % ("CONSTANT offset (%.0f%% variation): suspect "
                               "T_lidar_camera, not the trajectories"
                               % (100 * rel) if rel < 0.25 else
                               "GROWING (%.0f cm -> %.0f cm): one track is "
                               "drifting; the one WITHOUT absolute information "
                               "over that stretch is the suspect - check the "
                               "board-factor coverage line above"
                               % (gap.min() * 100, gap.max() * 100)))
            # the lidar track never used boards, so its own plane rms against
            # the map is an independent statement about it
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
                          "extrinsic convention and the other track is at fault")
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
                                  "surveyed pose: a real board IS there and it "
                                  "is NOT the surveyed one."
                                  % ("", float(np.linalg.norm(ctr - surv))))
                            print("      %-12s    Either a SECOND board of this "
                                  "design exists (survey it, or drop '%s' from "
                                  "this track's boards list), or the board MOVED "
                                  "after the survey - and if it moved, every "
                                  "session anchor derived from it is wrong too, "
                                  "which reaches other agents." % ("", bn))
                        elif spread < 0.5 and np.linalg.norm(ctr - surv) <= 0.2:
                            print("      %-12s    OK: the lidar track, this "
                                  "board's survey and the extrinsic convention "
                                  "all agree to %.0f cm - the map frame and the "
                                  "board frame ARE the same frame"
                                  % ("", 100 * float(np.linalg.norm(ctr - surv))))
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
        lid_r = next((r for r in results.values()
                      if r["kind"] == "lidar_icp"), None)
        if lid_r is not None and T_lidar_cam_ref is not None:
            resurvey_boards(results, lid_r, T_lidar_cam_ref, af, outd)
    except Exception as e:
        print("\n(re-survey failed: %s: %s)" % (type(e).__name__, e))
    try:
        save_paths_png(results, REF, bmap, outd, T_lidar_cam_ref)
    except Exception as e:
        print("\n(path plot failed: %s: %s)" % (type(e).__name__, e))
    print("\ndone -> %s" % outd)


SAMPLE_CONFIG = r"""
"08_reference": {
  "bag": "/path/to/mirc_dataset_coop2_20260828_merged",
  "session_anchor": "map_stages_20260828_outputs/session_anchor.json",
  "anchor_frame": "map_stages_20260828_outputs/anchor_frame.json",
  "ref_map": "map_final_20260828_nc_anchored.pcd",  <- MUST be the ANCHORED cloud
      (stage 03 output). It already is the denoised, dynamic-removed map; but
      denoised.pcd itself lives in the pre-anchor GLIM frame and would put a
      fixed R_align+offset error on every trajectory here.
  "out_dir": "map_stages_20260828_outputs/reference_coop2",
  "target_voxel": 0.05, "plane_voxel": 0.4,
  "tracks": [
    { "name": "mobile_1_lidar", "type": "lidar_icp",
      "points_topic": "/mobile_1/ouster/points",
      "odom_topic": "/mobile_1/zed/odom",
      "anchor_cam": "zed",
      "cam_extrinsic_xyzquat": [-0.010, 0.060, 0.015, -0.5, 0.5, -0.5, 0.5],
      "rate_hz": 5.0, "range_min": 0.7, "range_max": 15.0, "scan_voxel": 0.10 },
    { "name": "mobile_1_zed", "type": "arms",
      "odom_topic": "/mobile_1/zed/odom",
      "anchor_cam": "zed",
      "cam_extrinsic_xyzquat": [-0.010, 0.060, 0.015, -0.5, 0.5, -0.5, 0.5],
      "depth_topic": "/mobile_1/zed/depth/depth_registered",
      "depth_info_topic": "/mobile_1/zed/depth/camera_info",
      "depth_extrinsic_xyzquat": null,
      "image_topic": "/mobile_1/zed/left/image_rect_color",
      "camera_info_topic": "/mobile_1/zed/left/camera_info", "rectified": true,
      "boards": ["anchor", "anchor_b"],
      "rate_hz": 10.0, "img_stride": 2,
      "odom_sigma_t": 0.003, "odom_sigma_r": 0.001,
      "range_min": 0.4, "range_max": 5.0, "prior_beta": 0.10,
      "max_shift": 0.3, "max_rot_deg": 5.0, "icp_pts": 400, "gn_iters": 25 },

    { "name": "mobile_2", "type": "arms",
      "odom_topic": "/mobile_2/visual_slam/tracking/odometry",
      "anchor_cam": "realsense",
      "cam_extrinsic_xyzquat": null,
      "depth_topic": "/mobile_2/depth/image_rect_raw",
      "depth_info_topic": "/mobile_2/depth/camera_info",
      "depth_extrinsic_xyzquat": [0.05919025, -0.00000990, -0.00040596,
                                  -0.00296559, 0.00083214, 0.00130485, 0.99999441],
      "image_topic": "/mobile_2/color/image_raw",
      "camera_info_topic": "/mobile_2/color/camera_info", "rectified": false,
      "boards": ["rs_anchor", "anchor", "anchor_b"],
      "rate_hz": 10.0, "img_stride": 2,
      "odom_sigma_t": 0.003, "odom_sigma_r": 0.001,
      "range_min": 0.4, "range_max": 3.5, "prior_beta": 0.10,
      "max_shift": 0.3, "max_rot_deg": 5.0, "icp_pts": 400, "gn_iters": 25 }
  ]
}
"""

if __name__ == "__main__":
    main()

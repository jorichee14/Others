#!/usr/bin/env python3
"""
STAGE 09 - mobile_2 reference pose, three ways: ICP, boards, ICP+boards.

One estimator, three factor sets. All arms share the same nodes, the same
odometry factors and the same solver; they differ ONLY in which absolute
information is enabled, so the comparison is clean:

  arm A  (icp)     VSLAM relative motion + depth-cloud point-to-plane factors
                   against the frozen anchored map. Geometry-only: no boards
                   (the session anchor seeds the initialisation but no board
                   factor constrains the solution).
  arm B  (boards)  VSLAM relative motion + board-sighting pose factors + the
                   session-anchor prior. No map geometry.
  arm C  (joint)   everything.

THE ONE LESSON THAT SHAPES THIS FILE (measured, not theoretical): chained ICP
poses must NEVER enter the graph as absolute pose factors. Each chained pose is
seeded from the previous one, so the chain's own drift accumulates as fake
evidence - on a synthetic corridor it out-weighed the boards (2.3e6 vs 1.9e6 of
information along the blind axis) and arm C lost to arm B. Instead the ICP
factors are point-to-plane residuals RE-LINEARISED against the map at every
Gauss-Newton iteration: they contain no seed at all. The chained pass survives
only as the initialisation and as per-frame diagnostics.

Validated end-to-end on a synthetic corridor with ground truth (drifting,
mis-scaled odometry; boards with survey-grade error injected):
  arm A  med 1.55 / max  8.14 cm   rot 0.15 deg   (drifts where degenerate)
  arm B  med 4.11 / max 13.55 cm   rot 0.96 deg   (odometry drift between boards)
  arm C  med 1.49 / max  5.31 cm   rot 0.15 deg   (each arm fixes the other)

EVALUATION PRINTED AT THE END - the off-diagonal cells are the honest ones:
arm A never used a board, so its board residual is an independent check; arm B
never used the map, so its map rms is one. Arm C should be at least as good as
either specialist on the other's home turf.

GRAPH STATE is the COLOR optical frame (the session-anchor frame). Depth
clouds are pre-transformed color<-depth once at extraction, board measurements
are native color - so every factor speaks the same frame.

  python3 09_m2_arms.py [pipeline_config.json]

Config: "09_m2_arms" block, sample at the bottom of this file.
cam_extrinsic_xyzquat is REQUIRED: running with an identity guess bent the
whole trajectory by 8-9.6 m on this very bag (odometry child frames are body
convention, ~90 deg from optical).
"""
import os
import sys
import json
import math
import time
import numpy as np

from pipeline_common import load_pipeline, R_to_q
from pipeline_boards import Board, read_bag, pick_intrinsics

from scipy.spatial.transform import Rotation as Rot
from scipy import sparse, ndimage
from scipy.sparse.linalg import spsolve


# --------------------------------------------------------------------------- #
# SE(3) (validated: exact jr_inv, numerically checked Jacobians)
# --------------------------------------------------------------------------- #
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


def make_T_xyzq(v):
    return Rt(Rot.from_quat(v[3:7]).as_matrix(), np.asarray(v[0:3], float))


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


# --------------------------------------------------------------------------- #
# bag access
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


def img16(m):
    a = np.frombuffer(m.data, np.uint8)
    return a.view(np.uint16).reshape(m.height, m.step // 2)[:, :m.width]


def voxel_centroid(P, v):
    """Voxel CENTROID (never centres), float64 accumulator (float32 cumsum
    rounds away millimetres over 1e5+ points - measured)."""
    q = np.floor(P / v).astype(np.int64)
    q -= q.min(0)
    key = (q[:, 0] << 40) | (q[:, 1] << 20) | q[:, 2]
    o = np.argsort(key, kind="stable"); key = key[o]; Ps = P[o]
    br = np.r_[0, np.flatnonzero(np.diff(key)) + 1, len(key)]
    cs = np.vstack([np.zeros(3), np.cumsum(Ps.astype(np.float64), 0)])
    return ((cs[br[1:]] - cs[br[:-1]]) / np.diff(br)[:, None]).astype(np.float32)


def depth_to_cloud(d16, K, scale=0.001, rmin=0.4, rmax=3.5,
                   edge_jump=0.05, voxel=0.05, max_pts=8000):
    z = d16.astype(np.float32) * scale
    z[(z < rmin) | (z > rmax)] = 0
    zz = np.where(z > 0, z, np.nan)
    hi = ndimage.maximum_filter(np.nan_to_num(zz, nan=-1e3), size=3)
    lo = ndimage.minimum_filter(np.nan_to_num(zz, nan=1e3), size=3)
    z[(hi - lo) > edge_jump] = 0          # flying pixels sit BETWEEN surfaces
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


def read_map_xyz(path):
    import open3d as o3d
    P = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    if len(P) == 0:
        raise SystemExit("no points in %s" % path)
    return P


# --------------------------------------------------------------------------- #
# frozen map (soft planarity, voxel-membership matching - both paid for)
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
              huber=0.05, beta=0.10):
    """Chained-pass ICP (initialisation + diagnostics only, see module doc)."""
    T = T_init.copy(); rms = np.nan; eig = np.zeros(6)
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
            rms = float(np.sqrt(np.mean(ww * r * r) / max(ww.mean(), 1e-9)))
            ev = np.linalg.eigvalsh(S @ H @ S)
            eig = ev / max(ev.max(), 1e-12)
            if np.linalg.norm(d[:3]) < 1e-4 and np.linalg.norm(d[3:]) < 1e-5:
                break
    return T, rms, int((eig > 0.02).sum())


# --------------------------------------------------------------------------- #
# the one graph, three factor sets
# --------------------------------------------------------------------------- #
ICP_SIGMA = 0.02          # m; the map's own surface noise - a frame cannot
                          # claim to localise better than the map is built
HUBER = 0.05
GAUGE_W = 1e-2


def solve_graph(node_t, T_init, Z_rel, sig_rel, abs_meas, clouds, ref,
                use_icp, use_board, icp_pts=400, iters=12, verbose=True):
    """node_t (N,), T_init (N,4,4); Z_rel (N-1,4,4) odometry relative motions;
    abs_meas: [(node, T_meas, sig_t, sig_r)] board/anchor factors;
    clouds: {node_index: (M,3) float32 cloud in the STATE frame}.
    ICP factors are point-to-plane residuals re-linearised each iteration."""
    n = len(node_t); Ts = T_init.copy()
    dt = np.maximum(np.diff(node_t), 1e-3)
    st, sr = sig_rel
    sub = {}
    if use_icp:
        for k, P in clouds.items():
            if len(P) > icp_pts:
                P = P[np.linspace(0, len(P) - 1, icp_pts).astype(int)]
            sub[k] = np.asarray(P, float)
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
                B = W @ Jb
                rows, cc = np.meshgrid(np.arange(6), np.arange(6), indexing="ij")
                add(rows.ravel(), [6 * k + c for c in cc.ravel()], B.ravel(),
                    list(W @ res))
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
                ww = w * np.minimum(1.0, HUBER / np.maximum(np.abs(r), 1e-9)) \
                    / ICP_SIGMA
                Jp = np.hstack([nn, np.cross(p, nn @ R)]) * ww[:, None]
                rows, cc = np.meshgrid(np.arange(len(r)), np.arange(6),
                                       indexing="ij")
                add(rows.ravel(), [6 * k + cix for cix in cc.ravel()],
                    Jp.ravel(), list(r * ww))
        A = sparse.csr_matrix((V_, (I_, J_)), shape=(len(r_), 6 * n))
        rv = np.array(r_)
        Hn = (A.T @ A).tocsc() + sparse.identity(6 * n, format="csc") * 1e-6
        dx = spsolve(Hn, -(A.T @ rv))
        step = 0.0
        for k in range(n):
            dk = dx[6 * k:6 * k + 6]
            Ts[k] = Rt(Ts[k][:3, :3] @ exp_r(dk[3:]), Ts[k][:3, 3] + dk[:3])
            step = max(step, float(np.linalg.norm(dk[:3])))
        if verbose:
            print("    it%2d cost %.1f max step %.2f mm"
                  % (it, float(rv @ rv), step * 1000))
        if step < 1e-5:
            break
    return Ts


# --------------------------------------------------------------------------- #
def detect_boards(track, cfg, bmap, af, bag):
    board_cfgs = cfg.get("boards", {})
    axes = af.get("board_axes", "opencv"); borig = af.get("board_origin", "corner")
    wanted = track.get("boards") or sorted(bmap)
    designs = sorted({bmap[b][1].get("design", b) for b in wanted if b in bmap})
    dets = {}
    for dgn in designs:
        if dgn in board_cfgs:
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
            if d is not None:
                out.append((st, dgn, d.T @ fix))
    print("  %d raw sightings" % len(out))
    return out


def resolve_instances(sights, Ts_est, node_t, bmap, wanted, radius=2.0):
    out, dropped = [], 0
    for t, dgn, T_cb in sights:
        k = int(np.argmin(np.abs(node_t - t)))
        if abs(node_t[k] - t) > 0.05:
            dropped += 1; continue
        T_map_b = Ts_est[k] @ T_cb
        best, bd = None, radius
        for name in wanted:
            if name not in bmap or bmap[name][1].get("design", name) != dgn:
                continue
            dd = float(np.linalg.norm(bmap[name][0][:3, 3] - T_map_b[:3, 3]))
            if dd < bd:
                best, bd = name, dd
        if best is None:
            dropped += 1; continue
        out.append((k, best, T_cb))
    print("  %d sightings resolved, %d dropped" % (len(out), dropped))
    if dropped > len(out):
        print("  !! most sightings failed to resolve - check "
              "cam_extrinsic_xyzquat (frame convention) before blaming boards")
    return out


def eval_board_resid(Ts, res, bmap):
    """Predicted vs surveyed board position at every sighting. For an arm that
    never used boards this is an INDEPENDENT accuracy check."""
    e = [np.linalg.norm((Ts[k] @ T_cb)[:3, 3] - bmap[b][0][:3, 3])
         for k, b, T_cb in res]
    return np.array(e)


def eval_map_rms(Ts, clouds, ref, cap=300):
    """Point-to-plane rms of the depth clouds at the given poses (evaluated,
    not optimised). For an arm that never used the map: independent check."""
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
    return np.array(out)


# --------------------------------------------------------------------------- #
def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "pipeline_config.json"
    P = load_pipeline(cfg_path)
    s = P.cfg.get("09_m2_arms")
    if s is None:
        raise SystemExit("add a '09_m2_arms' block (sample at the bottom of "
                         "this file)")
    bag = s["bag"]
    outd = s.get("out_dir", "m2_arms_out")
    os.makedirs(outd, exist_ok=True)

    if not s.get("cam_extrinsic_xyzquat"):
        raise SystemExit(
            "cam_extrinsic_xyzquat is REQUIRED (odometry child frame -> color "
            "optical). Identity bent this bag's trajectory by 8-9.6 m. Get it:\n"
            "  ros2 run tf2_ros tf2_echo <odom child_frame_id> "
            "camera_color_optical_frame")
    X = make_T_xyzq(s["cam_extrinsic_xyzquat"])
    Xd = make_T_xyzq(s["depth_extrinsic_xyzquat"]) \
        if s.get("depth_extrinsic_xyzquat") else np.eye(4)

    sa = json.load(open(s["session_anchor"]))
    af = json.load(open(s["anchor_frame"]))
    bmap = {}
    for name, rec in (af.get("boards") or {}).items():
        T = np.eye(4)
        T[:3, :3] = Rot.from_quat(rec["qxyzw"]).as_matrix()
        T[:3, 3] = rec["xyz"]
        bmap[name] = (T, rec)
    arec = sa["cameras"][s.get("anchor_cam", "realsense")]
    A_T0 = make_T_xyzq(arec["map_to_cam"]["xyz"] + arec["map_to_cam"]["qxyzw"])
    print("session anchor (color optical): %s"
          % np.round(A_T0[:3, 3], 3).tolist())

    REF = Reference(read_map_xyz(s["ref_map"]),
                    voxel=float(s.get("target_voxel", 0.05)),
                    plane_voxel=float(s.get("plane_voxel", 0.4)))

    # ---- odometry, anchored in the color frame ----
    ot, oT, ochild = read_odom(bag, s["odom_topic"])
    t_anchor = arec.get("dwell_t_end") or ot[0]
    T_o0 = interp_traj(ot, oT, np.array([min(max(t_anchor, ot[0]), ot[-1])]))[0]
    T_map_origin = A_T0 @ inv(T_o0 @ X)

    # ---- chained ICP pass: initialisation + clouds + diagnostics ----
    Kd = None
    for _, ci in iter_topic(bag, s["depth_info_topic"], limit=1):
        Kd = np.array(ci.k).reshape(3, 3)
    rate = float(s.get("rate_hz", 10.0))
    keep_dt = 1.0 / rate
    print("\n== chained ICP pass (initialisation; depth fx=%.1f) ==" % Kd[0, 0])
    reg_t, reg_T, clouds_l, RMS, NOBS = [], [], [], [], []
    t_last = -1e18; T_prev = None; T_ol_prev = None; n_rej = 0
    t0w = time.time()
    for t, m in iter_topic(bag, s["depth_topic"]):
        if t - t_last < keep_dt:
            continue
        Pd = depth_to_cloud(img16(m), Kd,
                            rmin=float(s.get("range_min", 0.4)),
                            rmax=float(s.get("range_max", 3.5)))
        if len(Pd) < 500:
            continue
        Pc = apply(Xd, Pd).astype(np.float32)      # cloud in the COLOR frame
        T_ol = interp_traj(ot, oT, np.array([t]))[0]
        if T_prev is None:
            T_seed = T_map_origin @ T_ol @ X
        else:
            T_seed = T_prev @ (inv(X) @ inv(T_ol_prev) @ T_ol @ X)
        T_i, rms, nobs = icp_frame(np.asarray(Pc, float), T_seed, REF,
                                   beta=float(s.get("prior_beta", 0.10)))
        d = float(np.linalg.norm(T_i[:3, 3] - T_seed[:3, 3]))
        a = float(np.linalg.norm(log_R(T_seed[:3, :3].T @ T_i[:3, :3])))
        if d > float(s.get("max_shift", 0.3)) \
                or a > math.radians(float(s.get("max_rot_deg", 5.0))):
            T_i, rms, nobs = T_seed, np.nan, 0
            n_rej += 1
        T_prev, T_ol_prev = T_i, T_ol
        t_last = t
        reg_t.append(t); reg_T.append(T_i); clouds_l.append(Pc)
        RMS.append(rms); NOBS.append(nobs)
        if len(reg_t) % 200 == 0:
            print("  %5d frames  rms %5.2f cm  obs %d/6  %5.1fs"
                  % (len(reg_t), (rms if np.isfinite(rms) else 0) * 100,
                     nobs, time.time() - t0w), flush=True)
    reg_t = np.array(reg_t); reg_T = np.array(reg_T)
    print("  %d frames (%d rejected -> seed kept) | rms median %.2f cm | "
          "rank-deficient %.1f%%"
          % (len(reg_t), n_rej, np.nanmedian(RMS) * 100,
             100 * np.mean(np.array(NOBS) < 6)))

    # ---- board sightings ----
    print("\n== board sightings ==")
    sights = detect_boards(s, P.cfg, bmap, af, bag)
    # resolve against the anchored ODOMETRY (arm-B-clean; good enough now that
    # the extrinsic is right and drift is ~1 m against 7-17 m board spacing)
    To_all = np.array([T_map_origin @ interp_traj(ot, oT, np.array([t]))[0] @ X
                       for t in reg_t])
    res = resolve_instances(sights, To_all, reg_t, bmap,
                            s.get("boards") or sorted(bmap),
                            float(s.get("instance_radius", 2.0)))

    # ---- nodes: registration stamps ∪ sighting stamps ----
    st_extra = np.array(sorted({round(t, 6) for t, _, _ in sights}))
    node_t = np.unique(np.round(np.r_[reg_t, st_extra], 6))
    idx_of = {round(t, 6): i for i, t in enumerate(node_t)}
    clouds = {idx_of[round(t, 6)]: c for t, c in zip(np.round(reg_t, 6), clouds_l)}
    To_n = interp_traj(ot, oT, node_t)
    Z_rel = np.array([inv(X) @ inv(To_n[i]) @ To_n[i + 1] @ X
                      for i in range(len(node_t) - 1)])
    # init: chained ICP where available, anchored odometry elsewhere
    chain = interp_traj(reg_t, reg_T, node_t)
    T_init = chain.copy()
    abs_meas = []
    res_nodes = []
    for k_reg, bname, T_cb in res:
        k = idx_of.get(round(reg_t[k_reg], 6))
        if k is None:
            continue
        Tb, rec = bmap[bname]
        T_meas = Tb @ inv(T_cb)
        sig_t = math.hypot(float(rec.get("std_mm", 10)) * 1e-3, 0.010)
        lc = rec.get("loop_closure") or {}
        sig_t = max(sig_t, float(lc.get("mm", 0)) * 1e-3)
        sig_r = math.radians(max(float(lc.get("deg", 0.3)), 1.0))
        abs_meas.append((k, T_meas, sig_t, sig_r))
        res_nodes.append((k, bname, T_cb))
    # session-anchor prior (arms B and C; arm A stays geometry-only)
    anchor_prior = (0, T_init[0],
                    max(arec.get("std_mm", 10) * 1e-3, 0.005),
                    math.radians(1.0))
    print("\n%d nodes, %d board factors, %d clouds"
          % (len(node_t), len(abs_meas), len(clouds)))

    sig_rel = (float(s.get("odom_sigma_t", 0.003)),
               float(s.get("odom_sigma_r", 0.001)))
    ARMS = {}
    for arm, (ui, ub) in [("A_icp", (True, False)),
                          ("B_boards", (False, True)),
                          ("C_joint", (True, True))]:
        print("\n== arm %s ==" % arm)
        am = abs_meas + [anchor_prior] if ub else []
        Ts = solve_graph(node_t, T_init.copy(), Z_rel, sig_rel, am,
                         clouds, REF, use_icp=ui, use_board=ub,
                         icp_pts=int(s.get("icp_pts", 400)),
                         iters=int(s.get("gn_iters", 12)))
        ARMS[arm] = Ts
        write_tum(os.path.join(outd, "traj_m2_%s.tum" % arm), node_t, Ts)
    write_tum(os.path.join(outd, "traj_m2_odom_only.tum"), node_t,
              np.array([T_map_origin @ To_n[i] @ X for i in range(len(node_t))]))

    # ---- the ablation table ----
    print("\n== evaluation (off-diagonal cells are the independent ones) ==")
    print("%-10s %20s %20s %22s" % ("arm", "board resid (cm)", "map rms (cm)",
                                    "vs arm C (cm)"))
    for arm, Ts in ARMS.items():
        br = eval_board_resid(Ts, res_nodes, bmap) * 100
        mr = eval_map_rms(Ts, clouds, REF) * 100
        dv = np.linalg.norm(Ts[:, :3, 3] - ARMS["C_joint"][:, :3, 3], axis=1) * 100
        print("%-10s %9.1f med %5.1f p95 %7.2f med %5.2f p95 %9.1f med %6.1f max"
              % (arm, np.median(br), np.percentile(br, 95),
                 np.median(mr), np.percentile(mr, 95),
                 np.median(dv), dv.max()))
    print("""
Reading it:
  arm A's board resid  - independent (A never saw a board): its honest accuracy
  arm B's map rms      - independent (B never saw the map): its honest accuracy
  arm C               - should match or beat each specialist on the OTHER's column;
                        if it is worse than either, a weight is wrong (board sigma
                        first, odom_sigma_t second)""")
    print("done -> %s" % outd)


SAMPLE_CONFIG = r"""
"09_m2_arms": {
  "bag": "/home/wicoms-robot/workspaces/isaac_ros-dev/data/raw/20260828/mirc_dataset_coop2_20260828_merged",
  "session_anchor": "map_stages_20260828_outputs/session_anchor.json",
  "anchor_frame": "map_stages_20260828_outputs/anchor_frame.json",
  "ref_map": "map_final_20260828_nc_anchored.pcd",
  "out_dir": "map_stages_20260828_outputs/m2_arms",
  "anchor_cam": "realsense",
  "odom_topic": "/mobile_2/visual_slam/tracking/odometry",
  "cam_extrinsic_xyzquat": null,
  "depth_topic": "/mobile_2/depth/image_rect_raw",
  "depth_info_topic": "/mobile_2/depth/camera_info",
  "depth_extrinsic_xyzquat": [0.05919025, -0.00000990, -0.00040596,
                              -0.00296559, 0.00083214, 0.00130485, 0.99999441],
  "image_topic": "/mobile_2/color/image_raw",
  "camera_info_topic": "/mobile_2/color/camera_info",
  "rectified": false,
  "boards": ["rs_anchor", "anchor", "anchor_b"],
  "rate_hz": 10.0, "img_stride": 2,
  "odom_sigma_t": 0.003, "odom_sigma_r": 0.001,
  "range_min": 0.4, "range_max": 3.5, "prior_beta": 0.10,
  "max_shift": 0.3, "max_rot_deg": 5.0,
  "icp_pts": 400, "gn_iters": 12
}

cam_extrinsic_xyzquat: REQUIRED. From mobile_2:
  ros2 run tf2_ros tf2_echo <odom child_frame_id> camera_color_optical_frame
"""

if __name__ == "__main__":
    main()

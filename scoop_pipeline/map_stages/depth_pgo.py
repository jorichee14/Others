#!/usr/bin/env python3
"""
A depth track as one map-anchored pose graph (04 "tracks" entry with
"method": "pgo"; depth_track.py reads the frames, the odometry, the boards and
writes the outputs, this solves for the poses).

  variables   one planar pose (x, y, yaw) per depth frame; height and tilt
              from the map (ICP) where the frame registers, else the start's
              (a ground robot on a floor)
  odometry    between consecutive frames of an odometry piece: cuVSLAM's
              relative motion (odo_sigma_m + odo_sigma_frac of the step,
              odo_sigma_deg)
  dropouts    across a break the odometry may run on over (depth_track's
              dropouts: a camera dropout, IMU alone): its relative motion,
              weak (gap_sigma_m + gap_sigma_mps x dt, gap_sigma_deg +
              gap_sigma_dps x dt) and robust -- what it did without images may
              be wrong; across any other break only "it cannot have moved
              faster than speed_max"
  anchor      the start: 03's session anchor (anchor_sigma_m / _deg)
  boards      every board sighting, the copy of the design the depth frame
              registers at (depth_track.from_board), board_sigma_m / _deg
  map         a) search: every pgo_search_every_s a stack of pgo_stack_s of
                 frames is searched for on the map within pgo_search_m of the
                 current estimate (depth_track.refind: taken when unambiguous),
                 search_sigma_m / _deg -- once per radius in pgo_search_m
                 A stretch reached from the anchor only across a break the
                 odometry cannot run on (a reset), and with no board, is first
                 searched for within pgo_free_search_m and any heading, until
                 one stack of it is found (its odometry carries the rest).
              b) ICP: every frame registered to the map from the current
                 estimate (within pgo_icp_shift, at least pgo_icp_min_fit of
                 its points on the map), its information from the
                 point-to-plane Hessian over (x, y, yaw) -- a corridor fixes
                 across it, not along it; floor points fix nothing -- scaled
                 so a fully constrained frame is map_sigma_m; pgo_icp_iters
                 passes
  solve       Gauss-Newton on the sparse normal equations, a Cauchy loss
              (robust_c sigmas) on all but the odometry within a piece; the
              graph is solved after each step above, the frames re-matched to
              the map from where it puts them.

Each pose's uncertainty: its marginal covariance in the solved graph
(Graph.marginals), written per frame by depth_track (cov_<track>.csv: sigma x,
y, yaw, the x-y correlation, the major semi-axis of the xy ellipse) -- where
the graph holds the robot well (boards, map-rich views) and where it does not
(across dropouts).

Nothing sees the robot where it has no images; there the pose is where the
odometry within its pieces, the dropouts' weak links and the absolute fixes on
both sides put it together.
"""
import time

import numpy as np

DEFAULTS = dict(pgo_search_m=[2.0, 1.0], pgo_search_every_s=1.0, pgo_stack_s=1.0,
                pgo_search_yaw_deg=20.0, pgo_free_search_m=4.0, pgo_icp_iters=3,
                pgo_icp_shift=0.3, pgo_icp_min_fit=0.8,
                odo_sigma_m=0.01, odo_sigma_frac=0.02, odo_sigma_deg=0.3,
                gap_sigma_m=0.2, gap_sigma_mps=0.5, gap_sigma_deg=5.0, gap_sigma_dps=10.0,
                map_sigma_m=0.04, search_sigma_m=0.05, search_sigma_deg=1.0,
                board_sigma_m=0.05, board_sigma_deg=2.0,
                anchor_sigma_m=0.05, anchor_sigma_deg=1.0, robust_c=3.0, gn_iters=20)


def yaw_of(T):
    return np.arctan2(T[..., 1, 0], T[..., 0, 0])


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def planar_rel(Ta, Tb):
    """(dx, dy, dyaw) of Tb in Ta's body frame."""
    D = np.linalg.inv(Ta) @ Tb
    return np.array([D[0, 3], D[1, 3], yaw_of(D)])


class Graph:
    """Factors over planar poses X (n, 3)."""
    def __init__(self, n):
        self.n = n
        self.rel = []                                # (i, j, m(3), W(3,3), robust)
        self.abs = []                                # (k, m(3), W(3,3), robust, kind)

    def add_rel(self, i, j, m, W, robust):
        self.rel.append((i, j, m, W, robust))

    def add_abs(self, k, m, W, robust, kind):
        self.abs.append((k, m, W, robust, kind))

    def residuals(self, X):
        """Whitened residual norms per factor (rel, abs)."""
        out_r, out_a = [], []
        for i, j, m, W, _ in self.rel:
            out_r.append(np.linalg.norm(W @ self._e_rel(X, i, j, m)))
        for k, m, W, _, _ in self.abs:
            e = np.r_[X[k, :2] - m[:2], wrap(X[k, 2] - m[2])]
            out_a.append(np.linalg.norm(W @ e))
        return np.array(out_r), np.array(out_a)

    @staticmethod
    def _e_rel(X, i, j, m):
        c, s = np.cos(X[i, 2]), np.sin(X[i, 2])
        d = X[j, :2] - X[i, :2]
        return np.r_[c * d[0] + s * d[1] - m[0], -s * d[0] + c * d[1] - m[1],
                     wrap(X[j, 2] - X[i, 2] - m[2])]

    def _arrays(self):
        R, A_ = self.rel, self.abs
        return (np.array([f[0] for f in R], int), np.array([f[1] for f in R], int),
                np.array([f[2] for f in R]).reshape(-1, 3), np.array([f[3] for f in R]).reshape(-1, 3, 3),
                np.array([f[4] for f in R], bool),
                np.array([f[0] for f in A_], int), np.array([f[1] for f in A_]).reshape(-1, 3),
                np.array([f[2] for f in A_]).reshape(-1, 3, 3), np.array([f[3] for f in A_], bool))

    def linearize(self, X, c, arr=None):
        """Whitened, robust-weighted Jacobian (sparse) and residual at X."""
        from scipy import sparse
        ri, rj, rm, rW, rr, ak, am, aW, ar = arr if arr is not None else self._arrays()
        n, nR, nA = self.n, len(ri), len(ak)
        cr = float(c["robust_c"])
        # relative factors
        th = X[ri, 2]
        cs, sn = np.cos(th), np.sin(th)
        d = X[rj, :2] - X[ri, :2]
        e = np.column_stack([cs * d[:, 0] + sn * d[:, 1] - rm[:, 0],
                             -sn * d[:, 0] + cs * d[:, 1] - rm[:, 1],
                             wrap(X[rj, 2] - X[ri, 2] - rm[:, 2])])
        Ji = np.zeros((nR, 3, 3))
        Jj = np.zeros((nR, 3, 3))
        Ji[:, 0, 0], Ji[:, 0, 1] = -cs, -sn
        Ji[:, 1, 0], Ji[:, 1, 1] = sn, -cs
        Ji[:, 0, 2] = -sn * d[:, 0] + cs * d[:, 1]
        Ji[:, 1, 2] = -cs * d[:, 0] - sn * d[:, 1]
        Ji[:, 2, 2] = -1.0
        Jj[:, 0, 0], Jj[:, 0, 1] = cs, sn
        Jj[:, 1, 0], Jj[:, 1, 1] = -sn, cs
        Jj[:, 2, 2] = 1.0
        r_rel = np.einsum("kab,kb->ka", rW, e)
        Ji = np.einsum("kab,kbc->kac", rW, Ji)
        Jj = np.einsum("kab,kbc->kac", rW, Jj)
        w = np.ones(nR)
        nr = np.linalg.norm(r_rel, axis=1)
        w[rr] = 1.0 / (1.0 + (nr[rr] / cr) ** 2)
        sw = np.sqrt(w)
        r_rel *= sw[:, None]
        Ji *= sw[:, None, None]
        Jj *= sw[:, None, None]
        # absolute factors
        ea = np.column_stack([X[ak, :2] - am[:, :2], wrap(X[ak, 2] - am[:, 2])])
        r_abs = np.einsum("kab,kb->ka", aW, ea)
        wa = np.ones(nA)
        na = np.linalg.norm(r_abs, axis=1)
        wa[ar] = 1.0 / (1.0 + (na[ar] / cr) ** 2)
        swa = np.sqrt(wa)
        r_abs *= swa[:, None]
        Ja = aW * swa[:, None, None]
        # sparse Jacobian
        rows, cols, vals = [], [], []
        kk = np.arange(nR)
        for blk, idx in ((Ji, ri), (Jj, rj)):
            for a in range(3):
                for b in range(3):
                    rows.append(3 * kk + a)
                    cols.append(3 * idx + b)
                    vals.append(blk[:, a, b])
        ka = np.arange(nA)
        for a in range(3):
            for b in range(3):
                rows.append(3 * nR + 3 * ka + a)
                cols.append(3 * ak + b)
                vals.append(Ja[:, a, b])
        J = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(3 * (nR + nA), 3 * n))
        return J, np.concatenate([r_rel.ravel(), r_abs.ravel()])

    def solve(self, X0, c, log=print):
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        X = X0.copy()
        n = self.n
        arr = self._arrays()
        for it in range(int(c["gn_iters"])):
            J, r = self.linearize(X, c, arr)
            H = (J.T @ J + 1e-6 * sparse.identity(3 * n)).tocsc()
            dx = -spsolve(H, J.T @ r)
            X += dx.reshape(n, 3)
            if float(np.max(np.abs(dx))) < 1e-4:
                break
        return X

    def marginals(self, X, c, chunk=600):
        """Each pose's marginal covariance (n, 3, 3) in (x [m], y [m], yaw [rad]):
        the 3x3 diagonal blocks of the inverse information matrix at X, with
        the robust weights there (Laplace approximation; as calibrated as the
        factors' sigmas)."""
        from scipy import sparse
        from scipy.sparse.linalg import splu
        n = self.n
        J, _ = self.linearize(X, c)
        lu = splu((J.T @ J + 1e-6 * sparse.identity(3 * n)).tocsc())
        cov = np.zeros((n, 3, 3))
        for k0 in range(0, n, chunk):
            k1 = min(n, k0 + chunk)
            B = np.zeros((3 * n, 3 * (k1 - k0)))
            B[3 * k0:3 * k1, :] = np.eye(3 * (k1 - k0))
            S = lu.solve(B)
            for k in range(k0, k1):
                q = 3 * (k - k0)
                cov[k] = S[3 * k:3 * k + 3, q:q + 3]
        return 0.5 * (cov + cov.transpose(0, 2, 1))


def run_pgo(A, ref, scans, times, T_odom, piece, o_t, pc_o, soft, T_anchor, k_anchor,
            fix_at, c, refind, from_board, step, log=print):
    """-> (T (n,4,4) map poses, rows, history). T_odom: the odometry at every
    frame (interpolated over breaks too), soft: depth_track.dropouts."""
    c = dict(DEFAULTS, **c)
    n = len(scans)
    t_start = time.time()
    R_tilt = rotz(-yaw_of(T_anchor)) @ T_anchor[:3, :3]     # the start's tilt: a level floor
    z = np.full(n, T_anchor[2, 3])

    def lift(X, k):
        T = np.eye(4)
        T[:3, :3] = rotz(X[k, 2]) @ R_tilt
        T[:3, 3] = (X[k, 0], X[k, 1], z[k])
        return T

    # the odometry's piece at every frame (inside a break: the one before it)
    io = np.clip(np.searchsorted(o_t, times, side="right") - 1, 0, len(o_t) - 1)
    pk = pc_o[io]

    def W_diag(sm, sd):
        return np.diag([1 / sm, 1 / sm, 1 / np.radians(sd)])

    G = Graph(n)
    n_odo = n_gap = n_free = 0
    seg = np.zeros(n, int)                           # stretches between resets
    for k in range(n - 1):
        seg[k + 1] = seg[k]
        dt = times[k + 1] - times[k]
        m = planar_rel(T_odom[k], T_odom[k + 1])
        if piece[k] >= 0 and piece[k] == piece[k + 1]:
            G.add_rel(k, k + 1, m, W_diag(float(c["odo_sigma_m"]) + float(c["odo_sigma_frac"]) *
                                          np.hypot(m[0], m[1]),
                                          float(c["odo_sigma_deg"]) + 0.02 * np.degrees(abs(m[2]))),
                      False)
            n_odo += 1
            continue
        lo, hi = sorted((int(pk[k]), int(pk[k + 1])))
        if hi == lo or soft[lo + 1:hi + 1].all():
            G.add_rel(k, k + 1, m, W_diag(float(c["gap_sigma_m"]) + float(c["gap_sigma_mps"]) * dt,
                                          float(c["gap_sigma_deg"]) + float(c["gap_sigma_dps"]) * dt),
                      True)
            n_gap += 1
        else:
            G.add_rel(k, k + 1, np.zeros(3), W_diag(0.3 + float(c["speed_max"]) * dt,
                                                    min(170.0, 30.0 + float(c["yaw_rate_max"]) * dt)),
                      True)
            n_free += 1
            seg[k + 1] = seg[k] + 1
    G.add_abs(k_anchor, np.r_[T_anchor[0, 3], T_anchor[1, 3], yaw_of(T_anchor)],
              W_diag(float(c["anchor_sigma_m"]), float(c["anchor_sigma_deg"])), False, "anchor")
    # boards: the copy the depth frame registers at
    n_b = 0
    for j, cands in sorted(fix_at.items()):
        hit = from_board(A, ref, scans[j], cands, c)
        if hit is not None:
            Tb = hit[0]
            G.add_abs(j, np.r_[Tb[0, 3], Tb[1, 3], yaw_of(Tb)],
                      W_diag(float(c["board_sigma_m"]), float(c["board_sigma_deg"])), True, "board")
            n_b += 1
    held = {int(seg[k_anchor])} | {int(seg[f[0]]) for f in G.abs if f[4] == "board"}
    free = ~np.isin(seg, list(held))
    log("    pose graph: %d frames; %d odometry links, %d across dropouts (weak), %d across other "
        "breaks (speed only); the start anchor; %d board fixes" % (n, n_odo, n_gap, n_free, n_b))

    # the start: the odometry from the anchor
    T_init = np.einsum("ij,njk->nik", T_anchor @ np.linalg.inv(T_odom[k_anchor]), T_odom)
    X = np.column_stack([T_init[:, 0, 3], T_init[:, 1, 3], yaw_of(T_init)])
    fixed = list(G.abs)
    X = G.solve(X, c, log)
    history = []

    def report(stage, X_prev, n_map):
        dmove = np.linalg.norm(X[:, :2] - X_prev[:, :2], axis=1)
        bres = [np.linalg.norm(X[k, :2] - m[:2]) * 100 for k, m, _, _, kind in G.abs if kind == "board"]
        log("    %-34s %5d map fixes; moved median %.2f m, max %.2f m; boards: median %.1f cm, "
            "max %.1f cm  (%.0f s)" % (stage, n_map, np.median(dmove), dmove.max(),
                                      np.median(bres) if bres else float("nan"),
                                      max(bres) if bres else float("nan"), time.time() - t_start))
        history.append({"round": len(history) + 1, "stage": stage, "registered": n_map,
                        "frames": n, "not_placed": 0, "median_corr_cm": float(np.median(dmove) * 100),
                        "p95_corr_cm": float(np.percentile(dmove, 95) * 100),
                        "median_residual_cm": None})

    report("odometry + anchor + boards", np.column_stack([T_init[:, 0, 3], T_init[:, 1, 3],
                                                          yaw_of(T_init)]), 0)
    # map: search around the estimate
    search = []
    every = float(c["pgo_search_every_s"])
    for rnd, radius in enumerate(c["pgo_search_m"]):
        search = []
        t_next = -np.inf
        seg_found = set()
        for j in range(n):
            if times[j] < t_next or piece[j] < 0:
                continue
            if rnd == 0 and free[j] and int(seg[j]) in seg_found:
                continue
            t_next = times[j] + every
            T_j = lift(X, j)
            inv0 = np.linalg.inv(T_odom[j])
            stack, k = [], j
            while k < n and times[k] - times[j] <= float(c["pgo_stack_s"]) and piece[k] == piece[j]:
                M = inv0 @ T_odom[k]
                stack.append(scans[k] @ M[:3, :3].T + M[:3, 3])
                k += 1
            wide = rnd == 0 and free[j]              # after a reset: anywhere near, any heading
            T, fit, rival, _ = refind(A, ref, np.vstack(stack), T_j,
                                      float(c["pgo_free_search_m"]) if wide else float(radius),
                                      180.0 if wide else float(c["pgo_search_yaw_deg"]), c)
            if T is not None:
                search.append((j, np.r_[T[0, 3], T[1, 3], yaw_of(T)]))
                if wide:
                    seg_found.add(int(seg[j]))
                    log("    after the reset at %.1f s: found at %.1f s (%.1f, %.1f), fit %.2f"
                        % (times[np.flatnonzero(seg == seg[j])[0]] - times[0], times[j] - times[0],
                           T[0, 3], T[1, 3], fit))
        G.abs = fixed + [(j, m, W_diag(float(c["search_sigma_m"]), float(c["search_sigma_deg"])), True,
                          "search") for j, m in search]
        X_prev = X.copy()
        X = G.solve(X, c, log)
        report("map search within %.1f m" % float(radius), X_prev, len(search))

    # map: ICP from the estimate, information from the Hessian
    ci = dict(c, max_shift=float(c["pgo_icp_shift"]))
    icp_T = [None] * n
    sig2 = float(c["map_sigma_m"]) ** 2
    for it in range(int(c["pgo_icp_iters"])):
        facs = []
        icp_T = [None] * n
        for j in range(n):
            T0 = lift(X, j)
            T, status, row, share = step(A, ref, scans[j], T0, ci)
            if status != "ok" or share < float(c["pgo_icp_min_fit"]):
                continue
            P = scans[j] @ T[:3, :3].T + T[:3, 3]
            dd, ii = ref.tree.query(P, distance_upper_bound=0.1, workers=-1)
            ok = np.isfinite(dd)
            if ok.sum() < int(c["min_corr"]):
                continue
            nrm = ref.nrm[ii[ok]]
            q = P[ok] - T[:3, 3]
            Jr = np.column_stack([nrm[:, 0], nrm[:, 1], q[:, 0] * nrm[:, 1] - q[:, 1] * nrm[:, 0]])
            H = Jr.T @ Jr / len(Jr) / (sig2 / 3.0)
            ev, V = np.linalg.eigh(H)
            W = (V * np.sqrt(np.clip(ev, 1e-6, None))).T          # W^T W = H
            facs.append((j, np.r_[T[0, 3], T[1, 3], yaw_of(T)], W, True, "map"))
            icp_T[j] = (T, row)
        # height where the map registers it
        okz = [j for j in range(n) if icp_T[j] is not None]
        if okz:
            z[:] = np.interp(times, times[okz], [icp_T[j][0][2, 3] for j in okz])
        G.abs = fixed + facs
        X_prev = X.copy()
        X = G.solve(X, c, log)
        report("map ICP pass %d" % (it + 1), X_prev, len(facs))

    # out: the map's 6-DoF pose where the frame registered there (and agrees), else lifted
    T_out = np.zeros((n, 4, 4))
    rows = []
    for j in range(n):
        Tl = lift(X, j)
        if icp_T[j] is not None and np.linalg.norm(icp_T[j][0][:2, 3] - Tl[:2, 3]) <= 0.1:
            T_out[j], row = icp_T[j][0], icp_T[j][1]
            rows.append(row)
        else:
            T_out[j] = Tl
            rows.append(("pgo", 0, np.nan, 0.0, 0.0))
    rr, ra = G.residuals(X)
    kinds = [f[4] for f in G.abs]
    for kind in ("anchor", "board", "map"):
        v = ra[[i for i, kd in enumerate(kinds) if kd == kind]]
        if len(v):
            log("    residuals, %-6s %5d: median %.2f sigma, p95 %.2f, over 3 sigma %d"
                % (kind, len(v), np.median(v), np.percentile(v, 95), int((v > 3).sum())))
    gap_r = [np.linalg.norm(G._e_rel(X, i, j, m)[:2]) for i, j, m, W, rb in G.rel if rb]
    if gap_r:
        big = sorted(((np.linalg.norm(G._e_rel(X, i, j, m)[:2]), times[i] - times[0], times[j] - times[i])
                      for i, j, m, W, rb in G.rel if rb), reverse=True)[:5]
        log("    across breaks, where the graph and the odometry disagree most: "
            + "; ".join("%.2f m at %.1f s (%.1f s)" % b for b in big))
    t_cov = time.time()
    cov = G.marginals(X, c)
    major = sigma_major(cov)
    log("    uncertainty (marginal covariance, %.0f s): sigma xy major axis median %.1f cm, p95 %.1f cm, "
        "max %.1f cm; yaw median %.2f deg" % (time.time() - t_cov, 100 * np.median(major),
                                               100 * np.percentile(major, 95), 100 * major.max(),
                                               np.degrees(np.median(np.sqrt(cov[:, 2, 2])))))
    log("    least held (sigma xy major axis, 5 s apart): " + "; ".join(
        "%.0f cm at %.1f s" % (100 * major[k], times[k] - times[0]) for k in peaks(major, times, 5.0, 5)))
    return T_out, rows, history, cov


def sigma_major(cov):
    """The xy ellipse's major semi-axis (m) per pose."""
    a, b, d = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    return np.sqrt(np.maximum(0.5 * (a + d) + np.sqrt(0.25 * (a - d) ** 2 + b ** 2), 0.0))


def peaks(v, times, sep_s, k):
    """Indices of the k largest values at least sep_s apart."""
    out = []
    for i in np.argsort(-v):
        if all(abs(times[i] - times[j]) >= sep_s for j in out):
            out.append(int(i))
            if len(out) == k:
                break
    return out

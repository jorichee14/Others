"""Frozen-map reference and scan-to-map registration.

The map is loaded once into a `Reference` (voxel-centroid downsample plus a
local plane per cell, matched by voxel membership) and every range sensor -
lidar scan, depth frame, depth submap - is registered to it by the same
`icp_frame`. `Grid2D` is the 2D likelihood field used by grid localisers.

Extracted from stage 08 unchanged: the comments record what was measured on
the mapping sessions and why each choice is what it is.
"""
import math
import numpy as np
from scipy import ndimage

from .se3 import Rt, apply, exp_r, log_R, subsample


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


class Grid2D:
    """Wall likelihood field from the anchored map: points in a height band
    rasterised at `res`, distance transform, Gaussian likelihood."""
    def __init__(self, P, slice_z, res=0.05, sigma=0.10, max_d=1.0, pad=2.0):
        m = (P[:, 2] >= slice_z[0]) & (P[:, 2] <= slice_z[1])
        Q = P[m, :2]
        self.res = res
        self.x0 = Q[:, 0].min() - pad; self.y0 = Q[:, 1].min() - pad
        nx = int((Q[:, 0].max() + pad - self.x0) / res) + 1
        ny = int((Q[:, 1].max() + pad - self.y0) / res) + 1
        occ = np.zeros((ny, nx), bool)
        ix = ((Q[:, 0] - self.x0) / res).astype(int)
        iy = ((Q[:, 1] - self.y0) / res).astype(int)
        occ[iy, ix] = True
        self.dist = np.minimum(ndimage.distance_transform_edt(~occ) * res, max_d)
        self.sigma = sigma; self.max_d = max_d
        self.ny, self.nx = occ.shape
        # candidate cells for random injection: near a wall but not in it
        free = (self.dist > 0.25) & (self.dist < 2.0)
        self.free_iy, self.free_ix = np.nonzero(free)
        print("  likelihood field: %d x %d cells at %.2f m, %d wall cells, "
              "%d candidate free cells (z band %s)"
              % (nx, ny, res, int(occ.sum()), len(self.free_ix), list(slice_z)))

    def lookup(self, x, y):
        ix = np.clip(((x - self.x0) / self.res).astype(int), 0, self.nx - 1)
        iy = np.clip(((y - self.y0) / self.res).astype(int), 0, self.ny - 1)
        d = self.dist[iy, ix]
        out = (x < self.x0) | (x >= self.x0 + self.nx * self.res) | \
            (y < self.y0) | (y >= self.y0 + self.ny * self.res)
        return np.where(out, self.max_d, d)

    def random_poses(self, n, rng):
        j = rng.integers(0, len(self.free_ix), n)
        x = self.x0 + (self.free_ix[j] + rng.random(n)) * self.res
        y = self.y0 + (self.free_iy[j] + rng.random(n)) * self.res
        return np.column_stack([x, y, rng.uniform(-math.pi, math.pi, n)])



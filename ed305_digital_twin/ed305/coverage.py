"""Which cameras see which part of the room.

The target volume (room YAML `target:`) is cut into voxels. A camera sees a
voxel centre when it lands in the image, is close enough to be resolved
(`min_px_per_m`), and the line from the camera to it misses every static
obstacle box. Coverage of a voxel = how many cameras see it; a voxel is a
blind spot when that count is below k.
"""
from dataclasses import dataclass

import numpy as np
import yaml


@dataclass
class Room:
    x: tuple
    y: tuple
    height: float
    obstacles: list          # [(name, min xyz, max xyz)]
    z: np.ndarray            # target heights
    xy_step: float
    min_px_per_m: float
    margin_px: int
    k: int

    def grid(self):
        s = self.xy_step
        xs = np.arange(self.x[0] + s / 2, self.x[1], s)
        ys = np.arange(self.y[0] + s / 2, self.y[1], s)
        X, Y, Z = np.meshgrid(xs, ys, self.z, indexing="ij")
        return xs, ys, np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)


def load_room(path):
    with open(path) as f:
        doc = yaml.safe_load(f)
    r, t = doc["room"], doc.get("target", {})
    z0, z1 = t.get("z", [0.0, 2.0])
    zs = t.get("z_step", 0.25)
    obstacles = [(o.get("name", f"box_{i}"), np.asarray(o["min"], float), np.asarray(o["max"], float))
                 for i, o in enumerate(doc.get("obstacles", []) or [])]
    return Room(
        x=tuple(r["x"]), y=tuple(r["y"]), height=float(r["height"]),
        obstacles=obstacles,
        z=np.arange(z0, z1 + 1e-9, zs) if zs > 0 else np.array([z0]),
        xy_step=float(t.get("xy_step", 0.1)),
        min_px_per_m=float(t.get("min_px_per_m", 0.0)),
        margin_px=int(t.get("margin_px", 0)),
        k=int(t.get("k", 2)),
    )


def inside_box(pts, lo, hi):
    """Strictly inside: a point on the surface (a table top) stays visible."""
    return np.all((pts > lo) & (pts < hi), axis=1)


def segment_hits_box(a, pts, lo, hi, eps=1e-9):
    """Does the segment a -> pts[i] pass through the box (slab test)?

    The segment's own end is left out (t < 1 - eps) so a point on the box's
    surface is not hidden by the box it sits on.
    """
    d = pts - a
    par = d == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - a) / d
        t2 = (hi - a) / d
    tmin, tmax = np.minimum(t1, t2), np.maximum(t1, t2)
    # a zero direction component: inside the slab means no constraint, outside means a miss
    out = par & ((a < lo) | (a > hi))
    tmin = np.where(par, -np.inf, tmin)
    tmax = np.where(par, np.inf, tmax)
    t_in = np.max(tmin, axis=1)
    t_out = np.min(tmax, axis=1)
    return (~np.any(out, axis=1)) & (t_in <= t_out) & (t_out > eps) & (t_in < 1 - eps)


def obstacle_mask(room, pts):
    blocked = np.zeros(len(pts), bool)
    for _, lo, hi in room.obstacles:
        blocked |= inside_box(pts, lo, hi)
    return blocked


def camera_visibility(cam, room, pts, blocked):
    """(n_pts,) bool: `cam` sees each point."""
    ok, z = cam.in_view(pts, margin_px=room.margin_px)
    if room.min_px_per_m > 0:
        ok &= cam.K[0, 0] / np.maximum(z, 1e-9) >= room.min_px_per_m
    ok &= ~blocked
    for _, lo, hi in room.obstacles:
        idx = np.flatnonzero(ok)
        if idx.size:
            ok[idx[segment_hits_box(cam.center, pts[idx], lo, hi)]] = False
    return ok


def visibility(cameras, room, pts):
    """(n_cams, n_pts) bool: camera i sees point j."""
    blocked = obstacle_mask(room, pts)
    vis = np.zeros((len(cameras), len(pts)), bool)
    for i, cam in enumerate(cameras):
        vis[i] = camera_visibility(cam, room, pts, blocked)
    return vis, blocked


def covered(vis, k, cams=None):
    sub = vis if cams is None else vis[list(cams)]
    return sub.sum(0) >= k


def leave_one_out(vis, k, valid):
    """Covered fraction of the valid voxels with each camera removed."""
    full = vis.sum(0)
    return [float(np.mean((full - vis[i])[valid] >= k)) for i in range(len(vis))]


def critical(vis, k, valid):
    """Per camera: valid voxels that drop below k without it."""
    n = vis.sum(0)
    return [int(np.sum(vis[i] & (n == k) & valid)) for i in range(len(vis))]


def greedy_subset(vis, k, valid):
    """Fewest cameras (greedy) that cover every voxel the full set covers at k.

    Gain of a camera = covered-voxel deficit it fills: for each target voxel
    the deficit is max(0, k - count so far).
    """
    target = covered(vis, k) & valid
    need = np.where(target, k, 0)
    count = np.zeros(vis.shape[1], int)
    chosen = []
    while True:
        deficit = np.maximum(need - count, 0)
        if not deficit.any():
            return chosen
        gains = [(-1 if i in chosen else int(np.sum(vis[i] & (deficit > 0)))) for i in range(len(vis))]
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            return chosen
        chosen.append(best)
        count += vis[best]

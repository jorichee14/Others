"""Whether the geometry a sensor returned actually constrains the pose.

This is the instrument the sensor-constrained track is built on, and it is not
ATE. ATE tells you a method failed; it does not tell you whether the sensor
could have succeeded. A 87-degree depth camera facing a flat wall observes
nothing about motion parallel to that wall — no estimator can recover it, and a
benchmark that only reports the resulting drift has measured the method for a
property of the scene.

The quantity is the point-to-plane normal equation's information matrix, the
same H that a scan-to-map ICP solves against (and the same one stage 01a builds
in `_solve`). For a set of surface points p with unit normals n, observed from
sensor origin c:

    A_i = [ (p_i - c) x n_i , n_i ]        H = sum_i w_i A_i^T A_i

The translation block sum(n n^T) has an eigenvalue near zero exactly when no
observed surface has a normal component along that direction — the corridor
case, and the flat-wall case. The eigenvector names the direction, which is
what makes this a diagnosis rather than a score.

Computed from ONE SCAN and its own normals: no map, no reference, no estimator.
So it is available for every ablation cell before any SLAM system is run, and a
method that fails where observability has collapsed is not evidence about the
method.
"""
from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["Observability", "auto_voxel", "estimate_normals", "observability"]


@dataclasses.dataclass
class Observability:
    n_points: int                 # surfaces with a usable normal (NOT raw points)
    status: str                   # "ok" | "starved"
    trans_eigenvalues: np.ndarray  # (3,) ascending, normalised to [0, 1]
    trans_directions: np.ndarray   # (3,3) columns, matching eigenvectors
    rot_eigenvalues: np.ndarray    # (3,) ascending, normalised
    rot_directions: np.ndarray
    normal_isotropy: float         # 0 = one surface orientation, 1 = all
    voxel_m: float = 0.0           # the size the normals were estimated at

    @property
    def trans_weakest(self) -> float:
        """Smallest normalised translation eigenvalue.

        1/3 is isotropic (a corner: every direction equally constrained). Below
        ~0.05 one translation direction is effectively free, and the estimate
        along it comes from the prior or from nothing.
        """
        return float(self.trans_eigenvalues[0])

    @property
    def trans_weakest_direction(self) -> np.ndarray:
        return self.trans_directions[:, 0]

    @property
    def trans_condition(self) -> float:
        lo = max(float(self.trans_eigenvalues[0]), 1e-12)
        return float(self.trans_eigenvalues[2] / lo)

    def degenerate(self, threshold: float = 0.05) -> bool:
        """True when a translation direction is effectively unconstrained.

        False when `status` is "starved": too few surfaces were recovered to
        judge, which is NOT the same claim. Conflating them makes every sparse
        cloud read as degenerate geometry, which would have made the density
        axis of the constrained track measure the normal estimator instead of
        the sensor.
        """
        return self.status == "ok" and self.trans_weakest < threshold

    def starved(self) -> bool:
        return self.status == "starved"

    def as_dict(self) -> dict:
        return {
            "n_points": self.n_points, "status": self.status,
            "trans_eigenvalues": self.trans_eigenvalues.tolist(),
            "trans_weakest": self.trans_weakest,
            "trans_weakest_direction": self.trans_weakest_direction.tolist(),
            "trans_condition": self.trans_condition,
            "rot_eigenvalues": self.rot_eigenvalues.tolist(),
            "rot_weakest": float(self.rot_eigenvalues[0]),
            "normal_isotropy": self.normal_isotropy, "voxel_m": self.voxel_m,
        }


def auto_voxel(points: np.ndarray, target_per_voxel: int = 25,
               start_m: float = 0.25, bounds=(0.05, 4.0), iters: int = 3) -> float:
    """A voxel size giving roughly `target_per_voxel` points per occupied cell.

    Without this the density axis of the sensor-constrained track measures the
    NORMAL ESTIMATOR: at a fixed voxel size a 10x sparser cloud has 10x fewer
    points per cell, most cells fall below the PCA minimum, and the result reads
    as degenerate geometry when the geometry has not changed at all. Points lie
    on surfaces, so occupancy scales with the square of the voxel edge, which is
    the exponent used here.

    The size itself is then part of the result: two cells compared at different
    voxel sizes are asking slightly different questions, and the sweep reports
    the size it used per cell rather than hiding it.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    v = float(start_m)
    if len(pts) < 4:
        return v
    for _ in range(iters):
        key = np.floor(pts / v).astype(np.int64)
        n_occ = len(np.unique((key[:, 0] << 42) ^ (key[:, 1] << 21) ^ key[:, 2]))
        per = len(pts) / max(n_occ, 1)
        if abs(per - target_per_voxel) / target_per_voxel < 0.25:
            break
        v = float(np.clip(v * np.sqrt(target_per_voxel / max(per, 1e-9)), *bounds))
    return v


def estimate_normals(points: np.ndarray, voxel_m: float = 0.25,
                     min_points: int = 12, planarity: float = 0.3,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """One normal per occupied voxel, by PCA over the points in it.

    Returns (centroids, normals) for voxels whose points are planar enough to
    define one. Deliberately per-voxel rather than per-point k-NN: the quantity
    below is a sum over SURFACES, and a k-NN normal field weights each surface
    by how many points happened to land on it, so a near wall outvotes the
    whole rest of the room and the result becomes a density statistic.

    `planarity` is the ratio of the smallest to the middle singular value; a
    voxel above it is a corner or clutter and contributes no normal.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < min_points:
        return np.zeros((0, 3)), np.zeros((0, 3))
    key = np.floor(points / float(voxel_m)).astype(np.int64)
    packed = (key[:, 0] << 42) ^ (key[:, 1] << 21) ^ key[:, 2]
    order = np.argsort(packed, kind="stable")
    ks, ps = packed[order], points[order]
    first = np.concatenate(([True], ks[1:] != ks[:-1]))
    start = np.flatnonzero(first)
    end = np.concatenate((start[1:], [len(ks)]))

    cen, nrm = [], []
    for a, b in zip(start, end):
        if b - a < min_points:
            continue
        P = ps[a:b]
        c = P.mean(axis=0)
        w, V = np.linalg.eigh((P - c).T @ (P - c) / (b - a))
        if w[1] <= 1e-12:
            continue
        if np.sqrt(max(w[0], 0.0)) / np.sqrt(w[1]) > planarity:
            continue                                  # not a plane
        cen.append(c)
        nrm.append(V[:, 0])
    if not cen:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.array(cen), np.array(nrm)


def observability(points: np.ndarray, origin=(0.0, 0.0, 0.0),
                  voxel_m: float | str = 0.25, min_points: int = 12,
                  planarity: float = 0.3, min_surfaces: int = 12,
                  normals: np.ndarray | None = None,
                  centroids: np.ndarray | None = None) -> Observability:
    """Point-to-plane observability of one scan, in the frame the points are in.

    `origin` is the sensor position in that same frame; it only affects the
    rotation block, where it is the lever arm. Passing the wrong origin (world
    zero for a scan expressed in world) inflates the rotation block by the
    distance to the origin and makes every frame look well-constrained in
    rotation.

    Each surface contributes ONCE, with unit weight, regardless of how many
    points landed on it — see `estimate_normals`.
    """
    if normals is None or centroids is None:
        if voxel_m == "auto":
            voxel_m = auto_voxel(points, target_per_voxel=2 * min_points)
        centroids, normals = estimate_normals(points, float(voxel_m), min_points,
                                              planarity)
    n = len(normals)
    if n < max(3, min_surfaces):
        # Not "the geometry is degenerate" — "there is not enough of it to say".
        # A voxel-PCA normal field needs a minimum point density per voxel, so a
        # heavily subsampled cloud starves the estimator long before the SCENE
        # stops constraining the pose.
        z = np.zeros(3)
        return Observability(n, "starved", z, np.eye(3), z, np.eye(3), 0.0,
                             float(voxel_m) if voxel_m != "auto" else 0.0)

    Htt = normals.T @ normals                                  # sum n n^T
    lev = np.cross(centroids - np.asarray(origin, dtype=np.float64), normals)
    Hrr = lev.T @ lev

    wt, Vt = np.linalg.eigh(Htt / n)
    scale = float(np.trace(Hrr)) / 3.0
    wr, Vr = np.linalg.eigh(Hrr / max(scale * 3.0, 1e-12))

    # Spread of surface ORIENTATION, as the normalised entropy of the
    # translation spectrum: 0 when every observed surface faces the same way,
    # 1 when the three axes are equally constrained. Reported beside the
    # eigenvalues because it uses all three, where `trans_weakest` uses only
    # the worst -- a corridor (two axes held, one free) and a single wall (one
    # axis held, two free) have similar weakest eigenvalues and very different
    # isotropy, and they need different fixes.
    pw = np.maximum(wt, 0.0)
    tot = float(pw.sum())
    if tot <= 1e-12:
        iso = 0.0
    else:
        q = pw / tot
        nz = q[q > 1e-12]
        iso = float(-np.sum(nz * np.log(nz)) / np.log(3.0))

    return Observability(n_points=n, status="ok", trans_eigenvalues=wt,
                         trans_directions=Vt, rot_eigenvalues=wr, rot_directions=Vr,
                         normal_isotropy=min(max(iso, 0.0), 1.0),
                         voxel_m=float(voxel_m))

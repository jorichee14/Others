#!/usr/bin/env python3
"""
Another robot seen in mobile_1's LiDAR: every scan put in map through
mobile_1's track (04: traj_<name>.tum, T_map_lidar), the points the reference
map does not have, in the other robot's height band, clustered by footprint.

  off map   further than off_map_m from the reference map (its 0.05 m cloud)
  band      height in map within [z_lo, z_hi] (the robot's body)
  people    a cell with off-map points above z_hi + tall_m is a person, a
            door, ...: its cluster is dropped
  cluster   cells of cell_m with band points, joined to their 8 neighbours;
            kept when its footprint is between min_size_m and max_size_m and it
            has min_points points

-> [(t, [(x, y, n), ...])] per scan. A detection is a CANDIDATE: a moved
chair or a person's legs pass too, so depth_track only uses them to choose
among places mobile_2's own depth frames fit (and to centre its search).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULTS = dict(points_topic="/mobile_1/ouster/points", lidar_track="mobile_1_lidar",
                range=[0.8, 25.0], off_map_m=0.15, z_lo=None, z_hi=None, tall_m=0.6,
                cell_m=0.1, min_size_m=0.15, max_size_m=0.9, min_points=8, scan_stride=1)


def clusters(P, c):
    """Off-map points in map (N,3) -> [(x, y, n)] robot-sized clusters in the band."""
    from scipy import ndimage
    if not len(P):
        return []
    z_lo, z_hi = float(c["z_lo"]), float(c["z_hi"])
    cell = float(c["cell_m"])
    band = P[(P[:, 2] >= z_lo) & (P[:, 2] <= z_hi)]
    if len(band) < int(c["min_points"]):
        return []
    tall = P[P[:, 2] > z_hi + float(c["tall_m"])]
    lo = band[:, :2].min(0) - 2 * cell
    ij = np.floor((band[:, :2] - lo) / cell).astype(int)
    shape = ij.max(0) + 3
    if shape[0] * shape[1] > 4e7:                         # far-flung points: keep near ones
        return []
    occ = np.zeros(shape, bool)
    occ[ij[:, 0], ij[:, 1]] = True
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    if not n:
        return []
    tall_lab = set()
    if len(tall):
        tij = np.floor((tall[:, :2] - lo) / cell).astype(int)
        ok = (tij >= 0).all(1) & (tij < shape).all(1)
        tall_lab = set(np.unique(lab[tij[ok, 0], tij[ok, 1]]).tolist()) - {0}
    pl = lab[ij[:, 0], ij[:, 1]]
    out = []
    for k in range(1, n + 1):
        m = pl == k
        if m.sum() < int(c["min_points"]):
            continue
        xy = band[m, :2]
        ext = xy.max(0) - xy.min(0)
        size = float(max(ext))
        if size < float(c["min_size_m"]) or size > float(c["max_size_m"]):
            continue
        if k in tall_lab:
            continue
        out.append((float(xy[:, 0].mean()), float(xy[:, 1].mean()), int(m.sum())))
    return out


def detect(P, ref, c, cache, log=print):
    """[(t, [(x, y, n)])] for every scan of mobile_1 (cached in `cache`)."""
    import importlib
    from pipeline_common import iter_scans
    A = importlib.import_module("02_refine_poses")
    track = os.path.join(P.reference_dir(), "traj_%s.tum" % c["lidar_track"])
    bag = os.path.expanduser(P.dataset["bag"])
    key = "%s|%s|%s|%.2f|%.2f|%.2f|%.2f" % (bag, track, c["points_topic"], float(c["z_lo"]),
                                            float(c["z_hi"]), float(c["off_map_m"]),
                                            float(c["range"][1]))
    if os.path.exists(cache):
        d = json.load(open(cache))
        if d.get("key") == key and os.path.getmtime(cache) >= os.path.getmtime(track):
            return [(s["t"], [tuple(x) for x in s["c"]]) for s in d["scans"]]
    if not os.path.exists(track):
        raise SystemExit("no %s: run 04 for mobile_1 first" % track)
    tr_t, tr_T = A.load_traj(track)
    s01 = P.cfg.get("01_build_map", {})
    out, n, t0 = [], 0, time.time()
    log("mobile_1's LiDAR: other robots off the map, %.2f-%.2f m high, within %.0f m ..."
        % (float(c["z_lo"]), float(c["z_hi"]), float(c["range"][1])))
    for j, W in iter_scans(bag, c["points_topic"], tr_t, float(s01.get("time_tol", 0.04)),
                           float(c["range"][0]), float(c["range"][1]), tr_T,
                           bool(s01.get("deskew", True)), int(s01.get("deskew_bins") or 100),
                           frame="world", verbose=False):
        n += 1
        if (n - 1) % int(c["scan_stride"]):
            continue
        sel = (W[:, 2] >= float(c["z_lo"]) - 0.05) & \
            (W[:, 2] <= float(c["z_hi"]) + float(c["tall_m"]) + 1.5)
        W = W[sel]
        d, _ = ref.tree.query(W, workers=-1, distance_upper_bound=float(c["off_map_m"]))
        out.append((float(tr_t[j]), clusters(W[~np.isfinite(d)], c)))
        if n % 1000 == 0:
            log("    %d scans, %d candidate clusters (%.0f s)"
                % (n, sum(len(x[1]) for x in out), time.time() - t0))
    json.dump({"key": key, "scans": [{"t": t, "c": cl} for t, cl in out]}, open(cache, "w"))
    return out


class Sightings:
    """Lookup: candidate (x, y) near a time."""
    def __init__(self, scans):
        self.t = np.array([s[0] for s in scans]) if scans else np.zeros(0)
        self.c = [np.array([[x, y] for x, y, _ in cl]) if cl else np.zeros((0, 2))
                  for _, cl in scans]

    def at(self, t, tol=0.15):
        """Candidates (K,2) of the scan nearest t, within tol s (else empty)."""
        if not len(self.t):
            return np.zeros((0, 2))
        i = int(np.argmin(np.abs(self.t - t)))
        return self.c[i] if abs(self.t[i] - t) <= tol else np.zeros((0, 2))

    def support(self, times, xy, near_m=0.5, tol=0.15):
        """How many of the poses (times, xy) have a candidate within near_m."""
        k = 0
        for t, p in zip(times, xy):
            cc = self.at(t, tol)
            if len(cc) and np.min(np.linalg.norm(cc - p[:2], axis=1)) <= near_m:
                k += 1
        return k

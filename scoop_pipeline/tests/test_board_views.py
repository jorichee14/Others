#!/usr/bin/env python
"""Self-test for map_stages/board_views.py: a run's start from every opening
board sighting, combined through the run's LiDAR odometry.

The run's GLIM world is the map turned 137 deg and moved 6.3 m. The robot
parks 4 s in front of the board, then drives an arc around it for 8 s
(viewing directions over ~60 deg), the board out of view for good after
that, and seen again 300 s later. Each sighting's board pose carries
noise plus an error that depends on the viewpoint (the same at one view,
different at another), the way PnP on a small board errs.

  all views    must recover map <- GLIM to a few mm / tenths of a degree,
               closer than the parked part alone, and keep only the opening
               section (not the late re-sighting);
  few views    fewer than min_views in the opening -> None.

    python scoop_pipeline/tests/test_board_views.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "map_stages"))

from board_views import views_through_odom, cam_at          # noqa: E402
from pipeline_boards import avg_T                           # noqa: E402


def rot(axis, deg):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.radians(deg) * np.asarray(axis, float)).as_matrix()


def Tm(R=np.eye(3), t=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


T0 = 1_790_000_000.0
T_MAP_GLIM = Tm(rot([0, 0, 1], 137.0), (6.3, -2.1, 0.15))
T_MAP_BOARD = Tm(rot([0, 0, 1], 180.0), (0.0, 0.0, 1.0))     # board facing -x
T_LC = Tm(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float), (0.07, 0.0, -0.09))


def lidar_in_map(t):
    """Parked 1.5 m in front of the board for 4 s, then an arc around it."""
    a = 0.0 if t < 4 else np.radians(min(t - 4, 8) * 7.5)       # up to 60 deg
    p = np.array([-1.5 * np.cos(a), 1.5 * np.sin(a), 0.6]) if t < 12 else \
        np.array([-1.5 - (t - 12) * 0.5, 0.0, 0.6])
    yaw = -np.degrees(a)                                       # facing the board
    return Tm(rot([0, 0, 1], yaw), p)


def sightings(ts, rng):
    """A viewpoint's error repeats at that viewpoint: one draw per 7.5 deg of
    the arc (the parked 4 s are ONE viewpoint), plus per-frame noise."""
    biases = {}
    hits = []
    for t in ts:
        T_map_cam = lidar_in_map(t - T0) @ T_LC
        T_board_cam = np.linalg.inv(T_MAP_BOARD) @ T_map_cam
        view = int(max(t - T0 - 4, -1) // 1) if t - T0 < 12 else 100
        if view not in biases:
            biases[view] = (rot(rng.normal(size=3), rng.normal(0, 1.2)),
                            rng.normal(0, 0.008, 3))
        bias_r, bias_t = biases[view]
        E = Tm(bias_r @ rot(rng.normal(size=3), rng.normal(0, 0.3)),
               bias_t + rng.normal(0, 0.002, 3))
        Tb = T_board_cam @ E
        hits.append({"t": t, "T_board_cam": Tb, "T_map_cam": T_MAP_BOARD @ Tb,
                     "n": 40, "reproj": 0.4})
    return hits


def main():
    rng = np.random.default_rng(2)
    tr_t = T0 + np.arange(0.5, 400.0, 0.1)
    tr_T = np.array([np.linalg.inv(T_MAP_GLIM) @ lidar_in_map(t - T0) for t in tr_t])
    ts = T0 + np.concatenate([np.arange(0.2, 12.0, 0.1),          # opening (some before traj)
                              np.arange(312.0, 314.0, 0.1)])      # late re-sighting
    hits = sightings(ts, rng)
    failed = []

    r = views_through_odom(hits, tr_t, tr_T, T_LC)
    T_cam = cam_at(r, tr_t, tr_T, T_LC, r["t_first"])
    true_cam = lidar_in_map(r["t_first"] - T0) @ T_LC
    cmm = np.linalg.norm(T_cam[:3, 3] - true_cam[:3, 3]) * 1000
    ddeg = np.degrees(np.arccos(np.clip((np.trace(T_cam[:3, :3].T @ true_cam[:3, :3]) - 1) / 2,
                                        -1, 1)))
    print(f"all views: {r['n_views']} sightings, view span {r['view_angle_deg']:.0f} deg, "
          f"{r['n_before_traj']} before the trajectory, {r['n_sections']} sections; "
          f"camera at the first view off by {cmm:.1f} mm / {ddeg:.3f} deg; "
          f"scatter std {r['std_mm']:.1f} mm")

    parked = [h for h in hits if h["t"] - T0 < 4 and h["t"] >= tr_t[0]]
    Td, _ = avg_T([h["T_map_cam"] for h in parked])
    true_p = lidar_in_map(np.mean([h["t"] for h in parked]) - T0) @ T_LC
    pmm = np.linalg.norm(Td[:3, 3] - true_p[:3, 3]) * 1000
    pdeg = np.degrees(np.arccos(np.clip((np.trace(Td[:3, :3].T @ true_p[:3, :3]) - 1) / 2,
                                        -1, 1)))
    print(f"the parked part alone off by {pmm:.1f} mm / {pdeg:.2f} deg")
    if not (cmm < 6 and ddeg < 0.6 and r["std_mm"] < 50):
        failed.append(f"all views: {cmm:.1f} mm / {ddeg:.3f} deg, std {r['std_mm']:.1f} mm")
    if not (r["t_last"] - T0 < 20 and r["n_before_traj"] == 3):
        failed.append(f"section: last {r['t_last'] - T0:.1f} s, before {r['n_before_traj']}")
    if not (cmm < pmm and ddeg < pdeg):
        failed.append("the varied views were not better than the parked part")

    few = views_through_odom(hits[3:6], tr_t, tr_T, T_LC)
    if few is not None:
        failed.append("3 sightings were accepted")

    print("\n".join(failed) or "board views ok")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

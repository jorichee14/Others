#!/usr/bin/env python3
"""
A run's start in map from EVERY board sighting of its opening, moving or not.

Each sighting i of a board whose pose in map is known (stage 03) gives the
camera in map, T_map_cam_i = T_map_board @ inv(T_cam_board_i). The run's own
LiDAR odometry (its GLIM trajectory, mm-accurate over seconds) says where the
camera was in the GLIM world at the same instant, T_glim_cam(t_i) =
T_glim_lidar(t_i) @ T_lidar_cam. So every sighting is one estimate of the one
rigid transform between the two worlds,

    T_map_glim_i = T_map_cam_i @ inv(T_glim_cam(t_i)),

and the robust mean over the opening section is the start. It is averaged
as each view's estimate of the camera at the first view (T_map_glim_i @
T_glim_cam(t_first)), so the scatter is in millimetres at the board. Moving between
views is the point: a single viewpoint has PnP errors that repeat frame after
frame (depth along the view, the board's tilt), views from other angles have
other errors, and the mean over them cancels what averaging one view cannot.
The scatter of the per-view estimates is the check.

Used by 06_init_from_boards.py ("source": "lidar_odom" on a camera sensor).
"""
import numpy as np

from pipeline_common import traj_quats, interp_poses
from pipeline_boards import avg_T, spread, sectionize


def _pose(tr_t, tr_T, tr_q, t):
    R, p = interp_poses(tr_t, tr_T, tr_q, np.array([t], float))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R[0], p[0]
    return T


def _ang(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def views_through_odom(hits, tr_t, tr_T, T_lidar_cam, section_gap=3.0, min_views=5):
    """hits: 06's board sightings ({t, T_map_cam, T_board_cam, n, reproj}).
    -> dict with T_map_glim and the numbers that say whether to trust it,
    or None when the opening has fewer than min_views sightings with a pose."""
    tr_q = traj_quats(tr_T)
    T_lc = np.asarray(T_lidar_cam, float)
    hits = sorted(hits, key=lambda h: h["t"])
    inside = [h for h in hits if tr_t[0] <= h["t"] <= tr_t[-1]]
    before = sum(1 for h in hits if h["t"] < tr_t[0])
    secs = sectionize([h["t"] for h in inside], section_gap)
    sec = next((s for s in secs if len(s) >= min_views), None)
    if sec is None:
        return None
    use = [inside[i] for i in sec]
    T_gc = [_pose(tr_t, tr_T, tr_q, h["t"]) @ T_lc for h in use]
    # each view's estimate of the camera at the FIRST view (carried there by
    # the odometry): averaged and scattered where it is used, at the board --
    # not at the GLIM origin metres away, where a rotation error turns into
    # a lever-arm translation that says nothing about the start
    T_g0 = T_gc[0]
    ests = [h["T_map_cam"] @ np.linalg.inv(Tg) @ T_g0 for h, Tg in zip(use, T_gc)]
    w = [h["n"] / max(h["reproj"], 0.05) ** 2 for h in use]
    T_cam0, n_used = avg_T(ests, weights=w)
    sp = spread(ests, T_cam0)
    res_mm = np.array([np.linalg.norm(E[:3, 3] - T_cam0[:3, 3]) * 1000 for E in ests])
    res_deg = np.array([_ang(T_cam0[:3, :3].T @ E[:3, :3]) for E in ests])
    T = T_cam0 @ np.linalg.inv(T_g0)                 # map <- GLIM world
    # how varied the views were: camera positions in the board's frame
    pb = np.array([h["T_board_cam"][:3, 3] for h in use])
    dirs = pb / np.linalg.norm(pb, axis=1, keepdims=True)
    view_deg = float(np.degrees(np.arccos(np.clip((dirs @ dirs.T).min(), -1.0, 1.0))))
    rng = np.linalg.norm(pb, axis=1)
    moved = float(np.linalg.norm(pb.max(0) - pb.min(0)))
    return {"T_map_glim": T, "T_map_cam0": T_cam0, "n_views": len(use), "n_used": int(n_used),
            "n_before_traj": int(before), "n_sections": len(secs),
            "t_first": float(use[0]["t"]), "t_last": float(use[-1]["t"]),
            "std_mm": sp[0], "max_mm": sp[1], "max_deg": sp[2],
            "median_res_mm": float(np.median(res_mm)), "median_res_deg": float(np.median(res_deg)),
            "view_angle_deg": view_deg, "range_m": (float(rng.min()), float(rng.max())),
            "camera_moved_m": moved, "tr_q": tr_q}


def cam_at(r, tr_t, tr_T, T_lidar_cam, t):
    """The camera in map at t through the combined start: T_map_glim @ T_glim_cam(t)."""
    return r["T_map_glim"] @ _pose(tr_t, tr_T, r["tr_q"], t) @ np.asarray(T_lidar_cam, float)

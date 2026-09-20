#!/usr/bin/env python3
"""Self-tests for the evaluation layer. No bag, no ROS, no GPU, no network.

Run before trusting a single number out of this harness:

    python3 scripts/test_slambench.py

The tests that matter most are the ones that check a WRONG configuration is
caught: an unapplied extrinsic, a sim3 scale hidden under an se3 label, a
reference on a different clock, a map scored against an empty volume.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slambench import (Ablation, Trajectory, ablate, absolute_check,
                       agent_separation, ate, completeness_gain, crop_box, crop_fov,
                       decimate_beams, evaluate_map, extrinsic_matrix, fit, load_cloud,
                       load_tum, nearest_distances, auto_voxel, observability, observation_residual,
                       place_overlap, predict_observation, relative_trajectory_error,
                       revisit_residual, rpe, save_ply, save_tum, select_rate,
                       subsample, to_reference_frame, truncate_range, voxel_downsample)
from slambench import se3

PASS, FAIL = [], []


def test(fn):
    try:
        fn()
        PASS.append(fn.__name__)
    except Exception as e:                                       # noqa: BLE001
        FAIL.append((fn.__name__, e, traceback.format_exc()))
    return fn


def circle_traj(n=400, radius=3.0, t0=0.0, rate=10.0, name="circle") -> Trajectory:
    """A horizontal circle with the heading tangent to it — a pushcart loop."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    stamps = t0 + np.arange(n) / rate
    poses = np.zeros((n, 4, 4))
    for i, a in enumerate(th):
        T = se3.pose_from_rpy(radius * np.cos(a), radius * np.sin(a), 0.0,
                              0.0, 0.0, np.degrees(a) + 90.0)
        poses[i] = T
    return Trajectory(stamps, poses, name, frame="base", world="map")


# ----------------------------------------------------------------- se3 algebra
@test
def test_quat_roundtrip_preserves_rotation():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.normal(size=4)
        R = se3.quat_to_rot(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12), "not orthonormal"
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12), "not a rotation"
        assert np.allclose(se3.quat_to_rot(se3.rot_to_quat(R)), R, atol=1e-10)


@test
def test_rpy_roundtrip_and_convention():
    for rpy in [(0, 0, 0), (10, -20, 30), (0, 89.5267, -87.6614), (179, 0, -179)]:
        R = se3.rpy_to_rot(*rpy)
        back = se3.rot_to_rpy(R)
        assert np.allclose(se3.rpy_to_rot(*back), R, atol=1e-9), f"{rpy} -> {back}"
    # yaw of +90 deg must take +x to +y, not to -y
    assert np.allclose(se3.rpy_to_rot(0, 0, 90) @ [1, 0, 0], [0, 1, 0], atol=1e-12)


@test
def test_invert_is_exact_rigid_inverse():
    T = se3.pose_from_rpy(1.0, -2.0, 0.5, 12.0, -34.0, 56.0)
    assert np.allclose(se3.invert(T) @ T, np.eye(4), atol=1e-12)


# ------------------------------------------------------------------ trajectory
@test
def test_tum_roundtrip():
    tr = circle_traj(50)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.tum"
        save_tum(tr, p)
        back = load_tum(p)
    assert len(back) == len(tr)
    assert np.allclose(back.stamps, tr.stamps, atol=1e-9)
    assert np.allclose(back.poses, tr.poses, atol=1e-6)


@test
def test_tum_rejects_wrong_field_count():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.tum"
        p.write_text("1.0 0 0 0 0 0 0\n")            # 7 fields: KITTI-ish, not TUM
        try:
            load_tum(p)
        except ValueError as e:
            assert "8 fields" in str(e)
        else:
            raise AssertionError("an 7-field line was accepted as TUM")


@test
def test_nonmonotonic_stamps_are_refused():
    try:
        Trajectory([0.0, 2.0, 1.0], np.stack([np.eye(4)] * 3))
    except ValueError as e:
        assert "increasing" in str(e)
    else:
        raise AssertionError("out-of-order stamps were accepted")


@test
def test_interpolation_is_exact_on_its_own_samples():
    tr = circle_traj(100)
    got = tr.interpolate(tr.stamps[10:90])
    assert len(got) == 80
    assert np.allclose(got.poses, tr.poses[10:90], atol=1e-9)


@test
def test_interpolation_drops_queries_inside_a_dropout():
    tr = circle_traj(40)
    keep = np.ones(40, bool); keep[15:25] = False        # a 1 s hole at 10 Hz
    gappy = tr.subset(np.nonzero(keep)[0])
    got = gappy.interpolate(tr.stamps, max_gap_s=0.2)
    assert len(got) == 30, f"expected the 10 held-out stamps to be dropped, got {len(got)}"


@test
def test_path_length_of_a_circle():
    tr = circle_traj(2000, radius=3.0)
    assert abs(tr.path_length() - 2 * np.pi * 3.0) < 0.01


# ------------------------------------------------------------------- alignment
@test
def test_se3_alignment_recovers_a_known_transform():
    ref = circle_traj(200)
    T = se3.pose_from_rpy(5.0, -3.0, 0.7, 0.0, 0.0, 37.0)
    est = ref.transform_left(se3.invert(T))
    al = fit(est, ref, "se3")
    assert np.allclose(al.T, T, atol=1e-9), "se3 alignment did not recover the transform"
    assert al.scale == 1.0


@test
def test_sim3_recovers_scale_and_se3_reports_it_without_applying():
    ref = circle_traj(200)
    scaled = Trajectory(ref.stamps, ref.poses.copy(), "scaled")
    scaled.poses[:, :3, 3] *= 0.80                      # a monocular scale error
    a_sim = fit(scaled, ref, "sim3")
    assert abs(a_sim.scale - 1.25) < 1e-6, a_sim.scale
    a_se3 = fit(scaled, ref, "se3")
    assert a_se3.scale == 1.0
    assert abs(a_se3.scale_observed - 1.25) < 1e-6, "se3 must still report the scale it saw"


@test
def test_yaw_alignment_leaves_z_and_tilt_alone():
    ref = circle_traj(200)
    tilt = se3.pose_from_rpy(0, 0, 0, 5.0, 0, 0)        # 5 deg roll on the estimate
    est = ref.transform_left(tilt)
    a_yaw = fit(est, ref, "yaw")
    assert abs(se3.rot_to_rpy(a_yaw.T[:3, :3])[0]) < 1e-9, "yaw mode rotated in roll"
    a_se3 = fit(est, ref, "se3")
    assert abs(se3.rot_to_rpy(a_se3.T[:3, :3])[0]) > 1.0, "se3 mode should absorb the tilt"


@test
def test_alignment_does_not_mirror_a_planar_trajectory():
    ref = circle_traj(300)                              # exactly planar, z == 0
    T = se3.pose_from_rpy(1.0, 2.0, 0.0, 0, 0, 120.0)
    est = ref.transform_left(se3.invert(T))
    al = fit(est, ref, "se3")
    assert np.linalg.det(al.T[:3, :3]) > 0.999, "reflection guard failed on a planar path"


# ---------------------------------------------------------------------- errors
@test
def test_ate_of_an_identical_trajectory_is_zero():
    tr = circle_traj(200)
    r = ate(tr, tr, "se3", reference_uncertainty_m=0.015)
    assert r.trans.rmse < 1e-9, r.trans.rmse
    assert r.rot.rmse < 1e-7
    assert r.below_reference_uncertainty is True


@test
def test_ate_measures_a_known_lateral_offset():
    ref = circle_traj(200)
    est = Trajectory(ref.stamps, ref.poses.copy())
    est.poses[:, 2, 3] += 0.05                          # 5 cm up, everywhere
    r = ate(est, ref, "none")
    assert abs(r.trans.rmse - 0.05) < 1e-9, r.trans.rmse
    r_al = ate(est, ref, "se3")
    assert r_al.trans.rmse < 1e-9, "a constant offset must be absorbed by se3 alignment"


@test
def test_unapplied_extrinsic_shows_up_as_error_and_the_fix_removes_it():
    """The failure this harness exists to prevent."""
    ref = circle_traj(300)                              # map <- reference frame
    E = extrinsic_matrix({"x": 0.067064, "y": -0.092192, "z": -0.074147,
                          "roll": 2.2959, "pitch": 89.5267, "yaw": -87.6614})
    sensor = ref.transform_right(E)                     # what the method reports
    wrong = ate(sensor, ref, "se3")
    assert wrong.trans.rmse > 0.05, ("a 13 cm lever arm on a rotating platform must "
                                     f"show up; got {wrong.trans.rmse*1e3:.1f} mm")
    fixed = ate(to_reference_frame(sensor, E), ref, "se3")
    assert fixed.trans.rmse < 1e-9, fixed.trans.rmse


@test
def test_rpe_is_blind_to_a_global_transform():
    ref = circle_traj(300)
    est = ref.transform_left(se3.pose_from_rpy(10.0, 4.0, 1.0, 0, 0, 45.0))
    r = rpe(est, ref, 1.0, "m")
    assert r.trans.rmse < 1e-9, "RPE must not see a rigid re-anchoring"
    assert r.n_pairs > 100


@test
def test_rpe_sees_a_scale_error_that_se3_ate_hides():
    ref = circle_traj(400, radius=3.0)
    est = Trajectory(ref.stamps, ref.poses.copy())
    est.poses[:, :3, 3] *= 1.02                         # 2% scale
    r = rpe(est, ref, 1.0, "m")
    assert r.trans.rmse > 0.005, r.trans.rmse


@test
def test_rpe_refuses_a_delta_longer_than_the_trajectory():
    tr = circle_traj(50, radius=0.5)                    # ~3 m of path
    try:
        rpe(tr, tr, 100.0, "m")
    except ValueError as e:
        assert "covers" in str(e)
    else:
        raise AssertionError("an impossible delta was accepted")


@test
def test_association_failure_names_the_clock():
    a = circle_traj(50, t0=0.0)
    b = circle_traj(50, t0=5000.0)                      # a different epoch
    try:
        ate(a, b, "se3")
    except ValueError as e:
        assert "inside it" in str(e) or "clock" in str(e), str(e)
    else:
        raise AssertionError("trajectories on different clocks were compared")


@test
def test_absolute_check_refuses_a_window_without_an_observation():
    """A window alone measures the STANDOFF, not the error.

    The platform parks a metre from a board and looks at it. Comparing its
    position against the board's therefore returns ~1 m however good the
    estimate is, and against a 7 mm survey sigma that reads as "resolved above
    the survey's uncertainty" every time -- a confident number measuring
    nothing. The previous version of this test asserted exactly that residual
    (< 0.35 m on a 3 m circle), so it was pinning the defect in place.
    """
    tr = circle_traj(200, radius=3.0)
    rows = absolute_check(tr, [
        {"name": "no_evidence", "position": list(tr.positions[5]),
         "uncertainty_m": 0.007, "window": [tr.stamps[0], tr.stamps[10]]}])
    assert rows[0]["residual_m"] is None, rows[0]
    assert "REFUSED" in rows[0]["verdict"], rows[0]


@test
def test_absolute_check_scores_an_observed_board():
    """With a board detection the residual is a real absolute error."""
    t = np.arange(10, dtype=float)
    poses = np.tile(np.eye(4), (10, 1, 1))
    poses[:, 0, 3] = 0.72                      # parked 0.72 m from the board
    est = Trajectory(t, poses)
    board = {"name": "anchor", "position": [0.0, 0.0, 0.0],
             "uncertainty_m": 0.007, "window": [0.0, 9.0]}
    obs = [{"stamp": float(k), "p_board_cam": [-0.72, 0.0, 0.0]} for k in range(10)]

    r = absolute_check(est, [dict(board, observations=obs)])[0]
    assert r["residual_m"] < 1e-9 and r["source"] == "observed", r
    assert r["verdict"].startswith("indistinguishable"), r

    biased = poses.copy(); biased[:, 1, 3] = 0.03
    r = absolute_check(Trajectory(t, biased), [dict(board, observations=obs)])[0]
    assert abs(r["residual_m"] - 0.03) < 1e-9, r
    assert r["verdict"].startswith("resolved"), r

    # a surveyed standoff is the weaker substitute and must also work
    r = absolute_check(est, [dict(board, standoff=[0.72, 0.0, 0.0])])[0]
    assert r["residual_m"] < 1e-9 and r["source"] == "standoff", r

    # observations outside the window are not counted
    late = [{"stamp": 99.0, "p_board_cam": [-0.72, 0.0, 0.0]}]
    r = absolute_check(est, [dict(board, observations=late)])[0]
    assert r["n"] == 0 and r["residual_m"] is None, r


@test
def test_absolute_check_handles_a_missing_or_unreachable_window():
    tr = circle_traj(200, radius=3.0)
    rows = absolute_check(tr, [
        {"name": "unset", "position": [0, 0, 0], "uncertainty_m": 0.01,
         "window": None},
        {"name": "absent", "position": [0, 0, 0], "uncertainty_m": 0.01,
         "window": [1e9, 1e9 + 1], "standoff": [0, 0, 0]},
    ])
    assert rows[0]["n"] == 0 and "no dwell window" in rows[0]["verdict"], rows[0]
    assert rows[1]["n"] == 0 and rows[1]["residual_m"] is None, rows[1]


@test
def test_revisit_residual_finds_a_loop_and_is_silent_without_one():
    loop = circle_traj(600, radius=3.0, rate=10.0)      # one lap, 60 s
    two = Trajectory(np.concatenate([loop.stamps, loop.stamps + 60.0]),
                     np.concatenate([loop.poses, loop.poses]))
    r = revisit_residual(two, radius_m=0.3, min_separation_s=20.0)
    assert r["n_revisits"] > 100, r
    line = Trajectory(np.arange(200) / 10.0,
                      np.stack([se3.pose_from_rpy(i * 0.1, 0, 0, 0, 0, 0) for i in range(200)]))
    assert revisit_residual(line, 0.3, 20.0)["n_revisits"] == 0


# ------------------------------------------------------------------ pointcloud
@test
def test_ply_roundtrip_binary():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(5000, 3))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.ply"
        save_ply(pts, p)
        back = load_cloud(p)
    assert back.shape == (5000, 3)
    assert np.allclose(back, pts, atol=1e-6)


@test
def test_ply_ascii_is_read():
    body = "\n".join(f"{i} {i*2} {i*3}" for i in range(4))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.ply"
        p.write_text("ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\n"
                     "property float y\nproperty float z\nend_header\n" + body + "\n")
        got = load_cloud(p)
    assert np.allclose(got[3], [3, 6, 9])


@test
def test_voxel_downsample_is_one_point_per_cell():
    pts = np.repeat(np.array([[0.001, 0.001, 0.001], [0.5, 0.5, 0.5]]), 500, axis=0)
    out = voxel_downsample(pts, 0.05)
    assert len(out) == 2, len(out)


@test
def test_crop_box_keeps_only_the_volume():
    pts = np.array([[0, 0, 0], [5, 0, 0], [-1, 0, 0]], float)
    out = crop_box(pts, [-0.5, -1, -1], [1, 1, 1])
    assert len(out) == 1 and np.allclose(out[0], [0, 0, 0])


@test
def test_nearest_distances_are_exact_within_truncation():
    rng = np.random.default_rng(2)
    target = rng.uniform(-2, 2, size=(3000, 3))
    query = target + rng.normal(scale=0.003, size=target.shape)
    d, tr = nearest_distances(query, target, 0.5)
    brute = np.array([np.min(np.linalg.norm(target - q, axis=1)) for q in query[:200]])
    assert np.allclose(d[:200], brute, atol=1e-12), "voxel-hash NN disagrees with brute force"
    assert tr == 0.0


@test
def test_nearest_distances_report_truncation_instead_of_hiding_it():
    target = np.zeros((10, 3))
    query = np.array([[100.0, 0, 0]] * 5)
    d, tr = nearest_distances(query, target, 0.5)
    assert np.allclose(d, 0.5) and tr == 1.0


# ---------------------------------------------------------------- map metrics
@test
def test_map_metrics_on_a_perfect_reconstruction():
    rng = np.random.default_rng(3)
    ref = rng.uniform(-2, 2, size=(20000, 3))
    r = evaluate_map(ref.copy(), ref, volume={"min": [-2, -2, -2], "max": [2, 2, 2]},
                     resolution_m=0.02, truncate_m=0.5)
    assert r.chamfer_m < 1e-9
    assert r.fscore["0.02"]["f"] > 0.999


@test
def test_map_metrics_penalise_a_missing_half():
    rng = np.random.default_rng(4)
    ref = rng.uniform(-2, 2, size=(20000, 3))
    est = ref[ref[:, 0] > 0]                             # the method mapped half the room
    r = evaluate_map(est, ref, volume={"min": [-2, -2, -2], "max": [2, 2, 2]},
                     resolution_m=0.05, truncate_m=0.5)
    assert r.fscore["0.05"]["precision"] > 0.99, "the half it did map is correct"
    assert r.fscore["0.05"]["recall"] < 0.6, "the half it missed must cost recall"


@test
def test_map_metrics_refuse_an_empty_overlap():
    rng = np.random.default_rng(5)
    ref = rng.uniform(-2, 2, size=(2000, 3))
    est = ref + 1000.0                                   # an unapplied alignment
    try:
        evaluate_map(est, ref, volume={"min": [-2, -2, -2], "max": [2, 2, 2]})
    except ValueError as e:
        assert "alignment" in str(e)
    else:
        raise AssertionError("a map in the wrong frame was scored anyway")


@test
def test_downsampling_removes_the_density_advantage():
    rng = np.random.default_rng(6)
    ref = rng.uniform(-1, 1, size=(8000, 3))
    dense = np.repeat(ref, 8, axis=0) + rng.normal(scale=1e-4, size=(8000 * 8, 3))
    a = evaluate_map(ref.copy(), ref, volume={"min": [-1] * 3, "max": [1] * 3},
                     resolution_m=0.05)
    b = evaluate_map(dense, ref, volume={"min": [-1] * 3, "max": [1] * 3},
                     resolution_m=0.05)
    assert abs(a.fscore["0.05"]["recall"] - b.fscore["0.05"]["recall"]) < 0.02



# =========================================================== observability ====
def room_surfaces(rng, n=40000, walls=4):
    """Points on the floor plus `walls` vertical walls of an 8 x 14 m room."""
    out = [np.column_stack([rng.uniform(-4, 4, n), rng.uniform(-7, 7, n),
                            np.zeros(n)])]
    faces = [(np.full(n, -4.0), rng.uniform(-7, 7, n)),
             (np.full(n, 4.0), rng.uniform(-7, 7, n)),
             (rng.uniform(-4, 4, n), np.full(n, -7.0)),
             (rng.uniform(-4, 4, n), np.full(n, 7.0))]
    for x, y in faces[:walls]:
        out.append(np.column_stack([x, y, rng.uniform(0, 2.5, n)]))
    return np.concatenate(out)


@test
def test_observability_is_isotropic_in_a_closed_room():
    rng = np.random.default_rng(11)
    o = observability(room_surfaces(rng), origin=(0, 0, 1.0), voxel_m=0.5)
    assert o.n_points > 100, o.n_points
    assert o.trans_weakest > 0.15, f"a closed room should constrain every axis: {o.trans_eigenvalues}"
    assert o.trans_condition < 6.0, o.trans_condition


@test
def test_observability_finds_the_free_axis_of_a_corridor():
    """Two parallel walls and a floor: motion ALONG the corridor is free."""
    rng = np.random.default_rng(12)
    pts = room_surfaces(rng, walls=2)          # only the x = +/-4 walls, plus floor
    o = observability(pts, origin=(0, 0, 1.0), voxel_m=0.5)
    assert o.degenerate(0.05), f"corridor must read as degenerate: {o.trans_eigenvalues}"
    d = np.abs(o.trans_weakest_direction)
    assert d[1] > 0.95, f"the free axis should be y (along the corridor), got {d}"
    assert 0.4 < o.normal_isotropy < 0.85, o.normal_isotropy


@test
def test_observability_collapses_on_a_single_wall():
    rng = np.random.default_rng(13)
    n = 30000
    wall = np.column_stack([np.full(n, 3.0), rng.uniform(-5, 5, n),
                            rng.uniform(0, 2.5, n)])
    o = observability(wall, origin=(0, 0, 1.0), voxel_m=0.5)
    assert o.trans_weakest < 0.02, o.trans_eigenvalues
    # isotropy is what separates this from a corridor: both have a weakest
    # eigenvalue of ~0, and only one of them has TWO free axes
    assert o.normal_isotropy < 0.2, o.normal_isotropy


@test
def test_observability_is_not_a_point_count():
    """Ten times the points on the same surfaces must not change the verdict."""
    rng = np.random.default_rng(14)
    a = observability(room_surfaces(rng, n=4000), origin=(0, 0, 1.0), voxel_m=0.5)
    b = observability(room_surfaces(rng, n=40000), origin=(0, 0, 1.0), voxel_m=0.5)
    assert abs(a.trans_weakest - b.trans_weakest) < 0.08, \
        f"{a.trans_eigenvalues} vs {b.trans_eigenvalues}"


@test
def test_observability_survives_a_cloud_with_no_planes():
    rng = np.random.default_rng(15)
    o = observability(rng.normal(size=(5000, 3)), voxel_m=0.5)
    assert o.n_points >= 0 and np.isfinite(o.trans_eigenvalues).all()


# ================================================================= ablate =====
@test
def test_crop_fov_keeps_the_declared_wedge():
    rng = np.random.default_rng(21)
    az = rng.uniform(-np.pi, np.pi, 20000)
    p = np.column_stack([np.cos(az), np.sin(az), np.zeros(20000)]) * 5.0
    out = crop_fov(p, hfov_deg=90.0)
    frac = len(out) / len(p)
    assert abs(frac - 0.25) < 0.02, frac
    assert np.all(np.abs(np.degrees(np.arctan2(out[:, 1], out[:, 0]))) <= 45.001)


@test
def test_crop_fov_87_deg_matches_a_realsense_wedge():
    rng = np.random.default_rng(22)
    az = rng.uniform(-np.pi, np.pi, 40000)
    p = np.column_stack([np.cos(az), np.sin(az), np.zeros(40000)]) * 3.0
    frac = len(crop_fov(p, hfov_deg=87.0)) / len(p)
    assert abs(frac - 87.0 / 360.0) < 0.02, frac


@test
def test_truncate_range_is_a_spherical_shell():
    p = np.array([[1, 0, 0], [5, 0, 0], [11, 0, 0], [0, 0, 0.1]], float)
    out = truncate_range(p, 0.2, 10.0)
    assert len(out) == 2 and np.allclose(out[0], [1, 0, 0])


@test
def test_decimate_beams_reduces_rings_not_points_at_random():
    rng = np.random.default_rng(23)
    el = np.repeat(np.linspace(-0.3, 0.3, 128), 200)      # 128 rings
    az = rng.uniform(-np.pi, np.pi, el.size)
    p = np.column_stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                         np.sin(el)]) * 8.0
    out = decimate_beams(p, beams=32, total=128)
    frac = len(out) / len(p)
    assert 0.2 < frac < 0.3, frac
    # the survivors must lie on few distinct elevations, not be spread over all
    e = np.round(np.degrees(np.arcsin(out[:, 2] / 8.0)), 2)
    assert len(np.unique(e)) <= 40, len(np.unique(e))


@test
def test_decimate_beams_is_a_noop_when_asked_for_more_than_exist():
    rng = np.random.default_rng(24)
    p = rng.normal(size=(1000, 3)) * 3
    assert len(decimate_beams(p, beams=256, total=128)) == len(p)


@test
def test_subsample_is_seeded_and_exact():
    rng = np.random.default_rng(25)
    p = rng.normal(size=(10000, 3))
    a = subsample(p, 0.1, seed=7)
    b = subsample(p, 0.1, seed=7)
    c = subsample(p, 0.1, seed=8)
    assert len(a) == 1000 and np.array_equal(a, b)
    assert not np.array_equal(a, c)


@test
def test_select_rate_honours_elapsed_time_not_index():
    stamps = np.concatenate([np.arange(0, 1.0, 0.1),          # 10 Hz
                             np.arange(5.0, 6.0, 0.1)])       # after a 4 s hole
    idx = select_rate(stamps, 2.0)
    kept = stamps[idx]
    assert np.all(np.diff(kept) >= 0.5 - 1e-9), kept
    assert kept[0] == stamps[0]


@test
def test_ablation_pipeline_order_and_label():
    rng = np.random.default_rng(26)
    az = rng.uniform(-np.pi, np.pi, 50000)
    el = rng.uniform(-0.3, 0.3, 50000)
    r = rng.uniform(0.5, 30.0, 50000)
    p = np.column_stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                         np.sin(el)]) * r[:, None]
    ab = Ablation(hfov_deg=87.0, max_range_m=10.0, density=0.5, seed=1)
    out = ablate(p, ab)
    d = np.linalg.norm(out, axis=1)
    assert d.max() <= 10.001
    assert np.all(np.abs(np.degrees(np.arctan2(out[:, 1], out[:, 0]))) <= 43.51)
    assert ab.label() == "fov87-r10-d0.5"


# ================================================================= collab =====
def two_agents(n=300, t0=1000.0, rate=10.0):
    """Two platforms on separate loops in one room, in one world frame."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    stamps = t0 + np.arange(n) / rate
    A = np.stack([se3.pose_from_rpy(3 * np.cos(a), 3 * np.sin(a), 0.2,
                                    0, 0, np.degrees(a) + 90) for a in th])
    B = np.stack([se3.pose_from_rpy(6 + 2 * np.cos(-a), 2 * np.sin(-a), 0.2,
                                    0, 0, np.degrees(-a) - 90) for a in th])
    return (Trajectory(stamps, A, "ref/a", world="map"),
            Trajectory(stamps, B, "ref/b", world="map"))


@test
def test_relative_error_is_zero_in_an_arbitrary_common_frame():
    ra, rb = two_agents()
    W = se3.pose_from_rpy(17.0, -4.0, 2.5, 0, 0, 132.0)     # the method's own frame
    r = relative_trajectory_error(ra.transform_left(W), rb.transform_left(W), ra, rb)
    assert r.trans.rmse < 1e-9, r.trans.rmse
    assert r.rot.rmse < 1e-6
    assert "registered" in r.verdict()


@test
def test_relative_error_catches_a_map_merge_failure():
    """Agent b's frame is offset 40 cm and 3 deg: a place-recognition error."""
    ra, rb = two_agents()
    bad = rb.transform_left(se3.pose_from_rpy(0.4, 0, 0, 0, 0, 3.0))
    r = relative_trajectory_error(ra, bad, ra, rb)
    assert r.trans.rmse > 0.2, r.trans.rmse
    assert r.constant_trans_m > 0.2, r.constant_trans_m
    assert r.trans_after_constant.rmse < 0.5 * r.constant_trans_m
    assert "mis-registration" in r.verdict()


@test
def test_relative_error_separates_drift_from_mis_registration():
    """b's frame is right but its trajectory drifts: the constant part is small."""
    ra, rb = two_agents()
    drift = rb.poses.copy()
    # zero-mean, so the best-fit constant has nothing to absorb. A monotonic
    # ramp would be half-absorbed into the constant term -- see the docstring.
    drift[:, 0, 3] += np.linspace(-0.25, 0.25, len(rb))
    r = relative_trajectory_error(ra, Trajectory(rb.stamps, drift), ra, rb)
    assert r.trans_after_constant.rmse > 0.05, r.trans_after_constant.rmse
    assert r.constant_trans_m < r.trans_after_constant.rmse * 2
    assert "drift" in r.verdict()


@test
def test_relative_error_refuses_non_overlapping_clocks():
    ra, rb = two_agents()
    shifted = Trajectory(rb.stamps + 9000.0, rb.poses)
    try:
        relative_trajectory_error(ra, shifted, ra, shifted)
    except ValueError as e:
        assert "overlap in time" in str(e) or "shared stamps" in str(e), str(e)
    else:
        raise AssertionError("agents on disjoint clocks were scored anyway")


@test
def test_place_overlap_answers_the_loop_closure_question():
    ra, rb = two_agents()
    near = place_overlap(ra, rb, radius_m=5.0)
    assert near["fraction_of_b_near_a"] > 0.5, near
    assert "possible" in near["verdict"]
    far = Trajectory(rb.stamps, rb.poses.copy())
    far.poses[:, 0, 3] += 200.0
    none = place_overlap(ra, far, radius_m=5.0)
    assert none["fraction_of_b_near_a"] == 0.0
    assert "never came within" in none["verdict"]


@test
def test_agent_separation_reports_the_geometry():
    ra, rb = two_agents()
    s = agent_separation(ra, rb)
    assert s["min_m"] > 0 and s["max_m"] > s["min_m"]
    assert s["time_overlap_s"] > 25


@test
def test_completeness_gain_is_reported_against_the_solo_run():
    solo = {"fscore": {"0.05": {"precision": 0.9, "recall": 0.40, "f": 0.55}},
            "chamfer_m": 0.05, "completeness_m": {"mean": 0.09}}
    joint = {"fscore": {"0.05": {"precision": 0.88, "recall": 0.72, "f": 0.79}},
             "chamfer_m": 0.03, "completeness_m": {"mean": 0.04}}
    g = completeness_gain(solo, joint)
    assert abs(g["0.05"]["recall_gain"] - 0.32) < 1e-9
    assert g["completeness_gain_m"] > 0


# ======================================================== infrastructure ======
@test
def test_predicted_bearing_matches_a_hand_computed_case():
    """Node at the origin looking down +x; agent due left should read +90 deg."""
    stamps = np.array([0.0, 1.0])
    poses = np.stack([se3.pose_from_rpy(0, 4, 0, 0, 0, 0),
                      se3.pose_from_rpy(4, 0, 0, 0, 0, 0)])
    pred = predict_observation(Trajectory(stamps, poses), np.eye(4))
    assert abs(pred["azimuth_deg"][0] - 90.0) < 1e-9, pred["azimuth_deg"]
    assert abs(pred["azimuth_deg"][1] - 0.0) < 1e-9
    assert abs(pred["range_m"][0] - 4.0) < 1e-9


@test
def test_infrastructure_residual_separates_bias_from_noise():
    rng = np.random.default_rng(31)
    ra, _ = two_agents()
    infra = se3.pose_from_rpy(-5.508181, -2.590753, 1.998232,
                              -117.6730, 0.7379, -127.6503)   # coop2's node
    pred = predict_observation(ra, infra)
    obs = pred["azimuth_deg"] + 0.8 + rng.normal(scale=0.05, size=len(ra))
    r = observation_residual(pred, ra.stamps, obs, pred["range_m"])
    assert abs(r["azimuth_bias_deg"] - 0.8) < 0.05, r["azimuth_bias_deg"]
    assert abs(r["range_bias_m"]) < 1e-9
    assert r["n"] == len(ra)


@test
def test_infrastructure_residual_refuses_a_clock_mismatch():
    ra, _ = two_agents()
    pred = predict_observation(ra, np.eye(4))
    try:
        observation_residual(pred, ra.stamps + 500.0, pred["azimuth_deg"])
    except ValueError as e:
        assert "clock" in str(e)
    else:
        raise AssertionError("a detection stream on another clock was matched anyway")


@test
def test_observability_reports_starvation_not_degeneracy():
    """The density axis must measure the sensor, not the normal estimator."""
    rng = np.random.default_rng(41)
    room = room_surfaces(rng, n=20000)
    sparse = room[rng.choice(len(room), 200, replace=False)]
    fixed = observability(sparse, origin=(0, 0, 1.0), voxel_m=0.25)
    assert fixed.starved(), fixed.status
    assert not fixed.degenerate(), "starved must not be reported as degenerate"


@test
def test_auto_voxel_keeps_the_verdict_stable_across_density():
    rng = np.random.default_rng(42)
    room = room_surfaces(rng, n=40000)
    full = observability(room, origin=(0, 0, 1.0), voxel_m="auto")
    thin = observability(room[rng.choice(len(room), len(room) // 50, replace=False)],
                         origin=(0, 0, 1.0), voxel_m="auto")
    assert full.status == "ok" and thin.status == "ok", (full.status, thin.status)
    assert not full.degenerate() and not thin.degenerate(), \
        f"the same room at 1/50 density must not become degenerate: " \
        f"{full.trans_eigenvalues} vs {thin.trans_eigenvalues}"
    assert thin.voxel_m > full.voxel_m, (full.voxel_m, thin.voxel_m)


@test
def test_optical_axes_are_not_interchangeable_with_ros():
    """The bug this flag exists to stop: a node ABOVE the agents must see them
    below it, and the ros convention on an optical pose says the opposite."""
    stamps = np.array([0.0, 1.0])
    poses = np.stack([se3.pose_from_rpy(0, 4, 0, 0, 0, 0),
                      se3.pose_from_rpy(4, 0, 0, 0, 0, 0)])
    tr = Trajectory(stamps, poses)
    infra = se3.pose_from_rpy(-5.508181, -2.590753, 1.998232,
                              -117.6730, 0.7379, -127.6503)   # coop2's node, 2 m up
    opt = predict_observation(tr, infra, axes="optical")
    ros = predict_observation(tr, infra, axes="ros")
    assert opt["elevation_deg"][0] < 0, "a node 1.8 m above must look DOWN"
    assert abs(ros["azimuth_deg"][0]) > 150, "the ros reading should be the wrong one"
    assert np.allclose(opt["range_m"], ros["range_m"]), "range is frame-free"


@test
def test_predict_observation_rejects_an_unknown_frame():
    tr = Trajectory(np.array([0.0, 1.0]), np.stack([np.eye(4), np.eye(4)]))
    try:
        predict_observation(tr, np.eye(4), axes="nedd")
    except ValueError as e:
        assert "'ros' or 'optical'" in str(e)
    else:
        raise AssertionError("an unknown axis convention was accepted")


@test
def test_preflight_catches_a_null_nested_in_params():
    """Rule 3 has to hold one level down, not just at the top.

    `source_runs: {mobile_1: null}` is a dict, so a flat `v is None` scan passes
    it and the container starts on a value the config explicitly said could not
    be guessed. That is the failure rule 3 exists to prevent, wearing one extra
    level of indentation.

    This drives the REAL preflight over the REAL configs, because the bug it
    pins was in production code that a re-implementation of the walk would have
    happily agreed with.
    """
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    spec = importlib.util.spec_from_file_location("_rm", os.path.join(here, "run_method.py"))
    rm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rm)
    from slambench.config import load_dataset, load_method

    cfg = load_dataset(os.path.join(root, "configs", "coop2.yaml"))

    m = load_method(os.path.join(root, "configs", "methods",
                                 "decoupled_registration_nolidar.yaml"))
    _, problems = rm.preflight(cfg, m, "mobile_1.zed_rgbd")
    assert any("source_runs.mobile_1" in p for p in problems), (
        "a null nested inside params was not caught: " + repr(problems))

    # and the converse: a config with no nulls anywhere still starts, so the
    # stricter walk did not simply block everything.
    ok = load_method(os.path.join(root, "configs", "methods", "rtabmap_rgbd.yaml"))
    _, none_expected = rm.preflight(cfg, ok, "mobile_1.zed_rgbd")
    assert not none_expected, none_expected


# ---------------------------------------------------------------- stage 1 rig
#
# The container-side helpers. None of them needs ROS, and each of them is a
# place where a run completes and is silently wrong if the arithmetic slips.

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker" / "common"))


@test
def test_deproject_is_the_pinhole_model():
    from depth_to_cloud import deproject
    # a pixel AT the principal point lies on the boresight, at exactly z
    img = np.zeros((5, 7), np.float32); img[2, 3] = 3.0
    p = deproject(img, 100.0, 100.0, 3.0, 2.0)
    assert p.shape == (1, 3) and np.allclose(p[0], [0, 0, 3]), p
    # one pixel right of it, at depth z, is z/fx metres to the +x side
    img = np.zeros((5, 7), np.float32); img[2, 4] = 3.0
    p = deproject(img, 100.0, 100.0, 3.0, 2.0)
    assert np.allclose(p[0], [0.03, 0, 3]), p


@test
def test_deproject_drops_invalid_instead_of_putting_it_at_the_origin():
    from depth_to_cloud import deproject
    img = np.full((4, 6), 2.0, np.float32)
    img[0, 0] = 0.0          # ROS's "no return"
    img[1, 1] = np.nan
    img[2, 2] = 20.0         # past the stream's range
    p = deproject(img, 100.0, 100.0, 3.0, 2.0, r_min=0.3, r_max=12.0)
    assert len(p) == 24 - 3, len(p)
    assert not np.any(np.all(p == 0, axis=1)), "an invalid pixel became a point at the camera"


@test
def test_deproject_strides_coordinates_with_the_image():
    """Subsampling the image without subsampling the pixel coordinates is a
    silent focal-length error -- the geometry stays plausible and the scale is
    wrong by the stride."""
    from depth_to_cloud import deproject
    img = np.full((8, 8), 2.0, np.float32)
    full = deproject(img, 100.0, 100.0, 3.5, 3.5, stride=1)
    strided = deproject(img, 100.0, 100.0, 3.5, 3.5, stride=2)
    assert len(strided) == 16
    # every strided point must coincide with one of the full-resolution points
    for q in strided:
        assert np.min(np.linalg.norm(full - q, axis=1)) < 1e-6, q


@test
def test_depth_scale_mismatch_is_refused_not_corrected():
    """kiss_icp.yaml: metres vs millimetres is a factor of 1000, and the wrong
    one 'completes and produces a map of a room 1000 m across'."""
    from depth_to_cloud import check_scale
    assert check_scale("32FC1", 1.0) == 1.0
    assert check_scale("16UC1", 0.001) == 0.001
    for enc, scale in (("16UC1", 1.0), ("32FC1", 0.001), ("mono16", 0.001)):
        try:
            check_scale(enc, scale)
            raise AssertionError(f"{enc} at scale {scale} was accepted")
        except ValueError:
            pass


@test
def test_params_split_between_rtabmap_and_ros_without_dropping_either():
    from params_to_args import to_args, to_ros_params
    params = {"Vis/MinInliers": 15, "Grid/3D": "true", "Optimizer/GravitySigma": 0.3,
              "max_range": 20.0, "deskew": True}
    rtab = to_args(params)
    assert rtab == ["--Grid/3D", "true", "--Optimizer/GravitySigma", "0.3",
                    "--Vis/MinInliers", "15"], rtab
    ros = to_ros_params(params)
    assert ros == ["-p", "deskew:=true", "-p", "max_range:=20.0"], ros
    # every key ends up in exactly one of the two lists
    assert len(rtab) // 2 + len(ros) // 2 == len(params)


@test
def test_the_mapping_node_is_never_handed_an_odom_parameter():
    """Measured: rtabmap_slam does not declare Odom/*, and in ROS 2 setting an
    undeclared parameter throws -- the mapping node aborted at startup on
    `--Odom/ResetCountdown 8` and the loop-closing row became odometry-only
    with no error in the odometry log."""
    from params_to_args import to_args
    params = {"Odom/ResetCountdown": 8, "Vis/MinInliers": 15, "RGBD/LinearUpdate": 0.05}
    mapping = to_args(params, exclude=("Odom/",))
    assert "--Odom/ResetCountdown" not in mapping, mapping
    assert "--Vis/MinInliers" in mapping and "--RGBD/LinearUpdate" in mapping
    odometry = to_args(params)
    assert "--Odom/ResetCountdown" in odometry
    launch = (Path(__file__).resolve().parents[1] / "docker" / "rtabmap" / "launch.sh").read_text()
    assert 'exclude=("Odom/",)' in launch, "launch.sh does not filter the mapping node's args"
    assert 'odom_args:="$ODOM_ARGS"' in launch and 'rtabmap_args:="--delete_db_on_start $RTAB_ARGS"' in launch


@test
def test_booleans_survive_the_trip_to_rtabmap():
    """RTAB-Map parses `True` as false. YAML hands us both `true` (bool) and
    `"true"` (string, as Grid/3D is written) and both must come out lowercase."""
    from params_to_args import to_args
    assert to_args({"Grid/3D": True}) == ["--Grid/3D", "true"]
    assert to_args({"Grid/3D": "true"}) == ["--Grid/3D", "true"]
    assert to_args({"Grid/3D": "True"}) == ["--Grid/3D", "true"]
    assert to_args({"Grid/3D": False}) == ["--Grid/3D", "false"]


@test
def test_row_padding_does_not_shear_the_image():
    """A padded image reshaped by width alone comes out sheared, and a sheared
    image still tracks -- badly, with nothing anywhere to say why."""
    from bag_to_frames import to_rgb
    rows = [bytes([10, 20, 30, 255, 11, 21, 31, 255, 12, 22, 32, 255]) + b"\x00" * 4,
            bytes([40, 50, 60, 255, 41, 51, 61, 255, 42, 52, 62, 255]) + b"\x00" * 4]
    img = to_rgb(b"".join(rows), 2, 3, "bgra8", step=16)
    assert img.shape == (2, 3, 3)
    assert img[0, 0].tolist() == [30, 20, 10]      # bgra -> rgb
    assert img[1, 2].tolist() == [62, 52, 42]      # the padded row did not slide
    try:
        to_rgb(b"", 1, 1, "bayer_rggb8", 1)
        raise AssertionError("an unknown encoding was decoded anyway")
    except ValueError:
        pass


@test
def test_restamp_inverts_the_fabricated_clock_exactly():
    from restamp_tum import restamp
    stamps = [1700000000.5 + i * 0.068 for i in range(100)]      # 14.7 Hz, as recorded
    rows = restamp(["0.0 1 2 3 0 0 0 1", "1.0 4 5 6 0 0 0 1", "3.3 7 8 9 0 0 0 1"], stamps)
    assert [r.split()[0] for r in rows] == [f"{stamps[i]:.9f}" for i in (0, 30, 99)]
    assert rows[0].split()[1:] == ["1", "2", "3", "0", "0", "0", "1"], "pose columns moved"


@test
def test_restamp_refuses_rather_than_approximating():
    """A resynchronisation that 'mostly works' is a time offset nobody finds."""
    from restamp_tum import restamp
    stamps = [1700000000.5 + i * 0.068 for i in range(100)]
    for bad in (["0.017 1 2 3 0 0 0 1"],        # not an integer frame index
                ["4.0 1 2 3 0 0 0 1"],          # past the frames the bag held
                ["0.0 1 2 3"],                  # not a TUM row
                ["# only a comment"]):          # nothing to restamp
        try:
            restamp(bad, stamps)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass


@test
def test_an_inertial_method_is_actually_played_its_imu():
    """The container sees exactly `ros2 bag play --topics <SLAM_TOPICS>`, so a
    topic missing from that list is a sensor the method never receives. An
    inertial row starved of its IMU still completes, and its number then reads
    as 'the IMU did not help'."""
    import scripts.run_method as rm
    from slambench.config import load_dataset, load_method
    root = str(Path(__file__).resolve().parents[1])
    cfg = load_dataset(os.path.join(root, "configs", "coop2.yaml"))
    m = load_method(os.path.join(root, "configs", "methods", "rtabmap_rgbd_imu.yaml"))

    for stream, imu in (("mobile_1.zed_rgbd", "/mobile_1/zed/imu/data"),
                        ("mobile_2.realsense_rgbd", "/mobile_2/imu")):
        # mobile_2's inertial cell is blocked on its missing extrinsic (see
        # test_an_inertial_row_without_its_extrinsic_is_refused); the topic
        # assembly under test here happens either way.
        info, _ = rm.preflight(cfg, m, stream)
        assert imu in info["topics"], (stream, info["topics"])
        assert "/tf_static" in info["topics"], info["topics"]
        # /tf carries the recording pipeline's own map->sensor estimate; replaying
        # it would let a method read its answer off its input.
        assert "/tf" not in info["topics"], info["topics"]

    # and the IMU-free half of the pair is not handed one
    free = load_method(os.path.join(root, "configs", "methods", "rtabmap_rgbd.yaml"))
    info, _ = rm.preflight(cfg, free, "mobile_1.zed_rgbd")
    assert not any("imu" in t for t in info["topics"]), info["topics"]


@test
def test_the_stage_1_cells_are_runnable_or_blocked_for_a_recorded_reason():
    """docs/PLAN.md stage 1: three backbones on two platforms.

    Measured 2026-09-19, the grid is not six clean cells. mobile_2's inertial
    RTAB-Map is blocked on an extrinsic the bag cannot supply, so that platform's
    feature+depth backbone is the IMU-FREE half of the pair. Writing the real
    grid down here means a cell that silently starts working -- or stops --
    shows up as a test change rather than as a number in a table.
    """
    import scripts.run_method as rm
    from slambench.config import load_dataset, load_method
    root = str(Path(__file__).resolve().parents[1])
    cfg = load_dataset(os.path.join(root, "configs", "coop2.yaml"))

    runnable = [("kiss_icp", "mobile_1.zed_rgbd"),
                ("kiss_icp", "mobile_2.realsense_rgbd"),
                ("rtabmap_rgbd_imu", "mobile_1.zed_rgbd"),
                ("rtabmap_rgbd", "mobile_2.realsense_rgbd"),
                ("mast3r_slam", "mobile_1.zed_rgbd"),
                ("mast3r_slam", "mobile_2.realsense_rgbd")]
    for name, stream in runnable:
        m = load_method(os.path.join(root, "configs", "methods", f"{name}.yaml"))
        assert stream in m.streams, f"{name} does not declare {stream}"
        _, problems = rm.preflight(cfg, m, stream)
        assert not problems, (name, stream, problems)

    blocked = [("rtabmap_rgbd_imu", "mobile_2.realsense_rgbd", "extrinsic_from_camera")]
    for name, stream, why in blocked:
        m = load_method(os.path.join(root, "configs", "methods", f"{name}.yaml"))
        _, problems = rm.preflight(cfg, m, stream)
        assert any(why in p for p in problems), (name, stream, problems)


@test
def test_no_image_script_sources_ros_with_u_active():
    """`set -u` + ROS's setup.bash is instant death, at line 8, before a single
    line of ours runs: setup.bash reads AMENT_TRACE_SETUP_FILES with no default.
    The strictness is wanted everywhere else -- an unset SLAM_* expanding to
    nothing is how a container runs on starved input -- so every source is
    bracketed instead. This walks the scripts so the next image cannot forget.
    """
    root = Path(__file__).resolve().parents[1] / "docker"
    checked = 0
    for sh in sorted(root.rglob("*.sh")):
        u = False
        for n, line in enumerate(sh.read_text().splitlines(), 1):
            code = line.split("#", 1)[0].strip()
            if code.startswith("set ") and "u" in code.split()[1].lstrip("-+"):
                u = code.split()[1].startswith("-")
            elif ("setup.bash" in code and
                  (code.startswith("source ") or code.startswith(". "))):
                assert not u, f"{sh.name}:{n} sources ROS with `set -u` active: {code}"
                checked += 1
        assert not u or True
    assert checked >= 4, f"only found {checked} ROS sources; did the scripts move?"


@test
def test_tf_components_answer_the_lookup_question():
    """tf resolves through any path in either direction, so 'can these two
    frames be transformed' is exactly 'are they in one undirected component'."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from bag_probe import components
    comps = components([("base", "cam"), ("cam", "cam_optical"), ("imu", "imu_child")])
    assert [sorted(c) for c in comps] == [["base", "cam", "cam_optical"],
                                          ["imu", "imu_child"]], comps
    # direction must not matter: a child->parent edge connects just the same
    assert len(components([("a", "b"), ("c", "b")])) == 1


@test
def test_signed_stamp_gap_distinguishes_jitter_from_interleaving():
    """The MAGNITUDE against the frame period is the diagnostic. Jitter is a
    small fraction of a period and an interval rejects its tail; a gap near half
    a period means the closest available partner is half a frame of real motion
    away, and no interval recovers a pairing that was never recorded."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from bag_probe import nearest_dt
    period = 0.068                                   # 14.7 Hz, as recorded
    depth = np.arange(20) * period

    assert np.all(nearest_dt(depth.copy(), depth) == 0)          # identical stamps
    jitter = depth + 0.004                                       # well inside a frame
    assert np.allclose(nearest_dt(jitter, depth), 0.004)
    assert np.median(np.abs(nearest_dt(jitter, depth))) < 0.4 * period

    # Nearly half a period out: every colour frame sits close to midway between
    # two depth frames, so the closest partner is still ~half a frame of real
    # motion away. Magnitude catches it, and the sign is unambiguous here.
    inter = depth + 0.45 * period
    dt = nearest_dt(inter, depth)
    assert np.median(np.abs(dt)) > 0.4 * period, dt
    assert len(set(np.sign(dt[:-1]))) == 1, dt

    # Put the offset ON the tie and add jitter, and the sign starts flipping --
    # which is what the real bag showed. It is a symptom of sitting at the tie,
    # not an independent fault, and the magnitude already caught it.
    rng = np.random.default_rng(0)
    straddling = depth + period / 2 + rng.normal(0, 0.004, len(depth))
    flips = np.sign(nearest_dt(straddling, depth))
    assert np.sum(np.diff(flips) != 0) >= 2, flips
    assert np.median(np.abs(nearest_dt(straddling, depth))) > 0.4 * period


@test
def test_the_camera_imu_edge_is_supplied_when_the_bag_lacks_it():
    """Measured 2026-09-19: zed_imu_link is absent from the replayed tf_static,
    so RTAB-Map dropped every IMU sample and the inertial row would have become
    its own IMU-free control while still producing a trajectory."""
    import scripts.run_method as rm
    from slambench.config import load_dataset, load_method
    root = str(Path(__file__).resolve().parents[1])
    cfg = load_dataset(os.path.join(root, "configs", "coop2.yaml"))
    m = load_method(os.path.join(root, "configs", "methods", "rtabmap_rgbd_imu.yaml"))
    ext = cfg.raw["streams"]["mobile_1.imu"]["extrinsic_from_camera"]

    saved = sys.argv
    try:
        sys.argv = ["run_method.py", "--config", os.path.join(root, "configs", "coop2.yaml"),
                    "--method", os.path.join(root, "configs", "methods", "rtabmap_rgbd_imu.yaml"),
                    "--stream", "mobile_1.zed_rgbd",
                    "--runs-root", tempfile.mkdtemp()]
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert rm.main() == 0
        cmd = buf.getvalue()
    finally:
        sys.argv = saved

    assert "SLAM_STATIC_TF=" in cmd, cmd
    line = cmd.split("SLAM_STATIC_TF=", 1)[1].split("'")[0]
    parent, child, *nums = line.split()
    assert (parent, child) == (ext["parent"], ext["child"]), (parent, child)
    # the NUMBERS must be the config's, not a rounded or re-derived copy (rule 4)
    assert [float(v) for v in nums] == list(ext["xyz"]) + list(ext["quat_xyzw"]), nums

    # and the IMU-free half of the pair is handed no transform at all
    free = load_method(os.path.join(root, "configs", "methods", "rtabmap_rgbd.yaml"))
    assert not free.needs_imu


@test
def test_sync_mode_comes_from_the_recording_not_a_default():
    """mobile_1's colour and depth are stamped byte-identically (2288/2288) and
    mobile_2's are 10 us apart (0/4295 exact). Feeding the first to an
    approximate matcher is what paired frames a full 67 ms period apart on the
    first container run -- a whole frame of cart motion inside each RGB-D pair."""
    import io, contextlib
    import scripts.run_method as rm
    root = str(Path(__file__).resolve().parents[1])

    def command(method, stream):
        saved = sys.argv
        try:
            sys.argv = ["run_method.py",
                        "--config", os.path.join(root, "configs", "coop2.yaml"),
                        "--method", os.path.join(root, "configs", "methods", f"{method}.yaml"),
                        "--stream", stream, "--runs-root", tempfile.mkdtemp()]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                assert rm.main() == 0
            return buf.getvalue()
        finally:
            sys.argv = saved

    assert "SLAM_SYNC=exact" in command("rtabmap_rgbd", "mobile_1.zed_rgbd")
    assert "SLAM_SYNC=approx" in command("rtabmap_rgbd", "mobile_2.realsense_rgbd")


@test
def test_the_colour_depth_edge_is_published_only_when_the_frames_differ():
    """mobile_2's colour and depth sit in different optical frames and this
    bag's tf_static carries no RealSense frames at all, so the 59 mm offset has
    to come from the config or it lands on every point. On mobile_1 both images
    carry one frame_id, and publishing an identity edge there would be noise."""
    import io, contextlib
    import scripts.run_method as rm
    from slambench.config import load_dataset
    root = str(Path(__file__).resolve().parents[1])
    cfg = load_dataset(os.path.join(root, "configs", "coop2.yaml"))

    def command(stream):
        saved = sys.argv
        try:
            sys.argv = ["run_method.py",
                        "--config", os.path.join(root, "configs", "coop2.yaml"),
                        "--method", os.path.join(root, "configs", "methods", "rtabmap_rgbd.yaml"),
                        "--stream", stream, "--runs-root", tempfile.mkdtemp()]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                assert rm.main() == 0
            return buf.getvalue()
        finally:
            sys.argv = saved

    m1 = cfg.stream("mobile_1.zed_rgbd")
    assert m1["color_frame"] == m1["sensor_frame"]
    assert "SLAM_STATIC_TF" not in command("mobile_1.zed_rgbd")

    m2 = cfg.stream("mobile_2.realsense_rgbd")
    line = command("mobile_2.realsense_rgbd").split("SLAM_STATIC_TF=", 1)[1].split("'")[0]
    parent, child, x, y, z, *q = line.split()
    assert (parent, child) == (m2["color_frame"], m2["sensor_frame"]), (parent, child)
    # the translation is the config's, not a re-derived one (rule 4)
    ext = m2["reference_frame_from_sensor"]
    assert abs(float(x) - ext["x"]) < 1e-9 and abs(float(y) - ext["y"]) < 1e-9
    assert abs(np.linalg.norm([float(v) for v in q]) - 1.0) < 1e-9, q


@test
def test_an_inertial_row_without_its_extrinsic_is_refused():
    """It would not fail. RTAB-Map logs "Dropping imu data!" and CONTINUES, so
    the row lands in the table as evidence the IMU did not help, with no IMU in
    it. mobile_2.imu.extrinsic_from_camera is null and the bag's tf_static
    carries no inertial frames, so nothing can supply it at run time."""
    import scripts.run_method as rm
    from slambench.config import load_dataset, load_method
    root = str(Path(__file__).resolve().parents[1])
    cfg = load_dataset(os.path.join(root, "configs", "coop2.yaml"))
    m = load_method(os.path.join(root, "configs", "methods", "rtabmap_rgbd_imu.yaml"))

    _, problems = rm.preflight(cfg, m, "mobile_2.realsense_rgbd")
    assert any("extrinsic_from_camera is null" in p for p in problems), problems
    # mobile_1 has it (I13) and must still pass, so the gate is not blanket
    _, ok = rm.preflight(cfg, m, "mobile_1.zed_rgbd")
    assert not ok, ok


def _import_record_tum():
    """Import the recorder without ROS, by standing in for the three rclpy
    symbols it touches at module scope. The formatting and the choice of which
    trajectory to emit are plain Python and deserve testing on any machine."""
    import types
    stub = types.ModuleType("rclpy")
    node = types.ModuleType("rclpy.node")
    node.Node = object
    qos = types.ModuleType("rclpy.qos")
    qos.QoSProfile = lambda **kw: None
    qos.ReliabilityPolicy = types.SimpleNamespace(RELIABLE=1)
    saved = {k: sys.modules.get(k) for k in ("rclpy", "rclpy.node", "rclpy.qos")}
    sys.modules.update({"rclpy": stub, "rclpy.node": node, "rclpy.qos": qos})
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker" / "common"))
        import importlib
        return importlib.import_module("record_tum")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@test
def test_tum_row_unwraps_the_nested_pose_and_keeps_the_stamp():
    import types
    rt = _import_record_tum()
    h = types.SimpleNamespace(stamp=types.SimpleNamespace(sec=1700000000, nanosec=250000000))
    inner = types.SimpleNamespace(position=types.SimpleNamespace(x=1.0, y=-2.0, z=0.5),
                                  orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))
    bare = rt._row(h, inner)
    # Odometry and PoseWithCovariance nest the pose one level down; a recorder
    # that misses that writes the covariance block's address, not a position.
    nested = rt._row(h, types.SimpleNamespace(pose=inner))
    assert bare == nested, (bare, nested)
    cols = bare.split()
    assert cols[0] == "1700000000.250000000", cols[0]
    assert [float(c) for c in cols[1:4]] == [1.0, -2.0, 0.5]


@test
def test_a_loop_closing_method_is_scored_on_its_OPTIMISED_graph():
    """RTAB-Map publishes two different things. `/rtabmap/odom` is incremental
    and never benefits from a closure; the optimised graph applies every closure
    retroactively. Recording the first while the method config says
    `loop_closure: true` puts a drift number in a row that claims to have none.
    """
    import types
    rt = _import_record_tum()

    def stamped(t, x):
        return types.SimpleNamespace(
            header=types.SimpleNamespace(
                stamp=types.SimpleNamespace(sec=int(t), nanosec=0)),
            pose=types.SimpleNamespace(
                position=types.SimpleNamespace(x=x, y=0.0, z=0.0),
                orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)))

    out = Path(tempfile.mkdtemp())
    rec = rt.Recorder.__new__(rt.Recorder)
    rec.out, rec.n, rec.t0, rec.cloud, rec.seen, rec.diagnostics = out, 0, 0.0, None, 0, {}
    rec.fh = (out / "odometry.tum").open("w")
    rec.fh.write(rt.HEADER)
    for i in range(3):                               # drifting odometry
        rec.on_pose(stamped(i, i * 1.5))
    # the graph, re-optimised: same stamps, corrected positions
    rec.path = types.SimpleNamespace(poses=[stamped(i, i * 1.0) for i in range(3)])
    rec.finish()

    traj = [l.split() for l in (out / "trajectory.tum").read_text().splitlines()
            if not l.startswith("#")]
    odom = [l.split() for l in (out / "odometry.tum").read_text().splitlines()
            if not l.startswith("#")]
    assert [float(r[1]) for r in traj] == [0.0, 1.0, 2.0], traj
    assert [float(r[1]) for r in odom] == [0.0, 1.5, 3.0], odom
    meta = json.loads((out / "timing.json").read_text())
    assert meta["trajectory_source"] == "optimised_graph", meta

    # The graph must already be ON DISK before finish() -- a container that is
    # killed rather than asked to stop used to lose a completed run entirely.
    out3 = Path(tempfile.mkdtemp())
    rec3 = rt.Recorder.__new__(rt.Recorder)
    rec3.out, rec3.n, rec3.t0, rec3.cloud, rec3.path, rec3.seen, rec3.diagnostics = out3, 0, 0.0, None, None, 0, {}
    rec3.fh = (out3 / "odometry.tum").open("w")
    rec3.on_path(types.SimpleNamespace(poses=[stamped(i, i * 2.0) for i in range(4)]))
    rows = [l.split() for l in (out3 / "trajectory.tum").read_text().splitlines()
            if not l.startswith("#")]
    assert [float(r[1]) for r in rows] == [0.0, 2.0, 4.0, 6.0], rows
    assert not (out3 / "trajectory.tum.tmp").exists(), "left a temporary file behind"
    # a later, better graph replaces it wholesale
    rec3.on_path(types.SimpleNamespace(poses=[stamped(0, 99.0)]))
    rows = [l for l in (out3 / "trajectory.tum").read_text().splitlines()
            if not l.startswith("#")]
    assert len(rows) == 1 and rows[0].split()[1] == "99.000000", rows

    # and with no graph the odometry IS the estimate -- both files, same content
    out2 = Path(tempfile.mkdtemp())
    rec2 = rt.Recorder.__new__(rt.Recorder)
    rec2.out, rec2.n, rec2.t0, rec2.cloud, rec2.path, rec2.seen, rec2.diagnostics = out2, 0, 0.0, None, None, 0, {}
    rec2.fh = (out2 / "odometry.tum").open("w")
    rec2.fh.write(rt.HEADER)
    rec2.on_pose(stamped(0, 7.0))
    rec2.finish()
    assert (out2 / "trajectory.tum").read_text() == (out2 / "odometry.tum").read_text()
    assert json.loads((out2 / "timing.json").read_text())["trajectory_source"] == "odometry"


@test
def test_lost_tracking_is_measured_as_stretches_not_a_count():
    """Scattered zeros are frames a tracker rides through; one long stretch is
    where the trajectory leaves the room. Same distinction depth_health.py draws
    for the sensor, applied to the estimator."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from odom_health import runs_of_zero
    q = np.array([5, 0, 7, 0, 0, 0, 9])
    assert runs_of_zero(q) == [(1, 1), (3, 3)], runs_of_zero(q)
    assert runs_of_zero(np.array([0, 0, 3])) == [(0, 2)]        # starts lost
    assert runs_of_zero(np.array([3, 0, 0])) == [(1, 2)]        # never recovers
    assert runs_of_zero(np.array([1, 2, 3])) == []
    assert runs_of_zero(np.array([0, 0])) == [(0, 2)]
    # the two shapes that a bare zero-count cannot tell apart
    scattered = np.array([1, 0] * 10)
    one_gap = np.array([1] * 10 + [0] * 10)
    assert int(np.sum(scattered == 0)) == int(np.sum(one_gap == 0))
    assert len(runs_of_zero(scattered)) == 10
    assert len(runs_of_zero(one_gap)) == 1


@test
def test_stream_coverage_is_measured_at_both_ends_not_by_span():
    """Two spans side by side say nothing about alignment. A 152 s IMU and a
    156 s image stream could overlap perfectly for 152 s or miss by 4 s at
    either end, and only the ends distinguish them -- which matters because a
    front-end that needs an inertial sample newer than each frame drops every
    image past the IMU's last one."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from bag_probe import coverage_gap
    img = np.arange(0.0, 156.0, 1 / 14.9)
    imu = np.arange(0.0, 152.35, 1 / 201.0)          # same start, stops early
    late, early = coverage_gap(imu, img)
    assert abs(late) < 0.01 and 3.5 < early < 4.0, (late, early)
    # identical coverage -> no gap at either end
    assert all(abs(v) < 0.01 for v in coverage_gap(img.copy(), img))
    # short at the head instead
    late, early = coverage_gap(img[img > 5.0], img)
    assert late > 4.9 and abs(early) < 0.1, (late, early)


@test
def test_no_node_redeclares_use_sim_time():
    """rclpy's Node constructor already declares `use_sim_time` (TimeSource
    .attach_node), so declaring it again raises ParameterAlreadyDeclaredException
    and kills the node on construction. Backgrounded with `&`, that failure is
    invisible: every run replayed 156 s of bag into a dead recorder and produced
    a database with no trajectory. Sim time is switched on by a parameter
    OVERRIDE at launch instead, which the entrypoint must pass."""
    import ast as _ast
    common = Path(__file__).resolve().parents[1] / "docker" / "common"
    for py in sorted(common.glob("*.py")):
        # PARSED, not grepped. The fix leaves a comment explaining why the call
        # is absent, and that comment naturally contains the call's own text --
        # a substring check matches it and the test passes for the wrong reason,
        # which is worse than not having the test.
        for node in _ast.walk(_ast.parse(py.read_text())):
            if not isinstance(node, _ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name != "declare_parameter" or not node.args:
                continue
            first = node.args[0]
            assert not (isinstance(first, _ast.Constant)
                        and first.value == "use_sim_time"), f"{py.name}:{node.lineno}"

    entry = (common / "entrypoint.sh").read_text()
    for node in ("record_tum.py", "depth_to_cloud.py"):
        # Anchored to the INVOCATION. Both names also appear in error messages
        # earlier in the file, and matching one of those checks nothing.
        marker = f"python3 /opt/slambench/{node}"
        assert marker in entry, f"{node} is never launched"
        block = entry[entry.index(marker):].split("\n\n")[0]
        assert "use_sim_time:=true" in block, f"{node} is launched on wall time"


@test
def test_the_entrypoint_refuses_to_replay_into_a_dead_recorder():
    """A backgrounded process that dies on its first line leaves `&` perfectly
    happy. Without a check the bag replays for three minutes and the failure is
    only visible at the end, as an empty directory."""
    entry = (Path(__file__).resolve().parents[1]
             / "docker" / "common" / "entrypoint.sh").read_text()
    check = entry[entry.index("REC=$!"):entry.index("ros2 bag play")]
    assert 'kill -0 "$REC"' in check, "the recorder's liveness is never checked"
    assert "odometry.tum" in check, "nothing checks that it opened its output"
    assert "exit 4" in check, "a dead recorder does not stop the run"


@test
def test_recorder_hands_ros_args_to_rclpy_instead_of_rejecting_them():
    """The entrypoint appends `--ros-args -p use_sim_time:=true`, the only way
    to put the node on the bag's clock. A plain parse_args() exits 2 on those,
    which is a crash on the first line -- backgrounded, invisible, and the
    second silent death of this recorder in one afternoon."""
    rt = _import_record_tum()
    argv = ["record_tum.py", "--pose-topic", "/rtabmap/odom", "--path-topic",
            "/rtabmap/mapPath", "--out", "/out",
            "--ros-args", "-p", "use_sim_time:=true"]
    args, ros_argv = rt.parse(argv)
    assert args.pose_topic == "/rtabmap/odom" and args.out == "/out"
    assert args.path_topic == "/rtabmap/mapPath"
    # everything ROS needs, in order, with argv[0] in front as rclpy expects
    assert ros_argv == ["record_tum.py", "--ros-args", "-p", "use_sim_time:=true"], ros_argv
    # and with no ROS flags at all, rclpy still gets a well-formed argv
    _, plain = rt.parse(argv[:7])
    assert plain == ["record_tum.py"], plain


@test
def test_recorder_subscribes_to_the_pose_topic_exactly_once():
    """ROS 2 forbids two subscriptions on one topic with different types in one
    node. The original recorder subscribed three times -- Odometry, PoseStamped,
    PoseWithCovariance -- so that whichever the method published would land, and
    the second create_subscription() threw on every run from the first day."""
    import ast as _ast
    src = (Path(__file__).resolve().parents[1] / "docker" / "common" / "record_tum.py").read_text()
    tree = _ast.parse(src)
    pose_subs, in_loop = 0, 0
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.For, _ast.While)):
            for inner in _ast.walk(node):
                if (isinstance(inner, _ast.Call)
                        and getattr(inner.func, "attr", "") == "create_subscription"):
                    in_loop += 1
        if (isinstance(node, _ast.Call)
                and getattr(node.func, "attr", "") == "create_subscription"
                and len(node.args) >= 2
                and getattr(node.args[1], "id", "") == "pose_topic"):
            pose_subs += 1
    assert pose_subs == 1, f"{pose_subs} subscriptions on pose_topic"
    assert in_loop == 0, "a subscription inside a loop is the three-types trick again"

    # and the type is declared, not guessed: the image passes it through
    rt = _import_record_tum()
    args, _ = rt.parse(["x", "--pose-topic", "/t", "--out", "/o",
                        "--pose-type", "geometry_msgs/msg/PoseStamped"])
    assert args.pose_type == "geometry_msgs/msg/PoseStamped"
    entry = (Path(__file__).resolve().parents[1] / "docker" / "common" / "entrypoint.sh").read_text()
    assert "--pose-type" in entry and "SLAM_POSE_TYPE" in entry
    for df in ("Dockerfile.rtabmap",):                 # kiss-icp is offline: no recorder
        text = (Path(__file__).resolve().parents[1] / "docker" / df).read_text()
        assert "SLAM_POSE_TYPE=" in text, f"{df} does not declare its pose type"


@test
def test_a_zeroed_pose_is_a_lost_frame_not_a_corrupt_file():
    """RTAB-Map keeps publishing Odometry while lost, with the pose zeroed. Ten
    such rows crashed the evaluator on a run that was 99.4% tracked. They are
    skipped and counted, never silently dropped: a lost frame is a result."""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.tum"
        f.write_text("# h\n"
                     "1.0 0 0 0 0 0 0 1\n"
                     "2.0 0 0 0 0 0 0 0\n"          # lost
                     "3.0 1 0 0 0 0 0 1\n"
                     "4.0 0 0 0 0 0 0 0\n"          # lost
                     "5.0 2 0 0 0 0 0 1\n")
        t = load_tum(f)
        assert len(t) == 3 and t.lost == 2, (len(t), t.lost)
        assert list(t.stamps) == [1.0, 3.0, 5.0]
        # every row lost is still an error, with the count in the message
        f.write_text("1.0 0 0 0 0 0 0 0\n")
        try:
            load_tum(f)
            raise AssertionError("an all-lost file loaded")
        except ValueError as e:
            assert "all lost" in str(e), e


@test
def test_replay_rate_comes_from_the_method_config_and_is_recorded():
    """An offline backbone replays at half speed because the machine cannot keep
    up in real time; that is declared once in the method config (rule 7), lands
    in the docker command, and is written to run.json so a mixed table is
    visible. The CLI can still override it."""
    import io, contextlib
    import scripts.run_method as rm
    root = str(Path(__file__).resolve().parents[1])

    def run(extra):
        saved = sys.argv
        try:
            sys.argv = ["run_method.py",
                        "--config", os.path.join(root, "configs", "coop2.yaml"),
                        "--method", os.path.join(root, "configs", "methods", "rtabmap_rgbd_imu.yaml"),
                        "--stream", "mobile_1.zed_rgbd", "--runs-root", tempfile.mkdtemp()] + extra
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                assert rm.main() == 0
            out = buf.getvalue()
            manifest = json.loads(Path(out.split("manifest -> ", 1)[1].splitlines()[0]).read_text())
            return out, manifest
        finally:
            sys.argv = saved

    out, m = run([])
    assert "SLAM_RATE=0.5" in out and m["replay_rate"] == 0.5, (m["replay_rate"])
    out, m = run(["--rate", "1.0"])
    assert "SLAM_RATE=1.0" in out and m["replay_rate"] == 1.0
    # --dev mounts every script the image ships, post.sh included
    out, _ = run(["--dev"])
    assert "docker/rtabmap/post.sh:/opt/slambench/post.sh" in out, out[:300]
    assert "docker/rtabmap/launch.sh:/opt/slambench/launch.sh" in out


@test
def test_rtabmap_odometry_processes_every_frame_not_the_newest():
    """always_process_most_recent_frame=true drops any frame that arrives while
    the worker holds dataMutex_, silently. Measured 72% kept alone and ~40% with
    the mapping node subscribed to odom_info, identical at two replay rates.
    An offline backbone must see every frame."""
    launch = (Path(__file__).resolve().parents[1] / "docker" / "rtabmap" / "launch.sh").read_text()
    assert "odom_always_process_most_recent_frame:=false" in launch
    assert "sync_queue_size:=50" in launch and "topic_queue_size:=50" in launch
    # The actual drop mechanism: wait_imu_to_init parks frames whose stamp is
    # ahead of the newest IMU sample, and the parked frame's worker-thread
    # processing races the next image into a silent lockTry() drop. The bag
    # delivered 2291 frames; the node processed ~720 with it on.
    assert "wait_imu_to_init:=true" in launch      # off made the tracking worse, not the drops better
    assert "subscribe_odom_info:=false" in launch
    # THE frame-drop fix: qos:=0 is SYSTEM_DEFAULT, which rmw_fastrtps resolves
    # to a BEST_EFFORT reader. The recorder's RELIABLE witness saw 2291 frames
    # while the odometry saw a third, with its own drop counter at zero.
    assert "qos:=1" in launch


@test
def test_every_run_keeps_its_own_directory():
    """Two identical RTAB-Map runs scored 365 mm and 451 mm; the second erased
    the first. A method whose answer varies between runs needs every run kept."""
    import io, contextlib
    import scripts.run_method as rm
    root = str(Path(__file__).resolve().parents[1])
    runs = tempfile.mkdtemp()
    manifests = []
    for _ in range(2):
        saved = sys.argv
        try:
            sys.argv = ["run_method.py",
                        "--config", os.path.join(root, "configs", "coop2.yaml"),
                        "--method", os.path.join(root, "configs", "methods", "rtabmap_rgbd.yaml"),
                        "--stream", "mobile_1.zed_rgbd", "--runs-root", runs]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                assert rm.main() == 0
            manifests.append(Path(buf.getvalue().split("manifest -> ", 1)[1].splitlines()[0]))
        finally:
            sys.argv = saved
        import time as _t; _t.sleep(1.1)          # distinct second-resolution stamps
    assert manifests[0].exists() and manifests[1].exists()
    assert manifests[0].parent != manifests[1].parent, "the second run overwrote the first"
    cell = manifests[0].parent.parent
    assert (cell / "latest").is_symlink()
    assert (cell / "latest").resolve() == manifests[1].parent.resolve()


@test
def test_spread_scores_every_run_and_refuses_a_spread_below_three():
    """A cell is median [min, max] over N runs; two runs are two samples, not a
    spread. Interrupted runs and runs with no trajectory are listed, not scored,
    and `latest` (a symlink) is never scored twice."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from spread import score, summarise, run_dirs, KEYS, MIN_RUNS
    cell = Path(tempfile.mkdtemp())
    for i, val in enumerate((0.40, 0.45, 0.38)):
        d = cell / f"r{i}"; d.mkdir()
        (d / "trajectory.tum").write_text(f"{val}\n")
    (cell / "r3").mkdir(); (cell / "r3" / "INTERRUPTED").write_text("")
    (cell / "r4").mkdir()                                     # crashed before writing
    (cell / "latest").symlink_to("r2")
    assert [d.name for d in run_dirs(cell)] == ["r0", "r1", "r2", "r3", "r4"]

    def scorer(path):
        v = float(path.read_text())
        return {"poses": 1, "lost": 0, "rmse_mm": v * 1e3, "p90_mm": v * 1.2e3,
                "rot_deg": 5.0, "scale": 1.05, "drift_pct": 1.5}
    scored, skipped = score(cell, "trajectory.tum", scorer)
    assert [s["run"] for s in scored] == ["r0", "r1", "r2"]
    assert skipped == [("r3", "interrupted"), ("r4", "no trajectory.tum")], skipped
    summ = summarise(scored, KEYS)
    assert summ["n"] == 3 and MIN_RUNS == 3
    assert abs(summ["rmse_mm"]["median"] - 400.0) < 1e-9
    assert abs(summ["rmse_mm"]["min"] - 380.0) < 1e-9 and abs(summ["rmse_mm"]["max"] - 450.0) < 1e-9
    assert summarise(scored[:2], KEYS) is None
    # the front-end alone is a different file: skipped when absent, scored when present
    _, sk = score(cell, "odometry.tum", scorer)
    assert all(why.startswith("no odometry.tum") or why == "interrupted" for _, why in sk)


@test
def test_scoring_the_front_end_does_not_overwrite_the_graph_score():
    """`--trajectory-file odometry.tum` used to write the same metrics.json as
    the graph's score; aggregate.py then read the front-end as the row."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from eval_run import metrics_name
    assert metrics_name("trajectory.tum") == "metrics.json"
    assert metrics_name("odometry.tum") == "metrics_odometry.json"
    assert metrics_name("trajectory_mappath.tum") == "metrics_trajectory_mappath.json"


@test
def test_kiss_offline_runner_refuses_unknown_params_and_overrides_deskew_on_depth():
    """The library ignores a key it does not know; the manifest would then claim
    a parameter the run never had. Unknown keys are refused. On a depth stream
    deskew is forced off and reported, since a depth image has no per-point
    time. voxel_size absent -> the library's own rule, max_range/100."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker" / "kiss-icp"))
    import run_offline as ro

    class NS:                                   # a KISSConfig-shaped stand-in
        def __init__(self, **kw): self.__dict__.update(kw)
    def cfg():
        return NS(data=NS(max_range=100.0, min_range=5.0, deskew=False),
                  mapping=NS(voxel_size=None, max_points_per_voxel=20),
                  registration=NS(max_num_iterations=500, convergence_criterion=1e-4),
                  adaptive_threshold=NS(fixed_threshold=None, initial_threshold=2.0, min_motion_th=0.1))
    declared = {"max_range": 20.0, "min_range": 0.3, "voxel_size": 0.10, "deskew": True,
                "initial_threshold": 0.5, "max_points_per_voxel": 20}
    c = cfg(); ov = ro.apply_params(c, declared, "rgbd")
    assert c.data.max_range == 20.0 and c.mapping.voxel_size == 0.10
    assert c.adaptive_threshold.initial_threshold == 0.5
    assert c.data.deskew is False and ov == {"deskew": False}
    c = cfg(); ov = ro.apply_params(c, declared, "lidar")
    assert c.data.deskew is True and ov == {}
    c = cfg(); ro.apply_params(c, {"max_range": 20.0}, "lidar")
    assert abs(c.mapping.voxel_size - 0.2) < 1e-12
    try:
        ro.apply_params(cfg(), {"voxel": 0.1}, "rgbd")
        assert False, "unknown key accepted"
    except SystemExit as e:
        assert "voxel" in str(e)
    # every key the method config declares is one the runner takes
    import yaml
    mp = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "methods" / "kiss_icp.yaml").read_text())
    assert set(mp["params"]) <= set(ro.PARAM_MAP), set(mp["params"]) - set(ro.PARAM_MAP)


@test
def test_kiss_offline_runner_helpers():
    """Stamps normalised to [0,1] for the deskew (mid-pose 0.5); the pose row
    round-trips through the evaluator's loader; per-frame stats keep the stalls."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker" / "kiss-icp"))
    import run_offline as ro
    t = ro.normalise_stamps(np.array([10.0, 12.0, 14.0]))
    assert np.allclose(t, [0.0, 0.5, 1.0])
    assert np.allclose(ro.normalise_stamps(np.array([3.0, 3.0])), [0.5, 0.5])
    assert ro.normalise_stamps(np.array([])).size == 0
    T = se3.pose_from_rpy(1.0, -2.0, 0.5, 12.0, -34.0, 156.0)   # a big rotation, every branch
    for R in (T, se3.pose_from_rpy(0, 0, 0, 170.0, 5.0, 0.0), se3.pose_from_rpy(0, 0, 0, 5.0, 170.0, 0.0),
              se3.pose_from_rpy(0, 0, 0, 0.0, 5.0, 175.0)):
        q = ro.quat_xyzw(R[:3, :3])
        assert q[3] >= 0 and abs(np.linalg.norm(q) - 1) < 1e-12
        assert np.allclose(se3.quat_to_rot(q), R[:3, :3], atol=1e-9), R
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.tum"
        f.write_text("# h\n" + ro.pose_row(1700000000.5, T))
        back = load_tum(f)
    assert np.allclose(back.poses[0], T, atol=1e-5)
    assert abs(back.stamps[0] - 1700000000.5) < 1e-6
    st = ro.frame_stats([0.05] * 90 + [2.0] * 8 + [30.0, 65.0])
    assert st["n"] == 100 and st["over_5s"] == 2 and st["over_1s"] == 10
    assert abs(st["max"] - 65.0) < 1e-12 and abs(st["median"] - 0.05) < 1e-12
    assert ro.frame_stats([]) == {"n": 0}


@test
def test_kiss_image_is_offline_and_dev_mounts_its_runner():
    """No launch.sh, no pose topic: the image replaces the entrypoint like
    MASt3R's does. --dev mounts docker/kiss-icp/*.py too, or a fix to the
    runner would need a rebuild to be tested."""
    root = Path(__file__).resolve().parents[1]
    df = (root / "docker" / "Dockerfile.kiss-icp").read_text()
    assert 'ENTRYPOINT ["/opt/slambench/run.sh"]' in df
    assert "SLAM_POSE_TOPIC" not in df and "colcon" not in df
    assert "KISS_ICP_REF=v1.0.0" in df
    # the backend is pinned with the source: v1.0.0's pyproject fails under
    # scikit-build-core >= 0.8 (cmake.minimum-version was renamed)
    assert '"scikit-build-core[pyproject]<0.8"' in df and "--no-build-isolation" in df
    assert "import pyproject_metadata" in df          # proven before the compile starts
    assert not (root / "docker" / "kiss-icp" / "launch.sh").exists()
    assert (root / "docker" / "kiss-icp" / "run_offline.py").exists()
    rm = (root / "scripts" / "run_method.py").read_text()
    assert 'glob("*.py")' in rm.split("idir")[-1]
    from slambench.config import load_method
    m = load_method(root / "configs" / "methods" / "kiss_icp.yaml")
    assert "replay_rate" not in m.raw, "an offline run has no replay rate"
    assert m.outputs == ["trajectory", "map"]


@test
def test_mast3r_image_upgrades_setuptools_and_builds_without_isolation():
    """thirdparty/mast3r/setup.py needs an absolute __file__ (setuptools >= 61);
    the curope CUDA extension imports torch at build time, which an isolated
    build environment does not have. Both are one line in the Dockerfile and
    both cost a 20-minute build to discover."""
    import re
    df = (Path(__file__).resolve().parents[1] / "docker" / "Dockerfile.mast3r-slam").read_text()
    up = df.index("pip3 install --no-cache-dir -U pip setuptools wheel")
    assert up < df.index("-e thirdparty/mast3r"), "setuptools must be upgraded before the source installs"
    installs = re.findall(r"pip3 install [^\n]*-e (thirdparty/mast3r|thirdparty/in3d|\.)", df)
    assert sorted(installs) == [".", "thirdparty/in3d", "thirdparty/mast3r"], installs
    for line in re.findall(r"pip3 install [^\n]*-e [^\n]*", df):
        assert "--no-build-isolation" in line, line
    # setup.py asks torch for a GPU at build time; patched before the install
    assert df.index("patch_setup.py setup.py") < df.index("-e thirdparty/mast3r")
    assert "PIP_CONSTRAINT=" in df and df.index("PIP_CONSTRAINT=") < df.index("torch==")
    # the patch itself refuses a source it was not written for, and does its two edits
    patch = Path(__file__).resolve().parents[1] / "docker" / "mast3r-slam" / "patch_setup.py"
    src = Path(tempfile.mkdtemp()) / "setup.py"
    src.write_text('has_cuda = torch.cuda.is_available()\n'
                   '        "-gencode=arch=compute_86,code=sm_86",\n')
    import subprocess
    r = subprocess.run([sys.executable, str(patch), str(src)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = src.read_text()
    assert "has_cuda = True" in out and "compute_90,code=compute_90" in out and "sm_89" in out
    # the full opencv-python links libGL; the image proves the imports it ships
    assert "libgl1" in df and "libusb-1.0-0" in df and "check_imports.py main.py" in df
    # the checker walks a file's imports and names every one that fails
    chk = Path(__file__).resolve().parents[1] / "docker" / "mast3r-slam" / "check_imports.py"
    f = Path(tempfile.mkdtemp()) / "main.py"
    f.write_text("import json\nfrom pathlib import Path\nimport no_such_module_xyz\n")
    r = subprocess.run([sys.executable, str(chk), str(f)], capture_output=True, text=True)
    assert r.returncode != 0 and "no_such_module_xyz" in r.stderr, r.stderr
    f.write_text("import json\nfrom pathlib import Path\n")
    r = subprocess.run([sys.executable, str(chk), str(f)], capture_output=True, text=True)
    assert r.returncode == 0 and "2 imports ok" in r.stdout, r.stdout + r.stderr
    src.write_text("has_cuda = something_else\n")
    r = subprocess.run([sys.executable, str(patch), str(src)], capture_output=True, text=True)
    assert r.returncode != 0 and "not what this patch was written for" in r.stderr
    # in3d pulls the vendored pyimgui from a git checkout that holds only .pyx
    # sources; without Cython its setup.py compiles against a core.h that was
    # never generated. --no-build-isolation skips pyimgui's own build requires,
    # so Cython must be installed by hand, before in3d, at pyimgui's pin.
    cy = re.search(r'pip3 install [^\n]*"Cython>=0.24,<0.30"', df)
    assert cy, "Cython must be installed at pyimgui's pin (>=0.24,<0.30)"
    assert cy.start() < df.index("-e thirdparty/in3d"), "Cython must precede the in3d install"


@test
def test_register_depth_moves_points_into_the_colour_camera_nearest_wins():
    """Identity cameras: the output is the input. A colour camera 10 cm to the
    +x of the depth camera sees a 2 m plane shifted by fx*0.1/2 = 25 px. Two
    points on one output pixel: the nearer one survives. Half resolution
    scales the colour intrinsics by exactly 1/2, as RTAB-Map does."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker" / "common"))
    from register_depth import register, output_shape, quat_to_rot
    K = (500.0, 500.0, 320.0, 240.0)
    d = np.full((480, 640), 2.0, np.float32)
    out = register(d, K, K, np.eye(4), (480, 640), 1)
    assert np.allclose(out, 2.0)
    T = np.eye(4); T[0, 3] = 0.1
    out = register(d, K, K, T, (480, 640), 1)
    cols = np.nonzero(out[240] > 0)[0]
    assert cols[0] == 25 and cols[-1] == 639, (cols[0], cols[-1])
    assert (out[:, :25] == 0).all()                      # no depth there; holes stay 0
    # occlusion: a near pixel and a far pixel that project to the same place
    d2 = np.zeros((480, 640), np.float32)
    d2[240, 320] = 1.0                                   # at the centre, 1 m
    d2[240, 330] = 2.0                                   # 10 px right, 2 m: x = 10*2/500 = 0.04 m
    # move the colour camera so the 2 m point lands on the centre pixel too:
    # u = fx*(x + tx)/z + cx -> tx = -0.04 puts the 2 m point at u=320; the 1 m
    # point then lands at u = 500*(-0.04)/1 + 320 = 300.
    T2 = np.eye(4); T2[0, 3] = -0.04
    out = register(d2, K, K, T2, (480, 640), 1)
    assert abs(out[240, 320] - 2.0) < 1e-6 and abs(out[240, 300] - 1.0) < 1e-6
    # now put both on one pixel: a second 1 m point at u where 1 m maps to 320
    d3 = d2.copy(); d3[240, 340] = 1.0                   # x = 20*1/500 = 0.04 -> u = 320 after tx
    out = register(d3, K, K, T2, (480, 640), 1)
    assert abs(out[240, 320] - 1.0) < 1e-6, "the nearer point must win the z-buffer"
    h = register(d, K, K, np.eye(4), output_shape((480, 640), 2), 2)
    assert h.shape == (240, 320) and (h > 0).mean() == 1.0
    try:
        output_shape((720, 1280), 7); assert False
    except ValueError:
        pass
    assert np.allclose(quat_to_rot(0, 0, 0, 1), np.eye(3))
    assert np.allclose(quat_to_rot(0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)),
                       se3.pose_from_rpy(0, 0, 0, 0, 0, 90.0)[:3, :3])


@test
def test_rtabmap_on_the_realsense_registers_depth_and_declares_the_colour_frame():
    """The RealSense depth is not registered to its colour image; the ZED's is.
    run_method passes the registration only where the frames differ, records
    the pose frame, and refuses when the method has not declared the identity
    extrinsic for that stream -- the evaluator would otherwise move colour-frame
    poses by the depth-frame lever arm."""
    import io, contextlib, yaml
    import scripts.run_method as rm
    root = Path(__file__).resolve().parents[1]
    def run(method, stream):
        saved = sys.argv; buf = io.StringIO()
        try:
            sys.argv = ["run_method.py", "--config", str(root / "configs" / "coop2.yaml"),
                        "--method", str(method), "--stream", stream, "--runs-root", tempfile.mkdtemp()]
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                rc = rm.main()
        finally:
            sys.argv = saved
        text = buf.getvalue()
        m = None
        if "manifest -> " in text:
            m = json.loads(Path(text.split("manifest -> ", 1)[1].splitlines()[0]).read_text())
        return rc, text, m
    rc, text, m = run(root / "configs" / "methods" / "rtabmap_rgbd.yaml", "mobile_2.realsense_rgbd")
    assert rc == 0, text
    assert "SLAM_DEPTH_REGISTER=1" in m["command"] and "SLAM_COLOR_FRAME=camera_color_optical_frame" in m["command"]
    assert "SLAM_COLOR_FROM_DEPTH=" in m["command"]
    assert m["depth_registered_in_container"] is True and m["pose_frame"] == "camera_color_optical_frame"
    rc, text, m = run(root / "configs" / "methods" / "rtabmap_rgbd.yaml", "mobile_1.zed_rgbd")
    assert rc == 0 and "SLAM_DEPTH_REGISTER" not in m["command"]
    assert m["depth_registered_in_container"] is False and m["pose_frame"] == "zed_left_camera_optical_frame"
    # a method that needs registration but has not declared where its poses end up
    raw = yaml.safe_load((root / "configs" / "methods" / "rtabmap_rgbd.yaml").read_text())
    del raw["extrinsic_by_stream"]
    bad = Path(tempfile.mkdtemp()) / "rtabmap_rgbd.yaml"
    bad.write_text(yaml.safe_dump(raw))
    rc, text, _ = run(bad, "mobile_2.realsense_rgbd")
    assert rc == 1 and "extrinsic_by_stream" in text, text
    # and the evaluator will apply the identity, not the 59 mm depth->colour edge
    from slambench.config import load_dataset, load_method
    cfg = load_dataset(root / "configs" / "coop2.yaml")
    E = load_method(root / "configs" / "methods" / "rtabmap_rgbd.yaml").sensor_extrinsic(cfg, "mobile_2.realsense_rgbd")
    assert np.allclose(E, np.eye(4))
    E = load_method(root / "configs" / "methods" / "kiss_icp.yaml").sensor_extrinsic(cfg, "mobile_2.realsense_rgbd")
    assert abs(E[0, 3] - 0.059190) < 1e-9                # kiss reads the raw depth: still the depth frame
    # the entrypoint stops the bridge and counts the bag's depth, not the bridge's
    entry = (root / "docker" / "common" / "entrypoint.sh").read_text()
    assert 'register_depth.py' in entry and '"$REG"' in entry
    assert 'SLAM_WITNESS_TOPIC' in entry
    assert "register_depth.py" in (root / "docker" / "Dockerfile.base").read_text()
    launch = (root / "docker" / "rtabmap" / "launch.sh").read_text()
    assert 'FRAME_ID="$SLAM_FRAME_ID"' in launch


@test
def test_replay_rate_can_be_declared_per_stream():
    """One method, two frame rates: the RealSense delivers 27.5 Hz to the same
    front-end the ZED feeds at 15 Hz. The rate is per stream where declared,
    the method's rate otherwise, and run.json records what was used."""
    import io, contextlib
    import scripts.run_method as rm
    root = Path(__file__).resolve().parents[1]
    def rate_for(stream):
        saved = sys.argv; buf = io.StringIO()
        try:
            sys.argv = ["run_method.py", "--config", str(root / "configs" / "coop2.yaml"),
                        "--method", str(root / "configs" / "methods" / "rtabmap_rgbd.yaml"),
                        "--stream", stream, "--runs-root", tempfile.mkdtemp()]
            with contextlib.redirect_stdout(buf):
                assert rm.main() == 0
        finally:
            sys.argv = saved
        text = buf.getvalue()
        return json.loads(Path(text.split("manifest -> ", 1)[1].splitlines()[0]).read_text())["replay_rate"]
    assert rate_for("mobile_2.realsense_rgbd") == 0.25
    assert rate_for("mobile_1.zed_rgbd") == 0.5


@test
def test_anchor_check_evaluates_a_keyframe_trajectory_between_its_poses():
    """MASt3R emits 38 poses over 156 s. A board-derived pose at 15 Hz lands
    within 50 ms of one about never, so every anchor row came back empty and
    the sparse method scored nothing at the only independent tier. With
    interpolate it is evaluated between poses, and the row says how far the
    nearest real pose was."""
    # board poses offset from the keyframe grid, as a real 15 Hz detector is
    dense = circle_traj(400, radius=3.0, t0=100.0, rate=40.0)          # 25 ms apart
    sparse = Trajectory(dense.stamps[::40], dense.poses[::40], "kf")   # 1 s apart
    k = np.array([i for i in range(20, 360, 7) if 5 <= i % 40 <= 35])  # never near one
    board = Trajectory(dense.stamps[k] + 0.003, dense.poses[k], "board")
    anchor = [{"name": "a", "position": [0, 0, 0], "uncertainty_m": 0.007,
               "window": [float(board.stamps[0]), float(board.stamps[-1])],
               "observed_poses": board}]
    rows = absolute_check(sparse, anchor)
    assert rows[0]["n"] == 0 and "interpolate" in rows[0]["verdict"], rows[0]
    rows = absolute_check(sparse, anchor, interpolate=True)
    r = rows[0]
    assert r["n"] >= 15, r
    assert r["interp_gap_s"] > 0.05, r          # it says the gap it worked across
    # a chord across a 1 s arc is not the arc: real, bounded by the sagitta
    # r(1-cos(18 deg)) = 147 mm for this circle, and the row prints the gap
    assert 0.0 < r["residual_m"] < 0.16, r
    # a DENSE estimate still matches exactly and reports no interpolation
    r2 = absolute_check(dense, anchor, interpolate=True)[0]
    assert r2["residual_m"] < 1e-6 and "interp_gap_s" not in r2, r2


@test
def test_a_scale_free_method_gets_its_anchors_from_the_sim3_fit_and_says_so():
    """The machinery, not any one method: a row declaring `metric_scale: false`
    is placed by an se3 fit as a scale model of the room, and its distance to a
    surveyed board is the scale error in metres against a 7 mm survey. Such a
    row gets sim3 anchors and is marked not independent. NOTE mast3r_slam is
    NOT such a row -- it runs the *_metric.pth checkpoint and its scale is a
    measurement (1.0144 on mobile_2, 0.729 on mobile_1), not a free parameter."""
    import yaml
    root = Path(__file__).resolve().parents[1]
    ev = (root / "scripts" / "eval_run.py").read_text()
    assert 'mcfg.raw.get("metric_scale", True)' in ev
    assert "interpolate=True" in ev, "the anchor check must not depend on a method declaring sparsity"
    # every shipped method claims metric scale; none may claim otherwise silently
    for f in sorted((root / "configs" / "methods").glob("*.yaml")):
        r = yaml.safe_load(f.read_text()) or {}
        assert r.get("metric_scale", True) is True, (
            f"{f.name} declares metric_scale: false -- if that is right, say WHY "
            f"in the file, and check it is not a metric checkpoint like "
            f"mast3r_slam's *_metric.pth")
    df = (root / "docker" / "Dockerfile.mast3r-slam").read_text()
    assert "_metric.pth" in df, "the mast3r image must fetch the metric checkpoint"
    # a 27%-small estimate: se3 leaves metres of anchor residual, sim3 removes it
    ref = circle_traj(200, radius=3.0)
    small = Trajectory(ref.stamps, np.array([se3.pose_from_rpy(*(p[:3, 3] * 0.729),
                                                               0, 0, 0) for p in ref.poses]), "s")
    a_se3 = ate(small, ref, mode="se3")
    a_sim3 = ate(small, ref, mode="sim3")
    assert a_se3.trans.rmse > 0.5 and a_sim3.trans.rmse < 1e-6
    assert abs(a_sim3.alignment.scale_observed - 1 / 0.729) < 1e-6 or \
           abs(a_sim3.alignment.scale_observed - 0.729) < 1e-6


@test
def test_board_table_collects_tier_two_across_methods():
    """The board residual is the only number here whose error bar is not the
    reference's own. It has to be readable per method, and a row whose scale
    came from the reference has to be marked, or it reads as independent."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from board_table import collect, render
    root = Path(tempfile.mkdtemp())
    def write(sub, doc):
        d = root / sub; d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps(doc))
    write("rtabmap/m1/run1", {
        "method": "rtabmap_rgbd_imu", "agent": "mobile_1", "stream": "s", "run_dir": "run1",
        "anchor_alignment": {"mode": "se3", "independent": True},
        "absolute_check": [{"anchor": "anchor", "n": 90, "residual_m": 0.147,
                            "residual_deg": 1.2, "uncertainty_m": 0.007,
                            "verdict": "resolved above the survey's uncertainty"}]})
    write("mast3r/m1/run1", {
        "method": "mast3r_slam", "agent": "mobile_1", "stream": "s", "run_dir": "run1",
        "anchor_alignment": {"mode": "sim3", "independent": False,
                             "note": "scale taken from the reference"},
        "absolute_check": [{"anchor": "anchor", "n": 12, "residual_m": 0.091,
                            "residual_deg": 0.9, "uncertainty_m": 0.007,
                            "interp_gap_s": 1.8, "verdict": "resolved"}]})
    write("kiss/m2/run1", {
        "method": "kiss_icp", "agent": "mobile_2", "stream": "s", "run_dir": "run1",
        "anchor_alignment": {"mode": "se3", "independent": True},
        "absolute_check": [{"anchor": "rs_anchor", "n": 0, "residual_m": None,
                            "uncertainty_m": 0.015,
                            "verdict": "no board-derived pose inside the window"}]})
    rows = collect(root)
    assert len(rows) == 3
    assert {r["method"] for r in rows} == {"rtabmap_rgbd_imu", "mast3r_slam", "kiss_icp"}
    assert collect(root, agent="mobile_1") and len(collect(root, agent="mobile_2")) == 1
    text = render(rows)
    assert "147.0" in text and "91.0" in text
    assert "NOT independent" in text                  # the monocular row is marked
    assert "interpolated across 1.80 s" in text
    assert "no board-derived pose inside the window" in text
    # a metrics.json from before the fixes is marked, not silently compared
    write("old/m1/run1", {"method": "old_run", "agent": "mobile_1", "stream": "s",
                          "run_dir": "run1",
                          "absolute_check": [{"anchor": "anchor", "n": 62,
                                              "residual_m": 0.499, "uncertainty_m": 0.007,
                                              "verdict": "resolved"}]})
    text = render(collect(root))
    assert "STALE" in text and "re-run eval_run.py" in text
    assert "no absolute_check" in render([])


@test
def test_a_missing_trajectory_names_itself_and_the_tool_that_makes_it():
    """runs/reference/mobile_2.tum was never exported; the loader raised a bare
    FileNotFoundError from inside a rescore loop, so an entire agent was absent
    from the tier-2 table with nothing saying why."""
    missing = Path(tempfile.mkdtemp()) / "mobile_2.tum"
    try:
        load_tum(missing, name="reference/mobile_2")
        assert False, "a missing file loaded"
    except FileNotFoundError as e:
        msg = str(e)
    assert "mobile_2.tum" in msg and "reference/mobile_2" in msg
    assert "export_reference.py" in msg, msg


def main() -> int:
    for name, err, tb in FAIL:
        print(f"FAIL {name}: {err}\n{tb}")
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())

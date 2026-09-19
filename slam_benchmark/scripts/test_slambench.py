#!/usr/bin/env python3
"""Self-tests for the evaluation layer. No bag, no ROS, no GPU, no network.

Run before trusting a single number out of this harness:

    python3 scripts/test_slambench.py

The tests that matter most are the ones that check a WRONG configuration is
caught: an unapplied extrinsic, a sim3 scale hidden under an se3 label, a
reference on a different clock, a map scored against an empty volume.
"""
from __future__ import annotations

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


def main() -> int:
    for name, err, tb in FAIL:
        print(f"FAIL {name}: {err}\n{tb}")
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())

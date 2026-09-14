#!/usr/bin/env python3
"""Self-tests for the evaluation layer. No bag, no ROS, no GPU, no network.

Run before trusting a single number out of this harness:

    python3 scripts/test_slambench.py

The tests that matter most are the ones that check a WRONG configuration is
caught: an unapplied extrinsic, a sim3 scale hidden under an se3 label, a
reference on a different clock, a map scored against an empty volume.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slambench import (Trajectory, absolute_check, ate, crop_box, evaluate_map,
                       extrinsic_matrix, fit, load_cloud, load_tum, nearest_distances,
                       revisit_residual, rpe, save_ply, save_tum, to_reference_frame,
                       voxel_downsample)
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
def test_absolute_check_reports_verdict_per_anchor():
    tr = circle_traj(200, radius=3.0)
    anchors = [
        {"name": "tight", "position": list(tr.positions[5]), "uncertainty_m": 0.007,
         "window": [tr.stamps[0], tr.stamps[10]]},
        {"name": "loose", "position": [100.0, 0.0, 0.0], "uncertainty_m": 0.015,
         "window": [tr.stamps[0], tr.stamps[10]]},
        {"name": "absent", "position": [0, 0, 0], "uncertainty_m": 0.01,
         "window": [1e9, 1e9 + 1]},
    ]
    rows = absolute_check(tr, anchors)
    assert rows[0]["residual_m"] < 0.35
    assert rows[1]["verdict"].startswith("resolved")
    assert rows[2]["n"] == 0 and rows[2]["residual_m"] is None


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


def main() -> int:
    for name, err, tb in FAIL:
        print(f"FAIL {name}: {err}\n{tb}")
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())

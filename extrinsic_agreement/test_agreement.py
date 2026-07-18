#!/usr/bin/env python3
"""Synthetic self-test: prove the transform chains and the metrics are correct.

Builds a known static chair in the map frame and synthesizes each sensor's
observation of it through the *inverse* of the dataset transforms (so we have
ground truth). Then:

  A. With perfect observations, every disagreement must be ~0.
  B. With one extrinsic corrupted at generation time, the tool must report a bias
     matching the injected error (right axis, right magnitude).
  C. The 'auto' LiDAR interpretation must recover the true sheet direction.

Run: python3 test_agreement.py
"""

from __future__ import annotations

import numpy as np

import dataset as ds
from se3 import se3, inv, apply, chain
from agreement import Frame, SensorObs, rigid_rig, map_consensus, reprojection


rng = np.random.default_rng(0)


def make_zed_pose(i: int) -> np.ndarray:
    """A plausible moving-platform pose; varies per frame to exercise rotation."""
    from scipy.spatial.transform import Rotation
    yaw = np.radians(10 * i)
    R = Rotation.from_euler("zyx", [yaw, 0.03 * i, -0.02 * i]).as_matrix()
    t = np.array([0.5 + 0.3 * i, 0.1 * i, 0.18])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# Put the chair 6 m in front of the (fixed) Arducam so it lands in its FOV; this
# is the shared static target. The moving platforms happen to see it too.
CHAIR_MAP = apply(ds.T_map_arducam, np.array([0.3, 0.2, 6.0]))


def synth_frames(n=6, chair_map=CHAIR_MAP,
                 lidar_true="arrow",
                 corrupt_lidar_dz=0.0, corrupt_arducam_dz=0.0,
                 noise_m=0.0) -> list[Frame]:
    frames = []
    for i in range(n):
        T_map_zed = make_zed_pose(i)
        # ZED sees the chair
        p_zed = apply(inv(T_map_zed), chair_map) + noise_m * rng.standard_normal(3)

        # LiDAR: generate through the (possibly corrupted) true extrinsic
        T_zed_lidar_true = ds.T_zed_lidar(lidar_true).copy()
        if corrupt_lidar_dz:
            T_zed_lidar_true[2, 3] += corrupt_lidar_dz
        p_lidar = apply(inv(T_zed_lidar_true), p_zed) + noise_m * rng.standard_normal(3)

        # RealSense: independent pose
        T_map_rs = make_zed_pose(i) @ se3([0.05, 0.02, 0.0], [0, 0, 0, 1])
        p_rs = apply(inv(T_map_rs), chair_map) + noise_m * rng.standard_normal(3)

        # Arducam: fixed, possibly corrupted z
        T_map_ard = ds.T_map_arducam.copy()
        if corrupt_arducam_dz:
            T_map_ard[2, 3] += corrupt_arducam_dz
        p_ard = apply(inv(T_map_ard), chair_map)
        uv_ard = ds.ARDUCAM.project(p_ard)[0]
        uv_zed = ds.ZED_LEFT.project(p_zed)[0]

        frames.append(Frame(fid=str(i), t=float(i), sensors={
            "zed": SensorObs(point=p_zed, pixel=uv_zed, pose_map=T_map_zed),
            "lidar": SensorObs(point=p_lidar),
            "realsense": SensorObs(point=p_rs, pose_map=T_map_rs),
            "arducam": SensorObs(pixel=uv_ard),
        }))
    return frames


def approx(a, b, tol):
    return abs(a - b) <= tol


def test_perfect():
    f = synth_frames()
    rr = rigid_rig(f, lidar_interp="arrow")
    assert rr["lidar"]["rms"] < 1e-6, rr["lidar"]["rms"]
    mc = map_consensus(f)
    for s, r in mc["per_sensor"].items():
        assert r["rms"] < 1e-6, (s, r["rms"])
    rp = reprojection(f)
    assert rp["arducam"]["rms_px"] < 1e-6, rp["arducam"]
    print("A. perfect extrinsics -> zero disagreement            OK")


def test_lidar_corruption():
    # inject +40 mm z error into the LiDAR<->ZED extrinsic at generation time
    f = synth_frames(corrupt_lidar_dz=0.040)
    rr = rigid_rig(f, lidar_interp="arrow")
    rms = rr["lidar"]["rms"]
    # residual magnitude must match the injected 40 mm (rigid, so exact)
    assert approx(rms, 40.0, 1.0), rms
    print(f"B1. injected 40 mm LiDAR extrinsic error -> RMS {rms:.1f} mm   OK")


def test_lidar_interp_auto():
    # true direction is 'arrow'; feed observations built that way, ask 'auto'
    f = synth_frames(lidar_true="arrow")
    rr = rigid_rig(f, lidar_interp="auto")
    assert rr["lidar"]["interp"] == "arrow", rr["lidar"]["interp"]
    assert rr["lidar"]["rms"] < 1e-6
    assert rr["lidar"]["worse_alt"]["rms"] > 50.0  # wrong direction is far off
    print(f"C. auto interp picks the true sheet direction "
          f"(wrong one off by {rr['lidar']['worse_alt']['rms']:.0f} mm)  OK")


def test_arducam_z_symptom():
    # raise the fixed Arducam by 100 mm at generation -> a vertical reproj bias
    f = synth_frames(corrupt_arducam_dz=0.100)
    rp = reprojection(f)
    v_bias = rp["arducam"]["v_bias"]
    assert abs(v_bias) > 1.0, v_bias   # shows up as a nonzero vertical pixel bias
    print(f"B2. injected 100 mm Arducam z error -> v_bias {v_bias:+.1f} px   OK")


def test_noise_is_scatter_not_bias():
    f = synth_frames(noise_m=0.01)  # 1 cm localization noise, no extrinsic error
    rr = rigid_rig(f, lidar_interp="arrow")
    bias = np.linalg.norm(rr["lidar"]["bias_mm"])
    scatter = np.linalg.norm(rr["lidar"]["scatter_mm"])
    assert bias < scatter, (bias, scatter)  # noise -> scatter dominates, ~0 bias
    print(f"D. pure noise -> scatter {scatter:.1f} mm >> bias {bias:.1f} mm   OK")


if __name__ == "__main__":
    test_perfect()
    test_lidar_corruption()
    test_lidar_interp_auto()
    test_arducam_z_symptom()
    test_noise_is_scatter_not_bias()
    print("\nall self-tests passed")

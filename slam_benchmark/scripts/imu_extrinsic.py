#!/usr/bin/env python3
"""Recover the camera->IMU rotation from the bag, and measure the rotational
excitation that decides whether a VI estimator can initialise at all (I15).

WHY THIS EXISTS. `mobile_2.imu.extrinsic_from_camera` is null and blocks every
Stage 2 rung that adds inertial data. The bag cannot supply it the way I13 was
supplied: /tf_static carries the ZED and Ouster chains and nothing for the
RealSense. The device would report its factory value through
`rs-enumerate-devices -c`, but the device is not always to hand, and a datasheet
nominal is exactly the "cheap result that is an expensive wrong one" rule 2
exists to refuse.

THE MEASUREMENT. The IMU is bolted to the imager, so both sensors observe the
SAME rotation of the SAME rigid body in two different frames:

    omega_cam(t)  =  R_cam_imu @ omega_imu(t)      for every t

That is Wahba's problem over a few thousand samples: form M = sum omega_cam
omega_imu^T and take its SVD. No integration, no bias-free assumption on the
accelerometer, and no dependence on the translation lever arm -- angular
velocity of a rigid body is the same at every point of it.

WHAT IT CANNOT DO, STATED UP FRONT:

  * The TRANSLATION is not recovered. Its observability comes from the
    centripetal and tangential terms of the accelerometer, which on an indoor
    cart at walking pace are at the noise floor. This script reports the
    excitation available for it and refuses to emit a number.
  * The reference for mobile_2 is cuVSLAM-derived, and cuVSLAM already fuses
    THIS IMU under the extrinsic realsense2_camera published. So what comes
    back is the extrinsic the recording system itself used. That is the right
    answer for making this benchmark self-consistent, and it is NOT an
    independent confirmation of the factory calibration. The fit residual is
    the thing to read: a wrong assumed extrinsic would leave the reference
    inconsistent with the raw gyro, and the residual would show it.

RANK IS THE GATE. A rotation about one axis only cannot determine the other
two. The singular values of M say which axes were exercised; if the smallest is
a negligible fraction of the largest the fit is rank-deficient and this script
REFUSES rather than printing a confident wrong rotation for the unexcited axes.
That same number is the answer to I15.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slambench.config import load_dataset                            # noqa: E402
from slambench.se3 import rot_to_quat, rotation_angle, so3_log       # noqa: E402
from slambench.trajectory import load_tum                            # noqa: E402

# Below this ratio of smallest to largest singular value, one axis of the
# rotation is not determined by the data. Not tuned: 0.05 means the weakest
# axis carries 5% of the strongest's angular energy, and under that the fitted
# column for it is noise wearing a rotation matrix's clothes.
MIN_RANK_RATIO = 0.05
# Angular rate under which a sample says nothing about the rotation and only
# contributes gyro noise and reference quantisation. 2 deg/s is well above the
# ~0.1 deg/s bias-instability of the BMI055-class parts in these modules.
MIN_RATE_DEG_S = 2.0


def read_imu(bag: str, topic: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stamps, angular velocity (rad/s) and linear acceleration from the bag."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in types:
        raise SystemExit(f"{topic} is not in {bag}. Topics: {sorted(types)[:12]} ...")
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    cls = get_message(types[topic])

    t, w, a = [], [], []
    while reader.has_next():
        _, data, _ = reader.read_next()
        m = deserialize_message(data, cls)
        # The HEADER stamp, not the bag's receive stamp: the reference is
        # stamped on the same clock and a receive stamp carries driver latency.
        t.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        w.append([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
        a.append([m.linear_acceleration.x, m.linear_acceleration.y,
                  m.linear_acceleration.z])
    order = np.argsort(t)
    return (np.asarray(t)[order], np.asarray(w)[order], np.asarray(a)[order])


def camera_rates(traj, min_dt: float, max_dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Angular velocity of the camera, in the camera frame, per pose interval.

    Taken as the body-frame rotation between consecutive poses over its own dt,
    which is what the gyro integrates. Intervals with a gap outside
    [min_dt, max_dt] are dropped: a long gap means the reference was
    interpolated or dropped out, and the finite difference across it is not a
    rate.
    """
    R = traj.poses[:, :3, :3]
    dt = np.diff(traj.stamps)
    ok = (dt >= min_dt) & (dt <= max_dt)
    mid, rate = [], []
    for i in np.flatnonzero(ok):
        rate.append(so3_log(R[i].T @ R[i + 1]) / dt[i])
        mid.append(0.5 * (traj.stamps[i] + traj.stamps[i + 1]))
    return np.asarray(mid), np.asarray(rate)


def imu_rates_at(t_imu: np.ndarray, w_imu: np.ndarray, edges: np.ndarray,
                 mids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean gyro rate over each camera interval, and a mask of usable ones.

    Averaging over the SAME interval the camera difference spans is the honest
    pairing: a point sample at the midpoint of a 100 ms interval is a different
    quantity from the mean the camera difference measures.
    """
    lo = np.searchsorted(t_imu, edges[:, 0], side="left")
    hi = np.searchsorted(t_imu, edges[:, 1], side="right")
    out = np.zeros((len(mids), 3))
    ok = np.zeros(len(mids), dtype=bool)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b - a >= 2:                    # at least two samples to mean over
            out[i] = w_imu[a:b].mean(axis=0)
            ok[i] = True
    return out, ok


def fit_rotation(wc: np.ndarray, wi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rotation carrying IMU rates onto camera rates, plus M's spectrum."""
    M = wc.T @ wi
    U, S, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    return U @ D @ Vt, S


def main() -> int:                                                 # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--agent", default="mobile_2")
    ap.add_argument("--reference", required=True, help="the agent's reference TUM")
    ap.add_argument("--bag", default=None)
    ap.add_argument("--max-offset-ms", type=float, default=100.0,
                    help="half-width of the clock-offset search")
    ap.add_argument("--offset-step-ms", type=float, default=2.0)
    args = ap.parse_args()

    cfg = load_dataset(args.config)
    imu = cfg.raw.get("streams", {}).get(f"{args.agent}.imu") or {}
    if not imu.get("topic"):
        raise SystemExit(f"streams.{args.agent}.imu.topic is not declared")
    ref_decl = cfg.reference["agents"][args.agent]
    bag = str(Path(args.bag or cfg.raw["dataset"]["bag"]).expanduser())

    t_imu, w_imu, a_imu = read_imu(bag, imu["topic"])
    traj = load_tum(args.reference, name=args.agent)
    print(f"{len(t_imu)} IMU samples at "
          f"{(len(t_imu) - 1) / (t_imu[-1] - t_imu[0]):.1f} Hz, "
          f"{len(traj)} reference poses over {traj.duration:.1f} s")
    print(f"recovering {ref_decl['frame']} -> {imu['sensor_frame']}")

    max_gap = float(ref_decl.get("max_gap_ms", 250)) * 1e-3
    mids, wc = camera_rates(traj, min_dt=1e-3, max_dt=max_gap)
    if len(wc) < 100:
        raise SystemExit(f"only {len(wc)} usable reference intervals; "
                         f"not enough to fit a rotation")
    edges = np.column_stack([mids - 0.5 * max_gap, mids + 0.5 * max_gap])

    # ------------------------------------------------ excitation first (I15)
    # Reported BEFORE the fit, because if the answer is "the robot barely
    # rotated" then both this extrinsic and every VI rung downstream are
    # affected, and that is the finding regardless of what the SVD returns.
    speed = np.degrees(np.linalg.norm(wc, axis=1))
    total = np.degrees(np.abs(wc).sum(axis=0) * np.median(np.diff(traj.stamps)))
    print(f"\nROTATIONAL EXCITATION (I15)\n"
          f"  rate deg/s: median {np.median(speed):.2f}, p90 "
          f"{np.percentile(speed, 90):.2f}, max {speed.max():.2f}\n"
          f"  above {MIN_RATE_DEG_S} deg/s: {int((speed > MIN_RATE_DEG_S).sum())} "
          f"of {len(speed)} intervals "
          f"({100 * (speed > MIN_RATE_DEG_S).mean():.0f}%)\n"
          f"  cumulative rotation per camera axis, deg: "
          f"[{total[0]:.0f}, {total[1]:.0f}, {total[2]:.0f}]")

    keep = speed > MIN_RATE_DEG_S
    mids, wc, edges = mids[keep], wc[keep], edges[keep]
    if len(wc) < 100:
        raise SystemExit(f"only {len(wc)} intervals rotate faster than "
                         f"{MIN_RATE_DEG_S} deg/s. The sequence does not "
                         f"excite the gyro enough to identify the extrinsic, "
                         f"and that is the result: I15 is answered NO.")

    # ------------------------------------------------ clock offset, then fit
    # The two streams are on the same clock but not the same driver path, and a
    # tens-of-ms offset rotates the fit. Searched, not assumed.
    best = None
    for off in np.arange(-args.max_offset_ms, args.max_offset_ms + 1e-9,
                         args.offset_step_ms) * 1e-3:
        wi, ok = imu_rates_at(t_imu + off, w_imu, edges, mids)
        if ok.sum() < 100:
            continue
        R, S = fit_rotation(wc[ok], wi[ok])
        res = np.degrees(np.linalg.norm(wc[ok] - wi[ok] @ R.T, axis=1))
        m = float(np.median(res))
        if best is None or m < best[0]:
            best = (m, off, R, S, res, int(ok.sum()))
    if best is None:
        raise SystemExit("no clock offset gave 100 paired intervals; "
                         "check that the IMU and the reference overlap in time")
    med, off, R, S, res, n = best

    ratio = float(S[2] / S[0])
    print(f"\nFIT over {n} intervals, clock offset {off * 1e3:+.0f} ms\n"
          f"  singular values {S[0]:.3f} / {S[1]:.3f} / {S[2]:.3f} "
          f"-> weakest axis {100 * ratio:.1f}% of strongest\n"
          f"  residual rate error: median {med:.3f} deg/s, "
          f"p90 {np.percentile(res, 90):.3f}, "
          f"vs a median rate of {np.median(np.degrees(np.linalg.norm(wc, axis=1))):.2f} deg/s")

    if ratio < MIN_RANK_RATIO:
        print(f"\nREFUSED: the weakest axis carries {100 * ratio:.1f}% of the "
              f"strongest, under the {100 * MIN_RANK_RATIO:.0f}% this needs. "
              f"The motion is close to planar, so rotation about the two "
              f"in-plane axes is not identified and two columns of the matrix "
              f"below would be noise. Leave extrinsic_from_camera null.",
              file=sys.stderr)
        return 1

    q = rot_to_quat(R)
    ang = np.degrees(rotation_angle(R))
    print(f"\n  R ({ref_decl['frame']} <- {imu['sensor_frame']}), "
          f"{ang:.2f} deg from identity\n"
          f"  quat xyzw : [{q[0]: .6f}, {q[1]: .6f}, {q[2]: .6f}, {q[3]: .6f}]")
    for r in R:
        print(f"    [{r[0]: .6f}, {r[1]: .6f}, {r[2]: .6f}]")

    # ------------------------------------------------ what is NOT recovered
    # The lever arm shows up in the accelerometer as -omega x (omega x r) and
    # -domega/dt x r. Both scale with rotation rate; at these rates they sit
    # under the accelerometer noise, so the honest output is the magnitude of
    # the signal that would have to carry it, not a number.
    dt = np.diff(mids)
    dw = np.diff(wc, axis=0) / np.where(dt > 0, dt, np.nan)[:, None]
    centri = np.linalg.norm(np.cross(wc[:-1], np.cross(wc[:-1], wc[:-1])), axis=1)
    tang = np.linalg.norm(dw, axis=1)
    print(f"\nTRANSLATION NOT RECOVERED, and here is the reason in numbers.\n"
          f"  a lever arm r appears in the accelerometer as "
          f"-w x (w x r) - dw/dt x r\n"
          f"  |w x (w x .)| median {np.nanmedian(centri):.4f} 1/s^2, "
          f"|dw/dt| median {np.nanmedian(tang):.3f} 1/s^2\n"
          f"  so a 5 cm arm contributes about "
          f"{0.05 * np.nanmedian(tang):.4f} m/s^2, against a typical "
          f"accelerometer noise density of ~0.01 m/s^2/sqrt(Hz).\n"
          f"  Fill extrinsic_from_camera's translation from the device "
          f"(`rs-enumerate-devices -c`) or leave it null and declare the rung "
          f"blocked. Do not take it from a datasheet drawing.")
    return 0


if __name__ == "__main__":                                           # pragma: no cover
    sys.exit(main())

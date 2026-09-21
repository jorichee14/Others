#!/usr/bin/env python3
"""Where does the IMU think DOWN is, and does the survey agree?

    python3 scripts/gravity_check.py --config configs/coop2.yaml --agent mobile_1 \
        --reference runs/coop2_20260828/reference/mobile_1.tum

The question this answers. RTAB-Map's inertial row does not integrate the IMU;
it takes the `orientation` quaternion off each message and adds a GRAVITY
constraint at `Optimizer/GravitySigma 0.3`. So the only thing the IMU
contributes is a direction, and if that direction is wrong the constraint is a
systematic pull toward a tilted vertical -- which is exactly what an ablation
that makes the result WORSE looks like. On `mobile_1` the IMU row is worse than
the IMU-free row (323 mm -> 366-451 mm), and on `mobile_2` it is not, and the
two platforms differ in precisely where the orientation comes from: the ZED
SDK's own fusion on `mobile_1`, our Madgwick filter on `mobile_2`. That is a
hypothesis about the ZED's fusion or its declared extrinsic. This script is its
test, and it needs no new recording.

Three directions, all expressed in the IMU's own frame, all unit vectors:

  vendor   the orientation quaternion's own up:  R_world_imu^T @ +z.
           This is the one RTAB-Map consumes. Available while MOVING.
  survey   the up the anchored map defines, carried down the declared chain:
           (R_map_ref(t) @ R_ref_imu)^T @ +z. Reference-informed by
           construction -- it uses the LiDAR reference's ORIENTATION, so it is
           a diagnostic and never a tier-1 number (rule 5).
  accel    the accelerometer's up, a_imu / |a_imu|, valid only where the body
           is not accelerating. Uses NO reference and NO fusion.

The three pairwise angles separate the causes, which is the whole point of
computing all three rather than the one that matters:

  vendor vs survey   large   the constraint RTAB-Map applies is tilted. The
                            magnitude is what the ablation is paying.
  accel  vs survey   large   then the extrinsic (or the reference's own
                            vertical) is off, and the fusion is innocent.
  accel  vs vendor   large   the fusion disagrees with its OWN accelerometer,
                            which no extrinsic and no reference can explain.

A constant `vendor vs survey` offset is a bias -- extrinsic or fusion. One that
grows along the run is fusion drift. The script reports both the median and the
trend, because they have different fixes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slambench import load_dataset, load_tum, se3                  # noqa: E402

G = 9.80665

# optical (x right, y down, z forward) -> ROS body (x forward, y left, z up).
# The ZED publishes its IMU extrinsic against `zed_left_camera_frame` (body)
# while this benchmark's reference frame is `zed_left_camera_optical_frame`, so
# the chain crosses this convention change and getting it backwards yields a
# transform that is plausible, silent and wrong. Sanity check recorded in the
# config and asserted in the tests: IMU +z must land on optical -y (up).
R_BODY_FROM_OPTICAL = np.array([[0.0, 0.0, 1.0],
                                [-1.0, 0.0, 0.0],
                                [0.0, -1.0, 0.0]])


def ref_from_imu(cfg, agent: str) -> tuple[np.ndarray, str]:
    """R_ref_imu, where `ref` is the frame the reference trajectory is written in.

    Refuses rather than guessing the convention (rule 3). The extrinsic's
    `parent` either IS the reference frame, or is its body counterpart -- the
    same name with `_optical_frame` replaced by `_frame`, which is the ROS
    convention both vendors follow. Anything else and the composition would be
    a guess dressed as geometry.
    """
    imu = cfg.raw.get("streams", {}).get(f"{agent}.imu") or {}
    ext = imu.get("extrinsic_from_camera")
    if not isinstance(ext, dict) or not {"parent", "child", "xyz", "quat_xyzw"} <= set(ext):
        raise SystemExit(
            f"streams.{agent}.imu.extrinsic_from_camera is null or malformed. "
            f"Without camera<-IMU there is no chain from the reference frame to "
            f"the inertial one and no gravity direction to check.")
    ref_frame = ((cfg.raw["reference"].get("agents") or {}).get(agent) or {}).get("frame")
    parent = str(ext["parent"])
    R_pi = se3.quat_to_rot(np.asarray(ext["quat_xyzw"], float))
    if parent == ref_frame:
        return R_pi, f"{ref_frame} <- {ext['child']}  (declared directly)"
    if parent == str(ref_frame).replace("_optical_frame", "_frame"):
        # R_ref_imu = R_ref_body @ R_body_imu, and R_ref_body = R_body_ref^T.
        return R_BODY_FROM_OPTICAL.T @ R_pi, (
            f"{ref_frame} <- {parent} <- {ext['child']}  "
            f"(optical<-body convention change applied)")
    raise SystemExit(
        f"extrinsic parent `{parent}` is neither the reference frame "
        f"`{ref_frame}` nor its body counterpart. Composing them would be a "
        f"guess; declare the chain in the config instead.")


def read_imu(bag: str, topic: str) -> dict[str, np.ndarray]:     # pragma: no cover
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in types:
        raise SystemExit(f"{topic} is not in {bag}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    cls = get_message(types[topic])

    t, quat, acc, gyr = [], [], [], []
    while reader.has_next():
        _, data, _ = reader.read_next()
        m = deserialize_message(data, cls)
        t.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        q = m.orientation
        quat.append([q.x, q.y, q.z, q.w])
        a, w = m.linear_acceleration, m.angular_velocity
        acc.append([a.x, a.y, a.z])
        gyr.append([w.x, w.y, w.z])
    return {"t": np.array(t), "quat": np.array(quat),
            "acc": np.array(acc), "gyro": np.array(gyr)}


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-12)


def angles_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between corresponding unit vectors, in degrees."""
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def static_runs(acc: np.ndarray, gyro: np.ndarray, t: np.ndarray,
                positions: np.ndarray | None = None,
                accel_tol: float = 0.15, gyro_tol: float = 0.02,
                move_tol_m: float = 0.01, min_s: float = 0.5) -> list[tuple[int, int]]:
    """Contiguous [i, j) where the body is at rest.

    THE MAGNITUDE TEST IS NEARLY BLIND TO THE ERROR THAT MATTERS, and that is
    why `positions` exists. A body accelerating by `p` PERPENDICULAR to gravity
    tilts the measured up by atan(p/g) while lengthening the vector by only
    sqrt(p^2+g^2)-g ~= p^2/2g. At p = 1.5 m/s^2 that is an 8.7 deg tilt for a
    0.11 m/s^2 change in length -- inside any tolerance loose enough to pass a
    real accelerometer. The magnitude bounds the ALONG-gravity component and
    essentially nothing else. Caught by the tests, 2026-09-20; the first
    version of this function accepted exactly that motion as rest.

    So rest is established from the TRAJECTORY when one is supplied: the
    reference says where the body was, and a window it did not move in is a
    window it was not accelerating in, whatever the accelerometer reads. The
    inertial tests stay as necessary conditions -- the gyro rejects a steady
    turn (which holds |a| at g while rotating the measured vector), the
    magnitude rejects a gross along-gravity manoeuvre -- and the minimum
    duration rejects the instants a manoeuvre spends crossing a threshold on
    its way through.

    With `positions=None` the check degrades to the inertial tests alone and
    the caller is responsible for saying so; it is not the same test.
    """
    ok = ((np.abs(np.linalg.norm(acc, axis=1) - G) < accel_tol)
          & (np.linalg.norm(gyro, axis=1) < gyro_tol))
    if positions is not None:
        # PER SAMPLE, not per segment. Testing the spread of a whole contiguous
        # run instead would throw away the rest either side of a manoeuvre the
        # inertial tests failed to break the run at -- which is precisely the
        # case that exists, since the motion they are blind to is the one this
        # test is here to catch. Each sample is judged on the reference's
        # wander in the +/- min_s/2 around it.
        lo = np.searchsorted(t, t - min_s / 2, side="left")
        hi = np.searchsorted(t, t + min_s / 2, side="right")
        still = np.empty(len(t), bool)
        for k in range(len(t)):
            p = positions[lo[k]:hi[k]]
            still[k] = (np.linalg.norm(p - p.mean(axis=0), axis=1).max()
                        <= move_tol_m) if len(p) else False
        ok &= still

    out, i = [], None
    for k, flag in enumerate(ok):
        if flag and i is None:
            i = k
        elif not flag and i is not None:
            if t[k - 1] - t[i] >= min_s:
                out.append((i, k))
            i = None
    if i is not None and t[-1] - t[i] >= min_s:
        out.append((i, len(ok)))
    return out


def trend_deg_per_min(t: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope. Separates a fusion that DRIFTS from one that is
    merely offset -- an offset is an extrinsic or a bias and is correctable
    once; a drift is not, and it gets worse the longer the run."""
    if len(t) < 3:
        return float("nan")
    return float(np.polyfit(t - t[0], y, 1)[0] * 60.0)


def report(name: str, y: np.ndarray, t: np.ndarray) -> str:
    return (f"  {name:<22} median {np.median(y):6.2f}  p95 {np.percentile(y, 95):6.2f}  "
            f"max {y.max():6.2f}   trend {trend_deg_per_min(t, y):+6.2f} deg/min   "
            f"(n={len(y)})")


def main() -> int:                                                 # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--reference", required=True,
                    help="the agent's reference TUM, from scripts/export_reference.py")
    ap.add_argument("--bag", default=None, help="override dataset.bag")
    ap.add_argument("--max-gap-s", type=float, default=None,
                    help="refuse to interpolate the reference across a larger hole. "
                         "Default: the agent's declared reference.max_gap_ms -- at "
                         "9.7 Hz a hand-picked 50 ms would discard the whole run.")
    ap.add_argument("--accel-tol", type=float, default=0.15,
                    help="|a|-g tolerance for a static sample, m/s^2")
    ap.add_argument("--gyro-tol", type=float, default=0.02, help="|w| tolerance, rad/s")
    ap.add_argument("--move-tol-m", type=float, default=0.01,
                    help="how far the REFERENCE may wander inside a window still "
                         "called static. This is the test that actually works; see "
                         "static_runs.")
    args = ap.parse_args()

    cfg = load_dataset(args.config)
    ref_agent = (cfg.raw["reference"].get("agents") or {}).get(args.agent) or {}
    max_gap_s = args.max_gap_s
    if max_gap_s is None:
        if ref_agent.get("max_gap_ms") is None:
            raise SystemExit(f"reference.agents.{args.agent}.max_gap_ms is unset and "
                             f"--max-gap-s was not given; refusing to pick one (rule 3).")
        max_gap_s = float(ref_agent["max_gap_ms"]) / 1e3
    imu_stream = cfg.raw.get("streams", {}).get(f"{args.agent}.imu") or {}
    if not imu_stream.get("topic"):
        raise SystemExit(f"streams.{args.agent}.imu has no topic")
    R_ref_imu, chain = ref_from_imu(cfg, args.agent)

    bag = str(Path(args.bag or cfg.raw["dataset"]["bag"]).expanduser())
    d = read_imu(bag, imu_stream["topic"])
    ref = load_tum(args.reference)

    # Only where the reference actually covers the IMU. Extrapolating past
    # either end would invent the orientation the whole check is comparing to.
    lo, hi = ref.stamps[0], ref.stamps[-1]
    keep = (d["t"] >= lo) & (d["t"] <= hi)
    for k in ("t", "quat", "acc", "gyro"):
        d[k] = d[k][keep]
    if len(d["t"]) < 10:
        raise SystemExit(f"only {len(d['t'])} IMU samples inside the reference's "
                         f"{hi - lo:.1f} s window; nothing to check.")

    # `interpolate` DROPS the queries it cannot serve rather than returning
    # NaN for them, so the survivors have to be matched back by stamp. The
    # values are copied verbatim out of `d["t"]`, so searchsorted is exact.
    at_imu = ref.interpolate(d["t"], max_gap_s=max_gap_s)
    ok = np.searchsorted(d["t"], at_imu.stamps)
    if not np.array_equal(d["t"][ok], at_imu.stamps):
        raise SystemExit("interpolated stamps did not match the IMU stamps they came "
                         "from; the mapping back would be wrong.")
    dropped = len(d["t"]) - len(ok)
    for k in ("t", "quat", "acc", "gyro"):
        d[k] = d[k][ok]
    R_map_ref = at_imu.poses[:, :3, :3]

    up = np.array([0.0, 0.0, 1.0])
    # survey: the anchored map's vertical, carried into the IMU frame.
    R_map_imu = R_map_ref @ R_ref_imu
    u_survey = np.einsum("nji,j->ni", R_map_imu, up)

    has_orientation = np.linalg.norm(d["quat"], axis=1) > 0.5
    print(f"\n=== gravity check  {args.agent}\n")
    print(f"  chain      {chain}")
    print(f"  IMU        {imu_stream['topic']}  orientation={imu_stream.get('orientation')}")
    print(f"  window     {d['t'][-1] - d['t'][0]:.1f} s, {len(d['t'])} samples "
          f"inside the reference ({dropped} dropped across gaps > {max_gap_s * 1e3:.0f} ms)\n")

    if has_orientation.sum() < 10:
        print("  vendor: the stream carries NO orientation (all-zero quaternions), so "
              "there is no fused direction to check. That is the mobile_2 case -- the "
              "direction RTAB-Map consumes there comes from our own Madgwick filter, "
              "which this script cannot see from the bag.")
        u_vendor = None
    else:
        R_w_imu = np.stack([se3.quat_to_rot(q) for q in d["quat"][has_orientation]])
        u_vendor = np.einsum("nji,j->ni", R_w_imu, up)
        y = angles_deg(u_vendor, u_survey[has_orientation])
        print("VENDOR FUSION vs SURVEY -- the error in the constraint RTAB-Map applies")
        print(report("vendor vs survey", y, d["t"][has_orientation]))

    runs = static_runs(d["acc"], d["gyro"], d["t"],
                       positions=at_imu.poses[:, :3, 3],
                       accel_tol=args.accel_tol, gyro_tol=args.gyro_tol,
                       move_tol_m=args.move_tol_m)
    idx = np.concatenate([np.arange(i, j) for i, j in runs]) if runs else np.array([], int)
    print(f"\nSTATIC SEGMENTS  {len(runs)} found, "
          f"{sum(d['t'][j - 1] - d['t'][i] for i, j in runs):.1f} s total "
          f"({len(idx)} samples)")
    if len(idx) < 10:
        print("  Too little static data to read the accelerometer, so the vendor/survey "
              "disagreement above cannot be SPLIT between the fusion and the extrinsic. "
              "That is a property of the recording, not a failure of the check.")
    else:
        u_accel = unit(d["acc"][idx])
        print(report("accel vs survey", angles_deg(u_accel, u_survey[idx]), d["t"][idx]))
        if u_vendor is not None:
            # index u_vendor back into full-stream positions before selecting
            pos = np.full(len(d["t"]), -1)
            pos[np.flatnonzero(has_orientation)] = np.arange(len(u_vendor))
            both = idx[pos[idx] >= 0]
            if len(both) >= 10:
                print(report("accel vs vendor",
                             angles_deg(unit(d["acc"][both]), u_vendor[pos[both]]),
                             d["t"][both]))

    print("\nNOT A TIER-1 NUMBER. `survey` is built from the LiDAR reference's own "
          "orientation, so every angle above is a DISAGREEMENT with the reference and "
          "inherits whatever the reference's vertical is wrong by. It is diagnostic: it "
          "says whether the gravity constraint is tilted, not by how much the "
          "trajectory is wrong.")
    return 0


if __name__ == "__main__":                                          # pragma: no cover
    sys.exit(main())

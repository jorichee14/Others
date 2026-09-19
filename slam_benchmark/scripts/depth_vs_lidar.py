#!/usr/bin/env python3
"""Is the ZED's depth SCALED WRONG, and by how much?

T0 measured a +9.1% scale error on the ZED SDK's own odometry. Two very
different causes produce that, and they imply opposite plans:

  the DEPTH is biased    -> every depth-consuming backbone inherits it, and the
                            pseudo-GT should be built from depth-free ones
  the TRACKER is at fault -> the depth is fine and depth-based backbones are
                            still the strongest option

mobile_1 settles it directly, because it carries the Ouster and the ZED on one
rigid body: project LiDAR points into the depth image and compare, pixel by
pixel, what each sensor says the same surface's range is. Fit

    zed_depth = a * lidar_range + b

`a` is the scale error and `b` an offset. If a ~ 1.09 the depth is the culprit
and T0's +9% is explained; if a ~ 1.00 the depth is sound and the ZED's
tracker is where the scale went.

This is a MEASUREMENT OF THE SENSOR, not of any SLAM system, so it uses the
reference trajectory only to pair frames in time -- not to place anything.

    python3 scripts/depth_vs_lidar.py --config configs/coop2.yaml --bag <bag>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "rosbag_to_opv2v"))

from slambench import load_dataset, se3                      # noqa: E402


def fit_scale(lidar_r, zed_d):
    """Least squares zed = a*lidar + b, plus a robust ratio.

    Both are reported, and WHEN THEY DISAGREE THE DEPTH ERROR IS
    RANGE-DEPENDENT -- that is the finding, not a defect in either estimator.
    A median ratio is the error at the median sampled range; if the true error
    crosses 1.0 somewhere inside the sampled span, the median lands near 1.0
    and looks benign while the fit reveals the slope. That is exactly what
    coop2's ZED does: fit a=0.779 b=+0.282 over 1-2 m means it over-reads 6%
    at 1 m and under-reads 8% at 2 m, with a median ratio of 0.986 hiding it.
    """
    A = np.stack([lidar_r, np.ones_like(lidar_r)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, zed_d, rcond=None)
    ratio = zed_d / np.maximum(lidar_r, 1e-6)
    return float(a), float(b), float(np.median(ratio)), ratio


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--time-tol", type=float, default=0.05)
    ap.add_argument("--min-range", type=float, default=0.5)
    ap.add_argument("--max-range", type=float, default=8.0)
    ap.add_argument("--per-band-per-frame", type=int, default=200,
                    help="cap samples per range band per frame. Without a cap "
                         "the floor 1-2 m ahead supplies ~99.99%% of pairs and "
                         "the result describes the floor, not the sensor.")
    ap.add_argument("--out", default=None, help="per-pair CSV")
    args = ap.parse_args()

    from ros2opv2v.bagreader import BagReader
    sys.path.insert(0, str(ROOT / "scripts"))
    from phase0 import xyz_of                                  # tested decoder
    from depth_health import decode_depth                      # tested decoder

    cfg = load_dataset(args.config)
    zed = cfg.stream("mobile_1.zed_rgbd")
    ous = cfg.stream("mobile_1.ouster")
    # os_lidar -> zed_left_camera_optical_frame, from the config (rule 4)
    T_zed_lidar = se3.pose_from_rpy(**ous["reference_frame_from_sensor"])
    print(f"os_lidar -> zed optical: t={T_zed_lidar[:3,3].round(4).tolist()} m")

    reader = BagReader(str(Path(args.bag).expanduser()), stamp_source="header")
    topics = [zed["depth_topic"], zed["depth_info_topic"], ous["topic"]]

    K = None
    sweeps, depths = [], []
    for topic, stamp_ns, msg in reader.iter_messages(topics):
        t = stamp_ns * 1e-9
        if topic == zed["depth_info_topic"]:
            if K is None:
                K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        elif topic == ous["topic"]:
            sweeps.append((t, xyz_of(msg, max_points=200000)))
        elif len(depths) < args.max_frames * 3:
            depths.append((t, decode_depth(msg)))
    if K is None:
        print("error: no depth camera_info", file=sys.stderr); return 1
    print(f"{len(sweeps)} sweeps, {len(depths)} depth frames, "
          f"fx={K[0,0]:.1f} cx={K[0,2]:.1f}\n")

    st = np.array([s[0] for s in sweeps])
    L, Z, raw_all = [], [], []
    rng = np.random.default_rng(0)
    BANDS = [0.5, 1, 1.5, 2, 3, 4, 6, 8]
    used = 0
    for t, d in depths:
        if used >= args.max_frames:
            break
        i = int(np.argmin(np.abs(st - t)))
        if abs(st[i] - t) > args.time_tol:
            continue
        pts = sweeps[i][1]
        q = pts @ T_zed_lidar[:3, :3].T + T_zed_lidar[:3, 3]   # into zed optical
        keep = q[:, 2] > args.min_range
        q = q[keep]
        if not len(q):
            continue
        u = (K[0, 0] * q[:, 0] / q[:, 2] + K[0, 2]).round().astype(int)
        v = (K[1, 1] * q[:, 1] / q[:, 2] + K[1, 2]).round().astype(int)
        h, w = d.shape
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, q = u[ok], v[ok], q[ok]
        if not len(u):
            continue
        zed_d = d[v, u]
        lid_r = q[:, 2]                      # both are DEPTH along the optical axis
        good = (np.isfinite(zed_d) & (zed_d > args.min_range)
                & (zed_d < args.max_range) & (lid_r < args.max_range))
        if good.sum() < 50:
            continue
        lr, zr = lid_r[good], zed_d[good]
        raw_all.append(lid_r)                 # pre-validity, for the coverage report
        # STRATIFY. Without this one surface dominates: the floor a metre in
        # front of the cart is seen by both sensors on every frame and supplies
        # hundreds of thousands of samples in the 1-2 m band, while a wall at
        # 5 m supplies a handful. A median over that mixture describes the
        # floor, not the sensor. Cap each band per frame instead.
        for lo, hi in zip(BANDS[:-1], BANDS[1:]):
            m = (lr >= lo) & (lr < hi)
            n = int(m.sum())
            if not n:
                continue
            idx = np.flatnonzero(m)
            if n > args.per_band_per_frame:
                idx = rng.choice(idx, args.per_band_per_frame, replace=False)
            L.append(lr[idx]); Z.append(zr[idx])
        used += 1

    if not L:
        print("no paired samples -- check the time tolerance and the extrinsic",
              file=sys.stderr)
        return 1
    lid = np.concatenate(L); zd = np.concatenate(Z)
    a, b, ratio_med, ratio = fit_scale(lid, zd)
    print(f"{used} frame pairs, {len(lid)} pixel pairs\n")
    print(f"  least squares   zed = {a:.4f} * lidar {b:+.4f} m")
    print(f"  median ratio    zed / lidar = {ratio_med:.4f}")
    print(f"  ratio spread    p10 {np.percentile(ratio,10):.3f}  "
          f"p90 {np.percentile(ratio,90):.3f}")

    print("  by range band (is the error a SCALE or an OFFSET?):")
    bands, ratios = [], []
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        m = (lid >= lo) & (lid < hi)
        if m.sum() < 50:
            print(f"    {lo:.1f}-{hi:.1f} m  n={int(m.sum()):6d}  -- too few to read")
            continue
        bands.append((lo, hi, int(m.sum()), float(np.median(ratio[m])),
                      float(np.median(zd[m] - lid[m]))))
        print(f"    {lo:.1f}-{hi:.1f} m  n={int(m.sum()):6d}  "
              f"median ratio {bands[-1][3]:.4f}  "
              f"median residual {bands[-1][4]*100:+6.1f} cm")

    raw = np.concatenate(raw_all) if raw_all else np.array([])
    if len(raw):
        print(f"\n  coverage check -- LiDAR ranges landing in the image BEFORE "
              f"the depth-validity filter:")
        print("    " + "  ".join(
            f"{lo:.1f}-{hi:.1f}m:{np.mean((raw>=lo)&(raw<hi)):5.1%}"
            for lo, hi in zip(BANDS[:-1], BANDS[1:])))

    slope_disagrees = abs(a - ratio_med) > 0.03
    print(f"\n  VERDICT vs T0's +9.1% odometry scale error:")
    if slope_disagrees:
        print(f"    RANGE-DEPENDENT ERROR. The fit (a={a:.3f}, b={b:+.3f}) and the")
        print(f"    median ratio ({ratio_med:.4f}) disagree, which means the error is")
        print("    not a single number: it varies with range, and the median is")
        print("    merely its value at the median sampled range.")
        lo_r, hi_r = float(np.percentile(lid, 5)), float(np.percentile(lid, 95))
        print(f"    Over the sampled span {lo_r:.2f}-{hi_r:.2f} m the fit implies "
              f"ratio {(a*lo_r+b)/lo_r:.3f} -> {(a*hi_r+b)/hi_r:.3f}.")
        print("    Neither hypothesis holds: this is not a clean +9% scale, and the")
        print("    depth is not sound either.")
    elif abs(ratio_med - 1.091) < 0.03:
        print("    the depth carries the SAME error. It is the culprit; build the")
        print("    pseudo-GT from depth-free backbones, or rescale the depth first.")
    elif abs(ratio_med - 1.0) < 0.02:
        print("    the depth is SOUND. T0's scale error is the ZED's tracker, not")
        print("    the sensor -- depth-based backbones stay in contention.")
    else:
        print(f"    depth scale is {ratio_med:.3f}, neither 1.00 nor 1.09. Report")
        print("    it as its own finding rather than forcing it into either story.")

    if len(bands) < 3:
        print(f"\n  COVERAGE IS INADEQUATE: only {len(bands)} range band(s) have")
        print("    enough paired samples. Whatever the trend is over the sampled")
        print("    span, extrapolating it past that span is not evidence. The")
        print("    ranges that dominate a trajectory scale error (3-8 m) are")
        print("    unmeasured. Raise --max-frames and --per-band-per-frame, and")
        print("    check the coverage line above: if the LiDAR itself lands in")
        print("    the image only at short range, the camera spent the run close")
        print("    to surfaces and this comparison cannot reach further.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.out, np.stack([lid, zd], axis=1), delimiter=",",
                   header="lidar_depth_m,zed_depth_m", comments="")
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

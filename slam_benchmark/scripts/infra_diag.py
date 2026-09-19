#!/usr/bin/env python3
"""Chase the infra_1 residual: which frame convention makes it collapse?

phase0 found a -5.4 m SYSTEMATIC range residual on both agents against the
infra radar, with bearing medians of +3 and +8.6 deg. A constant error across
two independently moving agents is not noise — it is a wrong convention
somewhere in the chain. The suspects are enumerable, so enumerate them:

  * static_world_pose stored as map<-cam (declared) or cam<-map (inverted)
  * the cam->radar extrinsic stored forward or inverted
  * a clock offset between infra_1's machine and mobile_1's (the timeline)

(The ROS-vs-optical axes convention is NOT scanned: with point-cloud
detections the prediction and the detection are relabelled identically, so
any consistent convention gives the same residual — the synthetic test
proved it unidentifiable. Axes only matter against driver-reported angles.)

This script scores EVERY combination (2 x 2 x 2 poses/axes, x a dt scan) by
the residual it leaves against both reference trajectories, and prints the
ranking. The right convention is the one where BOTH agents' range residuals
collapse toward zero and the bearing spread tightens — and if none does, the
node's surveyed pose itself is wrong, which is a different (worse) finding.

Association is deliberately loose here (wide azimuth gate) because the point
is convention-ranking, not measurement; once the winner is found, phase0's
tight check is re-run under it for the publishable residuals.

    python3 scripts/infra_diag.py --config configs/coop2.yaml --bag <bag> \
        --ref-m1 runs/phase0/reference_mobile_1.tum \
        --ref-m2 runs/phase0/reference_mobile_2.tum --out runs/phase0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "rosbag_to_opv2v"))

from slambench import load_dataset, load_tum, se3          # noqa: E402
from slambench.collab import predict_observation           # noqa: E402


def candidate_transforms(T_map_cam, T_cam_radar):
    """All map<-radar hypotheses from the two direction ambiguities."""
    out = {}
    for cam_name, Tc in (("cam_fwd", T_map_cam), ("cam_inv", se3.invert(T_map_cam))):
        for ex_name, Te in (("ext_fwd", T_cam_radar), ("ext_inv", se3.invert(T_cam_radar))):
            out[f"{cam_name}/{ex_name}"] = Tc @ Te
    return out


def score(traj, T_map_radar, sweeps, dt=0.0, az_gate_deg=60.0):
    """Median |az| and median range residual, wide-gate nearest association.

    `dt` shifts the PREDICTIONS: positive dt means the radar's clock is ahead
    of the reference's.
    """
    pred = predict_observation(traj, T_map_radar, axes="ros")   # common language;
    # the choice cancels between prediction and detection (see module docstring)
    ps, paz, prng = pred["stamps"], pred["azimuth_deg"], pred["range_m"]
    az_res, rng_res = [], []
    for t, xyz in sweeps:
        i = int(np.argmin(np.abs(ps - (t + dt))))
        if abs(ps[i] - (t + dt)) > 0.15:
            continue
        det_az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))   # ros: atan2(-right,fwd)
        d = np.abs(((det_az - paz[i]) + 180.0) % 360.0 - 180.0)
        j = int(np.argmin(d))
        if d[j] > az_gate_deg:
            continue
        az_res.append(((det_az[j] - paz[i]) + 180.0) % 360.0 - 180.0)
        rng_res.append(float(np.linalg.norm(xyz[j]) - prng[i]))
    if len(az_res) < 10:
        return None
    return {"n": len(az_res),
            "az_med": float(np.median(az_res)),
            "az_spread": float(np.percentile(np.abs(az_res), 68)),
            "rng_med": float(np.median(rng_res)),
            "rng_spread": float(np.percentile(np.abs(np.asarray(rng_res)
                                                     - np.median(rng_res)), 68))}


def rank(combos_scores):
    """Order hypotheses by how completely they explain the data.

    The figure of merit is |range median| + a bearing term, summed over
    agents: the winning convention has to fix BOTH agents at once, since a
    frame error is common to them and a per-agent fix is a coincidence.
    """
    def merit(v):
        return sum(abs(s["rng_med"]) + 0.05 * abs(s["az_med"]) + 0.02 * s["az_spread"]
                   for s in v.values() if s)
    usable = {k: v for k, v in combos_scores.items()
              if all(s is not None for s in v.values())}
    return sorted(usable.items(), key=lambda kv: merit(kv[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--ref-m1", required=True)
    ap.add_argument("--ref-m2", required=True)
    ap.add_argument("--dt-scan", type=float, default=2.0,
                    help="clock-offset scan half-width, seconds")
    ap.add_argument("--out", default="runs/phase0")
    args = ap.parse_args()

    from ros2opv2v.bagreader import BagReader
    sys.path.insert(0, str(ROOT / "scripts"))
    from phase0 import xyz_of                                # reuse, tested

    cfg = load_dataset(args.config)
    st = cfg.stream("infra_1.radar")
    T_map_cam = se3.pose_from_rpy(**st["static_world_pose"])
    T_cam_radar = se3.pose_from_rpy(**st["reference_frame_from_sensor"])
    trajs = {"mobile_1": load_tum(args.ref_m1), "mobile_2": load_tum(args.ref_m2)}

    sweeps = []
    reader = BagReader(str(Path(args.bag).expanduser()), stamp_source="header")
    for topic, stamp_ns, msg in reader.iter_messages(["/infra_1/radar/points_dynamic"]):
        try:
            xyz = xyz_of(msg)
        except ValueError:
            continue
        if len(xyz):
            sweeps.append((stamp_ns * 1e-9, xyz))
    print(f"{len(sweeps)} dynamic sweeps with points\n")

    combos = candidate_transforms(T_map_cam, T_cam_radar)
    results = {name: {a: score(trajs[a], T, sweeps) for a in trajs}
               for name, T in combos.items()}

    print(f"{'hypothesis':28s} {'agent':9s} {'n':>4s} {'az med':>8s} {'az~sd':>7s} "
          f"{'rng med':>8s} {'rng~sd':>7s}")
    ranking = rank(results)
    for key, v in ranking:
        for a, s in v.items():
            print(f"{key:28s} {a:9s} {s['n']:4d} {s['az_med']:+8.2f} {s['az_spread']:7.2f} "
                  f"{s['rng_med']:+8.2f} {s['rng_spread']:7.2f}")
    dropped = [k for k in results if k not in dict(ranking)]
    if dropped:
        print(f"(no usable association under: {', '.join(dropped)})")

    if not ranking:
        print("\nNOTHING associates — the node's pose is likely wrong at a scale "
              "beyond a convention flip. That is a survey problem, not a code one.")
        return 1

    best_key, best = ranking[0]
    print(f"\nBEST: {best_key}")
    ok = all(abs(s["rng_med"]) < 0.5 and abs(s["az_med"]) < 3.0 for s in best.values())
    if ok:
        print("  residuals collapse for BOTH agents under this convention — "
              "adopt it, then re-run phase0's tight check for the publishable numbers.")
    else:
        print("  even the best convention leaves a systematic residual: "
              f"{ {a: (round(s['az_med'],2), round(s['rng_med'],2)) for a,s in best.items()} }\n"
              "  -> scan the clock offset before blaming the survey:")
        for dt in np.arange(-args.dt_scan, args.dt_scan + 1e-9, 0.25):
            v = {a: score(trajs[a], combos[best_key], sweeps, dt=float(dt))
                 for a in trajs}
            if all(v.values()):
                m = " ".join(f"{a}: az {v[a]['az_med']:+6.2f} rng {v[a]['rng_med']:+6.2f}"
                             for a in v)
                print(f"    dt {dt:+5.2f}s   {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

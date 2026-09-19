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
    az_res, rng_res, rng_near = [], [], []
    for t, xyz in sweeps:
        i = int(np.argmin(np.abs(ps - (t + dt))))
        if abs(ps[i] - (t + dt)) > 0.15:
            continue
        det_az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))   # ros: atan2(-right,fwd)
        d = np.abs(((det_az - paz[i]) + 180.0) % 360.0 - 180.0)
        ingate = d <= az_gate_deg
        if not ingate.any():
            continue
        j = int(np.argmin(d))
        az_res.append(((det_az[j] - paz[i]) + 180.0) % 360.0 - 180.0)
        rng_res.append(float(np.linalg.norm(xyz[j]) - prng[i]))
        # of everything on roughly the right bearing, the return CLOSEST to the
        # predicted range: if this is ~0 while the az-best pick sits at -5 m,
        # the cart is being detected at the right range and the az-pick is
        # grabbing something nearer on the same bearing; if this too is -5 m,
        # nothing returns at the predicted range at all and the error is
        # geometric, not associative.
        rr = np.linalg.norm(xyz[ingate], axis=1) - prng[i]
        rng_near.append(float(rr[np.argmin(np.abs(rr))]))
    if len(az_res) < 10:
        return None
    return {"n": len(az_res),
            "az_med": float(np.median(az_res)),
            "az_spread": float(np.percentile(np.abs(az_res), 68)),
            "rng_med": float(np.median(rng_res)),
            "rng_near_med": float(np.median(rng_near)),
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
    ap.add_argument("--tf-dump", action="store_true", default=True,
                    help="scan /tf and /tf_static for transforms touching the "
                         "infra chain and print their QUATERNIONS — the "
                         "convention-proof source the rpy re-derivation is not")
    ap.add_argument("--envelope", action="store_true", default=True,
                    help="range histogram of /infra_1/radar/points_all — does "
                         "this radar ever see past ~9 m at all?")
    ap.add_argument("--dump", type=int, default=6,
                    help="print N raw sweeps with predictions (0 = off)")
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
    all_ranges = []
    tf_hits = []
    topics = ["/infra_1/radar/points_dynamic"]
    if args.envelope:
        topics.append("/infra_1/radar/points_all")
    if args.tf_dump:
        topics += ["/tf", "/tf_static"]
    KEY = ("infra", "arducam", "radar", "map")
    reader = BagReader(str(Path(args.bag).expanduser()), stamp_source="header")
    for topic, stamp_ns, msg in reader.iter_messages(topics):
        if topic == "/infra_1/radar/points_dynamic":
            try:
                xyz = xyz_of(msg)
            except ValueError:
                continue
            if len(xyz):
                sweeps.append((stamp_ns * 1e-9, xyz))
        elif topic == "/infra_1/radar/points_all":
            try:
                all_ranges.append(np.linalg.norm(xyz_of(msg), axis=1))
            except ValueError:
                pass
        else:
            for tr in msg.transforms:
                pair = (str(tr.header.frame_id), str(tr.child_frame_id))
                if any(k in f.lower() for f in pair for k in KEY) and \
                        not any(h[0] == pair for h in tf_hits):
                    tf_hits.append((pair,
                        [tr.transform.translation.x, tr.transform.translation.y,
                         tr.transform.translation.z],
                        [tr.transform.rotation.x, tr.transform.rotation.y,
                         tr.transform.rotation.z, tr.transform.rotation.w]))
    print(f"{len(sweeps)} dynamic sweeps with points")

    if args.envelope and all_ranges:
        r = np.concatenate(all_ranges)
        q = np.percentile(r, [50, 90, 99, 99.9])
        print(f"\nRANGE ENVELOPE, /infra_1/radar/points_all ({len(r)} returns):")
        print(f"  median {q[0]:.2f}  p90 {q[1]:.2f}  p99 {q[2]:.2f}  "
              f"p99.9 {q[3]:.2f}  MAX {r.max():.2f} m")
        print(f"  returns beyond 9 m: {np.mean(r > 9.0):.2%}   beyond 12 m: "
              f"{np.mean(r > 12.0):.2%}")
        print("  If MAX plateaus below the room scale (7-18 m to the carts from"
              " this corner),\n  the chirp config cannot see the carts and the"
              " radar contributes NOTHING to\n  cross-room localization on this"
              " recording — a hardware-config finding, not code.")

    if args.tf_dump:
        print(f"\nTF QUATERNIONS touching the infra chain ({len(tf_hits)} found):")
        if not tf_hits:
            print("  none on /tf or /tf_static — the transform exists only in "
                  "the offline pipeline's files; get the quaternion from there.")
        for pair, t, q in tf_hits:
            print(f"  {pair[0]} -> {pair[1]}\n    t={np.round(t,4).tolist()}  "
                  f"q_xyzw={np.round(q,6).tolist()}")
    print()

    combos = candidate_transforms(T_map_cam, T_cam_radar)
    results = {name: {a: score(trajs[a], T, sweeps) for a in trajs}
               for name, T in combos.items()}

    print(f"{'hypothesis':28s} {'agent':9s} {'n':>4s} {'az med':>8s} {'az~sd':>7s} "
          f"{'rng med':>8s} {'rngNEAR':>8s} {'rng~sd':>7s}")
    ranking = rank(results)
    for key, v in ranking:
        for a, s in v.items():
            print(f"{key:28s} {a:9s} {s['n']:4d} {s['az_med']:+8.2f} {s['az_spread']:7.2f} "
                  f"{s['rng_med']:+8.2f} {s['rng_near_med']:+8.2f} {s['rng_spread']:7.2f}")
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

    # per-agent clock scan, BEARING ONLY (range is the suspect and cannot vote).
    # Each machine has its own clock, so the offsets need not match.
    print("\nNOTE: an in-container self-consistency test (2026-09-19) showed the"
          "\ncomposed ROTATION of this chain is wrong (predicted elevations 20-30 deg"
          "\noff pure geometry) while the translation is right. Bearings and therefore"
          "\nthe clock scan below are UNRELIABLE until the rotation is fixed from a"
          "\nquaternion source; ranges are rotation-invariant and remain valid.")
    print("\nper-agent clock scan on bearing (best convention, "
          f"dt in ±{args.dt_scan:.0f}s):")
    T_best = combos[best_key]
    for a in trajs:
        rows = []
        for dt in np.arange(-args.dt_scan, args.dt_scan + 1e-9, 0.5):
            v = score(trajs[a], T_best, sweeps, dt=float(dt))
            if v:
                rows.append((abs(v["az_med"]), float(dt), v["az_med"], v["rng_med"]))
        if rows:
            rows.sort()
            _, dt0, az0, rng0 = rows[0]
            print(f"  {a}: best dt {dt0:+5.1f}s  (az med {az0:+.2f} deg, "
                  f"rng med {rng0:+.2f} m at that dt)")
            if abs(dt0) >= args.dt_scan - 0.5:
                print(f"    at the scan edge — rerun with a wider --dt-scan")

    if args.dump:
        print(f"\nraw dump, {args.dump} sweeps (per agent: predicted, then every "
              "in-gate detection as az/rng):")
        idx = np.linspace(0, len(sweeps) - 1, args.dump).astype(int)
        for i in idx:
            t, xyz = sweeps[i]
            det_az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
            det_rng = np.linalg.norm(xyz, axis=1)
            print(f"  t={t:.2f}  ({len(xyz)} pts)")
            for a in trajs:
                pred = predict_observation(trajs[a], T_best, axes="ros")
                k = int(np.argmin(np.abs(pred["stamps"] - t)))
                if abs(pred["stamps"][k] - t) > 0.15:
                    continue
                print(f"    {a}: PRED az {pred['azimuth_deg'][k]:+7.2f}  "
                      f"rng {pred['range_m'][k]:6.2f}  el {pred['elevation_deg'][k]:+6.2f}")
            det_el = np.degrees(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])))
            order = np.argsort(det_rng)
            print("    DET  " + "  ".join(
                f"[{det_az[o]:+.1f}\u00b0,{det_rng[o]:.2f}m,el{det_el[o]:+.0f}\u00b0]"
                for o in order[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

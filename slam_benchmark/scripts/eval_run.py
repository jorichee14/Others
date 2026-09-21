#!/usr/bin/env python3
"""Score one run: trajectory tiers 1-3, and the map if the method produced one.

    python3 scripts/eval_run.py --config configs/coop2.yaml \
        --method configs/methods/kiss_icp.yaml --stream mobile_1.ouster \
        --run runs/coop2_20260828/kiss_icp/mobile_1.ouster \
        --reference runs/coop2_20260828/reference/mobile_1.tum

Reads `<run>/trajectory.tum` and, if present, `<run>/map.ply`; writes
`<run>/metrics.json`. Nothing in here touches a bag, a GPU or ROS, so a run can
be re-scored with a changed threshold in seconds without re-running the method.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slambench import (absolute_check, ate, evaluate_map, load_cloud, load_dataset,  # noqa: E402
                       load_method, load_tum, revisit_residual, rpe, to_reference_frame,
                       transform_points)
from slambench.report import write_json                                     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--stream", required=True,
                    help="the stream key the run used, e.g. mobile_1.ouster")
    ap.add_argument("--run", required=True, help="the run directory")
    ap.add_argument("--reference", required=True, help="reference trajectory, TUM")
    ap.add_argument("--reference-map", default=None, help="override reference.map_cloud")
    ap.add_argument("--alignment", default=None, help="override eval.alignment")
    ap.add_argument("--skip-map", action="store_true")
    ap.add_argument("--trajectory-file", default="trajectory.tum",
                    help="which file in the run directory to score. odometry.tum is the "
                         "front-end alone, recorded beside the graph on every run.")
    args = ap.parse_args()

    cfg, mcfg = load_dataset(args.config), load_method(args.method)
    run = Path(args.run)
    metrics_path = run / metrics_name(args.trajectory_file)
    ev = cfg.eval
    stream_key = args.stream
    stream = cfg.stream(stream_key)
    agent = stream["agent"]

    out: dict = {
        "dataset": cfg.name, "method": mcfg.name, "agent": agent,
        "stream": stream_key,
        # The comparison block comes from the STREAM, not the method: one
        # estimator run on a LiDAR and on a depth camera belongs in two blocks,
        # and that pair is the whole point of running it twice.
        "modality": [stream["modality"]], "method_accepts": mcfg.modality,
        "family": mcfg.raw.get("family"), "origin": mcfg.raw.get("origin"),
        "loop_closure": bool(mcfg.raw.get("loop_closure", False)),
        "run_dir": str(run), "reference_kind": cfg.reference.get("kind"),
        "reference_uncertainty_m": cfg.reference_uncertainty_m(),
        "status": "ok",
    }
    manifest = run / "run.json"
    if manifest.exists():
        import json
        out["run"] = json.loads(manifest.read_text())

    # ------------------------------------------------------------- trajectory
    traj_file = run / args.trajectory_file
    if not traj_file.exists():
        if "trajectory" in mcfg.outputs:
            out["status"] = f"no trajectory.tum in {run}"
            write_json(out, metrics_path)
            print(f"error: {out['status']}", file=sys.stderr)
            return 1
        est_ref = None
    else:
        est_raw = load_tum(traj_file, name=f"{mcfg.name}/{stream_key}")
        E = mcfg.sensor_extrinsic(cfg, stream_key)
        est_ref = to_reference_frame(est_raw, E,
                                     cfg.agent_reference(agent).get("frame", ""))
        out["estimate"] = {
            "n_poses": len(est_raw), "duration_s": est_raw.duration,
            "lost_poses": est_raw.lost,          # rows the method marked lost; beside the ATE, never dropped
            "path_length_m": est_raw.path_length(),
            "extrinsic_applied": True,
            "extrinsic_translation_m": float(np.linalg.norm(E[:3, 3])),
        }

        ref = load_tum(args.reference, name=f"reference/{agent}")
        # Reference coverage: a method that lost tracking for half the sequence
        # must not be scored as though it ran it. The fraction is reported beside
        # every number so an ATE over 40% of the run is visible as one.
        span = (min(ref.stamps[-1], est_raw.stamps[-1]) - max(ref.stamps[0], est_raw.stamps[0]))
        out["estimate"]["sequence_fraction"] = float(max(0.0, span) / max(ref.duration, 1e-9))

        # Precedence: command line, then the METHOD, then the dataset default.
        # A method may declare its own alignment because the frame its poses
        # live in is a property of the method: the construction's v2c rung is
        # anchored to the surveyed map and must be scored UNALIGNED, or an se3
        # fit to the reference quietly hands it back the thing it is being
        # measured against. Its v1 rung is in the backbone's own frame and
        # must be aligned. Both are declared in configs/methods/pseudo_gt_*.
        mode = args.alignment or mcfg.raw.get("alignment") or ev.get("alignment", "se3")
        a = ate(est_ref, ref, mode=mode, max_gap_s=ev.get("max_gap_s", 0.25),
                n_first=ev.get("align_n_first"),
                reference_uncertainty_m=cfg.reference_uncertainty_m())
        out["ate"] = a.as_dict()
        out["rpe"] = [rpe(est_ref, ref, d["delta"], d["unit"],
                          max_gap_s=ev.get("max_gap_s", 0.25)).as_dict()
                      for d in ev.get("rpe_deltas", [])]

        est_aligned = a.alignment.apply(est_ref)
        # TIER 2 NEEDS A METRICALLY CORRECT ESTIMATE. A method that declares
        # `metric_scale: false` (monocular: it recovers shape, not size) placed
        # in the world by an se3 fit is a scale model of the room, and its
        # distance to a surveyed board is the scale error wearing the units of
        # an absolute error -- metres, for a 7 mm survey. So the anchor check
        # for such a method is run on the sim3 fit, and the row says that the
        # scale came from the REFERENCE: tier 2 is then no longer independent
        # evidence for that method, which is the honest thing to print.
        metric = mcfg.raw.get("metric_scale", True)
        anchor_mode = mode
        est_for_anchors = est_aligned
        if not metric and mode != "sim3":
            a_s = ate(est_ref, ref, mode="sim3", max_gap_s=ev.get("max_gap_s", 0.25),
                      n_first=ev.get("align_n_first"),
                      reference_uncertainty_m=cfg.reference_uncertainty_m())
            est_for_anchors = a_s.alignment.apply(est_ref)
            anchor_mode = "sim3"
        out["anchor_alignment"] = {
            "mode": anchor_mode, "method_has_metric_scale": bool(metric),
            "independent": bool(metric),
            **({} if metric else {"note": "scale taken from the reference; this row "
                                          "is not independent evidence"}),
        }
        anchors = cfg.anchors(agent)
        # Interpolation is decided by the TRAJECTORY, not the method: RTAB-Map's
        # odometry is 2227 poses and its loop-closed graph export is 95 nodes,
        # the same method either way. So the anchor check always evaluates
        # between poses when it has to, and always reports the gap it worked
        # across — the number is then judgeable instead of absent.
        out["absolute_check"] = (absolute_check(est_for_anchors, anchors,
                                                interpolate=True) if anchors else
                                 {"note": f"no anchor has a dwell window for {agent}; "
                                          "tier 2 is silent. Fill "
                                          "reference.anchors[*].windows_by_agent in the "
                                          "dataset config — without it no independent "
                                          "evidence enters this benchmark at all."})
        rv = ev.get("revisit", {})
        out["revisit"] = revisit_residual(est_aligned, rv.get("radius_m", 0.30),
                                          rv.get("min_separation_s", 20.0))
        out["_alignment_T"] = a.alignment.T.tolist()
        out["_alignment_scale"] = a.alignment.scale

    # -------------------------------------------------------------------- map
    map_file = run / "map.ply"
    ref_map = args.reference_map or cfg.reference.get("map_cloud")
    if args.skip_map or "map" not in mcfg.outputs:
        out["map_status"] = "not scored: the method does not output a dense map"
    elif not map_file.exists():
        out["map_status"] = f"not scored: no map.ply in {run}"
    elif not ref_map:
        out["map_status"] = ("not scored: reference.map_cloud is null. The map tier "
                             "needs the anchored map cloud; until it is supplied this "
                             "benchmark reports trajectory only.")
    elif ev.get("volume", {}).get("min") is None:
        out["map_status"] = ("not scored: eval.volume is null. The volume changes the "
                             "ranking, so it is a declared choice, not a default.")
    else:
        est_pts = load_cloud(map_file)
        if ev.get("map_frame", "aligned") == "aligned":
            if est_ref is None:
                raise SystemExit("map_frame: aligned needs a trajectory to take the "
                                 "alignment from; use map_frame: raw or supply one")
            T = np.array(out["_alignment_T"])
            est_pts = transform_points(est_pts * out["_alignment_scale"], T)
        m = evaluate_map(est_pts, load_cloud(Path(ref_map).expanduser()),
                         volume=cfg.volume(), resolution_m=ev.get("resolution_m", 0.02),
                         truncate_m=ev.get("truncate_m", 0.5),
                         thresholds_m=tuple(ev.get("fscore_thresholds_m", (0.02, 0.05, 0.10))))
        out["map"] = m.as_dict()
        out["map"]["frame"] = ev.get("map_frame", "aligned")
        out["map_status"] = "ok"

    write_json(out, metrics_path)
    _print(out)
    return 0


def metrics_name(trajectory_file: str) -> str:
    """`metrics.json` for the run's own trajectory.tum; scoring any other file in
    the run directory (odometry.tum, the front-end alone) writes beside it, so
    the graph's score is never overwritten by the front-end's."""
    stem = Path(trajectory_file).stem
    return "metrics.json" if trajectory_file == "trajectory.tum" else f"metrics_{stem}.json"


def _print(out: dict) -> None:
    print(f"{out['method']} / {out['agent']} / {out.get('stream')}")
    a = out.get("ate")
    if a:
        t = a["ate_trans_m"]
        print(f"  ATE  RMSE {t['rmse']*1e3:8.1f} mm   median {t['median']*1e3:8.1f} mm"
              f"   p90 {t['p90']*1e3:8.1f} mm   rot {a['ate_rot_deg']['rmse']:.2f} deg")
        sc = a["alignment"]["scale_observed"]
        ratio = 1.0 / sc if sc else float("nan")
        note = ("within 0.5% of the reference's size" if abs(ratio - 1) < 0.005 else
                f"{abs(ratio - 1) * 100:.1f}% too {'large' if ratio > 1 else 'small'}")
        print(f"       align={a['alignment']['mode']} "
              f"scale_observed={sc:.4f} (the estimate is {note}) "
              f"coverage={a['coverage']:.2f} drift={a['drift_percent']:.2f}%")
        if a.get("below_reference_uncertainty"):
            print(f"       NOTE: below the reference's own uncertainty "
                  f"({out['reference_uncertainty_m']*1e3:.0f} mm) — "
                  f"indistinguishable from it, not better than its rivals")
    for r in out.get("rpe", []):
        print(f"  RPE  {r['delta']:g} {r['unit']:1s}  "
              f"{r['rpe_trans_m']['rmse']*1e3:8.1f} mm   "
              f"{r['rpe_rot_deg']['rmse']:.3f} deg   ({r['n_pairs']} pairs)")
    ac = out.get("absolute_check")
    if isinstance(ac, list):
        aa = out.get("anchor_alignment") or {}
        for row in ac:
            d = row.get("residual_m")
            extra = ""
            if row.get("n"):
                extra = f" (n={row['n']}"
                if row.get("residual_deg") is not None:
                    extra += f", {row['residual_deg']:.2f} deg"
                if row.get("interp_gap_s"):
                    extra += f", interpolated, median gap {row['interp_gap_s']:.2f} s"
                extra += ")"
            print(f"  ANCHOR {row['anchor']:<10s} "
                  f"{'—' if d is None else f'{d*1e3:6.1f} mm'}   {row['verdict']}{extra}")
        if ac and not aa.get("independent", True):
            print(f"       anchors aligned {aa['mode']}: {aa['note']}")
    elif ac:
        print(f"  ANCHOR {ac['note']}")
    rv = out.get("revisit", {})
    if rv.get("n_revisits"):
        print(f"  REVISIT {rv['n_revisits']} pairs, residual RMSE "
              f"{rv['residual_m']['rmse']*1e3:.1f} mm")
    elif rv:
        print(f"  REVISIT {rv.get('note')}")
    m = out.get("map")
    if m:
        print(f"  MAP  chamfer {m['chamfer_m']*1e3:.1f} mm   "
              f"F@5cm {m['fscore']['0.05']['f']:.3f}   "
              f"truncated acc/compl {m['accuracy_truncated']:.2f}/"
              f"{m['completeness_truncated']:.2f}   ({m['n_est']} pts, frame={m['frame']})")
    elif out.get("map_status"):
        print(f"  MAP  {out['map_status']}")


if __name__ == "__main__":
    raise SystemExit(main())

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
    args = ap.parse_args()

    cfg, mcfg = load_dataset(args.config), load_method(args.method)
    run = Path(args.run)
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
    traj_file = run / "trajectory.tum"
    if not traj_file.exists():
        if "trajectory" in mcfg.outputs:
            out["status"] = f"no trajectory.tum in {run}"
            write_json(out, run / "metrics.json")
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

        mode = args.alignment or ev.get("alignment", "se3")
        a = ate(est_ref, ref, mode=mode, max_gap_s=ev.get("max_gap_s", 0.25),
                n_first=ev.get("align_n_first"),
                reference_uncertainty_m=cfg.reference_uncertainty_m())
        out["ate"] = a.as_dict()
        out["rpe"] = [rpe(est_ref, ref, d["delta"], d["unit"],
                          max_gap_s=ev.get("max_gap_s", 0.25)).as_dict()
                      for d in ev.get("rpe_deltas", [])]

        est_aligned = a.alignment.apply(est_ref)
        anchors = cfg.anchors(agent)
        out["absolute_check"] = (absolute_check(est_aligned, anchors) if anchors else
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

    write_json(out, run / "metrics.json")
    _print(out)
    return 0


def _print(out: dict) -> None:
    print(f"{out['method']} / {out['agent']} / {out.get('stream')}")
    a = out.get("ate")
    if a:
        t = a["ate_trans_m"]
        print(f"  ATE  RMSE {t['rmse']*1e3:8.1f} mm   median {t['median']*1e3:8.1f} mm"
              f"   p90 {t['p90']*1e3:8.1f} mm   rot {a['ate_rot_deg']['rmse']:.2f} deg")
        print(f"       align={a['alignment']['mode']} "
              f"scale_observed={a['alignment']['scale_observed']:.4f} "
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
        for row in ac:
            d = row.get("residual_m")
            print(f"  ANCHOR {row['anchor']:<10s} "
                  f"{'—' if d is None else f'{d*1e3:6.1f} mm'}   {row['verdict']}")
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

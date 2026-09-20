#!/usr/bin/env python3
"""Build one rung of the construction from an existing backbone run.

    python3 scripts/build_pseudo_gt.py --config configs/coop2.yaml \\
        --run runs/coop2_20260828/rtabmap_rgbd_imu/mobile_1.zed_rgbd/<id> --rung v1
    python3 scripts/build_pseudo_gt.py ... --rung v2c [--hold-out rs_anchor]

Design: docs/STAGE2_GRAPH.md. Rung v1 is odometry factors only -- the wiring
gate, and it must reproduce the backbone it was given. Rung v2c adds the
surveyed-board anchor priors; v2c minus v1 is what the anchors buy.

The output is a RUN DIRECTORY the existing evaluator scores unchanged:

    runs/<dataset>/pseudo_gt_<rung>/<stream>/<UTC>/{trajectory.tum, run.json, graph.json}

so `scripts/rescore.py` then puts it in the same tables, with the same three
tiers, as every backbone. run.json carries a `construction` block naming the
source run, the odometry sigmas and their provenance, the boards consumed and
held out, and the solve report -- a pseudo-GT number without those is a number
nobody can check.

WHAT IS REFUSED, per rule 3: a backbone run that has not been scored (the
odometry sigmas come from its own RPE, and a guessed sigma decides how hard the
anchors pull), an anchor with no declared `uncertainty_deg`, and a v2c with no
anchor factor at all (that is v1 with the gauge removed, and it is singular).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slambench.align import fit                                              # noqa: E402
from slambench.config import load_dataset, load_method, to_reference_frame  # noqa: E402
from slambench.graph import (anchor_factors, gauge_prior, odometry_factors,  # noqa: E402
                             reset_edges, solve)
from slambench.trajectory import Trajectory, load_tum, save_tum             # noqa: E402


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serialisable: {type(o)}")


def edge_sigmas_from_metrics(metrics: dict, n_poses: int, path_length_m: float) -> dict:
    """Per-edge sigmas from the run's own RPE at 1 m.

    Model: per-edge errors independent, so an RMSE over one metre of E edges
    is sigma * sqrt(E). It is a declared model, and it is REFERENCE-INFORMED
    -- the RPE was scored against the LiDAR -- so it sets a weight, not an
    independent measurement. run.json says so. Check the sensitivity (x0.5,
    x2) before quoting anything that depends on it.
    """
    r1 = next((r for r in metrics.get("rpe", []) if r.get("unit") == "m" and
               abs(float(r.get("delta", 0)) - 1.0) < 1e-9), None)
    if r1 is None:
        raise SystemExit("the run's metrics carry no RPE at 1 m; re-score it with "
                         "eval.rpe_deltas including {delta: 1.0, unit: m}, or pass "
                         "--sigma-t/--sigma-r")
    epm = max((n_poses - 1) / max(path_length_m, 1e-9), 1.0)
    st = float(r1["rpe_trans_m"]["rmse"]) / np.sqrt(epm)
    sr = float(np.radians(r1["rpe_rot_deg"]["rmse"])) / np.sqrt(epm)
    return {"sigma_t_m": st, "sigma_r_deg": float(np.degrees(sr)),
            "source": "run's own RPE at 1 m / sqrt(edges per metre)",
            "rpe_1m_trans_rmse_m": float(r1["rpe_trans_m"]["rmse"]),
            "rpe_1m_rot_rmse_deg": float(r1["rpe_rot_deg"]["rmse"]),
            "edges_per_metre": float(epm), "reference_informed": True}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, help="a backbone run directory (has run.json)")
    ap.add_argument("--rung", required=True, choices=["v1", "v2c"])
    ap.add_argument("--hold-out", action="append", default=[], metavar="BOARD",
                    help="v2c: leave this board's observations OUT of the graph so the "
                         "evaluator's tier 2 against it stays independent. Repeatable.")
    ap.add_argument("--source", default="odometry.tum",
                    help="which file of the run is the backbone. odometry.tum is the dense "
                         "front-end; trajectory.tum is the loop-closed graph, which on "
                         "mobile_1 stops 27 s early (docs/STAGE2_GRAPH.md s2.1).")
    ap.add_argument("--sigma-t", type=float, default=None, help="per-edge translation sigma, m")
    ap.add_argument("--sigma-r", type=float, default=None, help="per-edge rotation sigma, deg")
    ap.add_argument("--reset-gap-factor", type=float, default=3.0)
    ap.add_argument("--reset-inflate", type=float, default=100.0)
    ap.add_argument("--anchor-max-gap-s", type=float, default=0.2,
                    help="an observation is interpolated onto a keyframe only across a "
                         "gap narrower than this")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--methods-dir", default=str(ROOT / "configs" / "methods"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_dataset(args.config)
    run = Path(args.run)
    manifest = json.loads((run / "run.json").read_text())
    method, stream_key, agent = manifest["method"], manifest["stream"], manifest["agent"]
    mcfg = load_method(Path(args.methods_dir) / f"{method}.yaml")

    # ---------------------------------------------------- the backbone, re-expressed
    src = run / args.source
    if not src.exists():
        raise SystemExit(f"{src} does not exist")
    est_raw = load_tum(src, name=f"{method}/{stream_key}")
    E = mcfg.sensor_extrinsic(cfg, stream_key)
    ref_frame = cfg.agent_reference(agent).get("frame", "")
    est = to_reference_frame(est_raw, E, ref_frame)
    N = len(est)
    print(f"backbone : {src}  {N} poses, {est.duration:.1f} s, {est.path_length():.2f} m "
          f"-> {ref_frame or 'reference frame'}")

    # ----------------------------------------------------------------- sigmas
    if args.sigma_t is not None and args.sigma_r is not None:
        sig = {"sigma_t_m": args.sigma_t, "sigma_r_deg": args.sigma_r,
               "source": "declared on the command line", "reference_informed": False}
    elif args.sigma_t is not None or args.sigma_r is not None:
        raise SystemExit("--sigma-t and --sigma-r go together")
    else:
        stem = Path(args.source).stem
        mfile = run / ("metrics.json" if args.source == "trajectory.tum" else f"metrics_{stem}.json")
        if not mfile.exists():
            raise SystemExit(f"{mfile} does not exist: the odometry sigmas come from the run's "
                             f"own RPE. Score the run first (scripts/rescore.py), or declare "
                             f"--sigma-t/--sigma-r.")
        sig = edge_sigmas_from_metrics(json.loads(mfile.read_text()), N, est.path_length())
        sig["metrics_file"] = str(mfile)
    sigma_t, sigma_r = float(sig["sigma_t_m"]), float(np.radians(sig["sigma_r_deg"]))
    print(f"sigma    : {sigma_t * 1e3:.2f} mm / {sig['sigma_r_deg']:.4f} deg per edge  "
          f"({sig['source']})")

    factors = odometry_factors(est.poses, est.stamps, sigma_t, sigma_r,
                               args.reset_gap_factor, args.reset_inflate)
    resets = reset_edges(est.stamps, args.reset_gap_factor)
    print(f"odometry : {len(factors.b_i)} between factors, {len(resets)} reset edges "
          f"(x{args.reset_inflate:g} sigma)")

    construction = {
        "rung": args.rung, "source_run": str(run), "source_file": args.source,
        "backbone_method": method, "n_poses": N, "sigma": sig,
        "reset_edges": {"count": int(len(resets)), "gap_factor": args.reset_gap_factor,
                        "inflate": args.reset_inflate,
                        "stamps": [float(est.stamps[i]) for i in resets]},
        "anchors": {"consumed": [], "held_out": list(args.hold_out), "skipped": []},
    }

    # ---------------------------------------------------------------- anchors
    if args.rung == "v1":
        factors = factors.extend(gauge_prior(est.poses, 0))
        poses0 = est.poses
        construction["frame"] = "backbone world, gauge pinned at pose 0"
        construction["initial_placement"] = "none (v1 keeps the backbone's own frame)"
    else:
        for a in cfg.anchors(agent):
            name = a["name"]
            if name in args.hold_out:
                continue
            obs = a.get("observed_poses")
            if obs is None:
                construction["anchors"]["skipped"].append(
                    {"anchor": name, "why": a.get("observed_poses_error") or
                     "no observed_poses declared for this agent"})
                continue
            if a.get("uncertainty_deg") is None:
                raise SystemExit(f"reference.anchors[{name}].uncertainty_deg is null. The anchor "
                                 f"factor weights rotation and position separately and "
                                 f"needs both; declare it with its provenance (rule 3).")
            fa, info = anchor_factors(obs.poses, obs.stamps, est.stamps,
                                      float(a["uncertainty_m"]),
                                      float(np.radians(a["uncertainty_deg"])), name,
                                      max_gap_s=args.anchor_max_gap_s, window=a.get("window"))
            if info["factors"] == 0:
                info["why"] = "no keyframe inside the window received an observation"
                construction["anchors"]["skipped"].append(info)
                continue
            construction["anchors"]["consumed"].append(info)
            factors = factors.extend(fa)
        if not construction["anchors"]["consumed"]:
            raise SystemExit("v2c with no anchor factor is v1 with the gauge removed. "
                             f"Skipped: {construction['anchors']['skipped']}")
        # Initial placement: the backbone rigidly fitted onto its own anchor
        # observations (docs/STAGE2_GRAPH.md s4). The anchors then fix the
        # gauge, so NO first-pose prior -- it would fight them.
        order = np.argsort(factors.p_i)
        pi = factors.p_i[order]
        obs_T = np.tile(np.eye(4), (len(pi), 1, 1))
        obs_T[:, :3, :3] = factors.p_R[order]; obs_T[:, :3, 3] = factors.p_t[order]
        al = fit(Trajectory(est.stamps[pi], est.poses[pi]),
                 Trajectory(est.stamps[pi], obs_T), mode="se3")
        est_map = al.apply(est)
        poses0 = est_map.poses
        d0 = np.linalg.norm(poses0[pi, :3, 3] - obs_T[:, :3, 3], axis=1)
        construction["frame"] = "map (anchored by the surveyed boards)"
        construction["initial_placement"] = {
            "mode": "se3 fit of the backbone onto its anchor observations",
            "n_pairs": int(len(pi)),
            "residual_before_solve_mm": {"median": float(np.median(d0) * 1e3),
                                         "p90": float(np.percentile(d0, 90) * 1e3)}}
        print(f"anchors  : " + ", ".join(f"{c['anchor']} x{c['factors']}"
                                         for c in construction["anchors"]["consumed"]) +
              (f"  held out: {args.hold_out}" if args.hold_out else "") +
              (f"  skipped: {[s_['anchor'] for s_ in construction['anchors']['skipped']]}"
               if construction["anchors"]["skipped"] else ""))
        print(f"placement: rigid fit on {len(pi)} pairs, residual median "
              f"{np.median(d0) * 1e3:.1f} mm before solving")

    # ------------------------------------------------------------------ solve
    out_poses, rep = solve(poses0, factors, verbose=args.verbose)
    construction["factor_counts"] = factors.counts()
    construction["solve"] = rep.as_dict()
    moved = np.linalg.norm(out_poses[:, :3, 3] - poses0[:, :3, 3], axis=1)
    construction["moved_from_initial_mm"] = {"median": float(np.median(moved) * 1e3),
                                             "max": float(moved.max() * 1e3)}
    print(f"solve    : {rep.iterations} it, converged={rep.converged}, cost "
          f"{rep.initial_cost:.3e} -> {rep.final_cost:.3e}, whitened rms "
          + ", ".join(f"{k} {v:.2f}" for k, v in rep.final_rms.items()))
    print(f"moved    : median {np.median(moved) * 1e3:.1f} mm, max {moved.max() * 1e3:.1f} mm "
          f"from the initial placement")
    for k, v in rep.final_rms.items():
        if v > 2.0:
            print(f"WARNING  : whitened {k} rms {v:.2f} > 2: the declared sigmas do not "
                  f"describe this data. Do not quote the result before understanding why.")

    # ------------------------------------------------------------------ write
    name = f"pseudo_gt_{args.rung}"
    # One directory per build, never overwritten (same rule as run_method.py).
    # The stamp is one-second resolution and a v1 build takes well under a
    # second, so a loop over four backbone runs collides -- it did, on the
    # fourth, 2026-09-20. Suffix rather than overwrite: a directory describing
    # a different build is worse than an extra one.
    cell = Path(args.runs_root) / cfg.name / name / stream_key
    base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir, n_try = cell / base, 0
    while out_dir.exists():
        n_try += 1
        out_dir = cell / f"{base}-{n_try}"
    out_dir.mkdir(parents=True, exist_ok=False)
    save_tum(Trajectory(est.stamps, out_poses, name=f"{name}/{stream_key}",
                        frame=ref_frame, world=construction["frame"]), out_dir / "trajectory.tum")
    run_json = {
        "dataset": cfg.name, "method": name, "agent": agent, "stream": stream_key,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "executed": True, "returncode": 0, "preflight_problems": [],
        "pose_frame": ref_frame, "construction": construction,
    }
    (out_dir / "run.json").write_text(json.dumps(run_json, indent=2, default=_json_default))
    (out_dir / "graph.json").write_text(json.dumps(construction, indent=2, default=_json_default))
    latest = out_dir.parent / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    os.symlink(out_dir.name, latest)
    print(f"\n{N} poses -> {out_dir / 'trajectory.tum'}\n"
          f"score it : python3 scripts/rescore.py --runs {args.runs_root}/{cfg.name} "
          f"--config {args.config}")
    if args.rung == "v2c":
        print("           tier 2 against a CONSUMED board is self-agreement; the LiDAR ATE "
              "and any held-out board are the independent numbers.\n"
              "           add --alignment none to eval_run.py to see the error in the "
              "map frame itself, without the se3 fit.")
    return 0


if __name__ == "__main__":                                              # pragma: no cover
    sys.exit(main())

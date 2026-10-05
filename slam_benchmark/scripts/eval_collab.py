#!/usr/bin/env python3
"""Score one collaborative run: the inter-agent transform, and what the pair added.

    python3 scripts/eval_collab.py --config configs/coop2.yaml \
        --method configs/methods/swarm_slam.yaml \
        --run runs/coop2_20260828/swarm_slam \
        --reference-dir runs/coop2_20260828/reference \
        [--solo runs/coop2_20260828/kiss_icp/mobile_1.ouster]

The run directory holds one `<agent>.tum` per agent, all in the METHOD'S OWN
frame — not in `map`. That is the whole point: the headline number needs no
alignment, so a system that lands in an arbitrary frame is scored exactly as
well as one that happens to land in the reference's.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slambench import (agent_separation, completeness_gain, evaluate_map,  # noqa: E402
                       load_cloud, load_dataset, load_method, load_tum,
                       place_overlap, relative_trajectory_error, to_reference_frame)
from slambench.report import write_json                                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--solo", default=None,
                    help="a single-agent run directory whose map is the solo "
                         "baseline for the completeness gain")
    ap.add_argument("--agents", default=None, help="comma-separated; default: both")
    args = ap.parse_args()

    cfg, mcfg = load_dataset(args.config), load_method(args.method)
    run, refdir = Path(args.run), Path(args.reference_dir)
    ev = cfg.eval

    streams = mcfg.streams
    by_agent = {cfg.stream(s)["agent"]: s for s in streams}
    agents = args.agents.split(",") if args.agents else sorted(by_agent)
    if len(agents) != 2:
        print(f"error: the collaborative track scores a PAIR; got {agents}",
              file=sys.stderr)
        return 2
    a, b = agents

    est, ref = {}, {}
    for name in (a, b):
        f = run / f"{name}.tum"
        if not f.exists():
            print(f"error: no {f}. Each agent's trajectory goes in the run "
                  f"directory as <agent>.tum, in the method's own frame.",
                  file=sys.stderr)
            return 1
        E = mcfg.sensor_extrinsic(cfg, by_agent[name])
        est[name] = to_reference_frame(load_tum(f, name=f"{mcfg.name}/{name}"), E)
        ref[name] = load_tum(refdir / f"{name}.tum", name=f"reference/{name}")

    out = {
        "dataset": cfg.name, "method": mcfg.name, "track": mcfg.raw.get("track"),
        "agents": [a, b], "streams": {n: by_agent[n] for n in (a, b)},
        "modality": sorted({cfg.stream(by_agent[n])["modality"] for n in (a, b)}),
        "heterogeneous": len({cfg.stream(by_agent[n])["modality"]
                              for n in (a, b)}) > 1,
        "reference_uncertainty_m": cfg.reference_uncertainty_m(),
        "run_dir": str(run),
    }
    for extra in ("run.json", "inter_robot_matches.json"):
        f = run / extra
        if f.exists():
            out[extra.replace(".json", "")] = json.loads(f.read_text())

    rel = relative_trajectory_error(est[a], est[b], ref[a], ref[b],
                                    max_gap_s=ev.get("max_gap_s", 0.25))
    out["relative"] = rel.as_dict()

    # Context, from the reference alone: an inter-agent error is read very
    # differently at 2 m of separation than at 16 m, and a method that found no
    # shared place cannot be faulted for not merging.
    rng = cfg.stream(by_agent[b]).get("range_m", [0.2, 10.0])[1]
    out["context"] = {"separation": agent_separation(ref[a], ref[b]),
                      "place_overlap": place_overlap(ref[a], ref[b], radius_m=float(rng))}

    # The map gain, only when there is a solo run to read it against.
    joint_map, ref_map = run / "map.ply", cfg.reference.get("map_cloud")
    if args.solo and joint_map.exists() and ref_map and ev.get("volume", {}).get("min"):
        solo_map = Path(args.solo) / "map.ply"
        if solo_map.exists():
            rc = load_cloud(Path(ref_map).expanduser())
            kw = dict(volume=cfg.volume(), resolution_m=ev.get("resolution_m", 0.02),
                      truncate_m=ev.get("truncate_m", 0.5),
                      thresholds_m=tuple(ev.get("fscore_thresholds_m",
                                                (0.02, 0.05, 0.10))))
            out["map_gain"] = completeness_gain(
                evaluate_map(load_cloud(solo_map), rc, **kw).as_dict(),
                evaluate_map(load_cloud(joint_map), rc, **kw).as_dict())
            out["map_gain"]["solo_run"] = str(args.solo)
        else:
            out["map_gain_status"] = f"no map.ply in {args.solo}"
    else:
        out["map_gain_status"] = ("not scored: needs --solo, a joint map.ply, "
                                  "reference.map_cloud and eval.volume. Without "
                                  "the solo baseline a joint map score is not "
                                  "interpretable — see collab.completeness_gain.")

    write_json(out, run / "metrics_collab.json")
    _print(out, rel)
    return 0


def _print(out, rel) -> None:
    a, b = out["agents"]
    het = " (heterogeneous)" if out["heterogeneous"] else ""
    print(f"{out['method']}  {a} + {b}{het}   "
          f"{' / '.join(out['streams'][n] for n in (a, b))}")
    print(f"  RELATIVE POSE  (gauge-free, no alignment)")
    print(f"    translation  RMSE {rel.trans.rmse * 1e3:8.1f} mm   "
          f"median {rel.trans.median * 1e3:8.1f} mm   p90 {rel.trans.p90 * 1e3:8.1f} mm")
    print(f"    rotation     RMSE {rel.rot.rmse:8.2f} deg")
    print(f"    over {rel.n} shared stamps, {rel.time_span_s:.0f} s, "
          f"separation {rel.separation_m['min']:.1f}..{rel.separation_m['max']:.1f} m")
    print(f"  DECOMPOSITION  (gauge fixed on {a})")
    print(f"    constant     {rel.constant_trans_m * 1e3:8.1f} mm   "
          f"{rel.constant_rot_deg:.2f} deg     <- map merge / place recognition")
    print(f"    varying      {rel.trans_after_constant.rmse * 1e3:8.1f} mm   "
          f"{rel.rot_after_constant.rmse:.2f} deg     <- relative drift")
    print(f"    -> {rel.verdict()}")
    ov = out["context"]["place_overlap"]
    print(f"  CONTEXT  {ov['verdict']}")
    m = out.get("map_gain")
    if m:
        print(f"  MAP GAIN  completeness {m['completeness_gain_m'] * 1e3:+.1f} mm; "
              f"recall@5cm {m['0.05']['recall_solo']:.3f} -> "
              f"{m['0.05']['recall_joint']:.3f} "
              f"({m['0.05']['recall_gain']:+.3f})")
    else:
        print(f"  MAP GAIN  {out.get('map_gain_status')}")
    n = out.get("inter_robot_matches")
    if n is not None:
        print(f"  INTER-ROBOT MATCHES  {n}")
    elif out.get("track") == "collaborative":
        print(f"  INTER-ROBOT MATCHES  not reported by the container. Without it, "
              f"a good\n                       relative pose cannot be "
              f"distinguished from two frames that\n                       "
              f"happened to agree.")


if __name__ == "__main__":
    raise SystemExit(main())

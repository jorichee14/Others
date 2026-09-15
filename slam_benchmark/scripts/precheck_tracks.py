#!/usr/bin/env python3
"""Answer each track's precondition BEFORE any SLAM system is scheduled.

    python3 scripts/precheck_tracks.py --config configs/coop2.yaml \
        --reference-dir runs/coop2_20260828/reference [--track all]

Every check here runs on the reference trajectories (and, for track A, one
sample scan) — no method, no container, no GPU, minutes not hours. Two of the
three preconditions can come back "not runnable on this sequence", and finding
that out from six flat result rows a week later is the expensive way.

Exit code is 1 if any checked precondition fails, so it can gate a sweep.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slambench import (Ablation, ablate, agent_separation, extrinsic_matrix,  # noqa: E402
                       load_cloud, load_dataset, load_tum, observability,
                       place_overlap, predict_observation, select_rate)
from slambench.report import write_json                                  # noqa: E402

OK, BAD, WARN = "PASS", "FAIL", "WARN"


def _hdr(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def check_collaborative(cfg, refs, tracks) -> dict:
    _hdr("TRACK B — collaborative SLAM: did the agents ever share a place?")
    tc = tracks["collaborative"]
    a_name, b_name = tc["agents"]
    if a_name not in refs or b_name not in refs:
        print(f"{BAD}  missing reference trajectory for "
              f"{[n for n in (a_name, b_name) if n not in refs]}")
        return {"status": BAD, "reason": "reference trajectory missing"}
    A, B = refs[a_name], refs[b_name]

    # The radius is the WEAKER sensor's usable range: overlap only counts where
    # the constrained agent could actually have seen the shared place.
    rng = cfg.stream(tc["streams"][b_name]).get("range_m", [0.2, 10.0])[1]

    sep = agent_separation(A, B)
    ov = place_overlap(A, B, radius_m=float(rng))
    print(f"  time overlap        {sep['time_overlap_s']:.1f} s over {sep['n']} stamps")
    print(f"  separation          min {sep['min_m']:.2f} m   median "
          f"{sep['median_m']:.2f} m   max {sep['max_m']:.2f} m")
    print(f"  closest approach    {ov['closest_approach_m']:.2f} m "
          f"(radius {rng} m = {b_name}'s usable range)")
    print(f"  b's path near a's   {100 * ov['fraction_of_b_near_a']:.1f}%")
    if "time_offset_s" in ov:
        t = ov["time_offset_s"]
        print(f"  revisit delay       median {t['median']:.1f} s, max {t['max']:.1f} s")
    print(f"\n  {ov['verdict']}")

    frac = ov["fraction_of_b_near_a"]
    if frac < 0.10:
        status, note = BAD, (
            "Below 10%: inter-robot place recognition has almost nothing to "
            "match on. A collaborative result here would measure each method's "
            "fallback behaviour, not its collaboration. Fix the ROUTE in the "
            "next session — send both platforms through one shared corridor — "
            "rather than widening the radius until the number looks acceptable.")
    elif frac < 0.30:
        status, note = WARN, (
            "Thin. The track can run, but a method that finds no inter-robot "
            "loop closure is reporting a property of this sequence and must be "
            "labelled that way, not scored as a failure.")
    else:
        status, note = OK, "Enough shared geometry for the track to mean something."
    print(f"\n  {status}: {note}")
    return {"status": status, "separation": sep, "overlap": ov, "note": note}


def check_infrastructure(cfg, refs, tracks) -> dict:
    _hdr("TRACK C — infrastructure: were the agents where the node could see them?")
    st = cfg.stream(tracks["infrastructure"]["node"] + ".radar")
    pose = extrinsic_matrix(st["static_world_pose"], "infra_1.static_world_pose")
    out, worst = {}, OK
    for name, traj in sorted(refs.items()):
        # infra_1's static_world_pose describes the Arducam OPTICAL frame
        # (z forward, y down), not a body frame. See predict_observation.
        p = predict_observation(traj, pose, axes="optical")
        r = p["range_m"]
        az, el = p["azimuth_deg"], p["elevation_deg"]
        # The node's own FoV is not declared anywhere; +/-45 deg azimuth is a
        # plain USB camera's order and is used only to say how much of the run
        # is plausibly observable. Replace it with the measured FoV before
        # quoting any coverage number.
        seen = np.abs(az) <= 45.0
        print(f"  {name:10s} range {r.min():5.2f}..{r.max():5.2f} m   "
              f"azimuth {az.min():7.1f}..{az.max():7.1f} deg   "
              f"elevation {el.min():6.1f}..{el.max():6.1f} deg")
        print(f"             within +/-45 deg azimuth for "
              f"{100 * seen.mean():.0f}% of its poses")
        out[name] = {"range_m": [float(r.min()), float(r.max())],
                     "azimuth_deg": [float(az.min()), float(az.max())],
                     "fraction_in_45deg": float(seen.mean())}
        if seen.mean() < 0.2:
            worst = BAD
        elif seen.mean() < 0.5 and worst == OK:
            worst = WARN
    print(f"\n  {WARN}: the node's pose UNCERTAINTY is not stated anywhere in "
          f"coop2, and an infrastructure-anchored fix cannot be better than its "
          f"anchor. Until it is a number, track C reports improvement over "
          f"ego-only and not an absolute accuracy.")
    print(f"  {WARN}: the Arducam publishes camera_info at 27.5 Hz against a "
          f"10.6 Hz image stream. Confirm both carry the same frame_id with "
          f"inspect_bag.py before projecting anything into that image.")
    print(f"\n  {worst}: geometric visibility above; the two warnings stand "
          f"regardless of it.")
    return {"status": worst, "per_agent": out,
            "unresolved": ["infra pose uncertainty", "arducam camera_info frame_id"]}


def check_sensor_constrained(cfg, tracks, cloud_path) -> dict:
    _hdr("TRACK A — sensor-constrained: does the geometry survive the ablation?")
    grid = yaml.safe_load((ROOT / "configs" / tracks["sensor_constrained"]["grid"]).read_text())
    if not cloud_path:
        print(f"  {WARN}: needs one sample scan in the sensor frame to answer.\n"
              f"         Export any single Ouster sweep to .ply/.pcd and pass\n"
              f"         --cloud <file>. Until then the grid is unverified and\n"
              f"         a tight cell may be measuring the room, not the sensor.")
        return {"status": WARN, "reason": "no sample scan supplied"}

    pts = load_cloud(cloud_path)
    fr = grid.get("frame", {})
    fwd, up = fr.get("forward", [1, 0, 0]), fr.get("up", [0, 0, 1])
    print(f"  sample scan: {len(pts)} points from {cloud_path}\n")
    print(f"  {'cell':<26} {'points':>8} {'weakest':>8} {'isotropy':>9} {'verdict'}")
    print(f"  {'-' * 26} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 28}")

    rows, degenerate, starved = [], [], []
    for c in grid["cells"]:
        kw = {k: v for k, v in c.items()
              if k not in ("name", "total_beams", "rate_hz")}
        ab = Ablation(**kw, name=c["name"], seed=grid.get("seed", 0))
        sub = ablate(pts, ab, forward=fwd, up=up)
        if len(sub) < 50:
            print(f"  {c['name']:<26} {len(sub):>8} {'—':>8} {'—':>9} "
                  f"empty after ablation")
            rows.append({"cell": c["name"], "n": len(sub), "status": BAD})
            starved.append(c["name"])
            continue
        # "auto": the voxel scales with density, so the density cells measure
        # the sensor rather than the normal estimator's minimum points per cell.
        o = observability(sub, origin=(0, 0, 0), voxel_m="auto")
        deg = o.degenerate(0.05)
        d = o.trans_weakest_direction
        if o.starved():
            v = f"too sparse to judge ({o.n_points} surfaces)"
        elif deg:
            v = f"DEGENERATE, free along [{d[0]:+.2f} {d[1]:+.2f} {d[2]:+.2f}]"
        else:
            v = "constrained"
        print(f"  {c['name']:<26} {len(sub):>8} {o.trans_weakest:>8.3f} "
              f"{o.normal_isotropy:>9.3f} {v}")
        rows.append({"cell": c["name"], "n": int(len(sub)),
                     "observability": o.as_dict(), "degenerate": bool(deg),
                     "starved": bool(o.starved())})
        if deg:
            degenerate.append(c["name"])
        elif o.starved():
            starved.append(c["name"])

    print(f"\n  Measured on ONE scan, so this bounds the grid rather than "
          f"characterising the run;\n  the sweep itself reports the per-frame "
          f"degenerate fraction for every cell.")
    if degenerate:
        print(f"\n  {WARN}: {len(degenerate)} cell(s) are already degenerate on "
              f"this scan: {', '.join(degenerate)}.\n"
              f"         A method's curve is not interpretable there — the "
              f"geometry stopped\n         constraining the pose, whatever the "
              f"estimator did. Either stop the grid\n         before them, or "
              f"report them as the observability floor rather than as\n"
              f"         method failures.")
        status = WARN
    else:
        print(f"\n  {OK}: every cell still constrains the pose on this scan.")
        status = OK
    if starved:
        print(f"\n  {WARN}: {len(starved)} cell(s) are too sparse for the normal "
              f"estimate to judge: {', '.join(starved)}.\n"
              f"         That is a statement about the INSTRUMENT, not about the "
              f"geometry — do not\n         report them as degenerate. A method "
              f"may still run there; nothing here can say\n         whether its "
              f"failure would be attributable.")
    return {"status": status, "cells": rows, "degenerate_cells": degenerate,
            "starved_cells": starved}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tracks", default=str(ROOT / "configs" / "tracks.yaml"))
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--cloud", default=None, help="one sample scan, sensor frame")
    ap.add_argument("--track", default="all",
                    choices=["all", "sensor_constrained", "collaborative",
                             "infrastructure"])
    ap.add_argument("--out", default=None, help="write the findings as JSON")
    args = ap.parse_args()

    cfg = load_dataset(args.config)
    tracks = yaml.safe_load(Path(args.tracks).read_text())
    refs = {}
    for f in sorted(Path(args.reference_dir).glob("*.tum")):
        refs[f.stem] = load_tum(f, name=f"reference/{f.stem}")
    if not refs:
        print(f"no *.tum in {args.reference_dir} — run scripts/export_reference.py "
              f"for each agent first", file=sys.stderr)
        return 2
    print(f"references: " + ", ".join(f"{k} ({len(v)} poses, {v.duration:.0f} s, "
                                      f"{v.path_length():.1f} m)"
                                      for k, v in sorted(refs.items())))

    res = {}
    want = ("sensor_constrained", "collaborative", "infrastructure") \
        if args.track == "all" else (args.track,)
    if "sensor_constrained" in want:
        res["sensor_constrained"] = check_sensor_constrained(cfg, tracks, args.cloud)
    if "collaborative" in want:
        res["collaborative"] = check_collaborative(cfg, refs, tracks)
    if "infrastructure" in want:
        res["infrastructure"] = check_infrastructure(cfg, refs, tracks)

    _hdr("SUMMARY")
    for k, v in res.items():
        print(f"  {v['status']:<5} {k}")
    if args.out:
        write_json(res, args.out)
        print(f"\n  -> {args.out}")
    return 1 if any(v["status"] == BAD for v in res.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())

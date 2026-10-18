#!/usr/bin/env python3
"""The checks of maps and trajectories, one command:

  python3 check.py <command> [args]      python3 check.py <command> -h

  map     <map.pcd>        surface thickness: walls, floors, doubled surfaces
  walls   <config>         the scans as placed at the map's wall spots: one
                           scan, 1 s ... whole run, the map
  maps    <A.pcd> <B.pcd>  completeness / accuracy of B against A
  scans   <config>         single scans and pairs: noise, deskew, time offset,
                           drift
  trajs   <config>         a pass's trajectories against each other / a
                           reference: ATE, RPE, heading vs turning
  time    <config>         is GLIM's pose time its scans'? (on time / OFF)
  imu     <config>         is the IMU on the LiDAR's time?
  loop    <config>         loop closure on the boards
  report  <config>         a mapping pass's results: GLIM, + time correction,
                           + scan-to-map, + BA against the reference; the TSDF
                           and union maps

The code: analysis/maps.py (map, walls, maps, scans) and analysis/trajs.py
(trajs, time, imu, loop); report is below. run_pass.py runs `time` and
`report` as steps of a pass.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, os.path.join(ROOT, "map_stages"))

import maps                                                              # noqa: E402
import trajs                                                             # noqa: E402
from pipeline_common import load_pipeline, load_traj                     # noqa: E402

DELTAS = (1.0, 10.0, 100.0)
REPORT_DOC = r'''A mapping pass's results in one place: its trajectories against the
reference, and its maps. The last step of a mapping pass in run_pass.py.

    python3 check.py report <pipeline config> [--samples 20000] [--no-loop]

Trajectories (odometry/<machine>/), each against the reference:
  GLIM                    GLIM's own trajectory (its dump in work/), as it
                          computed it
  GLIM + time correction  traj_lidar.txt: the same, its poses moved by
                          recording.yaml's glim.pose_time_offset on export
                          (left out when that is 0: then it is GLIM's own)
  GLIM + scan-to-map      02b's traj_lidar_refined.txt (from the corrected GLIM)
  GLIM + BA (reference)   02's traj_lidar_ba.txt (02_bundle_adjust.output)
per trajectory: ATE after the best rigid fit and RPE over 1, 10, 100 s
(RMSE, cm / deg, Sturm et al. 2012), its heading error against the
reference vs the turning (check.py time: dt, r), its path length, and
its loop closure on the boards (05 --check, run here when its report is
not there yet; --no-loop skips that).

Maps (mapping/), both from the reference poses:
  TSDF   01's map (denoised_<tag>.pcd, 01_build_map.fusion "tsdf"): the
         reference map
  union  01 --union's (denoised_union_<tag>.pcd): the comparison
per map: points, surface thickness (check.py map: walls,
floors, all; every layer of a patch on its own), patches holding one surface
twice (%), and the TSDF's completeness against the union (within 10 cm).

Writes mapping_report_<tag>.json and .md in the pass's processed folder.
'''


def trajectories(P):
    """[(key, label, file)] of the pass, the reference last."""
    corr = P.seed_traj()
    applied = trajs.applied_offset(corr)
    s2m = (P.cfg.get("02b_scan_to_map") or {}).get("output") or "traj_lidar_refined.txt"
    ba = trajs.reference_traj(P)
    rows = [("glim", "GLIM", P.glim_traj() if applied else corr)]
    if applied:
        rows.append(("glim_corr", "GLIM + time correction (%+.0f ms)" % (applied * 1000), corr))
    rows += [("glim_s2m", "GLIM + scan-to-map (02b)", P.outp(s2m)),
             ("glim_ba", "GLIM + BA (02, reference)", ba)]
    return rows, applied


def loop_closure(P, config, key, path, run):
    """The trajectory's loop closure on the anchor board (or the first board
    revisited): dict(board, mm, deg, noise_mm, significant), or None."""
    odo = lambda n: trajs.odometry_file(P, n)                              # noqa: E731
    report = {"glim": odo("glim_raw_loop_closure.json"),
              "glim_corr": odo("glim_loop_closure.json")}.get(
        key, os.path.splitext(path)[0] + "_loop_closure.json")
    if os.path.isfile(report):
        summ = trajs.loop_summary(report)
    elif run:
        summ = trajs.loop_check(config, path, report=report, label=key)
    else:
        return None
    if not summ:
        return None
    boards = summ["boards"]
    order = [summ.get("anchor_board")] + list(boards)
    for b in order:
        v = boards.get(b) or {}
        if v.get("mm") is not None:
            return dict(board=b, mm=v["mm"], deg=v["deg"], noise_mm=v["noise_mm"],
                        significant=v["significant"])
    return None


def map_stats(path, positions, n):
    pts = maps.load(path)
    st, _ = maps.patch_stats(pts, n_samples=n, positions=positions)
    out = maps.summary(st) if len(st) else {}
    out.update(file=path, points=int(len(pts)))
    return out, pts


def fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def report(argv=None):
    ap = argparse.ArgumentParser(description=REPORT_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--samples", type=int, default=20000, help="check.py map patches per map")
    ap.add_argument("--no-loop", action="store_true", help="do not run missing loop checks")
    a = ap.parse_args(argv)

    P = load_pipeline(a.config)
    rows, applied = trajectories(P)
    ref_key, _, ref_file = rows[-1]
    if not os.path.isfile(ref_file):
        raise SystemExit(f"no reference trajectory at {ref_file} (run 02 first)")
    tr, Tr = load_traj(ref_file)
    report = dict(pass_tag=P.tag, reference=ref_file, pose_time_offset_s=applied,
                  trajectories=[], maps={})

    print(f"reference: {ref_file} ({len(tr)} poses, {trajs.path_length(Tr):.1f} m)\n")
    for key, label, f in rows:
        if not os.path.isfile(f):
            print(f"{label}: none at {f}" + (" (a comparison: \"comparisons\": true)"
                                             if key == "glim_s2m" else ""))
            if key != "glim_s2m":
                report["trajectories"].append(dict(key=key, label=label, file=f, missing=True))
            continue
        t, T = load_traj(f)
        r = dict(key=key, label=label, file=f, poses=int(len(t)), path_m=trajs.path_length(T))
        if key != ref_key:
            c, _ = trajs.compare(key, ref_key, t, T, tr, Tr)
            r["ate_cm"] = c["after_rigid_fit_cm"]["rmse"]
            r["ate_deg"] = c["after_rigid_fit_deg"]["rmse"]
            r["rpe"] = {f"{d:g}s": (c["relative"][f"{d:g}s"]["translation_cm"]["rmse"],
                                    c["relative"][f"{d:g}s"]["rotation_deg"]["rmse"])
                        for d in DELTAS}
            h = trajs.time_check(t, T, tr, Tr, applied if key == "glim_corr" else 0.0)
            r["timing"] = {k: h.get(k) for k in ("dt_ms", "r_timing", "on_time")}
        r["loop"] = loop_closure(P, a.config, key, f, not a.no_loop)
        report["trajectories"].append(r)

    # maps, from the reference poses
    pos = Tr[:, :3, 3]
    tsdf = P.outp(P.pcd("denoised"))
    union = P.outp("denoised_union_%s.pcd" % P.tag if P.tag else "denoised_union.pcd")
    fusion = P.stage("01_build_map").get("fusion", "union")
    maps = [("tsdf", "TSDF (reference map)", tsdf), ("union", "union of points", union)] \
        if fusion == "tsdf" else [("union", "union of points (01's map)", tsdf)]
    clouds = {}
    for key, label, f in maps:
        if not os.path.isfile(f):
            print(f"map {label}: none at {f}" + (" (a comparison: \"comparisons\": true)"
                                                 if key == "union" and fusion == "tsdf" else ""))
            if not (key == "union" and fusion == "tsdf"):
                report["maps"][key] = dict(label=label, file=f, missing=True)
            continue
        print(f"map {label}: {f} ...")
        m, clouds[key] = map_stats(f, pos, a.samples)
        m["label"] = label
        report["maps"][key] = m
    if "tsdf" in clouds and "union" in clouds:
        comp, acc, *_ = maps.coverage(clouds["union"], clouds["tsdf"], 0.10)
        report["maps"]["tsdf"]["completeness_vs_union_pct"] = comp * 100
        report["maps"]["tsdf"]["accuracy_vs_union_pct"] = acc * 100

    # the tables
    md = [f"# Mapping pass {P.tag}", "",
          f"Reference trajectory: `{ref_file}`; GLIM's pose-time correction "
          f"{applied * 1000:+.0f} ms.", "",
          "| Trajectory | ATE cm / deg | RPE 1 s | RPE 10 s | RPE 100 s | pose time dt ms (r) "
          "| board loop mm / deg (noise) | path m |",
          "|---|---|---|---|---|---|---|---|"]
    for r in report["trajectories"]:
        if r.get("missing"):
            md.append(f"| {r['label']} | not there: `{r['file']}` | | | | | | |")
            continue
        lp = r.get("loop")
        loop = "-" if not lp else (f"{lp['mm']:.1f} / {lp['deg']:.2f} ({lp['noise_mm']:.1f})"
                                   + (" DRIFT" if lp["significant"] else ""))
        if "ate_cm" in r:
            rpe = [f"{x:.1f} / {y:.2f}" for x, y in r["rpe"].values()]
            tm = r["timing"]
            md.append(f"| {r['label']} | {r['ate_cm']:.1f} / {r['ate_deg']:.2f} | "
                      + " | ".join(rpe)
                      + f" | {tm['dt_ms']:+.1f} ({tm['r_timing']:.2f}) | {loop} | {r['path_m']:.1f} |")
        else:
            md.append(f"| **{r['label']}** | reference | | | | | {loop} | {r['path_m']:.1f} |")
    md += ["", "| Map (reference poses) | points | walls cm | floors cm | all cm | doubled % "
               "| completeness vs union |", "|---|---|---|---|---|---|---|"]
    for key, m in report["maps"].items():
        if m.get("missing"):
            md.append(f"| {m['label']} | not there: `{m['file']}` | | | | | |")
            continue
        cpl = m.get("completeness_vs_union_pct")
        md.append(f"| {m['label']} | {m['points'] / 1e6:.1f} M | {fmt(m.get('walls_cm'))} | "
                  f"{fmt(m.get('floors_cm'))} | {fmt(m.get('median_cm'))} | "
                  f"{fmt(m.get('doubled_pct'))} | {'' if cpl is None else f'{cpl:.1f} %'} |")
    md += ["", "ATE after the best rigid fit, RPE over 1 / 10 / 100 s (cm / deg), RMSE. "
               "Pose time: the heading error against the reference regressed on the turning; "
               "|dt| >= 10 ms with r >= 0.5 is a pose-time offset. Thickness: p95-p5 about "
               "planes in 0.25 m patches, medians."]
    text = "\n".join(md)
    print("\n" + text)
    base = os.path.join(P.out_dir, "mapping_report_%s" % P.tag if P.tag else "mapping_report")
    with open(base + ".json", "w") as fh:
        json.dump(report, fh, indent=1)
    with open(base + ".md", "w") as fh:
        fh.write(text + "\n")
    print(f"\nreport: {base}.md, {base}.json")


COMMANDS = {"map": maps.main_quality, "walls": maps.main_walls, "maps": maps.main_compare,
            "scans": maps.main_scans, "trajs": trajs.main_compare, "time": trajs.main_time,
            "imu": trajs.main_imu, "loop": trajs.main_loop, "report": report}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    cmd = sys.argv[1]
    sys.argv = ["check.py " + cmd] + sys.argv[2:]          # each command's usage line
    COMMANDS[cmd](sys.argv[1:])


if __name__ == "__main__":
    main()

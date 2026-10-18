#!/usr/bin/env python3
"""Keep a pass's current results, move everything else aside.

    python3 clean_pass.py <pipeline config> [--apply] [--work]

Looks at the pass's mapping/ and odometry/<machine>/ folders and sorts every
file and folder in them into

  KEEP   what the pipeline makes now and reads again: GLIM's export
         (traj_lidar.txt, traj_imu.txt, glim.yaml, glim_config/, its loop
         closures), 02's reference trajectory and its reports, 02b's
         comparison, dataset.traj, the pose-time check, 01's map (merged
         ... map_final, 06's cut), the union map, the seed map, and the
         names listed under "clean": {"keep": [...]} in the config (results
         a paper cites)
  STALE  one of those, but older than what it is made from (02b and the
         seed map older than GLIM's export; the map and the union map older
         than dataset.traj): run_pass.py makes it again
  OLD    everything else: earlier versions, side experiments, caches

STALE and OLD are moved to <processed>/<date>/<pass>/_old/, under the same
folders; nothing is deleted. Without --apply it only prints the lists (a dry
run); look at them, then run again with --apply, and delete _old/ yourself
once the pass runs again without it. --work also moves GLIM's side runs in
work/ (<machine>/glim_*/ other than glim/) to work/<date>/<pass>/_old/.
"""
import argparse
import fnmatch
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "map_stages"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))

from pipeline_common import load_pipeline                                # noqa: E402

GLIM_EXPORT = ("traj_lidar.txt", "traj_imu.txt", "glim.yaml", "glim_config",
               "glim_loop_closure.json", "glim_raw_loop_closure.json")
MAP_STAGES = ("merged", "static", "denoised", "colored", "flattened", "anchored", "map_final")


def size(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024


def with_reports(name):
    """A trajectory and the files stages write beside it."""
    b = os.path.splitext(name)[0]
    return [name, b + "_report.json", b + "_loop_closure.json", b + "_rounds.json", b + "_r*.txt"]


def plan(P):
    """{folder: [(name, verdict, why)]} for mapping/ and odometry/<machine>/."""
    odo = os.path.dirname(P.outp("traj_lidar.txt"))
    mp = P.folder_for("x.pcd")
    tag = P.tag
    keep_extra = list((P.cfg.get("clean") or {}).get("keep") or [])
    seed = os.path.join(odo, "traj_lidar.txt")
    traj = P.outp(P.dataset["traj"]) if P.dataset.get("traj") else None
    out02 = (P.cfg.get("02_bundle_adjust") or {}).get("output") or "traj_lidar_ba.txt"
    out02b = (P.cfg.get("02b_scan_to_map") or {}).get("output") or "traj_lidar_refined.txt"
    pcd = lambda base, extra="": f"{base}{extra}_{tag}.pcd" if tag else f"{base}{extra}.pcd"  # noqa: E731
    final01 = (P.cfg.get("01_build_map") or {}).get("output")
    final01 = os.path.basename(P.outp(final01)) if final01 else pcd("map_final")

    def mtime(p):
        return os.path.getmtime(p) if p and os.path.exists(p) else None

    # (patterns, made from: older than this = stale)
    odo_rules = [(list(GLIM_EXPORT), None, "GLIM's export"),
                 (["pose_time_check.json"], None, "the pose-time check"),
                 (with_reports(out02), None, "02's reference"),
                 (with_reports(out02b), seed, "02b's comparison")]
    if traj and os.path.dirname(traj) == odo:
        odo_rules.append((with_reports(os.path.basename(traj)), None, "dataset.traj"))
    map_rules = [([pcd(b) for b in MAP_STAGES] + [final01, "map_final_*"], traj, "01's map"),
                 ([pcd(b, "_union") for b in MAP_STAGES], traj, "the union map"),
                 ([pcd(b, "_seed") for b in MAP_STAGES], seed, "02b's seed map"),
                 (["cloud_in_ref_*"], None, "09's cloud")]
    out = {}
    for folder, rules in ((odo, odo_rules), (mp, map_rules)):
        rows = []
        if not os.path.isdir(folder):
            out[folder] = rows
            continue
        for name in sorted(os.listdir(folder)):
            verdict, why = "OLD", "not made by the pipeline now"
            for pats, src, what in rules:
                if any(fnmatch.fnmatch(name, p) for p in pats):
                    t, ts = mtime(os.path.join(folder, name)), mtime(src)
                    if ts is not None and t is not None and t < ts:
                        verdict, why = "STALE", f"{what}, older than {os.path.basename(src)}"
                    else:
                        verdict, why = "KEEP", what
                    break
            if verdict != "KEEP" and any(fnmatch.fnmatch(name, p) for p in keep_extra):
                verdict, why = "KEEP", "clean.keep in the config"
            rows.append((name, verdict, why))
        out[folder] = rows
    return out


def work_side_runs(P):
    """GLIM's side runs in work/: <machine>/glim_*/ other than glim/."""
    mdir = os.path.dirname(os.path.dirname(P.glim_traj()))
    if not os.path.isdir(mdir):
        return mdir, []
    return mdir, [n for n in sorted(os.listdir(mdir))
                  if n.startswith("glim_") and os.path.isdir(os.path.join(mdir, n))]


def move(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        dst += time.strftime("_%Y%m%d_%H%M%S")
    shutil.move(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--apply", action="store_true", help="move STALE and OLD to _old/")
    ap.add_argument("--work", action="store_true", help="also GLIM's side runs in work/")
    a = ap.parse_args()
    P = load_pipeline(a.config)
    old_root = os.path.join(P.out_dir, "_old")
    moves, freed = [], 0
    for folder, rows in plan(P).items():
        rel = os.path.relpath(folder, P.out_dir)
        print(f"\n{folder}")
        if not rows:
            print("  (nothing there)")
        for name, verdict, why in rows:
            p = os.path.join(folder, name)
            n = size(p)
            print(f"  {verdict:5s} {human(n):>9s}  {name}{'/' if os.path.isdir(p) else ''}  -- {why}")
            if verdict != "KEEP":
                moves.append((p, os.path.join(old_root, rel, name)))
                freed += n
    if a.work:
        mdir, runs = work_side_runs(P)
        wold = os.path.join(os.path.dirname(mdir), "_old")
        print(f"\n{mdir}")
        for n in runs:
            p = os.path.join(mdir, n)
            s = size(p)
            print(f"  OLD   {human(s):>9s}  {n}/  -- a GLIM side run (glim/ is the one in use)")
            moves.append((p, os.path.join(wold, os.path.basename(mdir), n)))
            freed += s
    print(f"\n{len(moves)} to move to _old/ ({human(freed)})")
    if not moves:
        return
    if not a.apply:
        print("dry run: nothing moved; run again with --apply")
        return
    for src, dst in moves:
        move(src, dst)
    print(f"moved; {old_root} holds them (and work's _old/ with --work). Delete it once the "
          "pass runs again without it (run_pass.py --status shows what it would redo).")


if __name__ == "__main__":
    main()

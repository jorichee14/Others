#!/usr/bin/env python3
"""One pass from raw to processed, every step in order.

    python scoop_pipeline/run_pass.py <raw pass folder> [--status] [--dry]
        [--from STEP] [--until STEP] [--only STEP] [--config FILE]

<raw pass folder> is data/raw/<date>/<pass>, e.g.
    ~/workspaces/isaac_ros-dev/data/raw/20260924/survey_1

Steps (a step whose output is there already is skipped):

  raw -> work
    process       processing/process_recording.py on every machine folder with an
                  Ouster packets bag or an SVO2: decoded, retimed, GLIM, ZED bags
    merge         processing/merge_session.py: check against record.yaml, merge
  work -> processed (map_stages/, the pipeline config of this pass)
    mapping pass (no dataset.reference_pass):
    01_seed       01_build_map.py --seed     map from the GLIM poses (02's reference)
    02_refine     02_refine_poses.py         every scan registered to that map
    01_map        01_build_map.py            the final map from the refined poses
    a run in a mapping pass's map (dataset.reference_pass):
    03_init       03_init_from_boards.py     start pose from the anchor board
    04_reference  04_reference_traj.py       every scan registered to the anchored map
    both:
    05_anchor     05_anchor.py               boards; a mapping pass: the anchored map
    06_cut        06_cut.py                  viewing copy, only when 06_cut has floor/ceil
    07_cameras    07_build_cameras.py        infra / parked cameras in map
    08_tfs        08_emit_tfs.py             frames/publish_tfs_<tag>.sh
    09_cloud      09_build_coop_cloud.py     a run's cloud in the map (runs only)
    10_poses      10_publish_poses.py        bags/<tag>_best_poses: poses + complete TF
    finalize      the merged bag again, with 10's /tf, /tf_static and poses in it
                  (built next to it, then put in its place)
    comms         processing/comms_tables.py: NTP, Wi-Fi, ping, iperf, CSI tables with
                  each machine's position (processed/<date>/<pass>/comms/)

--status  what is done, and stop      --dry     print the commands, run nothing
--from S  run S and every step after it, done or not
--until S stop after S                --only S  run S alone, done or not
--config  the pipeline config (default: the one in map_stages/ whose dataset is
          this pass, e.g. pipeline_config_survey_1.json; a run needs its own,
          "extends": "pipeline_config.json" with dataset.reference_pass)

Datasets for other tools (datasets/mcap_convert.py) are made afterwards from the
merged bag and the processed outputs.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
STAGES = os.path.join(ROOT, "map_stages")
sys.path.insert(0, ROOT)
sys.path.insert(0, STAGES)

MAPPING = ["01_seed", "02_refine", "01_map"]
RUN = ["03_init", "04_reference"]
ORDER = (["process", "merge"] + MAPPING + RUN
         + ["05_anchor", "06_cut", "07_cameras", "08_tfs", "09_cloud", "10_poses", "finalize",
            "comms"])


def py(script, *args):
    return [sys.executable, script, *args]


def machines(raw):
    """Machine folders with something process_recording.py works on."""
    from scoop import recording
    from scoop.bag import BagError
    out = []
    if not os.path.isdir(raw):
        return out
    for d in sorted(p for p in os.listdir(raw) if os.path.isdir(os.path.join(raw, p))):
        try:
            recording.find_recording(os.path.join(raw, d))
        except (BagError, OSError, ValueError):
            continue
        out.append(os.path.join(raw, d))
    return out


def merged_bag(raw):
    """Where merge_session.py writes this pass's merged bag; None without bags."""
    from scoop import recording, session
    from scoop.bag import BagError
    try:
        return _merged_bag(raw, recording, session)
    except (BagError, OSError, ValueError):
        return None


def _merged_bag(raw, recording, session):
    plan = session.RecordPlan(None)
    settings = recording.load_settings(None)
    rawp = recording.Path(raw).expanduser().resolve()
    mode = plan.mode_of(rawp.name, *(p.parent.name for p in rawp.rglob("metadata.yaml")))
    units = session.find_units(rawp, plan, settings, None, mode)
    return os.path.join(str(recording.work_dir_for(raw, None)),
                        "%s_merged" % session.session_name(units, plan, rawp))


def has_map_tf(bag):
    from scoop import tftree
    edges = tftree.tf_edges([bag], seconds=1.0)
    return any(e.parent == "map" for e in edges.values())


def plan_steps(raw, cfg):
    """[(step, applies, done, what, command or callable)]"""
    import pass_tf
    steps = []
    mb = merged_bag(raw)
    # once merged, the processed bags may be gone: processing is not redone for them
    have = mb is not None and os.path.isdir(mb)
    steps.append(("process", mb is not None, have, "%d machine folder(s)" % len(machines(raw))
                  if os.path.isdir(raw) else "no %s" % raw,
                  [py(os.path.join(ROOT, "processing", "process_recording.py"), m)
                   for m in machines(raw)]))
    steps.append(("merge", mb is not None, have, mb or "no bags below %s" % raw,
                  [py(os.path.join(ROOT, "processing", "merge_session.py"), raw)]))
    if not cfg:
        if mb:
            steps.append(comms_step(mb, None))
        return steps, None
    P = pass_tf.load_pipeline(cfg)
    run = bool(P.reference)
    st = lambda name: os.path.join(STAGES, name)                     # noqa: E731
    tag = P.tag
    blk = lambda k: P.cfg.get(k) or {}                               # noqa: E731
    stg = lambda k, f, d: P.stage(k).get(f) or P.outp(d) if k in P.cfg else P.outp(d)  # noqa: E731
    out = {
        "01_seed": P.outp("denoised_seed_%s.pcd" % tag),
        "02_refine": P.outp(blk("02_refine").get("output", "traj_lidar_refined.txt")),
        "01_map": P.outp(blk("01_build_map").get("output") or P.pcd("map_final")),
        "03_init": stg("03_init", "output", "session_anchor.json"),
        "04_reference": P.lidar_track_traj(),
        "05_anchor": P.outp("boards_{tag}.json") if run
        else stg("05_anchor", "frame_out", "anchor_frame.json"),
        "06_cut": stg("06_cut", "output", "map_final_{tag}_anchored_noceil.pcd"),
        "07_cameras": stg("07_build_cameras", "output", "cameras_in_map.yaml"),
        "08_tfs": P.outp("publish_tfs_{tag}.sh"),
        "09_cloud": P.outp("cloud_in_ref_{tag}.pcd"),
        "10_poses": P.outp(blk("10_publish").get("output_bag") or "{tag}_best_poses"),
    }
    cut = blk("06_cut")
    applies = {s: not run for s in MAPPING}
    applies.update({s: run for s in RUN})
    applies.update({"05_anchor": True, "06_cut": not run and (cut.get("floor") is not None
                                                             or cut.get("ceil") is not None),
                    "07_cameras": True, "08_tfs": True, "09_cloud": run, "10_poses": True})
    cmds = {
        "01_seed": [py(st("01_build_map.py"), cfg, "--seed")],
        "02_refine": [py(st("02_refine_poses.py"), cfg)],
        "01_map": [py(st("01_build_map.py"), cfg)],
        "03_init": [py(st("03_init_from_boards.py"), cfg)],
        "04_reference": [py(st("04_reference_traj.py"), cfg)],
        "05_anchor": [py(st("05_anchor.py"), cfg)],
        "06_cut": [py(st("06_cut.py"), cfg)],
        "07_cameras": [py(st("07_build_cameras.py"), cfg)],
        "08_tfs": [py(st("08_emit_tfs.py"), cfg) + [">", out["08_tfs"]]],
        "09_cloud": [py(st("09_build_coop_cloud.py"), cfg)],
        "10_poses": [py(st("10_publish_poses.py"), cfg)],
    }
    done = {k: bool(v) and os.path.exists(v) for k, v in out.items()}
    done["01_seed"] = done["01_seed"] or done["02_refine"]       # only 02 reads it
    for s in MAPPING + RUN + ["05_anchor", "06_cut", "07_cameras", "08_tfs", "09_cloud",
                              "10_poses"]:
        steps.append((s, applies[s], done[s], out[s], cmds[s]))
    db = P.dataset["bag"]
    if mb and os.path.normpath(db) != os.path.normpath(mb):
        print("note: dataset.bag of %s is %s; the merge writes %s" % (cfg, db, mb))
    steps.append(("finalize", True, lambda: finalized(db, out["10_poses"]), db,
                  lambda dry: finalize(db, out["10_poses"], dry)))
    steps.append(comms_step(db, cfg))
    return steps, P


def comms_step(bag, cfg):
    """NTP, Wi-Fi, ping, iperf, CSI tables (processed/<date>/<pass>/comms/)."""
    from pipeline_common import processed_dir_for
    summ = os.path.join(processed_dir_for(bag), "comms", "summary.json")
    cmd = py(os.path.join(ROOT, "processing", "comms_tables.py"), bag)
    return ("comms", True, os.path.exists(summ), os.path.dirname(summ),
            [cmd + (["--config", cfg] if cfg else [])])


def finalized(merged, poses):
    """The merged bag is newer than 10's bag and has the tree from map."""
    meta = lambda b: os.path.join(b, "metadata.yaml")                # noqa: E731
    return (os.path.exists(meta(merged)) and os.path.exists(meta(poses))
            and os.path.getmtime(meta(merged)) > os.path.getmtime(meta(poses))
            and has_map_tf(merged))


def finalize(merged, poses, dry):
    """merged + 10's bag (its /tf, /tf_static and pose topics) -> merged."""
    from scoop import bag, merge
    if not os.path.isdir(poses):
        if dry:
            print("  %s + %s (once 10_poses wrote it) -> in its place" % (merged, poses))
            return
        raise SystemExit("finalize: no %s (run 10_poses)" % poses)
    tp = sorted(bag.open_bag(poses).topics())
    new = merged.rstrip("/") + "_tf"
    print("  %s + %s (%s) -> %s, then in its place" % (merged, poses, ", ".join(tp), new))
    if dry:
        return
    if os.path.exists(new):
        raise SystemExit("%s exists: remove it (a finalize that stopped half way)" % new)
    size = merge.bag_bytes(merged)
    free = shutil.disk_usage(os.path.dirname(merged)).free
    if free < size + merge.MIN_FREE_GB * 1e9:
        raise SystemExit("finalize needs %.0f GB free next to the merged bag, %.0f GB are"
                         % (size / 1e9 + merge.MIN_FREE_GB, free / 1e9))
    merge.merge_bags([merged, poses], new, source={t: 1 for t in tp})
    shutil.rmtree(merged)
    os.rename(new, merged)
    print("  %s now has the complete TF and the poses" % merged)


def run_step(name, cmds, dry):
    if callable(cmds):
        return cmds(dry)
    for c in cmds:
        redirect = None
        if ">" in c:
            i = c.index(">")
            c, redirect = c[:i], c[i + 1]
        print("  $ " + " ".join(c) + (" > " + redirect if redirect else ""))
        if dry:
            continue
        if redirect:
            with open(redirect, "w") as fh:
                r = subprocess.run(c, stdout=fh)
        else:
            r = subprocess.run(c)
        if r.returncode:
            raise SystemExit("%s failed (exit %d); fix it, then run again (done steps are "
                             "skipped)" % (name, r.returncode))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--from", dest="start", choices=ORDER)
    ap.add_argument("--until", choices=ORDER)
    ap.add_argument("--only", choices=ORDER)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    import pass_tf
    raw = os.path.abspath(os.path.expanduser(a.raw_dir))
    cfg = a.config or pass_tf.find_config(raw)
    steps, P = plan_steps(raw, cfg)
    print("pass: %s\nconfig: %s" % (raw, cfg or "none in map_stages/ for this pass"))
    if P is not None:
        print("kind: %s" % ("run in %s's map" % P.reference if P.reference else "mapping pass"))

    if a.status:
        for name, applies, done, what, _ in steps:
            done = done() if callable(done) else done
            state = "-" if not applies else {True: "done", False: "todo", None: "runs"}[done]
            print("  %-13s %-5s %s" % (name, state, what))
        if not cfg:
            print("no pipeline config for this pass: the map stages need one (--config)")
        return

    names = [s[0] for s in steps]
    lo = names.index(a.only or a.start) if (a.only or a.start) else 0
    hi = names.index(a.only or a.until) if (a.only or a.until) else len(names) - 1
    forced = bool(a.only or a.start)
    for name, applies, done, what, cmds in steps[lo:hi + 1]:
        if not applies:
            continue
        if callable(done):
            done = done()
        if done and not forced:
            print("[skip] %s: %s" % (name, what))
            continue
        print("\n== %s ==  %s" % (name, what))
        run_step(name, cmds, a.dry)
    if P is None:
        print("\nno pipeline config for this pass: make map_stages/pipeline_config_%s.json "
              "(a run: {\"extends\": \"pipeline_config.json\", \"dataset\": {\"bag\": "
              "<the merged bag>, \"reference_pass\": <the mapping pass>}}) and run again"
              % os.path.basename(raw))


if __name__ == "__main__":
    main()

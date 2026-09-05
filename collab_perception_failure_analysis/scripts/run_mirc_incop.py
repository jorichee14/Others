#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep InCoP's pretrained models over the converted MIRC recording.

Every method, every seed, both fusion modes, one table. The pairing is the whole
point: `intermediate` is what the ego sees with its partner, `no` is what it sees
alone, and on this recording the ego is the 87-degree depth cart that finds the
chairs in about a third of frames by itself. The gap between the two columns is
what collaboration is worth on real hardware, which is the one number the
simulated arms of this study cannot supply.

    python3 scripts/run_mirc_incop.py \
        --incop-root ~/workspaces/isaac_robotics_data/InCoP \
        --dataset ~/workspaces/isaac_robotics_data/InCoP/mirc_coop2_real_world/validate \
        --out results/mirc_incop

`--video` additionally renders the ego-only vs fused comparison for the FIRST
seed of each method, over the whole split. Those passes are separate AND run
after every measured run, because --video_compare_fusion runs the model twice
per frame, rasterises four BEV panels per frame in Python, and produces no
metrics at all. One of them can take longer than the entire numeric sweep.

Nothing here is InCoP-specific beyond the CLI it shells out to; the results land
in the same shape as run_phase1.py's so the OPV2V and MIRC tables can sit side
by side.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time

import yaml

FUSION_MODES = ("intermediate", "no")     # the tool's own flag values
# "no" is NOT an ego-only run -- see ego_only_model_dir(). It stays
# selectable for comparison against the old numbers, but the default is
# the one mode that means something on its own.
DEFAULT_FUSION = ("intermediate",)
EGO_ONLY = "ego_only"                    # our floor, via comm_range 0
DEFAULT_METHODS = ("ours", "where2comm", "cobevt")


def checkpoints(incop_root: str, method: str, scene: str):
    """Every seed of one method, oldest first, so run order is reproducible."""
    pattern = os.path.join(incop_root, "opencood", "logs",
                           f"isaacsim_{method}_pretrained_{scene}_20*")
    # `_with_noise_` variants are a different model, not another seed of this one.
    return sorted(path for path in glob.glob(pattern)
                  if os.path.isdir(path) and "_with_noise_" not in os.path.basename(path))


def ego_only_model_dir(model_dir: str, out: str) -> str:
    """A shadow of `model_dir` whose config puts the partner out of range.

    THE ONLY HONEST EGO-ONLY FLOOR AVAILABLE HERE. `--fusion_method no` does not
    drop the partner: both modes reported `Communication: 158.951 KB/sample` to
    the milligram and recall equal to three decimals, so the "ego-only" column
    was the fused column computed twice and every lift came out 0.000.

    What does work is the mechanism the tool's own video path uses -- it prints
    `comm_range 50 -> 0 m` and then `record_len=[1]`, one agent in the sample.
    The dataset keeps a CAV when its distance to the ego is within comm_range;
    at 0 the ego survives (distance 0) and every partner is dropped.

    The weights are symlinked and only `comm_range` is rewritten, by line
    substitution rather than a yaml round-trip: these configs carry python tags
    that no safe dumper reproduces, and re-emitting one would quietly change
    fields nobody asked about.
    """
    name = os.path.basename(model_dir.rstrip("/"))
    # ABSOLUTE: inference runs with cwd set to the InCoP root, so a relative
    # --model_dir would resolve over there and the run dies on a missing config.
    shadow = os.path.abspath(os.path.join(out, "ego_only", name))
    os.makedirs(shadow, exist_ok=True)
    for path in glob.glob(os.path.join(model_dir, "net_epoch*.pth")):
        link = os.path.join(shadow, os.path.basename(path))
        if not os.path.exists(link):
            os.symlink(path, link)
    text = open(os.path.join(model_dir, "config.yaml")).read()
    text, count = re.subn(r"(?m)^comm_range:.*$", "comm_range: 0", text)
    if count != 1:
        raise SystemExit("expected one comm_range line in %s/config.yaml, found %d"
                         % (name, count))
    with open(os.path.join(shadow, "config.yaml"), "w") as handle:
        handle.write(text)
    return shadow


def newest_eval(model_dir: str, after: float):
    """The unified eval yaml this run just wrote, by mtime.

    `eval_all_isaac_*` specifically, not `eval_*`: the tool writes TWO summaries
    per run and the other one is class-agnostic, reporting 0.000 for a scene
    whose objects are all one class. Matched on time rather than by rebuilding
    the name, which encodes score threshold, NMS, epoch and dataset tag and would
    be a second implementation of the tool's own naming, free to drift from it.
    """
    best, best_time = None, after
    for path in glob.glob(os.path.join(model_dir, "eval_all_isaac_*.yaml")):
        stamp = os.path.getmtime(path)
        if stamp >= best_time:
            best, best_time = path, stamp
    return best


def x_to_world(pose) -> "np.ndarray":
    """OpenCOOD's ``transformation_utils.x_to_world``, ``[x,y,z,roll,yaw,pitch]``
    in degrees to a 4x4. Replicated rather than imported: this script shells out
    to InCoP and is never run inside it, so it must not need OpenCOOD on the path.
    """
    import numpy as np
    x, y, z, roll, yaw, pitch = [float(v) for v in pose[:6]]
    c_y, s_y = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    c_r, s_r = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    c_p, s_p = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    m = np.identity(4)
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    m[0, 0] = c_p * c_y
    m[0, 1] = c_y * s_p * s_r - s_y * c_r
    m[0, 2] = -c_y * s_p * c_r - s_y * s_r
    m[1, 0] = s_y * c_p
    m[1, 1] = s_y * s_p * s_r + c_y * c_r
    m[1, 2] = -s_y * s_p * c_r + c_y * s_r
    m[2, 0] = s_p
    m[2, 1] = -c_p * s_r
    m[2, 2] = c_p * c_r
    return m


def ego_folder(scenario_dir: str):
    """The agent OpenCOOD treats as ego: folders sorted, a LEADING negative id
    moved to the end, then the first one taken. Verified against `basedataset.py`
    on main; the negative-id rule exists because OPV2V numbers its RSU -1.
    """
    names = sorted(name for name in os.listdir(scenario_dir)
                   if os.path.isdir(os.path.join(scenario_dir, name)))
    if not names:
        return None
    if names[0].startswith("-"):
        names = names[1:] + names[:1]
    return names[0] if names else None


def labelled_gt_count(dataset_dir: str, class_name: str, gt_range,
                      limit: int = 0) -> dict:
    """How many labels of one class actually lie in the evaluation range.

    THE EVALUATOR'S OWN `gt_count` IS NOT THIS NUMBER. It comes from a
    visibility-subset block that only counts objects some prediction landed on,
    so it rises and falls with how many boxes the detector fires: on one MIRC
    split it read 476, 620, 728 and 757 for the same two chairs, and even
    disagreed between two fusion modes of a single checkpoint (476 vs 485). A
    denominator that moves with the model is not a denominator -- recall
    computed against it is `matched / what-the-model-covered`, which flatters
    every model and cannot be compared across models at all.

    So count the labels instead. Each frame's yaml holds `vehicles` in WORLD
    coordinates plus the ego's `lidar_pose`; OpenCOOD scores in the ego lidar
    frame and gates on `gt_range`, so do both here. Only the ego's yaml is read,
    because that is the frame the evaluation happens in.

    Returns a dict rather than an int so the caller can report the frame count
    and the per-scenario split without walking the tree twice.
    """
    import numpy as np
    lo = np.asarray([float(v) for v in gt_range[:3]], dtype=np.float64)
    hi = np.asarray([float(v) for v in gt_range[3:6]], dtype=np.float64)
    total, every, frames, scenarios = 0, 0, 0, {}
    for scenario in sorted(os.listdir(dataset_dir)):
        scenario_dir = os.path.join(dataset_dir, scenario)
        if not os.path.isdir(scenario_dir):
            continue
        ego = ego_folder(scenario_dir)
        if ego is None:
            continue
        here = 0
        frame_paths = sorted(glob.glob(os.path.join(scenario_dir, ego, "*.yaml")))
        # `--max-samples` caps how many frames each run sees, so the denominator
        # has to be capped the same way or recall is measured against labels the
        # run was never shown.
        if limit:
            frame_paths = frame_paths[:limit]
        for path in frame_paths:
            with open(path) as handle:
                frame = yaml.safe_load(handle) or {}
            pose = frame.get("lidar_pose")
            if pose is None:
                continue
            frames += 1
            world_to_ego = np.linalg.inv(x_to_world(pose))
            for entry in (frame.get("vehicles") or {}).values():
                if not isinstance(entry, dict):
                    continue
                kind = entry.get("obj_type") or entry.get("class_name")
                if kind is not None and str(kind) != class_name:
                    continue
                location = entry.get("location")
                if location is None or len(location) < 3:
                    continue
                every += 1
                point = np.append(np.asarray(location[:3], dtype=np.float64), 1.0)
                local = (world_to_ego @ point)[:3]
                # Centre-in-range, the same gate OpenCOOD applies before scoring.
                if bool(np.all(local >= lo) and np.all(local <= hi)):
                    here += 1
        scenarios[scenario] = here
        total += here
    # `gt_all` is every label of the class regardless of range. The gap between
    # it and gt_labelled is not noise: these checkpoints carry the hospital
    # scene's forward-only box, so an object the robot drove PAST is unscorable
    # even though both sensors saw it. Reported so that limit is visible rather
    # than quietly shrinking the denominator.
    return {"gt_labelled": total, "gt_all": every, "frames": frames,
            "per_scenario": scenarios}


class TolerantLoader(yaml.SafeLoader):
    """SafeLoader that skips tags it does not know instead of refusing the file.

    InCoP serialises `noise_setting` as
    `!!python/object/apply:collections.OrderedDict`, and SafeLoader rejects the
    WHOLE document over that one field -- including `postprocess.gt_range`,
    which is what we came for. Unknown tags become None here: nothing is
    constructed and no code runs, so this stays as safe as safe_load, it just
    does not throw away a config because of a field it was not asked about.
    """


TolerantLoader.add_multi_constructor("", lambda loader, suffix, node: None)


def read_hypes(path: str):
    """A checkpoint's config, or None if it genuinely cannot be read."""
    try:
        with open(path) as handle:
            return yaml.load(handle, Loader=TolerantLoader) or {}
    except Exception as error:
        print("  ! could not read %s: %s" % (path, error))
        return None


def eval_range(model_dir: str):
    """The range this checkpoint scores in, from its own config.

    `postprocess.gt_range` is what the evaluator gates ground truth on; it falls
    back to `cav_lidar_range`, which is the same list in every config seen so far.
    Returns None when the config cannot be read, so the caller degrades to
    reporting no corrected denominator rather than inventing one.
    """
    path = os.path.join(model_dir, "config.yaml")
    if not os.path.exists(path):
        return None
    hypes = read_hypes(path)
    if hypes is None:
        return None
    found = ((hypes.get("postprocess") or {}).get("gt_range")
             or hypes.get("cav_lidar_range"))
    if isinstance(found, (list, tuple)) and len(found) >= 6:
        return [float(v) for v in found[:6]]
    return None


def class_metrics(found: dict, class_name: str) -> dict:
    """Everything the unified summary says about one class.

    The headline mAP lives at the top level; recall, centre error (ATE) and size
    error (ASE) live inside per-subset `per_class` blocks, of which only the ones
    with ground truth carry numbers. Walk for them rather than hard-coding the
    path, so a change in how the subsets are nested does not silently return
    nulls that read as a model failing.
    """
    out = {}
    overall = (found.get("per_class_mAP") or {}).get(class_name) or {}
    for key, field in (("mAP@0.3", "ap30"), ("mAP@0.5", "ap50"), ("mAP@0.7", "ap70")):
        if overall.get(key) is not None:
            out[field] = float(overall[key])

    best = None
    stack = [found]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        block = node.get("per_class")
        if isinstance(block, dict):
            entry = block.get(class_name)
            # Several subsets carry the class; the one with ground truth is the
            # only one whose numbers exist.
            if isinstance(entry, dict) and (entry.get("gt_count") or 0) > 0:
                if best is None or entry["gt_count"] > best["gt_count"]:
                    best = entry
        stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
        for value in node.values():
            if isinstance(value, list):
                stack.extend(v for v in value if isinstance(v, dict))
    if best:
        # "gt_covered", not "gt": this block counts only the objects a
        # prediction landed on, so it moves with the detector. The honest
        # denominator comes from labelled_gt_count().
        for key, field in (("gt_count", "gt_covered"), ("matched_tp_0.5m", "matched_0.5m"),
                           ("recall@0.3", "recall30"), ("recall@0.5", "recall50"),
                           ("ATE_mean", "ate"), ("ASE_mean", "ase")):
            if best.get(key) is not None:
                out[field] = float(best[key])
    return out


def run_one(args, model_dir: str, fusion: str, video: bool, labelled=None):
    command = [
        sys.executable, "opencood/tools/inference_isaac.py",
        "--model_dir", model_dir,
        "--checkpoint_mode", args.checkpoint_mode,
        "--fusion_method", "intermediate" if fusion == EGO_ONLY else fusion,
        "--eval_split", "val",
        "--eval_dataset_dir", args.dataset,
    ]
    if video:
        command += ["--video_compare_fusion", "--video_fps", str(args.video_fps),
                    "--stream_video_output",
                    os.path.join(model_dir, "video_full_validate.mp4")]
    if args.max_samples:
        command += ["--max_samples", str(args.max_samples)]

    label = "%s [%s]%s" % (os.path.basename(model_dir), fusion,
                           " +video" if video else "")
    print("\n=== %s" % label, flush=True)
    if args.dry_run:
        print("  " + " ".join(command))
        return None

    started = time.time()
    result = subprocess.run(command, cwd=args.incop_root)
    elapsed = time.time() - started
    if result.returncode != 0:
        print("  FAILED (exit %d) after %.0fs" % (result.returncode, elapsed))
        return {"model_dir": model_dir, "fusion": fusion, "failed": True,
                "returncode": result.returncode, "runtime_s": round(elapsed, 1)}
    if video:
        return {"model_dir": model_dir, "fusion": fusion, "video": True,
                "runtime_s": round(elapsed, 1)}

    path = newest_eval(model_dir, started)
    if path is None:
        print("  ! no eval_*.yaml written — the run produced no metrics")
        return {"model_dir": model_dir, "fusion": fusion, "failed": True,
                "reason": "no eval yaml", "runtime_s": round(elapsed, 1)}
    with open(path) as handle:
        found = yaml.safe_load(handle) or {}
    metrics = class_metrics(found, args.eval_class)
    if "ap30" not in metrics:
        print("  ! %s has no %s numbers — the class may have no ground truth here"
              % (os.path.basename(path), args.eval_class))
    record = {"model_dir": model_dir, "fusion": fusion, "eval_yaml": path,
              "eval_class": args.eval_class, "runtime_s": round(elapsed, 1)}
    record.update(metrics)
    record.update(corrected_recall(metrics, labelled))
    print("  %s AP@0.3/0.5/0.7 = %.3f / %.3f / %.3f | recall@0.3 %s | "
          "ATE %.3f m ASE %.3f | gt %s (%.0fs)"
          % (args.eval_class, record.get("ap30", 0.0), record.get("ap50", 0.0),
             record.get("ap70", 0.0), _recall_text(record),
             record.get("ate", 0.0), record.get("ase", 0.0),
             _gt_text(record), elapsed))
    return record


def corrected_recall(metrics: dict, labelled) -> dict:
    """Recall against the labels, recovered from recall against coverage.

    The evaluator reports `recall@k = matched@k / gt_covered`. Multiplying back
    by `gt_covered` recovers the match count, which divided by the labelled
    total is the recall a reader assumes they are being given. Both are kept:
    the corrected number is the one to report, the raw one shows how much of
    the split each model actually engaged with.
    """
    if not labelled or not labelled.get("gt_labelled"):
        return {}
    total = float(labelled["gt_labelled"])
    covered = metrics.get("gt_covered")
    out = {"gt_labelled": int(total),
           "coverage": round(covered / total, 4) if covered else None}
    if not covered:
        return out
    for key in ("recall30", "recall50"):
        if metrics.get(key) is not None:
            out[key + "_true"] = round(metrics[key] * covered / total, 4)
    return out


def _recall_text(record: dict) -> str:
    if record.get("recall30_true") is not None:
        return "%.3f (raw %.3f)" % (record["recall30_true"], record.get("recall30", 0.0))
    return "%.3f" % record.get("recall30", 0.0)


def _gt_text(record: dict) -> str:
    if record.get("gt_labelled"):
        return "%d labelled, %d covered" % (record["gt_labelled"],
                                            int(record.get("gt_covered") or 0))
    return "%d covered" % int(record.get("gt_covered") or 0)


def summarise(records, methods, scene):
    """Mean and spread per method per fusion mode, across seeds."""
    rows = []
    for method in methods:
        row = {"method": method}
        for fusion in FUSION_MODES + (EGO_ONLY,):
            got = [r for r in records
                   if not r.get("failed") and not r.get("video")
                   and r["fusion"] == fusion
                   and ("isaacsim_%s_pretrained_%s_" % (method, scene))
                   in os.path.basename(r["model_dir"])]
            entry = {"seeds": len(got)}
            for key in ("ap30", "ap50", "ap70", "recall30", "recall50",
                        "recall30_true", "recall50_true", "coverage",
                        "ate", "ase", "gt_covered", "gt_labelled", "matched_0.5m"):
                values = [r[key] for r in got if key in r]
                if values:
                    entry[key] = {"mean": round(sum(values) / len(values), 4),
                                  "min": round(min(values), 4),
                                  "max": round(max(values), 4)}
            row[fusion] = entry
        rows.append(row)
    return rows


def markdown(rows):
    out = ["| method | seeds | AP@0.3 fused | AP@0.5 fused | AP@0.5 ego-only | lift | "
           "recall@0.3 | coverage | ATE m | ASE |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        # The real floor when it was measured; the tool's own "no" only as a
        # fallback, and that one is known to duplicate the fused run.
        fused = row.get("intermediate", {})
        alone = row.get(EGO_ONLY) or row.get("no", {})
        if not fused.get("ap50"):
            out.append("| %s | — | — | — | — | — | — | — | — | — |" % row["method"])
            continue
        floor = alone.get("ap50", {}).get("mean")
        lift = "" if floor is None else "%+.3f" % (fused["ap50"]["mean"] - floor)
        # recall30_true when the labelled denominator was available, because the
        # raw one is a fraction of what the model itself covered.
        recall = fused.get("recall30_true") or fused.get("recall30", {})
        coverage = fused.get("coverage", {}).get("mean")
        out.append("| %s | %d | %.3f | %.3f (%.3f-%.3f) | %s | %s | %.3f | %s | %.3f | %.3f |"
                   % (row["method"], fused["seeds"],
                      fused.get("ap30", {}).get("mean", 0.0),
                      fused["ap50"]["mean"], fused["ap50"]["min"], fused["ap50"]["max"],
                      "—" if floor is None else "%.3f" % floor, lift or "—",
                      recall.get("mean", 0.0),
                      "—" if coverage is None else "%.2f" % coverage,
                      fused.get("ate", {}).get("mean", 0.0),
                      fused.get("ase", {}).get("mean", 0.0)))
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--incop-root", required=True,
                        help="the InCoP checkout; inference runs with this as cwd")
    parser.add_argument("--dataset", required=True,
                        help="the split to evaluate, e.g. .../mirc_coop2_real_world/validate")
    parser.add_argument("--out", required=True, help="directory for results.json and the table")
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    parser.add_argument("--scene", default="hospital",
                        help="which pretrained weights to sweep (hospital/office/warehouse)")
    parser.add_argument("--checkpoint-mode", default="bestval", choices=["bestval", "latest"])
    parser.add_argument("--eval-class", default="chair",
                        help="which class's per-class numbers to report; the "
                             "multi-class mean is dragged to zero by the six "
                             "classes this scene does not contain")
    parser.add_argument("--seeds", type=int, default=0,
                        help="use only the first N seeds per method (0 = all)")
    parser.add_argument("--fusion", nargs="*", default=list(DEFAULT_FUSION),
                        choices=list(FUSION_MODES))
    parser.add_argument("--ego-floor", action="store_true",
                        help="also run each checkpoint with the partner out of "
                             "communication range, which is the only ego-only "
                             "baseline that actually drops the partner")
    parser.add_argument("--video", action="store_true",
                        help="also render the ego-only vs fused comparison for the "
                             "first seed of each method")
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="cap frames per run (0 = the whole split)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = parser.parse_args()

    args.incop_root = os.path.expanduser(args.incop_root)
    args.dataset = os.path.expanduser(args.dataset)
    args.out = os.path.abspath(os.path.expanduser(args.out))
    if not os.path.isdir(args.dataset):
        print("no such dataset directory: %s" % args.dataset, file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)

    # One sweep at a time. Two of these ran together once: both write eval yamls
    # into the SAME model directories, and results are collected by mtime, so one
    # sweep can read the other's file and attribute it to the wrong model. That
    # corrupts the table quietly, which is worse than the wasted GPU.
    lock = os.path.join(args.out, "sweep.lock")
    if os.path.exists(lock):
        with open(lock) as handle:
            owner = handle.read().strip()
        alive = owner.isdigit() and os.path.exists("/proc/%s" % owner)
        if alive:
            print("a sweep is already running here (pid %s). Stop it first:\n"
                  "  pkill -f run_mirc_incop\n"
                  "Two sweeps write into the same model directories and collect "
                  "results by mtime, so they read each other's files."
                  % owner, file=sys.stderr)
            return 2
        print("clearing a stale lock from pid %s" % owner)
    with open(lock, "w") as handle:
        handle.write(str(os.getpid()))

    plan = []
    # Video passes go LAST, in their own list. They render every frame of the
    # split through a Python BEV rasteriser and produce no metrics, so a single
    # one can outlast every measured run put together. Interleaved, one method's
    # video would sit between the next method's numbers and the operator.
    videos = []
    for method in args.methods:
        found = checkpoints(args.incop_root, method, args.scene)
        if not found:
            print("! no %s checkpoints for scene %s" % (method, args.scene))
            continue
        if args.seeds:
            found = found[:args.seeds]
        print("%-12s %d seed(s)" % (method, len(found)))
        for index, model_dir in enumerate(found):
            for fusion in args.fusion:
                plan.append((model_dir, fusion, False))
            if args.ego_floor:
                plan.append((ego_only_model_dir(model_dir, args.out),
                             EGO_ONLY, False))
            if args.video and index == 0:
                videos.append((model_dir, "intermediate", True))
    plan.extend(videos)

    if not plan:
        print("nothing to run", file=sys.stderr)
        return 2
    print("\n%d run(s) planned\n" % len(plan))

    # One honest denominator for the whole sweep. Counted once from the labels,
    # in the range the checkpoints score in, and reused for every run -- the
    # number of chairs in the split does not depend on which model is looking.
    labelled = None
    gt_range = eval_range(plan[0][0])
    if gt_range is None:
        print("! no gt_range in %s/config.yaml — reporting the evaluator's own "
              "counts uncorrected" % os.path.basename(plan[0][0]))
    else:
        labelled = labelled_gt_count(args.dataset, args.eval_class, gt_range,
                                     limit=args.max_samples)
        print("%d of %d %s label(s) lie inside %s, across %d ego frame(s)"
              % (labelled["gt_labelled"], labelled["gt_all"], args.eval_class,
                 [round(v, 1) for v in gt_range], labelled["frames"]))
        outside = labelled["gt_all"] - labelled["gt_labelled"]
        if outside:
            print("%d label(s) fall outside it and can never be scored by these "
                  "checkpoints.\n" % outside)
        else:
            print("")

    records = []
    for model_dir, fusion, video in plan:
        record = run_one(args, model_dir, fusion, video, labelled)
        if record is None:
            continue
        records.append(record)
        # Written after every run: a sweep this long should never lose finished
        # work to a failure in the run after it.
        with open(os.path.join(args.out, "results.json"), "w") as handle:
            json.dump(records, handle, indent=2)

    rows = summarise(records, args.methods, args.scene)
    table = markdown(rows)
    with open(os.path.join(args.out, "table.md"), "w") as handle:
        handle.write("# MIRC coop2, InCoP pretrained (%s), zero-shot\n\n" % args.scene)
        handle.write("Ego is `mobile_2`, the 87-degree depth cart; partner is `mobile_1`, "
                     "the Ouster.\n`lift` is fused minus ego-only at IoU 0.5.\n\n")
        handle.write(table + "\n")
    print("\n" + table)
    print("\nwrote %s/results.json and %s/table.md" % (args.out, args.out))
    try:
        os.remove(lock)
    except OSError:
        pass
    failed = [r for r in records if r.get("failed")]
    if failed:
        print("\n%d run(s) failed:" % len(failed))
        for record in failed:
            print("  %s [%s]" % (os.path.basename(record["model_dir"]), record["fusion"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

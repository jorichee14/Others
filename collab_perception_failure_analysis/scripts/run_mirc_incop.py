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
seed of each method, over the whole split. That pass is separate because
--video_compare_fusion runs the model twice per frame and disables metrics.

Nothing here is InCoP-specific beyond the CLI it shells out to; the results land
in the same shape as run_phase1.py's so the OPV2V and MIRC tables can sit side
by side.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import yaml

FUSION_MODES = ("intermediate", "no")     # with the partner; ego alone
DEFAULT_METHODS = ("ours", "where2comm", "cobevt")


def checkpoints(incop_root: str, method: str, scene: str):
    """Every seed of one method, oldest first, so run order is reproducible."""
    pattern = os.path.join(incop_root, "opencood", "logs",
                           f"isaacsim_{method}_pretrained_{scene}_20*")
    # `_with_noise_` variants are a different model, not another seed of this one.
    return sorted(path for path in glob.glob(pattern)
                  if os.path.isdir(path) and "_with_noise_" not in os.path.basename(path))


def newest_eval(model_dir: str, after: float):
    """The eval yaml this run just wrote, by mtime.

    Matched on time rather than name because the name encodes score threshold,
    NMS, epoch and dataset tag, and reconstructing it here would be a second
    implementation of the tool's own naming that could drift from it silently.
    """
    best, best_time = None, after
    for path in glob.glob(os.path.join(model_dir, "eval_*.yaml")):
        stamp = os.path.getmtime(path)
        if stamp >= best_time:
            best, best_time = path, stamp
    return best


def run_one(args, model_dir: str, fusion: str, video: bool):
    command = [
        sys.executable, "opencood/tools/inference_isaac.py",
        "--model_dir", model_dir,
        "--checkpoint_mode", args.checkpoint_mode,
        "--fusion_method", fusion,
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
    record = {
        "model_dir": model_dir, "fusion": fusion, "eval_yaml": path,
        "ap30": float(found.get("ap30", 0.0)),
        "ap50": float(found.get("ap_50", 0.0)),
        "ap70": float(found.get("ap_70", 0.0)),
        "runtime_s": round(elapsed, 1),
    }
    print("  AP@0.3/0.5/0.7 = %.3f / %.3f / %.3f  (%.0fs)"
          % (record["ap30"], record["ap50"], record["ap70"], elapsed))
    return record


def summarise(records, methods, scene):
    """Mean and spread per method per fusion mode, across seeds."""
    rows = []
    for method in methods:
        row = {"method": method}
        for fusion in FUSION_MODES:
            got = [r for r in records
                   if not r.get("failed") and not r.get("video")
                   and r["fusion"] == fusion
                   and ("isaacsim_%s_pretrained_%s_" % (method, scene))
                   in os.path.basename(r["model_dir"])]
            entry = {"seeds": len(got)}
            for key in ("ap30", "ap50", "ap70"):
                if got:
                    values = [r[key] for r in got]
                    entry[key] = {"mean": round(sum(values) / len(values), 4),
                                  "min": round(min(values), 4),
                                  "max": round(max(values), 4)}
            row[fusion] = entry
        rows.append(row)
    return rows


def markdown(rows):
    out = ["| method | seeds | AP@0.5 fused | AP@0.5 ego-only | lift | "
           "AP@0.3 fused | AP@0.7 fused |",
           "|---|---|---|---|---|---|---|"]
    for row in rows:
        fused, alone = row.get("intermediate", {}), row.get("no", {})
        if not fused.get("ap50") or not alone.get("ap50"):
            out.append("| %s | — | — | — | — | — | — |" % row["method"])
            continue
        lift = fused["ap50"]["mean"] - alone["ap50"]["mean"]
        out.append("| %s | %d | %.3f (%.3f-%.3f) | %.3f | %+.3f | %.3f | %.3f |"
                   % (row["method"], fused["seeds"],
                      fused["ap50"]["mean"], fused["ap50"]["min"], fused["ap50"]["max"],
                      alone["ap50"]["mean"], lift,
                      fused["ap30"]["mean"], fused["ap70"]["mean"]))
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
    parser.add_argument("--seeds", type=int, default=0,
                        help="use only the first N seeds per method (0 = all)")
    parser.add_argument("--fusion", nargs="*", default=list(FUSION_MODES),
                        choices=list(FUSION_MODES))
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
    args.out = os.path.expanduser(args.out)
    if not os.path.isdir(args.dataset):
        print("no such dataset directory: %s" % args.dataset, file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)

    plan = []
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
            if args.video and index == 0:
                plan.append((model_dir, "intermediate", True))

    if not plan:
        print("nothing to run", file=sys.stderr)
        return 2
    print("\n%d run(s) planned\n" % len(plan))

    records = []
    for model_dir, fusion, video in plan:
        record = run_one(args, model_dir, fusion, video)
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
    failed = [r for r in records if r.get("failed")]
    if failed:
        print("\n%d run(s) failed:" % len(failed))
        for record in failed:
            print("  %s [%s]" % (os.path.basename(record["model_dir"]), record["fusion"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

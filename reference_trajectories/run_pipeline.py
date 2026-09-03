#!/usr/bin/env python3
"""
One entry point for the whole pipeline.

    python3 run_pipeline.py --list
    python3 run_pipeline.py --run coop2 --stages 08,09
    python3 run_pipeline.py --run coop2 --from 08
    python3 run_pipeline.py --run coop2 --all --keep-going
    python3 run_pipeline.py --run coop2 --stages ntp --dry

Every stage is one entry in the registry below (or in the config's
"pipeline" block, which overrides it). A stage is either

  kind "script"  an existing standalone script, run as a subprocess exactly
                 the way you run it by hand today. This is what lets the
                 pipeline adopt 01-07 and ntp_analysis.py without rewriting
                 them: give the file and, if it does not take the config as
                 argv[1], an `args` template.
                 A stage may also name its own interpreter with "python"
                 (or a group default in "pipeline": {"python": {...}}). Bag
                 stages need the ROS interpreter; analysis stages read
                 parquet and are happier in a venv where numpy/pandas/
                 sklearn are one consistent set. That split is the fix for
                 "module numpy has no attribute _CopyMode": ROS puts the old
                 apt numpy ahead of the newer ~/.local one, so anything
                 importing sklearn must not run in the ROS interpreter.
  kind "module"  a python module with `run(cfg, ctx)`. These share one Ctx,
                 so the reference map and the parquet cache are built once
                 for the whole invocation instead of once per stage.

Placeholders usable in `args`: {config} {bag} {run} {out_root} {results_root}

Stages run in registry order; --stages selects a subset but keeps that
order, so a dependency can never run after its consumer. Each stage's output
is teed to <out_root>/logs/<stage>.log and a timing summary is printed at the
end.
"""
import argparse
import importlib
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mircpipe.config import Ctx, load_config      # noqa: E402

# ------------------------------------------------------------------ registry
# name, kind, target, config block (for the bag), argv template
DEFAULT_STAGES = [
    dict(name="01", kind="script", target="01_build_map.py", block="01_build_map",
         help="build the coloured lidar map from the mapping bag"),
    dict(name="02", kind="script", target="02_cut.py", block="02_cut",
         help="crop the map in z"),
    dict(name="03", kind="script", target="03_anchor.py", block="03_anchor",
         help="survey the boards, anchor the map"),
    dict(name="04", kind="script", target="04_build_cameras.py",
         block="04_build_cameras", help="static cameras in map"),
    dict(name="05", kind="script", target="05_emit_tfs.py", block="05_emit_tfs",
         help="publishable TFs"),
    dict(name="06", kind="script", target="06_init.py", block="06_init",
         help="session anchor per sensor from the opening dwell"),
    dict(name="08", kind="script", target="08_reference_traj.py",
         block="08_reference", help="reference trajectories per sensor"),
    dict(name="09", kind="script", target="09_publish_poses.py",
         block="09_publish", help="best pose per robot -> mcap bag"),
    dict(name="ntp", kind="script", target="ntp_analysis.py", block=None,
         args=["--bag", "{bag}", "--run", "{run}"], python="analysis",
         help="clock sync analysis (parquet cache + report)"),
    dict(name="wifi", kind="script", target="wifi_analysis.py", block=None,
         args=["--bag", "{bag}", "--run", "{run}"], optional=True,
         python="analysis",
         help="wifi/iperf analysis"),
    dict(name="csi", kind="script", target="csi_analysis.py", block=None,
         args=["--bag", "{bag}", "--run", "{run}"], optional=True,
         python="analysis",
         help="CSI analysis"),
    dict(name="radar", kind="script", target="radar_analysis.py", block=None,
         args=["--bag", "{bag}", "--run", "{run}"], optional=True,
         python="analysis",
         help="radar analysis"),
]


def registry(cfg):
    """Registry from the config's "pipeline".stages if present, else the
    default list. A config entry may also patch a default one by name."""
    pl = cfg.get("pipeline", {}) or {}
    if pl.get("stages"):
        return [dict(s) for s in pl["stages"]]
    out = [dict(s) for s in DEFAULT_STAGES]
    for patch in (pl.get("override") or []):
        for s in out:
            if s["name"] == patch.get("name"):
                s.update(patch)
    for extra in (pl.get("extra") or []):
        out.append(dict(extra))
    return out


def interpreter(cfg, st):
    """The python that runs a script stage.

    "python" on the stage is either a path or the name of a group defined in
    the config:  "pipeline": { "python": { "ros": "/usr/bin/python3",
                                           "analysis": "~/.venvs/mirc/bin/python3" } }
    An unknown or missing group falls back to the interpreter running this
    file, so a config without the block behaves exactly as before."""
    groups = ((cfg.get("pipeline", {}) or {}).get("python") or {})
    want = st.get("python")
    if not want:
        return sys.executable, None
    path = os.path.expanduser(groups.get(want, want))
    if os.path.isabs(path) and not os.path.exists(path):
        return sys.executable, ("group '%s' -> %s does not exist, using %s"
                                % (want, path, sys.executable))
    if not os.path.isabs(path):
        return sys.executable, None            # named group not configured
    return path, None


def resolve(cfg, st, run):
    """(path to the target, argv list) for a script stage."""
    target = st["target"]
    path = target if os.path.isabs(target) else os.path.join(HERE, target)
    blk = cfg.stage(st["block"], required=False) if st.get("block") else None
    try:
        bag = cfg.bag_of(blk, run)
    except SystemExit:
        bag = ""
    fields = dict(config=cfg_path_global, bag=bag, run=run or "",
                  out_root=cfg.out_root(), results_root=cfg.results_root())
    if st.get("args"):
        argv = [a.format(**fields) for a in st["args"]]
    else:
        argv = [cfg_path_global]                # the pipeline convention
    return path, argv


def run_script(cfg, st, run, log_path, dry):
    path, argv = resolve(cfg, st, run)
    if not os.path.exists(path):
        return ("missing", "%s not found" % path)
    py, warn = interpreter(cfg, st)
    if warn:
        print("  ! %s" % warn)
    cmd = [py, path] + argv
    print("  $ " + " ".join(cmd))
    if dry:
        return ("dry", "")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
        p.wait()
    return ("ok" if p.returncode == 0 else "fail:%d" % p.returncode, log_path)


def run_module(cfg, st, ctx, dry):
    mod = importlib.import_module(st["target"])
    if dry:
        print("  (module %s.run)" % st["target"])
        return ("dry", "")
    mod.run(cfg, ctx)
    return ("ok", "")


def main():
    ap = argparse.ArgumentParser(description="MIRC pipeline runner")
    ap.add_argument("--config", default="pipeline_config.json")
    ap.add_argument("--run", help="a key of the config's \"runs\" block, e.g. coop2")
    ap.add_argument("--stages", help="comma separated names, e.g. 08,09,ntp")
    ap.add_argument("--from", dest="from_", help="start at this stage, then all later ones")
    ap.add_argument("--all", action="store_true", help="every registered stage")
    ap.add_argument("--list", action="store_true", help="show the registry and exit")
    ap.add_argument("--dry", action="store_true", help="print what would run")
    ap.add_argument("--keep-going", action="store_true", help="do not stop on failure")
    args = ap.parse_args()

    global cfg_path_global
    cfg_path_global = os.path.abspath(os.path.expanduser(args.config))
    cfg = load_config(cfg_path_global)
    reg = registry(cfg)

    if args.list:
        print("config: %s" % cfg_path_global)
        print("runs:   %s" % ", ".join(sorted(cfg.get("runs", {}))) or "(none)")
        print("out:    %s\nresults:%s\n" % (cfg.out_root(), cfg.results_root()))
        print("%-7s %-8s %-26s %-16s %s" % ("stage", "kind", "target", "block", "what"))
        for s in reg:
            tgt = s["target"]
            here = os.path.exists(os.path.join(HERE, tgt)) or os.path.isabs(tgt)
            print("%-7s %-8s %-26s %-16s %s%s"
                  % (s["name"], s["kind"], tgt, s.get("block") or "-",
                     s.get("help", ""), "" if here else "   [script not present]"))
        return 0

    names = [s["name"] for s in reg]
    if args.stages:
        want = [n.strip() for n in args.stages.split(",") if n.strip()]
        unknown = [n for n in want if n not in names]
        if unknown:
            raise SystemExit("unknown stage(s): %s; known: %s"
                             % (", ".join(unknown), ", ".join(names)))
        sel = [s for s in reg if s["name"] in want]
    elif args.from_:
        if args.from_ not in names:
            raise SystemExit("unknown stage %s" % args.from_)
        sel = reg[names.index(args.from_):]
    elif args.all:
        sel = reg
    else:
        raise SystemExit("choose --stages, --from, --all or --list")

    ctx = Ctx(cfg, run=args.run)
    print("== pipeline: %s | run '%s' | %d stage(s) =="
          % (os.path.basename(cfg_path_global), args.run or "-", len(sel)))
    results, t_all = [], time.time()
    for st in sel:
        print("\n-- stage %s (%s) --" % (st["name"], st.get("help", st["target"])))
        t0 = time.time()
        try:
            if st["kind"] == "module":
                status, note = run_module(cfg, st, ctx, args.dry)
            else:
                log = os.path.join(cfg.out_root(), "logs", "%s.log" % st["name"])
                status, note = run_script(cfg, st, args.run, log, args.dry)
        except SystemExit as e:
            status, note = "fail", str(e)
        except Exception as e:
            status, note = "error", "%s: %s" % (type(e).__name__, e)
        dt = time.time() - t0
        results.append((st["name"], status, dt, note))
        if status == "missing" and st.get("optional"):
            print("  (optional stage not present - skipped)")
            continue
        if not status.startswith(("ok", "dry")) and not args.keep_going:
            print("  ! stage %s: %s - stopping (use --keep-going to continue)"
                  % (st["name"], note or status))
            break

    print("\n== summary (%.1f s total) ==" % (time.time() - t_all))
    print("  %-7s %-10s %9s  %s" % ("stage", "status", "seconds", "note"))
    for name, status, dt, note in results:
        print("  %-7s %-10s %9.1f  %s" % (name, status, dt, note))
    bad = [r for r in results if not r[1].startswith(("ok", "dry"))
           and r[1] != "missing"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

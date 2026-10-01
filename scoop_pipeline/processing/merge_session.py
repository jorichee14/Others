#!/usr/bin/env python3
"""Check a pass's bags against record.yaml, then merge them into one.

    python scoop_pipeline/processing/merge_session.py <raw pass folder>
        [--check] [--allow-missing] [--out DIR] [--name NAME]
        [--compression zstd|none] [--mode mapping|survey|coop|contention]
        [--record configs/record.yaml] [--static-tf configs/static_tf.yaml]
        [--settings configs/recording.yaml] [--work-root DIR]

<raw pass folder> holds one folder per machine, e.g.
    ~/workspaces/isaac_ros-dev/data/raw/20260924/mapping_A   (mobile_1/, mobile_2/, ...)

For every machine (configs/record.yaml) it lists the topics its bags must
have -- processed bags where they exist (mobile_1: retimed + ZED), raw ones
otherwise -- and flags MISSING and EMPTY ones, like record.sh's preflight.
What a pass mode does not record is left out (record.yaml's check section;
the mode comes from the pass folder's name, e.g. mapping_A, or --mode).
--check stops there. Otherwise, if nothing is missing (or --allow-missing),
all bags are merged in log-time order into
    data/work/<date>/<pass>/<prefix>_<pass>_..._<date>_merged/
(--out puts it elsewhere, e.g. another drive), zstd-compressed
(lossless; --compression none when an uncompressed merge fits), and the merged bag is
checked the same way. The calibrated
transforms of configs/static_tf.yaml are added to its /tf_static.
"""
import argparse
import os
import shutil
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", FutureWarning)

from scoop import merge, recording, session, tftree                  # noqa: E402
from scoop.bag import BagError                                      # noqa: E402


def show(check, indent="    "):
    w = max([len(t) for t in check.counts] + [20])
    for t in sorted(set(check.counts) - set(check.extra)):
        tag = "ok" if check.counts[t] else ("ok/none" if t in check.quiet else "EMPTY")
        print(f"{indent}{tag:8s}{t:{w}s} {check.counts[t]:9d}")
    if check.quiet:
        print(f"{indent}(ok/none: no messages, allowed by check.may_be_empty in record.yaml)")
    for t in check.missing:
        print(f"{indent}{'MISSING':8s}{t}")
    if check.extra:
        print(f"{indent}also there (not in record.yaml): {', '.join(check.extra)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir")
    ap.add_argument("--check", action="store_true", help="check the topics and stop")
    ap.add_argument("--allow-missing", action="store_true",
                    help="merge even if expected topics are missing or empty")
    ap.add_argument("--out", default=None, help="folder for the merged bag "
                    "(default: the pass folder below data/work)")
    ap.add_argument("--name", default=None, help="merged bag name without _merged "
                    "(default: <prefix>_<pass>_<date> from the folders)")
    ap.add_argument("--compression", default="zstd", choices=["zstd", "none"],
                    help="zstd (default): lossless, about half the size; none: faster")
    ap.add_argument("--mode", default=None, help="pass mode (default: from the folder name)")
    ap.add_argument("--record", default=None, help="default: configs/record.yaml")
    ap.add_argument("--static-tf", default=str(recording.ROOT / "configs" / "static_tf.yaml"),
                    help="calibrated transforms to add to /tf_static ('' for none)")
    ap.add_argument("--settings", default=None, help="default: configs/recording.yaml")
    ap.add_argument("--work-root", default=None)
    a = ap.parse_args()
    try:
        plan = session.RecordPlan(a.record)
        settings = recording.load_settings(a.settings)
        raw = recording.Path(a.raw_dir).expanduser().resolve()
        mode = a.mode or plan.mode_of(raw.name, *(p.parent.name for p in raw.rglob("metadata.yaml")))
        if mode is not None and mode not in plan.modes:
            raise ValueError(f"--mode {mode}: record.yaml has {list(plan.modes)}")
        print(f"mode: {mode or 'unknown (no mode name in the folder or bag names)'}")
        units = session.find_units(raw, plan, settings, a.work_root, mode)
        bad = []
        for u in units:
            print(f"\n{u.machine}  ({'processed' if u.processed else 'raw'}: "
                  f"{', '.join(b.name for b in u.bags)})")
            c = u.check()
            show(c)
            if not c.ok:
                bad.append(u.machine)
        missing = sorted(set(plan.machines) - {u.machine for u in units})
        if missing:
            print(f"\nno bags for: {', '.join(missing)}")
        if bad:
            print(f"\nMISSING or EMPTY topics on: {', '.join(bad)}")
        if a.check:
            return
        if bad and not a.allow_missing:
            sys.exit("not merged; fix the bags or pass --allow-missing")

        name = f"{a.name or session.session_name(units, plan, raw)}_merged"
        out_root = (os.path.expanduser(a.out) if a.out
                    else recording.work_dir_for(a.raw_dir, a.work_root))
        final = recording.Path(out_root) / name
        if final.exists():
            sys.exit(f"{final} exists; remove it to merge again")
        inputs = [b for u in units for b in u.bags]
        static_tf = []
        if a.static_tf:
            import yaml
            with open(os.path.expanduser(a.static_tf)) as fh:
                cals = tftree.load_calibrations(yaml.safe_load(fh))
            if cals:
                static_tf, notes = tftree.attach(cals, tftree.tf_edges(inputs))
                print("\nstatic transforms:")
                for n in notes:
                    print(f"    {n}")
        need = sum(merge.bag_bytes(b) for b in inputs) / 1e9
        final.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(final.parent).free / 1e9
        print(f"\ninputs {need:.0f} GB, {free:.0f} GB free at {final.parent}")
        if a.compression == "none" and free < need + merge.MIN_FREE_GB:
            sys.exit("not enough space uncompressed; leave --compression at zstd or use --out")
        tmp = final.parent / ".partial" / name
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        merge.merge_bags(inputs, tmp, compression=a.compression, static_tf=static_tf)
        os.replace(tmp, final)
        try:
            tmp.parent.rmdir()
        except OSError:
            pass
        print(f"\nmerged bag: {final}")
        # a may_be_empty topic that never had a message is not in the merged
        # bag (a topic is written with its first message): the recordings
        # having it is enough, as in the check before the merge
        c = session.check_topics([final], [t for u in units for t in u.expected],
                                 plan.may_be_empty, [b for u in units for b in u.raw_bags])
        show(c)
        if not c.ok:
            print("\nMISSING or EMPTY topics in the merged bag")
    except (BagError, ValueError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()

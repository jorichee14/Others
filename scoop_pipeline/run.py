#!/usr/bin/env python3
"""SCooP pipeline runner.

    python3 run.py status site_1                      # every pass + site stage
    python3 run.py status site_1/session_02/seq_05
    python3 run.py run site_1                         # passes A, B..., then site stages
    python3 run.py run site_1/mapping_A --until s02_refine
    python3 run.py run site_1 --from s03_final_map    # force s03, then what follows
    python3 run.py run site_1 --only s04_cut --force s04_cut
    python3 run.py run site_1 --dry-run
    python3 run.py stages                             # the stage list

A unit is ``site_X`` (all mapping passes, then the site stages),
``site_X/mapping_P`` (one pass), or ``site_X/session_YY/seq_ZZ``.
Config: configs/defaults.yaml + configs/sites/<site>.yaml; $SCOOP_DATA
overrides data_root.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scoop import runner                                            # noqa: E402
from scoop.config import ConfigError, load_config                   # noqa: E402
from scoop.layout import Layout, LayoutError, parse_unit            # noqa: E402
from scoop.stages import ALL, ORDER                                 # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("status", help="state of every stage for a unit")
    st.add_argument("unit")
    rn = sub.add_parser("run", help="run what is stale")
    rn.add_argument("unit")
    rn.add_argument("--from", dest="start", help="force this stage and continue after it")
    rn.add_argument("--until", help="stop after this stage")
    rn.add_argument("--only", action="append", default=[], help="run only these stages")
    rn.add_argument("--force", action="append", default=[], help="re-run even if current")
    rn.add_argument("--dry-run", action="store_true")
    sub.add_parser("stages", help="list the stages")
    a = ap.parse_args(argv)

    if a.cmd == "stages":
        for n in ORDER:
            s = ALL[n]
            mark = "" if s.implemented else f"  [phase {s.phase}, not implemented]"
            print(f"  {n:18s} {s.level:5s} {s.doc}{mark}")
        return 0

    try:
        unit = parse_unit(a.unit)
        cfg = load_config(unit.site)
        layout = Layout(cfg)
        items = runner.plan(unit, cfg, layout)
    except (ConfigError, LayoutError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"{unit.rel}  (data_root {cfg['data_root']})")
    if a.cmd == "status":
        runner.status(items)
        return 0
    items = runner.select(items, only=a.only, start=a.start, until=a.until)
    force = list(a.force) + ([a.start] if a.start else [])
    return runner.execute(items, cfg, force=force, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())

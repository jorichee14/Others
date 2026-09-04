#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage B: convert a rosbag2 recording into an OPV2V dataset OpenCOOD can read.

    python scripts/convert_rosbag.py --config configs/mirc_coop2.yaml [--dry-run]

``--dry-run`` runs the index + synchronisation + pose resolution stages and
prints the frame budget without writing anything — always do that first, because
it is where tolerance and alignment problems show up, and it costs seconds
instead of the full decode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ros2opv2v.bagreader import BagReader                       # noqa: E402
from ros2opv2v.config import ConfigError, load_config           # noqa: E402
from ros2opv2v.convert import (ConversionError, ConversionReport,  # noqa: E402
                               assign_scenarios, convert, plan)


def print_report(report: ConversionReport, dry_run: bool = False) -> None:
    print("\n" + "=" * 78)
    print("DRY RUN — nothing written" if dry_run else "CONVERSION COMPLETE")
    print("=" * 78)
    print(f"bag            : {report.bag}")
    if not dry_run:
        print(f"output         : {report.output}")
        print(f"scenarios      : {', '.join(report.scenarios)}")
    print(f"frames         : {report.frames_written} kept "
          f"of {report.frames_candidate} candidates")

    if report.dropped:
        print("\ndropped frames")
        for reason, count in sorted(report.dropped.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>6}  {reason}")

    if report.sync:
        print("\nper-agent synchronisation")
        print(f"  {'agent':<16} {'frames':>7} {'src Hz':>7} {'reuse':>7} "
              f"{'mean off':>9} {'max off':>9}")
        for name, entry in sorted(report.sync.items()):
            print(f"  {name:<16} {int(entry.get('frames', 0)):>7} "
                  f"{entry.get('source_rate_hz', 0):>7.2f} "
                  f"{entry.get('reuse_rate', 0) * 100:>6.1f}% "
                  f"{entry.get('mean_offset_ms', 0):>8.1f}ms "
                  f"{entry.get('max_offset_ms', 0):>8.1f}ms")

    if report.pose_stats:
        print("\nper-agent trajectory (in the shared world frame)")
        print(f"  {'agent':<16} {'path m':>9} {'max step m':>11}  extent xyz (m)")
        for name, entry in sorted(report.pose_stats.items()):
            print(f"  {name:<16} {entry['path_length_m']:>9.2f} "
                  f"{entry['max_step_m']:>11.3f}  {entry['extent_m']}")

    if report.points_per_agent:
        print("\npoints per frame")
        for name, entry in sorted(report.points_per_agent.items()):
            print(f"  {name:<16} mean {entry['mean_points']:>9.1f}  "
                  f"min {entry['min_points']:>7}  max {entry['max_points']:>7}")

    if report.warnings:
        print("\nwarnings")
        for warning in report.warnings:
            print(f"  ! {warning}")

    if not dry_run:
        print(f"\nelapsed: {report.duration_s:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="converter config yaml")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing scenario folders")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan only: no pcd/yaml written")
    parser.add_argument("--bag", default=None, help="override config's bag path")
    parser.add_argument("--out", default=None, help="override config's output.root")
    parser.add_argument("--duration", type=float, default=None,
                        help="override time.duration_s (seconds; 0 = whole bag)")
    parser.add_argument("--json", default=None, help="also write the report as json")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 2

    if args.bag:
        cfg.bag = os.path.expanduser(args.bag)
    if args.out:
        cfg.output.root = os.path.expanduser(args.out)
    if args.duration is not None:
        cfg.time.duration_s = args.duration

    try:
        if args.dry_run:
            report = ConversionReport(bag=cfg.bag)
            reader = BagReader(cfg.bag, cfg.time.stamp_source)
            frames, _, _, _, _ = plan(reader, cfg, report)
            report.frames_written = len(frames)
            report.scenarios = [name for name, _ in assign_scenarios(frames, cfg)]
        else:
            report = convert(
                cfg, overwrite=args.overwrite,
                progress=lambda done, total: print(
                    f"\r  writing {done}/{total} messages", end="", flush=True))
            print()
    except (ConversionError, ConfigError) as error:
        print(f"\nconversion failed: {error}", file=sys.stderr)
        return 1

    print_report(report, dry_run=args.dry_run)

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report.to_dict(), handle, indent=2)
        print(f"\nreport written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

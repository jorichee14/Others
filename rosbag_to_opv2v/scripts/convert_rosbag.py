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
        runs = getattr(report, "dropped_runs", None) or []
        if runs:
            print("  contiguous outages of 3+ frames (scattered single drops not listed):")
            for r in sorted(runs, key=lambda r: -r["frames"])[:12]:
                print(f"    {r['frames']:>4} frames  t = {r['t_start_s']:>6.1f} .. "
                      f"{r['t_end_s']:>6.1f} s   {r['reason']}")

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

    clocks = {k: v for k, v in (report.clocks or {}).items() if not k.startswith("_")}
    if clocks:
        meta = report.clocks.get("_meta", {})
        mode = next(iter(clocks.values())).get("mode", "?")
        print(f"\nhost clocks  (mode: {mode}, reference: {meta.get('reference_host', '?')})")
        if mode == "verify":
            print("  stamps are taken as already disciplined (chrony/ntpd); nothing is "
                  "shifted.\n  'ntp p95' is the residual the daemon itself reports; "
                  "'floor Δ' is the\n  delivery-floor estimate of the same offset, "
                  "independent of the daemon.")
        fields = meta.get("ntp_fields", {})
        print(f"  {'host':<12} {'ntp p95':>9} {'floor Δ':>9} {'agree?':>12} "
              f"{'residual':>9} {'bound':>8}  {'sync':>5} {'strat':>5} {'reach':>5}")
        for host, entry in sorted(clocks.items()):
            ntp = entry.get("ntp", {})
            p95 = f"{ntp['p95_abs_ms']:.2f}ms" if ntp else "—"
            detail = entry.get("cross_check_detail", {})
            floor = (f"{detail['delivery_floor_correction_ms']:+.1f}ms"
                     if detail else "—")
            info = fields.get(host, {})
            bound = (f"{info['bound_p95_ms']:.1f}ms"
                     if info.get("bound_p95_ms") is not None else "—")
            health = info.get("health") or {}
            sync = ("ok" if health.get("unsynced_samples") == 0 else
                    f"{health['unsynced_samples']}!" if health.get("unsynced_samples") else "—")
            strat = ("/".join(str(x) for x in health["strata"])
                     if health.get("strata") else "—")
            reach = (f"{health['reachability_min_pct']}%"
                     if health.get("reachability_min_pct") is not None else "—")
            print(f"  {host:<12} {p95:>9} {floor:>9} {entry['cross_check']:>12} "
                  f"{entry['residual_ms']:>7.2f}ms {bound:>8}  {sync:>5} {strat:>5} {reach:>5}")
        print("  residual = max(daemon offset p95, jitter p95), carried per frame; "
              "bound = the daemon's formal worst case (root dispersion)")
        if mode == "correct":
            for host, entry in sorted(clocks.items()):
                print(f"  {host:<12} applied {entry['correction_ms']:+.2f} ms "
                      f"({entry['correction_source']})")
        for host, rows in (report.clocks.get("_events") or {}).items():
            for row in rows[:8]:
                print(f"  {host:<12} event t={row['t_rel_s']:>6.1f}s  {row['text']}")
            if len(rows) > 8:
                print(f"  {host:<12} ... {len(rows) - 8} more event(s) in the report json")

    if report.pose_stats:
        print("\nper-agent trajectory (base frame, in the shared world frame)")
        print(f"  {'agent':<16} {'path m':>9} {'max step m':>11}  extent xyz (m)")
        for name, entry in sorted(report.pose_stats.items()):
            print(f"  {name:<16} {entry['path_length_m']:>9.2f} "
                  f"{entry['max_step_m']:>11.3f}  {entry['extent_m']}")
        # The start pose is the one number that catches a pose source in the
        # wrong frame: compare it with the published session anchor by eye
        # even when pose.expected_start is not set.
        for name, entry in sorted(report.pose_stats.items()):
            line = f"  {name:<16} starts at {entry['start_m']}"
            if 'expected_start_m' in entry:
                line += (f"  (expected {entry['expected_start_m']}, "
                         f"{entry['start_gap_m']:.3f} m off)")
            print(line + f"  ends at {entry['end_m']}")
        for name, entry in sorted(report.pose_stats.items()):
            steps = entry.get('largest_steps') or []
            if steps and steps[0]['step_m'] > 0.0:
                print(f"  {name:<16} fastest moves: " + ", ".join(
                    f"{d['speed_mps']:.2f} m/s at t={d['t_s']:.1f}s"
                    + (f" ({d['step_m']:.2f} m over a {d['dt_s']:.1f} s gap)" if d['gap']
                       else "")
                    for d in steps)
                    + f"   (z span {entry.get('z_span_m', 0):.3f} m)")

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

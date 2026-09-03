"""Command line entry point: ``python -m rosbag2opv2v <command>``."""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_topics(args) -> int:
    from mcap.reader import make_reader

    from .bag import resolve_bag_files

    files = resolve_bag_files(args.bag)
    totals = {}
    for path in files:
        with open(path, "rb") as handle:
            summary = make_reader(handle).get_summary()
            if summary is None:
                print("%s: no summary section" % path, file=sys.stderr)
                continue
            counts = (summary.statistics.channel_message_counts
                      if summary.statistics else {})
            for channel in summary.channels.values():
                schema = summary.schemas.get(channel.schema_id)
                entry = totals.setdefault(
                    channel.topic, {"type": schema.name if schema else "",
                                    "count": 0})
                entry["count"] += counts.get(channel.id, 0)
    width = max((len(t) for t in totals), default=10)
    for topic in sorted(totals):
        print("%-*s  %8d  %s" % (width, topic, totals[topic]["count"],
                                 totals[topic]["type"]))
    return 0


def _cmd_convert(args) -> int:
    from .config import Config, default_config_path
    from .convert import Converter

    cfg = Config.load(args.config or default_config_path())
    if args.split:
        cfg.split = args.split
        cfg.splits = None
    if args.max_frames:
        cfg.max_frames = args.max_frames
    if args.no_images:
        cfg.write_images = False
    converter = Converter(cfg, args.bag, args.out,
                          scenario_prefix=args.scenario_prefix,
                          verbose=not args.quiet)
    report = converter.convert(dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(report, indent=2, default=str))
        return 0

    if args.emit_hypes:
        from . import opencood_hypes

        text = opencood_hypes.build(args.out, fusion=args.fusion)
        with open(args.emit_hypes, "w") as handle:
            handle.write(text)
        print("wrote OpenCOOD config %s" % args.emit_hypes)

    if args.verify:
        from .verify import verify

        result = verify(args.out, sample=args.verify_samples,
                        verbose=not args.quiet)
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 1
    return 0


def _cmd_verify(args) -> int:
    from .verify import verify

    result = verify(args.root, sample=args.samples, verbose=not args.quiet)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


def _cmd_hypes(args) -> int:
    from . import opencood_hypes

    text = opencood_hypes.build(args.root, fusion=args.fusion,
                                voxel_xy=args.voxel)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
        print("wrote %s" % args.out)
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rosbag2opv2v",
        description="Convert a rosbag2 MCAP recording into an OPV2V dataset "
                    "that OpenCOOD can train on.")
    sub = parser.add_subparsers(dest="command", required=True)

    topics = sub.add_parser("topics", help="list the topics in a bag")
    topics.add_argument("--bag", required=True,
                        help="path to a .mcap file or a rosbag2 directory")
    topics.set_defaults(func=_cmd_topics)

    convert = sub.add_parser("convert", help="run the conversion")
    convert.add_argument("--bag", required=True)
    convert.add_argument("--config", help="converter config yaml "
                                          "(default: configs/mirc_coop2.yaml)")
    convert.add_argument("--out", required=True, help="output dataset root")
    convert.add_argument("--split", help="force every scenario into this split")
    convert.add_argument("--scenario-prefix")
    convert.add_argument("--max-frames", type=int)
    convert.add_argument("--no-images", action="store_true")
    convert.add_argument("--dry-run", action="store_true",
                         help="plan only: report what would be written")
    convert.add_argument("--verify", action="store_true",
                         help="validate the result after converting")
    convert.add_argument("--verify-samples", type=int, default=20)
    convert.add_argument("--emit-hypes", metavar="FILE",
                         help="also write an OpenCOOD training config")
    convert.add_argument("--fusion", default="intermediate",
                         choices=["intermediate", "early", "late"])
    convert.add_argument("--quiet", action="store_true")
    convert.set_defaults(func=_cmd_convert)

    verify = sub.add_parser("verify", help="check a converted dataset")
    verify.add_argument("--root", required=True)
    verify.add_argument("--samples", type=int, default=20,
                        help="frames per scenario to check geometrically")
    verify.add_argument("--quiet", action="store_true")
    verify.set_defaults(func=_cmd_verify)

    hypes = sub.add_parser("hypes", help="emit an OpenCOOD training config")
    hypes.add_argument("--root", required=True)
    hypes.add_argument("--out")
    hypes.add_argument("--voxel", type=float, default=0.1)
    hypes.add_argument("--fusion", default="intermediate",
                       choices=["intermediate", "early", "late"])
    hypes.set_defaults(func=_cmd_hypes)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

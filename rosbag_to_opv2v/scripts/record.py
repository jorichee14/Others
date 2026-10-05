#!/usr/bin/env python3
"""Record one machine's topics, or merge the machines' bags, from one yaml.

    python3 scripts/record.py --config configs/record_4machines.yaml --machine mobile_1
    python3 scripts/record.py --config configs/record_4machines.yaml --machine mobile_1 --dry-run
    python3 scripts/record.py --config configs/record_4machines.yaml --merge <dir>

Recording builds a `ros2 bag record` command from the config and execs it, so
Ctrl-C stops the recorder exactly as it would stop a hand-typed one. Merging
runs `ros2 bag convert` with every bag found under <dir> as an input.
"""
import argparse
import datetime
import os
import shlex
import sys
import tempfile
from pathlib import Path

import yaml


def load(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("session", "record", "machines"):
        if key not in cfg:
            sys.exit(f"{path}: missing top-level key '{key}'")
    return cfg


def write_qos(overrides):
    fd, path = tempfile.mkstemp(prefix="qos_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(overrides, f)
    return path


def record_cmd(cfg, machine):
    if machine not in cfg["machines"]:
        sys.exit(f"unknown machine '{machine}'; config has: {', '.join(cfg['machines'])}")
    rec = cfg["record"]
    topics = list(dict.fromkeys(cfg.get("common_topics", []) + cfg["machines"][machine]["topics"]))

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(os.path.expanduser(cfg["session"]["output_root"]))
    out = root / f"{cfg['session']['name']}_{stamp}_{machine}"

    cmd = ["ros2", "bag", "record", "-s", rec.get("storage", "mcap"), "-o", str(out)]
    if rec.get("max_bag_size_gb"):
        cmd += ["--max-bag-size", str(int(rec["max_bag_size_gb"] * 1024 ** 3))]
    if rec.get("max_cache_size_mb"):
        cmd += ["--max-cache-size", str(int(rec["max_cache_size_mb"] * 1024 ** 2))]
    if rec.get("compression", "none") != "none":
        cmd += ["--compression-mode", "file", "--compression-format", rec["compression"]]
    qos = {t: p for t, p in (cfg.get("qos_overrides") or {}).items() if t in topics}
    if qos:
        cmd += ["--qos-profile-overrides-path", write_qos(qos)]
    return cmd + topics, root


def merge_cmd(cfg, directory):
    bags = sorted(p.parent for p in Path(directory).expanduser().glob("*/metadata.yaml"))
    if not bags:
        sys.exit(f"no bags (*/metadata.yaml) under {directory}")
    out = Path(directory).expanduser() / f"{cfg['session']['name']}_merged"
    fd, opts = tempfile.mkstemp(prefix="merge_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump({"output_bags": [{"uri": str(out),
                                         "storage_id": cfg["record"].get("storage", "mcap"),
                                         "all": True}]}, f)
    cmd = ["ros2", "bag", "convert"]
    for b in bags:
        cmd += ["-i", str(b)]
    return cmd + ["-o", opts]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--machine", help="record this machine's topics")
    mode.add_argument("--merge", metavar="DIR", help="merge every bag under DIR into one")
    ap.add_argument("--dry-run", action="store_true", help="print the command, do not run it")
    args = ap.parse_args()

    cfg = load(args.config)
    env = dict(os.environ, ROS_DOMAIN_ID=str(cfg["session"].get("ros_domain_id", 0)))
    if args.machine:
        cmd, root = record_cmd(cfg, args.machine)
    else:
        cmd, root = merge_cmd(cfg, args.merge), None

    print(f"ROS_DOMAIN_ID={env['ROS_DOMAIN_ID']} " + shlex.join(cmd))
    if args.dry_run:
        return
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()

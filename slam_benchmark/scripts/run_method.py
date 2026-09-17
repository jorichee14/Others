#!/usr/bin/env python3
"""Run one method on one agent, in its own container, and record what was run.

    python3 scripts/run_method.py --config configs/coop2.yaml \
        --method configs/methods/kiss_icp.yaml --stream mobile_1.ouster [--execute]

Without --execute it prints the command and writes nothing but the manifest:
look at the command first, every time. The refusals below happen BEFORE the
container starts, because each of them produces a run that completes and is
wrong, and forty minutes of replay is a slow way to find that out.

What the container must do, and all it must do:
  * subscribe to the topics in `--topics`, replayed by `ros2 bag play --clock`
  * write `<out>/trajectory.tum` — poses of ITS OWN SENSOR FRAME, in its own
    world frame, on the bag's clock. Do not apply any extrinsic and do not try
    to put it in `map`: the evaluator does both, from the dataset config, the
    same way for every method.
  * write `<out>/map.ply` if it produces a dense map, in that same world frame.
  * write `<out>/timing.json`: {frames, wall_s, cpu_s, peak_rss_mb}.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slambench.config import ConfigError, load_dataset, load_method   # noqa: E402


def preflight(cfg, mcfg, stream_key: str) -> tuple[dict, list[str]]:
    """Everything that has to be true before a container is worth starting."""
    problems: list[str] = []

    if stream_key not in mcfg.streams:
        raise ConfigError(
            f"method {mcfg.name!r} does not declare stream {stream_key!r}. "
            f"It declares: {mcfg.streams}")
    stream = cfg.stream(stream_key)
    agent = stream.get("agent", "?")

    if stream.get("modality") not in mcfg.modality:
        problems.append(f"stream {stream_key} is {stream.get('modality')!r} but the "
                        f"method consumes {mcfg.modality}")

    try:
        mcfg.sensor_extrinsic(cfg, stream_key)
    except ConfigError as e:
        problems.append(str(e))

    if mcfg.needs_imu:
        key = mcfg.raw.get("imu_by_agent", {}).get(agent)
        imu = cfg.raw.get("streams", {}).get(key, {}) if key else {}
        if imu.get("present") is None:
            problems.append(
                f"{mcfg.name} needs an IMU and streams.{key or '<unset>'}.present is "
                f"null — nothing has established that this recording carries one. "
                f"Resolve it with rosbag_to_opv2v/scripts/inspect_bag.py before "
                f"scheduling any inertial method.")
        elif not imu.get("present"):
            problems.append(f"{mcfg.name} needs an IMU and this recording has none. "
                            f"The whole inertial family is out on this sequence.")
        elif not imu.get("topic"):
            problems.append(f"streams.{key}.present is true but its topic is null")

    def _nulls(obj, prefix=""):
        """Null params, NESTED ONES INCLUDED.

        A flat scan misses `source_runs: {mobile_1: null}` — a dict is not None,
        so the top-level check passes it and the run starts on a value the
        config explicitly said could not be guessed. That is the exact failure
        rule 3 exists to prevent, wearing one extra level of indentation.
        """
        out = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                out += _nulls(v, f"{prefix}{k}." if not prefix else f"{prefix}{k}.")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                out += _nulls(v, f"{prefix}{i}.")
        elif obj is None:
            out.append(prefix.rstrip("."))
        return out

    nulls = _nulls(mcfg.raw.get("params") or {})
    if nulls:
        problems.append(f"params left null in {mcfg.path.name}: {nulls}. Each is a value "
                        f"the config says cannot be guessed; a default here is a silent "
                        f"wrong answer, not a convenience.")

    topics = [v for k, v in stream.items()
              if k.endswith("topic") and isinstance(v, str) and v.startswith("/")]
    if not topics:
        problems.append(f"stream {stream_key} declares no topic (is it exported at all?)")

    return {"stream_key": stream_key, "stream": stream, "agent": agent,
            "topics": topics}, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--stream", required=True,
                    help="a stream key declared by both the dataset and the method, "
                         "e.g. mobile_1.ouster")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--bag", default=None)
    ap.add_argument("--rate", type=float, default=1.0,
                    help="bag replay rate. Anything but 1.0 makes the runtime column "
                         "meaningless and a dropped-frame difference a tuning artefact; "
                         "the harness records it so a mixed table is visible.")
    ap.add_argument("--execute", action="store_true", help="actually start the container")
    ap.add_argument("--force", action="store_true",
                    help="run despite preflight problems. Recorded in the manifest and "
                         "printed on every table row that comes out of it.")
    args = ap.parse_args()

    cfg, mcfg = load_dataset(args.config), load_method(args.method)
    try:
        info, problems = preflight(cfg, mcfg, args.stream)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    out = Path(args.runs_root) / cfg.name / mcfg.name / args.stream
    bag = Path(args.bag or cfg.raw["dataset"]["bag"]).expanduser()

    if problems:
        print(f"preflight: {len(problems)} problem(s) for {mcfg.name} on {args.stream}",
              file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        if not args.force:
            print("\nrefusing to start. Fix the config, or pass --force and accept that "
                  "every number from this run carries the caveat.", file=sys.stderr)
            return 1

    cmd = [
        "docker", "run", "--rm", "--network", "none",
        "-v", f"{bag}:/bag:ro",
        "-v", f"{out.resolve()}:/out",
        "-e", f"SLAM_TOPICS={','.join(info['topics'])}",
        "-e", f"SLAM_RATE={args.rate}",
        "-e", f"SLAM_PARAMS={json.dumps(mcfg.raw.get('params') or {})}",
        mcfg.raw.get("image", f"slambench/{mcfg.name}:humble"),
    ]

    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": cfg.name, "method": mcfg.name, "agent": info["agent"],
        "stream": info["stream_key"], "topics": info["topics"],
        "image": mcfg.raw.get("image"), "params": mcfg.raw.get("params"),
        "bag": str(bag), "replay_rate": args.rate,
        "command": " ".join(shlex.quote(c) for c in cmd),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_problems": problems,
        "forced": bool(problems and args.force),
        "executed": bool(args.execute),
        "host": os.uname().nodename,
    }

    print(manifest["command"])
    if args.execute:
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        manifest["wall_s"] = time.time() - t0
        manifest["returncode"] = rc
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (out / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest -> {out / 'run.json'}")
    if args.execute and manifest.get("returncode"):
        print(f"container exited {manifest['returncode']}", file=sys.stderr)
        return manifest["returncode"]
    if not args.execute:
        print("(dry run: pass --execute to start the container)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

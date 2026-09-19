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
        elif not (imu.get("extrinsic_from_camera") or {}):
            # MEASURED 2026-09-19. The bag's tf_static carries 7 edges in 2
            # disconnected trees -- the ZED chain and the Ouster chain -- and
            # nothing else, so no inertial frame is recoverable from the
            # recording. Without the extrinsic, RTAB-Map logs "Dropping imu
            # data!" and then CONTINUES, emitting a trajectory: the row does not
            # fail, it quietly becomes its own IMU-free control and lands in the
            # table as evidence the IMU did not help.
            problems.append(
                f"{mcfg.name} needs an IMU and streams.{key}.extrinsic_from_camera is "
                f"null. The bag's tf_static does not carry the inertial frames, so "
                f"nothing supplies camera<-IMU at run time and the method will drop "
                f"every sample WITHOUT failing. Recover it from the device "
                f"(rs-enumerate-devices -c, or the ZED SDK's camera_imu_transform) "
                f"as I13 was for mobile_1.")

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

    # The container is replayed with `ros2 bag play --topics <these>`, so this
    # list is not documentation: it is the entire universe the method can see.
    # Two things have to be added to it or the run completes on starved input,
    # which is the failure mode that looks most like a result.
    #
    #   * the IMU, for any method that declared it needs one. The stream block
    #     describes a camera; the inertial topic lives under its own stream key
    #     and was checked above but never played.
    #   * /tf_static, which carries the sensor extrinsics the method's own
    #     front-end resolves internally (RTAB-Map will not start RGB-D odometry
    #     without camera<-base).
    #
    # /tf is deliberately NOT added. The bag's dynamic tf carries map->sensor
    # from the recording pipeline -- that is a pose estimate, and replaying it
    # into a method that publishes its own odom->base is at best a tf conflict
    # and at worst a method reading the answer off its input.
    if mcfg.needs_imu:
        key = mcfg.raw.get("imu_by_agent", {}).get(agent)
        imu_topic = (cfg.raw.get("streams", {}).get(key) or {}).get("topic") if key else None
        if imu_topic:
            topics.append(imu_topic)

    topics.append("/tf_static")

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

    stream = info["stream"]
    # Static transforms the container has to publish because the bag does not
    # carry them. Each is a number already in the dataset config (rule 4), each
    # is a SENSOR EXTRINSIC the front-end needs in order to fuse at all rather
    # than a transform of the output (rule 6), and each travels in run.json.
    static_tf: list[list[str]] = []
    color_frame, sensor_frame = stream.get("color_frame"), stream.get("sensor_frame")
    if color_frame and sensor_frame and color_frame != sensor_frame:
        # colour optical <- depth optical. An RGB-D front-end told to work in the
        # depth frame still has to place the colour camera, and on this bag the
        # RealSense frames are absent from tf_static entirely.
        from slambench import se3
        from slambench.config import extrinsic_matrix
        T = extrinsic_matrix(stream["reference_frame_from_sensor"],
                             f"{args.stream}.reference_frame_from_sensor")
        t, q = se3.pose_to_quat(T)
        static_tf.append([color_frame, sensor_frame]
                         + [repr(float(v)) for v in list(t) + list(q)])
    cmd = [
        "docker", "run", "--rm", "--network", "none",
        "-v", f"{bag}:/bag:ro",
        "-v", f"{out.resolve()}:/out",
        "-e", f"SLAM_TOPICS={','.join(info['topics'])}",
        "-e", f"SLAM_RATE={args.rate}",
        "-e", f"SLAM_PARAMS={json.dumps(mcfg.raw.get('params') or {})}",
        # The stream's own description, so the image never has to know which
        # dataset it is looking at. Rule 6 in the other direction: the container
        # does not transform poses, but it does have to PARSE its input, and the
        # units and range of a depth image are properties of the stream.
        "-e", f"SLAM_MODALITY={stream.get('modality', '')}",
    ]
    for env, key in (("SLAM_DEPTH_TOPIC", "depth_topic"),
                     ("SLAM_DEPTH_INFO_TOPIC", "depth_info_topic"),
                     ("SLAM_COLOR_TOPIC", "color_topic"),
                     ("SLAM_COLOR_INFO_TOPIC", "color_info_topic"),
                     ("SLAM_CLOUD_TOPIC", "topic"),
                     ("SLAM_DEPTH_SCALE", "depth_scale")):
        if stream.get(key) is not None:
            cmd += ["-e", f"{env}={stream[key]}"]
    rng = stream.get("range_m")
    if rng:
        cmd += ["-e", f"SLAM_RANGE_MIN={rng[0]}", "-e", f"SLAM_RANGE_MAX={rng[1]}"]
    if mcfg.needs_imu:
        imu_key = mcfg.raw.get("imu_by_agent", {}).get(info["agent"])
        imu_stream = cfg.raw.get("streams", {}).get(imu_key) or {}
        if imu_stream.get("topic"):
            cmd += ["-e", f"SLAM_IMU_TOPIC={imu_stream['topic']}"]
        # THE CAMERA<-IMU EDGE, BECAUSE THE BAG DOES NOT CARRY IT.
        # Measured 2026-09-19: `zed_imu_link` is absent from the replayed
        # tf_static, so RTAB-Map dropped every IMU sample and never initialised
        # ("A valid TF between ... is required"). The transform is not lost --
        # it is I13, resolved from the wrapper's own published TF and recorded
        # in configs/coop2.yaml -- so the container publishes that one edge from
        # the config instead of doing without the IMU.
        #
        # This is not the container transforming poses (rule 6). It is a SENSOR
        # EXTRINSIC the front-end needs in order to fuse at all, it is the
        # config's number rather than a derived one (rule 4), and it travels in
        # run.json so any result can be checked against it.
        ext = imu_stream.get("extrinsic_from_camera")
        if isinstance(ext, dict) and ext.get("xyz") and ext.get("quat_xyzw"):
            static_tf.append([str(ext["parent"]), str(ext["child"])]
                             + [repr(float(v)) for v in
                                list(ext["xyz"]) + list(ext["quat_xyzw"])])
    # `gpu: required` in the method config is a hardware claim, so it belongs on
    # the command rather than in the image: an image that always asks for a GPU
    # cannot be smoke-tested on a machine without one.
    # Exact vs approximate synchronisation is a property of the RECORDING, not a
    # tuning knob: measured per stream by scripts/bag_probe.py and declared in
    # the dataset config. Getting it wrong is not a shrug -- an approximate
    # matcher fed byte-identical stamps paired mobile_1's frames a full period
    # apart on the first run.
    if stream.get("stamps_exact") is not None:
        cmd += ["-e", f"SLAM_SYNC={'exact' if stream['stamps_exact'] else 'approx'}"]
    if static_tf:
        cmd += ["-e", "SLAM_STATIC_TF=" + "\n".join(" ".join(e) for e in static_tf)]
    if mcfg.raw.get("gpu") == "required":
        cmd += ["--gpus", "all"]
    cmd.append(mcfg.raw.get("image", f"slambench/{mcfg.name}:humble"))

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

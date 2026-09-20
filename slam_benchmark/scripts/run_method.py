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
        elif imu.get("orientation") is None:
            problems.append(
                f"streams.{key}.orientation is null. RTAB-Map and every other "
                f"gravity-constrained method reads gravity off the message's "
                f"`orientation` quaternion and computes none itself, so a raw "
                f"gyro+accel stream is rejected sample by sample WITHOUT "
                f"failing the run. Establish whether this topic carries an "
                f"orientation before scheduling an inertial row on it.")
        elif imu.get("orientation") == "absent" and not mcfg.raw.get(
                "imu_orientation_filter"):
            problems.append(
                f"streams.{key}.orientation is absent and {mcfg.name} declares "
                f"no imu_orientation_filter. The run would start, drop every "
                f"IMU sample, and land in the table as an inertial row with no "
                f"inertial data in it.")
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
    ap.add_argument("--rate", type=float, default=None,
                    help="bag replay rate. Defaults to the method config's `replay_rate`, "
                         "else 1.0. Anything but 1.0 makes the runtime column meaningless; "
                         "the harness records it so a mixed table is visible.")
    ap.add_argument("--execute", action="store_true", help="actually start the container")
    ap.add_argument("--dev", action="store_true",
                    help="bind-mount docker/common/*.py and entrypoint.sh over the image's "
                         "copies, so a fix to the harness scripts runs without a 7-minute "
                         "rebuild. Recorded in run.json: the image tag no longer describes "
                         "what ran, so a --dev run is for getting the harness right, not "
                         "for a table.")
    ap.add_argument("--shell", action="store_true",
                    help="drop into bash inside the container instead of running the "
                         "entrypoint, with EXACTLY the same mounts and environment. For "
                         "when a run misbehaves and the question is what the method sees, "
                         "not what the harness thinks it sees.")
    ap.add_argument("--force", action="store_true",
                    help="run despite preflight problems. Recorded in the manifest and "
                         "printed on every table row that comes out of it.")
    args = ap.parse_args()

    cfg, mcfg = load_dataset(args.config), load_method(args.method)
    if args.rate is None:
        # Per stream first: the rate is a throughput statement about THIS
        # machine on THIS stream's frame rate, and one method sees 15 Hz on the
        # ZED and 27.5 Hz on the RealSense.
        by_stream = mcfg.raw.get("replay_rate_by_stream", {}) or {}
        args.rate = float(by_stream.get(args.stream, mcfg.raw.get("replay_rate", 1.0)))
    try:
        info, problems = preflight(cfg, mcfg, args.stream)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # One directory PER RUN, never overwritten. Two identical runs of RTAB-Map
    # scored 365 mm and 451 mm; the second erased the first, so the odometry
    # files that would have said whether the front-end or the graph moved were
    # gone. A method whose answer varies between runs needs every run kept, and
    # a spread reported beside the number (rule 8 in the other direction: a
    # good number is not a result either, without its variance).
    cell = Path(args.runs_root) / cfg.name / mcfg.name / args.stream
    out = cell / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
        color_from_depth = " ".join(repr(float(v)) for v in list(t) + list(q))
    else:
        color_from_depth = None
    # A method that consumes colour+depth as ONE image pair (RTAB-Map) needs the
    # depth registered into the colour camera. The RealSense in this bag is not,
    # and rgbd_odometry aborts on the size check (1280x720 vs 640x480). The
    # entrypoint registers it when the method declares the need and the frames
    # differ; the method's poses are then in the colour frame, which the method
    # config must declare with `extrinsic_by_stream` for that stream.
    register_depth = bool(mcfg.raw.get("depth_registered_to_color") == "required"
                          and color_from_depth is not None)
    if register_depth and args.stream not in mcfg.raw.get("extrinsic_by_stream", {}):
        print(f"{mcfg.name} will work in {color_frame} on {args.stream} (registered depth) "
              f"but declares no extrinsic_by_stream.{args.stream}; the evaluator would apply "
              f"the depth-frame extrinsic to colour-frame poses. Declare it.", file=sys.stderr)
        return 1
    cmd = [
        "docker", "run", "--rm", "--network", "none",
        # Docker gives a container 64 MB of /dev/shm. With --network none the DDS
        # layer has only shared memory and loopback, and three 1280x720 image
        # topics at 15 Hz exceed a 64 MB segment many times over; the failure is
        # silent -- messages are dropped, nothing is logged, and the estimator
        # simply never sees ~a quarter of the frames. Not a tuning knob: the
        # size of the pipe the bag flows through.
        "--shm-size", "2g",
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
    if register_depth:
        cmd += ["-e", "SLAM_DEPTH_REGISTER=1", "-e", f"SLAM_COLOR_FRAME={color_frame}",
                "-e", f"SLAM_COLOR_FROM_DEPTH={color_from_depth}"]
    if mcfg.needs_imu:
        imu_key = mcfg.raw.get("imu_by_agent", {}).get(info["agent"])
        imu_stream = cfg.raw.get("streams", {}).get(imu_key) or {}
        if imu_stream.get("topic"):
            cmd += ["-e", f"SLAM_IMU_TOPIC={imu_stream['topic']}"]
        # Inserted ONLY where the dataset says the stream carries no
        # orientation, so the filter cannot quietly appear on an agent whose
        # vendor already fused one and make the two rows different experiments.
        filt = mcfg.raw.get("imu_orientation_filter")
        if filt and imu_stream.get("orientation") == "absent":
            cmd += ["-e", f"SLAM_IMU_ORIENTATION_FILTER={filt}"]
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
    if args.dev:
        common = ROOT / "docker" / "common"
        mounts = sorted(common.glob("*.py")) + [common / "entrypoint.sh"]
        # The method's own launch script lives under docker/<image>/, named
        # after the image tag: slambench/rtabmap:humble -> docker/rtabmap/.
        image = mcfg.raw.get("image", f"slambench/{mcfg.name}:humble")
        idir = ROOT / "docker" / image.split("/", 1)[-1].split(":")[0]
        mounts += sorted(idir.glob("*.sh")) + sorted(idir.glob("*.py"))
        for f in mounts:
            cmd[2:2] = ["-v", f"{f}:/opt/slambench/{f.name}:ro"]
    if args.shell:
        # -it for a terminal; --entrypoint bash replaces the run script. Everything
        # else is byte-identical to the real run, which is the entire point: a
        # debugging shell that differs from the run teaches you about the shell.
        cmd[2:2] = ["-it", "--entrypoint", "bash"]
    cmd.append(mcfg.raw.get("image", f"slambench/{mcfg.name}:humble"))

    out.mkdir(parents=True, exist_ok=True)
    latest = cell / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out.name)
    manifest = {
        "dataset": cfg.name, "method": mcfg.name, "agent": info["agent"],
        "stream": info["stream_key"], "topics": info["topics"],
        "image": mcfg.raw.get("image"), "params": mcfg.raw.get("params"),
        "bag": str(bag), "replay_rate": args.rate,
        "command": " ".join(shlex.quote(c) for c in cmd),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_problems": problems,
        "dev_scripts_mounted": bool(args.dev),
        "depth_registered_in_container": register_depth,
        "pose_frame": color_frame if register_depth else stream.get("sensor_frame"),
        "forced": bool(problems and args.force),
        "executed": bool(args.execute),
        "host": os.uname().nodename,
    }

    print(manifest["command"])
    if args.shell:
        print("\ninside the container, the run is these four steps:\n"
              "  source /opt/ros/humble/setup.bash\n"
              "  python3 /opt/slambench/record_tum.py --pose-topic \"$SLAM_POSE_TOPIC\" "
              "--path-topic \"$SLAM_PATH_TOPIC\" --out /out --ros-args -p use_sim_time:=true &\n"
              "  /opt/slambench/launch.sh &\n"
              "  ros2 bag play /bag --clock --topics ${SLAM_TOPICS//,/ }\n",
              file=sys.stderr)
        return subprocess.run(cmd).returncode

    if args.execute:
        t0 = time.time()
        # A Ctrl-C used to propagate straight out of here, so run.json was never
        # written and the directory kept a STALE manifest from an earlier run --
        # a run directory describing a different run is worse than an empty one.
        try:
            rc = subprocess.run(cmd).returncode
        except KeyboardInterrupt:
            rc, manifest["interrupted"] = 130, True
            print("\ninterrupted — the container was asked to flush; "
                  "check for an INTERRUPTED marker in the run directory",
                  file=sys.stderr)
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

#!/usr/bin/env python3
"""KISS-ICP over a bag, offline: every frame, in order, however long each takes.

Why not the ROS node. Four runs of kiss_icp_node on mobile_1.zed_rgbd processed
155, 325, 566 and 320 of 2288 frames. Its input subscription is best-effort
(SensorDataQoS, hard-coded), and single frames took 5-65 s -- one frame per
half-second on average is not the shape of it; a stall of a minute on one
frame is. No replay rate covers a minute. This benchmark is offline, so the
frames are read from the bag and handed to the library one at a time through
its Python API, the same C++ core the node wraps, with the same parameters.
Nothing is dropped and nothing is timed against a clock. The time each frame
took is recorded instead, because that stall is a finding, not a nuisance.

Contract (docker/README.md): /out/trajectory.tum on the bag's clock, poses of
the sensor frame the method was given (the cloud's own frame: no transform
here), /out/map.ply in that same world frame, /out/timing.json.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/opt/slambench")

# SLAM_PARAMS key -> (section, field) in kiss_icp.config.KISSConfig. Anything
# not listed is refused: a parameter the library would silently ignore is a
# manifest that lies (rule 7).
PARAM_MAP = {
    "max_range": ("data", "max_range"),
    "min_range": ("data", "min_range"),
    "deskew": ("data", "deskew"),
    "voxel_size": ("mapping", "voxel_size"),
    "max_points_per_voxel": ("mapping", "max_points_per_voxel"),
    "max_num_iterations": ("registration", "max_num_iterations"),
    "convergence_criterion": ("registration", "convergence_criterion"),
    "initial_threshold": ("adaptive_threshold", "initial_threshold"),
    "fixed_threshold": ("adaptive_threshold", "fixed_threshold"),
    "min_motion_th": ("adaptive_threshold", "min_motion_th"),
}


def apply_params(config, params: dict, modality: str) -> dict:
    """Set every SLAM_PARAMS entry on the KISSConfig; refuse unknown keys.

    Returns the overrides applied on top of the declared params (structural,
    not tuning): a depth image is one exposure with no per-point time, so on
    rgbd `deskew` is forced off and said so.
    """
    unknown = sorted(set(params) - set(PARAM_MAP))
    if unknown:
        raise SystemExit(f"SLAM_PARAMS keys the library does not take: {unknown}. "
                         f"Known: {sorted(PARAM_MAP)}")
    overrides = {}
    for k, v in params.items():
        sec, field = PARAM_MAP[k]
        setattr(getattr(config, sec), field, v)
    if modality == "rgbd" and config.data.deskew:
        print("OVERRIDE deskew -> false (depth stream: no per-point time field)")
        config.data.deskew = False
        overrides["deskew"] = False
    if config.mapping.voxel_size is None:          # the library's own rule
        config.mapping.voxel_size = float(config.data.max_range / 100.0)
    return overrides


def normalise_stamps(t: np.ndarray) -> np.ndarray:
    """Per-point times -> [0, 1], which is what the deskew expects
    (Deskew.cpp: mid_pose_timestamp = 0.5). A scan with one time is all 0.5."""
    t = np.asarray(t, dtype=np.float64)
    if t.size == 0:
        return t
    lo, hi = float(t.min()), float(t.max())
    if hi <= lo:
        return np.full(t.shape, 0.5)
    return (t - lo) / (hi - lo)


def frame_stats(seconds: list[float]) -> dict:
    """The per-frame wall time, summarised beside the count: the node's stalls
    were the finding, and a run that only reports its total hides them."""
    s = np.asarray(seconds, dtype=np.float64)
    if s.size == 0:
        return {"n": 0}
    return {"n": int(s.size), "median": float(np.median(s)),
            "p90": float(np.percentile(s, 90)), "max": float(s.max()),
            "over_1s": int((s > 1.0).sum()), "over_5s": int((s > 5.0).sum()),
            "total": float(s.sum())}


def quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (x, y, z, w), w >= 0. Shepperd's
    method: pick the largest diagonal term so no division is near zero."""
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0]; m10, m11, m12 = R[1]; m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    q = np.array([x, y, z, w])
    if w < 0:
        q = -q
    return q / np.linalg.norm(q)


def pose_row(stamp: float, T: np.ndarray) -> str:
    x, y, z = np.asarray(T)[:3, 3]
    qx, qy, qz, qw = quat_xyzw(np.asarray(T)[:3, :3])
    return f"{stamp:.9f} {x:.6f} {y:.6f} {z:.6f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"


def main() -> int:                                           # pragma: no cover
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from kiss_icp.config import KISSConfig
    from kiss_icp.kiss_icp import KissICP
    from depth_to_cloud import ENCODINGS, check_scale, deproject

    out = Path(os.environ.get("OUT", "/out"))
    out.mkdir(parents=True, exist_ok=True)
    bag = os.environ.get("BAG", "/bag")
    modality = os.environ["SLAM_MODALITY"]
    params = json.loads(os.environ.get("SLAM_PARAMS") or "{}")

    config = KISSConfig()
    overrides = apply_params(config, params, modality)
    print(f"kiss-icp offline: modality={modality} params={params} overrides={overrides}")
    print(f"  data={config.data.model_dump()} mapping={config.mapping.model_dump()}\n"
          f"  registration={config.registration.model_dump()} "
          f"threshold={config.adaptive_threshold.model_dump()}")

    if modality == "rgbd":
        depth_topic, info_topic = os.environ["SLAM_DEPTH_TOPIC"], os.environ["SLAM_DEPTH_INFO_TOPIC"]
        depth_scale = float(os.environ["SLAM_DEPTH_SCALE"])
        r_min = float(os.environ.get("SLAM_RANGE_MIN", "0.0"))
        r_max = float(os.environ.get("SLAM_RANGE_MAX", "inf"))
        stride = int(os.environ.get("DEPTH_STRIDE", "2"))   # the bridge's declared stride
        topics = [depth_topic, info_topic]
    elif modality == "lidar":
        cloud_topic = os.environ["SLAM_CLOUD_TOPIC"]
        topics = [cloud_topic]
    else:
        raise SystemExit(f"modality {modality!r}: this runner takes rgbd or lidar")

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [t for t in topics if t not in types]
    if missing:
        raise SystemExit(f"not in the bag: {missing}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    cls = {t: get_message(types[t]) for t in topics}

    odometry = KissICP(config)
    K = None
    stamps: list[float] = []
    per_frame: list[float] = []
    traj = out / "trajectory.tum"
    fh = traj.open("w")
    fh.write("# timestamp tx ty tz qx qy qz qw\n")
    interrupted = {"flag": False}

    def on_signal(signum, _frame):
        interrupted["flag"] = True
        (out / "INTERRUPTED").write_text(f"signal {signum} after {len(stamps)} frames\n")

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    t_start = time.time()
    n_in = 0
    while reader.has_next() and not interrupted["flag"]:
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, cls[topic])
        if modality == "rgbd":
            if topic == info_topic:
                if K is None:
                    K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
                    print(f"intrinsics fx,fy,cx,cy = {K}")
                continue
            n_in += 1
            if K is None:
                print(f"depth frame before camera_info; skipped ({n_in})")
                continue
            scale = check_scale(msg.encoding, depth_scale)
            dtype, _ = ENCODINGS[msg.encoding]
            raw = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
            pts = deproject(raw.astype(np.float32) * scale, *K,
                            stride=stride, r_min=r_min, r_max=r_max).astype(np.float64)
            ts = np.full(len(pts), 0.5)
        else:
            from sensor_msgs_py import point_cloud2
            n_in += 1
            names = [f.name for f in msg.fields]
            tname = next((n for n in ("t", "timestamp", "time") if n in names), None)
            fields = ["x", "y", "z"] + ([tname] if tname else [])
            arr = point_cloud2.read_points(msg, field_names=fields, skip_nans=True)
            pts = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float64)
            ts = normalise_stamps(arr[tname]) if tname else np.full(len(pts), 0.5)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        t0 = time.time()
        odometry.register_frame(pts, ts)
        dt = time.time() - t0
        per_frame.append(dt)
        stamps.append(stamp)
        fh.write(pose_row(stamp, odometry.last_pose))
        if len(stamps) % 50 == 0:
            fh.flush()
        if len(stamps) % 100 == 0 or dt > 5.0:
            print(f"{len(stamps):5d} frames  last {dt * 1e3:7.1f} ms  {len(pts):6d} pts  "
                  f"median {np.median(per_frame) * 1e3:6.1f} ms  max {max(per_frame):6.2f} s",
                  flush=True)
    fh.close()

    try:
        from plyfile import PlyData, PlyElement
        cloud = np.asarray(odometry.local_map.point_cloud(), dtype=np.float32)
        arr = np.empty(len(cloud), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        arr["x"], arr["y"], arr["z"] = cloud[:, 0], cloud[:, 1], cloud[:, 2]
        el = PlyElement.describe(arr, "vertex")
        PlyData([el]).write(str(out / "map.ply"))
        map_points = int(len(cloud))
    except Exception as e:                                   # noqa: BLE001
        print(f"map.ply not written: {e}", file=sys.stderr)
        map_points = None

    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF)
    json.dump({
        "frames": len(stamps), "input_frames": n_in,
        "wall_s": time.time() - t_start, "cpu_s": ru.ru_utime + ru.ru_stime,
        "peak_rss_mb": ru.ru_maxrss / 1024.0,
        "frame_s": frame_stats(per_frame),
        "map_points": map_points,
        "depth_stride": stride if modality == "rgbd" else None,
        "trajectory_source": "offline",
        "overrides": overrides,
        "interrupted": interrupted["flag"],
    }, (out / "timing.json").open("w"), indent=2)
    print(f"{len(stamps)} of {n_in} frames -> {traj}")
    if interrupted["flag"]:
        return 130
    return 0 if stamps else 3


if __name__ == "__main__":                                   # pragma: no cover
    sys.exit(main())

#!/usr/bin/env python3
"""Ouster packets bag -> points bag, offline. Replaces the replay step.

    python3 scripts/decode_ouster.py <packets_bag> <out_bag_dir>
        [--packets-ns /ouster] [--out-ns /mobile_1/ouster]
        [--driver-min-range 1.30] [--driver-max-range 1000]
        [--frame lidar|sensor] [--no-imu] [--limit N]

Instead of

    ros2 launch ouster_ros replay.launch.xml bag_file:=<packets_bag> ...   (real time)
    ros2 bag record <out-ns>/points <out-ns>/imu ...

this reads <packets-ns>/lidar_packets, /imu_packets and /metadata from the
ORIGINAL recording and writes a rosbag2 folder (MCAP + metadata.yaml) with

    <out-ns>/points    sensor_msgs/PointCloud2, organized H x W, ouster_ros
                       default point layout: x y z intensity t reflectivity
                       ring ambient range; filtered / no-return points NaN
    <out-ns>/imu       sensor_msgs/Imu, m/s^2 and rad/s
    <out-ns>/metadata  std_msgs/String, the sensor metadata JSON

stamped in SENSOR time, exactly what a replay without timestamp_mode gives.
The next step is unchanged, in a ROS terminal:

    python3 retime_bag.py <packets_bag> <out_bag_dir> <retimed_bag_dir>

Log times are the original receive times, so the bag already sits on the
capture timeline and retime_bag's replay->capture shift comes out ~0.
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", FutureWarning)

from scoop import bag, ouster                                       # noqa: E402

# ouster_ros "original" point type (PointOS): PCL point4d padding after z,
# 16-byte aligned to 48. PointField datatypes: 7 float32, 6 uint32, 4 uint16.
POINT_FIELDS = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 16, 7),
                ("t", 20, 6), ("reflectivity", 24, 4), ("ring", 26, 4),
                ("ambient", 28, 4), ("range", 32, 6)]
POINT_DTYPE = np.dtype({"names": [f[0] for f in POINT_FIELDS],
                        "formats": ["<f4", "<f4", "<f4", "<f4", "<u4", "<u2", "<u2",
                                    "<u2", "<u4"],
                        "offsets": [f[1] for f in POINT_FIELDS], "itemsize": 48})
IMU_ACC_COV = 0.01                  # ouster_ros's fixed IMU covariances
IMU_GYR_COV = 6e-4


def typestore():
    from rosbags.typesys import Stores, get_typestore
    return get_typestore(Stores.ROS2_HUMBLE)


def cloud_array(f: ouster.OusterFrame) -> np.ndarray:
    H, W = f.valid.shape
    a = np.zeros(H * W, POINT_DTYPE)
    xyz = f.xyz.reshape(-1, 3)
    a["x"], a["y"], a["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    a["t"] = f.t_ns.reshape(-1)
    a["range"] = f.fields["range"].reshape(-1)
    if "signal" in f.fields:
        a["intensity"] = f.fields["signal"].reshape(-1)
    if "reflectivity" in f.fields:
        a["reflectivity"] = f.fields["reflectivity"].reshape(-1)
    if "near_ir" in f.fields:
        a["ambient"] = f.fields["near_ir"].reshape(-1)
    a["ring"] = np.repeat(np.arange(H, dtype=np.uint16), W)
    return a


class BagWriter:
    """Minimal rosbag2 (MCAP) writer: schemas from rosbags, metadata.yaml the
    way `ros2 bag record` writes it on Humble, so retime_bag.py, `ros2 bag`
    and GLIM take it as a recorded bag."""

    def __init__(self, out_dir: Path):
        from mcap.writer import CompressionType, Writer
        self.ts = typestore()
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=False)
        self.name = f"{self.dir.name}_0.mcap"
        self.fh = open(self.dir / self.name, "wb")
        self.w = Writer(self.fh, compression=CompressionType.NONE)
        self.w.start(profile="ros2", library="scoop decode_ouster.py")
        self.channels, self.types, self.counts = {}, {}, {}
        self.t_min = self.t_max = None
        self.n = 0

    def channel(self, topic, msgtype):
        if topic not in self.channels:
            text, _ = self.ts.generate_msgdef(msgtype, ros_version=2)
            sid = self.w.register_schema(msgtype, "ros2msg", text.encode())
            self.channels[topic] = self.w.register_channel(topic, "cdr", sid)
            self.types[topic] = msgtype
            self.counts[topic] = 0
        return self.channels[topic]

    def write(self, topic, msg, log_ns):
        ch = self.channel(topic, msg.__msgtype__)
        data = bytes(self.ts.serialize_cdr(msg, msg.__msgtype__))
        self.w.add_message(ch, log_time=int(log_ns), publish_time=int(log_ns),
                           data=data, sequence=self.counts[topic])
        self.counts[topic] += 1
        self.n += 1
        self.t_min = log_ns if self.t_min is None else min(self.t_min, log_ns)
        self.t_max = log_ns if self.t_max is None else max(self.t_max, log_ns)

    def close(self):
        self.w.finish()
        self.fh.close()
        t0, t1 = int(self.t_min or 0), int(self.t_max or 0)
        meta = {"rosbag2_bagfile_information": {
            "version": 5,
            "storage_identifier": "mcap",
            "duration": {"nanoseconds": t1 - t0},
            "starting_time": {"nanoseconds_since_epoch": t0},
            "message_count": self.n,
            "topics_with_message_count": [
                {"topic_metadata": {"name": t, "type": self.types[t],
                                    "serialization_format": "cdr",
                                    "offered_qos_profiles": ""},
                 "message_count": self.counts[t]} for t in self.channels],
            "compression_format": "",
            "compression_mode": "",
            "relative_file_paths": [self.name],
            "files": [{"path": self.name,
                       "starting_time": {"nanoseconds_since_epoch": t0},
                       "duration": {"nanoseconds": t1 - t0},
                       "message_count": self.n}]}}
        with open(self.dir / "metadata.yaml", "w") as fh:
            yaml.safe_dump(meta, fh, sort_keys=False)


def decode(packets_bag, out_dir, packets_ns="/ouster", out_ns="/mobile_1/ouster",
           driver_min_range=0.0, driver_max_range=1000.0, frame="lidar",
           imu=True, limit=0, metadata=None, log=print):
    reader = bag.open_bag(packets_bag)
    src = metadata or ouster.read_metadata(reader, packets_ns)
    info = ouster.sensor_info(src)
    meta_json = src if isinstance(src, str) else info.to_json_string()
    frame_id = {"lidar": "os_lidar", "sensor": "os_sensor"}[frame]
    w = BagWriter(out_dir)
    T = w.ts.types
    Time = T["builtin_interfaces/msg/Time"]
    Header = T["std_msgs/msg/Header"]
    PF = T["sensor_msgs/msg/PointField"]
    fields = [PF(name=n, offset=o, datatype=d, count=1) for n, o, d in POINT_FIELDS]

    def header(ns, fid):
        return Header(stamp=Time(sec=int(ns // 1_000_000_000),
                                 nanosec=int(ns % 1_000_000_000)), frame_id=fid)

    st = {}
    n_frames = 0
    t_first = None
    wall = time.perf_counter()
    try:
        for ev in ouster.iter_ouster(reader, packets_ns, clock=None, frame=frame,
                                     driver_min_range=driver_min_range,
                                     driver_max_range=driver_max_range, imu=imu,
                                     fields=True, metadata=info, stats=st):
            if t_first is None:
                t_first = ev.log_ns
                w.write(f"{out_ns}/metadata", T["std_msgs/msg/String"](data=meta_json),
                        ev.log_ns)
            if isinstance(ev, ouster.ImuSample):
                q = T["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0)
                V = T["geometry_msgs/msg/Vector3"]
                cov = lambda v: np.diag([v, v, v]).reshape(-1).astype(np.float64)  # noqa: E731
                ocov = np.zeros(9)
                ocov[0] = -1.0
                w.write(f"{out_ns}/imu", T["sensor_msgs/msg/Imu"](
                    header=header(ev.stamp_ns, "os_imu"), orientation=q,
                    orientation_covariance=ocov,
                    angular_velocity=V(x=float(ev.gyro[0]), y=float(ev.gyro[1]), z=float(ev.gyro[2])),
                    angular_velocity_covariance=cov(IMU_GYR_COV),
                    linear_acceleration=V(x=float(ev.accel[0]), y=float(ev.accel[1]), z=float(ev.accel[2])),
                    linear_acceleration_covariance=cov(IMU_ACC_COV)), ev.log_ns)
                continue
            H, W = ev.valid.shape
            a = cloud_array(ev)
            w.write(f"{out_ns}/points", T["sensor_msgs/msg/PointCloud2"](
                header=header(ev.stamp_ns, frame_id), height=H, width=W, fields=fields,
                is_bigendian=False, point_step=48, row_step=48 * W,
                data=np.frombuffer(a.tobytes(), np.uint8), is_dense=False), ev.log_ns)
            n_frames += 1
            if n_frames % 500 == 0:
                dt = time.perf_counter() - wall
                rec = (ev.log_ns - t_first) * 1e-9
                log(f"    {n_frames} scans, {rec:.0f} s of recording in {dt:.0f} s "
                    f"({rec / max(dt, 1e-9):.1f}x real time)")
            if limit and n_frames >= limit:
                break
    finally:
        w.close()
    dt = time.perf_counter() - wall
    rec = ((w.t_max or 0) - (w.t_min or 0)) * 1e-9
    log(f"wrote {w.dir}: {w.counts}  ({rec:.1f} s of recording in {dt:.1f} s, "
        f"{rec / max(dt, 1e-9):.1f}x real time)")
    return w.dir, w.counts, st


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("packets_bag")
    ap.add_argument("out_bag_dir")
    ap.add_argument("--packets-ns", default="/ouster")
    ap.add_argument("--out-ns", default="/mobile_1/ouster")
    ap.add_argument("--driver-min-range", type=float, default=1.30)
    ap.add_argument("--driver-max-range", type=float, default=1000.0)
    ap.add_argument("--frame", choices=("lidar", "sensor"), default="lidar")
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after N scans (for a test)")
    a = ap.parse_args()
    if os.path.exists(a.out_bag_dir):
        sys.exit(f"{a.out_bag_dir} exists; pick a new output folder")
    decode(a.packets_bag, a.out_bag_dir, a.packets_ns, a.out_ns, a.driver_min_range,
           a.driver_max_range, a.frame, not a.no_imu, a.limit)
    print(f"\nnext (ROS terminal):  python3 retime_bag.py {a.packets_bag} "
          f"{a.out_bag_dir} <retimed_out_dir> {a.packets_ns} {a.out_ns}")


if __name__ == "__main__":
    main()

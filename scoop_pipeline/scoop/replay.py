# -*- coding: utf-8 -*-
"""
Offline replacement for the ouster_ros replay: packets bag -> points bag.

The replay launch did two things: ``ros2 bag play --remap`` renamed topics,
and the ouster_ros driver turned lidar/IMU packets into point clouds and IMU
messages, which ``ros2 bag record`` saved. :func:`decode_ouster_bag` does both
in one pass over the original recording, without waiting for real time:

    <out_ns>/points    PointCloud2, ouster_ros layout (scoop.ouster.ros_points)
    <out_ns>/imu       Imu, m/s^2 and rad/s
    <out_ns>/metadata  String, the sensor metadata JSON
    + every remapped topic, copied byte for byte under its new name

Stamps are SENSOR time (a replay without timestamp_mode); log times are the
original receive times. ``retime_bag.py`` runs on the result unchanged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

from . import bag, ouster, rosmsg
from .bagwrite import BagWriter

__all__ = ["parse_remaps", "decode_ouster_bag", "ReplayResult"]


def parse_remaps(items: Optional[Iterable[str]]) -> Dict[str, str]:
    """``["/a:=/b", ...]`` -> ``{"/a": "/b"}``, the ``ros2 bag play --remap``
    syntax. ``/a:=/a`` copies a topic under its own name."""
    out: Dict[str, str] = {}
    for it in items or ():
        if ":=" not in it:
            raise ValueError(f"remap {it!r}: expected SRC:=DST")
        src, dst = (x.strip() for x in it.split(":=", 1))
        if not src.startswith("/") or not dst.startswith("/"):
            raise ValueError(f"remap {it!r}: topics must start with /")
        if src in out:
            raise ValueError(f"remap: {src} given twice")
        out[src] = dst
    if len(set(out.values())) != len(out):
        raise ValueError("remap: two sources onto one destination")
    return out


@dataclass
class ReplayResult:
    bag_dir: Path
    counts: Dict[str, int]                   # messages written per topic
    stats: Dict[str, int] = field(default_factory=dict)
    recording_s: float = 0.0
    wall_s: float = 0.0

    @property
    def speed(self) -> float:
        """Recording seconds per wall second (1.0 = a real-time replay)."""
        return self.recording_s / max(self.wall_s, 1e-9)


def decode_ouster_bag(packets_bag, out_dir, packets_ns: str = "/ouster",
                      out_ns: str = "/mobile_1/ouster", driver_min_range: float = 0.0,
                      driver_max_range: float = 1000.0, frame: str = "lidar",
                      imu: bool = True, remap: Optional[Dict[str, str]] = None,
                      limit: int = 0, metadata=None, log=print) -> ReplayResult:
    """Decode ``<packets_ns>/{lidar,imu}_packets`` of ``packets_bag`` into a new
    rosbag2 folder ``out_dir`` (which must not exist). ``remap`` = {src: dst}
    topics copied along; ``limit`` stops after that many scans; ``metadata``
    overrides the bag's metadata (JSON string or SensorInfo)."""
    reader = bag.open_bag(packets_bag)
    remap = dict(remap or {})
    points_t, imu_t, meta_t = f"{out_ns}/points", f"{out_ns}/imu", f"{out_ns}/metadata"
    clash = {points_t, imu_t} & set(remap.values())
    if clash:
        raise ValueError(f"remap onto {sorted(clash)}: those topics are decoded, not copied")
    schemas = bag.topic_schemas(reader) if remap else {}
    missing = sorted(t for t in remap if t not in schemas)
    if missing:
        raise bag.BagError(f"remap sources not in {packets_bag}: {missing}")
    write_meta = meta_t not in remap.values()      # a remapped metadata topic replaces it

    src = metadata or ouster.read_metadata(reader, packets_ns)
    info = ouster.sensor_info(src)
    meta_json = src if isinstance(src, str) else info.to_json_string()
    frame_id = ouster.ROS_FRAME_ID[frame]

    st: Dict[str, int] = {}
    n_scans = 0
    t_first = None
    wall = time.perf_counter()
    with BagWriter(out_dir, library="scoop.replay") as w:
        for ev in ouster.iter_ouster(reader, packets_ns, clock=None, frame=frame,
                                     driver_min_range=driver_min_range,
                                     driver_max_range=driver_max_range, imu=imu,
                                     fields=True, metadata=info,
                                     passthrough=remap.keys(), stats=st):
            if t_first is None:
                t_first = ev.log_ns
                if write_meta:
                    w.write(meta_t, rosmsg.string(meta_json), ev.log_ns)
            if isinstance(ev, ouster.RawMessage):
                w.write_raw(remap[ev.topic], schemas[ev.topic], ev.payload, ev.log_ns)
            elif isinstance(ev, ouster.ImuSample):
                w.write(imu_t, rosmsg.imu(ev.stamp_ns, ouster.ROS_IMU_FRAME_ID,
                                          ev.accel, ev.gyro,
                                          accel_cov=ouster.ROS_IMU_ACCEL_COV,
                                          gyro_cov=ouster.ROS_IMU_GYRO_COV), ev.log_ns)
            else:
                H, W = ev.valid.shape
                w.write(points_t, rosmsg.pointcloud2(ouster.ros_points(ev), ev.stamp_ns,
                                                     frame_id, height=H, width=W),
                        ev.log_ns)
                n_scans += 1
                if n_scans % 500 == 0:
                    rec = (ev.log_ns - t_first) * 1e-9
                    dt = time.perf_counter() - wall
                    log(f"    {n_scans} scans, {rec:.0f} s of recording in {dt:.0f} s "
                        f"({rec / max(dt, 1e-9):.1f}x real time)")
                if limit and n_scans >= limit:
                    break
    res = ReplayResult(w.dir, dict(w.counts), st,
                       recording_s=((w.t_max or 0) - (w.t_min or 0)) * 1e-9,
                       wall_s=time.perf_counter() - wall)
    log(f"wrote {res.bag_dir}: {res.counts}  ({res.recording_s:.1f} s of recording "
        f"in {res.wall_s:.1f} s, {res.speed:.1f}x real time)")
    return res

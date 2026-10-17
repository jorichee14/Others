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
    + the copied topics (all of them with copy_all), byte for byte with
      their QoS, under their own name or the one remap gives them

Stamps are SENSOR time (a replay without timestamp_mode); log times are the
original receive times. ``retime_bag.py`` runs on the result unchanged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

from . import bag, frames, ouster, rosmsg, tftree
from .bagwrite import BagWriter

__all__ = ["parse_remaps", "plan_copies", "decode_ouster_bag", "ReplayResult"]


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


def plan_copies(topics, remap: Dict[str, str], copy_all: bool = False,
                exclude=(), decoded=()) -> Dict[str, str]:
    """source topic -> output topic for everything copied verbatim.

    ``copy_all`` copies every topic of the original (under its own name
    unless ``remap`` renames it); without it only the remapped topics are
    copied. ``exclude`` leaves topics out. ``decoded`` are the topics this
    step writes itself; nothing copied may land on them, and no two sources
    may land on one name."""
    topics = set(topics)
    exclude = set(exclude or ())
    unknown = sorted((set(remap) | exclude) - topics)
    if unknown:
        raise bag.BagError(f"remap/exclude name topics not in the bag: {unknown}")
    srcs = (topics if copy_all else set(remap)) - exclude
    plan = {t: remap.get(t, t) for t in sorted(srcs)}
    clash = sorted(set(decoded) & set(plan.values()))
    if clash:
        raise ValueError(f"copied topics would land on decoded ones: {clash}; "
                         f"exclude or remap them")
    seen: Dict[str, str] = {}
    for src, dst in plan.items():
        if dst in seen:
            raise ValueError(f"{seen[dst]} and {src} would both be written as {dst}")
        seen[dst] = src
    return plan


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


LATCHED = ("- history: 1\n  depth: 1\n  reliability: 1\n  durability: 1\n  deadline:\n"
           "    sec: 9223372036\n    nsec: 854775807\n  lifespan:\n    sec: 9223372036\n"
           "    nsec: 854775807\n  liveliness: 1\n  liveliness_lease_duration:\n"
           "    sec: 9223372036\n    nsec: 854775807\n  avoid_ros_namespace_conventions: false")


def robot_static_tf(packets_bag, cals, own, rename=None, copied=True, log=print):
    """``[(parent, child, t, q)]``: the calibrations of ``cals`` reached from
    the robot's frames ``own``, attached to the TF the bag carries (after
    ``rename``), its Ouster tree kept a subtree (:func:`scoop.tftree.attach`).
    ``copied``: the bag's TF goes into the output (else only its own frames)."""
    mine = tftree.reachable(list(cals or ()), own)
    if not mine:
        return []
    edges = tftree.rename_edges(tftree.tf_edges([packets_bag]), rename or {}) if copied else {}
    keep = {tftree._root(f, edges) for f in own}
    out, notes = tftree.attach(mine, edges, keep=keep)
    for n in notes:
        log(f"    /tf_static += {n}")
    return out


def decode_ouster_bag(packets_bag, out_dir, packets_ns: str = "/ouster",
                      out_ns: str = "/mobile_1/ouster", driver_min_range: float = 0.0,
                      driver_max_range: float = 1000.0, frame: str = "lidar",
                      imu: bool = True, remap: Optional[Dict[str, str]] = None,
                      copy_all: bool = False, exclude=(),
                      limit: int = 0, metadata=None, frame_suffix: str = "", static_tf=(),
                      log=print) -> ReplayResult:
    """Decode ``<packets_ns>/{lidar,imu}_packets`` of ``packets_bag`` into a new
    rosbag2 folder ``out_dir`` (which must not exist). Topics are copied along
    byte for byte with their QoS (:func:`plan_copies`): ``remap`` = {src: dst}
    renames, ``copy_all`` copies every other topic too, ``exclude`` leaves
    some out. ``limit`` stops after that many scans; ``metadata`` overrides
    the bag's sensor metadata (JSON string or SensorInfo). ``frame_suffix``
    (e.g. ``_mobile2``) goes on every Ouster frame name -- os_lidar, os_sensor,
    os_imu -> os_lidar_mobile2, ... -- in the clouds, the IMU and the copied
    topics (their headers, /tf, /tf_static), so the TF tree still holds and
    two robots never share a frame. ``static_tf``: calibrations
    (:class:`scoop.tftree.Calibration`, configs/static_tf.yaml); the ones
    reached from this robot's Ouster frames go into the bag's /tf_static, so
    its processed bag carries its own TF (:func:`robot_static_tf`)."""
    reader = bag.open_bag(packets_bag)
    points_t, imu_t, meta_t = f"{out_ns}/points", f"{out_ns}/imu", f"{out_ns}/metadata"
    copies = plan_copies(reader.topics().keys(), dict(remap or {}), copy_all, exclude,
                         decoded=(points_t, imu_t) if imu else (points_t,))
    schemas = bag.topic_schemas(reader) if copies else {}
    qos = bag.topic_qos(reader)
    write_meta = meta_t not in copies.values()     # a copied metadata topic replaces it

    src = metadata or ouster.read_metadata(reader, packets_ns)
    info = ouster.sensor_info(src)
    meta_json = src if isinstance(src, str) else info.to_json_string()
    rename = ouster.frame_names(frame_suffix)
    frame_id = rename.get(ouster.ROS_FRAME_ID[frame], ouster.ROS_FRAME_ID[frame])
    imu_frame = rename.get(ouster.ROS_IMU_FRAME_ID, ouster.ROS_IMU_FRAME_ID)
    own = [rename.get(f, f) for f in (*ouster.ROS_FRAME_ID.values(), ouster.ROS_IMU_FRAME_ID)]
    calib = robot_static_tf(packets_bag, static_tf, own, rename,
                            copied="/tf_static" in copies or "/tf" in copies, log=log)

    st: Dict[str, int] = {}
    n_scans = 0
    t_first = None
    wall = time.perf_counter()
    with BagWriter(out_dir, library="scoop.replay") as w:
        for src, dst in copies.items():
            w.set_qos(dst, qos.get(src, ""))
        for ev in ouster.iter_ouster(reader, packets_ns, clock=None, frame=frame,
                                     driver_min_range=driver_min_range,
                                     driver_max_range=driver_max_range, imu=imu,
                                     fields=True, metadata=info,
                                     passthrough=copies.keys(), stats=st):
            if t_first is None:
                t_first = ev.log_ns
                if write_meta:
                    w.write(meta_t, rosmsg.string(meta_json), ev.log_ns)
                if calib:
                    w.set_qos("/tf_static", qos.get("/tf_static", "") or LATCHED)
                    w.write("/tf_static", rosmsg.tf_message(calib, ev.log_ns), ev.log_ns)
            if isinstance(ev, ouster.RawMessage):
                payload = frames.rename_payload(schemas[ev.topic], ev.payload, rename)
                w.write_raw(copies[ev.topic], schemas[ev.topic],
                            ev.payload if payload is None else payload, ev.log_ns)
            elif isinstance(ev, ouster.ImuSample):
                w.write(imu_t, rosmsg.imu(ev.stamp_ns, imu_frame,
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

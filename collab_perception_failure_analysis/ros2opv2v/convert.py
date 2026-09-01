# -*- coding: utf-8 -*-
"""
rosbag2 -> OPV2V conversion.

Output layout (exactly what ``opencood.data_utils.datasets.basedataset`` walks)::

    <root>/<split>/<scenario>/<cav_id>/<000000>.pcd
                                      /<000000>.yaml
                                      /<000000>_camera0.png   (optional)

The pipeline is three stages, in this order for a reason:

1. **Index** — stamps only, no deserialisation (:meth:`BagReader.index`).
2. **Plan** — frame times, per-agent message matching, pose resolution.  Every
   frame that cannot be fully resolved is dropped *here*, before anything is
   written, because OpenCOOD indexes all agents by the ego's timestamp keys and a
   half-populated frame is a ``KeyError`` at training time.
3. **Write** — one streaming pass that decodes only the chosen messages.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import labels
from .bagreader import BagReader
from .config import AgentConfig, ConverterConfig, ego_agent
from .geometry import invert, matrix_to_opencood_pose, quat_to_matrix
from .pointclouds import (apply_range_filter, cloud_from_depth_image,
                          cloud_from_pointcloud2, camera_intrinsics, subsample)
from .sync import (NS, Frame, FrameTable, PoseTrack, StampIndex,
                   build_frame_table, frame_times, reuse_statistics)
from .writers import image_to_array, write_frame_yaml, write_pcd, write_png


class ConversionError(RuntimeError):
    pass


@dataclass
class ConversionReport:
    """Everything worth knowing about a run, for the console and provenance file."""
    bag: str = ""
    output: str = ""
    scenarios: List[str] = field(default_factory=list)
    frames_written: int = 0
    frames_candidate: int = 0
    dropped: Dict[str, int] = field(default_factory=dict)
    points_per_agent: Dict[str, dict] = field(default_factory=dict)
    sync: Dict[str, dict] = field(default_factory=dict)
    pose_stats: Dict[str, dict] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "bag": self.bag,
            "output": self.output,
            "scenarios": self.scenarios,
            "frames_written": self.frames_written,
            "frames_candidate": self.frames_candidate,
            "dropped": self.dropped,
            "points_per_agent": self.points_per_agent,
            "sync": self.sync,
            "pose_stats": self.pose_stats,
            "warnings": self.warnings,
            "duration_s": round(self.duration_s, 2),
        }


# --------------------------------------------------------------------- helpers

def _pose_and_speed(msg) -> Tuple[np.ndarray, float]:
    """Extract ``(4x4 transform, speed m/s)`` from an odometry-like message."""
    pose = getattr(msg, "pose", None)
    if pose is None:
        raise ConversionError("pose topic message has no 'pose' field")
    inner = getattr(pose, "pose", pose)          # Odometry / PoseWithCovariance
    position = inner.position
    orientation = inner.orientation

    transform = quat_to_matrix(float(orientation.x), float(orientation.y),
                               float(orientation.z), float(orientation.w))
    transform[:3, 3] = [float(position.x), float(position.y), float(position.z)]

    speed = 0.0
    twist = getattr(msg, "twist", None)
    if twist is not None:
        linear = getattr(getattr(twist, "twist", twist), "linear", None)
        if linear is not None:
            speed = float(np.linalg.norm([linear.x, linear.y, linear.z]))
    return transform, speed


def _fill_speed_from_motion(track: PoseTrack) -> bool:
    """Derive speed by differencing positions when the bag carries no twist.

    Returns True if speeds were replaced.  ``ego_speed`` is only used by
    intermediate fusion (normalised by 30 km/h), so a wrong-but-plausible zero is
    worse than a derived value.
    """
    stamps = track.stamps
    if len(stamps) < 2 or any(abs(s) > 1e-9 for s in track.speeds):
        return False
    positions = np.array([t[:3, 3] for t in track.transforms])
    times = np.asarray(stamps, dtype=np.float64) / NS
    speeds = [0.0] * len(stamps)
    for i in range(1, len(stamps)):
        dt = times[i] - times[i - 1]
        if dt > 1e-6:
            speeds[i] = float(np.linalg.norm(positions[i] - positions[i - 1]) / dt)
    speeds[0] = speeds[1] if len(speeds) > 1 else 0.0
    track.set_speeds(speeds)
    return True


def _sensor_pose(world_from_base: np.ndarray, extrinsic: np.ndarray,
                 ground_lift: float) -> np.ndarray:
    """World pose of a sensor, including the ground-lift compensation.

    ``ground_lift`` shifts the *points* down inside the sensor frame and the
    sensor *pose* up by the same amount, so the world position of every point is
    unchanged while the apparent sensor height above the floor grows.  That is the
    knob for making a knee-high robot LiDAR look to a car-trained detector like a
    roof-mounted one.
    """
    pose = world_from_base @ extrinsic
    if ground_lift:
        pose = pose.copy()
        pose[2, 3] += float(ground_lift)
    return pose


def _apply_ground_lift(cloud: np.ndarray, ground_lift: float) -> np.ndarray:
    if not ground_lift or cloud.shape[0] == 0:
        return cloud
    cloud = cloud.copy()
    cloud[:, 2] -= float(ground_lift)
    return cloud


# ------------------------------------------------------------------- the stages

def load_pose_tracks(reader: BagReader, cfg: ConverterConfig,
                     report: ConversionReport) -> Dict[str, PoseTrack]:
    """Decode every pose topic once and keep the trajectories in memory."""
    topics = sorted({a.pose.topic for a in cfg.active_agents if a.pose.topic})
    tracks: Dict[str, PoseTrack] = {t: PoseTrack() for t in topics}
    if not topics:
        return tracks

    for topic, stamp, msg in reader.iter_messages(topics):
        transform, speed = _pose_and_speed(msg)
        tracks[topic].add(stamp, transform, speed)

    for topic, track in tracks.items():
        track.finish()
        if len(track) == 0:
            raise ConversionError(f"pose topic {topic!r} carried no messages")
        if _fill_speed_from_motion(track):
            report.warnings.append(
                f"{topic}: twist is zero in every message (stationary agent, or a "
                f"driver that does not fill it in) — ego_speed derived by "
                f"differencing positions instead")
    return tracks


def resolve_world_poses(cfg: ConverterConfig, tracks: Dict[str, PoseTrack],
                        t_ns: int) -> Dict[str, Optional[Tuple[np.ndarray, float]]]:
    """``world <- base`` for every active agent at one frame time.

    The chain is ``align @ odom_T_child @ child_to_base``: the odometry message
    gives the child frame in its own odom origin, ``child_to_base`` moves that to
    the robot body, and ``align`` — the operator-supplied transform — is what puts
    all the agents into one shared world.
    """
    out: Dict[str, Optional[Tuple[np.ndarray, float]]] = {}
    for agent in cfg.active_agents:
        if agent.pose.source == "static":
            out[agent.name] = (agent.pose.static_pose.copy(), 0.0)
            continue
        track = tracks[agent.pose.topic]
        found = track.lookup(t_ns, mode=agent.pose.interpolation,
                             max_gap_ns=int(agent.pose.max_gap_ms * 1e6))
        if found is None:
            out[agent.name] = None
            continue
        odom_from_child, speed = found
        out[agent.name] = (agent.pose.align @ odom_from_child @ agent.pose.child_to_base,
                           speed)
    return out


def plan(reader: BagReader, cfg: ConverterConfig,
         report: ConversionReport) -> Tuple[List[Frame], Dict[str, PoseTrack],
                                            Dict[str, dict], FrameTable]:
    """Index, synchronise and pose-resolve: everything before the first write."""
    available = reader.topics()
    wanted = cfg.topics()
    missing = [t for group in wanted.values() for t in group if t not in available]
    if missing:
        raise ConversionError(
            "these configured topics are not in the bag:\n  " +
            "\n  ".join(sorted(set(missing))) +
            "\n\nRun scripts/inspect_bag.py to list what the recording actually has.")

    cloud_topics = wanted["cloud"] + wanted["camera"]
    indexes = reader.index(cloud_topics)

    for topic, entry in indexes.items():
        if not entry.header_stamps:
            raise ConversionError(f"topic {topic!r} is present but empty")
        if entry.headerless:
            report.warnings.append(
                f"{topic}: {entry.headerless} messages had no usable header stamp; "
                f"log time was used for those")

    source = cfg.time.stamp_source
    cloud_indices = {a.name: StampIndex(indexes[a.cloud.topic].stamps(source))
                     for a in cfg.active_agents}
    camera_indices = {}
    for agent in cfg.active_agents:
        for slot, camera in enumerate(agent.cameras):
            camera_indices[(agent.name, slot)] = \
                StampIndex(indexes[camera.topic].stamps(source))

    master_name = cfg.time.master_agent or ego_agent(cfg).name
    master = cfg.agent_by_name(master_name)
    times = frame_times(cloud_indices[master.name].stamps,
                        rate_hz=cfg.time.rate_hz,
                        start_offset_s=cfg.time.start_offset_s,
                        duration_s=cfg.time.duration_s)
    if not times:
        raise ConversionError("no candidate frame times — check time.start_offset_s "
                              "and time.duration_s against the bag's span")

    table = build_frame_table(
        times=times,
        cloud_indices=cloud_indices,
        required={a.name: a.required for a in cfg.active_agents},
        tolerance_ns=int(cfg.time.match_tolerance_ms * 1e6),
        master=master.name,
        drop_incomplete=cfg.time.drop_incomplete_frames,
        camera_indices=camera_indices,
        stride=cfg.output.frame_stride)

    report.frames_candidate = len(times[::max(1, cfg.output.frame_stride)])
    report.dropped = dict(table.dropped)
    report.sync = reuse_statistics(table)
    for name, index in cloud_indices.items():
        report.sync.setdefault(name, {})["source_rate_hz"] = round(index.rate_hz(), 3)

    tracks = load_pose_tracks(reader, cfg, report)

    kept: List[Frame] = []
    frame_poses: Dict[str, dict] = {}
    pose_failures: Dict[str, int] = {}
    for frame in table.frames:
        poses = resolve_world_poses(cfg, tracks, frame.t_ns)
        blocking = [a.name for a in cfg.active_agents
                    if a.required and poses.get(a.name) is None]
        if blocking:
            for name in blocking:
                pose_failures[name] = pose_failures.get(name, 0) + 1
            continue
        frame.index = len(kept)
        kept.append(frame)
        frame_poses[frame.key] = poses

    for name, count in pose_failures.items():
        key = f"no_pose:{name}"
        report.dropped[key] = report.dropped.get(key, 0) + count

    if not kept:
        raise ConversionError(
            "every candidate frame was dropped. Most common causes: "
            "time.match_tolerance_ms too tight for the slowest agent, or a pose "
            "topic that does not span the bag (see the drop reasons above).")

    report.pose_stats = _pose_statistics(cfg, frame_poses, kept)
    return kept, tracks, frame_poses, table


def _pose_statistics(cfg: ConverterConfig, frame_poses: Dict[str, dict],
                     frames: List[Frame]) -> Dict[str, dict]:
    """Trajectory extent and worst inter-frame jump, per agent.

    A large jump usually means the alignment transform is fine but the SLAM
    relocalised — worth seeing before trusting any cross-agent geometry.
    """
    stats: Dict[str, dict] = {}
    for agent in cfg.active_agents:
        positions, jumps = [], []
        previous = None
        for frame in frames:
            entry = frame_poses[frame.key].get(agent.name)
            if entry is None:
                continue
            position = entry[0][:3, 3]
            positions.append(position)
            if previous is not None:
                jumps.append(float(np.linalg.norm(position - previous)))
            previous = position
        if not positions:
            continue
        positions = np.asarray(positions)
        stats[agent.name] = {
            "frames": int(positions.shape[0]),
            "path_length_m": round(float(np.sum(jumps)), 3) if jumps else 0.0,
            "extent_m": [round(float(v), 3)
                         for v in (positions.max(axis=0) - positions.min(axis=0))],
            "max_step_m": round(float(np.max(jumps)), 4) if jumps else 0.0,
        }
    return stats


def assign_scenarios(frames: List[Frame], cfg: ConverterConfig) -> List[Tuple[str, List[Frame]]]:
    """Chop the frame list into OPV2V scenario folders."""
    size = cfg.output.frames_per_scenario
    if size <= 0 or size >= len(frames):
        return [(cfg.output.scenario_name, frames)]
    chunks = []
    for start in range(0, len(frames), size):
        block = frames[start:start + size]
        chunks.append((f"{cfg.output.scenario_name}_{start // size:03d}", block))
    return chunks


def write_frame_yamls(cfg: ConverterConfig, scenarios, frame_poses: Dict[str, dict],
                      report: ConversionReport) -> Dict[str, str]:
    """Write every agent's per-frame yaml; returns frame key -> scenario dir."""
    agent_objects = {
        a.name: {"object_id": a.obj.object_id, "extent": a.obj.extent,
                 "center": a.obj.center}
        for a in cfg.active_agents if a.obj.emit}

    split_root = os.path.join(cfg.output.root, cfg.output.split)
    frame_dirs: Dict[str, str] = {}

    for scenario_name, frames in scenarios:
        scenario_dir = os.path.join(split_root, scenario_name)
        for local_index, frame in enumerate(frames):
            key = f"{local_index:06d}"
            frame_dirs[frame.key] = scenario_dir
            frame.local_key = key                     # consumed by the write pass
            poses = frame_poses[frame.key]
            world_poses = {name: entry[0] if entry else None
                           for name, entry in poses.items()}

            for agent in cfg.active_agents:
                entry = poses.get(agent.name)
                if entry is None:
                    continue
                world_from_base, speed = entry
                lidar_pose = matrix_to_opencood_pose(
                    _sensor_pose(world_from_base, agent.cloud.extrinsic,
                                 agent.cloud.ground_lift))
                body_pose = matrix_to_opencood_pose(world_from_base)

                params = {
                    "ego_speed": float(speed * 3.6),          # OPV2V stores km/h
                    "lidar_pose": lidar_pose,
                    "true_ego_pos": body_pose,
                    "predicted_ego_pos": body_pose,
                    "plan_trajectory": [],
                    "vehicles": labels.vehicles_for_viewer(
                        world_poses, agent_objects, viewer=agent.name,
                        include_self=cfg.output.include_self_in_vehicles),
                }
                for slot, camera in enumerate(agent.cameras):
                    params[f"camera{slot}"] = _camera_block(
                        camera, world_from_base, agent, cfg)

                # Provenance: the real sensor time behind this synthetic frame key.
                params["ros_stamp_ns"] = int(frame.cloud_stamps.get(agent.name,
                                                                    frame.t_ns))
                params["ros_frame_stamp_ns"] = int(frame.t_ns)
                params["source_agent"] = agent.name

                write_frame_yaml(
                    os.path.join(scenario_dir, str(agent.cav_id), f"{key}.yaml"),
                    params)
    return frame_dirs


def _camera_block(camera, world_from_base: np.ndarray, agent: AgentConfig,
                  cfg: ConverterConfig) -> dict:
    """OPV2V-shaped camera metadata (``cords`` / ``extrinsic`` / ``intrinsic``).

    Stock OpenCOOD's LiDAR pipeline never reads these; they are written so
    camera-capable forks and visual debugging have what they need.
    """
    world_from_camera = world_from_base @ camera.extrinsic
    lidar_pose = _sensor_pose(world_from_base, agent.cloud.extrinsic,
                              agent.cloud.ground_lift)
    camera_to_lidar = invert(lidar_pose) @ world_from_camera
    return {
        "cords": matrix_to_opencood_pose(world_from_camera),
        "extrinsic": [[float(v) for v in row] for row in camera_to_lidar],
        "intrinsic": camera.intrinsic or [[0.0, 0.0, 0.0]] * 3,
    }


def write_clouds(reader: BagReader, cfg: ConverterConfig, frames: List[Frame],
                 frame_dirs: Dict[str, str], camera_infos: Dict[str, object],
                 report: ConversionReport,
                 progress: Optional[Callable[[int, int], None]] = None) -> None:
    """The streaming write pass: decode only the selected messages."""
    # topic -> {stamp: [(agent, frame)]}, so one message can serve several frames
    selection: Dict[str, Dict[int, List[Tuple[AgentConfig, Frame, int]]]] = {}
    for frame in frames:
        for agent in cfg.active_agents:
            stamp = frame.cloud_stamps.get(agent.name)
            if stamp is not None:
                selection.setdefault(agent.cloud.topic, {}) \
                    .setdefault(stamp, []).append((agent, frame, -1))
        for (agent_name, slot), stamp in frame.camera_stamps.items():
            agent = cfg.agent_by_name(agent_name)
            selection.setdefault(agent.cameras[slot].topic, {}) \
                .setdefault(stamp, []).append((agent, frame, slot))

    stamp_sets = {topic: set(stamps) for topic, stamps in selection.items()}
    totals: Dict[str, List[int]] = {a.name: [] for a in cfg.active_agents}
    written = 0
    total_targets = sum(len(v) for stamps in selection.values() for v in stamps.values())

    for topic, stamp, msg in reader.iter_selected(stamp_sets):
        for agent, frame, slot in selection[topic].get(stamp, ()):
            target_dir = os.path.join(frame_dirs[frame.key], str(agent.cav_id))
            key = getattr(frame, "local_key", frame.key)
            if slot < 0:
                cloud = _build_cloud(agent, msg, camera_infos)
                count = write_pcd(os.path.join(target_dir, f"{key}.pcd"), cloud)
                totals[agent.name].append(count)
            else:
                image = image_to_array(msg)
                if image is not None:
                    write_png(os.path.join(target_dir, f"{key}_camera{slot}.png"), image)
            written += 1
            if progress and written % 200 == 0:
                progress(written, total_targets)

    for name, counts in totals.items():
        if not counts:
            continue
        report.points_per_agent[name] = {
            "frames": len(counts),
            "mean_points": round(float(np.mean(counts)), 1),
            "min_points": int(np.min(counts)),
            "max_points": int(np.max(counts)),
        }
        if int(np.min(counts)) == 0:
            report.warnings.append(
                f"{name}: at least one frame produced an empty cloud — check "
                f"cloud.range_filter and, for depth clouds, min/max_depth")


def _build_cloud(agent: AgentConfig, msg, camera_infos: Dict[str, object]) -> np.ndarray:
    """Message -> (N, 4) cloud in the agent's *sensor* frame, ready to write."""
    cloud_cfg = agent.cloud
    if cloud_cfg.kind == "pointcloud2":
        cloud = cloud_from_pointcloud2(msg, cloud_cfg.intensity)
    else:
        info = camera_infos.get(cloud_cfg.camera_info_topic)
        if info is None:
            raise ConversionError(
                f"agent {agent.name}: no CameraInfo seen on "
                f"{cloud_cfg.camera_info_topic!r}, cannot reproject depth")
        cloud = cloud_from_depth_image(msg, info, cloud_cfg)

    cloud = apply_range_filter(cloud, cloud_cfg.range_filter)
    cloud = subsample(cloud, cloud_cfg.max_points)
    return _apply_ground_lift(cloud, cloud_cfg.ground_lift)


def load_camera_infos(reader: BagReader, cfg: ConverterConfig) -> Dict[str, object]:
    """First ``CameraInfo`` per topic — intrinsics do not change within a bag."""
    topics = cfg.topics()["camera_info"]
    infos: Dict[str, object] = {}
    if not topics:
        return infos
    remaining = set(topics)
    for topic, stamp, msg in reader.iter_messages(topics):
        if topic in remaining:
            infos[topic] = msg
            remaining.discard(topic)
            if not remaining:
                break
    return infos


def convert(cfg: ConverterConfig, overwrite: bool = False,
            progress: Optional[Callable[[int, int], None]] = None) -> ConversionReport:
    """Run the full conversion described by ``cfg``."""
    started = time.time()
    report = ConversionReport(bag=cfg.bag,
                              output=os.path.join(cfg.output.root, cfg.output.split))

    reader = BagReader(cfg.bag, cfg.time.stamp_source)
    frames, tracks, frame_poses, table = plan(reader, cfg, report)

    scenarios = assign_scenarios(frames, cfg)
    report.scenarios = [name for name, _ in scenarios]

    split_root = os.path.join(cfg.output.root, cfg.output.split)
    for scenario_name, _ in scenarios:
        scenario_dir = os.path.join(split_root, scenario_name)
        if os.path.isdir(scenario_dir):
            if not overwrite:
                raise ConversionError(
                    f"{scenario_dir} already exists. Re-run with --overwrite to "
                    f"replace it, or change output.scenario_name.")
            shutil.rmtree(scenario_dir)
        for agent in cfg.active_agents:
            os.makedirs(os.path.join(scenario_dir, str(agent.cav_id)), exist_ok=True)

    camera_infos = load_camera_infos(reader, cfg)
    for agent in cfg.active_agents:
        for slot, camera in enumerate(agent.cameras):
            if camera.intrinsic is None and camera.camera_info_topic in camera_infos:
                fx, fy, cx, cy = camera_intrinsics(camera_infos[camera.camera_info_topic])
                camera.intrinsic = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

    frame_dirs = write_frame_yamls(cfg, scenarios, frame_poses, report)
    write_clouds(reader, cfg, frames, frame_dirs, camera_infos, report, progress)

    report.frames_written = len(frames)
    report.duration_s = time.time() - started

    if cfg.output.write_provenance:
        provenance = os.path.join(split_root, "conversion_report.json")
        os.makedirs(os.path.dirname(provenance), exist_ok=True)
        with open(provenance, "w") as handle:
            json.dump(report.to_dict(), handle, indent=2)
    return report

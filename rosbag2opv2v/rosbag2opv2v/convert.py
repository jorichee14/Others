"""rosbag2 (MCAP) -> OPV2V dataset conversion."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from .bag import BagSource, MessageRef, RefIndex, resolve_bag_files
from .config import (AgentSpec, CameraSpec, CloudSpec, Config, ExtrinsicSpec)
from .pcd_io import write_pcd
from . import ros_msgs as rm
from . import transforms as tf


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _sanitize(value):
    """Replace NaN/Inf with None so the yaml stays loadable by OpenCOOD."""
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, float):
        return None if (value != value or value in (float("inf"),
                                                    float("-inf"))) else value
    if isinstance(value, (np.floating, np.integer)):
        return _sanitize(value.item())
    return value


def _pose_from_msg(msg) -> Optional[Tuple[List[float], List[float]]]:
    """Extract (position, quaternion xyzw) from Pose*/Odometry/TransformStamped."""
    node = msg
    if hasattr(node, "pose"):
        node = node.pose
        if hasattr(node, "pose"):        # PoseWithCovariance
            node = node.pose
    elif hasattr(node, "transform"):
        node = node.transform
    if hasattr(node, "position") and hasattr(node, "orientation"):
        p, q = node.position, node.orientation
        return [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]
    if hasattr(node, "translation") and hasattr(node, "rotation"):
        p, q = node.translation, node.rotation
        return [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]
    return None


def _matrix_from_spec(xyz, rpy_deg, quat_xyzw=None) -> np.ndarray:
    if quat_xyzw is not None:
        rot = tf.quat_to_matrix(*quat_xyzw)
    else:
        rot = tf.rpy_deg_to_matrix(rpy_deg[0], rpy_deg[1], rpy_deg[2])
    return tf.make_matrix(xyz, rot)


@dataclass
class StreamPlan:
    """A resolved sensor stream: where its data is and how it sits on the agent."""

    key: str
    topic: str
    kind: str                       # cloud | cloud_merge | image
    spec: Any
    index: RefIndex
    extrinsic: np.ndarray           # T_pose_sensor
    extrinsic_method: str = "identity"
    frame_id: str = ""
    intrinsic: Optional[np.ndarray] = None


@dataclass
class FramePlan:
    t: float
    poses: Dict[int, np.ndarray] = field(default_factory=dict)      # agent id -> T_world_pose
    speeds: Dict[int, float] = field(default_factory=dict)
    clouds: Dict[str, MessageRef] = field(default_factory=dict)     # stream key -> ref
    images: Dict[str, MessageRef] = field(default_factory=dict)
    extras: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    complete: set = field(default_factory=set)   # agents with pose + point cloud


class Converter:
    def __init__(self, cfg: Config, bag_path: str, out_dir: str,
                 scenario_prefix: Optional[str] = None, verbose: bool = True):
        self.cfg = cfg
        self.bag_files = resolve_bag_files(bag_path)
        self.out_dir = os.path.abspath(out_dir)
        self.bag_path = os.path.abspath(bag_path)
        self.scenario_prefix = scenario_prefix or cfg.name
        self.verbose = verbose
        self.source = BagSource(self.bag_files)
        self.warnings: List[str] = []
        self.report: Dict[str, Any] = {}
        self.tree = tf.StaticTFTree()
        self.tracks: Dict[int, tf.PoseTrack] = {}
        self.static_poses: Dict[int, np.ndarray] = {}
        self.streams: Dict[int, Dict[str, StreamPlan]] = {}
        self.extras: Dict[int, Dict[str, List[Tuple[float, Any]]]] = {}
        self._warned_once: set = set()
        self.world_frames: Dict[int, str] = {}

    # -- logging -----------------------------------------------------------
    def log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def warn_once(self, key: str, message: str) -> None:
        if key not in self._warned_once:
            self._warned_once.add(key)
            self.warn(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print("WARNING: " + message, file=sys.stderr, flush=True)

    # -- pass 1 ------------------------------------------------------------
    def _topic_sets(self) -> Tuple[List[str], List[str]]:
        decode, index = set(self.cfg.tf_topics if self.cfg.use_tf_static else []), set()
        for agent in self.cfg.agents:
            if agent.pose.source == "topic":
                decode.add(agent.pose.topic)
            for extra in agent.extras:
                decode.add(extra.topic)
            cloud = agent.cloud
            if cloud is not None:
                for part in [cloud] + list(cloud.merge):
                    index.add(part.topic)
                    if part.info_topic:
                        decode.add(part.info_topic)
            for camera in agent.cameras:
                index.add(camera.image_topic)
                if camera.info_topic:
                    decode.add(camera.info_topic)
        return sorted(decode), sorted(index)

    def scan(self) -> None:
        decode_topics, index_topics = self._topic_sets()
        present = self.source.available_topics()
        missing = [t for t in decode_topics + index_topics if t not in present]
        for topic in missing:
            self.warn("topic '%s' is not in the bag" % topic)

        self.log("scanning %d file(s): %s" %
                 (len(self.bag_files),
                  ", ".join(os.path.basename(f) for f in self.bag_files)))
        decoded, index = self.source.scan(
            [t for t in decode_topics if t in present],
            [t for t in index_topics if t in present],
            time_source=self.cfg.time_source,
            progress=lambda n: self.log("  ... %d messages scanned" % n)
                if n and n % 100000 == 0 else None)
        self.decoded = decoded
        self.index = {topic: RefIndex(refs) for topic, refs in index.items()}

        # static transform tree
        for topic in self.cfg.tf_topics:
            for _, msg in decoded.get(topic, []):
                for transform in getattr(msg, "transforms", []):
                    pose = _pose_from_msg(transform)
                    if pose is None:
                        continue
                    self.tree.add(transform.header.frame_id,
                                  transform.child_frame_id,
                                  tf.make_matrix(pose[0],
                                                 tf.quat_to_matrix(*pose[1])))
        if self.cfg.use_tf_static:
            self.log("static TF frames: %s" % ", ".join(self.tree.frames()))

        self._build_poses()
        self._build_streams()
        self._build_extras()

    def _build_poses(self) -> None:
        for agent in self.cfg.agents:
            if agent.pose.source == "static":
                self.static_poses[agent.id] = _matrix_from_spec(
                    agent.pose.xyz, agent.pose.rpy_deg, agent.pose.quat_xyzw)
                continue
            track = tf.PoseTrack()
            for t, msg in self.decoded.get(agent.pose.topic, []):
                pose = _pose_from_msg(msg)
                if pose is None:
                    continue
                track.add(t, pose[0], pose[1])
                header = getattr(msg, "header", None)
                if header is not None and header.frame_id:
                    self.world_frames[agent.id] = header.frame_id
            track.finalize()
            if len(track) == 0:
                self.warn("agent '%s': no poses on %s" %
                          (agent.name, agent.pose.topic))
            self.tracks[agent.id] = track

        frames = {agent_id: name for agent_id, name in self.world_frames.items()}
        if len(set(frames.values())) > 1:
            self.warn("agents report different world frames %s; the export "
                      "assumes they are the same metric frame" % frames)

    def _resolve_extrinsic(self, spec: ExtrinsicSpec, agent: AgentSpec,
                           sensor_frame: str, label: str
                           ) -> Tuple[np.ndarray, str]:
        explicit = _matrix_from_spec(spec.xyz, spec.rpy_deg, spec.quat_xyzw)
        if spec.source == "identity":
            return np.eye(4), "identity"
        if spec.source == "explicit":
            return explicit, "explicit"
        frame = spec.frame or sensor_frame
        pose_frame = agent.pose.frame
        if pose_frame and frame:
            found = self.tree.lookup(pose_frame, frame)
            if found is not None:
                return found, "tf(%s<-%s)" % (pose_frame, frame)
            self.warn("%s: no static TF path %s <- %s; falling back to the "
                      "configured extrinsic" % (label, pose_frame, frame))
        else:
            self.warn("%s: extrinsic.source=tf but %s is unknown; falling back "
                      "to the configured extrinsic" %
                      (label, "pose.frame" if not pose_frame else
                       "the sensor frame_id"))
        return explicit, "explicit-fallback"

    def _stream_frame_id(self, topic: str) -> str:
        idx = self.index.get(topic)
        if idx is None or len(idx) == 0:
            return ""
        return idx.refs[0].frame_id

    def _camera_intrinsic(self, camera: CameraSpec) -> Optional[np.ndarray]:
        if not camera.info_topic:
            return None
        infos = self.decoded.get(camera.info_topic, [])
        if not infos:
            self.warn("no camera_info on %s" % camera.info_topic)
            return None
        return rm.camera_info_to_intrinsic(infos[0][1])

    def _build_streams(self) -> None:
        for agent in self.cfg.agents:
            plans: Dict[str, StreamPlan] = {}
            cloud = agent.cloud
            parts = [("lidar", cloud)] + [
                ("lidar_merge%d" % i, part)
                for i, part in enumerate(cloud.merge)]
            for key, part in parts:
                if part.topic not in self.index:
                    if part.required:
                        self.warn("agent '%s': required cloud topic %s missing"
                                  % (agent.name, part.topic))
                    continue
                frame_id = self._stream_frame_id(part.topic)
                extrinsic, method = self._resolve_extrinsic(
                    part.extrinsic, agent, frame_id,
                    "%s/%s" % (agent.name, key))
                intrinsic = None
                if part.source == "depth":
                    infos = self.decoded.get(part.info_topic or "", [])
                    if not infos:
                        self.warn("agent '%s': no camera_info on %s, cannot "
                                  "build a depth cloud" %
                                  (agent.name, part.info_topic))
                        continue
                    intrinsic = rm.camera_info_to_intrinsic(infos[0][1])
                    if part.to_body_frame:
                        # points get rotated optical -> body, so the extrinsic
                        # must absorb the inverse rotation
                        extrinsic = extrinsic @ tf.invert(rm.OPTICAL_TO_BODY)
                plans[key] = StreamPlan(
                    key=key, topic=part.topic,
                    kind="cloud" if key == "lidar" else "cloud_merge",
                    spec=part, index=self.index[part.topic],
                    extrinsic=extrinsic, extrinsic_method=method,
                    frame_id=frame_id, intrinsic=intrinsic)

            for camera in agent.cameras:
                key = "camera%d" % camera.index
                if camera.image_topic not in self.index:
                    if camera.required:
                        self.warn("agent '%s': required camera topic %s missing"
                                  % (agent.name, camera.image_topic))
                    continue
                frame_id = self._stream_frame_id(camera.image_topic)
                extrinsic, method = self._resolve_extrinsic(
                    camera.extrinsic, agent, frame_id,
                    "%s/%s" % (agent.name, key))
                plans[key] = StreamPlan(
                    key=key, topic=camera.image_topic, kind="image",
                    spec=camera, index=self.index[camera.image_topic],
                    extrinsic=extrinsic, extrinsic_method=method,
                    frame_id=frame_id,
                    intrinsic=self._camera_intrinsic(camera))
            self.streams[agent.id] = plans

    def _build_extras(self) -> None:
        for agent in self.cfg.agents:
            per_agent: Dict[str, List[Tuple[float, Any]]] = {}
            for extra in agent.extras:
                messages = self.decoded.get(extra.topic, [])
                if not messages:
                    self.warn("agent '%s': no messages on extra topic %s" %
                              (agent.name, extra.topic))
                per_agent[extra.key] = messages
            self.extras[agent.id] = per_agent

    # -- planning ----------------------------------------------------------
    def _time_window(self) -> Tuple[float, float]:
        starts, ends = [], []
        for agent in self.cfg.agents:
            if not agent.required:
                continue
            track = self.tracks.get(agent.id)
            if track is not None and len(track):
                starts.append(track.t_start)
                ends.append(track.t_end)
            for plan in self.streams.get(agent.id, {}).values():
                spec_required = getattr(plan.spec, "required", False)
                if spec_required and len(plan.index):
                    starts.append(plan.index.t_start)
                    ends.append(plan.index.t_end)
        if not starts:
            raise RuntimeError("no required stream produced any data; check the "
                               "topic names in the config")
        return max(starts), min(ends)

    def plan(self) -> List[FramePlan]:
        t0, t1 = self._time_window()
        if t1 <= t0:
            raise RuntimeError("required streams do not overlap in time "
                               "(%.3f .. %.3f)" % (t0, t1))
        period = self.cfg.sample_period
        n_samples = int(np.floor((t1 - t0) / period)) + 1
        self.log("sampling %d candidate frames at %.2f Hz over %.1f s" %
                 (n_samples, self.cfg.sample_rate_hz, t1 - t0))

        frames: List[FramePlan] = []
        drops: Dict[str, int] = {}
        missing_images = 0
        for i in range(n_samples):
            t = t0 + i * period
            frame = FramePlan(t=t)
            reasons: List[str] = []
            for agent in self.cfg.agents:
                pose = self._pose_at(agent, t)
                if pose is None:
                    reasons.append("%s:pose" % agent.name)
                    continue
                frame.poses[agent.id] = pose
                frame.speeds[agent.id] = self._speed_at(agent, t)
                frame.extras[agent.id] = self._extras_at(agent, t)

                has_cloud = False
                for key, plan in self.streams.get(agent.id, {}).items():
                    ref = plan.index.nearest(t, plan.spec.max_age)
                    if ref is None:
                        if getattr(plan.spec, "required", False):
                            reasons.append("%s:%s" % (agent.name, key))
                        elif plan.kind == "image":
                            missing_images += 1
                        continue
                    if plan.kind == "image":
                        frame.images["%d/%s" % (agent.id, key)] = ref
                    else:
                        frame.clouds["%d/%s" % (agent.id, key)] = ref
                        has_cloud = True
                if has_cloud and not any(
                        r.startswith(agent.name + ":") for r in reasons):
                    frame.complete.add(agent.id)

            blocking = [a for a in self.cfg.agents
                        if a.required and a.id not in frame.complete]
            if blocking:
                for reason in reasons or ["%s:incomplete" % blocking[0].name]:
                    drops[reason] = drops.get(reason, 0) + 1
                if self.cfg.drop_incomplete_frames:
                    continue
            frames.append(frame)
            if self.cfg.max_frames and len(frames) >= self.cfg.max_frames:
                break

        self.report["candidate_frames"] = n_samples
        self.report["accepted_frames"] = len(frames)
        self.report["dropped_frames"] = drops
        self.report["frames_missing_an_image"] = missing_images
        if drops:
            self.log("incomplete frames: %s" %
                     ", ".join("%s=%d" % kv for kv in sorted(drops.items())))
        return frames

    def _pose_at(self, agent: AgentSpec, t: float) -> Optional[np.ndarray]:
        if agent.pose.source == "static":
            return self.static_poses[agent.id]
        track = self.tracks.get(agent.id)
        if track is None or len(track) == 0:
            return None
        return track.sample(t, agent.pose.max_gap)

    def _speed_at(self, agent: AgentSpec, t: float) -> float:
        if agent.pose.source == "static":
            return 0.0
        track = self.tracks.get(agent.id)
        return track.velocity(t) if track is not None and len(track) else 0.0

    def _extras_at(self, agent: AgentSpec, t: float) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for extra in agent.extras:
            messages = self.extras.get(agent.id, {}).get(extra.key, [])
            if not messages:
                continue
            times = [item[0] for item in messages]
            import bisect
            idx = bisect.bisect_left(times, t)
            best, best_dt = None, None
            for candidate in (idx - 1, idx, idx + 1):
                if 0 <= candidate < len(messages):
                    dt = abs(times[candidate] - t)
                    if best_dt is None or dt < best_dt:
                        best, best_dt = messages[candidate], dt
            if best is None or best_dt > extra.max_age:
                continue
            plain = rm.msg_to_plain(best[1])
            if plain is None:
                continue
            if extra.fields:
                plain = {k: v for k, v in plain.items() if k in extra.fields}
            plain["_dt"] = round(best_dt, 4)
            out[extra.key] = plain
        return out

    # -- writing -----------------------------------------------------------
    def _scenarios(self, frames: List[FramePlan]) -> List[Tuple[str, str, List[FramePlan]]]:
        """Split the frame list into (split, scenario_name, frames) chunks."""
        if self.cfg.scenario_seconds:
            chunks: List[List[FramePlan]] = []
            current: List[FramePlan] = []
            chunk_start = frames[0].t if frames else 0.0
            for frame in frames:
                if frame.t - chunk_start >= self.cfg.scenario_seconds and current:
                    chunks.append(current)
                    current, chunk_start = [], frame.t
                current.append(frame)
            if current:
                chunks.append(current)
        else:
            chunks = [frames]

        chunks = [c for c in chunks if len(c) >= self.cfg.min_frames_per_scenario]
        out = []
        splits = self._split_assignment(len(chunks))
        for i, chunk in enumerate(chunks):
            name = "%s_%03d" % (self.scenario_prefix, i)
            out.append((splits[i], name, chunk))
        return out

    def _split_assignment(self, count: int) -> List[str]:
        if not self.cfg.splits:
            return [self.cfg.split] * count
        names = list(self.cfg.splits.keys())
        weights = np.asarray([float(self.cfg.splits[n]) for n in names])
        weights = weights / weights.sum()
        # deterministic largest-remainder allocation, in scenario order
        quota = weights * count
        counts = np.floor(quota).astype(int)
        for i in np.argsort(-(quota - counts))[:count - counts.sum()]:
            counts[i] += 1
        out: List[str] = []
        for name, n in zip(names, counts):
            out.extend([name] * int(n))
        return out[:count] or [names[0]] * count

    def convert(self, dry_run: bool = False) -> Dict[str, Any]:
        self.scan()
        frames = self.plan()
        if not frames:
            raise RuntimeError("no frame survived the completeness check; relax "
                               "max_age / max_gap or set "
                               "dataset.drop_incomplete_frames: false")
        scenarios = self._scenarios(frames)
        if not scenarios:
            raise RuntimeError("no scenario reached min_frames_per_scenario")

        self.report.update({
            "bag": self.bag_path,
            "files": [os.path.basename(f) for f in self.bag_files],
            "out_dir": self.out_dir,
            "sample_rate_hz": self.cfg.sample_rate_hz,
            "scenarios": [{"split": s, "name": n, "frames": len(f),
                           "t_start": f[0].t, "t_end": f[-1].t}
                          for s, n, f in scenarios],
            "agents": [{
                "id": a.id, "name": a.name, "role": a.role,
                "streams": {k: {"topic": p.topic, "frame_id": p.frame_id,
                                "extrinsic": p.extrinsic_method}
                            for k, p in self.streams.get(a.id, {}).items()},
            } for a in self.cfg.agents],
        })

        jobs: Dict[Tuple[str, int], List[Any]] = {}
        pending: Dict[str, Dict[str, Any]] = {}
        stats: Dict[str, List[float]] = {}

        kept_scenarios = []
        for split, scenario, chunk in scenarios:
            # OpenCOOD assumes every cav folder in a scenario holds the very
            # same timestamps, so an agent is written only if it is complete in
            # *every* frame of the chunk.
            agents = [a for a in self.cfg.agents
                      if all(a.id in f.complete for f in chunk)]
            if self.cfg.ego.id not in [a.id for a in agents]:
                self.warn("scenario %s dropped: the ego agent (%s) is not "
                          "complete in every frame" %
                          (scenario, self.cfg.ego.name))
                continue
            for skipped in [a for a in self.cfg.agents if a not in agents]:
                self.warn("scenario %s: agent '%s' is missing in some frames "
                          "and is left out of that scenario" %
                          (scenario, skipped.name))
            kept_scenarios.append((split, scenario, chunk, agents))
            for i, frame in enumerate(chunk):
                index = self.cfg.start_index + i * self.cfg.timestamp_step
                stamp = "%06d" % index
                for agent in agents:
                    folder = os.path.join(self.out_dir, split, scenario,
                                          agent.folder)
                    if not dry_run:
                        os.makedirs(folder, exist_ok=True)
                    params = self._frame_params(agent, frame, stats,
                                                agents)
                    if not dry_run:
                        with open(os.path.join(folder, stamp + ".yaml"),
                                  "w") as handle:
                            yaml.safe_dump(_sanitize(params), handle,
                                           default_flow_style=None,
                                           sort_keys=False)
                    if dry_run:
                        continue
                    self._queue_frame_jobs(agent, frame, folder, stamp,
                                           jobs, pending)

        scenarios = [(s, n, c) for s, n, c, _ in kept_scenarios]
        if not scenarios:
            raise RuntimeError("every scenario was dropped for incompleteness")
        self.report["scenarios"] = [
            {"split": s, "name": n, "frames": len(c),
             "agents": [a.folder for a in agents], "t_start": c[0].t,
             "t_end": c[-1].t} for s, n, c, agents in kept_scenarios]
        self.report["output_files"] = {
            "pcd": len(pending),
            "images": sum(1 for v in jobs.values()
                          for j in v if j[0] == "image"),
        }
        if dry_run:
            self.report["dry_run"] = True
            return self.report

        self.log("decoding %d selected messages for %d point clouds" %
                 (len(jobs), len(pending)))
        self._run_jobs(jobs, pending)
        for split, scenario, chunk, agents in kept_scenarios:
            self._write_scenario_meta(split, scenario, chunk, agents)
        report_path = os.path.join(self.out_dir, "conversion_report.json")
        os.makedirs(self.out_dir, exist_ok=True)
        self.report["warnings"] = self.warnings
        self.report["timing_stats"] = {
            k: {"mean_ms": float(np.mean(v) * 1e3),
                "max_ms": float(np.max(np.abs(v)) * 1e3), "count": len(v)}
            for k, v in stats.items() if v}
        with open(report_path, "w") as handle:
            json.dump(_sanitize(self.report), handle, indent=2)
        self.log("wrote %s" % report_path)
        return self.report

    # -- per-frame yaml ----------------------------------------------------
    def _frame_params(self, agent: AgentSpec, frame: FramePlan,
                      stats: Dict[str, List[float]],
                      agents: List[AgentSpec]) -> Dict[str, Any]:
        world_pose = frame.poses[agent.id]
        plans = self.streams.get(agent.id, {})
        lidar_plan = plans.get("lidar")
        lidar_world = (world_pose @ lidar_plan.extrinsic
                       if lidar_plan is not None else world_pose)

        params: Dict[str, Any] = {
            "ego_speed": float(frame.speeds.get(agent.id, 0.0) * 3.6),
            "lidar_pose": tf.matrix_to_opv2v_pose(lidar_world),
            "true_ego_pos": tf.matrix_to_opv2v_pose(world_pose),
            "predicted_ego_pos": tf.matrix_to_opv2v_pose(world_pose),
            "plan_trajectory": [],
            "vehicles": self._vehicles(agent, frame, agents),
        }
        if agent.role == "rsu":
            params["RSU"] = True

        extra: Dict[str, Any] = {
            "agent": agent.name,
            "role": agent.role,
            "stamp": float(frame.t),
            "ego_speed_mps": float(frame.speeds.get(agent.id, 0.0)),
        }
        for key, plan in plans.items():
            ref = (frame.clouds.get("%d/%s" % (agent.id, key)) or
                   frame.images.get("%d/%s" % (agent.id, key)))
            if ref is None:
                continue
            dt = ref.t - frame.t
            extra["%s_dt" % key] = round(float(dt), 4)
            stats.setdefault("%s/%s" % (agent.name, key), []).append(dt)
            if plan.kind == "image":
                sensor_world = world_pose @ plan.extrinsic
                params["camera%d" % plan.spec.index] = {
                    "cords": tf.matrix_to_opv2v_pose(sensor_world),
                    "extrinsic": (tf.invert(lidar_world) @
                                  sensor_world).tolist(),
                    "intrinsic": (plan.intrinsic.tolist()
                                  if plan.intrinsic is not None
                                  else np.eye(3).tolist()),
                }
        extra.update(frame.extras.get(agent.id, {}))
        params["mirc"] = extra
        return params

    def _vehicles(self, agent: AgentSpec, frame: FramePlan,
                  agents: List[AgentSpec]) -> Dict[int, Any]:
        """Ground-truth boxes this agent publishes: every *other* exported
        agent, in world coordinates, OPV2V-encoded."""
        vehicles: Dict[int, Any] = {}
        for other in agents:
            if other.id == agent.id or not other.obj.publish:
                continue
            if (not self.cfg.include_ego_as_object
                    and other.id == self.cfg.ego.id):
                # the ego never sees itself; OpenCOOD merges every agent's
                # vehicle dict, so publishing the ego's box makes it a target
                continue
            pose = frame.poses.get(other.id)
            if pose is None:
                continue
            encoded = tf.matrix_to_opv2v_pose(pose)
            vehicles[other.id] = {
                "angle": [encoded[3], encoded[4], encoded[5]],
                "center": [float(v) for v in other.obj.center],
                "extent": [float(v) for v in other.obj.extent],
                "location": [encoded[0], encoded[1], encoded[2]],
                "speed": float(frame.speeds.get(other.id, 0.0) * 3.6),
                "class": other.obj.class_name,
            }
        return vehicles

    # -- pass 2 ------------------------------------------------------------
    def _queue_frame_jobs(self, agent: AgentSpec, frame: FramePlan, folder: str,
                          stamp: str, jobs: Dict[Tuple[str, int], List[Any]],
                          pending: Dict[str, Dict[str, Any]]) -> None:
        plans = self.streams.get(agent.id, {})
        lidar_plan = plans.get("lidar")
        pcd_path = os.path.join(folder, stamp + ".pcd")
        parts = []
        for key, plan in plans.items():
            ref = frame.clouds.get("%d/%s" % (agent.id, key))
            if plan.kind in ("cloud", "cloud_merge") and ref is not None:
                parts.append((key, plan, ref))
        if parts:
            state = {"path": pcd_path, "remaining": len(parts), "xyz": [],
                     "intensity": []}
            pending[pcd_path] = state
            base = lidar_plan.extrinsic if lidar_plan is not None else np.eye(4)
            for key, plan, ref in parts:
                to_primary = tf.invert(base) @ plan.extrinsic
                jobs.setdefault((plan.topic, ref.ordinal), []).append(
                    ("cloud", pcd_path, plan, to_primary))

        if not self.cfg.write_images:
            return
        for key, plan in plans.items():
            ref = frame.images.get("%d/%s" % (agent.id, key))
            if plan.kind != "image" or ref is None:
                continue
            path = os.path.join(folder, "%s_camera%d.png" %
                                (stamp, plan.spec.index))
            jobs.setdefault((plan.topic, ref.ordinal), []).append(
                ("image", path, plan, None))

    def _run_jobs(self, jobs: Dict[Tuple[str, int], List[Any]],
                  pending: Dict[str, Dict[str, Any]]) -> None:
        written = {"pcd": 0, "png": 0, "points": 0}

        def handle(topic, ordinal, msg, payload):
            for kind, path, plan, matrix in payload:
                if kind == "cloud":
                    self._accumulate_cloud(msg, plan, matrix, pending[path],
                                           written)
                else:
                    self._write_image(msg, path, written)

        def progress(done, total):
            if done % 500 == 0:
                self.log("  ... %d/%d messages decoded" % (done, total))

        self.source.fetch(jobs, handle, progress)
        leftovers = [state for state in pending.values() if state["remaining"]]
        if leftovers:
            self.warn("%d point clouds stayed incomplete and were written from "
                      "the parts that were found" % len(leftovers))
            for state in leftovers:
                self._flush_cloud(state, written, force=True)
        self.report["written"] = written
        self.log("wrote %d pcd files (%d points) and %d images" %
                 (written["pcd"], written["points"], written["png"]))

    def _accumulate_cloud(self, msg, plan: StreamPlan, matrix: np.ndarray,
                          state: Dict[str, Any], written: Dict[str, int]) -> None:
        spec: CloudSpec = plan.spec
        if spec.source == "depth":
            depth = rm.image_to_array(msg)
            xyz = rm.depth_to_points(depth, plan.intrinsic, spec.depth_scale,
                                     spec.stride, spec.min_range, spec.max_range)
            if spec.to_body_frame:
                xyz = rm.transform_points(xyz, rm.OPTICAL_TO_BODY)
            intensity = np.zeros(xyz.shape[0], dtype=np.float32)
        else:
            names = [f.name for f in msg.fields]
            if spec.intensity_field and spec.intensity_field not in names:
                self.warn_once(
                    "intensity/" + plan.topic,
                    "%s: no '%s' field (available: %s); the exported intensity "
                    "channel will be zero" %
                    (plan.topic, spec.intensity_field, ", ".join(names)))
            xyz, intensity = rm.pointcloud2_to_xyzi(
                msg, spec.intensity_field, spec.intensity_scale)
            if spec.intensity_normalize == "percentile" and intensity.size:
                top = float(np.percentile(intensity, spec.intensity_percentile))
                intensity = intensity / max(top, 1e-6)
            if spec.min_range > 0.0 or np.isfinite(spec.max_range):
                radius = np.linalg.norm(xyz, axis=1)
                keep = (radius >= spec.min_range) & (radius <= spec.max_range)
                xyz, intensity = xyz[keep], intensity[keep]
        if not np.allclose(matrix, np.eye(4)):
            xyz = rm.transform_points(xyz, matrix)
        state["xyz"].append(xyz)
        state["intensity"].append(intensity)
        state["remaining"] -= 1
        if state["remaining"] <= 0:
            self._flush_cloud(state, written)

    def _flush_cloud(self, state: Dict[str, Any], written: Dict[str, int],
                     force: bool = False) -> None:
        if not state["xyz"]:
            return
        xyz = np.concatenate(state["xyz"], axis=0)
        intensity = np.concatenate(state["intensity"], axis=0)
        written["points"] += int(write_pcd(state["path"], xyz, intensity,
                                           binary=self.cfg.pcd_binary))
        written["pcd"] += 1
        state["xyz"], state["intensity"], state["remaining"] = [], [], 0

    def _write_image(self, msg, path: str, written: Dict[str, int]) -> None:
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover - depends on the environment
            self.warn("Pillow is not installed; images are skipped "
                      "(pip install pillow, or set write_images: false)")
            self.cfg.write_images = False
            return
        if msg.encoding.lower().startswith("bayer"):
            self.warn_once("bayer", "images on this topic are %s; they are "
                                    "written as raw grayscale, not demosaicked"
                           % msg.encoding)
        rgb = rm.image_to_rgb(msg)
        Image.fromarray(rgb).save(path)
        written["png"] += 1

    def _write_scenario_meta(self, split: str, scenario: str,
                             chunk: List[FramePlan],
                             agents: List[AgentSpec]) -> None:
        path = os.path.join(self.out_dir, split, scenario, "scenario_meta.yaml")
        meta = {
            "source_bag": self.bag_path,
            "scenario": scenario,
            "split": split,
            "frames": len(chunk),
            "sample_rate_hz": self.cfg.sample_rate_hz,
            "t_start_unix": float(chunk[0].t),
            "t_end_unix": float(chunk[-1].t),
            "ego_id": self.cfg.ego.id,
            "agents": {a.folder: {
                "name": a.name, "role": a.role,
                "is_object": a.obj.publish,
                # only tracked agents contribute a ground-truth box size
                "extent": a.obj.extent if a.obj.publish else None}
                for a in agents},
        }
        with open(path, "w") as handle:
            yaml.safe_dump(_sanitize(meta), handle, sort_keys=False)

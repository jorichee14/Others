# -*- coding: utf-8 -*-
"""
Converter configuration: the full description of how one rosbag2 recording maps
onto an OPV2V-shaped dataset.

Everything the conversion needs is declared here rather than hard-coded, because
the two things that cannot be recovered from the bag itself — the transform that
puts each agent's odometry into a *shared* world frame, and the physical extent
of each robot — have to come from the operator.  Missing values fail loudly at
load time instead of silently producing a geometrically meaningless dataset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import yaml

from .geometry import make_transform


class ConfigError(ValueError):
    """Raised for any structurally invalid or under-specified config."""


_REQUIRED = object()


def _get(mapping: dict, key: str, default=_REQUIRED, ctx: str = ""):
    if key in mapping and mapping[key] is not None:
        return mapping[key]
    if default is _REQUIRED:
        raise ConfigError(f"{ctx}: missing required key '{key}'")
    return default


def _transform_from(mapping: Optional[dict], ctx: str) -> np.ndarray:
    """Parse an ``{x, y, z, roll, pitch, yaw}`` block into a 4x4 (ROS RPY, degrees).

    ``None`` is *not* accepted as "identity": a null transform in the config is a
    placeholder the operator has not filled in yet, and silently treating it as
    identity is exactly the failure mode this converter must not have.
    """
    if mapping is None:
        raise ConfigError(
            f"{ctx}: transform is null. Fill in x/y/z/roll/pitch/yaw (degrees), "
            f"or write an explicit identity ({{x: 0, y: 0, z: 0, roll: 0, pitch: 0, yaw: 0}}) "
            f"if you are certain the frames already coincide.")
    unknown = set(mapping) - {"x", "y", "z", "roll", "pitch", "yaw"}
    if unknown:
        raise ConfigError(f"{ctx}: unknown transform keys {sorted(unknown)}")
    return make_transform(
        x=float(mapping.get("x", 0.0)), y=float(mapping.get("y", 0.0)),
        z=float(mapping.get("z", 0.0)), roll=float(mapping.get("roll", 0.0)),
        pitch=float(mapping.get("pitch", 0.0)), yaw=float(mapping.get("yaw", 0.0)),
        degrees=True)


@dataclass
class IntensityConfig:
    """How to turn a raw point attribute into OPV2V's [0, 1] intensity channel."""
    field_name: Optional[str] = "intensity"
    scale: float = 1.0
    offset: float = 0.0
    default: float = 0.5

    @staticmethod
    def parse(mapping: Optional[dict]) -> "IntensityConfig":
        mapping = mapping or {}
        name = mapping.get("field", "intensity")
        if isinstance(name, str) and name.lower() in ("none", "null", ""):
            name = None
        return IntensityConfig(
            field_name=name,
            scale=float(mapping.get("scale", 1.0)),
            offset=float(mapping.get("offset", 0.0)),
            default=float(mapping.get("default", 0.5)))


@dataclass
class CloudConfig:
    """The sensor that produces an agent's OPV2V point cloud (one per agent)."""
    kind: str                      # 'pointcloud2' | 'depth_image'
    topic: str
    extrinsic: np.ndarray          # base -> sensor
    intensity: IntensityConfig = field(default_factory=IntensityConfig)
    camera_info_topic: Optional[str] = None
    depth_scale: float = 0.001     # 16UC1 counts -> metres (ignored for 32FC1)
    max_depth: float = 20.0
    min_depth: float = 0.1
    pixel_stride: int = 2
    optical_frame: bool = True
    range_filter: Optional[List[float]] = None   # sensor frame [xmin..zmax]
    max_points: int = 0            # 0 = keep all; else random-free uniform subsample
    ground_lift: float = 0.0       # see docs/ROS2OPV2V.md ("ground lift")
    min_points: int = 1            # frames below this are treated as missing
    point_time_field: Optional[str] = None
    """Per-point acquisition offset from the message stamp, e.g. the Ouster's ``t``
    (nanoseconds). Present on most spinning LiDARs and on nothing else."""
    deskew: bool = False
    """Move every point to the frame's reference instant using the agent's own pose
    track. Needs ``point_time_field``. A 10 Hz sweep observes over ~100 ms — the
    same order as the whole inter-agent budget — and the resulting smear is
    azimuth-dependent, so it does not average out."""
    deskew_buckets: int = 64

    @staticmethod
    def parse(mapping: dict, ctx: str) -> "CloudConfig":
        kind = str(_get(mapping, "kind", ctx=ctx)).lower()
        if kind not in ("pointcloud2", "depth_image"):
            raise ConfigError(f"{ctx}: cloud kind must be pointcloud2 or depth_image, got {kind!r}")
        cfg = CloudConfig(
            kind=kind,
            topic=str(_get(mapping, "topic", ctx=ctx)),
            extrinsic=_transform_from(mapping.get("extrinsic"), f"{ctx}.extrinsic"),
            intensity=IntensityConfig.parse(mapping.get("intensity")),
            camera_info_topic=mapping.get("camera_info_topic"),
            depth_scale=float(mapping.get("depth_scale", 0.001)),
            max_depth=float(mapping.get("max_depth", 20.0)),
            min_depth=float(mapping.get("min_depth", 0.1)),
            pixel_stride=int(mapping.get("pixel_stride", 2)),
            optical_frame=bool(mapping.get("optical_frame", True)),
            range_filter=mapping.get("range_filter"),
            max_points=int(mapping.get("max_points", 0)),
            ground_lift=float(mapping.get("ground_lift", 0.0)),
            min_points=int(mapping.get("min_points", 1)),
            point_time_field=mapping.get("point_time_field"),
            deskew=bool(mapping.get("deskew", False)),
            deskew_buckets=int(mapping.get("deskew_buckets", 64)))
        if cfg.kind == "depth_image" and not cfg.camera_info_topic:
            raise ConfigError(f"{ctx}: depth_image clouds need a camera_info_topic")
        if cfg.range_filter is not None and len(cfg.range_filter) != 6:
            raise ConfigError(f"{ctx}.range_filter must be [xmin, ymin, zmin, xmax, ymax, zmax]")
        if cfg.pixel_stride < 1:
            raise ConfigError(f"{ctx}.pixel_stride must be >= 1")
        if cfg.deskew and not cfg.point_time_field:
            raise ConfigError(
                f"{ctx}: deskew needs point_time_field (the per-point time offset "
                f"field, 't' on an Ouster). Without it there is nothing to deskew "
                f"against, and silently skipping the correction would leave a "
                f"motion smear the config says was removed.")
        if cfg.deskew and cfg.kind != "pointcloud2":
            raise ConfigError(f"{ctx}: deskew applies to pointcloud2 clouds only "
                              f"(a depth image is a single exposure, not a sweep)")
        if cfg.deskew_buckets < 1:
            raise ConfigError(f"{ctx}.deskew_buckets must be >= 1")
        return cfg


@dataclass
class CameraConfig:
    """Optional RGB export (``<ts>_cameraN.png`` + a ``cameraN`` yaml block)."""
    topic: str
    extrinsic: np.ndarray          # base -> camera (body convention)
    camera_info_topic: Optional[str] = None
    intrinsic: Optional[List[List[float]]] = None
    optical_frame: bool = True

    @staticmethod
    def parse(mapping: dict, ctx: str) -> "CameraConfig":
        return CameraConfig(
            topic=str(_get(mapping, "topic", ctx=ctx)),
            extrinsic=_transform_from(mapping.get("extrinsic"), f"{ctx}.extrinsic"),
            camera_info_topic=mapping.get("camera_info_topic"),
            intrinsic=mapping.get("intrinsic"),
            optical_frame=bool(mapping.get("optical_frame", True)))


@dataclass
class PoseConfig:
    """How an agent's body pose in the shared world frame is obtained."""
    source: str                    # 'odometry' | 'pose' | 'static'
    topic: Optional[str] = None
    align: np.ndarray = field(default_factory=lambda: np.identity(4))   # world <- odom
    child_to_base: np.ndarray = field(default_factory=lambda: np.identity(4))
    static_pose: Optional[np.ndarray] = None                            # world <- base
    interpolation: str = "linear"  # 'linear' | 'nearest'
    max_gap_ms: float = 200.0

    @staticmethod
    def parse(mapping: dict, ctx: str) -> "PoseConfig":
        source = str(_get(mapping, "source", ctx=ctx)).lower()
        if source not in ("odometry", "pose", "static"):
            raise ConfigError(f"{ctx}: pose source must be odometry, pose or static")
        if source == "static":
            return PoseConfig(
                source=source,
                static_pose=_transform_from(mapping.get("world_pose"), f"{ctx}.world_pose"))
        interp = str(mapping.get("interpolation", "linear")).lower()
        if interp not in ("linear", "nearest"):
            raise ConfigError(f"{ctx}.interpolation must be linear or nearest")
        # An *absent* child_to_base means "the odometry child frame already is the
        # robot base"; an explicit null means "placeholder, not filled in yet".
        child_to_base = np.identity(4)
        if "child_to_base" in mapping:
            child_to_base = _transform_from(mapping["child_to_base"],
                                            f"{ctx}.child_to_base")
        return PoseConfig(
            source=source,
            topic=str(_get(mapping, "topic", ctx=ctx)),
            align=_transform_from(mapping.get("align"), f"{ctx}.align"),
            child_to_base=child_to_base,
            interpolation=interp,
            max_gap_ms=float(mapping.get("max_gap_ms", 200.0)))


@dataclass
class ObjectConfig:
    """The agent's own 3D box, as seen by the *other* agents (pseudo ground truth)."""
    emit: bool = False
    extent: Optional[List[float]] = None    # half-dims in base frame [dx, dy, dz]
    center: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    object_id: Optional[int] = None

    @staticmethod
    def parse(mapping: Optional[dict], ctx: str) -> "ObjectConfig":
        mapping = mapping or {}
        cfg = ObjectConfig(
            emit=bool(mapping.get("emit", False)),
            extent=mapping.get("extent"),
            center=list(mapping.get("center", [0.0, 0.0, 0.0])),
            object_id=mapping.get("object_id"))
        if cfg.emit and (cfg.extent is None or len(cfg.extent) != 3):
            raise ConfigError(f"{ctx}.extent must be [dx, dy, dz] half-dimensions in metres")
        if len(cfg.center) != 3:
            raise ConfigError(f"{ctx}.center must be [x, y, z]")
        return cfg


@dataclass
class AgentConfig:
    """One OPV2V CAV/RSU folder."""
    name: str
    cav_id: int
    pose: PoseConfig
    cloud: CloudConfig
    role: str = "cav"              # 'cav' | 'rsu'
    enabled: bool = True
    cameras: List[CameraConfig] = field(default_factory=list)
    obj: ObjectConfig = field(default_factory=ObjectConfig)
    speed_source: Optional[str] = None   # odometry topic for ego_speed; defaults to pose topic
    required: bool = True          # a frame missing a required agent is dropped
    host: Optional[str] = None     # clock domain; defaults to the agent's own name

    @property
    def clock_host(self) -> str:
        """The machine whose clock stamps this agent's messages.

        Defaults to the agent name because one robot is usually one host. Set it
        explicitly when two agents share a machine — they then share an offset,
        and estimating it twice from half the messages is worse than once from all
        of them.
        """
        return self.host or self.name

    @staticmethod
    def parse(mapping: dict) -> "AgentConfig":
        name = str(_get(mapping, "name", ctx="agent"))
        ctx = f"agent[{name}]"
        role = str(mapping.get("role", "cav")).lower()
        if role not in ("cav", "rsu"):
            raise ConfigError(f"{ctx}: role must be cav or rsu")
        cav_id = int(_get(mapping, "id", ctx=ctx))
        if role == "rsu" and cav_id >= 0:
            raise ConfigError(
                f"{ctx}: OpenCOOD identifies roadside units by a negative folder id "
                f"(it sorts them last so they are never chosen as ego); got {cav_id}")
        if role == "cav" and cav_id < 0:
            raise ConfigError(f"{ctx}: a cav id must be >= 0 (negative ids mean RSU)")
        return AgentConfig(
            name=name,
            cav_id=cav_id,
            role=role,
            enabled=bool(mapping.get("enabled", True)),
            pose=PoseConfig.parse(_get(mapping, "pose", ctx=ctx), f"{ctx}.pose"),
            cloud=CloudConfig.parse(_get(mapping, "cloud", ctx=ctx), f"{ctx}.cloud"),
            cameras=[CameraConfig.parse(c, f"{ctx}.cameras[{i}]")
                     for i, c in enumerate(mapping.get("cameras") or [])],
            obj=ObjectConfig.parse(mapping.get("object"), f"{ctx}.object"),
            speed_source=mapping.get("speed_source"),
            required=bool(mapping.get("required", True)),
            host=mapping.get("host"))


@dataclass
class ClockConfig:
    """Cross-host clock reconciliation (see ``ros2opv2v/clock.py``).

    Off by default, because on a single-host recording there is nothing to
    reconcile and a spurious correction is worse than none. Turn it on for any bag
    whose agents ran on different machines — which is every real testbed.
    """
    enabled: bool = False
    reference_host: Optional[str] = None    # defaults to the ego agent's host
    ntp_topics: Dict[str, str] = field(default_factory=dict)   # host -> topic
    offset_field: Optional[str] = None      # e.g. 'offset'; None = resolve at runtime
    offset_unit: Optional[str] = None       # 's' | 'ms' | 'us' | 'ns'; None = infer
    sign: str = "auto"                      # 'auto' | '+1' | '-1'
    cross_check_tolerance_ms: float = 20.0
    require_measured: bool = False
    """Refuse to convert while any host's offset is unmeasured. Off by default so a
    bag missing one NTP topic still converts — with the consequence reported in
    every frame — but worth turning on for a dataset that will be published."""

    @staticmethod
    def parse(mapping: Optional[dict]) -> "ClockConfig":
        mapping = mapping or {}
        sign = str(mapping.get("sign", "auto")).lower()
        if sign not in ("auto", "+1", "-1", "1"):
            raise ConfigError("clock.sign must be auto, +1 or -1")
        topics = mapping.get("ntp_topics") or {}
        if not isinstance(topics, dict):
            raise ConfigError("clock.ntp_topics must be a mapping of host -> topic")
        return ClockConfig(
            enabled=bool(mapping.get("enabled", False)),
            reference_host=mapping.get("reference_host"),
            ntp_topics={str(k): str(v) for k, v in topics.items()},
            offset_field=mapping.get("offset_field"),
            offset_unit=(str(mapping["offset_unit"]).lower()
                         if mapping.get("offset_unit") else None),
            sign=sign,
            cross_check_tolerance_ms=float(mapping.get("cross_check_tolerance_ms", 20.0)),
            require_measured=bool(mapping.get("require_measured", False)))


@dataclass
class TimeConfig:
    stamp_source: str = "header"   # 'header' | 'log'
    master_agent: Optional[str] = None
    rate_hz: float = 0.0           # 0 = use the master agent's own cloud stamps
    match_tolerance_ms: float = 60.0
    drop_incomplete_frames: bool = True
    start_offset_s: float = 0.0
    duration_s: float = 0.0        # 0 = to the end of the bag

    @staticmethod
    def parse(mapping: Optional[dict]) -> "TimeConfig":
        mapping = mapping or {}
        src = str(mapping.get("stamp_source", "header")).lower()
        if src not in ("header", "log"):
            raise ConfigError("time.stamp_source must be header or log")
        return TimeConfig(
            stamp_source=src,
            master_agent=mapping.get("master_agent"),
            rate_hz=float(mapping.get("rate_hz", 0.0)),
            match_tolerance_ms=float(mapping.get("match_tolerance_ms", 60.0)),
            drop_incomplete_frames=bool(mapping.get("drop_incomplete_frames", True)),
            start_offset_s=float(mapping.get("start_offset_s", 0.0)),
            duration_s=float(mapping.get("duration_s", 0.0)))


@dataclass
class OutputConfig:
    root: str
    split: str = "test"
    scenario_name: str = "scenario"
    frames_per_scenario: int = 0   # 0 = one scenario for the whole bag
    frame_stride: int = 1
    include_self_in_vehicles: bool = False
    write_provenance: bool = True

    @staticmethod
    def parse(mapping: dict) -> "OutputConfig":
        cfg = OutputConfig(
            root=os.path.expanduser(str(_get(mapping, "root", ctx="output"))),
            split=str(mapping.get("split", "test")),
            scenario_name=str(mapping.get("scenario_name", "scenario")),
            frames_per_scenario=int(mapping.get("frames_per_scenario", 0)),
            frame_stride=int(mapping.get("frame_stride", 1)),
            include_self_in_vehicles=bool(mapping.get("include_self_in_vehicles", False)),
            write_provenance=bool(mapping.get("write_provenance", True)))
        if cfg.frame_stride < 1:
            raise ConfigError("output.frame_stride must be >= 1")
        return cfg


@dataclass
class ConverterConfig:
    bag: str
    output: OutputConfig
    agents: List[AgentConfig]
    time: TimeConfig = field(default_factory=TimeConfig)
    clock: ClockConfig = field(default_factory=ClockConfig)
    world_frame: str = "world"

    @property
    def active_agents(self) -> List[AgentConfig]:
        return [a for a in self.agents if a.enabled]

    def agent_by_name(self, name: str) -> AgentConfig:
        for agent in self.agents:
            if agent.name == name:
                return agent
        raise ConfigError(f"no agent named {name!r}")

    def topics(self) -> Dict[str, List[str]]:
        """All topics the converter needs, grouped by role (for the bag reader)."""
        pose, cloud, info, camera = [], [], [], []
        for agent in self.active_agents:
            if agent.pose.topic:
                pose.append(agent.pose.topic)
            if agent.speed_source:
                pose.append(agent.speed_source)
            cloud.append(agent.cloud.topic)
            if agent.cloud.camera_info_topic:
                info.append(agent.cloud.camera_info_topic)
            for cam in agent.cameras:
                camera.append(cam.topic)
                if cam.camera_info_topic:
                    info.append(cam.camera_info_topic)
        dedup = lambda xs: sorted(set(xs))
        ntp = list(self.clock.ntp_topics.values()) if self.clock.enabled else []
        return {"pose": dedup(pose), "cloud": dedup(cloud),
                "camera_info": dedup(info), "camera": dedup(camera),
                "ntp": dedup(ntp)}


def load_config(path: str) -> ConverterConfig:
    """Parse and validate a converter config file."""
    with open(path, "r") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    agents = [AgentConfig.parse(a) for a in _get(raw, "agents", ctx="config")]
    if not agents:
        raise ConfigError("config: at least one agent is required")

    cfg = ConverterConfig(
        bag=os.path.expanduser(str(_get(raw, "bag", ctx="config"))),
        output=OutputConfig.parse(_get(raw, "output", ctx="config")),
        agents=agents,
        time=TimeConfig.parse(raw.get("time")),
        clock=ClockConfig.parse(raw.get("clock")),
        world_frame=str((raw.get("world") or {}).get("frame_id", "world")))

    validate(cfg)
    return cfg


def validate(cfg: ConverterConfig) -> None:
    """Structural checks that would otherwise surface as a broken dataset."""
    active = cfg.active_agents
    if not active:
        raise ConfigError("config: every agent is disabled")

    ids = [a.cav_id for a in active]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"config: duplicate cav ids {sorted(ids)}")
    names = [a.name for a in active]
    if len(set(names)) != len(names):
        raise ConfigError(f"config: duplicate agent names {sorted(names)}")

    cavs = [a for a in active if a.role == "cav"]
    if not cavs:
        raise ConfigError("config: at least one agent must be a cav — OpenCOOD "
                          "picks the ego from the non-negative ids")

    rsus = [a for a in active if a.role == "rsu"]
    if len(rsus) > 1:
        raise ConfigError(
            f"config: {len(rsus)} roadside units ({sorted(a.name for a in rsus)}). "
            f"Stock OpenCOOD moves only the *first* negative id to the end of the "
            f"agent list, so a second RSU can be picked as ego. Convert one RSU per "
            f"dataset, or give the extras a non-negative id and role: cav.")

    # OpenCOOD picks the smallest id as ego and never re-picks; say so out loud.
    ego = min(cavs, key=lambda a: a.cav_id)
    if not ego.required:
        raise ConfigError(f"config: agent {ego.name!r} is the ego (lowest cav id "
                          f"{ego.cav_id}) and cannot be optional")

    if cfg.clock.enabled:
        hosts = {a.clock_host for a in active}
        reference = cfg.clock.reference_host or ego_agent(cfg).clock_host
        if reference not in hosts:
            raise ConfigError(
                f"clock.reference_host={reference!r} is not a host of any enabled "
                f"agent (hosts: {sorted(hosts)})")
        unknown = set(cfg.clock.ntp_topics) - hosts
        if unknown:
            raise ConfigError(f"clock.ntp_topics names unknown hosts {sorted(unknown)}; "
                              f"agent hosts are {sorted(hosts)}")
        unmeasured = sorted(hosts - set(cfg.clock.ntp_topics) - {reference})
        if unmeasured and cfg.clock.require_measured:
            raise ConfigError(
                f"clock.require_measured is set but these hosts publish no NTP status "
                f"topic: {unmeasured}. Their offset would be estimated from delivery "
                f"floors alone, whose error is the transit asymmetry between links. "
                f"Add clock.ntp_topics entries, or clear require_measured and accept "
                f"the residual that every frame will report.")

    master = cfg.time.master_agent
    if master is not None and master not in names:
        raise ConfigError(f"time.master_agent={master!r} is not an enabled agent "
                          f"(enabled: {sorted(names)})")

    for agent in active:
        n_cams = len(agent.cameras)
        if n_cams > 4:
            raise ConfigError(f"agent[{agent.name}]: OPV2V has exactly 4 camera slots, "
                              f"got {n_cams}")
        if agent.obj.emit and agent.obj.object_id is None:
            agent.obj.object_id = 10000 + agent.cav_id if agent.cav_id >= 0 \
                else 20000 - agent.cav_id


def ego_agent(cfg: ConverterConfig) -> AgentConfig:
    """The agent OpenCOOD will treat as ego: the smallest non-negative cav id."""
    return min([a for a in cfg.active_agents if a.role == "cav"],
               key=lambda a: a.cav_id)

"""Converter configuration: schema, defaults and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


def _get(node: Dict[str, Any], key: str, default=None):
    value = node.get(key, default)
    return default if value is None else value


@dataclass
class ExtrinsicSpec:
    """Pose of a sensor expressed in its agent's pose frame (T_pose_sensor)."""

    source: str = "tf"            # tf | explicit | identity
    xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rpy_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    quat_xyzw: Optional[List[float]] = None
    frame: Optional[str] = None   # override the sensor frame used for TF lookup

    @staticmethod
    def parse(node) -> "ExtrinsicSpec":
        if node is None:
            return ExtrinsicSpec(source="tf")
        if isinstance(node, str):
            return ExtrinsicSpec(source=node)
        spec = ExtrinsicSpec(
            source=_get(node, "source", "explicit" if (
                "xyz" in node or "rpy_deg" in node or "quat_xyzw" in node)
                else "tf"),
            xyz=[float(v) for v in _get(node, "xyz", [0.0, 0.0, 0.0])],
            rpy_deg=[float(v) for v in _get(node, "rpy_deg", [0.0, 0.0, 0.0])],
            frame=node.get("frame"),
        )
        if node.get("quat_xyzw") is not None:
            spec.quat_xyzw = [float(v) for v in node["quat_xyzw"]]
        if spec.source not in ("tf", "explicit", "identity"):
            raise ValueError("extrinsic.source must be tf|explicit|identity")
        return spec


@dataclass
class CloudSpec:
    """A point-cloud stream that becomes (part of) the agent's ``.pcd``."""

    topic: str = ""
    source: str = "pointcloud"    # pointcloud | depth
    info_topic: Optional[str] = None
    max_age: float = 0.1
    required: bool = True
    intensity_field: Optional[str] = "intensity"
    intensity_scale: float = 1.0
    intensity_normalize: str = "scale"   # scale | percentile
    intensity_percentile: float = 99.0
    min_range: float = 0.0
    max_range: float = float("inf")
    depth_scale: float = 1e-3
    stride: int = 1
    to_body_frame: bool = True    # rotate optical-frame clouds into x-fwd/z-up
    extrinsic: ExtrinsicSpec = field(default_factory=ExtrinsicSpec)
    merge: List["CloudSpec"] = field(default_factory=list)

    @staticmethod
    def parse(node) -> Optional["CloudSpec"]:
        if node is None:
            return None
        spec = CloudSpec(
            topic=node["topic"],
            source=_get(node, "source", "pointcloud"),
            info_topic=node.get("info_topic"),
            max_age=float(_get(node, "max_age", 0.1)),
            required=bool(_get(node, "required", True)),
            intensity_field=node.get("intensity_field", "intensity"),
            intensity_scale=float(_get(node, "intensity_scale", 1.0)),
            intensity_normalize=str(_get(node, "intensity_normalize", "scale")),
            intensity_percentile=float(_get(node, "intensity_percentile", 99.0)),
            min_range=float(_get(node, "min_range", 0.0)),
            max_range=float(_get(node, "max_range", float("inf"))),
            depth_scale=float(_get(node, "depth_scale", 1e-3)),
            stride=int(_get(node, "stride", 1)),
            to_body_frame=bool(_get(node, "to_body_frame", True)),
            extrinsic=ExtrinsicSpec.parse(node.get("extrinsic")),
        )
        if spec.intensity_normalize not in ("scale", "percentile"):
            raise ValueError("intensity_normalize must be scale|percentile")
        if spec.source not in ("pointcloud", "depth"):
            raise ValueError("cloud.source must be pointcloud|depth")
        if spec.source == "depth" and not spec.info_topic:
            raise ValueError("depth cloud '%s' needs an info_topic" % spec.topic)
        spec.merge = [CloudSpec.parse(m) for m in _get(node, "merge", [])]
        return spec


@dataclass
class CameraSpec:
    index: int = 0
    image_topic: str = ""
    info_topic: Optional[str] = None
    max_age: float = 0.1
    required: bool = False
    extrinsic: ExtrinsicSpec = field(default_factory=ExtrinsicSpec)

    @staticmethod
    def parse(node, fallback_index: int) -> "CameraSpec":
        return CameraSpec(
            index=int(_get(node, "index", fallback_index)),
            image_topic=node["image_topic"],
            info_topic=node.get("info_topic"),
            max_age=float(_get(node, "max_age", 0.1)),
            required=bool(_get(node, "required", False)),
            extrinsic=ExtrinsicSpec.parse(node.get("extrinsic")),
        )


@dataclass
class ObjectSpec:
    """How this agent appears in the other agents' ``vehicles`` ground truth."""

    publish: bool = True
    extent: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.4])
    center: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    class_name: str = "robot"

    @staticmethod
    def parse(node) -> "ObjectSpec":
        if node is None:
            return ObjectSpec(publish=False)
        return ObjectSpec(
            publish=bool(_get(node, "publish", True)),
            extent=[float(v) for v in _get(node, "extent", [0.4, 0.3, 0.4])],
            center=[float(v) for v in _get(node, "center", [0.0, 0.0, 0.0])],
            class_name=str(_get(node, "class_name", "robot")),
        )


@dataclass
class ExtraSpec:
    """Auxiliary telemetry time-joined into every frame's yaml (wifi, ntp, ...)."""

    key: str
    topic: str
    max_age: float = 2.0
    fields: Optional[List[str]] = None

    @staticmethod
    def parse(node) -> "ExtraSpec":
        return ExtraSpec(
            key=node["key"],
            topic=node["topic"],
            max_age=float(_get(node, "max_age", 2.0)),
            fields=node.get("fields"),
        )


@dataclass
class PoseSpec:
    source: str = "topic"          # topic | static
    topic: Optional[str] = None
    frame: Optional[str] = None    # frame the pose describes (TF lookups start here)
    max_gap: float = 0.2
    xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rpy_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    quat_xyzw: Optional[List[float]] = None

    @staticmethod
    def parse(node) -> "PoseSpec":
        if node is None:
            raise ValueError("every agent needs a 'pose' block")
        spec = PoseSpec(
            source=_get(node, "source", "topic"),
            topic=node.get("topic"),
            frame=node.get("frame"),
            max_gap=float(_get(node, "max_gap", 0.2)),
            xyz=[float(v) for v in _get(node, "xyz", [0.0, 0.0, 0.0])],
            rpy_deg=[float(v) for v in _get(node, "rpy_deg", [0.0, 0.0, 0.0])],
        )
        if node.get("quat_xyzw") is not None:
            spec.quat_xyzw = [float(v) for v in node["quat_xyzw"]]
        if spec.source == "topic" and not spec.topic:
            raise ValueError("pose.source=topic requires pose.topic")
        if spec.source not in ("topic", "static"):
            raise ValueError("pose.source must be topic|static")
        return spec


@dataclass
class AgentSpec:
    id: int
    name: str
    role: str = "cav"              # cav | rsu
    required: bool = True
    pose: PoseSpec = field(default_factory=PoseSpec)
    cloud: Optional[CloudSpec] = None
    cameras: List[CameraSpec] = field(default_factory=list)
    obj: ObjectSpec = field(default_factory=ObjectSpec)
    extras: List[ExtraSpec] = field(default_factory=list)

    @property
    def folder(self) -> str:
        return str(self.id)

    @staticmethod
    def parse(node) -> "AgentSpec":
        agent = AgentSpec(
            id=int(node["id"]),
            name=str(_get(node, "name", node["id"])),
            role=_get(node, "role", "cav"),
            required=bool(_get(node, "required", True)),
            pose=PoseSpec.parse(node.get("pose")),
            cloud=CloudSpec.parse(node.get("lidar", node.get("cloud"))),
            obj=ObjectSpec.parse(node.get("object")),
        )
        agent.cameras = [CameraSpec.parse(c, i)
                         for i, c in enumerate(_get(node, "cameras", []))]
        agent.extras = [ExtraSpec.parse(e) for e in _get(node, "extras", [])]
        if agent.role not in ("cav", "rsu"):
            raise ValueError("agent.role must be cav|rsu")
        if agent.role == "rsu" and agent.id >= 0:
            raise ValueError(
                "RSU agents need a negative id (OpenCOOD never picks them as "
                "ego); agent '%s' has id %d" % (agent.name, agent.id))
        return agent


@dataclass
class Config:
    name: str = "dataset"
    sample_rate_hz: float = 10.0
    scenario_seconds: Optional[float] = None
    timestamp_step: int = 2
    start_index: int = 0
    split: str = "train"
    splits: Optional[Dict[str, float]] = None
    pcd_binary: bool = True
    write_images: bool = True
    use_tf_static: bool = True
    tf_topics: List[str] = field(default_factory=lambda: ["/tf_static"])
    time_source: str = "header"     # header | log
    include_ego_as_object: bool = True
    drop_incomplete_frames: bool = True
    min_frames_per_scenario: int = 5
    max_frames: Optional[int] = None
    agents: List[AgentSpec] = field(default_factory=list)

    @property
    def sample_period(self) -> float:
        return 1.0 / float(self.sample_rate_hz)

    @property
    def ego(self) -> AgentSpec:
        cavs = [a for a in self.agents if a.id >= 0]
        return sorted(cavs, key=lambda a: a.folder)[0]

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r") as handle:
            raw = yaml.safe_load(handle) or {}
        return Config.from_dict(raw)

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "Config":
        dataset = _get(raw, "dataset", {})
        cfg = Config(
            name=str(_get(dataset, "name", "dataset")),
            sample_rate_hz=float(_get(dataset, "sample_rate_hz", 10.0)),
            scenario_seconds=(float(dataset["scenario_seconds"])
                              if dataset.get("scenario_seconds") else None),
            timestamp_step=int(_get(dataset, "timestamp_step", 2)),
            start_index=int(_get(dataset, "start_index", 0)),
            split=str(_get(dataset, "split", "train")),
            splits=dataset.get("splits"),
            pcd_binary=bool(_get(dataset, "pcd_binary", True)),
            write_images=bool(_get(dataset, "write_images", True)),
            use_tf_static=bool(_get(dataset, "use_tf_static", True)),
            tf_topics=list(_get(dataset, "tf_topics", ["/tf_static"])),
            time_source=str(_get(dataset, "time_source", "header")),
            include_ego_as_object=bool(
                _get(dataset, "include_ego_as_object", True)),
            drop_incomplete_frames=bool(
                _get(dataset, "drop_incomplete_frames", True)),
            min_frames_per_scenario=int(
                _get(dataset, "min_frames_per_scenario", 5)),
            max_frames=(int(dataset["max_frames"])
                        if dataset.get("max_frames") else None),
        )
        cfg.agents = [AgentSpec.parse(a) for a in _get(raw, "agents", [])]
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.agents:
            raise ValueError("config defines no agents")
        ids = [a.id for a in self.agents]
        if len(set(ids)) != len(ids):
            raise ValueError("agent ids must be unique: %s" % ids)
        if not [a for a in self.agents if a.id >= 0]:
            raise ValueError("at least one agent must have a non-negative id "
                             "so OpenCOOD has an ego vehicle")
        for agent in self.agents:
            if agent.cloud is None:
                raise ValueError(
                    "agent '%s' has no lidar/cloud block; OpenCOOD requires a "
                    "point cloud per agent (use a depth-derived cloud for "
                    "RGBD-only agents)" % agent.name)
        if self.time_source not in ("header", "log"):
            raise ValueError("dataset.time_source must be header|log")
        if self.splits:
            total = sum(float(v) for v in self.splits.values())
            if total <= 0:
                raise ValueError("dataset.splits weights must be positive")


def default_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "configs", "mirc_coop2.yaml")

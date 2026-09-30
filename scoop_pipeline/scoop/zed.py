# -*- coding: utf-8 -*-
"""
ZED SVO2 -> rosbag2, on the capture clock.

Reading an SVO2 needs the ZED SDK, which comes with the ZED ROS 2 wrapper, so
the SVO is played through the wrapper and recorded (``scripts/run_zed.sh``,
where ROS + zed_wrapper are: this machine or the isaac_ros container):

    ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i svo_path:=<svo>
        (svo.use_svo_timestamps: true, svo.svo_loop: false, svo.svo_realtime: false)
    ros2 bag record <topics>            stopped at "SVO reached the end"

With ``use_svo_timestamps`` the header stamps are the SVO's capture times --
the ZED SDK stamps frames with the recording PC's clock, the clock of the raw
bag's log times -- but ``ros2 bag record`` logs each message at replay time.
:func:`restamp_bag` rewrites the recording into the final bag:

* log and publish time = header stamp;
* topics without a header (or with a zero stamp) move by the offset
  (stamp - log time) of the latest stamped message, which follows the replay
  speed;
* topics renamed (``remap``), QoS kept, written in log-time order.

The result sits on the same timeline as the retimed Ouster bag.

    from scoop import zed
    zed.svo_to_bag(svo, out_dir, topics=[...], remap=zed.prefix_remap(topics,
                   "/zed/zed_node", "/mobile_1/zed"))
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from . import bag, replay
from .bag import BagError, TopicSchema
from .bagwrite import BagWriter, write_in_order
from .retime import header_stamp_ns

__all__ = ["REORDER_S", "SVO_PARAMS", "find_svo", "has_header", "prefix_remap",
           "load_wrapper_config", "wrapper_params", "record_svo", "restamp_bag", "svo_to_bag", "RestampResult"]

REORDER_S = 5.0          # replay latency differs per topic (depth, point cloud)
SVO_PARAMS = {"use_svo_timestamps": True, "svo_loop": False}
RUN_ZED = Path(__file__).resolve().parents[1] / "scripts" / "run_zed.sh"
HEADER_TYPES = {"Header", "std_msgs/Header", "std_msgs/msg/Header"}


def find_svo(folder) -> Optional[Path]:
    """The .svo2 (or .svo) file in ``folder``, None if there is none."""
    found = sorted(p for p in Path(folder).iterdir()
                   if p.is_file() and p.suffix.lower() in (".svo2", ".svo"))
    if len(found) > 1:
        raise BagError(f"several SVO files in {folder}: {[p.name for p in found]}")
    return found[0] if found else None


def has_header(schema: TopicSchema) -> bool:
    """True when the message type's first field is a std_msgs/Header."""
    if schema.encoding != "ros2msg":
        return False
    text = bytes(schema.data).decode("utf-8", "replace")
    for line in text.split("====")[0].splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "=" in line:                       # blank, comment, constant
            continue
        return line.split()[0] in HEADER_TYPES
    return False


def prefix_remap(topics: Iterable[str], old: str, new: str) -> Dict[str, str]:
    """``{t: new + rest}`` for every topic ``old + rest``, e.g.
    /zed/zed_node/left/camera_info -> /mobile_1/zed/left/camera_info."""
    old, new = old.rstrip("/"), new.rstrip("/")
    return {t: new + t[len(old):] for t in topics
            if t == old or t.startswith(old + "/")}


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_wrapper_config(paths) -> dict:
    """The ``ros__parameters`` of zed_wrapper config files (e.g. the robot's
    common_stereo.yaml and zed2i.yaml), merged in order."""
    out: dict = {}
    for p in paths or ():
        with open(Path(p).expanduser()) as fh:
            doc = yaml.safe_load(fh) or {}
        for node in doc.values():                 # "/**" or a node name
            out = _merge(out, (node or {}).get("ros__parameters", {}))
    return out


def wrapper_params(params: Optional[dict] = None, realtime: bool = False,
                   base: Optional[dict] = None) -> dict:
    """The wrapper's ``ros_params_override_path`` file: ``base`` (the robot's
    wrapper config, :func:`load_wrapper_config`), then SVO timestamps, no
    loop, every frame (``realtime`` False), then ``params`` (nested, e.g.
    ``{"depth": {"depth_mode": "NEURAL"}}``). The SVO keys always win over
    ``base``, since the restamp relies on them."""
    ros = _merge(base or {}, {"svo": {**SVO_PARAMS, "svo_realtime": bool(realtime)}})
    ros = _merge(ros, params or {})
    return {"/**": {"ros__parameters": ros}}


def record_svo(svo, record_dir, topics: List[str], params_file, camera_model: str = "zed2i",
               script=None, log=print) -> Path:
    """Play ``svo`` through the ZED wrapper and record ``topics`` into
    ``record_dir`` (a new rosbag2 folder), via ``scripts/run_zed.sh``."""
    script = Path(script or RUN_ZED)
    if not topics:
        raise ValueError("no ZED topics to record")
    env = dict(os.environ, ZED_CAMERA_MODEL=camera_model)
    log(f"    recording {len(topics)} topics from {Path(svo).name} ({camera_model})")
    code = subprocess.call([str(script), str(svo), str(record_dir), str(params_file),
                            *topics], env=env)
    if code != 0 or not (Path(record_dir) / "metadata.yaml").is_file():
        raise BagError(f"the ZED recording failed (exit {code}); "
                       f"what was recorded is left in {record_dir}")
    return Path(record_dir)


@dataclass
class RestampResult:
    bag_dir: Path
    counts: Dict[str, int]                              # messages per output topic
    moved: Dict[str, int] = field(default_factory=dict)  # by offset, no usable stamp
    t_min: int = 0
    t_max: int = 0


def _stamp(schema: TopicSchema, data) -> int:
    if not has_header(schema) or len(data) < 12:
        return 0
    return max(header_stamp_ns(data), 0)


def restamp_bag(recorded, out_dir, remap: Optional[Dict[str, str]] = None,
                reorder_s: float = REORDER_S, log=print) -> RestampResult:
    """``recorded`` (header stamps = capture time, log times = replay time)
    -> ``out_dir`` with log and publish time = header stamp, topics renamed
    by ``remap``. See the module docstring."""
    reader = bag.open_bag(recorded)
    have = reader.topics().keys()
    remap = {k: v for k, v in (remap or {}).items() if k in have}
    copies = replay.plan_copies(have, remap, copy_all=True)
    qos = bag.topic_qos(reader)
    schemas = {}

    def schema_of(sc, ch):
        s = schemas.get(ch.topic)
        if s is None:
            s = schemas[ch.topic] = TopicSchema(sc.name, sc.encoding, bytes(sc.data),
                                                ch.message_encoding)
        return s

    offset = None                            # stamp - log of the latest stamped message
    for _, sc, ch, msg in bag.iter_mcap_records(reader):
        st = _stamp(schema_of(sc, ch), msg.data)
        if st:
            offset = st - msg.log_time
            break
    if offset is None:
        raise BagError(f"no message in {recorded} has a header stamp")

    moved: Dict[str, int] = {}

    def records():
        nonlocal offset
        for _, sc, ch, msg in bag.iter_mcap_records(reader):
            schema = schema_of(sc, ch)
            st = _stamp(schema, msg.data)
            if st:
                offset = st - msg.log_time
                log_ns = pub_ns = st
            else:
                log_ns, pub_ns = msg.log_time + offset, msg.publish_time + offset
                moved[copies[ch.topic]] = moved.get(copies[ch.topic], 0) + 1
            yield (log_ns, copies[ch.topic], schema, pub_ns, msg.sequence,
                   dict(ch.metadata), msg.data)

    with BagWriter(out_dir, library="scoop.zed") as w:
        for src, dst in copies.items():
            w.set_qos(dst, qos.get(src, ""))
        try:
            write_in_order(w, records(), reorder_s)
        except ValueError as e:
            raise BagError(f"{e}; pass a larger reorder_s") from None
    res = RestampResult(w.dir, dict(w.counts), moved, int(w.t_min or 0), int(w.t_max or 0))
    span = max((res.t_max - res.t_min) * 1e-9, 1e-9)
    log(f"wrote {w.n} messages -> {w.dir} ({span:.1f} s)")
    for t, c in sorted(res.counts.items()):
        note = f"  ({moved[t]} without a stamp, moved by offset)" if t in moved else ""
        log(f"    {t:45s} {c:7d}  {c / span:6.1f} Hz{note}")
    return res


def svo_to_bag(svo, out_dir, topics: List[str], remap: Optional[Dict[str, str]] = None,
               camera_model: str = "zed2i", params: Optional[dict] = None,
               realtime: bool = False, wrapper_config=(),
               reorder_s: float = REORDER_S, script=None,
               keep_record: bool = False, log=print) -> RestampResult:
    """SVO -> rosbag2 folder ``out_dir`` (must not exist), on capture time.
    ``wrapper_config``: the zed_wrapper config files to replay with (see
    :func:`wrapper_params`).

    The replay is recorded into ``<out_dir>.record`` first and restamped into
    ``out_dir``; the wrapper logs and the parameters used end up in
    ``out_dir/zed_logs/``. The intermediate recording is deleted unless
    ``keep_record``; after a failure it is left for a look."""
    out_dir = Path(out_dir)
    record_dir = out_dir.with_name(out_dir.name + ".record")
    params_file = out_dir.with_name(out_dir.name + ".params.yaml")
    for p in (out_dir, record_dir):
        if p.exists():
            raise BagError(f"{p} exists; remove it first")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(params_file, "w") as fh:
        yaml.safe_dump(wrapper_params(params, realtime, load_wrapper_config(wrapper_config)),
                       fh, sort_keys=False)
    record_svo(svo, record_dir, topics, params_file, camera_model, script, log)
    missing = sorted(set(topics) - set(bag.open_bag(record_dir).topics()))
    if missing:
        log(f"    WARNING: never published, so not in the bag: {missing}")
    res = restamp_bag(record_dir, out_dir, remap, reorder_s, log)
    logs = out_dir / "zed_logs"
    if (record_dir / "zed_logs").is_dir():
        shutil.move(str(record_dir / "zed_logs"), str(logs))
    logs.mkdir(exist_ok=True)
    shutil.move(str(params_file), str(logs / "params.yaml"))
    if not keep_record:
        shutil.rmtree(record_dir)
    return res

# -*- coding: utf-8 -*-
"""
Writing rosbag2 bags (MCAP + metadata.yaml) without a ROS install.

The output is what ``ros2 bag record`` leaves on Humble: one MCAP file with
ros2msg schemas and CDR messages, plus the ``metadata.yaml`` that
``ros2 bag info/play``, GLIM and ``retime_bag.py`` read. Two ways in:

* :meth:`BagWriter.write` -- a message object (see :mod:`scoop.rosmsg`);
* :meth:`BagWriter.write_raw` -- bytes copied from another bag, with that
  topic's own schema (:func:`scoop.bag.topic_schemas`), so custom types
  pass through untouched.

    with BagWriter(out_dir) as w:
        w.write("/points", rosmsg.pointcloud2(...), log_ns)
        w.write_raw("/zed/imu", schemas["/zed/zed_node/imu/data"], payload, log_ns)

Messages should arrive in log-time order, as a recorder writes them.
"""
from __future__ import annotations

import heapq
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml

from . import rosmsg
from .bag import TopicSchema

__all__ = ["BagWriter", "write_in_order"]


class BagWriter:
    def __init__(self, out_dir, library: str = "scoop", metadata_template=None):
        """``metadata_template``: another bag's metadata.yaml (path or dict).
        Its structure -- version, QoS profiles, topic order -- is kept and
        only the contents (files, counts, times) are rewritten, the way
        ``retime_bag.py`` did."""
        from mcap.writer import CompressionType, Writer
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=False)      # never overwrite a bag
        self.name = f"{self.dir.name}_0.mcap"
        self._fh = open(self.dir / self.name, "wb")
        self._w = Writer(self._fh, compression=CompressionType.NONE)
        self._w.start(profile="ros2", library=library)
        self._channels: Dict[str, int] = {}
        self.types: Dict[str, str] = {}
        self.qos: Dict[str, str] = {}             # topic -> offered_qos_profiles
        self.counts: Dict[str, int] = {}
        self.t_min: Optional[int] = None
        self.t_max: Optional[int] = None
        self.n = 0
        self._closed = False
        if metadata_template is not None and not isinstance(metadata_template, dict):
            with open(metadata_template) as fh:
                metadata_template = yaml.safe_load(fh)
        self._template = metadata_template

    # -- channels ------------------------------------------------------------
    def _channel(self, topic: str, schema: TopicSchema, metadata=None) -> int:
        ch = self._channels.get(topic)
        if ch is None:
            sid = self._w.register_schema(schema.name, schema.encoding, schema.data)
            ch = self._w.register_channel(topic, schema.message_encoding, sid,
                                          metadata=dict(metadata or {}))
            self._channels[topic] = ch
            self.types[topic] = schema.name
            self.counts[topic] = 0
        elif self.types[topic] != schema.name:
            raise ValueError(f"{topic}: {schema.name} written to a "
                             f"{self.types[topic]} topic")
        return ch

    def _add(self, ch: int, topic: str, data: bytes, log_ns: int,
             publish_ns: Optional[int] = None, sequence: Optional[int] = None):
        log_ns = int(log_ns)
        self._w.add_message(ch, log_time=log_ns,
                            publish_time=log_ns if publish_ns is None else int(publish_ns),
                            data=data,
                            sequence=self.counts[topic] if sequence is None else int(sequence))
        self.counts[topic] += 1
        self.n += 1
        self.t_min = log_ns if self.t_min is None else min(self.t_min, log_ns)
        self.t_max = log_ns if self.t_max is None else max(self.t_max, log_ns)

    # -- writing -------------------------------------------------------------
    def write(self, topic: str, msg, log_ns: int):
        """A message object; its schema is generated from the type."""
        name = msg.__msgtype__
        if topic not in self._channels:
            schema = TopicSchema(name, "ros2msg", rosmsg.msgdef(name).encode(), "cdr")
        else:
            schema = TopicSchema(name, "", b"", "")
        self._add(self._channel(topic, schema), topic, rosmsg.serialize(msg), log_ns)

    def set_qos(self, topic: str, profiles: str):
        """The topic's ``offered_qos_profiles`` for metadata.yaml, as the
        source bag had it (``scoop.bag.topic_qos``)."""
        self.qos[topic] = profiles or ""

    def write_raw(self, topic: str, schema: TopicSchema, payload: bytes, log_ns: int,
                  publish_ns: Optional[int] = None, sequence: Optional[int] = None,
                  channel_metadata=None):
        """Serialized bytes, copied verbatim under ``topic``. ``publish_ns``,
        ``sequence`` and ``channel_metadata`` default to the log time, a
        running count and none; pass the source's to copy a record exactly."""
        self._add(self._channel(topic, schema, channel_metadata), topic, bytes(payload),
                  log_ns, publish_ns, sequence)

    # -- closing -------------------------------------------------------------
    def close(self) -> Path:
        if self._closed:
            return self.dir
        self._closed = True
        self._w.finish()
        self._fh.close()
        t0, t1 = int(self.t_min or 0), int(self.t_max or 0)
        if self._template is not None:
            meta = self._template
            info = meta["rosbag2_bagfile_information"]
            info["relative_file_paths"] = [self.name]
            info["message_count"] = self.n
            info["starting_time"] = {"nanoseconds_since_epoch": t0}
            info["duration"] = {"nanoseconds": t1 - t0}
            listed = set()
            for te in info.get("topics_with_message_count", []):
                name = te.get("topic_metadata", {}).get("name")
                te["message_count"] = self.counts.get(name, 0)
                if self.qos.get(name):
                    te["topic_metadata"]["offered_qos_profiles"] = self.qos[name]
                listed.add(name)
            for t in self._channels:
                if t not in listed:
                    info.setdefault("topics_with_message_count", []).append(
                        {"topic_metadata": {"name": t, "type": self.types[t],
                                            "serialization_format": "cdr",
                                            "offered_qos_profiles": self.qos.get(t, "")},
                         "message_count": self.counts[t]})
            info["files"] = [{"path": self.name,
                              "starting_time": {"nanoseconds_since_epoch": t0},
                              "duration": {"nanoseconds": t1 - t0},
                              "message_count": self.n}]
            with open(self.dir / "metadata.yaml", "w") as fh:
                yaml.safe_dump(meta, fh, sort_keys=False)
            return self.dir
        meta = {"rosbag2_bagfile_information": {
            "version": 5,
            "storage_identifier": "mcap",
            "duration": {"nanoseconds": t1 - t0},
            "starting_time": {"nanoseconds_since_epoch": t0},
            "message_count": self.n,
            "topics_with_message_count": [
                {"topic_metadata": {"name": t, "type": self.types[t],
                                    "serialization_format": "cdr",
                                    "offered_qos_profiles": self.qos.get(t, "")},
                 "message_count": self.counts[t]} for t in self._channels],
            "compression_format": "",
            "compression_mode": "",
            "relative_file_paths": [self.name],
            "files": [{"path": self.name,
                       "starting_time": {"nanoseconds_since_epoch": t0},
                       "duration": {"nanoseconds": t1 - t0},
                       "message_count": self.n}]}}
        with open(self.dir / "metadata.yaml", "w") as fh:
            yaml.safe_dump(meta, fh, sort_keys=False)
        return self.dir

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def write_in_order(w: BagWriter, records: Iterable, reorder_s: float) -> None:
    """Write ``records`` -- ``(log_ns, topic, schema, publish_ns, sequence,
    channel_metadata, payload)`` that are out of log-time order by at most
    ``reorder_s`` -- in non-decreasing log-time order, through a bounded
    reorder buffer. Raises if a record arrives later than that."""
    horizon = int(reorder_s * 1e9)
    heap = []
    watermark = -1
    last = -1

    def emit(item):
        nonlocal last
        log_ns, _, topic, schema, pub, seq, meta, data = item
        if log_ns < last:
            raise ValueError(f"reorder window {reorder_s} s too small: "
                             f"{topic} would go back in time")
        last = log_ns
        w.write_raw(topic, schema, data, log_ns, publish_ns=pub, sequence=seq,
                    channel_metadata=meta)

    for i, (log_ns, topic, schema, pub, seq, meta, data) in enumerate(records):
        heapq.heappush(heap, (log_ns, i, topic, schema, pub, seq, meta, data))
        watermark = max(watermark, log_ns)
        while heap and heap[0][0] <= watermark - horizon:
            emit(heapq.heappop(heap))
    while heap:
        emit(heapq.heappop(heap))

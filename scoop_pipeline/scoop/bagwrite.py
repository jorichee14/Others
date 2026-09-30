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

from pathlib import Path
from typing import Dict, Optional

import yaml

from . import rosmsg
from .bag import TopicSchema

__all__ = ["BagWriter"]


class BagWriter:
    def __init__(self, out_dir, library: str = "scoop"):
        from mcap.writer import CompressionType, Writer
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=False)      # never overwrite a bag
        self.name = f"{self.dir.name}_0.mcap"
        self._fh = open(self.dir / self.name, "wb")
        self._w = Writer(self._fh, compression=CompressionType.NONE)
        self._w.start(profile="ros2", library=library)
        self._channels: Dict[str, int] = {}
        self.types: Dict[str, str] = {}
        self.counts: Dict[str, int] = {}
        self.t_min: Optional[int] = None
        self.t_max: Optional[int] = None
        self.n = 0
        self._closed = False

    # -- channels ------------------------------------------------------------
    def _channel(self, topic: str, schema: TopicSchema) -> int:
        ch = self._channels.get(topic)
        if ch is None:
            sid = self._w.register_schema(schema.name, schema.encoding, schema.data)
            ch = self._w.register_channel(topic, schema.message_encoding, sid)
            self._channels[topic] = ch
            self.types[topic] = schema.name
            self.counts[topic] = 0
        elif self.types[topic] != schema.name:
            raise ValueError(f"{topic}: {schema.name} written to a "
                             f"{self.types[topic]} topic")
        return ch

    def _add(self, ch: int, topic: str, data: bytes, log_ns: int):
        log_ns = int(log_ns)
        self._w.add_message(ch, log_time=log_ns, publish_time=log_ns, data=data,
                            sequence=self.counts[topic])
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

    def write_raw(self, topic: str, schema: TopicSchema, payload: bytes, log_ns: int):
        """Serialized bytes, copied verbatim under ``topic``."""
        self._add(self._channel(topic, schema), topic, bytes(payload), log_ns)

    # -- closing -------------------------------------------------------------
    def close(self) -> Path:
        if self._closed:
            return self.dir
        self._closed = True
        self._w.finish()
        self._fh.close()
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

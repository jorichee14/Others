# -*- coding: utf-8 -*-
"""
Reading a rosbag2 recording without a ROS installation.

Two backends, picked from the storage format:

* **mcap** (``mcap`` + ``mcap-ros2-support``) — decodes straight from the message
  definitions embedded in the file, so custom interfaces in the bag
  (``wifi_csi_msgs``, ``ouster_sensor_msgs``, ...) never need to be on the
  ``AMENT_PREFIX_PATH``.
* **sqlite3** (``rosbags``) — for ``.db3`` recordings, using that library's own
  typestore.

Decoding is lazy.  The converter needs two passes over the bag — one to learn
which messages exist and when, one to convert the chosen ones — and decoding
every LiDAR sweep twice is the difference between minutes and an hour.  So the
indexing pass reads the header stamp straight out of the CDR payload
(:func:`stamp_from_cdr`): in ROS 2 every stamped message begins with its
``std_msgs/Header``, whose ``sec``/``nanosec`` sit at fixed offsets right after
the 4-byte encapsulation prefix.  Only the messages that survive frame selection
are ever deserialised.
"""

from __future__ import annotations

import glob
import os
import struct
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import yaml


@dataclass
class TopicInfo:
    name: str
    msgtype: str
    count: int = 0


@dataclass
class TopicIndex:
    """Per-topic message stamps gathered by the indexing pass."""
    topic: str
    msgtype: str = "?"
    header_stamps: List[int] = field(default_factory=list)
    log_times: List[int] = field(default_factory=list)
    headerless: int = 0

    def stamps(self, source: str = "header") -> List[int]:
        return self.log_times if source == "log" else self.header_stamps


class BagError(RuntimeError):
    pass


def resolve_bag_files(path: str) -> Tuple[List[str], str]:
    """Resolve a bag path to ``(files, storage)`` where storage is mcap or sqlite3.

    Accepts a rosbag2 directory (with ``metadata.yaml``), a bare directory of
    split files, or a single ``.mcap``/``.db3`` file.
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise BagError(f"bag path does not exist: {path}")

    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".mcap":
            return [path], "mcap"
        if ext in (".db3", ".sqlite3"):
            return [path], "sqlite3"
        raise BagError(f"unrecognised bag file extension: {path}")

    meta_path = os.path.join(path, "metadata.yaml")
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as handle:
            meta = yaml.safe_load(handle) or {}
        info = meta.get("rosbag2_bagfile_information", {})
        storage = str(info.get("storage_identifier", "")).lower() or None
        rel = info.get("relative_file_paths") or []
        files = [f for f in (os.path.join(path, r) for r in rel) if os.path.isfile(f)]
        if files:
            if storage not in ("mcap", "sqlite3"):
                storage = "mcap" if files[0].endswith(".mcap") else "sqlite3"
            return sorted(files), storage

    mcaps = sorted(glob.glob(os.path.join(path, "*.mcap")))
    if mcaps:
        return mcaps, "mcap"
    db3s = sorted(glob.glob(os.path.join(path, "*.db3")))
    if db3s:
        return db3s, "sqlite3"
    raise BagError(f"no .mcap or .db3 files found under {path}")


def stamp_from_cdr(payload: bytes) -> Optional[int]:
    """Header stamp (ns) read directly from a CDR payload, without deserialising.

    Layout of any ROS 2 message whose first field is a ``std_msgs/Header``::

        [0:2]  CDR representation id (byte 1: 0 = big endian, 1 = little endian)
        [2:4]  options
        [4:8]  header.stamp.sec     (int32)
        [8:12] header.stamp.nanosec (uint32)

    Returns ``None`` for payloads too short to contain a header or whose stamp is
    unset, so the caller can fall back to log time.
    """
    if payload is None or len(payload) < 12:
        return None
    little_endian = bool(payload[1] & 0x01)
    fmt = "<iI" if little_endian else ">iI"
    sec, nanosec = struct.unpack_from(fmt, payload, 4)
    if sec < 0 or nanosec >= 1_000_000_000:
        return None                       # not a header after all
    value = sec * 1_000_000_000 + nanosec
    return value if value > 0 else None


def header_stamp_ns(msg, fallback_ns: int) -> int:
    """Header stamp of a *decoded* message in ns, or ``fallback_ns``."""
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None) if header is not None else None
    if stamp is None:
        return int(fallback_ns)
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", getattr(stamp, "nsec", None))
    if sec is None or nanosec is None:
        return int(fallback_ns)
    value = int(sec) * 1_000_000_000 + int(nanosec)
    return value if value > 0 else int(fallback_ns)


class BagReader:
    """Uniform, lazily-decoding read access to a rosbag2 recording."""

    def __init__(self, path: str, stamp_source: str = "header"):
        if stamp_source not in ("header", "log"):
            raise BagError("stamp_source must be 'header' or 'log'")
        self.path = os.path.expanduser(path)
        self.stamp_source = stamp_source
        self.files, self.storage = resolve_bag_files(self.path)
        self._topics: Optional[Dict[str, TopicInfo]] = None

    # ------------------------------------------------------------------ summary

    def topics(self) -> Dict[str, TopicInfo]:
        """Topic name -> ``TopicInfo``, read from the file index (no decoding)."""
        if self._topics is not None:
            return self._topics

        topics: Dict[str, TopicInfo] = {}
        if self.storage == "mcap":
            from mcap.reader import make_reader
            for file_path in self.files:
                with open(file_path, "rb") as handle:
                    summary = make_reader(handle).get_summary()
                    if summary is None:
                        raise BagError(
                            f"{file_path}: no summary section — the recording was not "
                            f"closed cleanly. Rebuild the index with `mcap recover`.")
                    counts = getattr(summary.statistics, "channel_message_counts", {}) \
                        if summary.statistics else {}
                    for channel_id, channel in summary.channels.items():
                        schema = summary.schemas.get(channel.schema_id)
                        info = topics.setdefault(
                            channel.topic,
                            TopicInfo(channel.topic, schema.name if schema else "?", 0))
                        info.count += int(counts.get(channel_id, 0))
        else:
            from pathlib import Path
            from rosbags.highlevel import AnyReader
            with AnyReader([Path(f) for f in self.files]) as reader:
                for connection in reader.connections:
                    info = topics.setdefault(
                        connection.topic,
                        TopicInfo(connection.topic, connection.msgtype, 0))
                    info.count += int(getattr(connection, "msgcount", 0) or 0)

        self._topics = topics
        return topics

    def time_range(self) -> Tuple[int, int]:
        """``(start_ns, end_ns)`` log-time span of the recording."""
        if self.storage == "mcap":
            from mcap.reader import make_reader
            starts, ends = [], []
            for file_path in self.files:
                with open(file_path, "rb") as handle:
                    summary = make_reader(handle).get_summary()
                    stats = summary.statistics if summary else None
                    if stats and stats.message_start_time:
                        starts.append(int(stats.message_start_time))
                        ends.append(int(stats.message_end_time))
            if starts:
                return min(starts), max(ends)
        from pathlib import Path
        from rosbags.highlevel import AnyReader
        with AnyReader([Path(f) for f in self.files]) as reader:
            return int(reader.start_time), int(reader.end_time)

    # ------------------------------------------------------------------- raw io

    def iter_raw(self, topics: Optional[Iterable[str]] = None,
                 start_ns: Optional[int] = None, end_ns: Optional[int] = None
                 ) -> Iterator[Tuple[str, int, bytes, Callable[[bytes], object]]]:
        """Yield ``(topic, log_time_ns, payload, decode)`` in log order.

        ``decode`` deserialises that payload on demand; callers that only need
        stamps never invoke it.
        """
        wanted = sorted(set(topics)) if topics is not None else None
        if self.storage == "mcap":
            yield from self._iter_raw_mcap(wanted, start_ns, end_ns)
        else:
            yield from self._iter_raw_rosbags(wanted, start_ns, end_ns)

    def _iter_raw_mcap(self, topics, start_ns, end_ns):
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory

        factory = DecoderFactory()
        for file_path in self.files:
            decoders: Dict[int, Callable[[bytes], object]] = {}
            with open(file_path, "rb") as handle:
                reader = make_reader(handle)
                for schema, channel, message in reader.iter_messages(
                        topics=topics, start_time=start_ns, end_time=end_ns):
                    decoder = decoders.get(channel.schema_id)
                    if decoder is None:
                        decoder = factory.decoder_for(channel.message_encoding, schema)
                        if decoder is None:
                            raise BagError(
                                f"{channel.topic}: no decoder for message encoding "
                                f"{channel.message_encoding!r} / schema "
                                f"{getattr(schema, 'name', '?')}")
                        decoders[channel.schema_id] = decoder
                    yield channel.topic, int(message.log_time), message.data, decoder

    def _iter_raw_rosbags(self, topics, start_ns, end_ns):
        from pathlib import Path
        from rosbags.highlevel import AnyReader
        with AnyReader([Path(f) for f in self.files]) as reader:
            connections = [c for c in reader.connections
                           if topics is None or c.topic in topics]
            for connection, timestamp, raw in reader.messages(
                    connections=connections, start=start_ns, stop=end_ns):
                msgtype = connection.msgtype
                yield (connection.topic, int(timestamp), raw,
                       lambda payload, _t=msgtype: reader.deserialize(payload, _t))

    # --------------------------------------------------------------- high level

    def index(self, topics: Iterable[str], start_ns: Optional[int] = None,
              end_ns: Optional[int] = None) -> Dict[str, TopicIndex]:
        """Stamp index for the given topics, built without deserialising anything."""
        wanted = sorted(set(topics))
        known = self.topics()
        out = {t: TopicIndex(topic=t, msgtype=known[t].msgtype if t in known else "?")
               for t in wanted}
        if not wanted:
            return out

        for topic, log_time, payload, _ in self.iter_raw(wanted, start_ns, end_ns):
            entry = out[topic]
            stamp = stamp_from_cdr(payload)
            if stamp is None:
                entry.headerless += 1
                stamp = log_time
            entry.header_stamps.append(stamp)
            entry.log_times.append(log_time)
        return out

    def iter_messages(self, topics: Optional[Iterable[str]] = None,
                      start_ns: Optional[int] = None, end_ns: Optional[int] = None
                      ) -> Iterator[Tuple[str, int, object]]:
        """Yield ``(topic, stamp_ns, decoded_msg)`` — decodes every message."""
        for topic, log_time, payload, decode in self.iter_raw(topics, start_ns, end_ns):
            if self.stamp_source == "log":
                stamp = log_time
            else:
                stamp = stamp_from_cdr(payload) or log_time
            yield topic, stamp, decode(payload)

    def iter_selected(self, selection: Dict[str, set], start_ns: Optional[int] = None,
                      end_ns: Optional[int] = None
                      ) -> Iterator[Tuple[str, int, object]]:
        """Decode only the messages named in ``{topic: {stamp_ns, ...}}``.

        This is the conversion pass: everything the frame table did not choose is
        skipped at the payload level.
        """
        topics = [t for t, stamps in selection.items() if stamps]
        if not topics:
            return
        for topic, log_time, payload, decode in self.iter_raw(topics, start_ns, end_ns):
            if self.stamp_source == "log":
                stamp = log_time
            else:
                stamp = stamp_from_cdr(payload) or log_time
            if stamp in selection.get(topic, ()):
                yield topic, stamp, decode(payload)

    def first_message(self, topic: str):
        """``(stamp, msg)`` of the first message on a topic, or ``None``."""
        for topic_name, stamp, msg in self.iter_messages([topic]):
            return stamp, msg
        return None

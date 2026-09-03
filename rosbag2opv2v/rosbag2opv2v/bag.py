"""Streaming access to a rosbag2 MCAP recording.

The converter reads every bag twice:

* **pass 1 (index)** decodes only the small topics (poses, camera_info, tf,
  telemetry) and merely *peeks* the header stamp of the bulky point-cloud and
  image messages, so planning never deserialises a LiDAR sweep;
* **pass 2 (fetch)** re-walks the bag and decodes only the messages the plan
  actually selected.

Messages are addressed across the two passes by ``(topic, ordinal)`` where the
ordinal counts messages of that topic in bag order -- stable because both
passes iterate the same files in the same order.
"""

from __future__ import annotations

import bisect
import glob
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from .ros_msgs import msg_stamp, peek_header_stamp


@dataclass
class MessageRef:
    topic: str
    ordinal: int
    t: float
    frame_id: str = ""


def resolve_bag_files(path: str) -> List[str]:
    """Accept a ``.mcap`` file, a rosbag2 directory, or a directory of bags."""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(path)

    metadata = os.path.join(path, "metadata.yaml")
    if os.path.isfile(metadata):
        with open(metadata, "r") as handle:
            meta = yaml.safe_load(handle) or {}
        info = meta.get("rosbag2_bagfile_information", {})
        rel = info.get("relative_file_paths") or []
        files = [os.path.join(path, r) for r in rel]
        files = [f for f in files if os.path.isfile(f)]
        if files:
            return files
    files = sorted(glob.glob(os.path.join(path, "**", "*.mcap"), recursive=True))
    if not files:
        raise FileNotFoundError("no .mcap files under %s" % path)
    return files


class BagSource:
    """Two-pass reader over one or more MCAP files."""

    def __init__(self, files: Sequence[str]):
        self.files = list(files)
        self._decoders: Dict[int, Callable] = {}
        self._factory = None

    # -- internals ---------------------------------------------------------
    def _decoder(self, schema, channel):
        from mcap_ros2.decoder import DecoderFactory

        if self._factory is None:
            self._factory = DecoderFactory()
        key = schema.id if schema is not None else -1
        if key not in self._decoders:
            decoder = self._factory.decoder_for(channel.message_encoding, schema)
            if decoder is None:
                raise ValueError(
                    "cannot decode messages on topic '%s' (encoding=%s, "
                    "schema=%s)" % (channel.topic, channel.message_encoding,
                                    schema.name if schema else "none"))
            self._decoders[key] = decoder
        return self._decoders[key]

    def _iter(self, topics: Sequence[str]):
        from mcap.reader import make_reader

        topics = list(topics)
        for path in self.files:
            with open(path, "rb") as handle:
                reader = make_reader(handle)
                for schema, channel, message in reader.iter_messages(
                        topics=topics or None):
                    yield schema, channel, message

    # -- public API --------------------------------------------------------
    def available_topics(self) -> Dict[str, str]:
        """topic -> schema name, from the summary of every file."""
        from mcap.reader import make_reader

        found: Dict[str, str] = {}
        for path in self.files:
            with open(path, "rb") as handle:
                summary = make_reader(handle).get_summary()
                if summary is None:
                    continue
                for channel in summary.channels.values():
                    schema = summary.schemas.get(channel.schema_id)
                    found[channel.topic] = schema.name if schema else ""
        return found

    def scan(self,
             decode_topics: Iterable[str],
             index_topics: Iterable[str],
             time_source: str = "header",
             progress: Optional[Callable[[int], None]] = None
             ) -> Tuple[Dict[str, List[Tuple[float, object]]],
                        Dict[str, List[MessageRef]]]:
        """Pass 1.

        Returns ``(decoded, index)`` where ``decoded`` maps a small topic to a
        time-ordered list of ``(t, msg)`` and ``index`` maps a bulky topic to
        its list of :class:`MessageRef`.
        """
        decode_topics = set(decode_topics)
        index_topics = set(index_topics)
        wanted = sorted(decode_topics | index_topics)
        decoded: Dict[str, List[Tuple[float, object]]] = {
            t: [] for t in decode_topics}
        index: Dict[str, List[MessageRef]] = {t: [] for t in index_topics}
        counters: Dict[str, int] = {t: 0 for t in wanted}

        for i, (schema, channel, message) in enumerate(self._iter(wanted)):
            topic = channel.topic
            if topic not in counters:
                continue
            ordinal = counters[topic]
            counters[topic] += 1
            log_time = message.log_time * 1e-9

            if topic in index_topics:
                stamp, frame_id = log_time, ""
                if time_source == "header":
                    peeked = peek_header_stamp(message.data)
                    if peeked is not None and peeked[0] > 0.0:
                        stamp, frame_id = peeked
                index[topic].append(MessageRef(topic, ordinal, stamp, frame_id))

            if topic in decode_topics:
                msg = self._decoder(schema, channel)(message.data)
                stamp = log_time
                if time_source == "header":
                    header_time = msg_stamp(msg)
                    if header_time and header_time > 0.0:
                        stamp = header_time
                decoded[topic].append((stamp, msg))

            if progress is not None and (i % 20000) == 0:
                progress(i)

        for values in decoded.values():
            values.sort(key=lambda item: item[0])
        for refs in index.values():
            refs.sort(key=lambda ref: ref.t)
        return decoded, index

    def fetch(self,
              wanted: Dict[Tuple[str, int], object],
              handler: Callable[[str, int, object, object], None],
              progress: Optional[Callable[[int, int], None]] = None) -> int:
        """Pass 2: decode exactly the selected ``(topic, ordinal)`` messages.

        ``handler(topic, ordinal, msg, payload)`` is called once per hit, with
        ``payload`` the value stored in ``wanted``.
        """
        topics = sorted({topic for topic, _ in wanted})
        counters: Dict[str, int] = {t: 0 for t in topics}
        done = 0
        total = len(wanted)
        for schema, channel, message in self._iter(topics):
            topic = channel.topic
            if topic not in counters:
                continue
            ordinal = counters[topic]
            counters[topic] += 1
            payload = wanted.get((topic, ordinal))
            if payload is None:
                continue
            msg = self._decoder(schema, channel)(message.data)
            handler(topic, ordinal, msg, payload)
            done += 1
            if progress is not None:
                progress(done, total)
        return done


class RefIndex:
    """Bisection lookup over a time-sorted list of :class:`MessageRef`."""

    def __init__(self, refs: Sequence[MessageRef]):
        self.refs = list(refs)
        self.times = [ref.t for ref in self.refs]

    def __len__(self) -> int:
        return len(self.refs)

    @property
    def t_start(self) -> float:
        return self.times[0]

    @property
    def t_end(self) -> float:
        return self.times[-1]

    def nearest(self, t: float, max_age: float) -> Optional[MessageRef]:
        """Closest message to ``t`` within ``max_age`` seconds."""
        if not self.refs:
            return None
        idx = bisect.bisect_left(self.times, t)
        best, best_dt = None, None
        for candidate in (idx - 1, idx, idx + 1):
            if 0 <= candidate < len(self.refs):
                dt = abs(self.times[candidate] - t)
                if best_dt is None or dt < best_dt:
                    best, best_dt = self.refs[candidate], dt
        if best is None or best_dt > max_age:
            return None
        return best

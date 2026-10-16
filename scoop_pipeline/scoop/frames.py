# -*- coding: utf-8 -*-
"""
Renaming TF frames in a bag, e.g. ``os_lidar`` -> ``os_lidar_mobile2``, so two
robots' sensors never share a frame name once their bags are merged.

What is renamed, wherever the old name appears:

* ``header.frame_id`` of every message whose type starts with a Header
  (PointCloud2, Image, CameraInfo, Imu, ..., custom types too);
* ``child_frame_id`` of nav_msgs/Odometry and geometry_msgs/TransformStamped;
* both frame names of every transform in tf2_msgs/TFMessage (/tf, /tf_static).

A header is rewritten in the bytes when the name's length changes by a
multiple of 8 (``os_lidar`` -> ``os_lidar_mobile2`` does): every later field
keeps its CDR alignment, so the rest of the message is copied as it is --
a 6 MB cloud costs one copy, not a decode. Other lengths go through the
rosbags decoder (standard types; a custom type is refused). Everything else
is copied byte for byte.

    from scoop import frames
    frames.rename_bag(src, out, {"os_lidar": "os_lidar_mobile2"})
"""
from __future__ import annotations

import dataclasses
import struct
from pathlib import Path
from typing import Dict, Iterable, Optional

from . import bag, rosmsg
from .bag import BagError, TopicSchema

__all__ = ["parse_renames", "rename_payload", "rename_bag"]

_CHILD = {"nav_msgs/msg/Odometry", "geometry_msgs/msg/TransformStamped"}
_TF = "tf2_msgs/msg/TFMessage"


def parse_renames(items: Optional[Iterable[str]]) -> Dict[str, str]:
    """``["old:=new", ...]`` -> ``{"old": "new"}``."""
    out: Dict[str, str] = {}
    for it in items or ():
        if ":=" not in it:
            raise ValueError(f"rename {it!r}: expected OLD:=NEW")
        old, new = (x.strip() for x in it.split(":=", 1))
        if not old or not new or old == new:
            raise ValueError(f"rename {it!r}: two different, non-empty names")
        out[old] = new
    return out


def _header_first(schema: TopicSchema) -> bool:
    for line in schema.data.decode("utf-8", "replace").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            return line.split()[0] in ("Header", "std_msgs/Header", "std_msgs/msg/Header")
    return False


def _header_frame(payload) -> tuple:
    """(endian, frame_id, end offset of the string) of a header-first message."""
    if len(payload) < 16 or payload[0] != 0:
        raise BagError("not a CDR message with a header")
    e = "<" if payload[1] & 1 else ">"
    (n,) = struct.unpack_from(e + "I", payload, 12)
    end = 16 + n
    return e, bytes(payload[16:end - 1]).decode(), end


def _decoded(name: str, payload, mapping: Dict[str, str]) -> Optional[bytes]:
    ts = rosmsg.typestore()
    if name not in ts.types:
        return None
    m = ts.deserialize_cdr(bytes(payload), name)
    r = lambda f: mapping.get(f, f)                           # noqa: E731
    if name == _TF:
        trs = [dataclasses.replace(t, header=dataclasses.replace(t.header, frame_id=r(t.header.frame_id)),
                                   child_frame_id=r(t.child_frame_id)) for t in m.transforms]
        m = dataclasses.replace(m, transforms=trs)
    else:
        kw = {"header": dataclasses.replace(m.header, frame_id=r(m.header.frame_id))}
        if hasattr(m, "child_frame_id"):
            kw["child_frame_id"] = r(m.child_frame_id)
        m = dataclasses.replace(m, **kw)
    return bytes(ts.serialize_cdr(m, name))


def rename_payload(schema: TopicSchema, payload, mapping: Dict[str, str]) -> Optional[bytes]:
    """The message with its frames renamed, or None when nothing changes."""
    if not mapping:
        return None
    if schema.name == _TF:
        ts = rosmsg.typestore()
        m = ts.deserialize_cdr(bytes(payload), _TF)
        if not any(t.header.frame_id in mapping or t.child_frame_id in mapping
                   for t in m.transforms):
            return None
        return _decoded(_TF, payload, mapping)
    if schema.name in _CHILD:
        m = rosmsg.typestore().deserialize_cdr(bytes(payload), schema.name)
        if m.header.frame_id not in mapping and m.child_frame_id not in mapping:
            return None
        return _decoded(schema.name, payload, mapping)
    if not _header_first(schema):
        return None
    e, frame, end = _header_frame(payload)
    if frame not in mapping:
        return None
    new = mapping[frame].encode() + b"\0"
    if (len(new) - (end - 16)) % 8 == 0:                      # alignment kept
        return bytes(payload[:12]) + struct.pack(e + "I", len(new)) + new + bytes(payload[end:])
    out = _decoded(schema.name, payload, mapping)
    if out is None:
        raise BagError(f"{schema.name}: unknown type, and renaming {frame!r} to "
                       f"{mapping[frame]!r} changes its length by "
                       f"{len(new) - (end - 16)} bytes (not a multiple of 8)")
    return out


def rename_bag(src, out_dir, mapping: Dict[str, str], compression: str = "zstd",
               log=print) -> Dict[str, int]:
    """``src`` -> new bag ``out_dir`` with ``mapping``'s frames renamed (see
    the module docstring); log times, QoS and order kept. Returns the number
    of messages renamed per topic."""
    from .bagwrite import BagWriter, write_in_order
    reader = bag.open_bag(src)
    qos = bag.topic_qos(reader)
    renamed: Dict[str, int] = {}

    def records():
        for _, sc, ch, msg in bag.iter_mcap_records(reader):
            schema = TopicSchema(sc.name, sc.encoding, bytes(sc.data), ch.message_encoding)
            data = msg.data
            new = rename_payload(schema, data, mapping)
            if new is not None:
                data = new
                renamed[ch.topic] = renamed.get(ch.topic, 0) + 1
            yield (msg.log_time, ch.topic, schema, msg.publish_time, msg.sequence,
                   dict(ch.metadata), data)

    with BagWriter(out_dir, library="scoop.frames", compression=compression) as w:
        for t, q in qos.items():
            w.set_qos(t, q)
        write_in_order(w, records(), 0.0)
    log(f"wrote {w.n} messages -> {Path(out_dir)}")
    for t, n in sorted(renamed.items()):
        log(f"    renamed {n} msgs on {t}")
    if not renamed:
        log(f"    WARNING: none of {sorted(mapping)} appears in {src}")
    return renamed

# -*- coding: utf-8 -*-
"""The link's state, written into every frame beside the sensor data.

A cooperative-perception frame is a set of messages that crossed a network,
and the testbed records what that network was doing: the Wi-Fi link status
(RSSI, bit rates, channel), round-trip time at ~100 Hz, goodput tests, the NTP
daemon's residual, and the CSI a sniffer hears per packet. Each is a ROS topic
of a custom message type; none of them has a fixed schema this converter
knows, so the fields are discovered from the message (every scalar field,
flattened as ``a.b.c``) rather than assumed.

Two ways a stream enters a frame:

* ``kind: nearest``   the one message closest to the frame time within
                      ``window_s`` (link status, goodput tests, NTP status): its
                      scalar fields and the signed ``dt_ms`` to the frame.
* ``kind: window``    every message within ``window_s`` centred on the frame
                      time (ping, CSI): ``n``, the median of every numeric field,
                      the fraction of every boolean field (a ``reply`` flag
                      becomes the delivery ratio), and p95 for the fields named
                      in ``p95``.

A stream with an ``agent`` goes into that agent's frame yaml under
``comms.<name>``; a stream without one (a CSI link heard by a sniffer) goes
into every agent's yaml under ``comms.links.<name>``. The full-rate record is
written once per scenario as ``comms/<name>.csv`` (one row per message, every
scalar field, stamps on the corrected timeline), so a per-frame summary can
always be traced back to the packets behind it.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

NS = 1_000_000_000


# ------------------------------------------------------------------ flattening

def _fields(msg) -> Sequence[str]:
    names = getattr(msg, "__slots__", None)
    if names is None:
        names = list(getattr(msg, "__dataclass_fields__", {}) or [])
    return [n.lstrip("_") for n in names]


def _is_time(v) -> bool:
    return hasattr(v, "sec") and hasattr(v, "nanosec")


def flatten(msg) -> Tuple[Dict[str, float], Dict[str, str], List[str]]:
    """``(numbers, strings, booleans)``: every scalar field of ``msg`` by its
    dotted path. Times become float seconds, booleans numbers 0/1 (their names
    returned separately so a window can report them as fractions), arrays and
    the header are left out."""
    numbers: Dict[str, float] = {}
    strings: Dict[str, str] = {}
    booleans: List[str] = []

    def walk(v, name):
        if name == "header":
            return
        if isinstance(v, bool):
            numbers[name] = 1.0 if v else 0.0
            booleans.append(name)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            numbers[name] = float(v)
        elif isinstance(v, str):
            strings[name] = v
        elif isinstance(v, (bytes, list, tuple, np.ndarray)):
            return
        elif _is_time(v):
            numbers[name] = float(v.sec) + float(v.nanosec) * 1e-9
        elif _fields(v):
            for k in _fields(v):
                walk(getattr(v, k), f"{name}.{k}" if name else k)

    for k in _fields(msg):
        walk(getattr(msg, k), k)
    return numbers, strings, booleans


# --------------------------------------------------------------------- streams

@dataclass
class StreamConfig:
    name: str
    topic: str
    kind: str                      # 'nearest' | 'window'
    agent: Optional[str] = None    # None: a link, written into every agent
    host: Optional[str] = None     # clock domain; defaults to the agent's name
    window_s: float = 1.0
    p95: List[str] = field(default_factory=list)

    @staticmethod
    def parse(mapping: dict, ctx: str) -> "StreamConfig":
        from .config import ConfigError
        topic = mapping.get("topic")
        if not topic:
            raise ConfigError(f"{ctx}: a comms stream needs a topic")
        kind = str(mapping.get("kind", "nearest")).lower()
        if kind not in ("nearest", "window"):
            raise ConfigError(f"{ctx}: kind must be nearest or window, got {kind!r}")
        name = mapping.get("name") or "_".join(str(topic).strip("/").split("/")[-2:])
        return StreamConfig(
            name=str(name), topic=str(topic), kind=kind,
            agent=mapping.get("agent"), host=mapping.get("host"),
            window_s=float(mapping.get("window_s", 1.0)),
            p95=[str(x) for x in (mapping.get("p95") or [])])

    @property
    def clock_host(self) -> Optional[str]:
        return self.host or self.agent

    @property
    def key(self) -> str:
        """``<agent>_<name>`` for an agent's stream, ``<name>`` for a link: the
        csv file's stem and the report's key."""
        return f"{self.agent}_{self.name}" if self.agent else self.name


@dataclass
class CommsConfig:
    enabled: bool = False
    streams: List[StreamConfig] = field(default_factory=list)
    tables_dir: Optional[str] = None   # copied into <scenario>/comms/ as is
    write_csv: bool = True

    @staticmethod
    def parse(mapping: Optional[dict]) -> "CommsConfig":
        from .config import ConfigError
        mapping = mapping or {}
        streams = [StreamConfig.parse(s, f"comms.streams[{i}]")
                   for i, s in enumerate(mapping.get("streams") or [])]
        keys = [s.key for s in streams]
        dup = sorted({k for k in keys if keys.count(k) > 1})
        if dup:
            raise ConfigError(f"comms: duplicate stream names {dup} (a name is unique per "
                              f"agent; a link's name is unique on its own)")
        tables = mapping.get("tables_dir")
        return CommsConfig(
            enabled=bool(mapping.get("enabled", bool(streams))),
            streams=streams,
            tables_dir=os.path.expanduser(str(tables)) if tables else None,
            write_csv=bool(mapping.get("write_csv", True)))

    def topics(self) -> List[str]:
        return sorted({s.topic for s in self.streams}) if self.enabled else []


# ------------------------------------------------------------------- the index

class StreamIndex:
    """Every message of one stream: corrected stamps and flattened scalars."""

    def __init__(self, cfg: StreamConfig) -> None:
        self.cfg = cfg
        self._stamps: List[int] = []
        self._raw: List[int] = []
        self._numbers: List[Dict[str, float]] = []
        self._strings: List[Dict[str, str]] = []
        self.booleans: set = set()
        self.stamps = np.zeros(0, dtype=np.int64)
        self.columns: List[str] = []
        self.string_columns: List[str] = []

    def add(self, stamp_ns: int, raw_ns: int, msg) -> None:
        numbers, strings, booleans = flatten(msg)
        self._stamps.append(int(stamp_ns))
        self._raw.append(int(raw_ns))
        self._numbers.append(numbers)
        self._strings.append(strings)
        self.booleans.update(booleans)

    def finish(self) -> None:
        order = np.argsort(np.asarray(self._stamps, dtype=np.int64), kind="stable")
        self.stamps = np.asarray(self._stamps, dtype=np.int64)[order]
        self._raw = [self._raw[i] for i in order]
        self._numbers = [self._numbers[i] for i in order]
        self._strings = [self._strings[i] for i in order]
        cols: set = set()
        scols: set = set()
        for d in self._numbers:
            cols.update(d)
        for d in self._strings:
            scols.update(d)
        self.columns = sorted(cols)
        self.string_columns = sorted(scols)
        self._table = {c: np.array([d.get(c, np.nan) for d in self._numbers], dtype=float)
                       for c in self.columns}

    def __len__(self) -> int:
        return len(self.stamps)

    # -- per frame
    def nearest(self, t_ns: int) -> Optional[dict]:
        if not len(self):
            return None
        i = int(np.searchsorted(self.stamps, t_ns))
        cands = [j for j in (i - 1, i) if 0 <= j < len(self)]
        j = min(cands, key=lambda k: abs(int(self.stamps[k]) - t_ns))
        dt = int(self.stamps[j]) - t_ns
        if abs(dt) > self.cfg.window_s * NS:
            return {"present": False, "dt_ms": None, "n_in_window": 0}
        block = {"present": True, "dt_ms": round(dt / 1e6, 3)}
        for c in self.columns:
            v = self._table[c][j]
            if np.isfinite(v):
                block[c] = (bool(v) if c in self.booleans else _num(v))
        block.update({c: self._strings[j][c] for c in self.string_columns
                      if c in self._strings[j]})
        return block

    def window(self, t_ns: int) -> dict:
        half = int(self.cfg.window_s * NS / 2)
        # [t - w/2, t + w/2): half-open, so a message on the boundary belongs
        # to exactly one of two adjacent windows
        lo = int(np.searchsorted(self.stamps, t_ns - half, side="left"))
        hi = int(np.searchsorted(self.stamps, t_ns + half, side="left"))
        block: dict = {"n": hi - lo, "window_s": self.cfg.window_s}
        if hi <= lo:
            return block
        for c in self.columns:
            v = self._table[c][lo:hi]
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            if c in self.booleans:
                block[c + "_fraction"] = _num(float(np.mean(v)))
            else:
                block[c + "_median"] = _num(float(np.median(v)))
                if c in self.cfg.p95:
                    block[c + "_p95"] = _num(float(np.percentile(v, 95)))
        return block

    def block(self, t_ns: int) -> Optional[dict]:
        return self.nearest(t_ns) if self.cfg.kind == "nearest" else self.window(t_ns)

    # -- the full-rate record
    def write_csv(self, path: str) -> int:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cols = ["t_ns", "t_raw_ns"] + self.columns + self.string_columns
        with open(path, "w", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(cols)
            for i in range(len(self)):
                row = [int(self.stamps[i]), self._raw[i]]
                row += [_csv_num(self._table[c][i]) for c in self.columns]
                row += [self._strings[i].get(c, "") for c in self.string_columns]
                w.writerow(row)
        return len(self)


def _num(v: float):
    return int(v) if float(v).is_integer() and abs(v) < 1e15 else round(float(v), 6)


def _csv_num(v: float) -> str:
    return "" if not np.isfinite(v) else repr(_num(v))


def load_streams(reader, cfg: CommsConfig, clocks=None) -> Dict[str, StreamIndex]:
    """Decode every comms topic once; stamps corrected per host like the sensors'."""
    out = {s.key: StreamIndex(s) for s in cfg.streams}
    by_topic: Dict[str, List[StreamIndex]] = {}
    for idx in out.values():
        by_topic.setdefault(idx.cfg.topic, []).append(idx)
    if not by_topic:
        return out
    for topic, stamp, msg in reader.iter_messages(sorted(by_topic)):
        for idx in by_topic[topic]:
            corrected = stamp
            if clocks is not None and idx.cfg.clock_host:
                corrected = stamp + clocks.correction_ns(idx.cfg.clock_host, stamp)[0]
            idx.add(corrected, stamp, msg)
    for idx in out.values():
        idx.finish()
    return out


def frame_block(streams: Dict[str, StreamIndex], agent_name: str, t_ns: int) -> dict:
    """The ``comms`` block of one agent's frame yaml."""
    block: dict = {}
    links: dict = {}
    for idx in streams.values():
        if idx.cfg.agent is None:
            links[idx.cfg.name] = idx.block(t_ns)
        elif idx.cfg.agent == agent_name:
            block[idx.cfg.name] = idx.block(t_ns)
    if links:
        block["links"] = links
    return block

# -*- coding: utf-8 -*-
"""
The link and time-base measurements of a pass as tables: NTP, Wi-Fi status,
ping, iperf and CSI, from the robots' comms_msgs topics.

One pass over the bag reads every comms topic. Each kind becomes one table
(rows of every machine, a ``machine`` column); CSI keeps its arrays in one
.npz per link. A message's scalar fields become columns, named as in its
definition (nested ones joined with '.'); ``t`` is its header stamp (that
machine's clock), ``t_log`` the time the bag logged it.

    tables = extract(bag)                     # {kind: Table}, csi arrays inside
    write(tables, out_dir)                    # <kind>.csv, csi_<tx>_to_<rx>.npz
    summary(tables)                           # per machine / link, for summary.json

Positions (:class:`Positions`) add where each machine was when it measured:
the pass's TF tree for fixed frames, its trajectory for the robot that moves.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from . import bag as scoop_bag

__all__ = ["KINDS", "kind_of", "ends_of", "flatten", "Table", "extract", "write",
           "summary", "Positions"]

KINDS = {"comms_msgs/msg/NtpStatus": "ntp", "comms_msgs/msg/PingStat": "ping",
         "comms_msgs/msg/IperfResult": "iperf", "comms_msgs/msg/WifiLinkStatus": "wifi"}
SEGMENTS = ("ntp", "wifi", "csi_publisher")


def kind_of(topic: str, msgtype: str) -> Optional[str]:
    if msgtype in KINDS:
        return KINDS[msgtype]
    parts = topic.strip("/").split("/")
    if parts[-1] == "csi":
        return "csi"
    if "csi_publisher" in parts:
        return "csi_status"
    return None


def ends_of(topic: str):
    """'/mobile_1/wifi/ping' -> ('mobile_1', None); '/mobile_1/sniffer/infra_1/csi'
    -> ('mobile_1/sniffer', 'infra_1'): the machine that measured, and for CSI
    the one whose packets it heard."""
    parts = topic.strip("/").split("/")
    if parts[-1] == "csi" and len(parts) >= 3:
        return "/".join(parts[:-2]), parts[-2]
    for i, p in enumerate(parts):
        if p in SEGMENTS:
            return "/".join(parts[:i]), None
    return "/".join(parts[:-1]), None


def _is_time(v):
    return hasattr(v, "sec") and hasattr(v, "nanosec") and len(_fields(v)) == 2


def _fields(m):
    names = getattr(m, "__slots__", None) or getattr(m, "__dataclass_fields__", None)
    if names is None and hasattr(m, "__dict__"):
        names = vars(m)
    return [k for k in (names or []) if not k.startswith("_")]


def flatten(msg, keep_arrays: bool = False):
    """A decoded message -> (scalars {name: value}, arrays {name: 1-D array}).
    header.stamp -> 't' (s), header.frame_id -> 'frame_id', a Time -> s.
    Numeric sequences go to arrays when keep_arrays, else into one text
    column ('a;b;c'); string sequences are joined with ' | '."""
    sc, ar = {}, {}

    def walk(v, name):
        if _is_time(v):
            sc[name] = v.sec + v.nanosec * 1e-9
        elif isinstance(v, (bytes, bytearray, memoryview)):
            a = np.frombuffer(bytes(v), np.uint8)
            if keep_arrays:
                ar[name] = a
            else:
                sc[name] = bytes(v).hex()
        elif isinstance(v, np.ndarray) or isinstance(v, (list, tuple)):
            if len(v) and not isinstance(v[0], (int, float, bool, np.number, np.bool_)):
                if all(isinstance(x, str) for x in v):
                    sc[name] = " | ".join(v)
                else:
                    for i, x in enumerate(v):
                        walk(x, "%s.%d" % (name, i))
                return
            a = np.asarray(v)
            if keep_arrays:
                ar[name] = a
            else:
                sc[name] = ";".join("%.6g" % x for x in a.astype(float))
        elif _fields(v) and not isinstance(v, (str, int, float, bool)):
            for k in _fields(v):
                walk(getattr(v, k), "%s.%s" % (name, k) if name else k)
        else:
            sc[name] = v.item() if isinstance(v, np.generic) else v

    walk(msg, "")
    if "header.stamp" in sc:
        sc["t"] = sc.pop("header.stamp")
    if "header.frame_id" in sc:
        sc["frame_id"] = sc.pop("header.frame_id")
    return sc, ar


@dataclass
class Table:
    kind: str
    rows: List[dict] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    arrays: Dict[str, Dict[str, list]] = field(default_factory=dict)   # link -> name -> [arr]

    def add(self, row: dict):
        for k in row:
            if k not in self._seen:
                self._seen.add(k)
                self.columns.append(k)
        self.rows.append(row)

    def __post_init__(self):
        self._seen = set(self.columns)

    def col(self, name, rows=None, dtype=float):
        rows = self.rows if rows is None else rows
        out = []
        for r in rows:
            v = r.get(name)
            try:
                out.append(dtype(v) if v is not None and v != "" else math.nan)
            except (TypeError, ValueError):
                out.append(math.nan)
        return np.array(out, float)


FIRST = ["machine", "peer", "topic", "t", "t_log"]


def extract(bag_dir, log=print) -> Dict[str, Table]:
    """Every comms topic of the bag, in one pass."""
    r = scoop_bag.open_bag(bag_dir)
    info = r.topics()
    topics = {t: kind_of(t, i.msgtype) for t, i in info.items()}
    topics = {t: k for t, k in topics.items() if k}
    if not topics:
        raise scoop_bag.BagError(f"{bag_dir}: no comms topics (comms_msgs, */csi)")
    for t in sorted(topics):
        log(f"    {topics[t]:10s} {t}  ({info[t].count} messages)")
    def messages():
        n = 0
        for topic, log_ns, payload, decode in r.iter_raw(sorted(topics)):
            yield topic, log_ns, decode(payload)
            n += 1
            if n % 50000 == 0:
                log(f"    read {n} comms messages")
    return extract_from(messages(), topics)


def extract_from(reader_iter, topics: Dict[str, str]) -> Dict[str, Table]:
    """The tables from (topic, log_ns, message) tuples: what extract() reads,
    split out so it can be fed without a bag."""
    tables: Dict[str, Table] = {}
    for topic, log_ns, msg in reader_iter:
        kind = topics.get(topic)
        if kind is None:
            continue
        machine, peer = ends_of(topic)
        sc, ar = flatten(msg, keep_arrays=(kind == "csi"))
        row = {"machine": machine, "peer": peer or "", "topic": topic,
               "t": sc.pop("t", math.nan), "t_log": log_ns * 1e-9}
        row.update(sc)
        tb = tables.setdefault(kind, Table(kind, columns=list(FIRST)))
        tb.add(row)
        if ar:
            link = "%s_to_%s" % (peer, machine.replace("/", "_"))
            d = tb.arrays.setdefault(link, {"t": [], "t_log": []})
            d["t"].append(row["t"])
            d["t_log"].append(row["t_log"])
            for k, a in ar.items():
                d.setdefault(k, []).append(a)
    return tables


def write(tables: Dict[str, Table], out_dir, log=print) -> List[str]:
    """<kind>.csv per table, rows in time order per machine; csi_<link>.npz
    with every array field stacked (rows x values) when the lengths agree,
    else concatenated with '<name>_offsets'."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for kind, tb in sorted(tables.items()):
        rows = sorted(tb.rows, key=lambda r: (r["machine"], r["peer"], r["t_log"]))
        p = os.path.join(out_dir, f"{kind}.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=tb.columns, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: _fmt(v) for k, v in row.items()})
        written.append(p)
        log(f"    {p}: {len(rows)} rows, {len(tb.columns)} columns")
        for link, d in sorted(tb.arrays.items()):
            out = {}
            for k, seq in d.items():
                if k in ("t", "t_log"):
                    out[k] = np.asarray(seq, float)
                    continue
                lens = {len(a) for a in seq}
                if len(lens) == 1:
                    out[k] = np.stack(seq)
                else:
                    out[k] = np.concatenate(seq)
                    out[k + "_offsets"] = np.cumsum([0] + [len(a) for a in seq])
            p = os.path.join(out_dir, f"csi_{link}.npz")
            np.savez_compressed(p, **out)
            written.append(p)
            log(f"    {p}: {len(d['t'])} packets, "
                + ", ".join(f"{k} {v.shape}" for k, v in out.items() if k not in ("t", "t_log")))
    return written


def _fmt(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else repr(v)
    if isinstance(v, bool):
        return int(v)
    return v


def _stats(a):
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "min": float(a.min()), "max": float(a.max())}


def summary(tables: Dict[str, Table]) -> dict:
    """Per machine (and link): what the tables say, in a few numbers."""
    out = {}
    by = lambda tb: {(r["machine"], r["peer"]) for r in tb.rows}           # noqa: E731
    for kind, tb in tables.items():
        res = out.setdefault(kind, {})
        for machine, peer in sorted(by(tb)):
            rows = [r for r in tb.rows if r["machine"] == machine and r["peer"] == peer]
            t = tb.col("t", rows)
            s = {"messages": len(rows),
                 "t_start": float(np.nanmin(t)) if np.isfinite(t).any() else None,
                 "t_end": float(np.nanmax(t)) if np.isfinite(t).any() else None}
            if kind == "ntp":
                s["synchronized_percent"] = 100 * float(np.nanmean(tb.col("synchronized", rows)))
                s["offset_ms"] = _scale(_stats(np.abs(tb.col("offset_seconds", rows))), 1e3)
                s["measured_offset_ms"] = _scale(_stats(np.abs(tb.col("last_offset_seconds", rows))), 1e3)
                s["clock_steps"] = int(np.nansum(tb.col("clock_stepped", rows)))
                s["sources"] = sorted({str(r.get("sync_source")) for r in rows})
                s["frequency_error_ppm"] = _stats(tb.col("frequency_error_ppm", rows))
            elif kind == "ping":
                reply = tb.col("reply", rows)
                s["loss_percent"] = 100 * float(1 - np.nanmean(reply)) if len(rows) else None
                s["rtt_ms"] = _stats(tb.col("rtt_ms", rows))
                s["targets"] = sorted({str(r.get("target")) for r in rows})
            elif kind == "iperf":
                for direction, rev in (("down", 1.0), ("up", 0.0)):
                    sel = [r for r, v in zip(rows, tb.col("reverse", rows)) if v == rev]
                    ok = [r for r, v in zip(sel, tb.col("success", sel)) if v == 1.0]
                    s[direction] = {"tests": len(sel), "failed": len(sel) - len(ok),
                                    "mbps": _stats(tb.col("bitrate_mbps", ok))}
            elif kind == "wifi":
                s["signal_dbm"] = _stats(tb.col("signal_dbm", rows))
                s["tx_bitrate_mbps"] = _stats(tb.col("tx_bitrate_mbps", rows))
                s["rx_bitrate_mbps"] = _stats(tb.col("rx_bitrate_mbps", rows))
                bss = [str(r.get("bssid")) for r in sorted(rows, key=lambda r: r["t_log"])]
                s["bssids"] = sorted(set(bss))
                s["roams"] = int(sum(a != b for a, b in zip(bss, bss[1:])))
                s["channels"] = sorted({str(r.get("channel")) for r in rows})
            elif kind == "csi":
                tt = np.sort(t[np.isfinite(t)])
                s["rate_hz"] = float((len(tt) - 1) / (tt[-1] - tt[0])) if len(tt) > 1 and tt[-1] > tt[0] else None
                s["longest_gap_s"] = float(np.diff(tt).max()) if len(tt) > 1 else None
            if "distance_m" in tb.columns:
                s["distance_m"] = _stats(tb.col("distance_m", rows))
            res[machine + (" <- " + peer if peer else "")] = s
    return out


def _scale(st, k):
    return None if st is None else {a: b * k for a, b in st.items()}


class Positions:
    """Where a frame was in `map` at time t: fixed frames from the static
    tree, the moving robot's frames through its trajectory (T_map_body at the
    trajectory's stamps, body somewhere in the same moving subtree)."""

    def __init__(self, edges, traj_t=None, traj_T=None, body=None, map_frame="map"):
        from . import tftree
        self.tf, self.edges, self.map = tftree, edges, map_frame
        self.traj_t, self.traj_T, self.body = traj_t, traj_T, body
        self._cache: Dict[str, Callable] = {}

    def _top(self, frame):
        """(top, T_top_frame, moving): up the static edges from frame."""
        M, f = np.eye(4), frame
        while f in self.edges and self.edges[f].static:
            M = self.edges[f].T @ M
            f = self.edges[f].parent
        return f, M, f in self.edges          # stopped on a /tf edge -> moving

    def of(self, frame) -> Optional[Callable]:
        if frame in self._cache:
            return self._cache[frame]
        top, T_top_f, moving = self._top(frame)
        fn = None
        if not moving and top == self.map:
            p = T_top_f[:3, 3].copy()
            fn = lambda t: np.broadcast_to(p, (len(np.atleast_1d(t)), 3)).copy()   # noqa: E731
        elif moving and self.traj_t is not None and self.body is not None:
            btop, T_top_b, _ = self._top(self.body)
            if btop == top:
                T_b_f = np.linalg.inv(T_top_b) @ T_top_f
                fn = self._moving(T_b_f)
        self._cache[frame] = fn
        return fn

    def _moving(self, T_b_f):
        t0, T = self.traj_t, self.traj_T
        P = np.einsum("nij,j->ni", T[:, :3, :], np.append(T_b_f[:3, 3], 1.0))

        def at(t):
            t = np.atleast_1d(np.asarray(t, float))
            out = np.full((len(t), 3), np.nan)
            ok = (t >= t0[0]) & (t <= t0[-1])
            for k in range(3):
                out[ok, k] = np.interp(t[ok], t0, P[:, k])
            return out
        return at

# -*- coding: utf-8 -*-
"""
Adding calibrated transforms (e.g. os_lidar -> zed_left_camera_optical_frame)
to the TF tree a bag already carries.

In TF every frame has at most one parent. When the child of a calibration
already has one (the ZED's optical frame hangs below zed_left_camera_frame),
:func:`attach` publishes the same geometry the other way round: the calibrated
child becomes the parent of the ROOT of the other tree (os_sensor above
os_lidar), with the transform composed through that tree's static
transforms. Looking up os_lidar -> zed_left_camera_optical_frame then gives
exactly the calibration.

    edges = tftree.tf_edges(bags)                       # what the bags publish
    publish, notes = tftree.attach(calibrations, edges)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import bag, rosmsg
from .bag import BagError

__all__ = ["Edge", "Calibration", "matrix", "to_tq", "tf_edges", "attach", "load_calibrations"]

TFMSG = "tf2_msgs/msg/TFMessage"


def matrix(t: Sequence[float], q: Sequence[float]) -> np.ndarray:
    """4x4 from translation xyz and quaternion xyzw (normalised)."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


def to_tq(M: np.ndarray) -> Tuple[List[float], List[float]]:
    """4x4 -> (translation xyz, quaternion xyzw with w >= 0)."""
    R = M[:3, :3]
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w]) / np.linalg.norm([x, y, z, w])
    return [float(v) for v in M[:3, 3]], [float(v) for v in q]


@dataclass
class Edge:
    parent: str
    child: str
    static: bool
    T: Optional[np.ndarray] = None          # pose of child in parent (static ones)


@dataclass
class Calibration:
    parent: str
    child: str
    translation: List[float]
    rotation_xyzw: List[float]


def load_calibrations(doc: dict) -> List[Calibration]:
    return [Calibration(c["parent"], c["child"], list(c["translation"]),
                        list(c["rotation_xyzw"])) for c in (doc or {}).get("transforms") or []]


def tf_edges(bags: Iterable, seconds: float = 30.0) -> Dict[str, Edge]:
    """child -> Edge, from every /tf_static message and the first
    ``seconds`` of /tf of each bag (dynamic edges repeat all the time)."""
    from mcap.reader import make_reader
    ts = rosmsg.typestore()
    edges: Dict[str, Edge] = {}
    for b in bags:
        r = bag.open_bag(b)
        tps = r.topics()
        for topic, static in (("/tf_static", True), ("/tf", False)):
            if topic not in tps:
                continue
            for f in r.files:
                with open(f, "rb") as fh:
                    rd = make_reader(fh)
                    summ = rd.get_summary()
                    t0 = summ.statistics.message_start_time if summ and summ.statistics else 0
                    end = None if static else t0 + int(seconds * 1e9)
                    for sc, _, msg in rd.iter_messages(topics=[topic], end_time=end):
                        if sc is None or sc.name != TFMSG:
                            break                             # not a TF topic after all
                        for tr in ts.deserialize_cdr(msg.data, TFMSG).transforms:
                            p, c = tr.header.frame_id.lstrip("/"), tr.child_frame_id.lstrip("/")
                            old = edges.get(c)
                            if old is not None and old.parent != p:
                                raise BagError(f"{c} has two parents in the bags: "
                                               f"{old.parent} and {p}")
                            if old is None or static:
                                t, q = tr.transform.translation, tr.transform.rotation
                                edges[c] = Edge(p, c, static,
                                                matrix([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
                                                if static else None)
                if not static:
                    break                                     # the first file is enough
    return edges


def _root(frame: str, edges: Dict[str, Edge]) -> str:
    seen = set()
    while frame in edges:
        if frame in seen:
            raise BagError(f"TF loop at {frame}")
        seen.add(frame)
        frame = edges[frame].parent
    return frame


def _static_pose(root: str, frame: str, edges: Dict[str, Edge]) -> np.ndarray:
    """Pose of ``frame`` in its ancestor ``root``, through static edges only."""
    M = np.eye(4)
    f = frame
    while f != root:
        e = edges[f]
        if not e.static:
            raise BagError(f"{e.parent} -> {f} is not static; cannot attach through it")
        M = e.T @ M
        f = e.parent
    return M


def attach(cals: List[Calibration], edges: Dict[str, Edge]):
    """The transforms to add to /tf_static for ``cals``, as
    ``[(parent, child, t, q)]``, and one note per calibration."""
    edges = dict(edges)
    out, notes = [], []
    for c in cals:
        T = matrix(c.translation, c.rotation_xyzw)
        rp, rc = _root(c.parent, edges), _root(c.child, edges)
        if rp == rc:
            raise BagError(f"{c.parent} and {c.child} are already connected (root {rp}); "
                           f"adding {c.parent} -> {c.child} would make a loop")
        if c.child not in edges:                              # free: add as given
            add = (c.parent, c.child, T)
            notes.append(f"{c.parent} -> {c.child}")
        else:                                                 # re-root the parent's tree
            T_root_parent = _static_pose(rp, c.parent, edges)
            add = (c.child, rp, np.linalg.inv(T) @ np.linalg.inv(T_root_parent))
            notes.append(f"{c.child} already has a parent ({edges[c.child].parent}): "
                         f"added {c.child} -> {rp} instead, so {c.parent} -> {c.child} "
                         f"is the calibration")
        t, q = to_tq(add[2])
        out.append((add[0], add[1], t, q))
        edges[add[1]] = Edge(add[0], add[1], True, add[2])
    return out, notes

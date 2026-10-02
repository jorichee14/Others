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

Replacing the odometry works the same way: :func:`edge_matcher` drops edges
(the ZED's map_zed -> odom_zed -> zed_camera_link), and :func:`odom_transforms`
turns a new trajectory -- poses of a body frame such as os_lidar, from a
topic (:func:`read_pose_topic`) or a TUM file (:func:`read_tum`) -- into
/tf transforms world -> <root of the body's tree>, so looking up
world -> body gives exactly the trajectory.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import bag, rosmsg
from .bag import BagError

__all__ = ["Edge", "Calibration", "matrix", "to_tq", "tf_edges", "attach", "load_calibrations",
           "edge_matcher", "read_tum", "read_pose_topic", "odom_transforms", "plan_edits",
           "format_tree", "TFEdits"]

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
    ``seconds`` of /tf of each bag, counted from its first /tf message
    (dynamic edges repeat all the time; a merged bag can start well before
    its /tf does)."""
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
                    end = None
                    for sc, _, msg in rd.iter_messages(topics=[topic]):
                        if sc is None or sc.name != TFMSG:
                            break                             # not a TF topic after all
                        if not static:
                            end = end or msg.log_time + int(seconds * 1e9)
                            if msg.log_time > end:
                                break
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


def edge_matcher(patterns):
    """``["map_zed->odom_zed", "odom_zed->*"]`` -> f(parent, child) -> drop?
    (shell patterns on either side)."""
    pairs = []
    for p in patterns or ():
        if "->" not in p:
            raise ValueError(f"--drop-tf {p}: expected PARENT->CHILD")
        a, b = (x.strip() for x in p.split("->", 1))
        pairs.append((a, b))

    def drop(parent: str, child: str) -> bool:
        return any(fnmatch.fnmatchcase(parent, a) and fnmatch.fnmatchcase(child, b)
                   for a, b in pairs)
    return drop


def read_tum(path) -> List[Tuple[int, np.ndarray]]:
    """TUM trajectory (``t tx ty tz qx qy qz qw`` per line, t in s) ->
    ``[(t_ns, 4x4)]``; GLIM's traj_lidar.txt is one."""
    out = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            v = line.split("#", 1)[0].split()
            if not v:
                continue
            if len(v) != 8:
                raise BagError(f"{path}:{n}: expected 8 numbers (t tx ty tz qx qy qz qw)")
            f = [float(x) for x in v]
            out.append((int(round(f[0] * 1e9)), matrix(f[1:4], f[4:8])))
    if not out:
        raise BagError(f"{path}: no poses")
    return out


def read_pose_topic(bags, topic: str):
    """Poses of a topic (nav_msgs/Odometry, geometry_msgs/PoseStamped or
    PoseWithCovarianceStamped) -> ``([(stamp_ns, 4x4)], parent frame, child
    frame or None)``."""
    for b in bags:
        r = bag.open_bag(b)
        if topic not in r.topics():
            continue
        out, parent, child = [], None, None
        for _, _, payload, decode in r.iter_raw([topic]):
            m = decode(payload)
            pose = m.pose.pose if hasattr(m.pose, "pose") else m.pose
            st = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
            p, q = pose.position, pose.orientation
            out.append((st, matrix([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])))
            parent = parent or m.header.frame_id
            child = child or getattr(m, "child_frame_id", None) or None
        return out, parent, child
    raise BagError(f"{topic} is in none of the bags")


def odom_transforms(poses, body: str, parent: str, edges: Dict[str, Edge]):
    """Poses of ``body`` in ``parent`` -> ``[(t_ns, parent, root, t, q)]``, the
    /tf transforms parent -> root of body's tree (through its static edges),
    so a lookup parent -> body gives the poses."""
    root = _root(body, edges)
    if _root(parent, edges) == root:
        raise BagError(f"{parent} is already in {body}'s tree (root {root}); drop the "
                       f"old odometry edges first (--drop-tf)")
    if parent in edges:
        raise BagError(f"{parent} already has a parent ({edges[parent].parent})")
    T_rb = _static_pose(root, body, edges)
    inv = np.linalg.inv(T_rb)
    out = []
    for t_ns, T_pb in poses:
        t, q = to_tq(T_pb @ inv)
        out.append((t_ns, parent, root, t, q))
    return out


@dataclass
class TFEdits:
    """What to change in a bag's TF (for :func:`scoop.merge.merge_bags`)."""
    drop: Optional[object] = None                # f(parent, child) -> True to remove
    static_tf: List = None                       # [(parent, child, t, q)] for /tf_static
    extra_tf: List = None                        # [(t_ns, parent, child, t, q)] for /tf
    notes: List[str] = None
    edges: Dict[str, Edge] = None                # the tree afterwards (static part exact)


def plan_edits(bags, drop=(), calibrations=(), odom=None) -> TFEdits:
    """Drop edges (``drop``: PARENT->CHILD patterns), add static transforms
    (``calibrations``: :class:`Calibration`, re-rooted as needed), and put in
    a new odometry (``odom`` = (poses, body frame, parent frame), see
    :func:`odom_transforms`), in that order, against the bags' TF tree."""
    edges = tf_edges(bags)
    notes: List[str] = []
    dropf = edge_matcher(drop) if drop else None
    if dropf:
        gone = [f"{e.parent} -> {c}" for c, e in edges.items() if dropf(e.parent, c)]
        if not gone:
            raise BagError(f"drop {' '.join(drop)}: matches no TF edge")
        notes.append("dropped " + ", ".join(gone))
        edges = {c: e for c, e in edges.items() if not dropf(e.parent, c)}
    static_tf: List = []
    if calibrations:
        static_tf, more = attach(list(calibrations), edges)
        notes += more
        for p, c, t, q in static_tf:
            edges[c] = Edge(p, c, True, matrix(t, q))
    extra_tf: List = []
    if odom is not None:
        poses, body, parent = odom
        extra_tf = odom_transforms(poses, body, parent, edges)
        root = extra_tf[0][2]
        notes.append(f"odometry {parent} -> {root}, {len(extra_tf)} poses "
                     f"(so {parent} -> {body} is the trajectory)")
        edges[root] = Edge(parent, root, False)
    return TFEdits(dropf, static_tf, extra_tf, notes, edges)


def format_tree(edges: Dict[str, Edge]) -> str:
    """The tree as text, roots first: static edges ``──``, dynamic ``~~``."""
    kids: Dict[str, List[str]] = {}
    for c, e in edges.items():
        kids.setdefault(e.parent, []).append(c)
    roots = sorted({e.parent for e in edges.values()} - set(edges))
    lines: List[str] = []

    def walk(f, prefix, last, first):
        if first:
            lines.append(f)
        else:
            e = edges[f]
            lines.append(f"{prefix}{'└' if last else '├'}{'──' if e.static else '~~'} {f}")
        ch = sorted(kids.get(f, []))
        for i, c in enumerate(ch):
            walk(c, prefix + ("" if first else ("   " if last else "│  ")), i == len(ch) - 1,
                 False)
    for r in roots:
        walk(r, "", True, True)
    return "\n".join(lines) + "\n(── static, ~~ /tf)"


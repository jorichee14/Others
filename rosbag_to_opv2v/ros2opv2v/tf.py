# -*- coding: utf-8 -*-
"""The bag's static transform tree, so an extrinsic can be *looked up* instead
of typed.

The offline pipeline publishes every sensor's mounting below the robot's base
link on ``/tf_static`` (``mobile_1_base_link -> zed_camera_link -> ... ->
os_lidar_mobile1``), and the pose topics describe the base link. The extrinsic
the converter needs, base -> sensor, is therefore the composition of a chain of
static edges, and copying its numbers into a config by hand is the one place a
sign or an inverse goes wrong without anything downstream noticing. ``extrinsic:
tf`` with a ``frame_id`` resolves it from the bag at run time; the resolved
values are written into the conversion report, so what was used is on record.

Only ``/tf_static`` is read. A sensor whose mounting is published on ``/tf``
(a moving edge) is not a fixed extrinsic, and the lookup refuses it by not
finding a path, naming the frames it does know.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np

from .geometry import invert, quat_to_matrix


class TfError(RuntimeError):
    """A frame is unknown, or two frames are not connected by static edges."""


class StaticTf:
    """Static edges ``parent -> child`` as 4x4 matrices (child pose in parent)."""

    def __init__(self) -> None:
        self.edges: Dict[Tuple[str, str], np.ndarray] = {}

    def add(self, parent: str, child: str, matrix: np.ndarray) -> None:
        self.edges[(str(parent), str(child))] = np.asarray(matrix, dtype=float)

    def add_message(self, msg) -> int:
        """Every transform of a ``tf2_msgs/TFMessage``; returns how many."""
        n = 0
        for tr in getattr(msg, "transforms", []):
            t = tr.transform.translation
            q = tr.transform.rotation
            matrix = quat_to_matrix(float(q.x), float(q.y), float(q.z), float(q.w))
            matrix[:3, 3] = [float(t.x), float(t.y), float(t.z)]
            self.add(tr.header.frame_id, tr.child_frame_id, matrix)
            n += 1
        return n

    @property
    def frames(self) -> List[str]:
        names = set()
        for parent, child in self.edges:
            names.add(parent)
            names.add(child)
        return sorted(names)

    def path(self, source: str, target: str) -> List[Tuple[str, str, bool]]:
        """The edges from ``source`` to ``target`` as ``(parent, child, forward)``,
        ``forward`` False when the edge is walked child -> parent."""
        if source == target:
            return []
        neighbours: Dict[str, List[Tuple[str, Tuple[str, str], bool]]] = {}
        for parent, child in self.edges:
            neighbours.setdefault(parent, []).append((child, (parent, child), True))
            neighbours.setdefault(child, []).append((parent, (parent, child), False))
        unknown = [f for f in (source, target) if f not in neighbours]
        if unknown:
            raise TfError(
                f"frame(s) {unknown} are not on /tf_static. Frames there: {self.frames}")
        previous: Dict[str, Tuple[str, Tuple[str, str], bool]] = {}
        queue = deque([source])
        seen = {source}
        while queue:
            node = queue.popleft()
            if node == target:
                break
            for nxt, edge, forward in neighbours.get(node, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                previous[nxt] = (node, edge, forward)
                queue.append(nxt)
        if target not in previous:
            raise TfError(
                f"no static chain from {source!r} to {target!r}: one of the edges "
                f"between them is published on /tf (moving) or missing. Frames on "
                f"/tf_static: {self.frames}")
        out: List[Tuple[str, str, bool]] = []
        node = target
        while node != source:
            prev, (parent, child), forward = previous[node]
            out.append((parent, child, forward))
            node = prev
        out.reverse()
        return out

    def lookup(self, source: str, target: str) -> np.ndarray:
        """``T_source_target``: the pose of ``target`` in ``source``, i.e. the
        extrinsic ``source -> target`` the converter composes with a base pose."""
        matrix = np.identity(4)
        for parent, child, forward in self.path(source, target):
            edge = self.edges[(parent, child)]
            matrix = matrix @ (edge if forward else invert(edge))
        return matrix


def read_static_tf(reader, topic: str = "/tf_static") -> StaticTf:
    """Every transform on ``topic``; later messages override earlier ones."""
    tree = StaticTf()
    if topic not in reader.topics():
        return tree
    for _topic, _stamp, msg in reader.iter_messages([topic]):
        tree.add_message(msg)
    return tree


def describe_chain(tree: StaticTf, source: str, target: str) -> str:
    """``a -> b -> c`` (``<-`` for an edge walked against its direction)."""
    parts = [source]
    for parent, child, forward in tree.path(source, target):
        parts.append(("-> " if forward else "<- ") + (child if forward else parent))
    return " ".join(parts)

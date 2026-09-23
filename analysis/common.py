#!/usr/bin/env python3
"""Helpers shared by the per-topic analyses.

Nothing here is specific to Wi-Fi, CSI or NTP: agent naming, the extraction
loaders, the ground-truth pose join, the map background, and the plotting
constants that keep every figure in the set looking like one figure.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

# The CSI publisher uses /mobile1 and /mobile2 while everything else uses
# /mobile_1 and /mobile_2; without this the same robot appears as two agents.
NODE_ALIASES = {"mobile1": "mobile_1", "mobile2": "mobile_2"}
# One fixed colour per agent, and a fixed categorical order for anything finer
# grained (a radio, a stream), so a series keeps its colour across every figure.
AGENT_COLOR = {"mobile_1": "#2a78d6", "mobile_2": "#eb6834", "infra_1": "#1baf7a"}
LINK_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
# RSSI is a magnitude, so one hue, light (weak) to dark (strong) -- the same ramp
# and the same limits in every coverage panel, so colours compare across agents.
RSSI_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
BAD_COLOR = "#e34948"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def node_of_topic(topic: str) -> str:
    """'/mobile1/csi' -> 'mobile_1'; '/mobile_2/wifi/status' -> 'mobile_2'."""
    node = topic.strip("/").split("/")[0]
    return NODE_ALIASES.get(node, node)


def link_of_topic(topic: str) -> tuple[str, str]:
    """(receiver, transmitter) from a CSI topic name.

    A CSI topic is one RECEIVER's view of one TRANSMITTER's channel, and the
    capture node names it <receiver namespace>/<transmitter>/csi:

        /mobile_1_sniffer/infra_1/csi  ->  ('mobile_1_sniffer', 'infra_1')
        /infra_1/csi                   ->  ('',                 'infra_1')

    An unnamespaced capture records no receiver, which is fine with one sniffer
    and ambiguous with two -- hence the namespace. Taking the FIRST segment
    instead (as node_of_topic does) collapses every stream from one namespaced
    sniffer onto the same name, which loses a channel silently.
    """
    parts = [p for p in topic.strip("/").split("/") if p]
    if parts and parts[-1] == "csi":
        parts = parts[:-1]
    if not parts:
        return "", ""
    tx = NODE_ALIASES.get(parts[-1], parts[-1])
    rx = "/".join(parts[:-1])
    return NODE_ALIASES.get(rx, rx), tx


def color_for(node: str) -> str:
    """'mobile_1' and 'mobile_1/A' share the agent's colour."""
    return AGENT_COLOR.get(node.split("/")[0], "#4a3aa7")


def read_pcd_xy(path: Path, max_pts: int = 120_000):
    """Points of a .pcd as (N,2) xy for the figure background.

    open3d if it is installed, otherwise a minimal reader for the ascii and
    uncompressed-binary layouts the mapping pipeline writes."""
    try:
        import open3d as o3d  # noqa: PLC0415

        P = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    except Exception:
        P = _read_pcd_raw(path)
    if P is None or not len(P):
        return None
    P = P[np.isfinite(P).all(1)]
    if len(P) > max_pts:
        P = P[np.linspace(0, len(P) - 1, max_pts).astype(int)]
    return P[:, :2]


def _read_pcd_raw(path: Path):
    with open(path, "rb") as f:
        fields, sizes, types, counts, npts, data = [], [], [], [], 0, None
        while True:
            line = f.readline()
            if not line:
                return None
            tok = line.decode("ascii", "replace").split()
            if not tok:
                continue
            key = tok[0].upper()
            if key == "FIELDS":
                fields = tok[1:]
            elif key == "SIZE":
                sizes = [int(x) for x in tok[1:]]
            elif key == "TYPE":
                types = tok[1:]
            elif key == "COUNT":
                counts = [int(x) for x in tok[1:]]
            elif key == "POINTS":
                npts = int(tok[1])
            elif key == "DATA":
                data = tok[1].lower()
                break
        if data == "binary_compressed":
            return None  # lzf; open3d handles it, this reader does not
        counts = counts or [1] * len(fields)
        names, fmts = [], []
        for fn, sz, ty, ct in zip(fields, sizes, types, counts):
            dt = {"F": "f", "U": "u", "I": "i"}.get(ty.upper(), "u") + str(sz)
            for c in range(ct):
                names.append(fn if ct == 1 else f"{fn}_{c}")
                fmts.append(dt)
        if data == "ascii":
            A = np.loadtxt(f, dtype=np.float64, max_rows=npts, ndmin=2)
            idx = [names.index(k) for k in ("x", "y", "z") if k in names]
            return A[:, idx] if len(idx) == 3 else None
        arr = np.frombuffer(f.read(), dtype=np.dtype(list(zip(names, fmts))), count=npts)
        return np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(float)


def load_poses(extracts: Path, suffix: str):
    """{agent: (t_s_array, xy array)} from PoseStamped or Odometry topics.

    `suffix` is one topic tail or several separated by commas, e.g.
    "zed/pose,visual_slam/tracking/vo_pose" when the agents name their pose
    topic differently; the first match per agent wins."""
    out = {}
    for one in suffix.split(","):
        one = one.strip().strip("/").replace("/", "__")
        for f in sorted(glob.glob(str(extracts / f"*{one}.parquet"))):
            topic = "/" + Path(f).stem.replace("__", "/")
            agent = node_of_topic(topic)
            if agent in out:
                continue
            df = pd.read_parquet(f)
            if "pose.position.x" in df.columns:
                cols = ["pose.position.x", "pose.position.y"]
            elif "pose.pose.position.x" in df.columns:          # nav_msgs/Odometry
                cols = ["pose.pose.position.x", "pose.pose.position.y"]
            else:
                continue
            out[agent] = (df["log_time_ns"].to_numpy(), df[cols].to_numpy(float))
    return out



def load_glob(extracts: Path, pattern: str):
    frames = []
    for f in sorted(glob.glob(str(extracts / pattern))):
        df = pd.read_parquet(f)
        df["topic"] = "/" + Path(f).stem.replace("__", "/")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None

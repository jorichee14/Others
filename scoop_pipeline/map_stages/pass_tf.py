#!/usr/bin/env python3
"""
The complete TF tree of a pass, from the pipeline's outputs, for the merge
(processing/merge_session.py adds it when a pipeline config's dataset is
that pass).

Every mobile platform hangs from its own origin frame, fixed below the board
it started at (map ── board ── map_zed, map ── board_rs ── map_realsense):

  board ── map_zed          mobile_1's origin: the ZED's map of this session
                            (05: T_map_mapzed in anchor_frame.json, or
                            boards_<tag>.json for a run); map_zed ~~ odom_zed
                            stays as recorded, so the ZED's own topics show in map
     map_zed ~~ zed_camera_link
                            mobile_1 from the LiDAR trajectory (a run: 04's
                            traj_<name>_in_cam.tum; a mapping pass: 02's refined
                            trajectory through 05's T_N_world), replacing the
                            ZED's odom_zed ~~ zed_camera_link: the ZED's tracking
                            is not trusted. Its whole tree (ZED, os_sensor,
                            radars) follows. (Without T_map_mapzed: map ~~
                            zed_camera_link.)
  board_rs ── map_realsense mobile_2's origin (07's camera "origin_frame", below
                            its "board"): at its place from 07 while it is parked
     map_realsense ── camera_link
  map ── <board frames>     05's boards (the reference pass's, plus the ones only
                            a run measured)
  map ── <camera tree>      07's cameras_in_map.yaml (arducam, RealSense): the
                            root of each camera's tree is hung under map (or
                            under its "origin_frame") so that map -> child_frame
                            is 07's pose. An edge from map_zed
                            into a camera (the arducam's ChArUco pose, in the
                            ZED's drifting map) is dropped first.

    python3 map_stages/pass_tf.py <pipeline_config.json> <bag> [<bag> ...]   # print the tree
"""
import glob
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from pipeline_common import (load_config, load_pipeline, load_cameras_yaml,  # noqa: E402
                             processed_dir_for, make_T, DEFAULT_TOPICS)
from scoop import bag as scoop_bag, tftree                                   # noqa: E402


def find_config(raw_dir, stages_dir=HERE):
    """The pipeline config whose dataset is the pass of data/raw/<date>/<pass>,
    or None. A run's config (pipeline_config_<pass>.json) wins over the base."""
    parts = os.path.normpath(os.path.abspath(os.path.expanduser(raw_dir))).split(os.sep)
    want = tuple(parts[-2:])
    hits = []
    for path in sorted(glob.glob(os.path.join(stages_dir, "pipeline_config*.json"))):
        try:
            d = load_config(path).get("dataset", {})
            if d.get("processed"):
                proc = os.path.expanduser(d["processed"])
            elif d.get("bag") and not d.get("out_dir"):
                proc = processed_dir_for(d["bag"])
            else:
                continue
        except (SystemExit, ValueError, OSError, KeyError):
            continue
        if tuple(os.path.normpath(proc).split(os.sep)[-2:]) == want:
            hits.append(path)
    return hits[-1] if hits else None


def _frame_id(bags, topic):
    """header.frame_id of the first message of `topic` in `bags`."""
    for b in bags:
        r = scoop_bag.open_bag(b)
        if topic not in r.topics():
            continue
        for _, _, _, msg in scoop_bag.iter_mcap_records(r, [topic]):
            d = msg.data
            n = struct.unpack_from("<I", d, 12)[0]
            return d[16:16 + n].rstrip(b"\0").decode()
    return None


def _T(rec):
    return make_T(rec["xyz"], rec["qxyzw"])


def _odom_edge(body, edges):
    """The first /tf (moving) edge above `body`: the odometry its tree hangs from."""
    f = body
    while f in edges:
        e = edges[f]
        if not e.static:
            return e.parent, f
        f = e.parent
    return None


def _load_poses(path, T_left=None):
    out = []
    for t_ns, T in tftree.read_tum(path):
        out.append((t_ns, T if T_left is None else T_left @ T))
    return out


def plan(P, edges, bags, log=print):
    """-> (drop [(parent, child)], static [(parent, child, t, q)],
    extra /tf [(t_ns, parent, child, t, q)], edges afterwards), against the
    bags' tree `edges` (child -> tftree.Edge, the calibrations already in)."""
    edges = dict(edges)
    s03 = P.cfg.get("05_anchor", {})
    mapf = s03.get("map_frame", "map")
    mapzed = s03.get("mapzed_frame", "map_zed")
    cam = s03.get("cam_frame", "zed_left_camera_optical_frame")
    frames = set(edges) | {e.parent for e in edges.values()}
    drop, static, extra = [], [], []

    def add_static(parent, child, T, why):
        t, q = tftree.to_tq(T)
        static.append((parent, child, t, q))
        edges[child] = tftree.Edge(parent, child, True, T)
        log("    %s -> %s  (%s)" % (parent, child, why))

    def drop_edge(parent, child, why):
        drop.append((parent, child))
        edges.pop(child, None)
        log("    dropped %s -> %s  (%s)" % (parent, child, why))

    if mapf in edges:
        raise SystemExit("%s already has a parent in the bags (%s)" % (mapf, edges[mapf].parent))

    # -- anchor files: this pass's, or for a run the reference's + this run's
    af_path = P.anchor_frame()
    af = json.load(open(af_path)) if os.path.exists(af_path) else {}
    run = json.load(open(P.outp("boards_{tag}.json"))) \
        if P.reference and os.path.exists(P.outp("boards_{tag}.json")) else {}
    session = run if P.reference else af
    if not af:
        log("    (no %s: no boards, map_zed or mapping trajectory)" % af_path)
    boards = dict((af.get("boards") or {}))
    for n, rec in (run.get("boards") or {}).items():
        boards.setdefault(n, rec)
    bframe = {n: rec.get("frame") or n for n, rec in boards.items()}
    s04 = P.cfg.get("07_build_cameras") or {}
    cfg_cams = {c["name"]: c for c in s04.get("cameras", [])}
    origins = {c.get("origin_frame") for c in cfg_cams.values()} - {None}

    # -- a bag that carries this tree already (merged after the map stages, or
    # 10's): what was put in below map, the boards and the parked origins goes,
    # and is put in again from the outputs as they are now
    old = [(e.parent, c) for c, e in edges.items()
           if e.parent == mapf or e.parent in set(bframe.values()) | origins]
    for p_, c in old:
        drop_edge(p_, c, "put in again from the pipeline outputs")
    frames = set(edges) | {e.parent for e in edges.values()}

    # -- boards, fixed in map
    T_board = {}
    for n, rec in sorted(boards.items()):
        fr = bframe[n]
        if fr in edges or fr in frames:
            continue
        T_board[n] = _T(rec)
        add_static(mapf, fr, T_board[n], "05: board '%s'" % n)

    def under_board(name, child, T_map_child, why):
        """child fixed in map, below board `name` when it is placed."""
        if name in T_board:
            add_static(bframe[name], child, np.linalg.inv(T_board[name]) @ T_map_child,
                       why + "; below board '%s'" % name)
        else:
            add_static(mapf, child, T_map_child, why)

    # -- mobile_1: its origin map_zed below the start board, the trajectory below it
    T_mz = session.get("T_map_mapzed")
    T_mz = np.array(T_mz, float) if T_mz is not None and mapzed not in edges else None
    origin = mapzed if T_mz is not None else mapf
    if P.reference:
        traj = P.lidar_track_traj("_in_cam")
        body, T_left = cam, None
    else:
        traj = P.outp(P.dataset["traj"]) if P.dataset.get("traj") else None
        pts = (P.cfg.get("04_reference") or {}).get("points_topic") \
            or P.dataset.get("points_topic") or DEFAULT_TOPICS["points_topic"]
        body = _frame_id(bags, pts)
        T_left = np.array(af["T_N_world"], float) if "T_N_world" in af else None
        if body is None or T_left is None:
            traj = None
    if traj and os.path.exists(traj):
        oe = _odom_edge(body, edges)
        if oe:
            drop_edge(oe[0], oe[1], "the old odometry; the LiDAR trajectory replaces it")
        if T_mz is not None:                            # poses in map -> in map_zed
            T_left = np.linalg.inv(T_mz) @ (np.eye(4) if T_left is None else T_left)
        poses = _load_poses(traj, T_left)
        extra = tftree.odom_transforms(poses, body, origin, edges)
        root = extra[0][2]
        edges[root] = tftree.Edge(origin, root, False)
        log("    %s ~~ %s  (%d poses of %s from %s)"
            % (origin, root, len(extra), body, os.path.relpath(traj, P.out_dir)))
    else:
        log("    (no LiDAR trajectory %s: mobile_1 stays on the ZED's tracking)" % traj)
    if T_mz is not None:
        start = session.get("anchor_board") or af.get("anchor_board") or "anchor"
        under_board(start, mapzed, T_mz, "mobile_1's origin, 05: the ZED's map at the start board")

    # -- infra / parked cameras from 07
    cy = P.stage("07_build_cameras")["output"] if "07_build_cameras" in P.cfg else None
    if cy and os.path.exists(cy):
        _, cams = load_cameras_yaml(cy)
        for c in cams:
            child = c["child_frame"]
            T = make_T(c["translation"], c["quaternion_xyzw"])
            e = edges.get(child)
            if e is not None and e.parent == mapzed:
                drop_edge(mapzed, child, "its pose in the ZED's map; 07's replaces it")
            root = tftree._root(child, edges) if child in edges else child
            if root == mapf:
                log("    (%s is already below %s: 07's %s skipped)" % (child, mapf, c["name"]))
                continue
            try:
                T_rc = tftree._static_pose(root, child, edges)
            except scoop_bag.BagError:
                log("    (%s moves in the bags: 07's static pose of %s skipped)"
                    % (child, c["name"]))
                continue
            T_root = T @ np.linalg.inv(T_rc)
            why = ("07: %s, so %s -> %s is its pose" % (c["name"], mapf, child)
                   if root != child else "07: %s" % c["name"])
            cc = cfg_cams.get(c["name"], {})
            og = cc.get("origin_frame")
            if og and og not in frames and og not in edges:
                under_board(cc.get("board"), og, T_root,
                            "%s's origin, where it is parked; %s" % (c["name"], why))
                add_static(og, root, np.eye(4), "parked")
            else:
                add_static(mapf, root, T_root, why)
    else:
        log("    (no %s: infra/parked cameras not placed)" % cy)
    return drop, static, extra, edges


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    P = load_pipeline(sys.argv[1])
    bags = sys.argv[2:]
    edges = tftree.tf_edges(bags)
    print("TF from the pipeline outputs (%s):" % P.out_dir)
    _, _, _, after = plan(P, edges, bags)
    print(tftree.format_tree(after))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
The complete TF tree of a pass, from the pipeline's outputs, for the merge
(processing/merge_session.py adds it when a pipeline config's dataset is
that pass).

Every mobile platform hangs from its own origin frame, fixed below the board
it started at (map ── board ── map_zed, map ── board_rs ── map_realsense):

  board ── map_zed          mobile_1's origin: the ZED's map of this session
                            (05: T_map_mapzed in anchor_frame.json, or
                            boards_<tag>.json for a run; a run without 05, coop:
                            where the ZED's own tracking puts the camera at the
                            LiDAR trajectory's first pose); map_zed ~~ odom_zed
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
  <board> ── <origin>       a robot with a track of its own (04 "tracks", e.g.
     <origin> ~~ <body>     mobile_2_depth in coop: map_realsense below rs_anchor's
                            board_rs at its first pose, camera_link moving below it)
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


def _tf_samples(bags, pairs, t_lo, t_hi):
    """/tf transforms of the (parent, child) `pairs` stamped in [t_lo, t_hi]
    (ns) -> {(parent, child): [(t_ns, T)]}, by stamp."""
    from mcap.reader import make_reader
    from scoop import rosmsg
    ts = rosmsg.typestore()
    out = {k: [] for k in pairs}
    for b in bags:
        r = scoop_bag.open_bag(b)
        if "/tf" not in r.topics():
            continue
        for f in r.files:
            with open(f, "rb") as fh:
                for sc, _, msg in make_reader(fh).iter_messages(
                        topics=["/tf"], start_time=t_lo - int(5e9), end_time=t_hi + int(5e9)):
                    for tr in ts.deserialize_cdr(msg.data, tftree.TFMSG).transforms:
                        k = (tr.header.frame_id.lstrip("/"), tr.child_frame_id.lstrip("/"))
                        if k not in out:
                            continue
                        t_ns = tr.header.stamp.sec * 1000000000 + tr.header.stamp.nanosec
                        if t_lo <= t_ns <= t_hi:
                            a, q = tr.transform.translation, tr.transform.rotation
                            out[k].append((t_ns, tftree.matrix([a.x, a.y, a.z],
                                                               [q.x, q.y, q.z, q.w])))
    for v in out.values():
        v.sort(key=lambda x: x[0])
    return out


def _mapzed_from_track(bags, edges, mapzed, body, poses, log=print):
    """T_map_mapzed for a pass without 05: the ZED's own tracking (map_zed ~~
    ... ~~ body in the bags) against the LiDAR trajectory (`poses`, body in
    map) at the trajectory's first poses where the ZED tracks -> T or None."""
    chain, f = [], body
    while f != mapzed:
        e = edges.get(f)
        if e is None:
            return None
        chain.append(e)
        f = e.parent
    dyn = [(e.parent, e.child) for e in chain if not e.static]
    t0 = poses[0][0]
    smp = _tf_samples(bags, dyn, t0, t0 + int(30e9))
    if any(not smp[k] for k in dyn):
        return None
    t_ref = max(smp[k][0][0] for k in dyn)          # every edge has a transform from here
    T = np.eye(4)
    for e in reversed(chain):                         # map_zed down to body
        if e.static:
            T = T @ e.T
        else:
            v = smp[(e.parent, e.child)]
            k = int(np.argmin([abs(t - t_ref) for t, _ in v]))
            T = T @ v[k][1]
    pt = np.array([t for t, _ in poses])
    k = int(np.argmin(np.abs(pt - t_ref)))
    if abs(pt[k] - t_ref) > int(0.2e9):
        return None
    log("    %s placed where the ZED's tracking puts %s at the LiDAR trajectory's pose "
        "%.2f s after its start" % (mapzed, body, (pt[k] - t0) * 1e-9))
    return poses[k][1] @ np.linalg.inv(T)


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
    for n, rec in P.extra_boards().items():            # dataset.boards_from
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
    if T_mz is None and P.reference and traj and os.path.exists(traj) and \
            mapzed in frames and mapzed not in edges and body in edges:
        # a run without 05 (coop): the ZED's map placed by its own tracking
        T_mz = _mapzed_from_track(bags, edges, mapzed, body, _load_poses(traj), log)
    origin = mapzed if T_mz is not None else mapf
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

    # -- the other robots with a track of their own (04 "tracks"): the origin
    # frame at the first pose, below the start board; the track below it
    for tr in (P.cfg.get("04_reference") or {}).get("tracks") or []:
        if not tr.get("enabled", True):
            continue
        body, og = tr.get("body_frame", "camera_link"), tr.get("origin_frame")
        traj = os.path.join(P.reference_dir(), "traj_%s.tum" % tr["name"])
        if not og or not os.path.exists(traj):
            log("    (no %s%s: %s not placed)" % (traj, "" if og else " / origin_frame",
                                                 tr["name"]))
            continue
        if body not in edges and body not in frames:
            log("    (%s not in the bags: %s not placed)" % (body, tr["name"]))
            continue
        oe = _odom_edge(body, edges)
        if oe:
            drop_edge(oe[0], oe[1], "the old odometry; %s replaces it" % tr["name"])
        poses = _load_poses(traj)
        T_og = poses[0][1]
        inv0 = np.linalg.inv(T_og)
        more = tftree.odom_transforms([(t, inv0 @ T) for t, T in poses], body, og, edges)
        root = more[0][2]
        edges[root] = tftree.Edge(og, root, False)
        extra = extra + more
        log("    %s ~~ %s  (%d poses of %s from %s)"
            % (og, root, len(more), body, os.path.relpath(traj, P.out_dir)))
        under_board(tr.get("board"), og, T_og, "%s's origin: its first pose" % tr["name"])
        frames |= {og}

    # -- infra / parked cameras from 07
    cy = P.stage("07_build_cameras")["output"] if "07_build_cameras" in P.cfg and \
        P.cfg["07_build_cameras"].get("enabled", True) else None
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

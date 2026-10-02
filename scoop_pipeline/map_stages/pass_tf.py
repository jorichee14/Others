#!/usr/bin/env python3
"""
The complete TF tree of a pass, from the pipeline's outputs, for the merge
(processing/merge_session.py adds it when a pipeline config's dataset is
that pass).

  map ── map_zed            the ZED's map of this session (03: T_map_mapzed in
                            anchor_frame.json, or boards_<tag>.json for a run);
                            map_zed ~~ odom_zed stays as recorded, so the ZED's
                            own topics show in map
  map ~~ zed_camera_link    mobile_1 from the LiDAR trajectory (a run: 08's
                            traj_<name>_in_cam.tum; a mapping pass: 01a's refined
                            trajectory through 03's T_N_world), replacing the
                            ZED's odom_zed ~~ zed_camera_link: the ZED's tracking
                            is not trusted. Its whole tree (ZED, os_sensor,
                            radars) follows.
  map ── <board frames>     03's boards (the reference pass's, plus the ones only
                            a run measured)
  map ── <camera tree>      04's cameras_in_map.yaml (arducam, RealSense): the
                            root of each camera's tree is hung under map so that
                            map -> child_frame is 04's pose. An edge from map_zed
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
    s03 = P.cfg.get("03_anchor", {})
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

    # -- mobile_1 trajectory
    if P.reference:
        traj = P.lidar_track_traj("_in_cam")
        body, T_left = cam, None
    else:
        traj = P.outp(P.dataset["traj"]) if P.dataset.get("traj") else None
        pts = (P.cfg.get("08_reference") or {}).get("points_topic") \
            or P.dataset.get("points_topic") or DEFAULT_TOPICS["points_topic"]
        body = _frame_id(bags, pts)
        T_left = np.array(af["T_N_world"], float) if "T_N_world" in af else None
        if body is None or T_left is None:
            traj = None
    if traj and os.path.exists(traj):
        oe = _odom_edge(body, edges)
        if oe:
            drop_edge(oe[0], oe[1], "the ZED's own tracking; the LiDAR trajectory replaces it")
        poses = _load_poses(traj, T_left)
        extra = tftree.odom_transforms(poses, body, mapf, edges)
        root = extra[0][2]
        edges[root] = tftree.Edge(mapf, root, False)
        log("    %s ~~ %s  (%d poses of %s from %s)"
            % (mapf, root, len(extra), body, os.path.relpath(traj, P.out_dir)))
    else:
        log("    (no LiDAR trajectory %s: mobile_1 stays on the ZED's tracking)" % traj)

    # -- map -> map_zed
    T_mz = session.get("T_map_mapzed")
    if T_mz is not None and mapzed in frames and mapzed not in edges:
        add_static(mapf, mapzed, np.array(T_mz, float), "03: the ZED's map at the start board")

    # -- boards
    boards = dict((af.get("boards") or {}))
    for n, rec in (run.get("boards") or {}).items():
        boards.setdefault(n, rec)
    for n, rec in sorted(boards.items()):
        fr = rec.get("frame") or n
        if fr in edges or fr in frames:
            continue
        add_static(mapf, fr, _T(rec), "03: board '%s'" % n)

    # -- infra / parked cameras from 04
    cy = P.stage("04_build_cameras")["output"] if "04_build_cameras" in P.cfg else None
    if cy and os.path.exists(cy):
        _, cams = load_cameras_yaml(cy)
        for c in cams:
            child = c["child_frame"]
            T = make_T(c["translation"], c["quaternion_xyzw"])
            e = edges.get(child)
            if e is not None and e.parent == mapzed:
                drop_edge(mapzed, child, "its pose in the ZED's map; 04's replaces it")
            root = tftree._root(child, edges) if child in edges else child
            if root == mapf:
                log("    (%s is already below %s: 04's %s skipped)" % (child, mapf, c["name"]))
                continue
            try:
                T_rc = tftree._static_pose(root, child, edges)
            except scoop_bag.BagError:
                log("    (%s moves in the bags: 04's static pose of %s skipped)"
                    % (child, c["name"]))
                continue
            add_static(mapf, root, T @ np.linalg.inv(T_rc),
                       "04: %s, so %s -> %s is its pose" % (c["name"], mapf, child)
                       if root != child else "04: %s" % c["name"])
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

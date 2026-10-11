#!/usr/bin/env python3
"""Extrinsics of all cameras in the ROOM FRAME, from the frame sets saved by capture.py --mode extr
plus the floor tags seen in any frame (intr or extr).

  python calibrate_extrinsics.py --data data --results results

Needs results/intrinsics/cam_N.json for every camera involved (calibrate_intrinsics.py).

Method
  1. Board pose per camera per event from solvePnP (initial values).
  2. Cameras that see 2+ surveyed floor tags get an initial room pose from PnP on the tags;
     the rest are chained through events they share with already-placed cameras.
  3. One bundle adjustment over all cameras and all board poses: reprojection of every ChArUco
     corner and every floor-tag corner (Huber loss). Floor tags pin the room frame.
Writes results/extrinsics/cameras.json (per camera: K, dist, R_cw, t_cw, position) and report.md.
"""
import argparse
import glob
import os
from collections import defaultdict

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from calib.common import (FloorTags, T_to_rt, inv_T, load_boards, load_intrinsics, project, read_json,
                          rt_to_T, write_json)


def pnp(obj, img, K, dist):
    ok, r, t = cv2.solvePnP(np.asarray(obj, np.float32), np.asarray(img, np.float32), K, dist,
                            flags=cv2.SOLVEPNP_ITERATIVE)
    return rt_to_T(r, t) if ok else None


def load_observations(data_dir, boards, tags, intr):
    """events: {event: {cam: (obj3 Nx3 in board frame, img Nx2)}}, tag_obs: [(cam, obj3 world, img)]."""
    events, tag_obs = defaultdict(dict), []
    world = tags.world_corners()
    for path in sorted(glob.glob(os.path.join(data_dir, "*", "*", "*", "*.json"))):
        j = read_json(path)
        cam = int(j["camera"])
        if cam not in intr:
            continue
        for tid, c in (j.get("tags") or {}).items():
            if int(tid) in world:
                tag_obs.append((cam, world[int(tid)], np.array(c, np.float32)))
        ch = j.get("charuco")
        if j.get("event") and ch and len(ch["ids"]) >= 8:
            b = boards[j["board"]]
            o, i = b.board.matchImagePoints(np.array(ch["corners"], np.float32).reshape(-1, 1, 2),
                                            np.array(ch["ids"], np.int32).reshape(-1, 1))
            events[j["event"]][cam] = (o.reshape(-1, 3), i.reshape(-1, 2))
    events = {e: v for e, v in events.items() if len(v) >= 2}
    return events, tag_obs


def initial_poses(events, tag_obs, intr):
    """T_cw per camera (world->camera) and T_wb per event (board->world)."""
    T_cb = {e: {c: pnp(o, i, *intr[c]) for c, (o, i) in v.items()} for e, v in events.items()}
    T_cw = {}
    by_cam = defaultdict(lambda: ([], []))
    for cam, obj, img in tag_obs:
        by_cam[cam][0].append(obj)
        by_cam[cam][1].append(img)
    for cam, (objs, imgs) in by_cam.items():
        if len(objs) >= 2:
            T = pnp(np.vstack(objs), np.vstack(imgs), *intr[cam])
            if T is not None:
                T_cw[cam] = T
    if not T_cw:  # no floor tags: gauge fixed by the most connected camera
        ref = max(set(c for v in events.values() for c in v), key=lambda c: sum(c in v for v in events.values()))
        T_cw[ref] = np.eye(4)
        print("WARNING: no surveyed floor tags seen; room frame = camera %d frame" % ref)
    changed = True
    while changed:  # chain unknown cameras through shared events
        changed = False
        for e, poses in T_cb.items():
            known = [c for c in poses if c in T_cw and poses[c] is not None]
            if not known:
                continue
            k = known[0]
            T_wb = inv_T(T_cw[k]) @ poses[k]
            for c, T in poses.items():
                if c not in T_cw and T is not None:
                    T_cw[c] = T @ inv_T(T_wb)
                    changed = True
    T_wb = {}
    for e, poses in T_cb.items():
        ks = [c for c in poses if c in T_cw and poses[c] is not None]
        if ks:
            T_wb[e] = inv_T(T_cw[ks[0]]) @ poses[ks[0]]
    return T_cw, T_wb


def bundle(events, tag_obs, intr, T_cw, T_wb, iters=50):
    cams = sorted(T_cw)
    evs = sorted(e for e in events if e in T_wb)
    ci = {c: i for i, c in enumerate(cams)}
    ei = {e: i for i, e in enumerate(evs)}
    x0 = np.concatenate([np.concatenate(T_to_rt(T_cw[c])) for c in cams] +
                        [np.concatenate(T_to_rt(T_wb[e])) for e in evs])
    nc = len(cams)
    blocks = []  # (cam, event or None, obj (board or world), img)
    for e in evs:
        for c, (o, i) in events[e].items():
            if c in ci:
                blocks.append((c, e, o, i))
    for c, o, i in tag_obs:
        if c in ci:
            blocks.append((c, None, o, i))
    n_res = sum(len(b[3]) * 2 for b in blocks)
    S = lil_matrix((n_res, len(x0)), dtype=int)
    row = 0
    for c, e, o, i in blocks:
        n = len(i) * 2
        S[row:row + n, ci[c] * 6:ci[c] * 6 + 6] = 1
        if e is not None:
            S[row:row + n, nc * 6 + ei[e] * 6:nc * 6 + ei[e] * 6 + 6] = 1
        row += n

    def residuals(x):
        out = []
        for c, e, o, i in blocks:
            rc, tc = x[ci[c] * 6:ci[c] * 6 + 3], x[ci[c] * 6 + 3:ci[c] * 6 + 6]
            if e is None:
                pw = o
            else:
                k = nc * 6 + ei[e] * 6
                Rb, _ = cv2.Rodrigues(x[k:k + 3])
                pw = o @ Rb.T + x[k + 3:k + 6]
            out.append((project(intr[c][0], intr[c][1], rc, tc, pw) - i).ravel())
        return np.concatenate(out)

    r0 = residuals(x0)
    sol = least_squares(residuals, x0, jac_sparsity=S, loss="huber", f_scale=2.0, max_nfev=iters,
                        x_scale="jac", verbose=0)
    x = sol.x
    out_cw = {c: rt_to_T(x[ci[c] * 6:ci[c] * 6 + 3], x[ci[c] * 6 + 3:ci[c] * 6 + 6]) for c in cams}
    # per camera rms after
    per_cam, row = defaultdict(list), 0
    r1 = residuals(x)
    for c, e, o, i in blocks:
        n = len(i) * 2
        per_cam[c].append(r1[row:row + n].reshape(-1, 2))
        row += n
    rms = {c: float(np.sqrt(np.mean(np.sum(np.vstack(v) ** 2, axis=1)))) for c, v in per_cam.items()}
    return out_cw, float(np.sqrt(np.mean(r0 ** 2))), float(np.sqrt(np.mean(r1 ** 2))), rms, len(evs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    boards, tags = load_boards(), FloorTags()
    intr = load_intrinsics(args.results)
    if not intr:
        raise SystemExit("no intrinsics in %s/intrinsics; run calibrate_intrinsics.py first" % args.results)
    events, tag_obs = load_observations(args.data, boards, tags, intr)
    print("%d events with 2+ cameras, %d floor-tag observations, %d cameras with intrinsics" % (
        len(events), len(tag_obs), len(intr)))
    if not events:
        raise SystemExit("no extrinsic events: run capture.py --mode extr and hold the board still")
    T_cw, T_wb = initial_poses(events, tag_obs, intr)
    missing = sorted(set(intr) - set(T_cw))
    if missing:
        print("WARNING: no path to cameras %s (they share no still board view with the others)" % missing)
    T_cw, rms0, rms1, per_cam, n_ev = bundle(events, tag_obs, intr, T_cw, T_wb)
    print("bundle adjustment: %d cameras, %d events, rms %.2f -> %.2f px" % (len(T_cw), n_ev, rms0, rms1))

    out = {}
    lines = ["# ED305 camera extrinsics", "", "Room frame from floor tags: %d tag observations. "
             "Bundle rms %.2f px over %d still-board events." % (len(tag_obs), rms1, n_ev), "",
             "| cam | x | y | z | rms px | tag obs |", "|---|---|---|---|---|---|"]
    tag_count = defaultdict(int)
    for c, _, _ in tag_obs:
        tag_count[c] += 1
    for c in sorted(T_cw):
        K, dist = intr[c]
        pos = (-T_cw[c][:3, :3].T @ T_cw[c][:3, 3])
        out[str(c)] = {"K": K, "dist": dist, "R_cw": T_cw[c][:3, :3], "t_cw": T_cw[c][:3, 3],
                       "position_world": pos, "rms_px": per_cam.get(c)}
        lines.append("| %d | %.3f | %.3f | %.3f | %.2f | %d |" % (c, *pos, per_cam.get(c, float("nan")), tag_count[c]))
    write_json(os.path.join(args.results, "extrinsics", "cameras.json"),
               {"frame": "room (floor tags), x y along walls, z up, metres", "cameras": out})
    lines += ["", "Cameras hang 2.5 to 2.7 m above the floor: z should be in that range.",
              "Re-run after any camera is moved. Verify with verify.py."]
    with open(os.path.join(args.results, "extrinsics", "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

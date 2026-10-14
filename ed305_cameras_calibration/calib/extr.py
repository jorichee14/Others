"""Extrinsics: observations from the saved frame sets, the calibrate/test split of events, initial
poses, the bundle adjustment, and the evaluation of a FIXED camera set on held-out events and on
floor tags (board pose refit per event, floor-point error per tag)."""
import glob
import os
from collections import defaultdict

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from calib.common import T_to_rt, inv_T, project, read_json, rt_to_T

TEST_EVERY = 5   # every 5th still-board event (in time order) is a TEST event


def pnp(obj, img, K, dist):
    ok, r, t = cv2.solvePnP(np.asarray(obj, np.float32), np.asarray(img, np.float32), K, dist,
                            flags=cv2.SOLVEPNP_ITERATIVE)
    return rt_to_T(r, t) if ok else None


def load_observations(data_dir, boards, tags, intr):
    """events: {event: {cam: (obj Nx3 in board frame, img Nx2)}} with 2+ cameras;
    tag_obs: [(cam, event or None, tag id, world 4x3, img 4x2)] from every frame (intr and extr)."""
    events, tag_obs = defaultdict(dict), []
    world = tags.world_corners()
    for path in sorted(glob.glob(os.path.join(data_dir, "*", "*", "*", "*.json"))):
        j = read_json(path)
        cam = int(j["camera"])
        if cam not in intr:
            continue
        for tid, c in (j.get("tags") or {}).items():
            if int(tid) in world:
                tag_obs.append((cam, j.get("event"), int(tid), world[int(tid)], np.array(c, np.float32)))
        ch = j.get("charuco")
        if j.get("event") and ch and len(ch["ids"]) >= 8:
            b = boards[j["board"]]
            o, i = b.board.matchImagePoints(np.array(ch["corners"], np.float32).reshape(-1, 1, 2),
                                            np.array(ch["ids"], np.int32).reshape(-1, 1))
            events[j["event"]][cam] = (o.reshape(-1, 3), i.reshape(-1, 2))
    events = {e: v for e, v in events.items() if len(v) >= 2}
    return events, tag_obs


def split_events(events, known=None):
    """{event: 'calib'|'test'}. Every 5th event in time order is a test event, except events whose
    split is already recorded (known, from an earlier events.csv), which keep it."""
    out = {}
    for k, e in enumerate(sorted(events)):
        if known and known.get(e) in ("calib", "test"):
            out[e] = known[e]
        else:
            out[e] = "test" if k % TEST_EVERY == TEST_EVERY - 1 else "calib"
    return out


def initial_poses(events, tag_obs, intr):
    """T_cw per camera (world->camera) and T_wb per event (board->world)."""
    T_cb = {e: {c: pnp(o, i, *intr[c]) for c, (o, i) in v.items()} for e, v in events.items()}
    T_cw = {}
    by_cam = defaultdict(lambda: ([], []))
    for cam, _, _, obj, img in tag_obs:
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
    """Joint refinement of all camera poses and all board poses. Returns
    (T_cw, T_wb, rms before, rms after, per-camera rms, number of events)."""
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
    for c, _, _, o, i in tag_obs:
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
    out_wb = {e: rt_to_T(x[nc * 6 + ei[e] * 6:nc * 6 + ei[e] * 6 + 3], x[nc * 6 + ei[e] * 6 + 3:nc * 6 + ei[e] * 6 + 6])
              for e in evs}
    per_cam, row = defaultdict(list), 0
    r1 = residuals(x)
    for c, e, o, i in blocks:
        n = len(i) * 2
        per_cam[c].append(r1[row:row + n].reshape(-1, 2))
        row += n
    rms = {c: float(np.sqrt(np.mean(np.sum(np.vstack(v) ** 2, axis=1)))) for c, v in per_cam.items()}
    return out_cw, out_wb, float(np.sqrt(np.mean(r0 ** 2))), float(np.sqrt(np.mean(r1 ** 2))), rms, len(evs)


# ---------------------------------------------------------------- evaluation with fixed cameras
def fit_board(obs, T_cw, intr):
    """Pose of a still board (T board->world) from all cameras that saw it, cameras held FIXED.
    Returns (T_wb, {cam: rms_px}) or (None, {})."""
    cams = [c for c in obs if c in T_cw]
    if not cams:
        return None, {}
    c0 = max(cams, key=lambda c: len(obs[c][1]))
    T_cb = pnp(*obs[c0], *intr[c0])
    if T_cb is None:
        return None, {}
    x0 = np.concatenate(T_to_rt(inv_T(T_cw[c0]) @ T_cb))
    rt = {c: T_to_rt(T_cw[c]) for c in cams}

    def res(x):
        Rb, _ = cv2.Rodrigues(x[:3])
        return np.concatenate([(project(intr[c][0], intr[c][1], rt[c][0], rt[c][1], obs[c][0] @ Rb.T + x[3:])
                                - obs[c][1]).ravel() for c in cams])

    sol = least_squares(res, x0, loss="huber", f_scale=2.0, max_nfev=50)
    r, per, row = res(sol.x), {}, 0
    for c in cams:
        n = len(obs[c][1]) * 2
        per[c] = float(np.sqrt(np.mean(np.sum(r[row:row + n].reshape(-1, 2) ** 2, axis=1))))
        row += n
    return rt_to_T(sol.x[:3], sol.x[3:]), per


def evaluate_events(events, names, T_cw, intr):
    """{event: (T_wb, {cam: rms_px})} for the named events, cameras fixed."""
    out = {}
    for e in names:
        T, per = fit_board(events[e], T_cw, intr)
        if T is not None:
            out[e] = (T, per)
    return out


def floor_error(cam, T_cw, K, dist, img4, world4):
    """Where the tag centre seen by this camera lands on the tag's plane, versus the surveyed centre (m)."""
    c = img4.mean(0).reshape(1, 1, 2).astype(np.float64)
    n = cv2.undistortPoints(c, K, dist).reshape(2)
    R_wc = T_cw[:3, :3].T
    pos = -R_wc @ T_cw[:3, 3]
    d = R_wc @ np.array([n[0], n[1], 1.0])
    z0 = world4[:, 2].mean()
    if abs(d[2]) < 1e-6:
        return float("nan")
    s = (z0 - pos[2]) / d[2]
    p = pos + s * d
    return float(np.linalg.norm(p[:2] - world4[:, :2].mean(0)))


def evaluate_tags(tag_obs, T_cw, intr):
    """Per tag observation: dict(camera, event, tag, rms_px, floor_err_m), cameras fixed."""
    out = []
    for cam, e, tid, world4, img4 in tag_obs:
        if cam not in T_cw:
            continue
        K, dist = intr[cam]
        r, t = T_to_rt(T_cw[cam])
        pr = project(K, dist, r, t, world4)
        out.append({"camera": cam, "event": e or "", "tag": tid,
                    "rms_px": float(np.sqrt(np.mean(np.sum((pr - img4) ** 2, axis=1)))),
                    "floor_err_m": floor_error(cam, T_cw[cam], K, dist, img4, world4)})
    return out


def per_camera(evals, split_of):
    """{cam: {'calib': [rms...], 'test': [rms...]}} from evaluate_events output."""
    out = defaultdict(lambda: {"calib": [], "test": []})
    for e, (T, per) in evals.items():
        for c, v in per.items():
            out[c][split_of[e]].append(v)
    return out


def rms_of(values):
    v = [x for x in values if np.isfinite(x)]
    return float(np.sqrt(np.mean(np.square(v)))) if v else float("nan")

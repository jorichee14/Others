#!/usr/bin/env python3
"""
Synthetic-room self-test for stages [6] detect and [7] synthesize.

There is no bag here, no camera and no GPU. What there IS is a room with known
ground truth, real camera poses, and the ACTUAL projection kernel stage [6]
uses -- project_visible() imported straight out of 01_build_map.py. The only
thing stubbed is the detector, which is exactly the part that cannot be
verified without weights, and stubbing it lets us inject the failure mode that
matters: masks that BLEED past the silhouette onto the wall behind, which is
what every guard downstream exists to survive.

The room contains the two cases from the brief:

  a TV flush on a wall   -> must be REMOVED and the wall behind it repaired,
                            because the LiDAR never measured that patch
  a chair on the floor   -> must be REPLACED, meaning the measured points stay
                            and a fitted asset joins them, upright, at the
                            right yaw, with its base on the floor

and the structure the rules depend on: floor, ceiling and four walls, none of
which YOLO has a class for.

    python3 test_semantics.py [-v]
"""

import importlib.util
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pipeline_detect as pdet          # noqa: E402
import pipeline_assets as past          # noqa: E402


def _load_stage01():
    """Import 01_build_map.py, whose name is not a valid module identifier.

    Worth the awkwardness: testing a reimplementation of project_visible would
    verify nothing about the code that actually runs."""
    spec = importlib.util.spec_from_file_location(
        "stage01", os.path.join(HERE, "01_build_map.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S01 = _load_stage01()

VERBOSE = "-v" in sys.argv
FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)
    return ok


# =============================================================================
# SYNTHETIC ROOM
# =============================================================================
ROOM = (6.0, 5.0, 2.6)          # x, y, z extents; origin at a floor corner
TV_C = np.array([3.0, 0.0, 1.40])
TV_WH = (1.10, 0.65)
TV_THICK = 0.06
CHAIR_XY = np.array([2.0, 2.6])
CHAIR_YAW = np.deg2rad(40.0)
CHAIR_SCALE = 1.12


def grid(u0, u1, v0, v1, step):
    u = np.arange(u0, u1 + 1e-9, step)
    v = np.arange(v0, v1 + 1e-9, step)
    U, V = np.meshgrid(u, v, indexing="ij")
    return np.stack([U.ravel(), V.ravel()], 1)


def build_room(step=0.025, rng=None):
    """Room surfaces as points, with the TV's wall patch OCCLUDED.

    The occlusion is the whole point of the fill test: a LiDAR cannot measure
    the wall behind a TV, so those points must genuinely not exist. A test that
    left them in would 'pass' hole filling while filling nothing."""
    rng = rng or np.random.default_rng(3)
    X, Y, Z = ROOM
    parts, kinds = [], []

    g = grid(0, X, 0, Y, step)
    parts.append(np.column_stack([g[:, 0], g[:, 1], np.zeros(len(g))]))
    kinds.append("floor")
    parts.append(np.column_stack([g[:, 0], g[:, 1], np.full(len(g), Z)]))
    kinds.append("ceiling")

    # wall y = 0, minus the rectangle the TV hides
    g = grid(0, X, 0, Z, step)
    hid = ((np.abs(g[:, 0] - TV_C[0]) < TV_WH[0] / 2)
           & (np.abs(g[:, 1] - TV_C[2]) < TV_WH[1] / 2))
    gg = g[~hid]
    parts.append(np.column_stack([gg[:, 0], np.zeros(len(gg)), gg[:, 1]]))
    kinds.append("wall_y0")

    g = grid(0, X, 0, Z, step)
    parts.append(np.column_stack([g[:, 0], np.full(len(g), Y), g[:, 1]]))
    kinds.append("wall_y1")
    g = grid(0, Y, 0, Z, step)
    parts.append(np.column_stack([np.zeros(len(g)), g[:, 0], g[:, 1]]))
    kinds.append("wall_x0")
    parts.append(np.column_stack([np.full(len(g), X), g[:, 0], g[:, 1]]))
    kinds.append("wall_x1")

    # TV: front face plus a shallow rim, standing proud of the wall
    g = grid(-TV_WH[0] / 2, TV_WH[0] / 2, -TV_WH[1] / 2, TV_WH[1] / 2, step)
    tv = np.column_stack([TV_C[0] + g[:, 0],
                          np.full(len(g), TV_THICK),
                          TV_C[2] + g[:, 1]])
    parts.append(tv)
    kinds.append("tv")

    # chair: the real asset, scaled/rotated/noised, with its far side removed
    # to imitate a one-sided scan
    lib = past.AssetLibrary(os.path.join(HERE, "assets"))
    meta = lib.pick("chair")
    ap, _ = lib.points(meta, 12000)
    ch = (ap * CHAIR_SCALE) @ past.rotz(CHAIR_YAW).T
    ch[:, 0] += CHAIR_XY[0]
    ch[:, 1] += CHAIR_XY[1]
    ch = ch[ch[:, 1] < CHAIR_XY[1] + 0.16]          # seen from -y only
    ch = ch + rng.normal(0, 0.004, ch.shape)
    parts.append(ch)
    kinds.append("chair")

    idx = {}
    a = 0
    for p, k in zip(parts, kinds):
        idx[k] = np.arange(a, a + len(p))
        a += len(p)
    pts = np.vstack(parts)

    cols = np.full((len(pts), 3), 0.72)
    cols[idx["floor"]] = (0.45, 0.40, 0.36)
    cols[idx["tv"]] = (0.08, 0.08, 0.10)
    cols[idx["chair"]] = (0.30, 0.55, 0.35)
    return pts, cols, idx


class FakeSensor:
    fx = fy = 480.0
    cx, cy = 320.0, 240.0


W, H = 640, 480


def look_at(eye, target):
    """Camera-to-world for the OpenCV convention project_visible assumes."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, (0.0, 0.0, 1.0))
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([r, d, f])
    T[:3, 3] = eye
    return T


def dilate(mask, px):
    m = mask
    for _ in range(int(px)):
        e = m.copy()
        e[1:, :] |= m[:-1, :]
        e[:-1, :] |= m[1:, :]
        e[:, 1:] |= m[:, :-1]
        e[:, :-1] |= m[:, 1:]
        m = e
    return m


def stub_masks(vis_idx, uu, vv, member, bleed_px):
    """A detector that is right about the object and wrong at its edges.

    Real masks overshoot the silhouette by a few pixels; that overshoot lands
    on whatever is behind, which for a wall-mounted TV is the wall it is
    supposed to be distinguished from. Reproducing it is the only way to test
    that the guards work."""
    m = np.zeros((H, W), bool)
    hit = np.isin(vis_idx, member)
    if not hit.any():
        return None
    m[vv[hit], uu[hit]] = True
    return dilate(m, bleed_px)


# =============================================================================
# TEST
# =============================================================================
def main():
    rng = np.random.default_rng(11)
    pts, cols, gt = build_room(rng=rng)
    N = len(pts)
    print(f"synthetic room: {N} points "
          f"(tv {gt['tv'].size}, chair {gt['chair'].size})\n")

    # ---- [6] projection + stubbed masks + vote fusion --------------------- #
    print("[6] detect")
    CLS = {"tv": 62, "chair": 56}
    n_cls = 80
    votes = pdet.VoteAccumulator(N, n_cls)
    tracker = pdet.InstanceTracker(N, link_frac=0.30)
    index = S01.BlockIndex(pts, block=1.0)

    poses = []
    for x in (2.2, 2.7, 3.2, 3.7):
        poses.append(look_at((x, 2.6, 1.45), (TV_C[0], 0.0, TV_C[2])))
    for a in np.linspace(-0.9, 0.9, 5):
        eye = (CHAIR_XY[0] + a, CHAIR_XY[1] - 1.7, 1.30)
        poses.append(look_at(eye, (CHAIR_XY[0], CHAIR_XY[1], 0.5)))

    n_frames = 0
    for Twc in poses:
        vis = S01.project_visible(index, pts, Twc, FakeSensor, W, H, 12.0)
        if vis is None:
            continue
        g, uu, vv, zz = vis
        n_frames += 1
        for key, cid in CLS.items():
            mk = stub_masks(g, uu, vv, gt[key], bleed_px=4)
            if mk is None:
                continue
            mk = pdet.erode_mask(mk, 3)
            sel = mk[vv, uu]
            if not sel.any():
                continue
            gi = g[sel]
            gi = gi[pdet.depth_gate(zz[sel], 0.6)]
            if gi.size < 25:
                continue
            votes.add(gi, cid, np.full(gi.size, 0.9, np.float32))
            tracker.add(gi, cid)

    check("every pose projected", n_frames == len(poses),
          f"{n_frames}/{len(poses)}")

    cls, conf, frames = votes.resolve(min_frames=2, min_ratio=0.5)
    instances = pdet.build_instances(pts, cls, conf, tracker.owner, tracker.cls,
                                     min_points=40, eps=0.10,
                                     min_frames_seen=1, hits=tracker.hits)
    names = {62: "tv", 56: "chair"}
    got = sorted(names[i["cls_id"]] for i in instances)
    check("both objects found, no spurious extras", got == ["chair", "tv"],
          f"instances={got}")

    # ---- structure -------------------------------------------------------- #
    models = pdet.fit_plane_models(pts, dist=0.03, min_points=4000,
                                   max_planes=10, fit_voxel=0.05)
    planes = pdet.assign_planes(pts, models, dist=0.03, min_points=4000)
    struct = pdet.Structure(planes, pts, tol_deg=15.0, min_wall_height=0.8)
    print(f"     structure: {struct.summary()}")
    check("one floor plane", len(struct.floors) == 1, str(len(struct.floors)))
    check("one ceiling plane", len(struct.ceilings) == 1,
          str(len(struct.ceilings)))
    check("four walls", len(struct.walls) == 4, str(len(struct.walls)))
    if struct.floors:
        check("floor is at z=0", abs(struct.floors[0]["z"]) < 0.02,
              f"z={struct.floors[0]['z']:.4f}")

    # the skirt trim needs structure, so it runs here, exactly as stage [6] does
    for ins in instances:
        new_idx, w = struct.trim_wall_skirt(ins["idx"], pts)
        if w is not None and new_idx.size >= 40:
            if VERBOSE:
                print(f"     trim {names[ins['cls_id']]}: "
                      f"{ins['idx'].size} -> {new_idx.size}")
            ins["idx"] = new_idx

    # purity: the guards exist to keep the wall out of the TV
    for key in ("tv", "chair"):
        ins = next((i for i in instances if names[i["cls_id"]] == key), None)
        if ins is None:
            check(f"{key} purity", False, "instance missing")
            continue
        inside = np.isin(ins["idx"], gt[key]).mean()
        recall = np.isin(gt[key], ins["idx"]).mean()
        check(f"{key}: >=95% of labelled points are really {key}",
              inside >= 0.95, f"precision {inside:.3f} recall {recall:.3f}")

    # ---- the rule the brief asks for -------------------------------------- #
    print("\n[7] synthesize")
    tv_ins = next(i for i in instances if names[i["cls_id"]] == "tv")
    ch_ins = next(i for i in instances if names[i["cls_id"]] == "chair")

    wall, frac = struct.wall_contact(pts[tv_ins["idx"]])
    check("TV is recognised as wall-mounted", wall is not None,
          f"wall fraction {frac:.2f}")
    ch_wall, ch_frac = struct.wall_contact(pts[ch_ins["idx"]])
    check("chair is NOT wall-mounted", ch_wall is None,
          f"wall fraction {ch_frac:.2f}")

    rules = {"tv": {"on_wall": "remove", "default": "replace"},
             "chair": "replace"}
    check("rule resolves TV-on-wall to remove",
          S01.resolve_rule(rules, "tv", True) == "remove")
    check("rule resolves free-standing TV to replace",
          S01.resolve_rule(rules, "tv", False) == "replace")

    # ---- removal + wall repair -------------------------------------------- #
    remove = np.zeros(N, bool)
    remove[tv_ins["idx"]] = True
    sidx = wall["idx"][~remove[wall["idx"]]]
    fp, fc = past.fill_plane_hole(wall, pts[tv_ins["idx"]], pts[sidx],
                                  cols[sidx], spacing=0.025, margin=0.03)
    check("wall hole was filled", len(fp) > 500, f"{len(fp)} points")
    if len(fp):
        off = np.abs(fp @ wall["n"] + wall["d"])
        check("patch lies in the wall plane", float(off.max()) < 1e-6,
              f"max |offset| {off.max():.2e} m")
        # it must cover the void and not spill over measured wall
        in_hole = ((np.abs(fp[:, 0] - TV_C[0]) < TV_WH[0] / 2 + 0.06)
                   & (np.abs(fp[:, 2] - TV_C[2]) < TV_WH[1] / 2 + 0.06))
        check("patch stays inside the void", float(in_hole.mean()) > 0.98,
              f"{100 * in_hole.mean():.1f}% inside")
        area = TV_WH[0] * TV_WH[1]
        cover = len(fp) * 0.025 ** 2 / area
        check("patch covers the void", 0.7 < cover < 1.5,
              f"{100 * cover:.0f}% of {area:.2f} m^2")
        check("patch takes the wall's colour, not the TV's",
              float(np.mean(np.abs(fc - 0.72))) < 0.05,
              f"mean |c - wall| {np.mean(np.abs(fc - 0.72)):.3f}")

    # ---- replacement: measured points KEPT, asset fitted to them ----------- #
    lib = past.AssetLibrary(os.path.join(HERE, "assets"))
    ip = pts[ch_ins["idx"]]
    ic = cols[ch_ins["idx"]]
    meta = lib.pick("chair", target_size=past.robust_extent(ip))
    plane, base_z = struct.support_under(ip, prefer=meta["support"])
    check("chair's support is the floor",
          plane is not None and plane["kind"] == "floor",
          "none" if plane is None else plane["kind"])
    check("support height is the floor height", abs(base_z) < 0.03,
          f"base_z={base_z:.4f}")

    ap, ac = lib.points(meta, 6000)
    fit = past.fit_asset(ap, meta["size"], ip, meta["yaw_symmetry"], base_z)
    check("a fit was produced", fit is not None)
    if fit is not None:
        check("fit explains the measurement", fit["coverage"] >= 0.35,
              f"coverage {fit['coverage']:.3f} rmse {fit['rmse']:.3f} m")
        check("asset base sits on the floor",
              abs(float(fit["pts"][:, 2].min())) < 0.03,
              f"min z {fit['pts'][:, 2].min():.4f}")
        check("asset is upright and roughly chair-sized",
              0.75 < float(fit["pts"][:, 2].max()) < 1.15,
              f"height {fit['pts'][:, 2].max():.3f} m")
        dy = np.degrees(fit["yaw"] - CHAIR_YAW) % 360.0
        dy = min(dy, 360.0 - dy)
        check("yaw recovered", dy < 20.0, f"error {dy:.1f} deg")
        cxy = fit["pts"][:, :2].mean(0)
        check("asset lands on the measured chair",
              float(np.linalg.norm(cxy - CHAIR_XY)) < 0.25,
              f"centre off by {np.linalg.norm(cxy - CHAIR_XY):.3f} m")

        newc = past.repaint(fit["pts"], ac, ip, ic, radius=0.08)
        seen_side = fit["pts"][:, 1] < CHAIR_XY[1]
        err = np.abs(newc[seen_side] - np.array([0.30, 0.55, 0.35])).mean()
        check("observed faces take the measured colour", err < 0.08,
              f"mean |c - measured| {err:.3f}")

        # the requirement from the brief: the measured points SURVIVE
        kept = (~remove)[ch_ins["idx"]].all()
        check("measured chair points are kept alongside the asset", bool(kept))

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED: {', '.join(FAILED)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Step 0 for the mobile_2 (sensor-constrained agent) reference pose: the board gates.

GOAL
  Before any estimator is built for mobile_2 (D455 + VSLAM, no LiDAR), three questions
  must be answered from the data, because a "no" to any of them changes the design:

  0a  Where are the surveyed boards in the GLIM map frame, and how good is each?
      Survey poses live in the normalised frame N (anchor board at origin); they are
      pulled back through inv(T_N_world). Per-board sigma is taken as
      max(section std, loop-closure disagreement) -- the raw std is optimistic for
      boards with a thin second section (anchor_b: 2 mm std but a *significant*
      12.9 mm / 2.6 deg loop closure).

  0b  How often does mobile_2 actually see a board?  (THE GATE)
      >= 4 well-separated sighting windows -> the fiducial/geometry/joint ablation is
      meaningful. Fewer -> the fiducial arm collapses to "VSLAM with one anchor";
      reframe, do not fuse. anchor and anchor_b are the SAME physical design
      (instances of it), so detections are disambiguated by cluster distance to the
      unambiguous DICT_5X5 board, never by marker id.

  0c  Are the boards still where the survey (96 min earlier) says, and is VSLAM's
      metric scale sane? Extrinsic-free joint test: inter-board baselines measured
      through VSLAM inside this bag vs the surveyed baselines.

  Plus: the board-frame axis convention check. PnP under ANY rigid axis convention
  reproduces the same image corners and the same board ORIGIN -- only the board's
  orientation frame changes -- so the convention cannot be picked from detections
  alone. It is picked by comparing board-to-board RELATIVE rotations against the
  survey (wrong conventions are off by ~90-180 deg, VSLAM drift by ~1 deg).

OUTPUT (in --out)
  census_raw.npz        raw detections (resume cache; delete to re-detect)
  step0_sightings.npz   assigned sightings + chosen axis convention + board poses,
                        consumed by the registration / pose-graph stage
  census.png            sighting timeline + range plot
  stdout                the 0a/0b/0c report and the PASS/FAIL gate

USAGE
  python3 m2_step0_boards.py \
      --bag    /path/to/mirc_dataset_coop2_20260828_merged \
      --survey map_stages_20260828_outputs/anchor_frame.json \
      --config pipeline_config.json \
      --out    m2_reference
  --config supplies the board designs (marker_len, min_corners, max_reproj) from the
  existing pipeline config; detection defaults fall back to it. Use --image-topic to
  switch to the color camera if the IR projector speckle ruins infra1 detection
  (watch mean reproj vs the survey's 0.22-0.39 px).
"""
import argparse, collections, json, math, sys, time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

# ----------------------------------------------------------------- SE(3) helpers
def Rt(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T

def inv(T):
    R = T[:3, :3]; o = np.eye(4); o[:3, :3] = R.T; o[:3, 3] = -R.T @ T[:3, 3]; return o

def q_to_R(q):        # q = xyzw
    return Rot.from_quat(q).as_matrix()

def interp_traj(ts_src, Ts_src, ts_q):
    """SLERP + linear interpolation onto query stamps; clamps at the ends."""
    from scipy.spatial.transform import Slerp
    ts_q = np.clip(ts_q, ts_src[0], ts_src[-1])
    i = np.clip(np.searchsorted(ts_src, ts_q) - 1, 0, len(ts_src) - 2)
    d = ts_src[i + 1] - ts_src[i]
    a = np.where(d > 0, (ts_q - ts_src[i]) / np.where(d > 0, d, 1), 0.0)
    sl = Slerp(ts_src, Rot.from_matrix(Ts_src[:, :3, :3]))
    out = np.tile(np.eye(4), (len(ts_q), 1, 1))
    out[:, :3, :3] = sl(ts_q).as_matrix()
    out[:, :3, 3] = Ts_src[i, :3, 3] * (1 - a)[:, None] + Ts_src[i + 1, :3, 3] * a[:, None]
    return out

# ----------------------------------------------------------------- bag access
def bag_reader(path):
    import rosbag2_py
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
           rosbag2_py.ConverterOptions("", ""))
    return r, {t.name: t.type for t in r.get_all_topics_and_types()}

def iter_topic(path, topic, stride=1, limit=None):
    """Yield (t_sec, msg); t is the header stamp when present, else bag time."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    try:
        from rosidl_runtime_py.utilities import get_message
    except ImportError:                      # very old distros
        from rosidl_runtime_py.utility import get_message
    r, types = bag_reader(path)
    if topic not in types:
        raise KeyError(f"{topic} not in bag; have {sorted(types)[:8]}...")
    cls = get_message(types[topic])
    f = rosbag2_py.StorageFilter(); f.topics = [topic]; r.set_filter(f)
    i = n = 0
    while r.has_next():
        _, data, t_bag = r.read_next()
        if i % stride: i += 1; continue
        i += 1
        m = deserialize_message(data, cls)
        h = getattr(m, "header", None)
        t = (h.stamp.sec + h.stamp.nanosec * 1e-9) if h is not None else t_bag * 1e-9
        yield t, m
        n += 1
        if limit and n >= limit: break

def camera_K(path, topic_ci):
    for _, ci in iter_topic(path, topic_ci, limit=1):
        return np.array(ci.k).reshape(3, 3), ci.width, ci.height
    raise RuntimeError(f"no camera_info on {topic_ci}")

def img_to_np(m):
    a = np.frombuffer(m.data, dtype=np.uint8)
    enc = m.encoding
    if enc in ("mono8", "8UC1"):   return a.reshape(m.height, m.step)[:, :m.width]
    if enc == "16UC1":             return a.view(np.uint16).reshape(m.height, m.step // 2)[:, :m.width]
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}[enc]
    im = a.reshape(m.height, m.step // ch, ch)[:, :m.width]
    return im[..., ::-1][..., :3] if enc.startswith("rgb") else im[..., :3]

def odom_to_T(m):
    p = m.pose.pose.position; o = m.pose.pose.orientation
    return Rt(q_to_R([o.x, o.y, o.z, o.w]), np.array([p.x, p.y, p.z]))

# ----------------------------------------------------------------- ChArUco
# Candidate board-frame axis conventions (map from the surveyed frame to OpenCV's
# X-right / Y-down / Z-out board frame). Which one the survey used is checked
# against the data by axis_check(); the config's name ("ros") does not define the
# rotation unambiguously.
AXIS_CANDIDATES = {
    "cv":       np.eye(3),
    "xy_flip":  np.diag([1.0, -1.0, -1.0]),
    "ros":      np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]]),
    "ros_180":  np.array([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]]),
}
BOARD_AXES = "cv"          # overwritten by axis_check()
_DET = {}

def make_board(spec):
    import cv2
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
    sx, sy = spec["squares"]
    try:                                  # OpenCV >= 4.7
        b = cv2.aruco.CharucoBoard((sx, sy), spec["square_len"], spec["marker_len"], d)
        b.setLegacyPattern(True)
    except AttributeError:                # OpenCV 4.6
        b = cv2.aruco.CharucoBoard_create(sx, sy, spec["square_len"], spec["marker_len"], d)
    return b, d

def board_object_points(spec, axes=None):
    """Interior ChArUco corners, (sx-1)*(sy-1) x 3, in the SURVEYED board frame."""
    sx, sy = spec["squares"]; sq = spec["square_len"]
    j, i = np.meshgrid(np.arange(1, sy), np.arange(1, sx), indexing="ij")
    P = np.column_stack([i.ravel() * sq, j.ravel() * sq, np.zeros(i.size)])
    P -= np.array([sx * sq / 2, sy * sq / 2, 0.0])           # board_origin = "center"
    Rc = AXIS_CANDIDATES[axes or BOARD_AXES]
    return P @ np.linalg.inv(Rc).T                           # p_board = Rc^-1 p_cv

def detect_charuco(gray, spec):
    """-> (corner_ids (n,), uv (n,2)) or (None, None)."""
    import cv2
    key = (spec["dictionary"], spec["squares"], spec["square_len"])
    if key not in _DET: _DET[key] = make_board(spec)
    board, dic = _DET[key]
    if hasattr(cv2.aruco, "CharucoDetector"):
        cc, ci, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
        if cc is None or len(cc) < 4: return None, None
        return ci.ravel().astype(int), cc.reshape(-1, 2)
    mc, mi, _ = cv2.aruco.detectMarkers(gray, dic)
    if mi is None or len(mi) < 2: return None, None
    n, cc, ci = cv2.aruco.interpolateCornersCharuco(mc, mi, gray, board)
    if n is None or n < 4: return None, None
    return ci.ravel().astype(int), cc.reshape(-1, 2)

def pnp_board(ids, uv, spec, K, axes=None):
    """-> (T_cam_board, mean_reproj_px, ambiguity_ratio). Planar targets have a
    two-fold pose ambiguity at grazing views; the IPPE ratio (2nd/1st solution
    reprojection error) flags it -- gated by the config's min_ambiguity_ratio."""
    import cv2
    P = board_object_points(spec, axes)[ids].astype(np.float64)
    if len(P) < 4: return None, np.inf, 0.0
    uv = uv.astype(np.float64)
    ratio = np.inf
    try:
        ok, rvs, tvs, errs = cv2.solvePnPGeneric(P, uv.reshape(-1, 1, 2), K, None,
                                                 flags=cv2.SOLVEPNP_IPPE)
        if not ok or len(rvs) == 0: return None, np.inf, 0.0
        errs = np.asarray(errs).ravel()
        if len(errs) > 1 and errs[0] > 0: ratio = float(errs[1] / errs[0])
        rv, tv = rvs[0], tvs[0]
    except cv2.error:
        ok, rv, tv = cv2.solvePnP(P, uv, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: return None, np.inf, 0.0
    rv, tv = cv2.solvePnPRefineLM(P, uv, K, None, rv, tv)
    proj, _ = cv2.projectPoints(P, rv, tv, K, None)
    err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - uv, axis=1)))
    return Rt(cv2.Rodrigues(rv)[0], tv.ravel()), err, ratio

# ================================================================= stage 0a
def load_survey(survey_path, cfg_boards):
    S = json.loads(Path(survey_path).read_text())
    T_world_N = inv(np.array(S["T_N_world"]))
    # detection params come from the pipeline config, matched by design
    def design_of(b):
        for d in (cfg_boards or {}).values():
            if (d.get("dictionary") == b["dictionary"]
                    and (d.get("squares_x"), d.get("squares_y")) == tuple(b["squares"])
                    and abs(d.get("square_len", 0) - b["square_len"]) < 1e-9):
                return d
        return {}
    boards = {}
    for name, b in S["boards"].items():
        d = design_of(b)
        lc = b.get("loop_closure", {}) or {}
        boards[name] = dict(
            name=name,
            T_map_board=T_world_N @ Rt(q_to_R(b["qxyzw"]), np.array(b["xyz"])),
            squares=tuple(b["squares"]), square_len=b["square_len"],
            marker_len=d.get("marker_len", 0.75 * b["square_len"]),
            dictionary=b["dictionary"], id_offset=b.get("id_offset", 0),
            min_corners=d.get("min_corners", 8),
            max_reproj=d.get("max_reproj", 1.5),
            min_ambiguity=d.get("min_ambiguity_ratio", 1.5),
            sigma_t=max(b.get("std_mm", 0.0), lc.get("mm", 0.0)) * 1e-3,
            sigma_r=math.radians(max(lc.get("deg", 0.0), 0.2)),
            n_views=b.get("n_views", 0),
            drift_warning=bool(b.get("drift_warning", False)),
            lc_significant=bool(lc.get("significant", False)),
        )
        if not d:
            print(f"  (no design in --config matches '{name}'; "
                  f"marker_len defaulted to {boards[name]['marker_len']:.4f})")
    return boards

def report_survey(boards):
    print(f"\n=== 0a: surveyed boards in the map frame ===")
    print(f"{'board':11s} {'x':>8s} {'y':>8s} {'z':>8s}  {'sig_t':>7s} {'sig_R':>7s} {'views':>5s}  flags")
    for n, b in boards.items():
        p = b["T_map_board"][:3, 3]
        fl = ",".join(f for f, on in [("DRIFT", b["drift_warning"]),
                                      ("LC-SIG", b["lc_significant"])] if on) or "-"
        print(f"{n:11s} {p[0]:8.3f} {p[1]:8.3f} {p[2]:8.3f}  "
              f"{b['sigma_t']*1000:6.1f}mm {math.degrees(b['sigma_r']):6.2f}d {b['n_views']:5d}  {fl}")
    grp = collections.defaultdict(list)
    for n, b in boards.items():
        grp[(b["dictionary"], b["id_offset"], b["squares"])].append(n)
    ambig = {k: v for k, v in grp.items() if len(v) > 1}
    for k, v in ambig.items():
        print(f"!! ID COLLISION {v} share {k[0]} offset {k[1]} {k[2]} "
              "-> disambiguated by position, not id")
    names = list(boards)
    print("surveyed baselines (rangefinder targets - long, one in a corridor):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = np.linalg.norm(boards[names[i]]["T_map_board"][:3, 3] -
                               boards[names[j]]["T_map_board"][:3, 3])
            print(f"  {names[i]:11s} -> {names[j]:11s} {d:7.3f} m")
    return ambig

# ================================================================= stage 0b
def run_census(args, boards, K):
    """Detect every distinct board design over the image topic. Cached."""
    import cv2
    cache = Path(args.out) / "census_raw.npz"
    if cache.exists() and not args.force:
        rows = list(np.load(cache, allow_pickle=True)["rows"])
        print(f"\n=== 0b: census (cached) === {len(rows)} raw sightings from {cache}")
        return rows
    uniq, seen = [], set()
    for b in boards.values():
        k = (b["dictionary"], b["squares"], b["square_len"])
        if k not in seen: seen.add(k); uniq.append((k, b))
    print(f"\n=== 0b: census over {args.image_topic} === "
          f"designs: {[k[0] for k, _ in uniq]}")
    rows, t0 = [], time.time()
    for fi, (t, m) in enumerate(iter_topic(args.bag, args.image_topic,
                                           stride=args.stride, limit=args.limit or None)):
        im = img_to_np(m)
        gray = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        for k, spec in uniq:
            ids, uv = detect_charuco(gray, spec)
            if ids is None or len(ids) < spec["min_corners"]: continue
            T_cb, err, ratio = pnp_board(ids, uv, spec, K)
            if T_cb is None or err > spec["max_reproj"]: continue
            if ratio < spec["min_ambiguity"]: continue       # planar-pose ambiguity
            rows.append(dict(t=t, frame=fi, key=k[0] + str(k[1]), n=len(ids),
                             rng=float(np.linalg.norm(T_cb[:3, 3])), err=err,
                             ratio=ratio, ids=ids, uv=uv))
        if fi and fi % 200 == 0:
            print(f"  {fi:5d} frames  {len(rows):4d} sightings  {time.time()-t0:5.1f}s",
                  flush=True)
    np.savez_compressed(cache, rows=np.array(rows, dtype=object), allow_pickle=True)
    print(f"{len(rows)} raw sightings in {time.time()-t0:.1f}s -> {cache}")
    if rows:
        mr = np.mean([r["err"] for r in rows])
        print(f"mean reproj {mr:.3f} px (survey was 0.22-0.39 px; much worse => "
              f"IR projector speckle, retry with --image-topic <color topic>)")
    return rows

def load_vslam(args):
    vo_t, vo_T = [], []
    for t, m in iter_topic(args.bag, args.vo_topic, limit=args.limit or None):
        vo_t.append(t); vo_T.append(odom_to_T(m))
    vo_t = np.array(vo_t); vo_T = np.array(vo_T)
    print(f"VSLAM odometry: {len(vo_t)} poses, {vo_t[-1]-vo_t[0]:.1f} s, "
          f"path {np.sum(np.linalg.norm(np.diff(vo_T[:,:3,3],axis=0),axis=1)):.1f} m")
    return vo_t, vo_T

def assign_instances(raw, boards, ambig, vo_t, vo_T, K):
    """Cluster sightings by board origin in the VSLAM frame (origins are invariant to
    the axis convention), then name each cluster by its distance to a cluster of an
    unambiguous design. Instances of one design can never be told apart by id."""
    sight = []
    for r in raw:
        spec = next(b for b in boards.values()
                    if b["dictionary"] + str(b["squares"]) == r["key"])
        T_cb, err, _ = pnp_board(r["ids"], r["uv"], spec, K)
        if T_cb is None or err > spec["max_reproj"]: continue
        T_vo = interp_traj(vo_t, vo_T, np.array([r["t"]]))[0]
        sight.append(dict(t=r["t"], n=r["n"], rng=r["rng"], err=err, key=r["key"],
                          ids=r["ids"], uv=r["uv"], T_cb=T_cb,
                          p_vo=(T_vo @ T_cb)[:3, 3]))
    def cluster(pts, tol=0.6):
        lab = -np.ones(len(pts), int); c = 0
        for i in range(len(pts)):
            if lab[i] >= 0: continue
            lab[np.linalg.norm(pts - pts[i], axis=1) < tol] = c; c += 1
        return lab, c
    clus = {}
    for key in sorted({s["key"] for s in sight}):
        idx = [i for i, s in enumerate(sight) if s["key"] == key]
        lab, nc = cluster(np.array([sight[i]["p_vo"] for i in idx]))
        for c in range(nc):
            sel = [idx[k] for k in np.where(lab == c)[0]]
            clus[f"{key}#{c}"] = dict(key=key, idx=sel,
                p_vo=np.median([sight[i]["p_vo"] for i in sel], axis=0))
    print(f"\n{len(sight)} sightings -> {len(clus)} spatial clusters")
    unamb = {n: b for n, b in boards.items() if not any(n in v for v in ambig.values())}
    ref_cid = next((cid for cid, c in clus.items()
                    if any(c["key"] == b["dictionary"] + str(b["squares"])
                           for b in unamb.values())), None)
    assign = {}
    if ref_cid is None:
        print("!! no unambiguous board seen - name the clusters by hand "
              "(edit ASSIGN in the saved npz).")
    else:
        ref_name = next(n for n, b in unamb.items()
                        if b["dictionary"] + str(b["squares"]) == clus[ref_cid]["key"])
        assign[ref_cid] = ref_name
        p_ref = clus[ref_cid]["p_vo"]
        print(f"reference cluster {ref_cid} = '{ref_name}'")
        for cid, c in clus.items():
            if cid == ref_cid: continue
            d_meas = float(np.linalg.norm(c["p_vo"] - p_ref))
            cands = {n: float(np.linalg.norm(b["T_map_board"][:3, 3] -
                                             boards[ref_name]["T_map_board"][:3, 3]))
                     for n, b in boards.items()
                     if n != ref_name and b["dictionary"] + str(b["squares"]) == c["key"]}
            if not cands: continue
            best = min(cands, key=lambda n: abs(cands[n] - d_meas))
            assign[cid] = best
            print(f"  {cid:26s} d_meas={d_meas:6.2f} m -> '{best}'  "
                  f"(surveyed: {' '.join(f'{n}={v:.2f}' for n, v in cands.items())})")
    for cid, name in assign.items():
        for i in clus[cid]["idx"]: sight[i]["board"] = name
    sight = [s for s in sight if "board" in s]
    print(f"{len(sight)} assigned sightings across "
          f"{len({s['board'] for s in sight})} boards")
    return sight, assign

def axis_check(sight, boards, vo_t, vo_T, K):
    """Pick the surveyed board-frame convention from board-to-board relative
    rotations (see module docstring). Needs >= 2 distinct boards sighted."""
    global BOARD_AXES
    def score(axes):
        Rm = {}
        for s in sight:
            T_cb, err, _ = pnp_board(s["ids"], s["uv"], boards[s["board"]], K, axes=axes)
            if T_cb is None: continue
            T_vo = interp_traj(vo_t, vo_T, np.array([s["t"]]))[0]
            Rm.setdefault(s["board"], []).append((T_vo @ T_cb)[:3, :3])
        names = sorted(Rm)
        if len(names) < 2: return None
        worst = 0.0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                R_meas = Rm[a][len(Rm[a]) // 2].T @ Rm[b][len(Rm[b]) // 2]
                R_srv = boards[a]["T_map_board"][:3, :3].T @ boards[b]["T_map_board"][:3, :3]
                worst = max(worst, math.degrees(np.linalg.norm(
                    Rot.from_matrix(R_srv.T @ R_meas).as_rotvec())))
        return worst
    print("\n=== board-axis convention check ===")
    scores = {a: score(a) for a in AXIS_CANDIDATES}
    if all(v is None for v in scores.values()):
        print("!! <2 distinct boards sighted - convention unchecked; keeping "
              f"'{BOARD_AXES}'. Set it from the survey tool's board-frame code.")
        return BOARD_AXES
    for a, v in sorted(scores.items(), key=lambda kv: (kv[1] is None, kv[1])):
        print(f"  {a:9s} relative-rotation disagreement "
              + ("   n/a" if v is None else f"{v:8.2f} deg"))
    BOARD_AXES = min((a for a in scores if scores[a] is not None), key=lambda a: scores[a])
    print(f"-> BOARD_AXES = '{BOARD_AXES}'  ({scores[BOARD_AXES]:.2f} deg)")
    if scores[BOARD_AXES] > 10:
        print("!! best candidate still >10 deg off - none of the enumerated conventions "
              "matches the survey; add the survey tool's rotation to AXIS_CANDIDATES.")
    for s in sight:                       # re-solve stored poses under the winner
        T_cb, err, _ = pnp_board(s["ids"], s["uv"], boards[s["board"]], K, axes=BOARD_AXES)
        if T_cb is not None: s["T_cb"], s["err"] = T_cb, err
    return BOARD_AXES

def gate_report(sight, vo_t, out_dir):
    t_rel = np.array([s["t"] for s in sight]) - vo_t[0]
    bnames = sorted({s["board"] for s in sight})
    dur = vo_t[-1] - vo_t[0]
    windows = []
    for b in bnames:
        tb = np.sort(t_rel[[i for i, s in enumerate(sight) if s["board"] == b]])
        if not len(tb): continue
        br = np.where(np.diff(tb) > 2.0)[0]
        for a, z in zip(np.r_[0, br + 1], np.r_[br, len(tb) - 1]):
            windows.append((b, tb[a], tb[z], z - a + 1))
    print(f"\n=== 0b: THE GATE ===\ntrajectory {dur:.1f} s")
    print(f"{'board':11s} {'t_start':>8s} {'t_end':>8s} {'n':>5s}")
    for b, a, z, n in sorted(windows, key=lambda w: w[1]):
        print(f"{b:11s} {a:8.1f} {z:8.1f} {n:5d}")
    gaps = np.diff(np.r_[0, sorted(w[1] for w in windows), dur])
    print(f"{len(windows)} windows; longest board-free stretch {gaps.max():.1f} s "
          f"({gaps.max()*0.8:.1f} m at 0.8 m/s)")
    ok = len(windows) >= 4
    print("GATE:", "PASS - three-arm ablation is meaningful" if ok else
          "FAIL - reframe as anchored vs unanchored VSLAM, not a fusion ablation")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), height_ratios=[1, 2])
    for i, b in enumerate(bnames):
        m = [j for j, s in enumerate(sight) if s["board"] == b]
        ax[0].scatter(t_rel[m], np.full(len(m), i), s=8, label=b)
        ax[1].scatter(t_rel[m], [sight[j]["rng"] for j in m], s=8, label=b)
    ax[0].set_yticks(range(len(bnames))); ax[0].set_yticklabels(bnames)
    ax[0].set_xlim(0, dur); ax[0].set_title("board sightings over the trajectory")
    ax[0].grid(alpha=.3)
    ax[1].set_xlim(0, dur); ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("range [m]")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    plt.tight_layout()
    png = Path(out_dir) / "census.png"; plt.savefig(png, dpi=130); plt.close()
    print(f"plot: {png}")
    return windows, ok

def stability_check(sight, boards):
    """0c: measured inter-board baselines vs the survey. Joint test of board
    stability and VSLAM scale; agreement validates both, disagreement flags a
    problem without saying which."""
    bnames = sorted({s["board"] for s in sight})
    print(f"\n=== 0c: board stability / VSLAM scale ===")
    print(f"{'pair':26s} {'surveyed':>9s} {'measured':>9s} {'diff':>8s} {'n':>5s}")
    rows = []
    for i in range(len(bnames)):
        for j in range(i + 1, len(bnames)):
            a, b = bnames[i], bnames[j]
            d_srv = float(np.linalg.norm(boards[a]["T_map_board"][:3, 3] -
                                         boards[b]["T_map_board"][:3, 3]))
            pa = np.array([s["p_vo"] for s in sight if s["board"] == a])
            pb = np.array([s["p_vo"] for s in sight if s["board"] == b])
            d_msr = float(np.linalg.norm(np.median(pa, 0) - np.median(pb, 0)))
            rows.append((d_srv, d_msr))
            print(f"{a+' <-> '+b:26s} {d_srv:8.3f}m {d_msr:8.3f}m "
                  f"{1000*(d_msr-d_srv):+7.0f}mm {len(pa)+len(pb):5d}")
    if len(rows) >= 2:
        s = np.polyfit([r[0] for r in rows], [r[1] for r in rows], 1)[0]
        print(f"implied VSLAM scale {s:.5f}  ({(s-1)*1e6:+.0f} ppm)")
        print("  consistent scale != 1 across pairs => VSLAM stereo scale error "
              "(estimable downstream); one pair off while others match => that board moved.")
    elif rows:
        print("only one pair - cannot separate 'board moved' from 'VSLAM scale'; "
              "treat a large diff as a warning, not a diagnosis.")

def save_outputs(sight, assign, boards, out_dir):
    out = Path(out_dir) / "step0_sightings.npz"
    np.savez_compressed(
        out,
        sightings=np.array(sight, dtype=object),
        board_axes=BOARD_AXES,
        assign=np.array(list(assign.items()), dtype=object),
        boards=np.array([(n, b["T_map_board"], b["sigma_t"], b["sigma_r"])
                         for n, b in boards.items()], dtype=object),
        allow_pickle=True)
    print(f"\nwrote {out}: {len(sight)} assigned sightings, BOARD_AXES='{BOARD_AXES}'")
    print("next stage (depth->clouds, scan-to-map, A/B/C pose graph) consumes this file.")

# ================================================================= main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bag", required=True, help="coop bag directory (rosbag2/mcap)")
    ap.add_argument("--survey", required=True, help="anchor_frame.json from stage 03")
    ap.add_argument("--config", default=None,
                    help="pipeline_config.json - supplies board marker_len + gates")
    ap.add_argument("--out", default="m2_reference")
    ap.add_argument("--image-topic", default="/mobile_2/infra1/image_rect_raw")
    ap.add_argument("--camera-info-topic", default="/mobile_2/infra1/camera_info")
    ap.add_argument("--vo-topic", default="/mobile_2/visual_slam/tracking/odometry")
    ap.add_argument("--stride", type=int, default=2, help="detect every Nth frame")
    ap.add_argument("--limit", type=int, default=0, help="frame cap for smoke tests")
    ap.add_argument("--force", action="store_true", help="ignore the detection cache")
    args = ap.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    cfg_boards = (json.loads(Path(args.config).read_text()).get("boards", {})
                  if args.config else {})
    boards = load_survey(args.survey, cfg_boards)
    ambig = report_survey(boards)

    K, w, h = camera_K(args.bag, args.camera_info_topic)
    print(f"\ncamera {args.image_topic}: {w}x{h}, fx={K[0,0]:.1f}")
    raw = run_census(args, boards, K)
    if not raw:
        print("!! NO SIGHTINGS - check the topic, lower min_corners in --config, or "
              "confirm mobile_2 ever faced a board. Gate: FAIL."); return 1
    vo_t, vo_T = load_vslam(args)
    sight, assign = assign_instances(raw, boards, ambig, vo_t, vo_T, K)
    if not sight:
        print("!! sightings could not be assigned to boards. Gate: FAIL."); return 1
    axis_check(sight, boards, vo_t, vo_T, K)
    gate_report(sight, vo_t, args.out)
    stability_check(sight, boards)
    save_outputs(sight, assign, boards, args.out)
    return 0

if __name__ == "__main__":
    sys.exit(main())

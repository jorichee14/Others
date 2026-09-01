#!/usr/bin/env python3
"""Step 0 for the mobile_2 (sensor-constrained agent) reference pose: the board gates.

GOAL
  Before any estimator is built for mobile_2 (D455 + VSLAM, no LiDAR), three questions
  must be answered from the data, because a "no" to any of them changes the design:

  0a  Where are the surveyed boards in the map frame, and how good is each?
      Poses come from stage 03's anchor_frame.json, pulled back through inv(T_N_world).
      Per-board sigma = max(section std, loop-closure disagreement) -- the raw std is
      optimistic for boards with a thin second section (anchor_b: 2 mm std but a
      *significant* 12.9 mm / 2.6 deg loop closure).

  0b  How often does mobile_2 actually see a board?  (THE GATE)
      >= 4 well-separated sighting windows -> the fiducial/geometry/joint ablation is
      meaningful. Fewer -> the fiducial arm collapses to "VSLAM with one anchor";
      reframe, do not fuse. anchor / anchor_b are instances of ONE design, so
      sightings are named by cluster distance to a single-instance design
      (rs_anchor), never by marker id.

  0c  Are the boards still where the survey (96 min earlier) says, and is VSLAM's
      metric scale sane? Extrinsic-free joint test: inter-board baselines measured
      through VSLAM inside this bag vs the surveyed baselines.

DETECTION
  Primary: the pipeline's own detector (pipeline_boards.bank_from_config + bank.detect),
  imported from the directory of --config -- the SAME code and frame_fix(board_axes,
  board_origin) that produced the survey, so detected poses are in the survey's board
  frame by construction. Fallback (pipeline modules not importable): a built-in
  ChArUco detector that tries both legacy and modern square layouts and, because its
  frame convention is then a guess, checks it against the survey via board-to-board
  relative rotations (PnP origins are convention-invariant; only orientations
  discriminate).

  Zero sightings triggers a probe: n frames sampled across the bag, aruco MARKER
  counts vs interpolated CORNER counts per dictionary (raw and CLAHE-equalised),
  frames saved with detections drawn. Markers 0 everywhere -> board too small/far,
  image too dark (--equalize), wrong topic. Markers > 0 but corners 0 -> board
  layout mismatch. Corners > 0 but 0 accepted -> the gates are rejecting everything.

OUTPUT (--out)
  step0_sightings.npz   assigned sightings + board poses, for the registration stage
  census_raw.npz        detection cache (delete or --force to re-detect)
  census.png            sighting timeline + range plot
  probe_*.png           only when the census comes back empty

USAGE
  python3 m2_step0_boards.py \
      --bag    ../../raw/20260828/mirc_dataset_coop2_20260828_merged \
      --survey map_stages_20260828_outputs/anchor_frame.json \
      --config pipeline_config.json --out m2_reference
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
def iter_topic(path, topic, stride=1, limit=None):
    """Yield (t_sec, msg); t is the header stamp when present, else bag time."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    try:
        from rosidl_runtime_py.utilities import get_message
    except ImportError:
        from rosidl_runtime_py.utility import get_message
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
           rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
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

def to_gray(m):
    import cv2
    im = img_to_np(m)
    return im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

def odom_to_T(m):
    p = m.pose.pose.position; o = m.pose.pose.orientation
    return Rt(q_to_R([o.x, o.y, o.z, o.w]), np.array([p.x, p.y, p.z]))

def _clahe(gray):
    import cv2
    return cv2.createCLAHE(3.0, (8, 8)).apply(gray)

# ================================================================= stage 0a
def load_survey(survey_path):
    S = json.loads(Path(survey_path).read_text())
    T_world_N = inv(np.array(S["T_N_world"]))
    boards = {}
    for name, b in S["boards"].items():
        lc = b.get("loop_closure", {}) or {}
        boards[name] = dict(
            name=name, design=b.get("design", name),
            T_map_board=T_world_N @ Rt(q_to_R(b["qxyzw"]), np.array(b["xyz"])),
            squares=tuple(b["squares"]), square_len=b["square_len"],
            dictionary=b["dictionary"], id_offset=b.get("id_offset", 0),
            sigma_t=max(b.get("std_mm", 0.0), lc.get("mm", 0.0)) * 1e-3,
            sigma_r=math.radians(max(lc.get("deg", 0.0), 0.2)),
            n_views=b.get("n_views", 0),
            drift_warning=bool(b.get("drift_warning", False)),
            lc_significant=bool(lc.get("significant", False)),
        )
    meta = dict(board_axes=S.get("board_axes", "opencv"),
                board_origin=S.get("board_origin", "corner"))
    return boards, meta

def report_survey(boards):
    print(f"\n=== 0a: surveyed boards in the map frame ===")
    print(f"{'board':11s} {'design':10s} {'x':>8s} {'y':>8s} {'z':>8s}  "
          f"{'sig_t':>7s} {'sig_R':>7s} {'views':>5s}  flags")
    for n, b in boards.items():
        p = b["T_map_board"][:3, 3]
        fl = ",".join(f for f, on in [("DRIFT", b["drift_warning"]),
                                      ("LC-SIG", b["lc_significant"])] if on) or "-"
        print(f"{n:11s} {b['design']:10s} {p[0]:8.3f} {p[1]:8.3f} {p[2]:8.3f}  "
              f"{b['sigma_t']*1000:6.1f}mm {math.degrees(b['sigma_r']):6.2f}d "
              f"{b['n_views']:5d}  {fl}")
    multi = {d: [n for n, b in boards.items() if b["design"] == d]
             for d in {b["design"] for b in boards.values()}}
    for d, v in multi.items():
        if len(v) > 1:
            print(f"!! design '{d}' has instances {v} -> named by position, not id")
    names = list(boards)
    print("surveyed baselines (rangefinder targets - long, one in a corridor):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = np.linalg.norm(boards[names[i]]["T_map_board"][:3, 3] -
                               boards[names[j]]["T_map_board"][:3, 3])
            print(f"  {names[i]:11s} -> {names[j]:11s} {d:7.3f} m")
    return multi

# ================================================================= detection: pipeline
def census_pipeline(args, boards, meta):
    """Primary path: the survey's own detector + frame_fix, so detected poses are in
    the survey board frame by construction. Returns rows or None if unavailable."""
    cfg_dir = str(Path(args.config).resolve().parent)
    if cfg_dir not in sys.path: sys.path.insert(0, cfg_dir)
    try:
        from pipeline_boards import bank_from_config
        import pipeline_common  # noqa: F401  (bank may need it)
    except ImportError as e:
        print(f"(pipeline detector not importable from {cfg_dir}: {e})")
        return None
    import cv2
    cfg = json.loads(Path(args.config).read_text())
    designs = sorted({b["design"] for b in boards.values() if b["design"] in cfg.get("boards", {})})
    if not designs:
        print("(no survey design matches the --config boards registry)")
        return None
    try:
        bank = bank_from_config(cfg, designs)
        FIX = {b.name: b.frame_fix(meta["board_axes"], meta["board_origin"])
               for b in bank.designs}
    except Exception as e:
        print(f"(pipeline detector setup failed: {type(e).__name__}: {e})")
        return None
    K, w, h = camera_K(args.bag, args.camera_info_topic)
    D = np.zeros(5)                       # image_rect_raw is rectified
    print(f"\n=== 0b: census over {args.image_topic} ({w}x{h}, fx={K[0,0]:.1f}) === "
          f"pipeline detector, designs {designs}, frame_fix"
          f"({meta['board_axes']},{meta['board_origin']})"
          + ("  [CLAHE]" if args.equalize else ""))
    rows, t0, nfr = [], time.time(), 0
    try:
        for fi, (t, m) in enumerate(iter_topic(args.bag, args.image_topic,
                                               stride=args.stride,
                                               limit=args.limit or None)):
            nfr += 1
            gray = to_gray(m)
            if args.equalize: gray = _clahe(gray)
            for d in bank.detect(gray, K, D, stamp=t):
                T_cb = d.T @ FIX[d.design]
                rows.append(dict(t=t, frame=fi, design=d.design, T_cb=T_cb,
                                 n=int(d.n), err=float(d.reproj),
                                 ratio=float(getattr(d, "ratio", np.inf)),
                                 rng=float(np.linalg.norm(T_cb[:3, 3])),
                                 ids=None, uv=None, src="pipeline"))
            if fi and fi % 200 == 0:
                print(f"  {fi:5d} frames  {len(rows):4d} sightings  "
                      f"{time.time()-t0:5.1f}s", flush=True)
    except Exception as e:
        print(f"(pipeline detector failed mid-run: {type(e).__name__}: {e})")
        return None
    print(f"{len(rows)} sightings over {nfr} frames in {time.time()-t0:.1f}s")
    for b in bank.designs:
        try:    print(f"  {b.name:12s} {b.reject_str()}")
        except Exception: pass
    return rows

# ================================================================= detection: builtin
_DET, _PARAMS = {}, None
BOARD_AXES = "cv"          # builtin fallback only; checked by axis_check()
AXIS_CANDIDATES = {
    "cv":       np.eye(3),
    "xy_flip":  np.diag([1.0, -1.0, -1.0]),
    "ros":      np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]]),
    "ros_180":  np.array([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]]),
}

def detector_params():
    import cv2
    global _PARAMS
    if _PARAMS is None:
        try:               p = cv2.aruco.DetectorParameters()
        except AttributeError: p = cv2.aruco.DetectorParameters_create()
        try:               p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        except AttributeError: pass
        _PARAMS = p
    return _PARAMS

def make_board(spec, legacy=True):
    import cv2
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
    sx, sy = spec["squares"]
    if hasattr(cv2.aruco, "CharucoBoard_create"):    # old API (<=4.6): legacy layout inherent
        b = cv2.aruco.CharucoBoard_create(sx, sy, spec["square_len"], spec["marker_len"], d)
    else:
        b = cv2.aruco.CharucoBoard((sx, sy), spec["square_len"], spec["marker_len"], d)
        b.setLegacyPattern(legacy)
    return b, d

def board_object_points(spec, axes=None):
    """Interior ChArUco corners in the surveyed board frame (builtin path)."""
    sx, sy = spec["squares"]; sq = spec["square_len"]
    j, i = np.meshgrid(np.arange(1, sy), np.arange(1, sx), indexing="ij")
    P = np.column_stack([i.ravel() * sq, j.ravel() * sq, np.zeros(i.size)])
    P -= np.array([sx * sq / 2, sy * sq / 2, 0.0])
    Rc = AXIS_CANDIDATES[axes or BOARD_AXES]
    return P @ np.linalg.inv(Rc).T

def detect_charuco(gray, spec):
    """-> (ids, uv, n_markers, legacy). n_markers reports markers of this DICTIONARY
    even when interpolation failed - separates 'not visible' from 'layout mismatch'.
    On OpenCV >= 4.7 both legacy and modern layouts are tried."""
    import cv2
    key = (spec["dictionary"], spec["squares"], spec["square_len"])
    old_api = hasattr(cv2.aruco, "CharucoBoard_create")
    best_ids, best_uv, best_leg, nm_max = None, None, None, 0
    for legacy in ([True] if old_api else [True, False]):
        k = key + (legacy,)
        if k not in _DET: _DET[k] = make_board(spec, legacy)
        board, dic = _DET[k]
        cc = ci = None
        if old_api:
            mc, mi, _ = cv2.aruco.detectMarkers(gray, dic, parameters=detector_params())
            nm = 0 if mi is None else len(mi)
            if nm >= 2:
                n, cc, ci = cv2.aruco.interpolateCornersCharuco(mc, mi, gray, board)
                if not n: cc = ci = None
        else:
            cc, ci, mc, mi = cv2.aruco.CharucoDetector(board).detectBoard(gray)
            nm = 0 if mi is None else len(mi)
            if cc is None or len(cc) < 4: cc = ci = None
        nm_max = max(nm_max, nm)
        if ci is not None and (best_ids is None or len(ci) > len(best_ids)):
            best_ids, best_uv, best_leg = ci.ravel().astype(int), cc.reshape(-1, 2), legacy
    return best_ids, best_uv, nm_max, best_leg

def pnp_board(ids, uv, spec, K, axes=None):
    """-> (T_cam_board, mean_reproj_px, IPPE ambiguity ratio)."""
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

def census_builtin(args, boards, cfg_boards):
    """Fallback detector, fully instrumented."""
    import cv2
    print(f"\n=== 0b: census over {args.image_topic} === builtin detector, opencv "
          f"{cv2.__version__}" + ("  [CLAHE]" if args.equalize else ""))
    K, w, h = camera_K(args.bag, args.camera_info_topic)
    uniq, seen = [], set()
    for b in boards.values():
        d = cfg_boards.get(b["design"], {})
        spec = dict(b, marker_len=d.get("marker_len", 0.75 * b["square_len"]),
                    min_corners=d.get("min_corners", 8),
                    max_reproj=d.get("max_reproj", 1.5),
                    min_ambiguity=d.get("min_ambiguity_ratio", 1.5))
        if b["design"] not in seen: seen.add(b["design"]); uniq.append(spec)
    stats = {s_["design"]: collections.Counter() for s_ in uniq}
    rows, t0, nfr = [], time.time(), 0
    for fi, (t, m) in enumerate(iter_topic(args.bag, args.image_topic,
                                           stride=args.stride, limit=args.limit or None)):
        nfr += 1
        gray = to_gray(m)
        if args.equalize: gray = _clahe(gray)
        for spec in uniq:
            st = stats[spec["design"]]
            ids, uv, nm, leg = detect_charuco(gray, spec)
            if nm: st["frames_markers"] += 1; st["max_markers"] = max(st["max_markers"], nm)
            if ids is None: continue
            st["frames_corners"] += 1
            if len(ids) < spec["min_corners"]: st["rej_corners"] += 1; continue
            T_cb, err, ratio = pnp_board(ids, uv, spec, K)
            if T_cb is None or err > spec["max_reproj"]: st["rej_reproj"] += 1; continue
            if ratio < spec["min_ambiguity"]: st["rej_ambiguity"] += 1; continue
            st["accepted"] += 1
            rows.append(dict(t=t, frame=fi, design=spec["design"], T_cb=T_cb,
                             n=len(ids), err=err, ratio=ratio,
                             rng=float(np.linalg.norm(T_cb[:3, 3])),
                             ids=ids, uv=uv, legacy=leg, src="builtin"))
        if fi and fi % 200 == 0:
            print(f"  {fi:5d} frames  {len(rows):4d} sightings  {time.time()-t0:5.1f}s",
                  flush=True)
    print(f"{len(rows)} accepted sightings over {nfr} frames in {time.time()-t0:.1f}s")
    for dgn, st in stats.items():
        print(f"  {dgn:12s} frames w/ markers {st['frames_markers']:5d} "
              f"(max {st['max_markers']}/frame)  w/ corners {st['frames_corners']:5d}  "
              f"accepted {st['accepted']:5d}  rejects c/r/a: "
              f"{st['rej_corners']}/{st['rej_reproj']}/{st['rej_ambiguity']}")
    if rows:
        legs = collections.Counter(r.get("legacy") for r in rows)
        print(f"mean reproj {np.mean([r['err'] for r in rows]):.3f} px "
              f"(survey was 0.22-0.39 px); layout used: {dict(legs)}")
    return rows, K

def probe(args, boards, cfg_boards, n=12):
    """Why zero? Sample n frames, report markers vs corners per dictionary
    (raw and CLAHE), save the frames with detections drawn."""
    import cv2
    print(f"\n--- probe: {n} frames spread over {args.image_topic} ---")
    total = None
    meta = Path(args.bag) / "metadata.yaml"
    if meta.exists():
        try:
            import yaml
            y = yaml.safe_load(meta.read_text())["rosbag2_bagfile_information"]
            for t in y["topics_with_message_count"]:
                if t["topic_metadata"]["name"] == args.image_topic:
                    total = t["message_count"]
        except Exception:
            pass
    stride = max(1, (total or 3000) // n)
    uniq, seen = [], set()
    for b in boards.values():
        d = cfg_boards.get(b["design"], {})
        spec = dict(b, marker_len=d.get("marker_len", 0.75 * b["square_len"]))
        if b["design"] not in seen: seen.add(b["design"]); uniq.append(spec)
    for fi, (t, m) in enumerate(iter_topic(args.bag, args.image_topic,
                                           stride=stride, limit=n)):
        gray = to_gray(m); eq = _clahe(gray)
        line = f"  frame {fi*stride:6d}  mean_px {gray.mean():5.1f}"
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        drew = False
        for spec in uniq:
            ids, uv, nm, leg = detect_charuco(gray, spec)
            ids2, uv2, nm2, _ = detect_charuco(eq, spec)
            nc = 0 if ids is None else len(ids)
            nc2 = 0 if ids2 is None else len(ids2)
            line += (f" | {spec['dictionary'].replace('DICT_','')}: "
                     f"mk {nm}({nm2}eq) corn {nc}({nc2}eq)")
            dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
            mc, mi, _ = cv2.aruco.detectMarkers(eq, dic, parameters=detector_params())
            if mi is not None and len(mi):
                cv2.aruco.drawDetectedMarkers(vis, mc, mi); drew = True
        print(line)
        cv2.imwrite(str(Path(args.out) / f"probe_{'hit' if drew else 'raw'}_{fi*stride:06d}.png"), vis)
    print(f"  frames written to {args.out}/probe_*.png - LOOK at them.")
    print("  markers 0 everywhere -> too small/far, too dark (--equalize), wrong topic/dict")
    print("  markers >0, corners 0 -> board layout mismatch (squares/marker_len/legacy)")
    print("  corners >0, 0 accepted -> gates rejecting (min_corners/max_reproj/ambiguity)")

# ================================================================= stages 0b/0c
def load_vslam(args):
    vo_t, vo_T = [], []
    for t, m in iter_topic(args.bag, args.vo_topic, limit=args.limit or None):
        vo_t.append(t); vo_T.append(odom_to_T(m))
    vo_t = np.array(vo_t); vo_T = np.array(vo_T)
    print(f"VSLAM odometry: {len(vo_t)} poses, {vo_t[-1]-vo_t[0]:.1f} s, "
          f"path {np.sum(np.linalg.norm(np.diff(vo_T[:,:3,3],axis=0),axis=1)):.1f} m")
    return vo_t, vo_T

def assign_instances(rows, boards, vo_t, vo_T):
    """Cluster sightings by board origin in the VSLAM frame, then name each cluster
    by its distance to a cluster of a single-instance design."""
    sight = []
    for r in rows:
        T_vo = interp_traj(vo_t, vo_T, np.array([r["t"]]))[0]
        sight.append(dict(r, p_vo=(T_vo @ r["T_cb"])[:3, 3]))
    def cluster(pts, tol=0.6):
        lab = -np.ones(len(pts), int); c = 0
        for i in range(len(pts)):
            if lab[i] >= 0: continue
            lab[np.linalg.norm(pts - pts[i], axis=1) < tol] = c; c += 1
        return lab, c
    clus = {}
    for dgn in sorted({s["design"] for s in sight}):
        idx = [i for i, s in enumerate(sight) if s["design"] == dgn]
        lab, nc = cluster(np.array([sight[i]["p_vo"] for i in idx]))
        for c in range(nc):
            sel = [idx[k] for k in np.where(lab == c)[0]]
            clus[f"{dgn}#{c}"] = dict(design=dgn, idx=sel,
                p_vo=np.median([sight[i]["p_vo"] for i in sel], axis=0))
    print(f"\n{len(sight)} sightings -> {len(clus)} spatial clusters")
    inst_of = collections.defaultdict(list)
    for n, b in boards.items(): inst_of[b["design"]].append(n)
    ref_cid = next((cid for cid, c in clus.items()
                    if len(inst_of[c["design"]]) == 1), None)
    assign = {}
    if ref_cid is None:
        if len(clus) == 1:
            cid, c = next(iter(clus.items()))
            cands = inst_of[c["design"]]
            print(f"!! single cluster of multi-instance design '{c['design']}' and no "
                  f"unambiguous board seen - cannot name it ({cands}); FIX BY HAND.")
        else:
            print("!! no single-instance design sighted - name clusters by hand.")
    else:
        ref_name = inst_of[clus[ref_cid]["design"]][0]
        assign[ref_cid] = ref_name
        p_ref = clus[ref_cid]["p_vo"]
        print(f"reference cluster {ref_cid} = '{ref_name}'")
        for cid, c in clus.items():
            if cid == ref_cid: continue
            d_meas = float(np.linalg.norm(c["p_vo"] - p_ref))
            cands = {n: float(np.linalg.norm(boards[n]["T_map_board"][:3, 3] -
                                             boards[ref_name]["T_map_board"][:3, 3]))
                     for n in inst_of[c["design"]] if n != ref_name}
            if not cands: continue
            best = min(cands, key=lambda n: abs(cands[n] - d_meas))
            if best in assign.values():
                print(f"  {cid:16s} d_meas={d_meas:6.2f} m -> '{best}' AGAIN: two "
                      f"clusters of one physical board = VSLAM drift between visits "
                      f"split them; merging is correct, and their separation is a "
                      f"free drift measurement between those visit times")
            else:
                print(f"  {cid:16s} d_meas={d_meas:6.2f} m -> '{best}'  "
                      f"(surveyed: {' '.join(f'{n}={v:.2f}' for n, v in cands.items())})")
            assign[cid] = best
    for cid, name in assign.items():
        for i in clus[cid]["idx"]: sight[i]["board"] = name
    sight = [s for s in sight if "board" in s]
    print(f"{len(sight)} assigned sightings across "
          f"{len({s['board'] for s in sight})} boards")
    return sight, assign

def axis_check(sight, boards, vo_t, vo_T, K):
    """Builtin path only: the frame convention was a guess there, so verify it via
    board-to-board relative rotations against the survey (origins cannot discriminate
    conventions; orientations disagree by ~90-180 deg when wrong)."""
    global BOARD_AXES
    if not sight or sight[0].get("src") != "builtin":
        print("\n(frame convention taken from the pipeline's frame_fix - no check needed)")
        return "pipeline_fix"
    if any(s.get("ids") is None for s in sight):
        return BOARD_AXES
    def score(axes):
        Rm = {}
        for s in sight:
            spec = dict(boards[s["board"]],
                        marker_len=0.75 * boards[s["board"]]["square_len"])
            T_cb, err, _ = pnp_board(s["ids"], s["uv"], spec, K, axes=axes)
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
    print("\n=== board-axis convention check (builtin detector) ===")
    scores = {a: score(a) for a in AXIS_CANDIDATES}
    if all(v is None for v in scores.values()):
        print(f"!! <2 distinct boards - convention unchecked; keeping '{BOARD_AXES}'")
        return BOARD_AXES
    for a, v in sorted(scores.items(), key=lambda kv: (kv[1] is None, kv[1])):
        print(f"  {a:9s} " + ("n/a" if v is None else f"{v:8.2f} deg"))
    BOARD_AXES = min((a for a in scores if scores[a] is not None), key=lambda a: scores[a])
    print(f"-> BOARD_AXES = '{BOARD_AXES}'  ({scores[BOARD_AXES]:.2f} deg)")
    for s in sight:
        spec = dict(boards[s["board"]], marker_len=0.75 * boards[s["board"]]["square_len"])
        T_cb, err, _ = pnp_board(s["ids"], s["uv"], spec, K, axes=BOARD_AXES)
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
    """0c: measured inter-board baselines vs the survey. Joint test of board stability
    and VSLAM scale; agreement validates both."""
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
        sc = np.polyfit([r[0] for r in rows], [r[1] for r in rows], 1)[0]
        print(f"implied VSLAM scale {sc:.5f}  ({(sc-1)*1e6:+.0f} ppm)")
        print("  consistent scale != 1 across pairs => VSLAM stereo scale error "
              "(estimable downstream); one pair off while others match => that board moved.")
    elif rows:
        print("only one pair - cannot separate 'board moved' from 'VSLAM scale'.")

def save_outputs(sight, assign, boards, axes_used, out_dir):
    out = Path(out_dir) / "step0_sightings.npz"
    np.savez_compressed(
        out,
        sightings=np.array(sight, dtype=object),
        board_axes=axes_used,
        assign=np.array(list(assign.items()), dtype=object),
        boards=np.array([(n, b["T_map_board"], b["sigma_t"], b["sigma_r"])
                         for n, b in boards.items()], dtype=object),
        allow_pickle=True)
    print(f"\nwrote {out}: {len(sight)} assigned sightings (frame convention: {axes_used})")
    print("next stage (depth->clouds, scan-to-map, A/B/C pose graph) consumes this file.")

# ================================================================= main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bag", required=True, help="coop bag directory (rosbag2/mcap)")
    ap.add_argument("--survey", required=True, help="anchor_frame.json from stage 03")
    ap.add_argument("--config", default="pipeline_config.json",
                    help="pipeline config; its directory also provides pipeline_boards.py")
    ap.add_argument("--out", default="m2_reference")
    ap.add_argument("--image-topic", default="/mobile_2/infra1/image_rect_raw")
    ap.add_argument("--camera-info-topic", default="/mobile_2/infra1/camera_info")
    ap.add_argument("--vo-topic", default="/mobile_2/visual_slam/tracking/odometry")
    ap.add_argument("--stride", type=int, default=2, help="detect every Nth frame")
    ap.add_argument("--limit", type=int, default=0, help="frame cap for smoke tests")
    ap.add_argument("--force", action="store_true", help="ignore the detection cache")
    ap.add_argument("--equalize", action="store_true",
                    help="CLAHE-equalise frames before detection (dark IR images)")
    ap.add_argument("--builtin", action="store_true",
                    help="skip the pipeline detector, use the builtin one")
    ap.add_argument("--boards", default=None,
                    help="comma-separated survey board names to look for "
                         "(e.g. rs_anchor); default: all surveyed boards")
    args = ap.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    boards, meta = load_survey(args.survey)
    if args.boards:
        keep = set(args.boards.split(","))
        missing = keep - set(boards)
        if missing:
            raise SystemExit(f"--boards {sorted(missing)} not in the survey "
                             f"(have: {sorted(boards)})")
        boards = {n: b for n, b in boards.items() if n in keep}
    report_survey(boards)
    cfg_boards = {}
    if args.config and Path(args.config).exists():
        cfg_boards = json.loads(Path(args.config).read_text()).get("boards", {})

    cache = Path(args.out) / "census_raw.npz"
    rows = None
    if cache.exists() and not args.force:
        z = np.load(cache, allow_pickle=True)
        if "ver" in z.files and int(z["ver"]) == 3:
            rows = list(z["rows"])
            print(f"\n=== 0b: census (cached) === {len(rows)} sightings from {cache} "
                  f"(--force to re-detect)")
    K = None
    if rows is None:
        rows = None if args.builtin else census_pipeline(args, boards, meta)
        if rows is None:
            rows, K = census_builtin(args, boards, cfg_boards)
        np.savez_compressed(cache, rows=np.array(rows, dtype=object), ver=3,
                            allow_pickle=True)
    if not rows:
        probe(args, boards, cfg_boards)
        print("\n!! NO SIGHTINGS - read the probe table above. Gate: FAIL.")
        return 1
    if K is None:
        K, _, _ = camera_K(args.bag, args.camera_info_topic)

    vo_t, vo_T = load_vslam(args)
    sight, assign = assign_instances(rows, boards, vo_t, vo_T)
    if not sight:
        print("!! sightings could not be assigned to boards. Gate: FAIL.")
        return 1
    axes_used = axis_check(sight, boards, vo_t, vo_T, K)
    gate_report(sight, vo_t, args.out)
    stability_check(sight, boards)
    save_outputs(sight, assign, boards, axes_used, args.out)
    return 0

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Detect a ChArUco board in the bag's images and write camera-in-map poses.

    # 1. PROVE IT on a pair the pipeline already solved
    python3 scripts/board_detect.py --config configs/coop2.yaml --bag <bag> \
        --agent mobile_1 --anchor anchor --validate-against <zed_cam_in_map.tum>

    # 2. then run it on a pair that has none
    python3 scripts/board_detect.py --config configs/coop2.yaml --bag <bag> \
        --agent mobile_1 --anchor rs_anchor --out runs/boards/mobile_1_rs_anchor.tum

This fills the `observed_poses_by_agent` entries that are null, which is the
only tier in this benchmark whose error bar does not come from the reference.
Three board/agent pairs have a declared dwell window and no detections, and
that is roughly half the available independent evidence.

WHY --validate-against COMES FIRST. Two pairs already have pipeline-produced
files (`anchor`/mobile_1, `rs_anchor`/mobile_2). A detector that cannot
reproduce those, on the same frames, to within their own scatter (2.0 mm and
6.9 mm) is not trustworthy on the three it is being written for. The board
frame convention in particular is easy to get silently wrong -- the outward
normal here is +x, not OpenCV's +z, and the first attempt at this in the
mapping pipeline rejected every view without saying so.

The output is `camera optical frame in map`, TUM, on the bag clock -- exactly
what `absolute_check` consumes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slambench import load_dataset, load_tum, se3                      # noqa: E402
from slambench.trajectory import Trajectory, save_tum                  # noqa: E402

CAMERA = {"mobile_1": ("/mobile_1/zed/left/image_rect_color",
                       "/mobile_1/zed/left/camera_info"),
          "mobile_2": ("/mobile_2/color/image_raw",
                       "/mobile_2/color/camera_info")}

# The board's own axes. OpenCV builds a ChArUco board in its XY plane with +z
# as the normal; this dataset declares `axes: ros`, whose outward normal is +x.
# So a board point (u, v, 0) from OpenCV maps to (0, -u, -v) in the board
# frame: +x out of the board, +y to the board's left, +z up, which is the ROS
# body convention applied to a plane. Written once, checked against the
# pipeline's own detections by --validate-against rather than by argument.
# MEASURED, not derived. `--solve-axes` re-solved 90 detections of `anchor`
# under all eight possible planar conventions and scored each against the
# mapping pipeline's own zed_cam_in_map.tum:
#
#     ros+0      1448.6 mm   179.84 deg        rosflip+0      412.3 mm  179.94 deg
#     ros+90     1441.6 mm   179.77 deg        rosflip+90     295.0 mm   90.03 deg
#     ros+180    1399.9 mm   179.57 deg    ->  rosflip+180      6.0 mm    0.52 deg
#     ros+270    1407.2 mm   179.66 deg        rosflip+270    288.1 mm   89.97 deg
#
# 6.0 mm against that board's own 6.86 mm survey scatter, with the runner-up
# 288 mm away: decisive. The winner is the cyclic permutation below --
# x_board = z_opencv, y_board = x_opencv, z_board = y_opencv.
#
# TWO EARLIER GUESSES WERE WRONG, both from reading prose. The first read
# OpenCV's z as the outward normal with y flipped (412 mm, 179.9 deg); the
# second, from stage 03's own docstring, flipped the normal instead (1449 mm).
# Conventions are cheap to measure and expensive to argue about.
OPENCV_TO_ROS = np.array([[0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])


def _spin_x(deg: float) -> np.ndarray:
    c, s_ = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]])


# The other chirality, kept so --solve-axes still searches the COMPLETE set:
# eight layouts, four spins in each of two mirror images. It has to remain a
# PROPER rotation -- a board frame with det -1 would report a left-handed
# orientation that no physical board has -- so the in-plane axis and the
# NORMAL are both flipped. That flips the 2-D corner layout while keeping
# det +1, which is exactly the pair of possibilities a planar target leaves.
_MIRROR = OPENCV_TO_ROS @ np.diag([-1.0, 1.0, -1.0])

AXES_CANDIDATES = {}
for _tag, _base in (("ros", OPENCV_TO_ROS), ("rosmirror", _MIRROR)):
    for _d in (0, 90, 180, 270):
        AXES_CANDIDATES[f"{_tag}+{_d}"] = _spin_x(_d) @ _base


def board_to_map(position, orientation_xyzw) -> np.ndarray:
    """4x4 `map <- board`, from the two numbers the dataset config declares."""
    T = np.eye(4)
    T[:3, :3] = se3.quat_to_rot(np.asarray(orientation_xyzw, dtype=np.float64))
    T[:3, 3] = np.asarray(position, dtype=np.float64)
    return T


def to_board_frame(corners_opencv: np.ndarray, squares, square_m: float,
                   origin: str, axes: str) -> np.ndarray:
    """OpenCV's chessboard corners -> the BOARD frame this dataset declares.

    The corner LIST comes from the OpenCV board object, so its ordering matches
    the charuco ids exactly and cannot drift from whatever that version does.
    Only the frame is changed here: `origin: center` puts the board's centre at
    the origin, where the survey put it, and `axes: ros` turns OpenCV's plane
    (normal +z) into this dataset's (normal +x).
    """
    pts = np.asarray(corners_opencv, dtype=np.float64).reshape(-1, 3).copy()
    if origin == "center":
        pts[:, 0] -= squares[0] * square_m / 2.0
        pts[:, 1] -= squares[1] * square_m / 2.0
    elif origin != "corner":
        raise ValueError(f"board origin {origin!r}: expected 'center' or 'corner'")
    if axes == "opencv":
        return pts
    if axes in AXES_CANDIDATES:
        return pts @ AXES_CANDIDATES[axes].T
    if axes == "ros":
        return pts @ OPENCV_TO_ROS.T
    raise ValueError(f"board axes {axes!r}: expected 'opencv', 'ros' or one of "
                     f"{sorted(AXES_CANDIDATES)}")


def object_points(squares, square_m: float, origin: str, axes: str) -> np.ndarray:
    """The inner corners analytically, for tests and for checking the board
    object agrees. A 9x7-square board has 8x6 of them, row-major."""
    nx, ny = int(squares[0]) - 1, int(squares[1]) - 1
    u, v = np.meshgrid(np.arange(nx), np.arange(ny))
    pts = np.stack([(u.ravel() + 1) * square_m,
                    (v.ravel() + 1) * square_m,
                    np.zeros(nx * ny)], axis=1)
    return to_board_frame(pts, squares, square_m, origin, axes)


def charuco_board(b: dict, cv2):
    """(board, dictionary, detect) across the 4.7 ArUco rewrite.

    4.5 has `CharucoBoard_create` + `detectMarkers` + `interpolateCornersCharuco`;
    4.7 replaced them with `CharucoBoard` + `CharucoDetector`. The robot runs
    4.5.4 and a newer machine will not, so both are supported rather than one
    being pinned -- a detector that only runs in one place is not reproducible.
    """
    sx, sy = int(b["squares"][0]), int(b["squares"][1])
    sl, ml = float(b["square_m"]), float(b["marker_m"])
    dict_id = getattr(cv2.aruco, b["dictionary"])
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        adict = cv2.aruco.getPredefinedDictionary(dict_id)
    else:
        adict = cv2.aruco.Dictionary_get(dict_id)

    def tune(params):
        """Defaults are set for large, sharp markers. These boards span 11-15 px
        per marker, i.e. about 2 pixels per bit, so every default that assumes
        a comfortable marker works against them. Each change below is aimed at
        that regime and at nothing else."""
        # a small marker is a small fraction of the frame
        params.minMarkerPerimeterRate = 0.01
        # sweep more thresholds: the boards are lit unevenly and one window
        # size will not binarise both a bright and a shadowed corner
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 43
        params.adaptiveThreshWinSizeStep = 4
        # at 2 px per bit the sampled cell is tiny; take more of it and demand
        # less contrast before calling a bit set
        params.perspectiveRemovePixelPerCell = 8
        params.perspectiveRemoveIgnoredMarginPerCell = 0.1
        params.maxErroneousBitsInBorderRate = 0.5
        params.errorCorrectionRate = 0.8
        # sub-pixel corners: the reprojection is already 0.2 px, and the corner
        # positions are what the pose is made of
        if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return params

    if hasattr(cv2.aruco, "CharucoDetector"):                      # >= 4.7
        board = cv2.aruco.CharucoBoard((sx, sy), sl, ml, adict)
        cp = cv2.aruco.CharucoParameters()
        det = cv2.aruco.CharucoDetector(board, cp,
                                        tune(cv2.aruco.DetectorParameters()))

        def detect(gray):
            corners, ids, _, _ = det.detectBoard(gray)
            return (None, None) if ids is None or not len(ids) else (corners, ids)
    else:                                                          # 4.5 / 4.6
        board = cv2.aruco.CharucoBoard_create(sx, sy, sl, ml, adict)
        params = tune(cv2.aruco.DetectorParameters_create())

        def detect(gray):
            mc, mids, rej = cv2.aruco.detectMarkers(gray, adict, parameters=params)
            # A marker whose ID failed to decode is still a quadrilateral in
            # `rej`. The board says where its markers must be, so most of those
            # can be recovered -- which matters when the median frame is
            # finding one corner out of 48.
            if hasattr(cv2.aruco, "refineDetectedMarkers"):
                mc, mids, rej, _ = cv2.aruco.refineDetectedMarkers(
                    gray, board, mc, mids, rej, parameters=params)
            if mids is None or not len(mids):
                return None, None
            n, cc, cids = cv2.aruco.interpolateCornersCharuco(mc, mids, gray, board)
            if not n or cids is None or not len(cids):
                return None, None
            return cc, cids

    raw = (board.getChessboardCorners() if hasattr(board, "getChessboardCorners")
           else np.asarray(board.chessboardCorners))
    return board, adict, detect, np.asarray(raw, dtype=np.float64).reshape(-1, 3)


def solve_view(obj, img, K, dist, cv2):
    """PnP for a PLANAR target: the best pose, its reprojection, and how much
    better it is than the runner-up.

    A planar board always admits two solutions. Taking the better one without
    asking how much better is how a flipped pose gets into a file looking
    perfectly clean -- so the ratio of the two reprojection errors is returned
    and gated on, exactly as the mapping pipeline's `min_ambiguity_ratio`
    does. Infinite ratio means only one solution was offered.
    """
    if hasattr(cv2, "solvePnPGeneric"):
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not n:
            return None, None, float("inf"), 0.0
        e = np.asarray(errs, dtype=np.float64).ravel()
        order = np.argsort(e)
        best = int(order[0])
        ratio = float(e[order[1]] / max(e[best], 1e-9)) if n > 1 else float("inf")
        return rvecs[best], tvecs[best], float(e[best]), ratio
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None, float("inf"), 0.0
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum(
        (proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, axis=1))))
    return rvec, tvec, err, float("inf")


def camera_in_map(rvec, tvec, T_map_board) -> np.ndarray:
    """PnP gives `board -> camera`. The output wants `map <- camera`."""
    import cv2
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    T_cam_board = np.eye(4)
    T_cam_board[:3, :3] = R
    T_cam_board[:3, 3] = np.asarray(tvec, dtype=np.float64).ravel()
    return T_map_board @ se3.invert(T_cam_board)


def sanity_gate(T_map_cam: np.ndarray, t: float, ref: Trajectory,
                max_pos_m: float, max_dt_s: float = 0.05):
    """Is this pose anywhere near where the robot actually was?

    `anchor` and `anchor_b` are the SAME board design with the SAME marker ids
    (DICT_4X4_50, id_offset 0 on both), so nothing in the image distinguishes
    them: a run aimed at one can solve the other and produce a confident pose
    tens of metres away. A mirrored PnP solution looks the same from here.
    Either way the reference says where the camera was to a few centimetres,
    so a detection that disagrees with it by more than `max_pos_m` is the
    wrong board or the wrong solution, and is dropped with its distance
    reported rather than silently kept.

    Returns (ok, distance_m). A stamp the reference does not cover cannot be
    gated, and is kept -- the reference's absence is not evidence against a
    detection.
    """
    j = int(np.argmin(np.abs(ref.stamps - t)))
    if abs(ref.stamps[j] - t) > max_dt_s:
        return True, float("nan")
    d = float(np.linalg.norm(T_map_cam[:3, 3] - ref.positions[j]))
    return d <= max_pos_m, d


def compare(got: Trajectory, ref: Trajectory, max_dt_s: float = 0.02) -> dict:
    """Detections against the pipeline's, on matched stamps. The verdict this
    script exists to print before anyone trusts a new file."""
    dp, dr, n = [], [], 0
    for k, t in enumerate(got.stamps):
        j = int(np.argmin(np.abs(ref.stamps - t)))
        if abs(ref.stamps[j] - t) > max_dt_s:
            continue
        n += 1
        dp.append(np.linalg.norm(got.positions[k] - ref.positions[j]))
        dr.append(np.degrees(se3.rotation_angle(
            got.poses[k][:3, :3].T @ ref.poses[j][:3, :3])))
    if not n:
        return {"matched": 0}
    return {"matched": n, "pos_median_mm": float(np.median(dp)) * 1e3,
            "pos_p90_mm": float(np.percentile(dp, 90)) * 1e3,
            "rot_median_deg": float(np.median(dr))}


def main() -> int:                                           # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--validate-against", default=None,
                    help="a pipeline-produced *_cam_in_map.tum for this same pair")
    ap.add_argument("--min-corners", type=int, default=None,
                    help="overrides the board's declared min_corners")
    ap.add_argument("--max-reproj-px", type=float, default=None)
    ap.add_argument("--min-ambiguity-ratio", type=float, default=None)
    ap.add_argument("--window", type=float, nargs=2, default=None,
                    metavar=("T0", "T1"),
                    help="override the declared dwell window. The rs_anchor window "
                         "for mobile_2 was WIDENED to the union with the pipeline's "
                         "detections so tier 2 could intersect them; for a fresh "
                         "detector run that union carries ~330 frames where the board "
                         "is too far to read, and they drown the rate.")
    ap.add_argument("--reference", default=None,
                    help="the agent's reference TUM. Used ONLY to reject a pose that "
                         "cannot be where the robot was -- see sanity_gate. Required "
                         "for a board whose markers are shared with another.")
    ap.add_argument("--solve-axes", action="store_true",
                    help="with --validate-against: score EVERY candidate board-axis "
                         "convention against the pipeline's own detections and print "
                         "which one is right. The first guess was 179.96 deg out, and "
                         "arguing about conventions is slower than measuring them.")
    ap.add_argument("--max-offset-m", type=float, default=0.5,
                    help="how far a detection may sit from the reference before it is "
                         "treated as the wrong board or a mirrored solution")
    args = ap.parse_args()

    try:
        import cv2
    except ImportError:
        print("opencv is not installed: pip3 install opencv-python", file=sys.stderr)
        return 2
    if not hasattr(cv2, "aruco"):
        print("this opencv has no aruco module: pip3 install opencv-contrib-python",
              file=sys.stderr)
        return 2

    cfg = load_dataset(args.config)
    anchor = next((a for a in cfg.reference.get("anchors", [])
                   if a["name"] == args.anchor), None)
    if anchor is None:
        raise SystemExit(f"no anchor {args.anchor!r} in {args.config}")
    for k in ("orientation", "board"):
        if not anchor.get(k):
            raise SystemExit(f"anchor {args.anchor!r} declares no {k}; a detector "
                             f"cannot be run on a board whose geometry is not stated")
    window = args.window or (anchor.get("windows_by_agent") or {}).get(args.agent)
    if args.window:
        print(f"window overridden: {window[0]:.3f} .. {window[1]:.3f} "
              f"({window[1] - window[0]:.1f} s)")
    if not window:
        raise SystemExit(f"anchor {args.anchor!r} declares no window for {args.agent}: "
                         f"this agent never dwelled there, and a detector run over the "
                         f"whole bag would be a fishing expedition, not a measurement")

    b = anchor["board"]
    # The gates are the mapping pipeline's own, carried in the dataset config.
    min_corners = args.min_corners or int(b.get("min_corners", 8))
    max_reproj = args.max_reproj_px or float(b.get("max_reproj_px", 1.5))
    min_ratio = args.min_ambiguity_ratio or float(b.get("min_ambiguity_ratio", 1.0))
    print(f"gates: >= {min_corners} corners, reprojection <= {max_reproj} px, "
          f"ambiguity ratio >= {min_ratio}")
    shared = [a["name"] for a in cfg.reference.get("anchors", [])
              if a.get("board") and a["name"] != args.anchor
              and a["board"].get("dictionary") == b.get("dictionary")
              and a["board"].get("id_offset", 0) == b.get("id_offset", 0)]
    ref_traj = load_tum(args.reference, name=f"reference/{args.agent}") if args.reference else None
    if shared and ref_traj is None:
        raise SystemExit(
            f"{args.anchor!r} carries the same markers as {shared} (same dictionary, "
            f"same id_offset), so the image cannot tell them apart. Pass --reference "
            f"so a solved pose can be checked against where the robot actually was.")
    T_map_board = board_to_map(anchor["position"], anchor["orientation"])
    obj = object_points(b["squares"], float(b["square_m"]),
                        b.get("origin", "center"), b.get("axes", "ros"))
    board, adict, detect, raw = charuco_board(b, cv2)
    obj_cv = to_board_frame(raw, b["squares"], float(b["square_m"]),
                            b.get("origin", "center"), b.get("axes", "ros"))
    # The board object and the analytic construction must agree, or one of the
    # two is wrong about this version's corner layout and every pose is wrong.
    if obj_cv.shape != obj.shape or not np.allclose(np.sort(obj_cv, axis=0),
                                                    np.sort(obj, axis=0), atol=1e-9):
        raise SystemExit(f"opencv {cv2.__version__} lays this board out differently "
                         f"from configs/coop2.yaml ({obj_cv.shape} vs {obj.shape}); "
                         f"refusing to guess which is right")
    obj = obj_cv                     # the board object's ordering, which the ids index

    img_topic, info_topic = CAMERA[args.agent]
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    sys.path.insert(0, str(ROOT / "docker" / "common"))
    from bag_to_frames import to_rgb

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id=""),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=[img_topic, info_topic]))
    cls = {t: get_message(types[t]) for t in (img_topic, info_topic)}

    K = dist = None
    stamps, poses, tried, detected, rejected = [], [], 0, 0, []
    from collections import Counter
    why, n_corners, reproj, solved, ratios = Counter(), [], [], [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, cls[topic])
        if topic == info_topic:
            if K is None:
                K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
                # NOT assumed zero. mobile_1's topic is `*_rect_*` and its
                # distortion really is out; mobile_2's is `image_raw` and is
                # NOT rectified, and zeroing it there bends every corner by a
                # few pixels -- which is the difference between a detection and
                # a rejection on a marker that spans 14 px.
                d = np.array(msg.d, dtype=np.float64).ravel()
                rect = "rect" in img_topic
                if rect and np.any(np.abs(d) > 1e-9):
                    print(f"{img_topic} says rectified but camera_info carries "
                          f"non-zero distortion {d[:5]}; using it anyway",
                          file=sys.stderr)
                dist = np.zeros(5) if rect else (d if len(d) else np.zeros(5))
                print(f"intrinsics {K[0,0]:.1f} {K[1,1]:.1f} {K[0,2]:.1f} {K[1,2]:.1f}"
                      f"  distortion {'zeroed (rectified topic)' if rect else list(np.round(dist, 5))}")
            continue
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if not (window[0] <= t <= window[1]) or K is None:
            continue
        tried += 1
        gray = cv2.cvtColor(to_rgb(bytes(msg.data), msg.height, msg.width,
                                   msg.encoding, msg.step), cv2.COLOR_RGB2GRAY)
        corners, ids = detect(gray)
        # Count WHERE frames are lost. "7 of 521" says nothing; "500 saw no
        # marker at all" and "500 saw five corners" want different fixes.
        if ids is None:
            why["no corners"] += 1
            continue
        n_corners.append(len(ids))
        if len(ids) < min_corners:
            why[f"< {min_corners} corners"] += 1
            continue
        rvec, tvec, err, ratio = solve_view(obj[ids.ravel()],
                                            corners.reshape(-1, 2), K, dist, cv2)
        if rvec is None:
            why["solvePnP failed"] += 1
            continue
        ratios.append(ratio)
        if ratio < min_ratio:
            why[f"ambiguity ratio < {min_ratio}"] += 1
            continue
        reproj.append(err)
        if err > max_reproj:
            why[f"reprojection > {max_reproj} px"] += 1
            continue
        solved.append(((corners, ids), t))
        T_map_cam = camera_in_map(rvec, tvec, T_map_board)
        if ref_traj is not None:
            ok, d = sanity_gate(T_map_cam, t, ref_traj, args.max_offset_m)
            if not ok:
                rejected.append(d)
                continue
        detected += 1
        stamps.append(t)
        poses.append(T_map_cam)

    print(f"{detected} of {tried} frames in the window gave a pose "
          f"(>= {min_corners} corners, reprojection <= {max_reproj} px)")
    for reason, n in why.most_common():
        print(f"  {n:5d} frames lost: {reason}")
    if n_corners:
        print(f"  corners per frame that saw any: median {int(np.median(n_corners))}, "
              f"max {max(n_corners)} of {len(obj)}")
    if ratios:
        finite = [r for r in ratios if np.isfinite(r)]
        print(f"  ambiguity ratio: median "
              f"{np.median(finite):.2f}" if finite else "  ambiguity: single solution",
              end="")
        print(f" over {len(ratios)} solved views")
    if reproj:
        print(f"  reprojection of solved frames: median {np.median(reproj):.2f} px, "
              f"p90 {np.percentile(reproj, 90):.2f} px")
    if rejected:
        print(f"{len(rejected)} solved poses REJECTED for sitting "
              f"{np.median(rejected):.2f} m (median) from the reference, over the "
              f"{args.max_offset_m} m gate"
              + (f" -- {args.anchor} shares its markers with {shared}, so these are "
                 f"very likely the other board" if shared else
                 " -- likely mirrored PnP solutions"))
    if args.solve_axes:
        if not args.validate_against:
            raise SystemExit("--solve-axes needs --validate-against: it is scored "
                             "against the pipeline's own detections, not guessed")
        truth = load_tum(args.validate_against)
        print(f"\n{len(solved)} raw detections available. The sanity gate is NOT "
              f"applied here: it compares a pose against the reference, and a pose "
              f"is only meaningful once the convention is known -- which is what "
              f"this is for.")
        print("candidate board-axis conventions, scored against the pipeline:")
        best = None
        for name, M in AXES_CANDIDATES.items():
            pts = to_board_frame(raw, b["squares"], float(b["square_m"]),
                                 b.get("origin", "center"), "opencv") @ M.T
            poses_n = []
            for rv, tv in solved:
                r2, t2, _e, _r = solve_view(pts[np.asarray(rv[1]).ravel()],
                                            np.asarray(rv[0]).reshape(-1, 2),
                                            K, dist, cv2)
                if r2 is not None:
                    poses_n.append((tv, camera_in_map(r2, t2, T_map_board)))
            if not poses_n:
                continue
            cand = Trajectory(np.array([t for t, _ in poses_n]),
                              np.stack([P for _, P in poses_n]), name)
            v = compare(cand, truth)
            if not v["matched"]:
                print(f"  {name:<8} no matched stamp")
                continue
            print(f"  {name:<8} position median {v['pos_median_mm']:8.1f} mm   "
                  f"rotation median {v['rot_median_deg']:7.2f} deg   n={v['matched']}")
            if best is None or v["pos_median_mm"] < best[1]["pos_median_mm"]:
                best = (name, v)
        if best:
            n, v = best
            print(f"\n-> `axes: {n}` wins at {v['pos_median_mm']:.1f} mm / "
                  f"{v['rot_median_deg']:.2f} deg. Put that in coop2.yaml's board "
                  f"block for every anchor and re-run the validation before "
                  f"writing any new file.")
        return 0


    if not stamps:
        print("no detection: the board was in frame by the reference's reckoning but "
              "the detector found nothing. Occlusion, blur or exposure -- the three "
              "things board_visibility.py says it cannot settle.", file=sys.stderr)
        return 1
    traj = Trajectory(np.array(stamps), np.stack(poses),
                      f"{args.agent}/{args.anchor}", frame="camera", world="map")

    if args.validate_against:
        v = compare(traj, load_tum(args.validate_against))
        if not v["matched"]:
            print("VALIDATION IMPOSSIBLE: no stamp matched the pipeline's file",
                  file=sys.stderr)
            return 1
        print(f"vs the pipeline, {v['matched']} matched frames: "
              f"position median {v['pos_median_mm']:.1f} mm, p90 {v['pos_p90_mm']:.1f} mm, "
              f"rotation median {v['rot_median_deg']:.2f} deg")
        print("The board's own scatter is 2.0 mm (anchor/zed) and 6.9 mm "
              "(rs_anchor/realsense). Agreement at that level means the frame "
              "convention and the intrinsics are right; anything larger means "
              "they are not, and no new file should be written until they are.")

    if args.out:
        save_tum(traj, args.out)
        print(f"{len(traj)} poses -> {args.out}\n"
              f"point reference.anchors[{args.anchor}].observed_poses_by_agent."
              f"{args.agent} at it, then re-run scripts/rescore.py")
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    sys.exit(main())

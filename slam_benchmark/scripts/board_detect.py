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
OPENCV_TO_ROS = np.array([[0.0, 0.0, 1.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0]])


def board_to_map(position, orientation_xyzw) -> np.ndarray:
    """4x4 `map <- board`, from the two numbers the dataset config declares."""
    T = np.eye(4)
    T[:3, :3] = se3.quat_to_rot(np.asarray(orientation_xyzw, dtype=np.float64))
    T[:3, 3] = np.asarray(position, dtype=np.float64)
    return T


def object_points(squares, square_m: float, origin: str, axes: str) -> np.ndarray:
    """The ChArUco inner corners in the BOARD frame, in OpenCV corner order.

    A 9x7-square board has 8x6 inner corners. OpenCV numbers them row-major
    from the corner it calls the origin; `origin: center` shifts them so the
    board's centre is the frame origin, which is where the survey put it.
    """
    nx, ny = int(squares[0]) - 1, int(squares[1]) - 1
    u, v = np.meshgrid(np.arange(nx), np.arange(ny))
    pts = np.stack([(u.ravel() + 1) * square_m,
                    (v.ravel() + 1) * square_m,
                    np.zeros(nx * ny)], axis=1)
    if origin == "center":
        pts[:, 0] -= squares[0] * square_m / 2.0
        pts[:, 1] -= squares[1] * square_m / 2.0
    elif origin != "corner":
        raise ValueError(f"board origin {origin!r}: expected 'center' or 'corner'")
    if axes == "ros":
        return pts @ OPENCV_TO_ROS.T
    if axes == "opencv":
        return pts
    raise ValueError(f"board axes {axes!r}: expected 'ros' or 'opencv'")


def camera_in_map(rvec, tvec, T_map_board) -> np.ndarray:
    """PnP gives `board -> camera`. The output wants `map <- camera`."""
    import cv2
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    T_cam_board = np.eye(4)
    T_cam_board[:3, :3] = R
    T_cam_board[:3, 3] = np.asarray(tvec, dtype=np.float64).ravel()
    return T_map_board @ se3.invert(T_cam_board)


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
    ap.add_argument("--min-corners", type=int, default=6,
                    help="inner corners needed before a view is used at all")
    ap.add_argument("--max-reproj-px", type=float, default=1.5)
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
    window = (anchor.get("windows_by_agent") or {}).get(args.agent)
    if not window:
        raise SystemExit(f"anchor {args.anchor!r} declares no window for {args.agent}: "
                         f"this agent never dwelled there, and a detector run over the "
                         f"whole bag would be a fishing expedition, not a measurement")

    b = anchor["board"]
    T_map_board = board_to_map(anchor["position"], anchor["orientation"])
    obj = object_points(b["squares"], float(b["square_m"]),
                        b.get("origin", "center"), b.get("axes", "ros"))
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, b["dictionary"]))
    board = cv2.aruco.CharucoBoard((int(b["squares"][0]), int(b["squares"][1])),
                                   float(b["square_m"]), float(b["marker_m"]), adict)
    detector = cv2.aruco.CharucoDetector(board)

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
    stamps, poses, tried, detected = [], [], 0, 0
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, cls[topic])
        if topic == info_topic:
            if K is None:
                K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
                # *_rect_* / aligned images: the distortion is already out, and
                # applying the plumb-bob a second time would bend every corner.
                dist = np.zeros(5)
                print(f"intrinsics {K[0,0]:.1f} {K[1,1]:.1f} {K[0,2]:.1f} {K[1,2]:.1f}")
            continue
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if not (window[0] <= t <= window[1]) or K is None:
            continue
        tried += 1
        gray = cv2.cvtColor(to_rgb(bytes(msg.data), msg.height, msg.width,
                                   msg.encoding, msg.step), cv2.COLOR_RGB2GRAY)
        corners, ids, _, _ = detector.detectBoard(gray)
        if ids is None or len(ids) < args.min_corners:
            continue
        ok, rvec, tvec = cv2.solvePnP(obj[ids.ravel()], corners.reshape(-1, 2),
                                      K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj[ids.ravel()], rvec, tvec, K, dist)
        err = float(np.sqrt(np.mean(np.sum(
            (proj.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1))))
        if err > args.max_reproj_px:
            continue
        detected += 1
        stamps.append(t)
        poses.append(camera_in_map(rvec, tvec, T_map_board))

    print(f"{detected} of {tried} frames in the window gave a pose "
          f"(>= {args.min_corners} corners, reprojection <= {args.max_reproj_px} px)")
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

#!/usr/bin/env python3
"""Interactively pull chair observations out of the rosbag into an observations
YAML that agreement.py consumes.

Requires ROS 2 (rclpy, rosbag2_py, sensor_msgs, cv_bridge, sensor_msgs_py) and
OpenCV with a GUI backend. This is the environment-specific glue; the analysis
in agreement.py is what carries the math (and is unit-tested independently).

Workflow per frame you want to capture:
  1. Pick a time where the chair is clearly visible to several sensors.
  2. For each RGBD sensor: an image pops up -> click the chair. Depth at the click
     (median over a small window) is deprojected to a 3D point in the optical
     frame; the platform pose nearest that stamp is looked up.
  3. For the LiDAR: give a rough 3D ROI box (--lidar-roi) around the chair; the
     centroid of the points inside is taken in os_lidar.
  4. For the Arducam: an image pops up -> click the chair; the pixel is stored.
  5. Rows are appended to the output YAML.

Example:
  python3 extract_observations.py mirc_dataset_20260706_complete \\
      --times 12.48 14.96 21.30 \\
      --out observations.yaml

Topics default to the mirc_dataset names; override with the --*-topic flags.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import yaml

# ROS 2 imports are deferred so `--help` and import-time errors are legible.
def _need_ros():
    try:
        import rclpy  # noqa: F401
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions  # noqa: F401
        from rclpy.serialization import deserialize_message  # noqa: F401
        from sensor_msgs.msg import Image, PointCloud2, CameraInfo  # noqa: F401
        from geometry_msgs.msg import PoseStamped  # noqa: F401
        from sensor_msgs_py import point_cloud2  # noqa: F401
        from cv_bridge import CvBridge  # noqa: F401
        import cv2  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        sys.exit(f"This extractor needs ROS 2 + OpenCV in your environment.\n"
                 f"Import failed: {e}\n"
                 f"(The analysis in agreement.py has no ROS dependency.)")


DEFAULT_TOPICS = {
    "zed_image": "/zed/zed_node/left/image_rect_color",
    "zed_depth": "/zed/zed_node/depth/depth_registered",
    "zed_pose": "/glim/camera_pose",
    "rs_image": "/camera/camera/color/image_raw",
    "rs_depth": "/camera/camera/depth/image_rect_raw",
    "rs_pose": "/vo_pose",
    "lidar": "/ouster/points",
    "arducam": "/arducam/image_raw",
}


def read_bag(path, topics):
    """Yield (topic, stamp_sec, raw_bytes) in log order for the wanted topics."""
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

    reader = SequentialReader()
    reader.open(StorageOptions(uri=path, storage_id="sqlite3"),
                ConverterOptions("", ""))
    wanted = set(topics)
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic in wanted:
            yield topic, t_ns * 1e-9, data


def nearest(msgs, t):
    """msgs: list of (stamp, payload). Return payload nearest t (or None)."""
    if not msgs:
        return None
    i = int(np.argmin([abs(s - t) for s, _ in msgs]))
    return msgs[i][1]


def deproject_click(image_msg, depth_msg, K, win=5):
    """Show the image, take one click, deproject via the aligned depth. -> (point, pixel)."""
    import cv2
    from cv_bridge import CvBridge

    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
    depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32)
    if depth.max() > 100:      # millimetres -> metres
        depth = depth / 1000.0

    click = {}
    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["uv"] = (x, y)
    cv2.namedWindow("click the chair")
    cv2.setMouseCallback("click the chair", on_click)
    while "uv" not in click:
        cv2.imshow("click the chair", img)
        if cv2.waitKey(20) & 0xFF == 27:  # ESC to skip
            cv2.destroyAllWindows()
            return None, None
    cv2.destroyAllWindows()

    u, v = click["uv"]
    patch = depth[max(0, v - win):v + win, max(0, u - win):u + win]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        print("  no valid depth at that pixel; skipping")
        return None, [float(u), float(v)]
    d = float(np.median(valid))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) / fx * d
    y = (v - cy) / fy * d
    return [float(x), float(y), d], [float(u), float(v)]


def click_pixel(image_msg):
    import cv2
    from cv_bridge import CvBridge
    img = CvBridge().imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
    click = {}
    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["uv"] = (x, y)
    cv2.namedWindow("click the chair")
    cv2.setMouseCallback("click the chair", on_click)
    while "uv" not in click:
        cv2.imshow("click the chair", img)
        if cv2.waitKey(20) & 0xFF == 27:
            cv2.destroyAllWindows()
            return None
    cv2.destroyAllWindows()
    return [float(click["uv"][0]), float(click["uv"][1])]


def lidar_centroid(pc_msg, roi):
    """Centroid of os_lidar points inside an axis-aligned ROI box [xmin..zmax]."""
    from sensor_msgs_py import point_cloud2
    pts = np.array([[p[0], p[1], p[2]] for p in
                    point_cloud2.read_points(pc_msg, field_names=("x", "y", "z"),
                                             skip_nans=True)])
    xmin, xmax, ymin, ymax, zmin, zmax = roi
    m = ((pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
         (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax) &
         (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax))
    if m.sum() == 0:
        print("  no LiDAR points in ROI; skipping")
        return None
    return pts[m].mean(axis=0).tolist()


def pose_to_dict(pose_msg):
    p = pose_msg.pose.position if hasattr(pose_msg, "pose") else pose_msg.position
    q = pose_msg.pose.orientation if hasattr(pose_msg, "pose") else pose_msg.orientation
    return {"trans": [p.x, p.y, p.z], "quat": [q.x, q.y, q.z, q.w]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="rosbag2 directory (sqlite3)")
    ap.add_argument("--times", type=float, nargs="+", required=True,
                    help="bag-relative times (s) to capture frames at")
    ap.add_argument("--out", default="observations.yaml")
    ap.add_argument("--lidar-roi", type=float, nargs=6, default=None,
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                    help="os_lidar ROI box around the chair (metres)")
    for k, v in DEFAULT_TOPICS.items():
        ap.add_argument(f"--{k}-topic", default=v)
    args = ap.parse_args()

    _need_ros()
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image, PointCloud2
    from geometry_msgs.msg import PoseStamped

    topics = {getattr(args, f"{k}_topic"): k for k in DEFAULT_TOPICS}
    # buffer everything we need (bag is short: ~97 s)
    buf = {k: [] for k in DEFAULT_TOPICS}
    typemap = {"zed_image": Image, "zed_depth": Image, "zed_pose": PoseStamped,
               "rs_image": Image, "rs_depth": Image, "rs_pose": PoseStamped,
               "lidar": PointCloud2, "arducam": Image}
    t0 = None
    for topic, t, data in read_bag(args.bag, topics.keys()):
        t0 = t if t0 is None else t0
        key = topics[topic]
        buf[key].append((t - t0, deserialize_message(data, typemap[key])))
    print({k: len(v) for k, v in buf.items()})

    frames = []
    for i, tt in enumerate(args.times):
        print(f"\n=== frame {i} @ t={tt:.3f}s ===")
        sensors = {}

        # ZED
        img = nearest(buf["zed_image"], tt); dep = nearest(buf["zed_depth"], tt)
        if img is not None and dep is not None:
            from dataset import ZED_LEFT
            print("ZED: click the chair (ESC to skip)")
            pt, px = deproject_click(img, dep, ZED_LEFT.K)
            pose = nearest(buf["zed_pose"], tt)
            if pt:
                sensors["zed"] = {"point": pt, "pixel": px,
                                  "pose_map": pose_to_dict(pose) if pose else None}

        # RealSense
        img = nearest(buf["rs_image"], tt); dep = nearest(buf["rs_depth"], tt)
        if img is not None and dep is not None:
            from dataset import REALSENSE
            print("RealSense: click the chair (ESC to skip)")
            pt, px = deproject_click(img, dep, REALSENSE.K)
            pose = nearest(buf["rs_pose"], tt)
            if pt:
                sensors["realsense"] = {"point": pt, "pixel": px,
                                        "pose_map": pose_to_dict(pose) if pose else None}

        # LiDAR
        if args.lidar_roi:
            pc = nearest(buf["lidar"], tt)
            if pc is not None:
                c = lidar_centroid(pc, args.lidar_roi)
                if c:
                    sensors["lidar"] = {"point": c}

        # Arducam
        img = nearest(buf["arducam"], tt)
        if img is not None:
            print("Arducam: click the chair (ESC to skip)")
            px = click_pixel(img)
            if px:
                sensors["arducam"] = {"pixel": px}

        frames.append({"id": i, "t": float(tt), "sensors": sensors})

    with open(args.out, "w") as f:
        yaml.safe_dump({"frames": frames}, f, sort_keys=False)
    print(f"\nwrote {len(frames)} frame(s) -> {args.out}")
    print("now run:  python3 agreement.py", args.out, "--lidar-interp auto")


if __name__ == "__main__":
    main()

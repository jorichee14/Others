#!/usr/bin/env python3
"""ROS 2 node: watch every camera for the board and publish its pose on ONE topic for all cameras.

Every frame in which a camera sees the board (enough corners, low reprojection error) gives one
geometry_msgs/PoseStamped on /ed305/board_pose:
  header.frame_id   which camera: "cam_N" (the pose is in that camera's optical frame)
  header.stamp      when the frame arrived here (RTSP carries no capture time)
  pose              the board in that camera: its origin corner (m) and orientation
Several cameras seeing the board at once each publish their own message. Nothing is written to disk.
PoseStamped is the same in Humble and Jazzy, so the Jetson (Humble) can subscribe to a Jazzy host.

On a camera host (streams read from localhost, no VPN in between):
  source /opt/ros/jazzy/setup.bash
  python3 board_pose_node.py --cams 3 6 8 10 12 13 15 16 --rtsp rtsp://localhost:8554   # right host
  python3 board_pose_node.py --cams 1 2 4 5 7 9 11 14 --rtsp rtsp://localhost:8554      # left host
  ros2 topic echo /ed305/board_pose
Without --rtsp the addresses come from config/cameras.yaml. Intrinsics: results/intrinsics/cam_N.json
(this folder's calibration) or --intrinsics N=FILE (any file read_intrinsics_file takes).
"""
import argparse
import threading
import time

import cv2
from scipy.spatial.transform import Rotation as Rot

from calib.common import camera_urls, load_boards, load_intrinsics, read_intrinsics_file
from calib.network import BoardPose
from calib.streams import open_streams


def pose_msg(PoseStamped, cam, T, t):
    m = PoseStamped()
    m.header.frame_id = "cam_%d" % cam
    m.header.stamp.sec, m.header.stamp.nanosec = int(t), int((t % 1) * 1e9)
    m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in T[:3, 3])
    q = Rot.from_matrix(T[:3, :3]).as_quat()
    m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = (float(v) for v in q)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cams", nargs="*", type=int, help="cameras to watch (default: every camera in config/cameras.yaml)")
    ap.add_argument("--rtsp", help="stream base address, e.g. rtsp://localhost:8554 (camera N = <base>/cam_N)")
    ap.add_argument("--board", default="E", help="board from config/boards.yaml (default E, the 4 x 4 extrinsics board)")
    ap.add_argument("--topic", default="/ed305/board_pose")
    ap.add_argument("--hz", type=float, default=10.0, help="most detections per second per camera (default 10)")
    ap.add_argument("--results", default="results", help="folder with intrinsics/cam_N.json")
    ap.add_argument("--intrinsics", action="append", default=[], metavar="N=FILE", help="camera N's intrinsics from FILE")
    ap.add_argument("--min-corners", type=int, default=8, help="fewest board corners for a pose (default 8)")
    ap.add_argument("--max-reproj", type=float, default=2.0, help="largest reprojection rms in px (default 2)")
    args = ap.parse_args(argv)

    import rclpy
    from geometry_msgs.msg import PoseStamped

    board = load_boards([args.board])[args.board].board
    intr = load_intrinsics(args.results)
    for spec in args.intrinsics:
        n, path = spec.split("=", 1)
        K, dist, _, model = read_intrinsics_file(path)
        if model == "fisheye":
            raise SystemExit("cam %s: fisheye intrinsics are not supported here" % n)
        intr[int(n)] = (K, dist)
    cams = args.cams or sorted(camera_urls())
    for c in [c for c in cams if c not in intr]:
        print("cam %d: no intrinsics, skipped" % c)
    cams = [c for c in cams if c in intr]
    if not cams:
        raise SystemExit("no camera with intrinsics (results/intrinsics/cam_N.json or --intrinsics N=FILE)")
    urls = {c: "%s/cam_%d" % (args.rtsp.rstrip("/"), c) for c in cams} if args.rtsp else camera_urls(cams)
    streams = open_streams(urls)

    rclpy.init()
    node = rclpy.create_node("ed305_board_pose")
    pub = node.create_publisher(PoseStamped, args.topic, 10)
    lock = threading.Lock()
    count = {c: 0 for c in cams}

    def watch(cam):
        detect, (K, D), last_n = BoardPose(board, args.min_corners), intr[cam], -1
        while rclpy.ok():
            t0, s = time.time(), streams[cam]
            if s.n != last_n:                          # only frames not looked at yet
                last_n = s.n
                frame, t = s.latest()
                if frame is not None:
                    T, reproj, _ = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), K, D)
                    if T is not None and reproj <= args.max_reproj:
                        with lock:
                            pub.publish(pose_msg(PoseStamped, cam, T, t))
                            count[cam] += 1
            time.sleep(max(0.0, 1.0 / args.hz - (time.time() - t0)))

    for c in cams:
        threading.Thread(target=watch, args=(c,), daemon=True).start()
    print("watching %s for board %s -> %s" % (" ".join("cam_%d" % c for c in cams), args.board, args.topic))
    try:
        while rclpy.ok():
            time.sleep(5)
            with lock:
                seen = ", ".join("cam_%d %d" % (c, n) for c, n in count.items() if n) or "none"
                for c in count:
                    count[c] = 0
            offline = [c for c in cams if not streams[c].ok]
            print("poses in the last 5 s: %s%s" % (seen, "; offline: %s" % offline if offline else ""))
    except KeyboardInterrupt:
        pass
    finally:
        for s in streams.values():
            s.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

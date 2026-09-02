#!/usr/bin/env python3
"""
run_rtabmap.py - RTAB-Map on a bag, pose out, one script.

  map       play the MAPPING bag through rtabmap (RGB-D + IMU) and save the
            database (the visual map).
  localize  play a COOP bag through rtabmap in localisation mode on that
            database and write the camera pose as TUM.

    python3 run_rtabmap.py map      --platform mobile_1 --bag <mapping bag> --db rtab_mobile_1.db
    python3 run_rtabmap.py localize --platform mobile_1 --bag <coop bag>    --db rtab_mobile_1.db --out rtab_out

Outputs (localize):
    <out>/rtabmap_<platform>.tum        pose of the platform's base frame
                                        (zed_camera_link / camera_link) in
                                        RTAB-Map's map frame, TUM format
    <out>/rtabmap_<platform>_odom.tum   the same before map corrections
    <out>/rtabmap_<platform>.log        rtabmap's own output

Feed the .tum to stage 08 as "odom_file" (child = the base frame); the stage
anchors it on the session anchor and compares it to the lidar as usual.

Needs: ROS 2 with ros-<distro>-rtabmap-ros installed, rclpy, tf2_ros, and the
bag readable by `ros2 bag play`. The bag's own /tf is NOT played (it carries
the broken tracker); only /tf_static, images, depth, camera_info and IMU.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
import threading

PLATFORMS = {
    "mobile_1": dict(
        frame_id="zed_camera_link",
        rgb="/mobile_1/zed/left/image_rect_color",
        depth="/mobile_1/zed/depth/depth_registered",
        info="/mobile_1/zed/left/camera_info",
        imu="/mobile_1/zed/imu/data",
        extra=[],
        # stage-08 extrinsic of this base frame -> optical, for the README
        note="cam_extrinsic_xyzquat [-0.010, 0.060, 0.015, -0.5, 0.5, -0.5, 0.5]"),
    "mobile_2": dict(
        frame_id="camera_link",
        rgb="/mobile_2/infra1/image_rect_raw",          # same frame as the depth
        depth="/mobile_2/depth/image_rect_raw",
        info="/mobile_2/infra1/camera_info",
        imu="/mobile_2/imu",
        extra=[],
        note="cam_extrinsic_xyzquat derived from the depth extrinsic (child = camera_link)"),
}


def rtabmap_cmd(p, args, localize):
    """ros2 launch rtabmap_launch rtabmap.launch.py with the platform topics."""
    params = [
        "--Odom/Strategy", "0",             # frame-to-map F2M visual odometry
        "--Vis/CorType", "0",
        "--Odom/ResetCountdown", "1",       # re-init instead of stopping when lost
        "--Vis/MinInliers", "12",
        "--RGBD/ProximityBySpace", "true",
        "--RGBD/OptimizeMaxError", "3.0",
        "--Reg/Force3DoF", "false",
        "--Mem/STMSize", "30",
    ]
    if localize:
        params += ["--Mem/IncrementalMemory", "false", "--Mem/InitWMWithAllNodes", "true"]
    else:
        params += ["--delete_db_on_start"]
    if args.markers:
        # ArUco markers of the ChArUco boards as landmarks (one dictionary):
        # 0 = DICT_4X4_50 (anchor boards, 15 mm markers)
        params += ["--RGBD/MarkerDetection", "true", "--Marker/Dictionary",
                   str(args.marker_dict), "--Marker/Length", str(args.marker_len),
                   "--Marker/MaxRange", "2.0"]
    cmd = ["ros2", "launch", "rtabmap_launch", "rtabmap.launch.py",
           "rtabmap_viz:=false", "rviz:=false",
           "use_sim_time:=true",
           "qos:=1",                                    # bag topics are reliable
           "frame_id:=%s" % p["frame_id"],
           "rgb_topic:=%s" % p["rgb"],
           "depth_topic:=%s" % p["depth"],
           "camera_info_topic:=%s" % p["info"],
           "approx_sync:=true", "approx_sync_max_interval:=0.04",
           "imu_topic:=%s" % p["imu"],
           "wait_imu_to_init:=true",
           "database_path:=%s" % os.path.abspath(args.db),
           "localization:=%s" % ("true" if localize else "false"),
           "rtabmap_args:=%s" % " ".join(params)]
    return cmd


def bag_cmd(p, args):
    topics = [p["rgb"], p["depth"], p["info"], p["imu"], "/tf_static"] + p["extra"]
    return ["ros2", "bag", "play", args.bag, "--clock", "--rate", str(args.rate),
            "--topics"] + topics


class Recorder:
    """Records T_map_base at every /rtabmap/odom stamp: map->odom from TF
    (rtabmap's correction, slowly varying) composed with the odometry."""
    def __init__(self, frame_id):
        import rclpy
        from rclpy.node import Node
        from nav_msgs.msg import Odometry
        import tf2_ros
        self.rclpy = rclpy
        rclpy.init()
        self.node = Node("rtabmap_pose_recorder")
        self.node.set_parameters([rclpy.parameter.Parameter(
            "use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True)])
        self.buf = tf2_ros.Buffer(); self.tl = tf2_ros.TransformListener(self.buf, self.node)
        self.frame_id = frame_id
        self.rows, self.rows_odom, self.n_corr = [], [], 0
        self.node.create_subscription(Odometry, "/rtabmap/odom", self.on_odom, 50)
        self.spin = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin.start()

    @staticmethod
    def _q(o):
        return [o.x, o.y, o.z, o.w]

    def on_odom(self, m):
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        p, o = m.pose.pose.position, m.pose.pose.orientation
        if not (p.x == p.x):              # NaN = odometry lost
            return
        T_ob = np.eye(4); T_ob[:3, :3] = R.from_quat(self._q(o)).as_matrix()
        T_ob[:3, 3] = [p.x, p.y, p.z]
        self.rows_odom.append((t, T_ob))
        try:
            from rclpy.time import Time
            tf = self.buf.lookup_transform("map", m.header.frame_id, Time())
            tr, rq = tf.transform.translation, tf.transform.rotation
            T_mo = np.eye(4); T_mo[:3, :3] = R.from_quat(self._q(rq)).as_matrix()
            T_mo[:3, 3] = [tr.x, tr.y, tr.z]
        except Exception:
            T_mo = np.eye(4)
        self.rows.append((t, T_mo @ T_ob))

    def write(self, path, rows):
        from scipy.spatial.transform import Rotation as R
        with open(path, "w") as f:
            for t, T in rows:
                q = R.from_matrix(T[:3, :3]).as_quat()
                f.write("%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n"
                        % (t, T[0, 3], T[1, 3], T[2, 3], q[0], q[1], q[2], q[3]))
        print("wrote %s (%d poses)" % (path, len(rows)))

    def close(self):
        self.node.destroy_node(); self.rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["map", "localize"])
    ap.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
    ap.add_argument("--bag", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default="rtab_out")
    ap.add_argument("--rate", type=float, default=0.5,
                    help="bag playback rate; lower if rtabmap drops frames")
    ap.add_argument("--markers", action="store_true",
                    help="detect the boards' ArUco markers as landmarks")
    ap.add_argument("--marker_dict", type=int, default=0)
    ap.add_argument("--marker_len", type=float, default=0.015)
    ap.add_argument("--settle", type=float, default=6.0,
                    help="seconds to let rtabmap start before playing")
    args = ap.parse_args()
    p = PLATFORMS[args.platform]
    os.makedirs(args.out, exist_ok=True)
    localize = args.mode == "localize"
    if localize and not os.path.exists(args.db):
        sys.exit("database %s not found - run 'map' on the mapping bag first" % args.db)

    log = open(os.path.join(args.out, "rtabmap_%s.log" % args.platform), "w")
    print("starting rtabmap (%s) ..." % args.mode)
    rt = subprocess.Popen(rtabmap_cmd(p, args, localize), stdout=log, stderr=subprocess.STDOUT,
                          preexec_fn=os.setsid)
    time.sleep(args.settle)
    rec = Recorder(p["frame_id"]) if localize else None
    print("playing %s at rate %.2f ..." % (args.bag, args.rate))
    bp = subprocess.run(bag_cmd(p, args))
    time.sleep(3.0)
    os.killpg(os.getpgid(rt.pid), signal.SIGINT)     # rtabmap saves the db on SIGINT
    try:
        rt.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(rt.pid), signal.SIGKILL)
    log.close()
    if rec is not None:
        rec.write(os.path.join(args.out, "rtabmap_%s.tum" % args.platform), rec.rows)
        rec.write(os.path.join(args.out, "rtabmap_%s_odom.tum" % args.platform), rec.rows_odom)
        rec.close()
        print("child frame of the poses: %s (%s)" % (p["frame_id"], p["note"]))
    else:
        print("database saved: %s" % args.db)
    if bp.returncode:
        print("bag play exited with %d" % bp.returncode)


if __name__ == "__main__":
    main()

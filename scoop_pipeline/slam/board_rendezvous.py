#!/usr/bin/env python3
"""The anchor boards as one inter-robot measurement for Swarm-SLAM: each robot
knows where its odom frame lies in map from the board it started at
(datasets/swarm_slam_lidar.py's board_start.json), so its first keyframe's
pose in map is known as soon as cslam makes it. One node per robot, in the
robot's ROS domain:

  - on its robot's keyframe 0 (/r<i>/cslam/keyframe_odom) it publishes that
    keyframe's pose in map on /cslam/board_start (geometry_msgs/PoseStamped,
    frame_id r<i>; reliable, transient local), which the channel relay
    carries to the other robots like any cslam message;
  - for every other robot j > i whose /cslam/board_start it receives, it
    publishes the measurement between the two first keyframes, inv(P_i) P_j,
    on /cslam/inter_robot_loop_closure (cslam's InterRobotLoopClosure, robot
    i's keyframe 0 to robot j's), as cslam's front-end does a loop closure:
    robot i's back-end takes it here, robot j's through the relay.

Swarm-SLAM then connects the robots from their first keyframes rather than
from their first loop closure; GNC judges the measurement with the loop
closures (it is no known inlier).

    python3 board_rendezvous.py --robot-id 0 --board-start <export>/board_start.json
"""
import argparse
import json

import numpy as np


def quat_of(R):
    """Rotation matrix -> (x, y, z, w), by its largest component (Shepperd)."""
    t = np.trace(R)
    if t > max(R[0, 0], R[1, 1], R[2, 2]):
        w = np.sqrt(1.0 + t) / 2
        q = [(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w]
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        v = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) / 2
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / (4 * v)]
        q[i], q[j], q[k] = v, (R[j, i] + R[i, j]) / (4 * v), (R[k, i] + R[i, k]) / (4 * v)
    q = np.asarray(q)
    return q / np.linalg.norm(q) * (1 if q[3] >= 0 else -1)


def T_of(p, q):
    """(x, y, z), (x, y, z, w) -> 4x4."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = p
    return T


def pose_T(pose):
    """geometry_msgs/Pose -> 4x4."""
    p, o = pose.position, pose.orientation
    return T_of([p.x, p.y, p.z], [o.x, o.y, o.z, o.w])


def set_pose(pose, T):
    q = quat_of(T[:3, :3])
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in T[:3, 3])
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (float(v) for v in q)


def set_transform(tf, T):
    q = quat_of(T[:3, :3])
    tf.translation.x, tf.translation.y, tf.translation.z = (float(v) for v in T[:3, 3])
    tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w = (float(v) for v in q)


def measurement(P_i, P_j):
    """cslam's between factor from robot i's keyframe to robot j's: inv(P_i) P_j."""
    return np.linalg.inv(P_i) @ P_j


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-id", type=int, required=True)
    ap.add_argument("--board-start", required=True, help="the export's board_start.json")
    a = ap.parse_args()
    rec = {r["robot_id"]: r for r in json.load(open(a.board_start))["robots"]}
    if a.robot_id not in rec:
        raise SystemExit("%s has no robot %d" % (a.board_start, a.robot_id))
    T_map_odom = np.asarray(rec[a.robot_id]["T_map_odom"], float)

    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import PoseStamped
    from cslam_common_interfaces.msg import KeyframeOdom, InterRobotLoopClosure

    rclpy.init()
    i = a.robot_id
    node = rclpy.create_node("board_rendezvous_r%d" % i)
    latched = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub_start = node.create_publisher(PoseStamped, "/cslam/board_start", latched)
    pub_lc = node.create_publisher(InterRobotLoopClosure, "/cslam/inter_robot_loop_closure", 100)
    st = {"own": None, "others": {}, "sent": set()}

    def send():
        if st["own"] is None or pub_lc.get_subscription_count() == 0:
            return
        for j, P_j in sorted(st["others"].items()):
            if j <= i or j in st["sent"]:
                continue
            m = InterRobotLoopClosure()
            m.robot0_id, m.robot0_keyframe_id, m.robot1_id, m.robot1_keyframe_id = i, 0, j, 0
            m.success = True
            M = measurement(st["own"], P_j)
            set_transform(m.transform, M)
            pub_lc.publish(m)
            st["sent"].add(j)
            node.get_logger().info("board measurement r%d kf 0 -> r%d kf 0: %.3f %.3f %.3f m, yaw %.2f deg"
                                   % (i, j, *M[:3, 3], np.degrees(np.arctan2(M[1, 0], M[0, 0]))))

    def on_keyframe(msg):
        if msg.id != 0 or st["own"] is not None:
            return
        st["own"] = T_map_odom @ pose_T(msg.odom.pose.pose)
        out = PoseStamped()
        out.header.stamp = msg.odom.header.stamp
        out.header.frame_id = "r%d" % i
        set_pose(out.pose, st["own"])
        pub_start.publish(out)
        node.get_logger().info("r%d keyframe 0 in map by the board: %.3f %.3f %.3f" % (i, *st["own"][:3, 3]))
        send()

    def on_start(msg):
        j = int(msg.header.frame_id.lstrip("r"))
        if j != i and j not in st["others"]:
            st["others"][j] = pose_T(msg.pose)
            send()

    node.create_subscription(KeyframeOdom, "/r%d/cslam/keyframe_odom" % i, on_keyframe, 100)
    node.create_subscription(PoseStamped, "/cslam/board_start", on_start, latched)
    node.create_timer(1.0, send)                     # until cslam's back-end has subscribed
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

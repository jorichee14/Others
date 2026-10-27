#!/usr/bin/env python3
"""What a method was given and what it gave back, message by message: a ROS 2
node that subscribes to the method's input and output topics as raw bytes
(nothing deserialized but the header stamp), and logs every message.

    python3 probe.py --dir OUT --in /mobile_1/ouster/points=sensor_msgs/msg/PointCloud2 \\
        --in /mobile_1/ouster/imu=sensor_msgs/msg/Imu \\
        --out /Odometry=nav_msgs/msg/Odometry --tum /Odometry

Writes into OUT:
  messages.csv     wall_ns, role (in/out), topic, stamp_ns, bytes, src_ns: one
                   row per message; wall_ns when it arrived here (time.time_ns),
                   src_ns when its publisher sent it (the middleware's source
                   timestamp, the same clock): latency is timed between
                   publishers, so this probe's own delivery does not enter it
  trajectory.tum   the --tum topic's poses (Odometry, PoseStamped,
                   PoseWithCovarianceStamped or TransformStamped),
                   't x y z qx qy qz qw' on the message stamps
  probe.json       the frames of the --tum topic's first message, QoS used

Each topic is subscribed once a publisher for it exists, with that publisher's
reliability (a reliable subscription would not match a best-effort publisher;
a best-effort one would drop large inputs that were sent reliably). Stops on
SIGINT / SIGTERM, flushing what it has.
"""
import argparse
import json
import os
import signal
import struct
import sys
import time


def header_stamp_ns(raw):
    """The header stamp of a CDR-serialized message that starts with a
    std_msgs/Header (sec int32, nanosec uint32 right after the 4-byte
    encapsulation), or -1."""
    if len(raw) < 12:
        return -1
    end = "<" if raw[1] == 1 else ">"
    sec, nsec = struct.unpack_from(end + "iI", raw, 4)
    return sec * 1_000_000_000 + nsec


def pose_of(msg):
    """(x y z qx qy qz qw) of an Odometry / PoseWithCovarianceStamped / PoseStamped /
    TransformStamped."""
    if hasattr(msg, "transform"):
        t, q = msg.transform.translation, msg.transform.rotation
        return (t.x, t.y, t.z, q.x, q.y, q.z, q.w)
    p = msg.pose
    p = getattr(p, "pose", p)
    return (p.position.x, p.position.y, p.position.z,
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--in", dest="ins", action="append", default=[], metavar="TOPIC=TYPE")
    ap.add_argument("--out", dest="outs", action="append", default=[], metavar="TOPIC=TYPE")
    ap.add_argument("--tum", default=None, help="an --out topic whose poses go to trajectory.tum")
    ap.add_argument("--flush-s", type=float, default=1.0)
    a, ros_args = ap.parse_known_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from rclpy.executors import SingleThreadedExecutor, await_or_execute
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    from rclpy.subscription import Subscription

    class InfoExecutor(SingleThreadedExecutor):
        """Hands a subscription callback the message info as well (Humble's
        executor takes it from the middleware and drops it; from Jazzy on, a
        callback taking two arguments gets it anyway: Subscription.CallbackType)."""

        def _take_subscription(self, sub):
            with sub.handle:
                return sub.handle.take_message(sub.msg_type, sub.raw)

        async def _execute_subscription(self, sub, msg_info):
            if msg_info:
                await await_or_execute(sub.callback, msg_info[0], msg_info[1])

    topics = {}
    for role, specs in (("in", a.ins), ("out", a.outs)):
        for s in specs:
            t, _, ty = s.partition("=")
            if not ty:
                sys.exit(f"probe: {s}: want TOPIC=TYPE")
            topics[t] = (role, ty)
    if a.tum and a.tum not in topics:
        sys.exit(f"probe: --tum {a.tum} is not one of the --out topics")
    os.makedirs(a.dir, exist_ok=True)
    rows, tum, meta = [], [], {"qos": {}, "tum_topic": a.tum}
    fh = open(os.path.join(a.dir, "messages.csv"), "w")
    fh.write("wall_ns,role,topic,stamp_ns,bytes,src_ns\n")
    tum_fh = open(os.path.join(a.dir, "trajectory.tum"), "w") if a.tum else None

    rclpy.init(args=ros_args)
    node = Node("scoop_probe")
    subs = {}

    def make_cb(topic, role):
        tum_type = get_message(topics[topic][1]) if topic == a.tum else None

        def cb(raw, info):
            now = time.time_ns()
            rows.append("%d,%s,%s,%d,%d,%d\n" % (now, role, topic, header_stamp_ns(raw), len(raw),
                                                info.get("source_timestamp", -1)))
            if tum_type is not None:
                m = deserialize_message(raw, tum_type)
                st = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
                tum.append("%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n" % ((st * 1e-9,) + pose_of(m)))
                if "frame_id" not in meta:
                    meta["frame_id"] = m.header.frame_id
                    meta["child_frame_id"] = getattr(m, "child_frame_id", None)
        return cb

    def subscribe():
        for t, (role, ty) in topics.items():
            if t in subs:
                continue
            pubs = node.get_publishers_info_by_topic(t)
            if not pubs:
                continue
            rel = pubs[0].qos_profile.reliability
            qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=200, reliability=rel,
                             durability=DurabilityPolicy.VOLATILE)
            subs[t] = node.create_subscription(get_message(ty), t, make_cb(t, role), qos, raw=True)
            meta["qos"][t] = "reliable" if rel == ReliabilityPolicy.RELIABLE else "best_effort"

    def flush():
        if rows:
            fh.write("".join(rows))
            rows.clear()
            fh.flush()
        if tum and tum_fh:
            tum_fh.write("".join(tum))
            tum.clear()
            tum_fh.flush()

    node.create_timer(0.1, subscribe)
    node.create_timer(a.flush_s, flush)
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    ex = SingleThreadedExecutor() if hasattr(Subscription, "CallbackType") else InfoExecutor()
    ex.add_node(node)
    try:
        while not stop and rclpy.ok():
            ex.spin_once(timeout_sec=0.1)
    finally:
        flush()
        fh.close()
        if tum_fh:
            tum_fh.close()
        meta["subscribed"] = sorted(subs)
        meta["never_published"] = sorted(set(topics) - set(subs))
        json.dump(meta, open(os.path.join(a.dir, "probe.json"), "w"), indent=1)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

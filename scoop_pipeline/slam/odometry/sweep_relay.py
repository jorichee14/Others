#!/usr/bin/env python3
"""Hand the method each LiDAR cloud when a live robot would have it: at the end
of its sweep, not its start.

    python3 sweep_relay.py --in /replay/mobile_1/ouster/points --out /mobile_1/ouster/points \\
        --hold-s 0.1 [--rate 1.0]

The pass bags are retimed (scoop/retime.py): a cloud is logged, so replayed,
at its header stamp, the sweep's first point. A live driver publishes it once
the sweep is complete, a sweep later. Played as recorded, a LiDAR-inertial
method then waits a sweep for the IMU to cover the cloud, and that wait is
counted as its latency (RKO-LIO, FAST-LIO2: ~110-120 ms against KISS-ICP's 9).
This relay takes the clouds from the player (ros2 bag play --remap'd to --in)
and publishes each, bytes untouched, --hold-s / --rate later on --out, the
topic the method and the probe listen to. Reliable, like the player.
"""
import argparse
import collections
import signal
import threading
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", default="sensor_msgs/msg/PointCloud2")
    ap.add_argument("--hold-s", type=float, required=True, help="the sweep's duration (s, bag time)")
    ap.add_argument("--rate", type=float, default=1.0, help="the player's --rate")
    a, ros_args = ap.parse_known_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from rosidl_runtime_py.utilities import get_message

    rclpy.init(args=ros_args)
    node = Node("scoop_sweep_relay")
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=20, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.VOLATILE)
    ty = get_message(a.type)
    pub = node.create_publisher(ty, a.out, qos)
    hold = a.hold_s / a.rate
    queue, cv, stop = collections.deque(), threading.Condition(), []

    def cb(raw):
        with cv:
            queue.append((time.monotonic() + hold, raw))
            cv.notify()

    def sender():
        while not stop:
            with cv:
                while not queue and not stop:
                    cv.wait(0.1)
                if stop:
                    return
                due, raw = queue[0]
                wait = due - time.monotonic()
                if wait > 0:
                    cv.wait(wait)
                    continue
                queue.popleft()
            pub.publish(raw)

    node.create_subscription(ty, a.src, cb, qos, raw=True)
    t = threading.Thread(target=sender, daemon=True)
    t.start()
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    try:
        while not stop and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        stop.append(1)
        with cv:
            cv.notify()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

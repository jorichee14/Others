#!/usr/bin/env python3
"""Robot-to-robot ROS 2 messages through a channel (scoop/channel.py): each
robot in its own ROS domain, this relay carries what one robot publishes
for the others, holding each message until the channel says it arrives, or
dropping it.

    python3 scoop_pipeline/slam/channel_relay.py --channel logged:<pass> \\
        --robots mobile_1 mobile_2 [--domains 0 1] [--topics '^/(r[0-9]+/)?cslam/(?!viz/)'] \\
        [--log relay.csv] [--seed 0]

Robot i is in ROS domain --domains[i] (default i) and is --robots[i] for the
channel. A topic matching --topics is carried from domain A to domain B when
A has a publisher on it (other than the relay) and B has a subscriber (other
than the relay), so only what the robots really exchange is carried
(Swarm-SLAM: /cslam/local_descriptors, /cslam/inter_robot_loop_closure,
/cslam/pose_graph, /r<j>/cslam/heartbeat, ...). Messages go as serialized
bytes: their size is what the channel charges.

Time is the bag's (/clock in the first robot's domain: play the data with
--clock), the same clock as the link traces, so a slow playback slows the
link with it. Until /clock arrives, messages go at once (logged as such).

Every message is logged (--log, CSV): t_send, topic, src, dst, bytes, lost,
measured (the logged channel had a measurement at t_send), t_arrive,
t_published.
"""
import argparse
import csv
import hashlib
import heapq
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from scoop import channel as CH                                     # noqa: E402

TOPICS = r"^/(r[0-9]+/)?cslam/(?!viz/)"
NODE = "scoop_channel_relay"


class Relay:
    """The scheduling, without ROS: messages in, (due time, domain, topic,
    bytes) out, every message logged."""

    def __init__(self, ch, robots, log_path=None):
        self.ch, self.robots = ch, list(robots)
        self.heap, self.seq = [], 0
        self.rows = []
        self.log = None
        if log_path:
            self.log = open(log_path, "w", newline="")
            self.w = csv.writer(self.log)
            self.w.writerow(["t_send", "topic", "src", "dst", "bytes", "lost", "measured",
                             "t_arrive", "t_published"])

    def send(self, t_now, src, topic, data, dsts):
        """A message robot index src published at t_now, for robot indices dsts."""
        for dst in dsts:
            if dst == src:
                continue
            if t_now is None:
                d = CH.Delivery(None, False, False, 0.0)
                t_arr = None
            else:
                d = self.ch.deliver(self.robots[src], self.robots[dst], t_now, len(data))
                t_arr = d.t_arrive
            row = [t_now, topic, self.robots[src], self.robots[dst], len(data), int(d.lost),
                   int(d.measured), t_arr, None]
            if d.lost:
                self._write(row)
                continue
            heapq.heappush(self.heap, (t_arr if t_arr is not None else -1.0, self.seq, dst, topic, data, row))
            self.seq += 1

    def due(self, t_now):
        """[(dst, topic, data)] whose arrival time has come (all of them when
        there is no clock yet)."""
        out = []
        while self.heap and (t_now is None or self.heap[0][0] <= t_now):
            _, _, dst, topic, data, row = heapq.heappop(self.heap)
            row[8] = t_now
            self._write(row)
            out.append((dst, topic, data))
        return out

    def _write(self, row):
        self.rows.append(row)
        if self.log:
            self.w.writerow(["" if v is None else ("%.6f" % v if isinstance(v, float) else v) for v in row])
            self.log.flush()

    def close(self):
        if self.log:
            self.log.close()


def ros_main(a):
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from rosidl_runtime_py.utilities import get_message
    from rosgraph_msgs.msg import Clock

    domains = a.domains or list(range(len(a.robots)))
    relay = Relay(CH.make(a.channel, seed=a.seed), a.robots, a.log)
    pat = re.compile(a.topics)
    nodes, execs = [], []
    for d in domains:
        c = Context()
        rclpy.init(context=c, domain_id=d)
        n = rclpy.create_node(NODE, context=c)
        e = SingleThreadedExecutor(context=c)
        e.add_node(n)
        nodes.append(n)
        execs.append(e)
    clock = {"t": None}
    # rosbag2 publishes /clock best effort: a reliable subscription gets nothing
    nodes[0].create_subscription(Clock, "/clock", lambda m: clock.__setitem__(
        "t", m.clock.sec + m.clock.nanosec * 1e-9),
        QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
    subs, pubs, echo, types = {}, {}, {}, {}

    def others(infos):
        return [i for i in infos if i.node_name != NODE]

    def qos_of(infos):
        if infos:
            q = infos[0].qos_profile
            return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100, reliability=q.reliability,
                              durability=q.durability)
        return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100,
                          reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)

    def receive(i, topic):
        def cb(data):
            h = hashlib.sha1(data).digest()
            k = (i, topic)
            if echo.get(k, {}).get(h):                 # what the relay itself published here
                echo[k][h] -= 1
                return
            dsts = [j for j in range(len(nodes)) if j != i and
                    others(nodes[j].get_subscriptions_info_by_topic(topic))]
            relay.send(clock["t"], i, topic, bytes(data), dsts)
        return cb

    def discover():
        for i, n in enumerate(nodes):
            for topic, tys in n.get_topic_names_and_types():
                if not pat.search(topic) or (i, topic) in subs:
                    continue
                pub_infos = others(n.get_publishers_info_by_topic(topic))
                if not pub_infos:
                    continue
                types[topic] = get_message(tys[0])
                subs[(i, topic)] = n.create_subscription(types[topic], topic, receive(i, topic),
                                                         qos_of(pub_infos), raw=True)
                n.get_logger().info("domain %d: carrying %s (%s)" % (domains[i], topic, tys[0]))

    def publish(j, topic, data):
        k = (j, topic)
        if k not in pubs:
            infos = others(nodes[j].get_subscriptions_info_by_topic(topic))
            pubs[k] = nodes[j].create_publisher(types[topic], topic, qos_of(infos))
        h = hashlib.sha1(data).digest()
        echo.setdefault(k, {})[h] = echo.get(k, {}).get(h, 0) + 1
        pubs[k].publish(data)

    nodes[0].create_timer(1.0, discover)
    started = nodes[0].get_clock().now()

    def watch_clock():
        if clock["t"] is None and (nodes[0].get_clock().now() - started).nanoseconds > 30e9:
            nodes[0].get_logger().warn("no /clock after 30 s: messages go through at once "
                                       "(play the bag with --clock in domain %d)" % domains[0])

    nodes[0].create_timer(10.0, watch_clock)
    try:
        while all(rclpy.ok(context=n.context) for n in nodes):
            for e in execs:
                e.spin_once(timeout_sec=0.001)
            for j, topic, data in relay.due(clock["t"]):
                publish(j, topic, data)
    except KeyboardInterrupt:
        pass
    finally:
        relay.close()
        n_lost = sum(1 for r in relay.rows if r[5])
        print("relay: %d messages carried, %d lost, %d bytes" % (
            len(relay.rows) - n_lost, n_lost, sum(r[4] for r in relay.rows)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", required=True, help="scoop/channel.py spec, e.g. logged:<pass>")
    ap.add_argument("--robots", nargs="+", required=True, help="machine names, robot 0 first")
    ap.add_argument("--domains", nargs="+", type=int, default=None)
    ap.add_argument("--topics", default=TOPICS)
    ap.add_argument("--log", default="relay.csv")
    ap.add_argument("--seed", type=int, default=0)
    ros_main(ap.parse_args())


if __name__ == "__main__":
    main()

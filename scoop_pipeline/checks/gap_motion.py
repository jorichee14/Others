#!/usr/bin/env python3
"""Did GLIM bridge the LiDAR's outages? Its motion across each gap in its
trajectory against an odometry that kept running (the robot's live VSLAM).

    python3 check.py gaps <traj_lidar.txt> [--bag <bag with the odometry>]
        [--odom /mobile_2/visual_slam/tracking/odometry] [--min-gap 0.5]

For each gap of more than --min-gap s between GLIM's poses: how far it moved
and turned across it, and the same from the odometry between the same two
times (distance and angle do not depend on either's frame). A gap where they
disagree by more than 0.10 m or 3 deg is marked: GLIM lost track there.
Distance and angle only: a wrong step of the right length is not seen (look
at the trajectory there when a gap's GLIM speed is far from the speed near it).
Without --bag, GLIM's motion alone (a jump shows as a speed far above the
speed around it).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rot(q):
    """xyzw -> 3x3."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def angle(qa, qb):
    """deg between two orientations."""
    R = rot(qa).T @ rot(qb)
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def motion(t, p, q, ta, tb):
    """(distance m, angle deg) between the poses nearest ta and tb, or None
    when either is more than 0.1 s away."""
    i, j = (int(np.argmin(np.abs(t - x))) for x in (ta, tb))
    if abs(t[i] - ta) > 0.1 or abs(t[j] - tb) > 0.1:
        return None
    return float(np.linalg.norm(p[j] - p[i])), float(angle(q[i], q[j]))


def load_tum(path):
    a = np.loadtxt(path, comments="#")
    return a[:, 0], a[:, 1:4], a[:, 4:8]


def load_odom(bag, topic):
    from scoop.bag import open_bag
    t, p, q = [], [], []
    for _, stamp, m in open_bag(bag).iter_messages([topic]):
        pose = m.pose.pose if hasattr(m.pose, "pose") else m.pose
        t.append(stamp * 1e-9)
        p.append([pose.position.x, pose.position.y, pose.position.z])
        q.append([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    if not t:
        raise SystemExit("no %s messages in %s" % (topic, bag))
    o = np.argsort(t)
    return np.array(t)[o], np.array(p)[o], np.array(q)[o]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traj", help="GLIM's traj_lidar.txt (TUM: t x y z qx qy qz qw)")
    ap.add_argument("--bag", help="a bag with the odometry (the raw recording)")
    ap.add_argument("--odom", default="/mobile_2/visual_slam/tracking/odometry")
    ap.add_argument("--min-gap", type=float, default=0.5)
    a = ap.parse_args()
    t, p, q = load_tum(a.traj)
    dt = np.diff(t)
    gaps = np.flatnonzero(dt > a.min_gap)
    print("%s: %d poses over %.1f s; %d gaps > %.2f s, %.1f s in all"
          % (a.traj, len(t), t[-1] - t[0], len(gaps), a.min_gap, dt[gaps].sum()))
    if not len(gaps):
        return
    odom = load_odom(os.path.expanduser(a.bag), a.odom) if a.bag else None
    hdr = "  %8s %6s | %13s %8s" % ("at s", "gap s", "GLIM m / deg", "m/s near")
    if odom:
        hdr += " | %13s | %s" % ("odom m / deg", "diff")
    print(hdr)
    bad = 0
    for k in gaps:
        ta, tb = t[k], t[k + 1]
        d, r = motion(t, p, q, ta, tb)
        near = (t > ta - 2) & (t <= ta)
        idx = np.flatnonzero(near)
        v = (np.linalg.norm(p[idx[-1]] - p[idx[0]]) / max(t[idx[-1]] - t[idx[0]], 1e-6)
             if len(idx) > 1 else float("nan"))
        line = "  %8.1f %6.2f | %6.3f %6.2f %8.3f" % (ta - t[0], tb - ta, d, r, v)
        if odom:
            m = motion(*odom, ta, tb)
            if m is None:
                line += " | %13s |" % "no odometry"
            else:
                dd, dr = abs(d - m[0]), abs(r - m[1])
                flag = dd > 0.10 or dr > 3.0
                bad += flag
                line += " | %6.3f %6.2f | %.3f m %.2f deg%s" % (m[0], m[1], dd, dr,
                                                                 "  <- lost" if flag else "")
        print(line)
    if odom:
        print("%d of %d gaps disagree with the odometry by > 0.10 m or 3 deg" % (bad, len(gaps)))


if __name__ == "__main__":
    main()

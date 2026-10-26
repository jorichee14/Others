#!/usr/bin/env python3
"""Swarm-SLAM's pose graph optimization judges loop closures only: the prior and
the odometry edges are GNC's known inliers.

    python3 cslam_gnc_known_inliers.py [/Swarm-SLAM]      (in Swarm-SLAM's container)
    then: cd /Swarm-SLAM && . /opt/ros/$ROS_DISTRO/setup.sh && colcon build --packages-select cslam

cslam/src/back_end/decentralized_pgo.cpp optimizes with gtsam's GncOptimizer at
its default parameters: no known inliers, so it may reject any factor,
odometry included. On the 20261015 site_2 run (3 inter-robot loop closures,
all within 10 cm of the ground truth; every odometry edge within 1 cm) it
rejected robot 1's odometry edge (1,21)-(1,22) and the one loop closure that
held the rest of robot 1, which then lay 180 deg turned 19 m away: robot 1's
own ATE 9.6 m against 2 cm before optimization. With few loop closures there
is not the redundancy GNC needs to tell them from the odometry. The patch
marks as known inliers every factor that is not a Pose3 between-factor (the
prior) and every between-factor of consecutive keyframes of one robot (the
odometry), so only loop closures are judged. Run it again: it does nothing
twice.
"""
import os
import sys

MARK = "// scoop patch: GNC known inliers"
OLD = """    gtsam::GncParams<gtsam::LevenbergMarquardtParams> params;
"""
NEW = OLD + """    """ + MARK + """: the prior and the odometry (consecutive keyframes
    // of one robot) are never outliers; GNC judges the loop closures alone
    {
      gtsam::GncParams<gtsam::LevenbergMarquardtParams>::IndexVector known_inliers;
      for (size_t i = 0; i < graph->size(); ++i)
      {
        if (!graph->at(i))
        {
          continue;
        }
        auto between = boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(graph->at(i));
        if (!between)
        {
          known_inliers.push_back(i);
          continue;
        }
        gtsam::LabeledSymbol a(between->key1()), b(between->key2());
        if (a.label() == b.label() && (a.index() + 1 == b.index() || b.index() + 1 == a.index()))
        {
          known_inliers.push_back(i);
        }
      }
      params.setKnownInliers(known_inliers);
    }
"""


def patch(path):
    """'patched', 'already' or 'not found' (the code is not the one this patch knows)."""
    s = open(path).read()
    if MARK in s:
        return "already"
    if s.count(OLD) != 1:
        return "not found"
    open(path, "w").write(s.replace(OLD, NEW))
    return "patched"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/Swarm-SLAM"
    found = []
    for d, _, files in os.walk(root):
        if "decentralized_pgo.cpp" in files and d.endswith(os.path.join("src", "back_end")):
            p = os.path.join(d, "decentralized_pgo.cpp")
            found.append((p, patch(p)))
    for p, r in found:
        print("%-10s %s" % (r, p))
    if not found or any(r == "not found" for _, r in found):
        sys.exit("decentralized_pgo.cpp not found or not the code this patch knows: nothing changed there")
    print("now rebuild: cd %s && . /opt/ros/$ROS_DISTRO/setup.sh && colcon build --packages-select cslam" % root)


if __name__ == "__main__":
    main()

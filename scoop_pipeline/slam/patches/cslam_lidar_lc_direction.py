#!/usr/bin/env python3
"""Swarm-SLAM's LiDAR loop closures, put the right way round in its pose graph.

    python3 cslam_lidar_lc_direction.py [/Swarm-SLAM]      (in Swarm-SLAM's container)

cslam/lidar_pr/icp_utils.py compute_transform(src, dst) returns TEASER++'s
(refined by ICP) solution as it is: dst = R src + t, the pose of the src
keyframe in the dst keyframe. Its callers (inter-robot matches and intra-robot
loop closures, lidar_handler_node.py) send it as the measurement from the src
keyframe to the dst keyframe, which the pose graph reads as the pose of dst in
src: the inverse. Measured on the 20261015 site_1 runs (analysis/slam/
cslam_eval.py, every edge against the ground truth): odometry edges 0.9 cm /
0.1 deg as written; loop closures 0.5-0.9 m / 14-15 deg as written and
2.6-5.5 cm / 0.3-0.8 deg inverted. The patch inverts the solution in
compute_transform, in every copy of icp_utils.py under the Swarm-SLAM folder
(the source and the installed package). Run it again: it does nothing twice.
"""
import os
import sys

MARK = "# scoop patch: loop-closure direction"
OLD = """    valid, translation, rotation = solve_teaser(src, dst, voxel_size,
                                                min_inliers)
"""
NEW = OLD + """    """ + MARK + """: the solution maps src into dst (the pose
    # of src in dst); the measurement src -> dst is the pose of dst in src
    rotation = np.asarray(rotation).T
    translation = -rotation @ np.asarray(translation)
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
        if "icp_utils.py" in files and d.endswith(os.path.join("cslam", "lidar_pr")):
            p = os.path.join(d, "icp_utils.py")
            found.append((p, patch(p)))
    for p, r in found:
        print("%-10s %s" % (r, p))
    if not found or any(r == "not found" for _, r in found):
        sys.exit("icp_utils.py not found or not the code this patch knows: nothing changed there")


if __name__ == "__main__":
    main()

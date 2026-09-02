#!/usr/bin/env bash
# Board + session-anchor frames from anchor_frame.json (mapping-session survey),
# plus where mobile_1's LIDAR track says rs_anchor actually was during the coop
# session. Run alongside the anchored map cloud in RViz (fixed frame: map).
#
#   ros2 run pcl_ros pcd_to_pointcloud --ros-args \
#     -p file_name:=map_final_20260828_nc_anchored.pcd -p frame_id:=map
#
# WHAT TO LOOK FOR: a board frame should sit ON a wall surface in the cloud,
# flush with it, at roughly chest height. board_rs (surveyed) and
# board_rs_SEEN are 9.20 m apart - exactly one of them can be on a real wall.
set -e

# ---------- surveyed boards (mapping session, 2026-09-01) ----------
ros2 run tf2_ros static_transform_publisher \
  --x 0.000000 --y 0.000000 --z 0.000000 \
  --qx -0.005077 --qy 0.002281 --qz 0.000012 --qw 0.999985 \
  --frame-id map --child-frame-id board &

ros2 run tf2_ros static_transform_publisher \
  --x 8.033500 --y -7.181630 --z -0.101557 \
  --qx -0.059071 --qy -0.024090 --qz -0.857719 --qw 0.510146 \
  --frame-id map --child-frame-id board_anchor_b &

ros2 run tf2_ros static_transform_publisher \
  --x 8.460388 --y -14.266981 --z -0.055759 \
  --qx 0.007510 --qy 0.013642 --qz 0.999879 --qw 0.000177 \
  --frame-id map --child-frame-id board_rs &

# ---------- where the LIDAR track saw the 5x5 board in the coop bag ----------
# 258 sightings, 2 cm scatter, 9.20 m from the surveyed pose above.
# Rotation is a placeholder (identity) - position is the point at issue.
# The full re-surveyed pose lands in anchor_frame_resurveyed.json.
ros2 run tf2_ros static_transform_publisher \
  --x -0.580000 --y -12.570000 --z 0.040000 \
  --qx 0.0 --qy 0.0 --qz 0.0 --qw 1.0 \
  --frame-id map --child-frame-id board_rs_SEEN &

# ---------- session-start poses from stage 06 ----------
ros2 run tf2_ros static_transform_publisher \
  --x 0.697952 --y -0.062696 --z 0.193334 \
  --qx -0.477626 --qy -0.496853 --qz 0.524467 --qw 0.499946 \
  --frame-id map --child-frame-id zed_pose_start &

# NOTE: this one assumes the STALE board pose, so if rs_anchor moved this is
# where mobile_2 is wrongly believed to have started.
ros2 run tf2_ros static_transform_publisher \
  --x 7.704074 --y -14.308017 --z 0.165040 \
  --qx -0.499221 --qy 0.490747 --qz -0.510928 --qw 0.498898 \
  --frame-id map --child-frame-id rs_pose_start_STALE &

wait

#!/bin/bash
# KISS-ICP, the geometry-only backbone. Two jobs: on mobile_1.ouster it is the
# LiDAR row; on a depth stream it is stage 1's first consensus backbone, fed by
# the entrypoint's depth->cloud bridge.
set -euo pipefail
source /opt/ros/humble/setup.bash
source /opt/kiss_ws/install/setup.bash
: "${SLAM_CLOUD_TOPIC:?no cloud topic: either the stream carries one or SLAM_NEEDS_CLOUD=1}"

# THE LAUNCH FILE IS NOT USED, ON PURPOSE. kiss-icp's odometry.launch.py hard-codes
# max_range/min_range/voxel_size and says in a comment that they are "not exposed
# through the launch system". Launching through it would run the 100 m automotive
# defaults while configs/methods/kiss_icp.yaml claimed the indoor ones -- a run
# tuned differently from its own manifest, which is rule 7's exact failure.
OVERRIDE=()
DROP=()
if [[ "${SLAM_MODALITY:-}" == "rgbd" ]]; then
    # Structural, not tuning: a depth image is one exposure with no per-point
    # timestamps, so there is nothing to deskew against. Said out loud because a
    # silent override is a manifest that lies.
    echo "OVERRIDE deskew -> false (depth stream: no per-point time field)"
    DROP=(deskew)
    OVERRIDE=(-p deskew:=false)
fi
mapfile -t PARAMS < <(python3 -c '
import json, os, sys
sys.path.insert(0, "/opt/slambench")
from params_to_args import to_ros_params
for a in to_ros_params(json.loads(os.environ.get("SLAM_PARAMS") or "{}"),
                       drop=tuple(sys.argv[1:])):
    print(a)' ${DROP[@]+"${DROP[@]}"})

set -x
exec ros2 run kiss_icp kiss_icp_node --ros-args \
    -r pointcloud_topic:="$SLAM_CLOUD_TOPIC" \
    -p use_sim_time:=true \
    -p base_frame:="" \
    -p publish_odom_tf:=false \
    -p publish_debug_clouds:=true \
    "${PARAMS[@]}" ${OVERRIDE[@]+"${OVERRIDE[@]}"}

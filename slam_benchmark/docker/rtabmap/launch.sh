#!/bin/bash
# RTAB-Map RGB-D, with or without the IMU. Stage 1's feature+depth(+inertial)
# backbone, and the only tier-3 entry that also produces a dense map.
set -euo pipefail
source /opt/ros/humble/setup.bash
: "${SLAM_DEPTH_TOPIC:?}" "${SLAM_COLOR_TOPIC:?}" "${SLAM_DEPTH_INFO_TOPIC:?}"

# WHICH FRAME THE POSES COME OUT IN -- read off the bag, never assumed.
# RTAB-Map's frame_id is the body frame it expresses odometry in. Setting it to
# the DEPTH IMAGE'S OWN optical frame makes that transform the identity, so the
# container emits the sensor frame it was given and the evaluator applies
# reference_frame_from_sensor exactly once (container contract rule 1). Setting
# it to a base_link instead would emit body poses that look fine and are wrong by
# the camera's lever arm.
FRAME_ID=$(python3 /opt/slambench/sniff_frame.py --bag /bag --topic "$SLAM_DEPTH_TOPIC")
echo "frame_id : $FRAME_ID  (from the first $SLAM_DEPTH_TOPIC header)"

# Slash keys are RTAB-Map parameters. They go to BOTH nodes: the odometry node
# owns Vis/* and Reg/*, the mapping node owns RGBD/*, Grid/* and Mem/*, and each
# warns about the other's. One warning stream is cheaper than two params blocks
# that can disagree.
RTAB_ARGS=$(python3 -c '
import json, os, sys
sys.path.insert(0, "/opt/slambench")
from params_to_args import to_args
print(" ".join(to_args(json.loads(os.environ.get("SLAM_PARAMS") or "{}"))))')
echo "rtabmap  : $RTAB_ARGS"

IMU_ARGS=()
if [[ -n "${SLAM_IMU_TOPIC:-}" ]]; then
    # always_check_imu_tf stays at its default: if the bag's tf_static does not
    # reach the IMU from $FRAME_ID, RTAB-Map must FAIL rather than fuse an
    # unrotated gravity vector, which would tilt the whole map by the mounting
    # angle and still complete.
    IMU_ARGS=(imu_topic:="$SLAM_IMU_TOPIC" wait_imu_to_init:=true)
    echo "imu      : $SLAM_IMU_TOPIC"
else
    echo "imu      : none (the IMU-free half of the pair)"
fi

set -x
exec ros2 launch rtabmap_launch rtabmap.launch.py \
    use_sim_time:=true \
    frame_id:="$FRAME_ID" \
    rgb_topic:="$SLAM_COLOR_TOPIC" \
    depth_topic:="$SLAM_DEPTH_TOPIC" \
    camera_info_topic:="$SLAM_COLOR_INFO_TOPIC" \
    approx_sync:=true \
    publish_tf_odom:=false \
    publish_tf_map:=false \
    rtabmap_viz:=false rviz:=false \
    database_path:=/out/rtabmap.db \
    rtabmap_args:="--delete_db_on_start $RTAB_ARGS" \
    odom_args:="$RTAB_ARGS" \
    ${IMU_ARGS[@]+"${IMU_ARGS[@]}"}

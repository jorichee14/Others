#!/bin/bash
# RTAB-Map RGB-D, with or without the IMU. Stage 1's feature+depth(+inertial)
# backbone, and the only tier-3 entry that also produces a dense map.
set -euo pipefail
# Sourcing ROS under `set -u` kills the script before it starts: setup.bash reads
# AMENT_TRACE_SETUP_FILES with no default, and an unbound variable under -u is a
# fatal error. The strictness is wanted everywhere else -- an unset SLAM_* that
# expands to nothing is exactly how a container runs on starved input -- so it is
# lifted for the source and put straight back.
set +u
source /opt/ros/humble/setup.bash
set -u
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

# EXACT vs APPROXIMATE, from the stream's measured stamp alignment rather than
# a default. On mobile_1 colour and depth are stamped BYTE-IDENTICALLY (2288/2288
# pairs), and feeding that to an approximate matcher is what paired frames a full
# 67 ms period apart on the first run -- a whole frame of cart motion inside each
# RGB-D pair. Exact sync cannot mispair; approximate sync can, even when a
# perfect partner exists.
INTERVAL=()
case "${SLAM_SYNC:-approx}" in
    exact)  APPROX=false; echo "sync     : EXACT (stamps are identical on this stream)" ;;
    approx) APPROX=true
            # Tight, because the measured jitter is microseconds. The interval's
            # job is to REJECT a pairing when the real partner was dropped, not
            # to absorb a systematic offset -- there is none to absorb.
            INTERVAL=(approx_sync_max_interval:=0.005)
            echo "sync     : approximate, max interval 5 ms" ;;
    *)      echo "SLAM_SYNC=${SLAM_SYNC} is neither exact nor approx" >&2; exit 2 ;;
esac

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
    approx_sync:="$APPROX" ${INTERVAL[@]+"${INTERVAL[@]}"} \
    publish_tf_odom:=true \
    publish_tf_map:=true \
    rtabmap_viz:=false rviz:=false \
    database_path:=/out/rtabmap.db \
    rtabmap_args:="--delete_db_on_start $RTAB_ARGS" \
    odom_args:="$RTAB_ARGS" \
    ${IMU_ARGS[@]+"${IMU_ARGS[@]}"}

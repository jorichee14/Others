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

# WHY THE ODOMETRY TRANSFORM IS NOT PUBLISHED, on the second attempt.
#
# I turned it on once, reasoning that nothing else publishes `odom` so there
# could be no conflict. That was the wrong half of the edge to look at. A frame
# in tf has exactly ONE parent, and the bag's tf_static already gives
# zed_left_camera_optical_frame a parent: zed_left_camera_frame. Publishing
# odom -> zed_left_camera_optical_frame RE-PARENTS it, which silently deletes
# the static link -- and with it the only path from the camera to the IMU.
#
# The symptom is unmistakable once seen: "Lookup would require extrapolation
# into the future ... looking up transform from [zed_imu_link] to
# [zed_left_camera_optical_frame]". A lookup between two STATICALLY related
# frames can never extrapolate. That it did proves the path was routing through
# a dynamic edge that should not have been in it.
#
# The mapping node warns that it cannot look odom up in tf. That warning is
# cosmetic here: it also subscribes to the odometry TOPIC, which is how it built
# a 58-node map in the run before this one.
#
# WHICH FRAME THE POSES COME OUT IN -- read off the bag, never assumed.
# RTAB-Map's frame_id is the body frame it expresses odometry in. Setting it to
# the DEPTH IMAGE'S OWN optical frame makes that transform the identity, so the
# container emits the sensor frame it was given and the evaluator applies
# reference_frame_from_sensor exactly once (container contract rule 1). Setting
# it to a base_link instead would emit body poses that look fine and are wrong by
# the camera's lever arm.
FRAME_ID=$(python3 /opt/slambench/sniff_frame.py --bag /bag --topic "$SLAM_DEPTH_TOPIC")
echo "frame_id : $FRAME_ID  (from the first $SLAM_DEPTH_TOPIC header)"

# Slash keys are RTAB-Map parameters. The ODOMETRY node accepts every group,
# so it gets the whole block. The MAPPING node does not declare Odom/*, and an
# undeclared parameter in ROS 2 is an exception, not a warning: it aborted at
# startup on `--Odom/ResetCountdown 8` and the run quietly became odometry-only.
# So the mapping node gets everything except Odom/*.
ODOM_ARGS=$(python3 -c '
import json, os, sys
sys.path.insert(0, "/opt/slambench")
from params_to_args import to_args
print(" ".join(to_args(json.loads(os.environ.get("SLAM_PARAMS") or "{}"))))')
RTAB_ARGS=$(python3 -c '
import json, os, sys
sys.path.insert(0, "/opt/slambench")
from params_to_args import to_args
print(" ".join(to_args(json.loads(os.environ.get("SLAM_PARAMS") or "{}"), exclude=("Odom/",))))')
echo "odometry : $ODOM_ARGS"
echo "mapping  : $RTAB_ARGS"

# NO odom_info. Measured across every run: with the mapping node dead the
# odometry processes consecutive frames (period 67 ms, 72% kept); with it
# alive, exactly every other frame (period 134 ms, 20-41% kept), at both replay
# rates, with the bag delivering all 2291 frames and the callback's own cost at
# p90 46 ms. The one thing the mapping node changes INSIDE the odometry node is
# subscribe_odom_info: with a subscriber present, the odometry builds and
# publishes the whole local feature map on every frame, in the hot path. It is
# switched off. Gravity still reaches the graph: the mapping node subscribes to
# the IMU itself, and Mem/UseOdomGravity (which would take gravity from
# odom_info instead) is left at its published default, false.
#
# PROCESS EVERY FRAME, NOT THE MOST RECENT ONE. rgbd_odometry defaults to
# always_process_most_recent_frame=true: the worker thread holds dataMutex_ for
# the whole of processData() -- estimation plus publishing odom_info, a large
# message built only when the mapping node subscribes -- and the image callback
# does lockTry() and DROPS any frame that lands while it is held, with no log
# line unless the timing looks "flaky". A real-time design: newest frame wins.
# Measured: ~72% of frames kept with no mapping node, ~40% with one, identical
# at 1.0x and 0.5x replay, callback p90 46 ms against a 134 ms period -- the
# loss is the lock, not the compute. RTAB-Map's own warning text says what to do
# for a bag: always_process_most_recent_frame:=false, so the callback processes
# synchronously and the sync queue absorbs bursts. The queues are raised from 10
# so a burst queues instead of overflowing the subscription. Harness plumbing,
# not method tuning: the estimator's parameters are unchanged.
#
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
    # wait_imu_to_init stays ON. Turning it off (2026-09-19) did not touch the
    # frame loss and made the tracking less stable: 4 resets instead of 2.
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
    odom_always_process_most_recent_frame:=false \
    topic_queue_size:=50 sync_queue_size:=50 \
    subscribe_odom_info:=false \
    publish_tf_odom:=false \
    publish_tf_map:=false \
    rtabmap_viz:=false rviz:=false \
    database_path:=/out/rtabmap.db \
    rtabmap_args:="--delete_db_on_start $RTAB_ARGS" \
    odom_args:="$ODOM_ARGS" \
    ${IMU_ARGS[@]+"${IMU_ARGS[@]}"}

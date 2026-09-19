#!/bin/bash
# Replay the bag and the method side by side, record the trajectory, stop when
# the bag stops. Deliberately plain: a benchmark harness that is hard to read is
# a benchmark whose runs cannot be audited.
set -euo pipefail
source /opt/ros/humble/setup.bash

OUT=${OUT:-/out}
mkdir -p "$OUT"
: "${SLAM_TOPICS:?SLAM_TOPICS is required}"
: "${SLAM_POSE_TOPIC:?the image must declare which topic carries the estimate}"

echo "topics   : $SLAM_TOPICS"
echo "modality : ${SLAM_MODALITY:-?}"
echo "rate     : ${SLAM_RATE:-1.0}"
echo "params   : ${SLAM_PARAMS:-{}}"

# --- the depth->cloud bridge, for geometry-only estimators on a depth camera.
# Started here rather than inside each image because it is a FORMAT adapter, not
# a method: the same twenty lines would otherwise be copied into every image
# that wants points out of a camera, and copied code drifts.
BRIDGE=""
if [[ "${SLAM_NEEDS_CLOUD:-0}" == "1" && "${SLAM_MODALITY:-}" == "rgbd" ]]; then
    : "${SLAM_DEPTH_TOPIC:?a cloud is needed but the stream declares no depth topic}"
    : "${SLAM_DEPTH_SCALE:?the stream must declare depth_scale; see depth_to_cloud.py}"
    export DEPTH_TOPIC="$SLAM_DEPTH_TOPIC"
    export DEPTH_INFO_TOPIC="${SLAM_DEPTH_INFO_TOPIC:?depth_info_topic is required}"
    export CLOUD_TOPIC="/slambench/cloud"
    export DEPTH_SCALE="$SLAM_DEPTH_SCALE"
    export RANGE_MIN="${SLAM_RANGE_MIN:-0.0}"
    export RANGE_MAX="${SLAM_RANGE_MAX:-inf}"
    python3 /opt/slambench/depth_to_cloud.py --ros-args -p use_sim_time:=true &
    BRIDGE=$!
    # The estimator reads the bridge's output, not the bag's.
    export SLAM_CLOUD_TOPIC="$CLOUD_TOPIC"
    echo "bridge   : $SLAM_DEPTH_TOPIC -> $CLOUD_TOPIC (scale $SLAM_DEPTH_SCALE)"
fi

python3 /opt/slambench/record_tum.py \
    --pose-topic "$SLAM_POSE_TOPIC" \
    --map-topic "${SLAM_MAP_TOPIC:-}" \
    --out "$OUT" &
REC=$!

# Each image ships its own launch script. It is a FILE and not an environment
# variable on purpose: a Dockerfile `ENV` expands ${...} at BUILD time, so a
# launch line that referred to a run-time variable -- the cloud topic, the IMU
# topic -- silently baked in the empty string and the method started subscribed
# to nothing. That bug shipped once; this is the fix.
: "${SLAM_LAUNCH_SCRIPT:=/opt/slambench/launch.sh}"
test -x "$SLAM_LAUNCH_SCRIPT" || { echo "image ships no $SLAM_LAUNCH_SCRIPT" >&2; exit 2; }
"$SLAM_LAUNCH_SCRIPT" &
SLAM=$!

sleep 5   # let the method's nodes come up before the first message arrives
ros2 bag play /bag --clock -r "${SLAM_RATE:-1.0}" --topics ${SLAM_TOPICS//,/ }

# The bag is done; give the method a moment to flush its last optimisation.
sleep 10
for pid in "$SLAM" "$BRIDGE" "$REC"; do
    [[ -n "$pid" ]] || continue
    kill -INT "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
done

test -s "$OUT/trajectory.tum" || { echo "no trajectory written" >&2; exit 3; }
wc -l "$OUT/trajectory.tum"

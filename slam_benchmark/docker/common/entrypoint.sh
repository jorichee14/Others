#!/bin/bash
# Replay the bag and the method side by side, record the trajectory, stop when
# the bag stops. Deliberately plain: a benchmark harness that is hard to read is
# a benchmark whose runs cannot be audited.
set -euo pipefail
# Sourcing ROS under `set -u` kills the script before it starts: setup.bash reads
# AMENT_TRACE_SETUP_FILES with no default, and an unbound variable under -u is a
# fatal error. The strictness is wanted everywhere else -- an unset SLAM_* that
# expands to nothing is exactly how a container runs on starved input -- so it is
# lifted for the source and put straight back.
set +u
source /opt/ros/humble/setup.bash
set -u

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

# --- the camera<-IMU edge, when the bag's tf_static does not carry it.
# One transform, from configs/coop2.yaml, published so the method's own
# front-end can rotate the IMU into the camera frame. Without it RTAB-Map drops
# every inertial sample and the row silently becomes its own IMU-free control.
TFS=()
if [[ -n "${SLAM_STATIC_TF:-}" ]]; then
    while read -r P C X Y Z QX QY QZ QW; do
        [[ -n "$P" ]] || continue
        echo "static tf: $P -> $C  t=[$X $Y $Z]"
        ros2 run tf2_ros static_transform_publisher \
            --x "$X" --y "$Y" --z "$Z" --qx "$QX" --qy "$QY" --qz "$QZ" --qw "$QW" \
            --frame-id "$P" --child-frame-id "$C" \
            --ros-args -p use_sim_time:=true &
        TFS+=($!)
    done <<<"$SLAM_STATIC_TF"
fi

python3 /opt/slambench/record_tum.py \
    --pose-topic "$SLAM_POSE_TOPIC" \
    --map-topic "${SLAM_MAP_TOPIC:-}" \
    --path-topic "${SLAM_PATH_TOPIC:-}" \
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
for pid in "$SLAM" "$BRIDGE" ${TFS[@]+"${TFS[@]}"} "$REC"; do
    [[ -n "$pid" ]] || continue
    kill -INT "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
done

test -s "$OUT/trajectory.tum" || { echo "no trajectory written" >&2; exit 3; }
wc -l "$OUT/trajectory.tum"

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
    --pose-type "${SLAM_POSE_TYPE:-nav_msgs/msg/Odometry}" \
    --out "$OUT" \
    --ros-args -p use_sim_time:=true &
REC=$!

# CHECK THAT IT IS ACTUALLY ALIVE. A backgrounded process that dies on its first
# line leaves `&` perfectly happy, and the run then replays the whole bag into
# nothing -- which is exactly what happened here: the recorder was raising
# ParameterAlreadyDeclaredException on construction and every run produced a
# database and no trajectory. Three minutes is far too long to find that out at
# the end.
sleep 3
if ! kill -0 "$REC" 2>/dev/null || [[ ! -f "$OUT/odometry.tum" ]]; then
    echo "the recorder did not start -- refusing to replay the bag into nothing" >&2
    wait "$REC" 2>/dev/null || true
    exit 4
fi
echo "recorder : up, writing to $OUT"

# Each image ships its own launch script. It is a FILE and not an environment
# variable on purpose: a Dockerfile `ENV` expands ${...} at BUILD time, so a
# launch line that referred to a run-time variable -- the cloud topic, the IMU
# topic -- silently baked in the empty string and the method started subscribed
# to nothing. That bug shipped once; this is the fix.
: "${SLAM_LAUNCH_SCRIPT:=/opt/slambench/launch.sh}"
test -x "$SLAM_LAUNCH_SCRIPT" || { echo "image ships no $SLAM_LAUNCH_SCRIPT" >&2; exit 2; }
"$SLAM_LAUNCH_SCRIPT" &
SLAM=$!

# Shut down in one place, so an INTERRUPTED run still yields what it computed.
# `docker run` forwards Ctrl-C to PID 1, and without a trap bash dies on the spot:
# the recorder never reaches its finish(), and three minutes of replay produce an
# empty directory. A partial trajectory is a result (CLAUDE.md rule 8); nothing
# at all is not.
# Ask a process to stop, then INSIST. `wait` with no bound is how a single node
# that ignores SIGINT turns a completed 156 s replay into a run that never
# returns -- the replay is done and the poses are computed by then, so the cost
# of hanging here is the whole run for no reason.
stop_pid() {
    local pid=$1 limit=$2 waited=0
    kill -INT "$pid" 2>/dev/null || return 0
    while kill -0 "$pid" 2>/dev/null && (( waited < limit )); do
        sleep 1; waited=$((waited + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "pid $pid ignored SIGINT for ${limit}s — killing" >&2
        kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
}

shutdown() {
    # Order matters: the method first, so its final optimised graph is published
    # while the recorder is still listening, and the recorder last so it can
    # write it. The recorder gets the longest grace for the same reason -- it is
    # the one process whose clean exit IS the deliverable.
    for pid in "$SLAM" "$BRIDGE" ${TFS[@]+"${TFS[@]}"}; do
        [[ -n "$pid" ]] && stop_pid "$pid" 20
    done
    [[ -n "$REC" ]] && stop_pid "$REC" 45
    return 0
}
interrupted() {
    echo "INTERRUPTED — flushing whatever has been computed so far" >&2
    shutdown
    touch "$OUT/INTERRUPTED"      # so no table ever quotes this run as complete
    exit 130
}
trap interrupted INT TERM

sleep 5   # let the method's nodes come up before the first message arrives
ros2 bag play /bag --clock -r "${SLAM_RATE:-1.0}" --topics ${SLAM_TOPICS//,/ }

# The bag is done; give the method a moment to flush its last optimisation.
sleep 10
shutdown

# An image may ship a post-run step -- typically exporting the final graph
# from a database the method wrote, which is only consistent once it has
# stopped. It runs after the recorder too, so it can replace what the live
# capture produced and say so in timing.json.
if [[ -x /opt/slambench/post.sh ]]; then
    /opt/slambench/post.sh || echo "post.sh failed (non-fatal)" >&2
fi

test -s "$OUT/trajectory.tum" || { echo "no trajectory written" >&2; exit 3; }
wc -l "$OUT/trajectory.tum"

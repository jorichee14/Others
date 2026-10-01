#!/usr/bin/env bash
# Play an SVO/SVO2 through the ZED ROS 2 wrapper and record it:
#
#   processing/run_zed.sh <svo> <out_bag> <params.yaml> <topic>...
#
#   svo          the .svo2 file
#   out_bag      rosbag2 folder to create (must not exist)
#   params.yaml  wrapper parameter overrides (ros_params_override_path);
#                scoop.zed writes it: svo.use_svo_timestamps true, no loop, ...
#   topic...     the topics to record, as the wrapper names them
#
# Recording stops when the wrapper reports "SVO reached the end"; both logs go
# to <out_bag>/zed_logs/. Log times are replay times -- scoop.zed.restamp_bag
# puts them on capture time (process_recording.py and svo_to_bag.py do both).
#
# Needs ros2 + zed_wrapper and the GPU. Where they are not installed, this
# script runs itself inside the running container $ZED_CONTAINER (default
# isaac_ros_dev-x86_64-container) with docker exec; paths below $ISAAC_ROS_WS
# (default ~/workspaces/isaac_ros-dev) are the same files below
# $ZED_CONTAINER_WS (default /workspaces/isaac_ros-dev) in there.
# Also: ZED_CAMERA_MODEL (zed2i), ZED_ROS_SETUP (a setup.bash to source).
set -euo pipefail
die() { echo "run_zed: $*" >&2; exit 1; }

if [ $# -lt 4 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    sed -n '2,21p' "$0"; exit 1
fi
if [ -n "${ZED_IN_CONTAINER:-}" ]; then SVO=$1; OUT=$2; PARAMS=$3   # already translated
else SVO=$(realpath -m "$1"); OUT=$(realpath -m "$2"); PARAMS=$(realpath -m "$3"); fi
shift 3
MODEL=${ZED_CAMERA_MODEL:-zed2i}

have_zed() { command -v ros2 >/dev/null 2>&1 && ros2 pkg prefix zed_wrapper >/dev/null 2>&1; }

if [ -z "${ZED_IN_CONTAINER:-}" ]; then
    [ -f "$SVO" ] || die "no SVO at $SVO"
    [ -f "$PARAMS" ] || die "no parameter file at $PARAMS"
    [ ! -e "$OUT" ] || die "$OUT exists; remove it or pick another output"
    mkdir -p "$(dirname "$OUT")"
    if ! have_zed; then
        CONTAINER=${ZED_CONTAINER:-isaac_ros_dev-x86_64-container}
        HOST_WS=$(realpath -m "${ISAAC_ROS_WS:-$HOME/workspaces/isaac_ros-dev}")
        CONT_WS=${ZED_CONTAINER_WS:-/workspaces/isaac_ros-dev}
        inside() {
            case "$1" in
                "$HOST_WS"/*) echo "$CONT_WS${1#"$HOST_WS"}" ;;
                *) die "$1 is not below $HOST_WS, the folder the container sees" ;;
            esac
        }
        command -v docker >/dev/null 2>&1 \
            || die "no ros2 + zed_wrapper here and no docker; run this where the ZED wrapper is"
        docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" \
            || die "no ros2 + zed_wrapper here and container $CONTAINER is not running; start it or set ZED_CONTAINER"
        C_SVO=$(inside "$SVO"); C_OUT=$(inside "$OUT"); C_PARAMS=$(inside "$PARAMS")
        PIDFILE="$(dirname "$OUT")/.$(basename "$OUT").pid"; C_PIDFILE=$(inside "$PIDFILE")
        rm -f "$PIDFILE"
        echo "run_zed: running in container $CONTAINER"
        docker exec -i -e ZED_IN_CONTAINER=1 -e ZED_CAMERA_MODEL="$MODEL" \
            -e ZED_ROS_SETUP="${ZED_ROS_SETUP:-}" -e ZED_WS="$CONT_WS" \
            -e ZED_OWNER="$(id -u):$(id -g)" -e ZED_PIDFILE="$C_PIDFILE" \
            "$CONTAINER" bash -s -- "$C_SVO" "$C_OUT" "$C_PARAMS" "$@" < "$0" &
        DX=$!
        # Ctrl+C does not reach processes started by docker exec: stop the
        # replay in there ourselves, and wait until it has cleaned up.
        stop_inside() {
            trap '' INT TERM
            if [ -f "$PIDFILE" ]; then
                echo "run_zed: stopping the replay in $CONTAINER ..."
                docker exec "$CONTAINER" bash -c "p=\$(cat '$C_PIDFILE'); kill -TERM \$p 2>/dev/null;
                    while kill -0 \$p 2>/dev/null; do sleep 1; done" || true
            fi
            wait "$DX" 2>/dev/null || true
            rm -f "$PIDFILE"
            exit 130
        }
        trap stop_inside INT TERM
        code=0; wait "$DX" || code=$?
        rm -f "$PIDFILE"
        exit "$code"
    fi
fi

if ! have_zed; then                    # in the container: source ROS and the workspace
    set +u
    for f in "/opt/ros/${ROS_DISTRO:-humble}/setup.bash" \
             "${ZED_WS:-/workspaces/isaac_ros-dev}/install/setup.bash" ${ZED_ROS_SETUP:-}; do
        [ -f "$f" ] && source "$f"
    done
    set -u
    have_zed || die "zed_wrapper not found; set ZED_ROS_SETUP to the setup.bash that has it"
fi

# Background jobs of a script ignore SIGINT; give the recorder and the launch
# the default back, so they stop cleanly (bag closed, metadata.yaml written).
if env --default-signal=INT true 2>/dev/null; then SIGDFL=(env --default-signal=INT)
else set -m; SIGDFL=(); fi

LOGS=$(mktemp -d)
REC=""; ZED=""
[ -n "${ZED_PIDFILE:-}" ] && echo $$ > "$ZED_PIDFILE"      # so the host can stop us
stop() {                               # SIGINT, then wait (at most $2 s) for it to exit
    local pid=$1 t=0
    [ -n "$pid" ] || return 0
    kill -INT "$pid" 2>/dev/null || return 0
    while kill -0 "$pid" 2>/dev/null && [ "$t" -lt "$2" ]; do sleep 1; t=$((t + 1)); done
    kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null || true
}
finish() {
    stop "$ZED" 30; stop "$REC" 120
    if [ -d "$OUT" ]; then
        mkdir -p "$OUT/zed_logs"; cp "$LOGS"/* "$OUT/zed_logs/" 2>/dev/null || true
    else                               # nothing recorded: show the logs before they go
        tail -n 20 "$LOGS"/*.log >&2 2>/dev/null || true
    fi
    if [ -n "${ZED_OWNER:-}" ] && [ -e "$OUT" ]; then chown -R "$ZED_OWNER" "$OUT" || true; fi
    rm -rf "$LOGS"
    [ -n "${ZED_PIDFILE:-}" ] && rm -f "$ZED_PIDFILE"
    return 0
}
trap finish EXIT
trap 'exit 130' INT TERM

echo "svo:    $SVO"
echo "out:    $OUT"
echo "topics: $*"
# in pieces of $ZED_SPLIT_BYTES (4 GiB), so the restamp can delete each one once read
"${SIGDFL[@]}" ros2 bag record -s mcap --max-cache-size $((512 * 1024 * 1024)) \
    --max-bag-size "${ZED_SPLIT_BYTES:-4294967296}" -o "$OUT" "$@" \
    </dev/null >"$LOGS/record.log" 2>&1 &
REC=$!
for _ in $(seq 30); do                 # until the recorder listens
    grep -q "Listening for topics" "$LOGS/record.log" 2>/dev/null && break
    kill -0 "$REC" 2>/dev/null || { cat "$LOGS/record.log"; die "ros2 bag record did not start"; }
    sleep 1
done

"${SIGDFL[@]}" ros2 launch zed_wrapper zed_camera.launch.py camera_model:="$MODEL" \
    svo_path:="$SVO" ros_params_override_path:="$PARAMS" \
    </dev/null >"$LOGS/zed.log" 2>&1 &
ZED=$!
echo "playing the SVO (log: $OUT/zed_logs/zed.log when done) ..."
start=$SECONDS; shown=0
until grep -q "SVO reached the end" "$LOGS/zed.log"; do
    if ! kill -0 "$ZED" 2>/dev/null; then
        tail -n 30 "$LOGS/zed.log"; die "the ZED wrapper exited before the end of the SVO"
    fi
    kill -0 "$REC" 2>/dev/null || { tail -n 30 "$LOGS/record.log"; die "ros2 bag record stopped early"; }
    if [ $((SECONDS - start)) -ge $((shown + 30)) ]; then
        shown=$((SECONDS - start))
        echo "    ${shown} s, $(du -sh "$OUT" 2>/dev/null | cut -f1) recorded"
    fi
    sleep 1
done
echo "end of the SVO after $((SECONDS - start)) s"
sleep "${ZED_DRAIN_S:-3}"              # the last messages still on their way
stop "$REC" 120; REC=""
stop "$ZED" 30; ZED=""
[ -f "$OUT/metadata.yaml" ] || die "ros2 bag record left no metadata.yaml in $OUT"
echo "recorded $OUT"

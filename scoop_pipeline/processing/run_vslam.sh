#!/usr/bin/env bash
# Play a robot's stereo + IMU out of a bag through Isaac ROS cuVSLAM and record
# its odometry:
#
#   processing/run_vslam.sh <bag> <out_bag> <launch.py> <rate> <topic>...
#
#   bag        the bag to play (the merged pass bag)
#   out_bag    rosbag2 folder to create (must not exist): /visual_slam/* topics
#   launch.py  the cuVSLAM launch file (processing/vslam.py writes it)
#   rate       ros2 bag play --rate
#   topic...   the topics to play (images, camera_info, imu, /tf_static)
#
# The bag plays with --clock and cuVSLAM runs on sim time, so its odometry is
# stamped with the images' header stamps. Recording stops when the play ends.
# Both logs go to <out_bag>/vslam_logs/.
#
# Needs ros2 + isaac_ros_visual_slam and the GPU. Where they are not installed,
# this script runs itself inside the running container $VSLAM_CONTAINER
# (default isaac_ros_dev-x86_64-container) with docker exec; paths below
# $ISAAC_ROS_WS (default ~/workspaces/isaac_ros-dev) are the same files below
# $VSLAM_CONTAINER_WS (default /workspaces/isaac_ros-dev) in there.
# Also: VSLAM_ROS_SETUP (a setup.bash to source).
#
# VSLAM_MONITOR=1: log what cuVSLAM costs and how fast it answers, as
# slam/odometry/run_odometry.sh does for the other methods (slam/monitor/):
# <out_bag>/monitor/ gets messages.csv (every image, IMU and odometry message),
# resources.csv (the component container's CPU and memory, the GPU),
# trajectory.tum (the odometry, of base_frame) and summary.json (latency per
# left image, drops, rates, MB/s, whether it kept up). Meaningful at rate 1.
#
# Hooks, beside the launch file (processing/vslam_loc.py writes them; a plain
# run has none): record_extra.txt, more topics to record (one per line);
# during_play.json, `python3 vslam_srv.py during <its folder>` runs alongside the
# play (it waits for its own moment on /clock: localize in a saved map);
# after_play.json, `python3 vslam_srv.py after <its folder>` runs once the bag
# has played, before cuVSLAM is stopped (save the map, dump its keyframes).
set -euo pipefail
die() { echo "run_vslam: $*" >&2; exit 1; }

if [ $# -lt 5 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    sed -n '2,23p' "$0"; exit 1
fi
if [ -n "${VSLAM_IN_CONTAINER:-}" ]; then BAG=$1; OUT=$2; LAUNCH=$3   # already translated
else BAG=$(realpath -m "$1"); OUT=$(realpath -m "$2"); LAUNCH=$(realpath -m "$3"); fi
RATE=$4
shift 4

have_vslam() { command -v ros2 >/dev/null 2>&1 && ros2 pkg prefix isaac_ros_visual_slam >/dev/null 2>&1; }

if [ -z "${VSLAM_IN_CONTAINER:-}" ]; then
    [ -e "$BAG" ] || die "no bag at $BAG"
    [ -f "$LAUNCH" ] || die "no launch file at $LAUNCH"
    [ ! -e "$OUT" ] || die "$OUT exists; remove it or pick another output"
    mkdir -p "$(dirname "$OUT")"
    if ! have_vslam; then
        CONTAINER=${VSLAM_CONTAINER:-isaac_ros_dev-x86_64-container}
        HOST_WS=$(realpath -m "${ISAAC_ROS_WS:-$HOME/workspaces/isaac_ros-dev}")
        CONT_WS=${VSLAM_CONTAINER_WS:-/workspaces/isaac_ros-dev}
        inside() {
            case "$1" in
                "$HOST_WS"/*) echo "$CONT_WS${1#"$HOST_WS"}" ;;
                *) die "$1 is not below $HOST_WS, the folder the container sees" ;;
            esac
        }
        command -v docker >/dev/null 2>&1 \
            || die "no ros2 + isaac_ros_visual_slam here and no docker"
        docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" \
            || die "no ros2 + isaac_ros_visual_slam here and container $CONTAINER is not running; start it or set VSLAM_CONTAINER"
        C_BAG=$(inside "$BAG"); C_OUT=$(inside "$OUT"); C_LAUNCH=$(inside "$LAUNCH")
        C_MON=""
        if [ "${VSLAM_MONITOR:-0}" = 1 ]; then        # the container sees only the workspace
            MON="$(dirname "$OUT")/.monitor_$(basename "$OUT")"
            mkdir -p "$MON" && cp "$(dirname "$(realpath "$0")")"/../slam/monitor/*.py "$MON"/
            C_MON=$(inside "$MON")
        fi
        PIDFILE="$(dirname "$OUT")/.$(basename "$OUT").pid"; C_PIDFILE=$(inside "$PIDFILE")
        rm -f "$PIDFILE"
        echo "run_vslam: running in container $CONTAINER"
        docker exec -i -e VSLAM_IN_CONTAINER=1 -e VSLAM_ROS_SETUP="${VSLAM_ROS_SETUP:-}" \
            -e PYTHONDONTWRITEBYTECODE=1 \
            -e VSLAM_WS="$CONT_WS" -e VSLAM_OWNER="$(id -u):$(id -g)" \
            -e VSLAM_PIDFILE="$C_PIDFILE" -e VSLAM_MONITOR_SCRIPTS="$C_MON" \
            "$CONTAINER" bash -s -- "$C_BAG" "$C_OUT" "$C_LAUNCH" "$RATE" "$@" < "$0" &
        DX=$!
        # Ctrl+C does not reach processes started by docker exec: stop it in
        # there ourselves, and wait until it has cleaned up.
        stop_inside() {
            trap '' INT TERM
            if [ -f "$PIDFILE" ]; then
                echo "run_vslam: stopping in $CONTAINER ..."
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

if [ -z "${VSLAM_IN_CONTAINER:-}" ] && [ "${VSLAM_MONITOR:-0}" = 1 ]; then
    VSLAM_MONITOR_SCRIPTS=$(realpath "$(dirname "$(realpath "$0")")/../slam/monitor")
fi
if ! have_vslam; then                  # in the container: source ROS and the workspace
    set +u
    for f in "/opt/ros/${ROS_DISTRO:-humble}/setup.bash" \
             "${VSLAM_WS:-/workspaces/isaac_ros-dev}/install/setup.bash" ${VSLAM_ROS_SETUP:-}; do
        [ -f "$f" ] && source "$f"
    done
    set -u
    have_vslam || die "isaac_ros_visual_slam not found; set VSLAM_ROS_SETUP to the setup.bash that has it"
fi

if env --default-signal=INT true 2>/dev/null; then SIGDFL=(env --default-signal=INT)
else set -m; SIGDFL=(); fi

LOGS=$(mktemp -d)
HOOKS=$(dirname "$LAUNCH")
EXTRA=(); [ -f "$HOOKS/record_extra.txt" ] && mapfile -t EXTRA < <(grep -v '^\s*$' "$HOOKS/record_extra.txt")
REC=""; SLAM=""; PLAY=""; PROBE=""; RES=""; DURING=""
[ -n "${VSLAM_PIDFILE:-}" ] && echo $$ > "$VSLAM_PIDFILE"  # so the host can stop us
stop() {                               # SIGINT, then wait (at most $2 s) for it to exit
    local pid=$1 t=0
    [ -n "$pid" ] || return 0
    kill -INT "$pid" 2>/dev/null || return 0
    while kill -0 "$pid" 2>/dev/null && [ "$t" -lt "$2" ]; do sleep 1; t=$((t + 1)); done
    kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null || true
}
finish() {
    stop "$DURING" 5; stop "$PLAY" 10; stop "$SLAM" 30; stop "$REC" 60; stop "$PROBE" 10; stop "$RES" 5
    if [ -n "$PROBE" ] && [ -d "$OUT/monitor" ]; then
        python3 "$VSLAM_MONITOR_SCRIPTS/summarize.py" "$OUT/monitor" || true
        [ -n "${VSLAM_IN_CONTAINER:-}" ] && rm -rf "$VSLAM_MONITOR_SCRIPTS"
    fi
    if [ -d "$OUT" ]; then
        mkdir -p "$OUT/vslam_logs"; cp "$LOGS"/* "$OUT/vslam_logs/" 2>/dev/null || true
    else
        tail -n 30 "$LOGS"/*.log >&2 2>/dev/null || true
    fi
    if [ -n "${VSLAM_OWNER:-}" ]; then               # all it wrote, the launch's folder too
        chown -R "$VSLAM_OWNER" "$(dirname "$LAUNCH")" 2>/dev/null || true
        [ -e "$OUT" ] && chown -R "$VSLAM_OWNER" "$OUT" || true
    fi
    rm -rf "$LOGS"
    [ -n "${VSLAM_PIDFILE:-}" ] && rm -f "$VSLAM_PIDFILE"
    return 0
}
trap finish EXIT
trap 'exit 130' INT TERM

echo "bag:    $BAG"
echo "out:    $OUT"
echo "play:   $* (rate $RATE)"
"${SIGDFL[@]}" ros2 launch "$LAUNCH" </dev/null >"$LOGS/vslam.log" 2>&1 &
SLAM=$!
"${SIGDFL[@]}" ros2 bag record -s mcap -o "$OUT" --use-sim-time \
    /visual_slam/tracking/odometry /visual_slam/status "${EXTRA[@]}" \
    </dev/null >"$LOGS/record.log" 2>&1 &
REC=$!
for _ in $(seq 30); do                 # until the recorder listens
    grep -q "Listening for topics" "$LOGS/record.log" 2>/dev/null && break
    kill -0 "$REC" 2>/dev/null || { cat "$LOGS/record.log"; die "ros2 bag record did not start"; }
    sleep 1
done
if [ -n "${VSLAM_MONITOR_SCRIPTS:-}" ]; then
    mkdir -p "$OUT/monitor"
    PARGS=(); KEY=""
    for t in "$@"; do
        case "$t" in
            */camera_info|/tf_static|/tf) ;;
            *imu*) PARGS+=(--in "$t=sensor_msgs/msg/Imu") ;;
            *) PARGS+=(--in "$t=sensor_msgs/msg/Image"); [ -n "$KEY" ] || KEY=$t ;;
        esac
    done
    python3 - "$OUT/monitor/run.json" "$KEY" <<'PYEOF'
import json, sys
json.dump({"method": "cuvslam", "key_topic": sys.argv[2], "out_topic": "/visual_slam/tracking/odometry",
           "stamp_tol_s": 0.0, "tum_frame": "odometry child frame (base_frame)"}, open(sys.argv[1], "w"), indent=1)
PYEOF
    "${SIGDFL[@]}" python3 "$VSLAM_MONITOR_SCRIPTS/probe.py" --dir "$OUT/monitor" "${PARGS[@]}" \
        --out /visual_slam/tracking/odometry=nav_msgs/msg/Odometry --tum /visual_slam/tracking/odometry \
        </dev/null >"$LOGS/probe.log" 2>&1 &
    PROBE=$!
    "${SIGDFL[@]}" python3 "$VSLAM_MONITOR_SCRIPTS/resources.py" --dir "$OUT/monitor" \
        --match component_container </dev/null >"$LOGS/resources.log" 2>&1 &
    RES=$!
    echo "monitor: $OUT/monitor (key $KEY)"
fi
sleep "${VSLAM_START_S:-8}"            # cuVSLAM up before the first image
kill -0 "$SLAM" 2>/dev/null || { tail -n 30 "$LOGS/vslam.log"; die "cuVSLAM exited at start"; }

# with the monitor, the player's publishers exist a second before the first message
DELAY=(); [ -n "$PROBE" ] && DELAY=(--delay 1)
"${SIGDFL[@]}" ros2 bag play "$BAG" --clock --rate "$RATE" "${DELAY[@]}" --topics "$@" \
    </dev/null >"$LOGS/play.log" 2>&1 &
PLAY=$!
echo "playing the bag (logs: $OUT/vslam_logs/ when done) ..."
if [ -f "$HOOKS/during_play.json" ]; then
    "${SIGDFL[@]}" python3 "$HOOKS/vslam_srv.py" during "$HOOKS" </dev/null >"$LOGS/during.log" 2>&1 &
    DURING=$!
    echo "during the play: vslam_srv.py during (log: during.log)"
fi
start=$SECONDS; shown=0
while kill -0 "$PLAY" 2>/dev/null; do
    kill -0 "$SLAM" 2>/dev/null || { tail -n 30 "$LOGS/vslam.log"; die "cuVSLAM exited during the play"; }
    kill -0 "$REC" 2>/dev/null || { tail -n 30 "$LOGS/record.log"; die "ros2 bag record stopped early"; }
    if [ $((SECONDS - start)) -ge $((shown + 30)) ]; then
        shown=$((SECONDS - start))
        echo "    ${shown} s"
    fi
    sleep 1
done
wait "$PLAY" || { tail -n 20 "$LOGS/play.log"; die "ros2 bag play failed"; }
PLAY=""
echo "end of the bag after $((SECONDS - start)) s"
sleep "${VSLAM_DRAIN_S:-3}"
if [ -n "$DURING" ]; then
    kill -0 "$DURING" 2>/dev/null && echo "the during-play hook is still running: stopped"
    stop "$DURING" 5; DURING=""; cat "$LOGS/during.log"
fi
if [ -f "$HOOKS/after_play.json" ]; then
    echo "after the play: vslam_srv.py after (save the map, keyframes) ..."
    timeout "${VSLAM_AFTER_TIMEOUT_S:-900}" python3 "$HOOKS/vslam_srv.py" after "$HOOKS" </dev/null \
        >"$LOGS/after.log" 2>&1 || { cat "$LOGS/after.log"; die "the after-play hook failed"; }
    cat "$LOGS/after.log"
fi
stop "$REC" 60; REC=""
stop "$SLAM" 30; SLAM=""
[ -f "$OUT/metadata.yaml" ] || die "ros2 bag record left no metadata.yaml in $OUT"
echo "recorded $OUT"

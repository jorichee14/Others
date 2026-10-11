#!/bin/bash
# Hydra on a datasets/hydra.py export, every robot streaming its scene graph to
# a server through a channel (scoop/channel.py): Hydra's counterpart of
# slam/run_swarm_slam.sh. Run inside Hydra's container (hydra_ros/docker,
# PROFILE=dev), with the data and this repo mounted at the same paths as outside.
#
#   scoop_pipeline/slam/run_hydra.sh <hydra export dir> [channel ideal] [rate 0.5] [robots "mobile_1 mobile_2"]
#
# channel: any scoop/channel.py spec (ideal, v2xvit, logged:<pass>, fit:<pass>,
#   synthetic:...). A scene-graph update is the whole map, MBs: logged, fit and
#   synthetic channels get ",queue", so an update waits until the one before it
#   is through (the same link cannot send two at once).
# Robot i runs Hydra (exit_after_clock: it saves and stops when its bag ends) in
#   ROS domain DOMAIN_BASE+i (default 20+i) with its bag_r<i> (--clock), and
#   publishes /r<i>/hydra/backend/dsg; the server is domain DOMAIN_BASE+N, where
#   slam/channel_relay.py delivers every robot's stream ("server" for the
#   channel: one hop, the robot's uplink) and analysis/slam/dsg_stream.py logs
#   what arrives.
#
# Writes <export>/runs/<channel>/ (REPEAT=<n>: runs/<channel>_r<n>):
#   hydra_out_r<i>/ (Hydra's map and trajectory: analysis/slam/hydra_eval.py),
#   relay.csv (every update: sent, size, lost, arrival), server_dsg.csv (what the
#   server received), hydra_r<i>.log, relay.log, server.log.
set -e
OUT=$(realpath "$1"); CHANNEL=${2:-ideal}; RATE=${3:-0.5}; ROBOTS=(${4:-mobile_1 mobile_2})
N=${#ROBOTS[@]}; BASE=${DOMAIN_BASE:-20}; SERVER=$((BASE + N))
HERE=$(dirname "$(realpath "$0")")
for ((i = 0; i < N; i++)); do
  [ -d "$OUT/bag_r$i" ] && [ -f "$OUT/hydra_r$i.yaml" ] || { echo "no $OUT/bag_r$i or hydra_r$i.yaml: run datasets/hydra.py first"; exit 1; }
done
grep -q dsg_topic "$OUT/scoop_hydra.launch.yaml" || { echo "$OUT/scoop_hydra.launch.yaml is older than this script: run datasets/hydra.py <swarm_slam export> --configs-only"; exit 1; }
source /opt/ros/$ROS_DISTRO/setup.bash
source /root/hydra_ws/install/setup.bash
TAG=$(echo "$CHANNEL" | sed -E 's#:[^,]*/([^/,]+)#:\1#; s#[:,=/]#_#g')${REPEAT:+_r$REPEAT}
RUN="$OUT/runs/$TAG"
[ -e "$RUN" ] && { echo "$RUN exists: remove it to run this channel again"; exit 1; }
SPEC=$CHANNEL
case "$SPEC" in
  logged:*|fit:*|synthetic:*) [[ "$SPEC" == *queue* ]] || SPEC="$SPEC,queue" ;;
esac
# Hydra nodes left in these domains would publish into this run
for D in $(seq $BASE $SERVER); do
  if ROS_DOMAIN_ID=$D ros2 node list --no-daemon --spin-time 3 2>/dev/null | grep -qE "/hydra$|dsg_stream_logger|scoop_channel_relay"; then
    echo "Hydra, a logger or a relay is already running in ROS domain $D: stop it first"; exit 1
  fi
done
mkdir -p "$RUN"
PIDS=(); PLAY=(); TOPICS=()
for ((i = 0; i < N; i++)); do TOPICS+=(--topic /r$i/hydra/backend/dsg); done
# each in its own process group, SIGINT restored (a background job of a script starts
# with it ignored); the container runs as root: hand the run folder back to the export's owner
trap 'set +e; for p in ${PLAY[@]} ${PIDS[@]}; do kill -INT -- -$p 2>/dev/null; done; sleep 5
      for p in ${PLAY[@]} ${PIDS[@]}; do kill -KILL -- -$p 2>/dev/null; done; wait 2>/dev/null
      chown -R --reference="$OUT" "$RUN" 2>/dev/null' EXIT
for ((i = 0; i < N; i++)); do
  ROS_DOMAIN_ID=$((BASE + i)) setsid env --default-signal=INT ros2 launch "$OUT/scoop_hydra.launch.yaml" \
      robot_id:=$i config:="$OUT/hydra_r$i.yaml" log_path:="$RUN/hydra_out_r$i" \
      dsg_topic:=/r$i/hydra/backend/dsg visualize:=false > "$RUN/hydra_r$i.log" 2>&1 &
  PIDS+=($!)
done
ROS_DOMAIN_ID=$SERVER setsid env --default-signal=INT python3 "$HERE/../analysis/slam/dsg_stream.py" log \
    "${TOPICS[@]}" --out "$RUN/server_dsg.csv" > "$RUN/server.log" 2>&1 &
PIDS+=($!)
DOMAINS=$(seq -s " " $BASE $SERVER)
setsid env --default-signal=INT python3 "$HERE/channel_relay.py" --channel "$SPEC" --robots "${ROBOTS[@]}" server \
    --domains $DOMAINS --topics '^/r[0-9]+/hydra/backend/dsg$' --log "$RUN/relay.csv" > "$RUN/relay.log" 2>&1 &
PIDS+=($!)
# play once every Hydra waits for its camera (the segmentation engine is loaded by then)
for ((i = 0; i < N; i++)); do
  for k in $(seq 300); do
    grep -q "Waiting for CameraInfo" "$RUN/hydra_r$i.log" 2>/dev/null && break
    grep -qE "process has died|Caught exception" "$RUN/hydra_r$i.log" 2>/dev/null && { echo "Hydra r$i failed: see $RUN/hydra_r$i.log"; exit 1; }
    sleep 1
  done
  grep -q "Waiting for CameraInfo" "$RUN/hydra_r$i.log" || { echo "Hydra r$i not ready after 300 s: see $RUN/hydra_r$i.log"; exit 1; }
done
echo "Hydra up for $N robots, channel $SPEC; playing the bags at x$RATE ($RUN)"
for ((i = 0; i < N; i++)); do
  ROS_DOMAIN_ID=$((BASE + i)) setsid env --default-signal=INT ros2 bag play "$OUT/bag_r$i" -r "$RATE" --clock \
      > "$RUN/play_r$i.log" 2>&1 & PLAY+=($!)
done
T0=$SECONDS
wait ${PLAY[@]}
if (( SECONDS - T0 < 20 )); then
  echo "the bags played for $((SECONDS - T0)) s: nothing was played (check ros2 bag info $OUT/bag_r0)"; exit 1
fi
echo "bags done after $((SECONDS - T0)) s; waiting for every Hydra to save and stop (up to 10 min)"
for k in $(seq 600); do
  pgrep -f "hydra_ros_node.*hydra_out_r" > /dev/null || break
  sleep 1
done
pgrep -f "hydra_ros_node.*hydra_out_r" > /dev/null && echo "a Hydra is still running after 10 min: stopping it (its map may not be saved)"
sleep 5                                                          # the last updates through the relay
for ((i = 0; i < N; i++)); do
  ls "$RUN/hydra_out_r$i/backend/trajectory.csv" 2>/dev/null || echo "r$i: no trajectory saved (see $RUN/hydra_r$i.log)"
done

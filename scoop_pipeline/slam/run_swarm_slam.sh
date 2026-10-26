#!/bin/bash
# Swarm-SLAM (cslam) on a datasets/swarm_slam.py export (RGB-D) or a
# datasets/swarm_slam_lidar.py one (LiDAR, any odometry), the robots
# talking through a channel (scoop/channel.py). Run inside Swarm-SLAM's
# container (cslam_experiments/docker), with the data and this repo mounted
# at the same paths as outside.
#
#   scoop_pipeline/slam/run_swarm_slam.sh <export dir> [channel ideal] [rate 0.5] [robots "mobile_1 mobile_2"]
#
# channel: any scoop/channel.py spec (ideal, v2xvit, cobevflow:0.3,
#   synthetic:delay_ms=50,mbps=100,loss=0.01, fit:<pass>, logged:<pass>[,queue]):
#   robot i runs in ROS domain DOMAIN_BASE+i (default 10+i) with its half of
#   the bag (--clock), and slam/channel_relay.py carries what they exchange.
#   "direct": every robot in one domain, no relay (Swarm-SLAM's own setup).
#
# Writes <export dir>/runs/<channel>/: results/ (cslam's logs), keyframes_r<i>/
# (each robot's /r<i>/cslam/keyframe_odom), relay.csv
# (every message: sent, size, lost, arrival), cslam_r<i>.log, relay.log.
set -e
OUT=$(realpath "$1"); CHANNEL=${2:-ideal}; RATE=${3:-0.5}; ROBOTS=(${4:-mobile_1 mobile_2})
N=${#ROBOTS[@]}; BASE=${DOMAIN_BASE:-10}
HERE=$(dirname "$(realpath "$0")")
if [ -f "$OUT/scoop_lidar.yaml" ]; then CFG=scoop_lidar.yaml; LAUNCH=cslam_lidar.launch.py
elif [ -f "$OUT/scoop_rgbd.yaml" ]; then CFG=scoop_rgbd.yaml; LAUNCH=cslam_rgbd.launch.py
else echo "no $OUT/scoop_rgbd.yaml or scoop_lidar.yaml: run datasets/swarm_slam.py or swarm_slam_lidar.py first"; exit 1; fi
source /opt/ros/$ROS_DISTRO/setup.bash
source /Swarm-SLAM/install/setup.bash 2>/dev/null || source ~/Swarm-SLAM/install/setup.bash
if [ "$CFG" = scoop_rgbd.yaml ] && [ ! -f "$OUT/resnet18_64.pth" ]; then
  echo "fetching the CosPlace ResNet-18/64 weights -> $OUT/resnet18_64.pth"
  python3 -c "import torch; m = torch.hub.load('gmberton/cosplace', 'get_trained_model', backbone='ResNet18', fc_output_dim=64); torch.save(m.state_dict(), '$OUT/resnet18_64.pth')"
fi
# REPEAT=<n> (environment): a repeat of the same channel, in runs/<channel>_r<n>
TAG=$(echo "$CHANNEL" | sed -E 's#:[^,]*/([^/,]+)#:\1#; s#[:,=/]#_#g')${REPEAT:+_r$REPEAT}
RUN="$OUT/runs/$TAG"
[ -e "$RUN" ] && { echo "$RUN exists: remove it to run this channel again"; exit 1; }
# cslam nodes left in these domains (e.g. from an earlier run in another container
# on the host network) would get the same data and publish the same keyframe ids
for ((i = 0; i < N; i++)); do
  D=$([ "$CHANNEL" = direct ] && echo 0 || echo $((BASE + i)))
  if ROS_DOMAIN_ID=$D ros2 node list --no-daemon --spin-time 3 2>/dev/null | grep -q cslam; then
    echo "cslam nodes are already running in ROS domain $D: stop them first (docker stop <the old container>)"; exit 1
  fi
done
mkdir -p "$RUN/results"
sed "s#log_folder: .*#log_folder: \"$RUN/results\"#" "$OUT/$CFG" > "$RUN/$CFG"
PIDS=(); PLAY=(); REC=()
# each launch in its own process group, so that the nodes it starts stop with it, and
# with SIGINT restored (a background job of a script starts with it ignored);
# the container runs as root: hand the run folder back to the export's owner
# (set +e: under set -e the first kill of an already finished group would end the trap)
# (the keyframe recorders first, and given time to write their bag's end)
trap 'set +e; for p in ${REC[@]}; do kill -INT -- -$p 2>/dev/null; done
      for k in $(seq 30); do kill -0 ${REC[@]} 2>/dev/null || break; sleep 1; done
      for p in ${PIDS[@]} ${PLAY[@]}; do kill -INT -- -$p 2>/dev/null; done; sleep 3
      for p in ${PIDS[@]} ${PLAY[@]} ${REC[@]}; do kill -KILL -- -$p 2>/dev/null; done; wait 2>/dev/null
      chown -R --reference="$OUT" "$RUN" 2>/dev/null' EXIT
for ((i = 0; i < N; i++)); do
  D=$([ "$CHANNEL" = direct ] && echo 0 || echo $((BASE + i)))
  ROS_DOMAIN_ID=$D setsid env --default-signal=INT ros2 launch cslam_experiments $LAUNCH config_path:="$RUN/" \
      config_file:=$CFG robot_id:=$i namespace:=/r$i max_nb_robots:=$N > "$RUN/cslam_r$i.log" 2>&1 &
  PIDS+=($!)
done
# every robot's keyframe ids and stamps (cslam logs them only once a robot is in an optimization)
for ((i = 0; i < N; i++)); do
  D=$([ "$CHANNEL" = direct ] && echo 0 || echo $((BASE + i)))
  ROS_DOMAIN_ID=$D setsid env --default-signal=INT ros2 bag record -s mcap -o "$RUN/keyframes_r$i" /r$i/cslam/keyframe_odom \
      > "$RUN/record_r$i.log" 2>&1 &
  REC+=($!)
done
if [ "$CHANNEL" != direct ]; then
  DOMAINS=$(for ((i = 0; i < N; i++)); do echo -n "$((BASE + i)) "; done)
  setsid env --default-signal=INT python3 "$HERE/channel_relay.py" --channel "$CHANNEL" --robots "${ROBOTS[@]}" --domains $DOMAINS \
      --log "$RUN/relay.csv" > "$RUN/relay.log" 2>&1 &
  PIDS+=($!)
fi
echo "cslam up for $N robots, channel $CHANNEL; playing the bag at x$RATE in 15 s ($RUN)"
sleep 15
PLAY=()
if [ "$CHANNEL" = direct ]; then
  ROS_DOMAIN_ID=0 setsid env --default-signal=INT ros2 bag play "$OUT/bag" -r "$RATE" & PLAY+=($!)
else
  for ((i = 0; i < N; i++)); do
    # ros2 bag play's --regex must match the whole topic name
    ROS_DOMAIN_ID=$((BASE + i)) setsid env --default-signal=INT ros2 bag play "$OUT/bag" -r "$RATE" --clock --regex "/r$i/.*" & PLAY+=($!)
  done
fi
T0=$SECONDS
wait ${PLAY[@]}
if (( SECONDS - T0 < 20 )); then
  echo "the bag played for $((SECONDS - T0)) s: no topics were played (check ros2 bag info $OUT/bag)"; exit 1
fi
echo "bag done after $((SECONDS - T0)) s; letting the pose graph settle 30 s"
sleep 30
ls "$RUN/results"

#!/bin/bash
# Swarm-SLAM (cslam, RGB-D) on a datasets/swarm_slam.py export, the robots
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
# Writes <export dir>/runs/<channel>/: results/ (cslam's logs), relay.csv
# (every message: sent, size, lost, arrival), cslam_r<i>.log, relay.log.
set -e
OUT=$(realpath "$1"); CHANNEL=${2:-ideal}; RATE=${3:-0.5}; ROBOTS=(${4:-mobile_1 mobile_2})
N=${#ROBOTS[@]}; BASE=${DOMAIN_BASE:-10}
HERE=$(dirname "$(realpath "$0")")
[ -f "$OUT/scoop_rgbd.yaml" ] || { echo "no $OUT/scoop_rgbd.yaml: run datasets/swarm_slam.py first"; exit 1; }
source /opt/ros/$ROS_DISTRO/setup.bash
source /Swarm-SLAM/install/setup.bash 2>/dev/null || source ~/Swarm-SLAM/install/setup.bash
if [ ! -f "$OUT/resnet18_64.pth" ]; then
  echo "fetching the CosPlace ResNet-18/64 weights -> $OUT/resnet18_64.pth"
  python3 -c "import torch; m = torch.hub.load('gmberton/cosplace', 'get_trained_model', backbone='ResNet18', fc_output_dim=64); torch.save(m.state_dict(), '$OUT/resnet18_64.pth')"
fi
TAG=$(echo "$CHANNEL" | sed -E 's#:[^,]*/([^/,]+)#:\1#; s#[:,=/]#_#g')
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
sed "s#log_folder: .*#log_folder: \"$RUN/results\"#" "$OUT/scoop_rgbd.yaml" > "$RUN/scoop_rgbd.yaml"
PIDS=(); PLAY=()
# each launch in its own process group, so that the nodes it starts stop with it;
# the container runs as root: hand the run folder back to the export's owner
trap 'for p in ${PIDS[@]} ${PLAY[@]}; do kill -INT -- -$p 2>/dev/null; done; sleep 3
      for p in ${PIDS[@]} ${PLAY[@]}; do kill -KILL -- -$p 2>/dev/null; done; wait 2>/dev/null
      chown -R --reference="$OUT" "$RUN" 2>/dev/null' EXIT
for ((i = 0; i < N; i++)); do
  D=$([ "$CHANNEL" = direct ] && echo 0 || echo $((BASE + i)))
  ROS_DOMAIN_ID=$D setsid ros2 launch cslam_experiments cslam_rgbd.launch.py config_path:="$RUN/" \
      config_file:=scoop_rgbd.yaml robot_id:=$i namespace:=/r$i max_nb_robots:=$N > "$RUN/cslam_r$i.log" 2>&1 &
  PIDS+=($!)
done
if [ "$CHANNEL" != direct ]; then
  DOMAINS=$(for ((i = 0; i < N; i++)); do echo -n "$((BASE + i)) "; done)
  setsid python3 "$HERE/channel_relay.py" --channel "$CHANNEL" --robots "${ROBOTS[@]}" --domains $DOMAINS \
      --log "$RUN/relay.csv" > "$RUN/relay.log" 2>&1 &
  PIDS+=($!)
fi
echo "cslam up for $N robots, channel $CHANNEL; playing the bag at x$RATE in 15 s ($RUN)"
sleep 15
PLAY=()
if [ "$CHANNEL" = direct ]; then
  ROS_DOMAIN_ID=0 setsid ros2 bag play "$OUT/bag" -r "$RATE" & PLAY+=($!)
else
  for ((i = 0; i < N; i++)); do
    # ros2 bag play's --regex must match the whole topic name
    ROS_DOMAIN_ID=$((BASE + i)) setsid ros2 bag play "$OUT/bag" -r "$RATE" --clock --regex "/r$i/.*" & PLAY+=($!)
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

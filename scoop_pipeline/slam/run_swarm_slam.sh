#!/bin/bash
# Swarm-SLAM (cslam, RGB-D) on a datasets/swarm_slam.py export: one cslam
# instance per robot (/r0, /r1, ...), then the bag. Run inside Swarm-SLAM's
# container (cslam_experiments/docker), with the data mounted at the same path.
#
#   scoop_pipeline/slam/run_swarm_slam.sh <export dir> [rate 0.5] [robots 2]
#
# Writes <export dir>/results/ (cslam's logs) and <export dir>/cslam_*.log.
set -e
OUT=$(realpath "$1"); RATE=${2:-0.5}; N=${3:-2}
[ -f "$OUT/scoop_rgbd.yaml" ] || { echo "no $OUT/scoop_rgbd.yaml: run datasets/swarm_slam.py first"; exit 1; }
source /opt/ros/$ROS_DISTRO/setup.bash
source /Swarm-SLAM/install/setup.bash 2>/dev/null || source ~/Swarm-SLAM/install/setup.bash
if [ ! -f "$OUT/resnet18_64.pth" ]; then
  echo "fetching the CosPlace ResNet-18/64 weights -> $OUT/resnet18_64.pth"
  python3 -c "import torch; m = torch.hub.load('gmberton/cosplace', 'get_trained_model', backbone='ResNet18', fc_output_dim=64); torch.save(m.state_dict(), '$OUT/resnet18_64.pth')"
fi
mkdir -p "$OUT/results"
PIDS=()
for ((i = 0; i < N; i++)); do
  ros2 launch cslam_experiments cslam_rgbd.launch.py config_path:="$OUT/" config_file:=scoop_rgbd.yaml \
      robot_id:=$i namespace:=/r$i max_nb_robots:=$N > "$OUT/cslam_r$i.log" 2>&1 &
  PIDS+=($!)
done
trap 'kill ${PIDS[@]} 2>/dev/null' EXIT
echo "cslam up for $N robots; playing the bag at x$RATE in 15 s (logs: $OUT/cslam_r*.log)"
sleep 15
ros2 bag play "$OUT/bag" -r "$RATE"
echo "bag done; letting the pose graph settle 30 s"
sleep 30
ls -la "$OUT/results"

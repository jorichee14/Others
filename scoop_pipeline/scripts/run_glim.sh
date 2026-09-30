#!/usr/bin/env bash
# Run GLIM (koide3/glim_ros2 docker image) offline on one bag, with its output
# written next to the bag and owned by you afterwards (no sudo needed).
#
#   scripts/run_glim.sh <bag_dir> [config_dir] [dump_dir]
#
#   bag_dir     the retimed bag, e.g. .../20260924/mapping_A/mobile_1/retimed_full
#   config_dir  GLIM config folder (default: $GLIM_CONFIG if set,
#               else ~/workspaces/isaac_ros-dev/glim_config)
#   dump_dir    where GLIM's dump goes (default: <folder of the bag>/glim_dump);
#               must be new or empty, so an earlier run is never overwritten
#
# The dump holds traj_lidar.txt (the trajectory stage 01 reads) and the map.
set -euo pipefail

if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    sed -n '2,13p' "$0"; exit 1
fi
BAG=$(realpath "$1")
CONFIG=$(realpath "${2:-${GLIM_CONFIG:-$HOME/workspaces/isaac_ros-dev/glim_config}}")
DUMP=$(realpath -m "${3:-$(dirname "$BAG")/glim_dump}")
IMAGE=${GLIM_IMAGE:-koide3/glim_ros2:humble-mcap}

[ -f "$BAG/metadata.yaml" ] || { echo "no metadata.yaml in $BAG: not a rosbag2 folder"; exit 1; }
[ -d "$CONFIG" ] || { echo "no GLIM config folder at $CONFIG"; exit 1; }
if [ -e "$DUMP" ] && [ -n "$(ls -A "$DUMP" 2>/dev/null)" ]; then
    echo "$DUMP already holds a GLIM run; remove it or pass another dump_dir"; exit 1
fi
mkdir -p "$DUMP"        # created by you, so docker does not create it as root

echo "bag:    $BAG"
echo "config: $CONFIG"
echo "dump:   $DUMP"
xhost +local: >/dev/null 2>&1 || true      # let the container open the viewer

# The image keeps its ROS install under /root, so GLIM has to run as root.
# Instead, the dump is handed back to you when GLIM exits -- also when it
# fails or is stopped with Ctrl+C -- so nothing needs sudo afterwards.
OWNER="$(id -u):$(id -g)"
docker run -it --rm --gpus all \
    -e DISPLAY="${DISPLAY:-:0}" -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$BAG":/bag:ro \
    -v "$CONFIG":/config:ro \
    -v "$DUMP":/tmp/dump \
    "$IMAGE" \
    bash -c "trap 'chown -R $OWNER /tmp/dump' EXIT; \
             ros2 run glim_ros glim_rosbag /bag --ros-args -p config_path:=/config"

echo
if [ -f "$DUMP/traj_lidar.txt" ]; then
    echo "trajectory: $DUMP/traj_lidar.txt ($(wc -l < "$DUMP/traj_lidar.txt") poses)"
else
    echo "GLIM finished but $DUMP/traj_lidar.txt is missing; check the output above"
fi

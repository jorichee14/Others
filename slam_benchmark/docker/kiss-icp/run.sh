#!/bin/bash
# KISS-ICP, the geometry-only backbone -- run offline, not as the ROS node.
#
# It replaces the shared entrypoint the way MASt3R's run.sh does, and meets the
# same contract: /out/trajectory.tum on the bag's clock, /out/map.ply, and
# /out/timing.json. See run_offline.py for why the node is not used.
set -euo pipefail
set +u
source /opt/ros/humble/setup.bash
set -u
: "${SLAM_MODALITY:?}" "${SLAM_PARAMS:?}"
echo "modality : $SLAM_MODALITY"
echo "params   : $SLAM_PARAMS"
exec python3 /opt/slambench/run_offline.py

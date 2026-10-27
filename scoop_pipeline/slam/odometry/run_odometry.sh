#!/bin/bash
# One odometry method on one robot of a pass, as it would run live: the bag
# played at 1x with --clock, the method on sim time, its output and its cost
# logged (slam/monitor/).
#
#   scoop_pipeline/slam/odometry/run_odometry.sh <method> <bag> <machine> <run dir> \
#       [--start S] [--duration S] [--imu-noise FILE] [--map PCD] [--ref TUM]
#
#   method   kiss_icp | rko_lio | fast_lio | fast2lamaa | fast2lamaa_loc (mobile_1, mobile_2) |
#            openvins | rtabmap_rgbd (mobile_2)
#   bag      the pass's rosbag2 folder (the finalized merged bag)
#   run dir  created; must not hold a run
#
# Environment: RATE (play rate, default 1.0: anything else is not a real-time
# run), ODOM_IMAGE (default scoop/odometry:humble), ODOM_IMAGE_2F2L (the
# fast2lamaa methods', default scoop/odometry:2f2l-jazzy), PYTHON (the scoop env's
# python, for make_config.py), START_WAIT (s the method gets to start, 8),
# DRAIN (s after the bag for the last outputs, 5).
#
# Writes into <run dir>: run.json and config/ (make_config.py), trajectory.tum
# (the method's estimate, of run.json's tum_frame, on the bag's clock: give it
# to datasets/swarm_slam_lidar.py --odom-tum <machine> <tum> <tum_frame>),
# messages.csv, resources.csv, host.json, probe.json, summary.json
# (slam/monitor/summarize.py), accuracy.json (score.py: ATE, RPE, drift against
# stage 04's reference, also in summary.json), status.json, logs/.
set -euo pipefail
if [ $# -lt 4 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then sed -n '2,26p' "$0"; exit 1; fi
HERE=$(dirname "$(realpath "$0")")
MONITOR=$(realpath "$HERE/../monitor")
METHOD=$1; BAG=$(realpath "$2"); MACHINE=$3; RUN=$(realpath -m "$4"); shift 4
IMAGE=${ODOM_IMAGE:-scoop/odometry:humble}
case $METHOD in fast2lamaa*) IMAGE=${ODOM_IMAGE_2F2L:-scoop/odometry:2f2l-jazzy};; esac

"${PYTHON:-python3}" "$HERE/make_config.py" "$METHOD" "$BAG" "$MACHINE" "$RUN" "$@"

GPU=()
if command -v nvidia-smi >/dev/null 2>&1 && docker info 2>/dev/null | grep -qi nvidia; then
    GPU=(--gpus all)                   # for the GPU columns of resources.csv
fi
TTY=(); [ -t 0 ] && TTY=(-t)
# --network none: the method, the player and the probe talk over loopback and
# shared memory only, and nothing else on the host's ROS domain hears them
docker run --rm -i "${TTY[@]}" "${GPU[@]}" --network none --shm-size 4g \
    -v "$BAG":/bag:ro -v "$RUN":/work -v "$MONITOR":/opt/monitor:ro -v "$HERE":/opt/odometry:ro \
    -e OWNER="$(id -u):$(id -g)" -e RATE="${RATE:-1.0}" \
    -e START_WAIT="${START_WAIT:-8}" -e DRAIN="${DRAIN:-5}" \
    "$IMAGE" bash /opt/odometry/inside.sh

# accuracy against stage 04's reference (score.py; ATE, RPE, drift, coverage, jumps)
"${PYTHON:-python3}" "$HERE/score.py" "$RUN" || echo "(no accuracy: see above; score.py $RUN --ref <TUM> to give the reference)"

#!/bin/bash
# Replay the bag and the method side by side, record the trajectory, stop when
# the bag stops. Deliberately plain: a benchmark harness that is hard to read is
# a benchmark whose runs cannot be audited.
set -euo pipefail
source /opt/ros/humble/setup.bash

OUT=${OUT:-/out}
mkdir -p "$OUT"
: "${SLAM_TOPICS:?SLAM_TOPICS is required}"
: "${SLAM_POSE_TOPIC:?the image must declare which topic carries the estimate}"

echo "topics : $SLAM_TOPICS"
echo "rate   : ${SLAM_RATE:-1.0}"
echo "params : ${SLAM_PARAMS:-{}}"

python3 /opt/slambench/record_tum.py \
    --pose-topic "$SLAM_POSE_TOPIC" \
    --map-topic "${SLAM_MAP_TOPIC:-}" \
    --out "$OUT" &
REC=$!

eval "${SLAM_LAUNCH:?the image must declare SLAM_LAUNCH}" &
SLAM=$!

sleep 5   # let the method's nodes come up before the first message arrives
ros2 bag play /bag --clock -r "${SLAM_RATE:-1.0}" --topics ${SLAM_TOPICS//,/ }

# The bag is done; give the method a moment to flush its last optimisation.
sleep 10
kill -INT "$SLAM" 2>/dev/null || true
wait "$SLAM" 2>/dev/null || true
kill -INT "$REC" 2>/dev/null || true
wait "$REC" 2>/dev/null || true

test -s "$OUT/trajectory.tum" || { echo "no trajectory written" >&2; exit 3; }
wc -l "$OUT/trajectory.tum"

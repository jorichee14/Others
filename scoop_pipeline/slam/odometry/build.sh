#!/bin/bash
# Build the odometry image (Dockerfile next to this script): scoop/odometry:humble.
#
#   scoop_pipeline/slam/odometry/build.sh [extra docker build args, e.g. --no-cache]
#
# About 20-40 min on 8+ cores the first time; 3-4 GB.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
IMAGE=${ODOM_IMAGE:-scoop/odometry:humble}
docker build -t "$IMAGE" -f "$HERE/Dockerfile" "$@" "$HERE"
docker run --rm "$IMAGE" cat /opt/versions.txt

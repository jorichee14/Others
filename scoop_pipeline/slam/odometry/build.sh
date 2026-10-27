#!/bin/bash
# Build an odometry image (a Dockerfile next to this script).
#
#   scoop_pipeline/slam/odometry/build.sh [2f2l|rtabmap] [extra docker build args, e.g. --no-cache]
#
#   (none)  Dockerfile       -> scoop/odometry:humble (KISS-ICP, RKO-LIO, FAST-LIO2,
#                               OpenVINS, RTAB-Map); about 20-40 min on 8+ cores, 3-4 GB
#   2f2l    Dockerfile.2f2l  -> scoop/odometry:2f2l-jazzy (2Fast-2Lamaa); about 10-20 min
#   rtabmap Dockerfile.rtabmap -> scoop/odometry:rtabmap (RTAB-Map's SLAM node); a few min
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
if [ "${1:-}" = "2f2l" ]; then
    shift
    IMAGE=${ODOM_IMAGE_2F2L:-scoop/odometry:2f2l-jazzy}; FILE=Dockerfile.2f2l
elif [ "${1:-}" = "rtabmap" ]; then
    shift
    IMAGE=${ODOM_IMAGE_RTAB:-scoop/odometry:rtabmap}; FILE=Dockerfile.rtabmap
else
    IMAGE=${ODOM_IMAGE:-scoop/odometry:humble}; FILE=Dockerfile
fi
docker build -t "$IMAGE" -f "$HERE/$FILE" "$@" "$HERE"
docker run --rm "$IMAGE" cat /opt/versions.txt

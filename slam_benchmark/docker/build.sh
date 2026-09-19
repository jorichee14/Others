#!/bin/bash
# Build every image, in dependency order, every time.
#
# The method images are FROM slambench/base:humble, and Docker snapshots the
# base's layers into them at build time. Rebuilding the base therefore changes
# NOTHING in an already-built method image -- it keeps running the entrypoint
# and recorder it was built with. That cost a full afternoon: a fix to the
# recorder was rebuilt into base, and every run kept using rtabmap's stale copy.
# So there is one build command, it builds everything, and "rebuild" never means
# "the one I think changed".
set -euo pipefail
cd "$(dirname "$0")/.."

build() { echo "=== $2"; docker build -f "docker/$1" -t "$2" . ; }

build Dockerfile.base     slambench/base:humble
build Dockerfile.kiss-icp slambench/kiss-icp:humble
build Dockerfile.rtabmap  slambench/rtabmap:humble

# GPU image, ~10 GB and a CUDA compile; opt in.
if [[ "${1:-}" == "--all" ]]; then
    build Dockerfile.mast3r-slam slambench/mast3r-slam:humble
else
    echo "(skipped slambench/mast3r-slam:humble -- pass --all to include it)"
fi

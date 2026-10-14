#!/bin/bash
# Build isaac_ros_visual_slam on mobile_2 (run inside the container).
#
# The container is started with --rm, so anything installed to /usr/local is
# gone after a restart. Everything this needs therefore lives in the workspace,
# which is a host mount and survives: magic_enum is installed to
# $WS/.local/magic_enum (no sudo), and the build output goes to $WS/install.
# Once built, install/ is all that is needed to RUN; this only matters when you
# rebuild after a restart, and it is safe to run again.
#
#   ./build_vslam.sh                 # magic_enum if missing, then colcon build
#   ./build_vslam.sh isaac_ros_nitros  # build up to another package
set -eo pipefail

WS=${WS:-/workspaces/isaac_ros-dev}
PKG=${1:-isaac_ros_visual_slam}
PREFIX=$WS/.local/magic_enum

if [ ! -f "$PREFIX/include/magic_enum.hpp" ]; then
    echo "installing magic_enum into $PREFIX"
    rm -rf /tmp/magic_enum
    git clone --depth 1 https://github.com/Neargye/magic_enum.git /tmp/magic_enum
    cmake -S /tmp/magic_enum -B /tmp/magic_enum/build -DCMAKE_INSTALL_PREFIX="$PREFIX" \
          -DMAGIC_ENUM_OPT_BUILD_EXAMPLES=OFF -DMAGIC_ENUM_OPT_BUILD_TESTS=OFF
    cmake --install /tmp/magic_enum/build
    # GXF does `#include "magic_enum.hpp"`; newer magic_enum nests the header.
    if [ ! -f "$PREFIX/include/magic_enum.hpp" ]; then
        ln -s "$(find "$PREFIX/include" -name magic_enum.hpp | head -1)" "$PREFIX/include/magic_enum.hpp"
    fi
fi

cd "$WS"
export CPATH="$PREFIX/include${CPATH:+:$CPATH}"
colcon build --packages-up-to "$PKG" --symlink-install \
    --cmake-args -DCMAKE_PREFIX_PATH="$PREFIX"
echo
echo "built. now: source $WS/install/setup.bash"

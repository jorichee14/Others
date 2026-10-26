# The odometry methods of PLAN.md in one ROS 2 Humble image: KISS-ICP,
# RKO-LIO, FAST-LIO2, OpenVINS, RTAB-Map's rgbd_odometry. FAST-LIO2 includes
# Livox's CustomMsg, which it never receives here (Ouster clouds): livox_msgs/
# is a package of those two messages under the driver's name, in place of the
# driver and the Livox SDK.
#
#   scoop_pipeline/slam/odometry/build.sh        (docker build, tag scoop/odometry:humble)
#
# Every source is pinned to a commit or tag below; /opt/versions.txt records
# them and the apt version of RTAB-Map. The harness (inside.sh, the monitor)
# is not baked in: run_odometry.sh mounts it, so a fix to it needs no rebuild.
ARG BASE=ros:humble-ros-base
FROM ${BASE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

ARG FAST_LIO=cc3c9e8b1469b33b8072764d6db2348ed6dc597f
ARG KISS_ICP=v1.3.0
ARG RKO_LIO=04566e598f700bc7d5efcfcf3a83327ce842f4f7
ARG OPENVINS=v2.7

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake python3-dev python3-colcon-common-extensions python3-rosdep \
        ros-humble-rosbag2-storage-mcap \
        ros-humble-rtabmap-odom \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws/src
RUN fetch() { git clone -q "$1" "$2" && git -C "$2" checkout -q "$3" \
                && git -C "$2" submodule update -q --init --recursive; } \
    && fetch https://github.com/hku-mars/FAST_LIO.git FAST_LIO ${FAST_LIO} \
    && fetch https://github.com/PRBonn/kiss-icp.git kiss-icp ${KISS_ICP} \
    && touch kiss-icp/cpp/COLCON_IGNORE kiss-icp/python/COLCON_IGNORE \
    && fetch https://github.com/PRBonn/rko_lio.git rko_lio ${RKO_LIO} \
    && fetch https://github.com/rpng/open_vins.git open_vins ${OPENVINS} \
    && for d in */; do echo "${d%/} $(git -C "$d" rev-parse HEAD)"; done > /opt/versions.txt
COPY livox_msgs /ws/src/livox_ros_driver2

RUN apt-get update \
    && (rosdep init 2>/dev/null || true) && rosdep update --rosdistro humble \
    && rosdep install --from-paths /ws/src --ignore-src -y -r --rosdistro humble \
    && rm -rf /var/lib/apt/lists/*

# --continue-on-error: one failing package does not hide the others' errors
WORKDIR /ws
RUN source /opt/ros/humble/setup.bash \
    && colcon build --merge-install --continue-on-error --event-handlers console_cohesion+ \
         --packages-up-to fast_lio kiss_icp rko_lio ov_msckf \
         --cmake-args -DCMAKE_BUILD_TYPE=Release -DRKO_LIO_FETCH_CONTENT_DEPS=ON \
    && rm -rf build log \
    && dpkg-query -W ros-humble-rtabmap-odom ros-humble-rtabmap >> /opt/versions.txt

# rgbd_odometry's library path, and the workspace, for every shell
RUN echo "source /opt/ros/humble/setup.bash && source /ws/install/setup.bash" >> /root/.bashrc

# The odometry methods of PLAN.md in one ROS 2 Humble image: KISS-ICP,
# RKO-LIO, FAST-LIO2 (with Livox-SDK2 and livox_ros_driver2, which it does not
# run but cannot compile without), OpenVINS, RTAB-Map's rgbd_odometry.
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

ARG LIVOX_SDK2=v1.2.5
ARG LIVOX_DRIVER2=21445540f0d100dc86a7e6df312dd70bbdb4afdf
ARG FAST_LIO=cc3c9e8b1469b33b8072764d6db2348ed6dc597f
ARG KISS_ICP=v1.3.0
ARG RKO_LIO=04566e598f700bc7d5efcfcf3a83327ce842f4f7
ARG OPENVINS=v2.7

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake python3-dev python3-colcon-common-extensions python3-rosdep \
        libapr1-dev \
        ros-humble-rosbag2-storage-mcap \
        ros-humble-rtabmap-odom \
    && rm -rf /var/lib/apt/lists/*

# Livox-SDK2: livox_ros_driver2 links it, FAST_LIO includes the driver's CustomMsg
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git /tmp/livox_sdk2 \
    && git -C /tmp/livox_sdk2 checkout -q ${LIVOX_SDK2} \
    && cmake -S /tmp/livox_sdk2 -B /tmp/livox_sdk2/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/livox_sdk2/build -j"$(nproc)" \
    && cmake --install /tmp/livox_sdk2/build \
    && rm -rf /tmp/livox_sdk2

WORKDIR /ws/src
RUN fetch() { git clone -q "$1" "$2" && git -C "$2" checkout -q "$3" \
                && git -C "$2" submodule update -q --init --recursive; } \
    && fetch https://github.com/Livox-SDK/livox_ros_driver2.git livox_ros_driver2 ${LIVOX_DRIVER2} \
    && cp livox_ros_driver2/package_ROS2.xml livox_ros_driver2/package.xml \
    && fetch https://github.com/hku-mars/FAST_LIO.git FAST_LIO ${FAST_LIO} \
    && fetch https://github.com/PRBonn/kiss-icp.git kiss-icp ${KISS_ICP} \
    && touch kiss-icp/cpp/COLCON_IGNORE kiss-icp/python/COLCON_IGNORE \
    && fetch https://github.com/PRBonn/rko_lio.git rko_lio ${RKO_LIO} \
    && fetch https://github.com/rpng/open_vins.git open_vins ${OPENVINS} \
    && for d in */; do echo "${d%/} $(git -C "$d" rev-parse HEAD)"; done > /opt/versions.txt

RUN apt-get update \
    && (rosdep init 2>/dev/null || true) && rosdep update --rosdistro humble \
    && rosdep install --from-paths /ws/src --ignore-src -y -r --rosdistro humble \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
RUN source /opt/ros/humble/setup.bash \
    && colcon build --merge-install \
         --packages-up-to fast_lio kiss_icp rko_lio ov_msckf \
         --cmake-args -DCMAKE_BUILD_TYPE=Release \
                      -DROS_EDITION=ROS2 -DHUMBLE_ROS=humble \
                      -DRKO_LIO_FETCH_CONTENT_DEPS=ON \
    && rm -rf build log \
    && dpkg-query -W ros-humble-rtabmap-odom ros-humble-rtabmap >> /opt/versions.txt

# rgbd_odometry's library path, and the workspace, for every shell
RUN echo "source /opt/ros/humble/setup.bash && source /ws/install/setup.bash" >> /root/.bashrc

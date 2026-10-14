# Reference copy of isaac_ros_visual_slam_realsense.launch.py on mobile_2.
# It starts BOTH the RealSense node and Isaac VSLAM, under namespace mobile_2.
# Edit it on the robot; keep this copy in step so the camera settings are recorded.
# SPDX-License-Identifier: Apache-2.0 (NVIDIA original, modified)
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Launch file which brings up visual slam node configured for RealSense."""
    realsense_camera_node = Node(
        name='camera',
        namespace='mobile_2',
        package='realsense2_camera',
        executable='realsense2_camera_node',
        parameters=[{
            'enable_infra1': True,
            'enable_infra2': True,
            'enable_color': LaunchConfiguration('enable_color'),
            'enable_depth': LaunchConfiguration('enable_depth'),
            'depth_module.emitter_enabled': 0,
            'depth_module.profile': '640x480x30',
            'rgb_camera.profile': '640x480x30',   # realsense-ros 4.51: no color_profile/infra_profile
            'align_depth.enable': False,
            'pointcloud.enable': False,
            'enable_gyro': True,
            'enable_accel': True,
            'gyro_fps': 200,
            'accel_fps': 200,
            'unite_imu_method': 2,
        }],
    )
    visual_slam_node = ComposableNode(
        name='visual_slam_node',
        namespace="mobile_2",
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        parameters=[{
            'enable_image_denoising': False,
            'rectified_images': True,
            'enable_imu_fusion': True,
            'gyro_noise_density': 0.000244,
            'gyro_random_walk': 0.000019393,
            'accel_noise_density': 0.001862,
            'accel_random_walk': 0.003,
            'calibration_frequency': 200.0,
            'image_jitter_threshold_ms': 22.00,
            'base_frame': 'camera_link',
            'imu_frame': 'camera_gyro_optical_frame',
            'map_frame': 'map_realsense',
            'odom_frame': 'odom_realsense',
            'enable_slam_visualization': True,
            'enable_landmarks_view': True,
            'enable_observations_view': True,
            'camera_optical_frames': [
                'camera_infra1_optical_frame',
                'camera_infra2_optical_frame',
            ],
            'use_vslam_gxf_memory_pool': False,
            'disable_gxf_memory_pool': True
        }],
        remappings=[
            ('visual_slam/image_0', '/mobile_2/infra1/image_rect_raw'),
            ('visual_slam/camera_info_0', '/mobile_2/infra1/camera_info'),
            ('visual_slam/image_1', '/mobile_2/infra2/image_rect_raw'),
            ('visual_slam/camera_info_1', '/mobile_2/infra2/camera_info'),
            ('visual_slam/imu', '/mobile_2/imu'),
        ],
    )
    visual_slam_launch_container = ComposableNodeContainer(
        name='visual_slam_launch_container',
        namespace='mobile_2',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[visual_slam_node],
        output='screen',
    )
    return launch.LaunchDescription([
        # VSLAM only needs infra1/infra2 + IMU. Colour and depth are recorded
        # for the dataset; turn them off if the camera or USB link struggles:
        #   ros2 launch ... enable_depth:=false enable_color:=false
        DeclareLaunchArgument('enable_color', default_value='true'),
        DeclareLaunchArgument('enable_depth', default_value='true'),
        visual_slam_launch_container,
        realsense_camera_node,
    ])

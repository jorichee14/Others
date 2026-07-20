#!/usr/bin/env python3
"""
Launch: merge N radar clouds into one shared frame using saved extrinsics.

    ros2 launch wicoms_utils radar_merge.launch.py

No calibration rig. Point `extrinsic_files` at the JSON each radar's
`radar_camera_calib.py` run produced, list the matching live topics, and pick a
`target_frame` (empty → the shared camera/parent frame). Edit the paths below.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = {
        # index-matched: file[i] calibrates the radar publishing topic[i]
        'extrinsic_files': [
            '/root/extrinsics/extrinsic_zed_left__radar1.json',
            '/root/extrinsics/extrinsic_zed_left__radar2.json',
        ],
        'radar_topics': [
            '/radar1/radar/points_all',
            '/radar2/radar/points_all',
        ],

        # '' → shared camera/parent frame from the first file;
        # or set to a radar link (e.g. 'radar1_link') to merge in that radar's frame.
        'target_frame': 'zed_left_camera_optical_frame',

        # IWR6843ISK points_all field names (SNR is 'intensity')
        'pc_field_x': 'x', 'pc_field_y': 'y', 'pc_field_z': 'z',
        'pc_field_snr': 'intensity', 'pc_field_doppler': 'doppler',

        'output_topic': '/radar_merged/points',
        'publish_rate_hz': 15.0,   # 0 → publish on every incoming cloud
        'max_age_s': 0.25,         # drop a radar's cloud if older than this
        'add_source_field': True,  # tag each point with its radar index
        'publish_tf': True,        # static parent→radar_i TF for RViz
    }

    return LaunchDescription([
        Node(
            package='wicoms_utils',
            executable='radar_merge',
            name='radar_merge',
            output='screen',
            parameters=[params],
        ),
    ])

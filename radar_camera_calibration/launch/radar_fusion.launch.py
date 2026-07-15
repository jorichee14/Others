#!/usr/bin/env python3
"""
Launch: radar_fusion_reflector — fuse two calibrated radars into a smooth,
maneuvering-target track of the corner reflector, drawn on the ZED image.

    ros2 launch wicoms_utils radar_fusion.launch.py

EVERY node parameter is listed with its default and a note (full reference). The
r1_*/r2_* extrinsics default to this rig's solved values; replace them with your
own from the calibration output (or the sessions/*.md transforms). All values are
typed here so the CLI YAML-boolean trap can't happen.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = {
        # ── camera + radar topics ──
        'image_topic': '/zed/zed_node/left/image_rect_color',
        'info_topic':  '/zed/zed_node/left/camera_info',
        'radar1_topic': '/radar1/radar/points_all',
        'radar2_topic': '/radar2/radar/points_all',
        'pc_field_x': 'x', 'pc_field_y': 'y', 'pc_field_z': 'z',
        'pc_field_snr': 'intensity',

        # ── per-radar extrinsics T_cam_radar (REPLACE with your solved values) ──
        'r1_t_xyz': [0.2218, -0.0067, -0.1721],
        'r1_quat_xyzw': [-0.5345, 0.5853, -0.4196, -0.4424],
        'r2_t_xyz': [-0.0999, -0.0124, -0.0011],
        'r2_quat_xyzw': [0.7882, -0.0406, 0.6121, 0.0499],
        'r1_range_scale': 1.039, 'r2_range_scale': 1.026,   # per-radar ingest range scale

        # ── radar noise model (same chip; drives the fusion weighting) ──
        'sigma_range_m': 0.05, 'sigma_az_deg': 3.0, 'sigma_el_deg': 8.0,

        # ── reflector selection ──
        'min_range': 0.3, 'max_range': 6.0, 'min_snr': 100.0,
        'select_radius_m': 0.5,             # SNR-weighted blob-centroid radius around the prediction (m)

        # ── maneuvering-target tracker ──
        'process_accel': 1.0,               # quiet-state accel std (m/s²): smoothing floor when still
        'maneuver_gain': 3.0,               # speed→accel-noise gain (1/s): higher = snappier on a fast reflector
        'maneuver_deadband': 0.15,          # ignore speed below this (m/s): keeps a still reflector smooth
        'innov_gate_chi2': 11.35,           # Mahalanobis gate, 3-DOF 99% = 11.34
        'adapt_window': 12,                 # recent innovations used to inflate R per radar
        'adapt_max_scale': 4.0,             # cap adaptive R inflation at this × model
        'reinit_gap_s': 1.0,                # hard-reinit the track after this gap with no update (s)
        'coast_s': 0.5,                     # keep drawing the tracked point this long after last update (s)
        'trail_len': 60, 'trail_s': 3.0,    # motion-trace length / max age (s)

        # ── output / display ──
        'publish_point': True,              # publish tracked reflector on /radar_fusion/reflector
        'debug_image_topic': '/radar_fusion/debug_image',
        'show_window': True,
    }

    return LaunchDescription([
        Node(
            package='wicoms_utils',
            executable='radar_fusion_reflector',
            name='radar_fusion_reflector',
            output='screen',
            parameters=[params],
        ),
    ])

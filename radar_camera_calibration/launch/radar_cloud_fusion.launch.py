#!/usr/bin/env python3
"""
Launch: radar_cloud_fusion — fuse the two calibrated radars' FULL point clouds
into a single denser/less-noisy scene cloud in the camera frame, project it onto
the ZED image, and validate the fusion live.

    ros2 launch wicoms_utils radar_cloud_fusion.launch.py

Every node parameter is listed with its default and a note (full reference). The
r1_*/r2_* extrinsics default to this rig's FINAL solved values
(sessions/2026-07-22_zed_radar1_radar2_final.md); replace them with your own from
the calibration output. All values are typed here so the CLI YAML-boolean trap
can't happen.
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
        'camera_frame': 'zed_left_camera_optical_frame',

        # ── per-radar extrinsics T_cam_radar (FINAL values; REPLACE if re-solved) ──
        'r1_t_xyz': [0.2368, 0.0190, -0.0542],
        'r1_quat_xyzw': [-0.4995, 0.6007, -0.4224, -0.4596],
        'r2_t_xyz': [-0.1194, -0.0096, -0.0157],
        'r2_quat_xyzw': [0.7572, 0.0539, 0.6506, -0.0217],
        'r1_range_scale': 0.958, 'r2_range_scale': 0.967,

        # ── radar noise model (same chip; sets each point's anisotropic covariance) ──
        'sigma_range_m': 0.05, 'sigma_az_deg': 3.0, 'sigma_el_deg': 8.0,

        # ── extrinsic 1σ (from the calibration) — folded into each point's cov so
        #    the cross-radar χ² is calibrated (≈3). Without it, χ² reads ~2× high. ──
        'r1_ext_sigma_t_m': 0.035, 'r1_ext_sigma_rot_deg': 4.0,
        'r2_ext_sigma_t_m': 0.030, 'r2_ext_sigma_rot_deg': 3.5,

        # ── gating ──
        'min_range': 0.3, 'max_range': 8.0, 'min_snr': 0.0,
        'max_points': 400,                  # cap per cloud before O(n·m) association

        # ── fusion ──
        'assoc_gate_chi2': 7.815,           # 3-DOF 95% = 7.815 (99% = 11.345)
        'valid_chi2_max': 6.0,              # VALID if windowed mean χ² ≤ this (2× ideal)
        'stats_window': 300,               # report χ²/shrink over the last N matches
        'require_both': False,              # True → publish ONLY 2-radar confirmed points
        'sync_s': 0.15,                     # both clouds must be within this to cross-fuse
        'accum_s': 0.0,                     # >0: temporal accumulate+voxel merge (STATIC scenes)
        'voxel_m': 0.10,                    # voxel size for the temporal merge

        # ── display / output ──
        'draw_ellipse': True,              # per-point 1σ projected uncertainty ellipse
        'point_radius': 4,
        'fused_cloud_topic': '/radar_fusion/cloud',
        'debug_image_topic': '/radar_fusion/cloud_image',
        'publish_cloud': True,
        'report_every_s': 2.0,             # cadence of the [validate] log line
        'show_window': True,
    }

    return LaunchDescription([
        Node(
            package='wicoms_utils',
            executable='radar_cloud_fusion',
            name='radar_cloud_fusion',
            output='screen',
            parameters=[params],
        ),
    ])

#!/usr/bin/env python3
"""
Launch: radar_camera_calib for IWR6843ISK (3D People-Counting) + ZED.

    ros2 launch wicoms_utils iwr6843isk.launch.py

All parameters are typed here, so the YAML-boolean trap that bites the CLI
(`-p pc_field_y:=y` → parsed as `true`) can't happen. Edit the dict below for
your board, priors, and topics. Priors are pre-wired but OFF by default — flip
use_extrinsic_prior / offset_prior_sigma_m to use them.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = {
        # ── camera ──
        'image_topic': '/zed/zed_node/left/image_rect_color',
        'info_topic':  '/zed/zed_node/left/camera_info',

        # ── board (EDIT to match your printed ChArUco) ──
        'squares_x': 9, 'squares_y': 7,
        'square_len': 0.020, 'marker_len': 0.015,
        'dictionary': 'DICT_4X4_50',
        'min_corners': 6,            # small board at close range → allow fewer corners
        'max_reproj_px': 1.5,
        # A planar board has two poses that project almost identically. Drop the
        # frame when IPPE's two hypotheses are within this reprojection ratio —
        # otherwise ~10% of poses past ~1.5 m come back flipped, and every degree
        # of board error is multiplied by the apex offset. 0 disables the guard.
        'pnp_ambiguity_ratio': 1.2,
        'rectified_input': 'auto',   # 'auto' reads P (not k/d) on a *_rect_* topic

        # ── radar (IWR6843ISK / points_all: x,y,z,doppler,intensity) ──
        'radar_topic': '/radar1/radar/points_all',
        'pc_field_x': 'x', 'pc_field_y': 'y', 'pc_field_z': 'z',
        'pc_field_snr': 'intensity',      # SNR field is named 'intensity'
        'pc_field_doppler': 'doppler',
        'select_by': 'snr',
        'min_range': 0.5, 'max_range': 2.5,
        'range_gate_margin_m': 0.5,       # gate radar around the camera board-distance
        'gate_radius': 0.4,               # tight 3-D gate once an extrinsic/prior exists
        'max_abs_doppler': -1.0,          # static reflector ≈ 0 Doppler → don't gate it out

        # ── radar noise model (drives ML weighting; el is weak on the ISK) ──
        'sigma_range_m': 0.03,
        'sigma_az_deg': 1.5,
        'sigma_el_deg': 10.0,
        'radar_range_bias_m': 0.0,        # already compensated in-chip
        'force_2d_radar': False,

        # ── apex offset (board frame) + OFFSET PRIOR ──
        #   measured well → tight sigma pins it; unknown → 0 + loose sigma (0.10)
        'reflector_offset_x': 0.0,
        'reflector_offset_y': 0.0,
        'reflector_offset_z': 0.0,
        'solve_offset': True,
        'offset_prior_sigma_m': 0.10,

        # ── EXTRINSIC PRIOR (opt-in): rough known radar-in-camera pose ──
        #   turn on and fill from a tape measure / CAD / a first rough solve.
        #   USE THE QUATERNION. This rig's extrinsic is a ~90° frame swap, which
        #   is exactly the gimbal-lock singularity of the rpy convention: there
        #   the euler triple is not unique and a 1° change rewrites it by 90°, so
        #   pasting a printed rpy back in as a prior injects a real error. Every
        #   solve prints a ready-to-paste prior_quat_xyzw line.
        'use_extrinsic_prior': False,
        'prior_t_xyz': [0.0, 0.0, 0.0],       # radar position in camera frame (m)
        'prior_quat_xyzw': [0.0, 0.0, 0.0, 0.0],   # zero norm = unset → rpy used
        'prior_rpy_deg': [0.0, 0.0, 0.0],     # legacy/rough entry only
        'prior_t_sigma_m': 0.05,
        'prior_rot_sigma_deg': 10.0,
        # fail the verdict when the data-only and prior-pulled rotations disagree
        # by more than this — a wrong prior fits just as well as a right one, so
        # the residual can never catch it and this is the only thing that can
        'prior_disagree_warn_deg': 5.0,
        'unobs_rot_sigma_deg': 10.0, 'unobs_t_sigma_mm': 100.0,

        # ── capture / strictness ──
        'capture_mode': 'auto',
        'stable_window': 16,
        'stable_std': 0.01,
        'stable_std_radar': 0.04,         # strict: demand a steady radar point
        'min_baseline': 0.10,
        'min_points': 14,
        'min_snr': 150.0,                 # strict: reject weak (mis-associable) returns
        'sync_slop': 0.08,

        # ── validation thresholds ──
        'val_pass_reproj_px': 20.0,
        'val_pass_3d_mm': 150.0,
        'val_pass_bias_mm': 50.0,
        'measured_baseline_m': -1.0,      # >0 → tape-measured |t| check
        'baseline_tol_m': 0.03,

        # ── frames / output / display ──
        'parent_frame': 'zed_left_camera_optical_frame',
        'child_frame': 'radar1_link',
        'camera_name': 'zed_left', 'radar_name': 'radar1',
        'publish_tf': True,
        'debug_image': True,
        'debug_image_topic': '/radar_camera_calib/debug_image',
        'show_window': True,
        'radar_watchdog_s': 3.0,
    }

    return LaunchDescription([
        Node(
            package='wicoms_utils',
            executable='radar_camera_calibration',
            name='radar_camera_calib',
            output='screen',
            parameters=[params],
        ),
    ])

#!/usr/bin/env python3
"""
Launch: radar_camera_calib for IWR6843ISK (3D People-Counting) + ZED.

    ros2 launch wicoms_utils iwr6843isk.launch.py

EVERY node parameter is listed below with its default and a one-line note, so this
doubles as the full parameter reference. All values are typed here, so the
YAML-boolean trap that bites the CLI (`-p pc_field_y:=y` → parsed as `true`) can't
happen. Edit the dict for your board, topics, priors, and strictness.

The values below are a known-good STATIC profile (step-and-settle, diversity HUD,
cluster selection). For the exact radar1 / radar2 configs see
`sessions/2026-07-15_zed_radar1.md` / `..._radar2.md`.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = {
        # ── camera ──
        'image_topic': '/zed/zed_node/left/image_rect_color',
        'info_topic':  '/zed/zed_node/left/camera_info',

        # ── board (EDIT to match your printed ChArUco) ──
        'squares_x': 4, 'squares_y': 4,          # inner grid (squares), not corners
        'square_len': 0.12, 'marker_len': 0.09,  # metres
        'dictionary': 'DICT_4X4_50',
        'min_corners': 4,                        # allow fewer corners on a small/close board
        'max_reproj_px': 1.5,                    # reject a board pose whose PnP reproj exceeds this

        # ── radar topic + fields (IWR6843ISK points_all: x,y,z,doppler,intensity) ──
        'radar_topic': '/radar1/radar/points_all',
        'pc_field_x': 'x', 'pc_field_y': 'y', 'pc_field_z': 'z',
        'pc_field_snr': 'intensity',             # SNR field is named 'intensity' on this driver
        'pc_field_doppler': 'doppler',

        # ── reflector selection ──
        'select_by': 'cluster',                  # 'snr' | 'nearest' | 'cluster'
        'cluster_eps': 0.20,                     # cluster: points within this join one blob (m)
        'min_cluster_size': 1,                   # cluster: min points to accept a blob
        'cluster_apex_radius': 0.40,             # cluster: only points within this of the predicted apex (m)
        'cluster_strict': False,                 # cluster: True → reject capture if no blob near the apex

        # ── range / doppler gating ──
        'min_range': 0.5, 'max_range': 2.5,
        'range_gate_margin_m': 0.5,              # gate radar around the camera board-distance
        'gate_radius': 0.5,                      # 3-D gate once an extrinsic/prior exists (m)
        'max_abs_doppler': -1.0,                 # keep |doppler| ≤ this (−1 = off; static reflector ≈ 0)
        'min_abs_doppler': -1.0,                 # keep |doppler| ≥ this (−1 = off; use for moving-only)
        'use_doppler_consistency': False,        # DYNAMIC: match radial velocity to camera-predicted motion
        'doppler_match_tol': 0.30, 'doppler_sign': 'auto', 'min_motion_mps': 0.05,

        # ── background subtraction (optional) ──
        'bg_accum_frames': 15, 'bg_match_dist': 0.2, 'require_background': False,

        # ── radar range correction (applied at ingest, before everything) ──
        'radar_range_scale': 1.0,                # r' = scale·r + bias   (per-radar; fix scale first)
        'radar_range_bias_m': 0.0,               # constant range offset (m); catches phase-center / chip bias

        # ── radar noise model (drives ML weighting; el is weak on the ISK) ──
        'sigma_range_m': 0.05,
        'sigma_az_deg': 3.0,
        'sigma_el_deg': 8.0,
        'force_2d_radar': False,                 # ignore elevation entirely (auto-detected otherwise)

        # ── robust solver ──
        'huber_f_scale': 1.5,                    # robust-loss knee (σ)
        'reject_sigma': 4.0,                     # drop a match whose RMS-across-axes residual exceeds this (σ)
        'reject_axis_sigma': 0.0,                # opt-in (0=off): also drop if ANY single axis exceeds this (σ) — catches multipath ghosts; try 3.5

        # ── apex offset (board frame) + OFFSET PRIOR ──
        #   measured well → tight sigma pins it; unknown → 0 + loose sigma (0.10)
        'reflector_offset_x': 0.0, 'reflector_offset_y': 0.0, 'reflector_offset_z': 0.0,
        'solve_offset': True, 'offset_prior_sigma_m': 0.05,

        # ── EXTRINSIC PRIOR (opt-in): rough known radar-in-camera pose ──
        #   turn on and fill from a tape measure / CAD / a first rough solve.
        'use_extrinsic_prior': False,
        'prior_t_xyz': [0.0, 0.0, 0.0],          # radar position in camera frame (m)
        'prior_rpy_deg': [0.0, 0.0, 0.0],        # radar orientation (xyz euler, deg)
        'prior_t_sigma_m': 0.05, 'prior_rot_sigma_deg': 10.0,

        # ── capture / strictness ──
        'capture_mode': 'auto',                  # 'auto' | 'manual' (~/capture)
        'stable_window': 12, 'stable_std': 0.02, # per-pose camera stability gate (frames / m)
        'stable_std_radar': 0.10,                # demand a steady radar point (m)
        'min_baseline': 0.15, 'min_points': 25,  # capture spacing (m) / min poses before a solve
        'min_snr': 100.0,                        # strict: reject a pick weaker than this SNR
        'sync_slop': 0.06,                       # image↔radar time-sync tolerance (s)
        'max_sync_dt': -1.0,                     # DYNAMIC: reject a capture whose img/radar Δt exceeds this (s)
        'max_capture_speed': -1.0,               # DYNAMIC: reject a capture while moving faster than this (m/s)

        # ── validation thresholds (VERDICT line) ──
        'val_pass_reproj_px': 20.0, 'val_pass_3d_mm': 150.0, 'val_pass_bias_mm': 50.0,
        'measured_baseline_m': -1.0,             # >0 → tape-measured |t| check
        'baseline_tol_m': 0.03,

        # ── frames / output / display ──
        'parent_frame': 'zed_left_camera_optical_frame',
        'child_frame': 'radar1_link',
        'camera_name': 'zed_left', 'radar_name': 'radar1',
        'output_path': '',                       # '' → auto (./extrinsic_<cam>__<radar>.*); saves yaml/json/session
        'publish_tf': True,
        'debug_image': True, 'debug_image_topic': '/radar_camera_calib/debug_image',
        'show_window': True,
        'show_diversity_hud': True,              # pitch/roll/yaw + az/el/range readiness bars (static)
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

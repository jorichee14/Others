#!/usr/bin/env python3
"""
ARDUCAM radar ↔ camera extrinsic calibration  (raw feed, in-node rectification)
===============================================================================
The SAME step-and-settle pipeline used for the ZED (`radar_camera_calib_static`),
applied to an Arducam. The only difference is the camera feed: an Arducam
publishes a RAW / unrectified image with real lens distortion in `camera_info`,
which breaks ArUco detection and makes reprojection blow up. This profile turns on
**in-node rectification** (`rectify_image:=true`): the node undistorts every frame
from `camera_info`, then runs ALL detection and projection with the rectified K and
ZERO distortion. So the full ZED-proven solver (measurement-space ML, diversity
HUD, cluster/SNR selection, joint apex offset, session auto-save, priors, range-fit
diagnostic) applies to the Arducam completely unchanged — nothing else differs.

Everything about the *process* is identical to the ZED: follow
`CALIBRATION_PROTOCOL.md` (measure the rig offset once, tape the per-radar
extrinsic prior, collect until the HUD is green, tune `radar_range_scale` to a≈1,
save). Only the camera topics / frame change.

Run (fill in your Arducam topics + the usual offset/prior params):

    ros2 run wicoms_utils radar_camera_calib_arducam --ros-args \
      -p image_topic:=/arducam/image_raw \
      -p info_topic:=/arducam/camera_info \
      -p parent_frame:=arducam_optical_frame \
      -p radar_topic:=/radar1/radar/points_all -p pc_field_snr:=intensity \
      -p reflector_offset_x:=0.235 -p reflector_offset_y:=0.57 -p reflector_offset_z:=0.0 \
      -p use_extrinsic_prior:=true -p prior_t_xyz:="[...]" -p prior_rpy_deg:="[...]" \
      -p child_frame:=radar1_link -p radar_name:=radar1
      # ... plus the frozen block from CALIBRATION_PROTOCOL.md

`rectify_alpha`: 0 = keep only valid pixels (zoomed), 1 = keep the whole FoV (with
black borders). If you accidentally point this at an already-rectified feed (D≈0),
rectification auto-disables and it behaves exactly like the ZED static profile.
"""
try:                                             # flat-module or package install layout
    from radar_camera_calib import main as _run
except ImportError:
    from wicoms_utils.radar_camera_calib import main as _run

# Arducam profile: the STATIC step-and-settle preset + in-node rectification for
# the raw/distorted feed. Override topics / frame / priors on the command line.
ARDUCAM_DEFAULTS = {
    'rectify_image': True,             # undistort the raw Arducam feed in-node
    'rectify_alpha': 0.0,              # 0 = crop to valid pixels; 1 = keep full FoV
    'capture_mode': 'auto',            # stability-gated capture (rig held still)
    'use_doppler_consistency': False,  # a still reflector is ~0 doppler
    'min_abs_doppler': -1.0,           # no moving-reflector gate
    'max_abs_doppler': -1.0,
    'max_sync_dt': -1.0,               # stationary → time mismatch is harmless
    'max_capture_speed': -1.0,
    'show_diversity_hud': True,        # pitch/roll/yaw + az/el/range readiness cue
    # Real-time debug on a remote/big feed WITHOUT dropping frames: publish only the
    # JPEG-compressed overlay (~tens of KB) — the board axes + apex show EVERY frame,
    # just small over the wire. View '<debug_image_topic>/compressed' in rqt_image_view.
    # (A raw bgr8 frame is ~MBs and is what stalls the link — so debug_raw is off here.)
    'debug_raw': False,                # don't publish the heavy raw Image stream
    'debug_compressed': True,          # publish the compressed JPEG stream instead
    'debug_jpeg_quality': 40,          # 1–100; raise for crisper, lower for lighter
}


def main():
    _run(ARDUCAM_DEFAULTS)


if __name__ == '__main__':
    main()

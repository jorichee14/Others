#!/usr/bin/env python3
"""
ARDUCAM profile for the radar ↔ LIDAR calibration.

    python3 radar_lidar_calib_arducam.py --ros-args -p radar_topic:=... [...]

Identical to `radar_lidar_calib.py` in every way that matters — the SOLVE never
touches the camera, so the pipeline, gates, coverage HUD and RViz layer are
unchanged. This only presets the camera half:

  * the Arducam publishes a RAW feed with real lens distortion, so
    `rectify_image` is on. Every frame is undistorted from `camera_info` and
    projection then uses the new K with ZERO distortion. Projecting onto a raw
    image is not wrong in itself (projectPoints re-applies D), but the two
    conventions differ by tens of pixels near the frame edge, and the overlay has
    to match whichever image the rest of the pipeline consumes.
  * the topics and the optical frame.

If you point this at an already-rectified feed (D ≈ 0) rectification disables
itself with a log line and it behaves exactly like the plain node.

Everything else — the reflector/lidar detection tuning, which is a property of
your reflector and mount rather than of the camera — stays on the command line.

The camera transform still has to be supplied, either as
`lidar_camera_xyz` / `lidar_camera_quat_xyzw`, or (better, when a
static_transform_publisher is already broadcasting it) with
`camera_transform_from_tf:=true`, which looks it up from the frame the CLOUD
arrives in and so cannot be quoted against the wrong frame or inverted.

An Arducam whose INTRINSICS are not calibrated yet costs nothing here: they feed
the overlay only, never the solve or the composed T_cam_radar. Run with
`show_image_overlay:=false` and verify in RViz until they exist.
"""
try:                                    # flat-module or installed-package layout
    from radar_lidar_calib import main as _run
except ImportError:
    from wicoms_utils.radar_lidar_calib import main as _run

ARDUCAM_DEFAULTS = {
    'image_topic': '/arducam/image_raw',
    'info_topic': '/arducam/camera_info',
    'camera_frame': 'arducam_optical_frame',
    'rectify_image': True,      # RAW feed — undistort in-node
    'rectify_alpha': 0.0,       # 0 = crop to valid pixels, 1 = keep the full FoV
}


def main():
    _run(ARDUCAM_DEFAULTS)


if __name__ == '__main__':
    main()

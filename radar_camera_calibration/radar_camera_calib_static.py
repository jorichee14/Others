#!/usr/bin/env python3
"""
STATIC radar–camera calibration.

Hold the rig STILL at each pose. The reflector is ~0 Doppler, so we do NOT
Doppler-filter; the reflector is isolated by SNR + the camera-range gate, and
each capture is a 16-frame average once the board AND radar are steady. Cleanest
correspondences — best final accuracy — but you must hold still per pose.

Run:  ros2 run wicoms_utils radar_camera_calibration_static
Only overrides vs radar_camera_calib.DEFAULTS live here; -p still overrides these.
"""
try:                                    # core module name may differ per package
    from radar_camera_calib import main
except ImportError:
    from radar_camera_calibration import main

STATIC = {
    'capture_mode': 'auto',      # capture when the pose is held STILL
    'min_abs_doppler': -1.0,     # do NOT require motion (reflector is 0-Doppler)
    'max_abs_doppler': -1.0,     # do NOT reject static returns
    'min_snr': 100.0,            # only strong reflector returns
    'stable_window': 16,         # frames averaged per pose (more = cleaner)
    'stable_std': 0.01,          # board must be still to ~10 mm
    'stable_std_radar': 0.05,    # radar reasonably steady (angular noise is cm-dm)
    'min_baseline': 0.12,        # move ≥12 cm to a new spot between captures
    'min_points': 15,
}


def entry():
    main(STATIC)


if __name__ == '__main__':
    entry()

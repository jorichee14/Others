#!/usr/bin/env python3
"""
DYNAMIC radar–camera calibration.

SWEEP the rig continuously through the FoV — no holding still. The reflector is
MOVING, so we keep only moving returns (min_abs_doppler), which isolates it from
all static clutter automatically. A capture is taken every min_baseline of
movement (3-frame averaged). Far easier to collect; noisier per capture (motion
+ sync + weaker moving returns), so lean on many poses + robust rejection.

Tips: move SLOWLY (keeps the board sharp and the sync error small) and keep the
reflector's opening pointed at the radar so it stays bright.

Run:  ros2 run wicoms_utils radar_camera_calibration_dynamic
Only overrides vs radar_camera_calib.DEFAULTS live here; -p still overrides these.
"""
import os, sys
# Ensure the sibling core module imports whether this file is run directly,
# via `ros2 run`, or as an installed console_script — the core lives next to it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from radar_camera_calib import main
except ImportError:                     # installed as a package submodule
    from radar_camera_calibration import main

DYNAMIC = {
    'capture_mode': 'continuous',  # capture on movement, no stillness needed
    'min_abs_doppler': 0.1,        # keep only MOVING points → drops static clutter
    'max_abs_doppler': -1.0,
    'min_snr': 60.0,               # moving reflector returns are weaker than static
    'min_baseline': 0.08,          # capture every 8 cm of sweep
    'min_points': 20,
    'sync_slop': 0.08,
}


def entry():
    main(DYNAMIC)


if __name__ == '__main__':
    entry()

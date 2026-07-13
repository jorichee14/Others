#!/usr/bin/env python3
"""
STATIC radar ↔ camera extrinsic calibration  (step-and-settle)
==============================================================
Profile of `radar_camera_calib` for a rig held STILL at each pose. This is the
most accurate way to calibrate a hand-held rig: move, pause ~0.5 s, let the tool
capture during the settle, move on. Because both sensors are stationary at
capture time, there is NO moving-correspondence error (no time-mismatch, no
tracker lag) — the two failure modes that plague continuous sweeping.

The catch with a single reflector is that ROTATION is only observable if the
pose set is DIVERSE. You will typically get good translation but bad rotation
unless you deliberately vary the geometry. This script therefore turns the live
**diversity HUD** on by default (`show_diversity_hud:=true`): six bars —

    board  PITCH / ROLL / YAW spread   → makes the apex OFFSET observable
    radar  AZ / EL / RANGE spread      → the lever arm that makes the
                                         EXTRINSIC ROTATION observable

Each bar turns green when it crosses the target that makes rotation (and, via
the offset, translation) well-determined. When all bars are green and the
measured `rot 1σ` is small, the HUD reads **READY — rotation observable**.
Practically: tilt the board in pitch and yaw, roll it, AND move the rig
near↔far, left↔right, up↔down. Watch the red bars fill green.

Everything else (solver, gating, validation, save/TF) is identical to the shared
node — see `radar_camera_calib.py` / README. Run with:

    ros2 run wicoms_utils radar_camera_calib_static -p ...params...
"""
try:                                             # flat-module or package install layout
    from radar_camera_calib import main as _run
except ImportError:
    from wicoms_utils.radar_camera_calib import main as _run

# Static profile: stillness-gated auto capture, no Doppler/motion machinery,
# diversity HUD on. Override any of these on the command line as usual.
STATIC_DEFAULTS = {
    'capture_mode': 'auto',            # stability-gated capture (rig held still)
    'use_doppler_consistency': False,  # not needed: a still reflector is ~0 doppler
    'min_abs_doppler': -1.0,           # no moving-reflector gate
    'max_abs_doppler': -1.0,
    'max_sync_dt': -1.0,               # stationary → time mismatch is harmless
    'max_capture_speed': -1.0,
    'show_diversity_hud': True,        # the pitch/roll/yaw + az/el/range readiness cue
}


def main():
    _run(STATIC_DEFAULTS)


if __name__ == '__main__':
    main()

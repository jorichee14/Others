#!/usr/bin/env python3
"""
DYNAMIC radar ↔ camera extrinsic calibration  (continuous sweep)
================================================================
Profile of `radar_camera_calib` for a rig you keep MOVING (never stopping).
Sweeping is the easy way to collect wide pose diversity, but a hand-held moving
rig introduces three correspondence errors a static capture never has:

  1. TIME MISMATCH   — camera and radar are sampled at slightly different instants;
                       at speed v and sync gap Δt the "same" point is v·Δt apart.
  2. TRACKER LAG     — the people-counting firmware smooths a moving target, so
                       its reported position lags the truth (scales with speed).
  3. FEATURE AMBIGUITY — your hand/arm/body are strong MOVING reflectors, so a
                       static |doppler|≈0 gate cannot isolate the trihedral.

This profile fights all three by turning the motion into the discriminator:

  • use_doppler_consistency — the reflector is rigidly tied to the board, so its
    radar radial velocity must equal the camera's d|range|/dt. Keeps only radar
    points whose Doppler matches the camera-predicted value; rejects moving
    clutter that a static gate can't. (See _select_radar / README.)
  • max_sync_dt      — drop image/radar pairs whose stamps differ too much.
  • max_capture_speed — skip captures while you move too fast (bounds errors 1–2).

If your solves are still noisy, prefer `radar_camera_calib_static` (step-and-
settle) — pausing removes errors 1–2 entirely and is usually more accurate.

Everything else (solver, gating, validation, save/TF) is identical to the shared
node — see `radar_camera_calib.py` / README. Run with:

    ros2 run wicoms_utils radar_camera_calib_dynamic -p ...params...
"""
try:                                             # flat-module or package install layout
    from radar_camera_calib import main as _run
except ImportError:
    from wicoms_utils.radar_camera_calib import main as _run

# Dynamic profile: continuous capture on movement + Doppler↔motion consistency
# + sync/speed guards. Tune tolerances on the command line for your radar.
DYNAMIC_DEFAULTS = {
    'capture_mode': 'continuous',      # capture every min_baseline of movement
    'use_doppler_consistency': True,   # isolate the reflector by its radial velocity
    'doppler_match_tol': 0.30,         # m/s
    'doppler_sign': 'auto',
    'min_motion_mps': 0.05,
    'max_sync_dt': 0.03,               # s; drop poorly time-aligned pairs
    'max_capture_speed': 0.25,         # m/s; skip captures while moving faster
    'min_baseline': 0.10,              # m between captures
    'show_diversity_hud': False,       # diversity accrues fast while sweeping; opt in if wanted
}


def main():
    _run(DYNAMIC_DEFAULTS)


if __name__ == '__main__':
    main()

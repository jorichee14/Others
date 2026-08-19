#!/usr/bin/env python3
"""
Where to put the tripod — turns an extrinsic into a list of physical positions.

    python3 plan_captures.py --quat x,y,z,w --xyz x,y,z [--lidar-fov 22.5] [--n 15]

The coverage HUD tells you a bar is red; this tells you where to stand to fix it.
It searches placements that fill all nine az x el cells AND stay inside the
lidar's vertical field of view, then reports each one as (forward, side, height)
relative to the RADAR — numbers you can pace out and measure, not angles.

Two rig facts drive the whole answer and are easy to get backwards:

  * The radar's boresight is usually not level. Elevation is measured from that
    tilted axis, so "same height as the radar" is NOT elevation zero: with a
    nose-up radar the el=0 line RISES with range, and past a few metres it is
    above head height and unreachable. Close range is the only place both signs
    of elevation are available.
  * The LIDAR is what has to see the reflector, and it sits at its own height.
    Whichever way it is offset from the radar, that side of the field of view
    runs out first — with the lidar above the radar, LOW shots leave the FoV
    before high ones do, which is the opposite of what the radar geometry alone
    would suggest.

Needing an extrinsic to plan captures for solving the extrinsic is not circular:
a rough one is enough, since the plan only has to be approximately right, and
the answer is insensitive to the input (a 5 deg error moves the positions by a
couple of centimetres). Feed it the previous run's result.
"""
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as Rot

CELL_AZ = (-60.0, -20.0, 20.0, 60.0)
CELL_EL = (-40.0, -10.0, 10.0, 40.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quat', required=True, help='T_lidar_radar rotation, xyzw')
    ap.add_argument('--xyz', required=True, help='T_lidar_radar translation, metres')
    ap.add_argument('--lidar-fov', type=float, default=22.5,
                    help='lidar HALF vertical FoV in deg (OS1 22.5, OS0 45)')
    ap.add_argument('--margin', type=float, default=1.5, help='deg of FoV kept in hand')
    a = ap.parse_args()

    R = Rot.from_quat([float(v) for v in a.quat.split(',')]).as_matrix()
    t = np.array([float(v) for v in a.xyz.split(',')])
    fov = np.radians(a.lidar_fov - a.margin)

    # Room axes as the radar experiences them: f = where it faces (horizontal),
    # l = its left, z = up. Everything below is quoted in these.
    b = R @ [1, 0, 0]
    f = np.array([b[0], b[1], 0.0]); f /= np.linalg.norm(f)
    l = np.cross([0, 0, 1.0], f)
    nose = np.degrees(np.arcsin(b[2]))

    def radar_pt(fwd, side, high):
        return R.T @ (fwd * f + side * l + high * np.array([0, 0, 1.0]))

    def lidar_el(fwd, side, high):
        q = R @ radar_pt(fwd, side, high) + t
        return np.arcsin(q[2] / np.linalg.norm(q))

    def rae(p):
        r = np.linalg.norm(p)
        return r, np.degrees(np.arctan2(p[1], p[0])), np.degrees(np.arcsin(p[2] / r))

    def h_limits(fwd):
        """Height range at this distance that keeps the target inside the lidar's
        vertical FoV. Solved numerically because the lidar's offset from the radar
        makes the two limits asymmetric."""
        out = []
        for sgn in (+1, -1):
            lo, hi = 0.0, 4.0
            for _ in range(40):                      # bisect on |lidar_el| = fov
                mid = 0.5 * (lo + hi)
                if abs(lidar_el(fwd, 0.0, sgn * mid)) < fov:
                    lo = mid
                else:
                    hi = mid
            out.append(sgn * lo)
        return out[1], out[0]                        # (lowest, highest)

    print(f'radar nose pitched {abs(nose):.1f} deg {"UP" if nose > 0 else "DOWN"}; '
          f'lidar sits {-t[2]*100:+.0f} cm relative to it (+ = above)')
    print(f'lidar vertical FoV +-{a.lidar_fov:.1f} deg, planning to '
          f'+-{a.lidar_fov - a.margin:.1f}\n')
    print('reachable height band, relative to the RADAR:')
    print('  forward     lowest    highest      -> radar elevation')
    for d in (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5):
        lo, hi = h_limits(d)
        e_lo = rae(radar_pt(d, 0, lo))[2]
        e_hi = rae(radar_pt(d, 0, hi))[2]
        print(f'   {d:.1f} m    {lo*100:+7.0f} cm {hi*100:+7.0f} cm      '
              f'{e_lo:+6.1f} .. {e_hi:+6.1f} deg')

    # Fill the 3x3 map from the close ring (best angular leverage), then add a
    # mid and a far arc purely for range spread.
    print('\nCAPTURE PLAN')
    print('   #   forward       side            height vs RADAR   radar range   el')
    n = 0
    rows = []
    for fwd, frac in ((1.0, 0.90), (2.2, 0.55), (3.3, 0.40)):
        lo, hi = h_limits(fwd)
        mid = 0.5 * (lo + hi)
        heights = ([hi * frac if hi > 0 else hi, mid, lo * frac if lo < 0 else lo]
                   if fwd < 1.5 else [hi * frac, mid])
        side = fwd * np.tan(np.radians(35.0))
        for h in heights:
            for s in (-side, 0.0, side):
                if len(rows) >= 15:
                    break
                p = radar_pt(fwd, s, h)
                r, az, el = rae(p)
                rows.append((fwd, s, h, r, az, el))
    for i, (fwd, s, h, r, az, el) in enumerate(rows[:15], 1):
        sd = (f'{abs(s)*100:3.0f} cm {"LEFT " if s > 0 else "RIGHT"}' if abs(s) > 0.01
              else '   centre   ')
        print(f'  {i:<3}  {fwd:.1f} m    {sd}    {h*100:+5.0f} cm         '
              f'{r:.2f} m   {el:+5.1f}')

    P = np.array([radar_pt(f_, s_, h_) for f_, s_, h_, *_ in rows[:15]])
    raz = np.array([rae(p) for p in P])
    cell = {(0 if x < CELL_AZ[1] else (1 if x < CELL_AZ[2] else 2),
             0 if y < CELL_EL[1] else (1 if y < CELL_EL[2] else 2))
            for x, y in zip(raz[:, 1], raz[:, 2])}
    print(f'\n  cells {len(cell)}/9   range spread {np.ptp(raz[:,0]):.1f} m   '
          f'az {np.ptp(raz[:,1]):.0f} deg   el {np.ptp(raz[:,2]):.0f} deg   '
          f'el both sides {min(max(raz[:,2].max(),0), max(-raz[:,2].min(),0)):.0f} deg')
    print('  az -60      0     +60')
    for r_, lab in ((2, 'el +10..+40'), (1, 'el -10..+10'), (0, 'el -40..-10')):
        print('  ' + ' '.join('[X]' if (c, r_) in cell else '[ ]' for c in range(3))
              + '  ' + lab)


if __name__ == '__main__':
    main()

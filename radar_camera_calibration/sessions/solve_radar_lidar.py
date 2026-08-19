#!/usr/bin/env python3
"""
Offline re-solve of a radar -> lidar extrinsic from a saved capture set.

    python3 solve_radar_lidar.py <session.json> [--sig-r 0.05] [--sig-az 3] [--sig-el 8]
                                 [--reject 4.0] [--reject-axis 3.5]
                                 [--cam-quat x,y,z,w --cam-xyz x,y,z]

Reads the `captures` list written by `radar_lidar_calib.py` (`~/save`) and
reproduces its solve without ROS. Use it to audit a run, merge several
sessions (concatenate their `captures`), or re-weight with a different radar
noise model. Prints the same numbers the node prints, plus a per-capture
residual table so you can see exactly which shots were rejected and why.

The solve is a measurement-space maximum-likelihood fit: residuals are formed
in the radar's OWN coordinates (range, azimuth, elevation), each divided by
its own sigma. That matters because the radar's error is wildly anisotropic
(5 cm in range, 3 deg in azimuth, 8 deg in elevation) — a plain Cartesian
least-squares fit would treat those as equal and bias the answer toward the
axis the radar knows least about.
"""
import argparse, json, sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot


# Observability of the capture set — mirrors COVERAGE_TARGETS in
# radar_lidar_calib.py, so a saved session grades the same offline as it did
# live. Each row is the spread that unlocks one DOF: a rotation d about axis k
# displaces a point by d(k x p), and the fit only sees it if the radar measures
# that displacement on an axis it is good at. AZ spread -> yaw. EL spread ->
# pitch and roll (in a single horizontal plane both are unobservable). RANGE
# spread -> separates rotation from translation. The BAL rows demand coverage on
# BOTH sides of boresight, because a wide one-sided smear fits a rotation error
# and an apex bias equally well and the two trade off.
COVERAGE_TARGETS = {'range': 1.50, 'az': 60.0, 'az_bal': 20.0,
                    'el': 30.0, 'el_bal': 10.0, 'near': 6.0}
NEAR_RANGE_M = 1.5


def coverage(P):
    """{name: (value, target, ok)} for the six observability rows."""
    raz = np.array([to_raz(p) for p in P])
    rng, az, el = raz[:, 0], np.degrees(raz[:, 1]), np.degrees(raz[:, 2])
    bal = lambda a: float(min(max(a.max(), 0.0), max(-a.min(), 0.0)))
    vals = {'range': float(rng.max() - rng.min()),
            'az': float(az.max() - az.min()), 'az_bal': bal(az),
            'el': float(el.max() - el.min()), 'el_bal': bal(el),
            'near': float((rng < NEAR_RANGE_M).sum())}
    return {k: (v, COVERAGE_TARGETS[k], v >= COVERAGE_TARGETS[k]) for k, v in vals.items()}


def to_raz(p):
    """Cartesian -> (range, azimuth, elevation) in the radar's own frame."""
    r = float(np.linalg.norm(p))
    return np.array([r, np.arctan2(p[1], p[0]), np.arcsin(p[2] / r)])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def kabsch(P, Q):
    """Isotropic closed-form fit — only ever used as the starting guess."""
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, mq - R @ mp


def residual(x, P, Q, sig):
    """Whitened (range, az, el) error for every correspondence, stacked."""
    R = Rot.from_rotvec(x[:3]).as_matrix()
    t = x[3:6]
    out = np.empty((len(P), 3))
    for i, (p, q) in enumerate(zip(P, Q)):
        m, pred = to_raz(p), to_raz(R.T @ (q - t))
        out[i] = [(m[0] - pred[0]) / sig[0],
                  wrap(m[1] - pred[1]) / sig[1],
                  wrap(m[2] - pred[2]) / sig[2]]
    return out.ravel()


def solve(P, Q, sig, reject=4.0, reject_axis=3.5, f_scale=1.5, iters=5):
    R0, t0 = kabsch(P, Q)
    x = np.concatenate([Rot.from_matrix(R0).as_rotvec(), t0])
    mask = np.ones(len(P), bool)
    for _ in range(iters):
        sol = least_squares(residual, x, args=(P[mask], Q[mask], sig),
                            loss='huber', f_scale=f_scale)
        x = sol.x
        d = residual(x, P, Q, sig).reshape(-1, 3)
        # Two independent gates. The RMS one catches a shot that is wrong in
        # general; the per-axis one catches an elevation mirror ghost, which is
        # right in range and azimuth and so slips under an RMS threshold.
        keep = (np.linalg.norm(d, axis=1) / np.sqrt(3) < reject)
        if reject_axis > 0:
            keep &= np.abs(d).max(1) < reject_axis
        if keep.sum() < max(4, int(0.5 * len(P))) or np.array_equal(keep, mask):
            break
        mask = keep

    d = residual(x, P, Q, sig).reshape(-1, 3)
    rms = float(np.sqrt((np.linalg.norm(d[mask], axis=1) ** 2 / 3).mean()))

    # Per-DOF 1-sigma from the Gauss-Newton covariance at the solution.
    J = sol.jac
    try:
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((6, 6), np.nan)
    # Pose-diversity conditioning: how well-spread the radar points are. Same
    # definition the live node prints, so the two numbers are comparable.
    Pc = P[mask] - P[mask].mean(0)
    cond = float(np.linalg.cond(Pc)) if mask.sum() >= 3 else float('inf')
    return dict(R=Rot.from_rotvec(x[:3]).as_matrix(), t=x[3:6], mask=mask,
                d=d, rms=rms, cov=cov, cond=cond)


def loo(P, Q, sig, **kw):
    """Leave-one-out: refit without each inlier, score it with the others' fit."""
    errs = []
    for i in range(len(P)):
        k = np.arange(len(P)) != i
        try:
            s = solve(P[k], Q[k], sig, **kw)
        except Exception:
            continue
        errs.append(np.linalg.norm(residual(
            np.concatenate([Rot.from_matrix(s['R']).as_rotvec(), s['t']]),
            P[i:i + 1], Q[i:i + 1], sig)) / np.sqrt(3))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--sig-r', type=float, default=0.05)
    ap.add_argument('--sig-az', type=float, default=3.0)
    ap.add_argument('--sig-el', type=float, default=8.0)
    ap.add_argument('--reject', type=float, default=4.0)
    ap.add_argument('--reject-axis', type=float, default=3.5)
    ap.add_argument('--cam-quat', default='', help='T_lidar_camera xyzw, to compose T_cam_radar')
    ap.add_argument('--cam-xyz', default='', help='T_lidar_camera translation, metres')
    a = ap.parse_args()

    doc = json.load(open(a.session))
    caps = doc['captures']
    P = np.array([c['p_radar'] for c in caps], float)
    Q = np.array([c['p_lidar'] for c in caps], float)
    sig = np.array([a.sig_r, np.radians(a.sig_az), np.radians(a.sig_el)])

    s = solve(P, Q, sig, a.reject, a.reject_axis)
    R, t, mask = s['R'], s['t'], s['mask']
    q = Rot.from_matrix(R).as_quat()
    err = (R @ P[mask].T).T + t - Q[mask]

    print(f"\ncaptures {len(P)}   inliers {mask.sum()}   "
          f"residual {s['rms']:.2f}s   LOO {loo(P[mask], Q[mask], sig):.2f}s   "
          f"cond {s['cond']:.1f}")
    print(f"\nT_{doc.get('parent_frame','lidar')}_{doc.get('child_frame','radar')}")
    print(f"  t (m)      : {t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}   |t| = {np.linalg.norm(t)*100:.1f} cm")
    print(f"  quat xyzw  : {q[0]:+.5f} {q[1]:+.5f} {q[2]:+.5f} {q[3]:+.5f}")
    print(f"  rpy (deg)  : " + ' '.join(f'{v:+.2f}' for v in Rot.from_matrix(R).as_euler('xyz', degrees=True)))

    sd = np.sqrt(np.clip(np.diag(s['cov']), 0, None))
    print(f"  1s rot deg : " + ' '.join(f'{np.degrees(v):.2f}' for v in sd[:3]))
    print(f"  1s t   mm  : " + ' '.join(f'{v*1000:.1f}' for v in sd[3:6]))
    print(f"  bias   mm  : " + ' '.join(f'{v:+.1f}' for v in err.mean(0) * 1000))
    print(f"  RMS    mm  : " + ' '.join(f'{v:.0f}' for v in np.sqrt((err ** 2).mean(0)) * 1000))

    ax = {k: R @ v for k, v in (('X', [1, 0, 0]), ('Y', [0, 1, 0]), ('Z', [0, 0, 1]))}
    print('  radar axes in lidar frame: ' + '  '.join(
        f'{k}->[{v[0]:+.2f} {v[1]:+.2f} {v[2]:+.2f}]' for k, v in ax.items()))

    cov = coverage(P[mask])
    print('\ncoverage of the inlier set — what each spread makes observable')
    for k, unlocks in (('range', 't vs R'), ('az', 'yaw'), ('az_bal', 'yaw, one-sidedness'),
                       ('el', 'pitch + roll'), ('el_bal', 'pitch, one-sidedness'),
                       ('near', f'all (captures under {NEAR_RANGE_M:.1f} m)')):
        v, tgt, ok = cov[k]
        print(f"  {k:<7} {v:7.1f} / {tgt:5.1f}   {'PASS' if ok else 'FAIL'}   {unlocks}")

    if a.cam_quat and a.cam_xyz:
        # Saved transform is T_lidar_camera; invert it to get T_cam_lidar, then
        # compose:  T_cam_radar = T_cam_lidar . T_lidar_radar
        Rg = Rot.from_quat([float(v) for v in a.cam_quat.split(',')]).as_matrix()
        tg = np.array([float(v) for v in a.cam_xyz.split(',')])
        R_cl, t_cl = Rg.T, -Rg.T @ tg
        R_cr, t_cr = R_cl @ R, R_cl @ t + t_cl
        q_cr = Rot.from_matrix(R_cr).as_quat()
        print('\nT_camera_radar (composed through the lidar)')
        print(f"  t (m)     : {t_cr[0]:+.6f} {t_cr[1]:+.6f} {t_cr[2]:+.6f}")
        print(f"  quat xyzw : {q_cr[0]:+.6f} {q_cr[1]:+.6f} {q_cr[2]:+.6f} {q_cr[3]:+.6f}")

    print('\n  #   range   |  dr    daz   del   (sigma)  |  3D err mm  |')
    for i, c in enumerate(caps):
        r = np.linalg.norm(P[i])
        e = (R @ P[i] + t) - Q[i]
        flag = '' if mask[i] else '  <-- REJECTED'
        print(f"  {c.get('idx', i+1):<3} {r:5.2f} m  | {s['d'][i][0]:+5.2f} {s['d'][i][1]:+5.2f} "
              f"{s['d'][i][2]:+5.2f}          | {np.linalg.norm(e)*1000:7.0f}    |{flag}")


if __name__ == '__main__':
    sys.exit(main())

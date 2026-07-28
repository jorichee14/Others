#!/usr/bin/env python3
"""
Dump the 6x6 extrinsic covariance of T_cam_radar from a logged *_poses.json.

Re-runs the measurement-space ML solve (same estimator as solve_from_poses.py:
range/az/el residuals weighted by the radar sigmas, Huber loss) and then reads
the parameter covariance the same way the joint solver does:

    cov6 = pinv(J^T J)      over params x = [rotvec(3), t(3)]

evaluated at the solution. The apex offset is held fixed (these logs carry only
(p_cam, p_radar), not the per-pose board pose), so this is exactly the [:6,:6]
extrinsic block of solve_from_poses_joint.py's cov6 with the offset pinned.

Reports per-axis 1-sigma (rotvec in deg, t in mm), the full 6x6, and the
correlation matrix (which axes trade off against each other).

Usage:  python3 dump_extrinsic_cov.py <session_poses.json>
"""
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares

np.set_printoptions(precision=4, suppress=True, linewidth=120)


def kabsch(P, Q):
    muP, muQ = P.mean(0), Q.mean(0)
    H = (P - muP).T @ (Q - muQ)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1., 1., float(np.sign(np.linalg.det(Vt.T @ U.T)))])
    R = Vt.T @ D @ U.T
    return R, muQ - R @ muP


def cart_to_raz(p):
    r = float(np.linalg.norm(p))
    if r < 1e-9:
        return np.array([0., 0., 0.])
    return np.array([r, np.arctan2(p[1], p[0]), np.arcsin(np.clip(p[2] / r, -1, 1))])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def resid_fn(P, Q, sig_r, sig_az, sig_el):
    def resid(x):
        R = Rot.from_rotvec(x[:3]).as_matrix(); t = x[3:6]
        out = []
        for i in range(len(P)):
            pr = R.T @ (Q[i] - t)
            rp = cart_to_raz(pr); rm = cart_to_raz(P[i])
            out += [(rm[0] - rp[0]) / sig_r,
                    _wrap(rm[1] - rp[1]) / sig_az,
                    _wrap(rm[2] - rp[2]) / sig_el]
        return np.array(out)
    return resid


def main():
    path = sys.argv[1]
    d = json.load(open(path))
    P = np.array([c["p_radar"] for c in d["captures"]], float)
    Q = np.array([c["p_cam"] for c in d["captures"]], float)
    prm = d["params"]
    sig_r = prm["sigma_range_m"]
    sig_az = np.radians(prm["sigma_az_deg"])
    sig_el = np.radians(prm["sigma_el_deg"])

    R0, t0 = kabsch(P, Q)
    x0 = np.concatenate([Rot.from_matrix(R0).as_rotvec(), t0])
    resid = resid_fn(P, Q, sig_r, sig_az, sig_el)
    sol = least_squares(resid, x0, method='trf', loss='huber', f_scale=1.5, max_nfev=6000)
    x = sol.x
    R = Rot.from_rotvec(x[:3]).as_matrix(); t = x[3:6]
    rpy = Rot.from_matrix(R).as_euler('xyz', degrees=True)
    q = Rot.from_matrix(R).as_quat()

    # cov6 = pinv(J^T J) at the solution (same as solve_from_poses_joint.py)
    J = sol.jac
    cov6 = np.linalg.pinv(J.T @ J)
    sig = np.sqrt(np.clip(np.diag(cov6), 0, None))
    sig_deg = np.degrees(sig[:3])     # rotvec axis 1-sigma -> deg
    sig_mm = sig[3:] * 1000.0         # translation 1-sigma -> mm

    # correlation matrix
    dinv = np.diag(1.0 / np.where(sig > 0, sig, 1.0))
    corr = dinv @ cov6 @ dinv

    # rescale rotvec block of the *display* covariance to deg^2 / deg*mm / mm^2
    # so the printed 6x6 has interpretable units.
    scl = np.diag([np.degrees(1)] * 3 + [1000.0] * 3)
    cov6_u = scl @ cov6 @ scl

    n = len(P)
    print(f"\n=== {d.get('session','?')} ===")
    print(f"file: {path}   N={n} poses")
    print(f"t (m)    : {t.round(4).tolist()}   |t| {np.linalg.norm(t)*100:.1f} cm")
    print(f"quat xyzw: {q.round(4).tolist()}")
    print(f"rpy (deg): {rpy.round(2).tolist()}")
    print(f"\nper-axis 1-sigma:")
    print(f"  rot  (deg): rx {sig_deg[0]:.2f}  ry {sig_deg[1]:.2f}  rz {sig_deg[2]:.2f}")
    print(f"  t    (mm) : tx {sig_mm[0]:.1f}  ty {sig_mm[1]:.1f}  tz {sig_mm[2]:.1f}")
    print(f"\n6x6 covariance  [order: rx ry rz (deg), tx ty tz (mm)]  units deg^2 / deg*mm / mm^2:")
    print(cov6_u)
    print(f"\n6x6 correlation matrix  [rx ry rz tx ty tz]:")
    print(corr)


if __name__ == "__main__":
    main()

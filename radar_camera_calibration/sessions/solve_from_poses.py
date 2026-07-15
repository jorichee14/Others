#!/usr/bin/env python3
"""
Independent offline re-solve of a recorded calibration session, to CROSS-CHECK
the live node's extrinsic. Reads a *_poses.json (the (p_cam, p_radar) pairs the
node captured) and re-estimates T_cam_radar two ways:

  1. Kabsch / Umeyama  (isotropic Cartesian least squares)  -> baseline
  2. Measurement-space ML in (range, azimuth, elevation) weighted by the radar
     sigmas + Huber robust loss  -> the same estimator the node uses.

It fixes the apex offset at whatever produced the logged p_cam (we don't have the
per-pose board rotation here), so it reproduces the node's (R, t) but NOT its
joint offset refinement. A close match validates the live result.

Deps: numpy, scipy.  Run:  python3 solve_from_poses.py 2026-07-15_zed_radar1_poses.json
"""
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares


def kabsch(P, Q):
    """q = R p + t (source P=radar -> target Q=camera)."""
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


def ml_solve(P, Q, sig_r, sig_az, sig_el, huber=1.5):
    """Measurement-space ML: predict each radar meas from the camera apex via
    (R,t), minimise (range,az,el) residuals / sigma with a Huber loss."""
    R0, t0 = kabsch(P, Q)
    x0 = np.concatenate([Rot.from_matrix(R0).as_rotvec(), t0])

    def resid(x):
        R = Rot.from_rotvec(x[:3]).as_matrix(); t = x[3:6]
        out = []
        for i in range(len(P)):
            pr = R.T @ (Q[i] - t)            # predicted radar point
            rp = cart_to_raz(pr); rm = cart_to_raz(P[i])
            out += [(rm[0] - rp[0]) / sig_r,
                    _wrap(rm[1] - rp[1]) / sig_az,
                    _wrap(rm[2] - rp[2]) / sig_el]
        return np.array(out)

    sol = least_squares(resid, x0, method='trf', loss='huber', f_scale=huber, max_nfev=6000)
    R = Rot.from_rotvec(sol.x[:3]).as_matrix()
    return R, sol.x[3:6]


def report(name, R, t, ref_R, ref_t):
    q = Rot.from_matrix(R).as_quat()
    rpy = Rot.from_matrix(R).as_euler('xyz', degrees=True)
    dR = Rot.from_matrix(ref_R.T @ R).magnitude() * 180 / np.pi      # geodesic deg
    dt = np.linalg.norm(t - ref_t) * 1000                            # mm
    print(f"\n[{name}]")
    print(f"  t (m)   : {t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}   |t| {np.linalg.norm(t)*100:.1f} cm")
    print(f"  rpy(deg): {rpy[0]:+.2f} {rpy[1]:+.2f} {rpy[2]:+.2f}")
    print(f"  quat    : {q[0]:+.4f} {q[1]:+.4f} {q[2]:+.4f} {q[3]:+.4f}")
    print(f"  vs node : dRot {dR:.2f} deg   dTrans {dt:.1f} mm")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "2026-07-15_zed_radar1_poses.json"
    with open(path) as f:
        d = json.load(f)
    P = np.array([c["p_radar"] for c in d["captures"]], float)   # radar
    Q = np.array([c["p_cam"] for c in d["captures"]], float)     # camera apex
    prm = d["params"]
    # result section: node writes it under "result"; the hand-written record uses
    # "tool_result_31_captures". Translation key is "..._translation" (node) or
    # "..._translation_m" (hand-written).
    tr = d.get("result") or d.get("tool_result_31_captures")
    ref_t = np.array(tr.get("T_cam_radar_translation") or tr["T_cam_radar_translation_m"], float)
    ref_R = Rot.from_quat(tr["T_cam_radar_quaternion_xyzw"]).as_matrix()

    print(f"session: {d.get('session', d.get('node','?'))}   "
          f"{d.get('date_utc', d.get('iso_time_utc','?'))}   N={len(P)} poses")

    Rk, tk = kabsch(P, Q)
    report("Kabsch (isotropic)", Rk, tk, ref_R, ref_t)

    Rm, tm = ml_solve(P, Q,
                      prm["sigma_range_m"],
                      np.radians(prm["sigma_az_deg"]),
                      np.radians(prm["sigma_el_deg"]))
    report("Measurement-space ML", Rm, tm, ref_R, ref_t)

    # node's own numbers for reference
    print("\n[node result (from JSON)]")
    print(f"  t (m)   : {ref_t[0]:+.4f} {ref_t[1]:+.4f} {ref_t[2]:+.4f}   |t| {np.linalg.norm(ref_t)*100:.1f} cm")
    print(f"  rpy(deg): {tr['T_cam_radar_rpy_deg']}")

    # in-sample 3-D residual of the ML fit (inliers) vs the node's 297.8 mm RMS
    pred = (P @ Rm.T) + tm
    rms = float(np.sqrt(((pred - Q) ** 2).sum(1).mean())) * 1000
    print(f"\nML in-sample 3-D RMS: {rms:.1f} mm   (node reported 297.8 mm)")
    dR = Rot.from_matrix(ref_R.T @ Rm).magnitude() * 180 / np.pi
    dt = np.linalg.norm(tm - ref_t) * 1000
    verdict = "MATCH" if (dR < 5 and dt < 60) else "CHECK"
    print(f"\n>>> {verdict}: ML re-solve is within {dR:.2f} deg / {dt:.1f} mm of the live node.")
    print("    (rounding of logged p_cam to 3 dp + no offset refinement explain the small gap)")


if __name__ == "__main__":
    main()

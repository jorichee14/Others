#!/usr/bin/env python3
"""
Offline JOINT re-solve of a logged calibration session.

Unlike solve_from_poses.py (which fixes the apex offset at whatever produced the
logged p_cam), this reads the full per-capture BOARD POSE (board_R_quat_xyzw,
board_t) and re-runs the measurement-space ML solve with the apex offset as a
FREE / MAP parameter — so you can decide how much to trust your tape measure:

  offset_prior_sigma = 0      -> fully FREE (ignore tape, pure data)
  offset_prior_sigma = 0.15   -> loose MAP (tape is only a hint)
  offset_prior_sigma = 0.05   -> tight (reproduces the live run default)

The key readout is the apex 1-sigma PER AXIS. Small sigma on an axis => the DATA
determined it; large sigma => it is unobservable from these poses and freeing it
just lets it wander (and corrupt t / rotation). No ROS required.

Usage:
  python3 solve_from_poses_joint.py <session.json> [offset_prior_sigma_m] [--no-ext-prior]
"""
import sys, json
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot


def cart_to_raz(p):
    r = float(np.linalg.norm(p))
    if r < 1e-9:
        return np.array([0., 0., 0.])
    return np.array([r, np.arctan2(p[1], p[0]), np.arcsin(np.clip(p[2] / r, -1, 1))])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def kabsch(P, Q):
    muP, muQ = P.mean(0), Q.mean(0)
    H = (P - muP).T @ (Q - muQ)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1., 1., float(np.sign(np.linalg.det(Vt.T @ U.T)))])
    R = Vt.T @ D @ U.T
    return R, muQ - R @ muP


def solve(P, Rb, Tb, a0, sig_r, sig_az, sig_el, use_el=True,
          offset_prior_sigma=0.0, R_prior=None, t_prior=None,
          rot_prior_sigma=None, t_prior_sigma=None,
          huber=1.5, reject_sigma=4.0, max_iter=5):
    P = np.asarray(P, float); Rb = np.asarray(Rb, float)
    Tb = np.asarray(Tb, float); a0 = np.asarray(a0, float)
    k = 3 if use_el else 2
    prior = offset_prior_sigma and offset_prior_sigma > 0
    ext_rot = R_prior is not None and rot_prior_sigma and rot_prior_sigma > 0
    ext_t = t_prior is not None and t_prior_sigma and t_prior_sigma > 0

    def unpack(x):
        return Rot.from_rotvec(x[:3]).as_matrix(), x[3:6], x[6:9]

    def resid(x, idx):
        R, t, a = unpack(x); out = []
        for i in idx:
            pr = R.T @ ((Rb[i] @ a) + Tb[i] - t)
            rp = cart_to_raz(pr); rm = cart_to_raz(P[i])
            out.append((rm[0] - rp[0]) / sig_r); out.append(_wrap(rm[1] - rp[1]) / sig_az)
            if use_el:
                out.append(_wrap(rm[2] - rp[2]) / sig_el)
        if prior:
            out.extend(list((x[6:9] - a0) / offset_prior_sigma))
        if ext_rot:
            out.extend(list(Rot.from_matrix(R_prior.T @ R).as_rotvec() / rot_prior_sigma))
        if ext_t:
            out.extend(list((t - np.asarray(t_prior, float)) / t_prior_sigma))
        return np.array(out)

    def pps(x, idx):
        R, t, a = unpack(x); s = []
        for i in idx:
            pr = R.T @ ((Rb[i] @ a) + Tb[i] - t)
            rp = cart_to_raz(pr); rm = cart_to_raz(P[i])
            d = [(rm[0] - rp[0]) / sig_r, _wrap(rm[1] - rp[1]) / sig_az]
            if use_el:
                d.append(_wrap(rm[2] - rp[2]) / sig_el)
            s.append(np.linalg.norm(d) / np.sqrt(k))
        return np.array(s)

    if ext_rot and ext_t:
        R0, t0 = R_prior, np.asarray(t_prior, float)
    else:
        Q0 = np.array([(Rb[i] @ a0) + Tb[i] for i in range(len(P))]); R0, t0 = kabsch(P, Q0)
    x = np.concatenate([Rot.from_matrix(R0).as_rotvec(), t0, a0])
    allidx = np.arange(len(P)); mask = np.ones(len(P), bool); sol = None
    for _ in range(max_iter):
        sol = least_squares(resid, x, args=(allidx[mask],), method='trf',
                            loss='huber', f_scale=huber, max_nfev=6000)
        x = sol.x; pn = pps(x, allidx); new = pn < reject_sigma
        if new.sum() < max(4, int(0.5 * len(P))):
            keep = max(4, int(0.6 * len(P))); new = pn <= np.sort(pn)[keep - 1]
        if np.array_equal(new, mask):
            break
        mask = new
    R, t, a = unpack(x)
    try:
        full = np.linalg.pinv(sol.jac.T @ sol.jac)
    except Exception:
        full = np.full((len(x), len(x)), np.nan)
    apex_sig = np.sqrt(np.clip(np.diag(full)[6:9], 0, None))
    rms = float(np.sqrt((pps(x, allidx[mask]) ** 2).mean()))
    return dict(R=R, t=t, a=a, apex_sigma=apex_sig, rms=rms, n_in=int(mask.sum()), n=len(P))


def main():
    path = sys.argv[1]
    ops = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('-') else 0.0
    no_ext = '--no-ext-prior' in sys.argv
    d = json.load(open(path))
    caps = d['captures']; pr = d['params']
    P = np.array([c['p_radar'] for c in caps], float)
    Rb = np.array([Rot.from_quat(c['board_R_quat_xyzw']).as_matrix() for c in caps])
    Tb = np.array([c['board_t'] for c in caps], float)
    a0 = np.array([pr['reflector_offset_x'], pr['reflector_offset_y'], pr['reflector_offset_z']], float)
    sig_r = pr['sigma_range_m']; sig_az = np.radians(pr['sigma_az_deg']); sig_el = np.radians(pr['sigma_el_deg'])
    Rp = tp = rps = tps = None
    if pr.get('use_extrinsic_prior') and not no_ext:
        Rp = Rot.from_euler('xyz', pr['prior_rpy_deg'], degrees=True).as_matrix()
        tp = np.array(pr['prior_t_xyz'], float)
        rps = np.radians(pr['prior_rot_sigma_deg']); tps = pr['prior_t_sigma_m']
    r = solve(P, Rb, Tb, a0, sig_r, sig_az, sig_el, True, ops, Rp, tp, rps, tps)
    t = r['t']; rpy = Rot.from_matrix(r['R']).as_euler('xyz', degrees=True)
    zsig = r['apex_sigma'][2] * 1000
    print(f"\n=== offset_prior_sigma = {ops} m  ({'FREE' if ops <= 0 else 'MAP'})   ext_prior={'off' if no_ext else 'on'} ===")
    print(f"  inliers {r['n_in']}/{r['n']}   residual {r['rms']:.2f} sigma")
    print(f"  t (m)   : {t.round(4).tolist()}   |t| {np.linalg.norm(t) * 100:.1f} cm")
    print(f"  rpy(deg): {rpy.round(2).tolist()}")
    print(f"  apex off: {r['a'].round(4).tolist()} m")
    print(f"  apex 1s : {(r['apex_sigma'] * 1000).round(1).tolist()} mm"
          f"   (z {'DATA-DETERMINED' if zsig < 40 else 'WEAK/unobservable'})")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Radar ↔ Camera Extrinsic Calibration  —  ChArUco board + trihedral reflector
============================================================================
ONE rigid target: a ChArUco board with a trihedral corner reflector bolted to
it at a KNOWN, FIXED offset. The reflector apex is the single brightest radar
return; because it is rigidly tied to the board, the CAMERA can locate the same
apex from the board pose.

    camera : detect ChArUco board → T_cam_board (6-DOF) → p_cam = T_cam_board·apex_board
    radar  : reflector = strongest gated return → p_radar   (a 3-D point)

A radar point has no orientation, so ONE view can't give a 6-DOF transform.
Move the rig to N ≥ 3 non-collinear spots, collect corresponding point pairs
{(p_cam^i, p_radar^i)}, and solve for X = T_cam_radar (p_cam = R·p_radar + t).

THE ESTIMATOR — why NOT plain Kabsch
────────────────────────────────────
A radar measures RANGE precisely (~cm) but ANGLE poorly (degrees), and the
cross-range error GROWS with range (≈ range·σ_az: ~9 cm at 5 m for σ_az=1°).
Isotropic Cartesian Kabsch (min ‖R·p_radar+t−p_cam‖²) weights that large,
range-dependent angular error the same as the tiny range error, so it is
biased. Monte-Carlo on this exact geometry: Kabsch ≈ 1.7°/120 mm vs the
estimator below ≈ 1.0°/38 mm (≈1.6× rotation, ≈3× translation better).

Instead we do MAXIMUM-LIKELIHOOD estimation in the radar's MEASUREMENT space:
predict each radar measurement from the (accurate) camera apex via the current
(R,t), convert to (range, azimuth, elevation), and minimise residuals weighted
by each component's real σ:

    predicted radar pt = R^T (p_cam − t)
    min_{R,t} Σ ρ( [ (r_m−r_p)/σ_r , (az_m−az_p)/σ_az , (el_m−el_p)/σ_el ] )

  • ρ = Huber → robust to the occasional bad radar match; plus iterative
    σ-gated rejection of gross outliers (both proven necessary: one wrong
    match blows plain L2 up to ~50°).
  • Kabsch provides the initial guess. When an extrinsic prior is supplied the
    problem is ALSO solved without it, from Kabsch, and the two rotations are
    compared — a wrong prior fits as well as a right one, so the residual can
    never catch it and only that disagreement can.
  • 2-D radar (one reporting z == 0 for every point) is auto-detected and the
    elevation residual is dropped — but then out-of-plane rotation and height
    are UNOBSERVABLE; the covariance readout flags exactly which DOFs are weak.

READING THE ROTATION — rpy is a TRAP on this rig
────────────────────────────────────────────────
The extrinsic is a ~90° frame swap, i.e. |pitch| = 90°, which is EXACTLY the
singularity of the 'xyz' euler convention. There the triple is not unique and a
1° physical change rewrites it by 90°:  [-90,-90,0] -> [0,-89,-90]. A perfectly
good calibration can therefore print an rpy that looks catastrophically wrong,
and pasting a printed triple back in as prior_rpy_deg turns that display artefact
into a genuinely wrong prior. So: the rotation is reported as quaternion +
axis-angle + an explicit "radar axis -> camera axis" map (exact, singularity
free, eyeball-checkable against the mounting), rotations are only ever COMPARED
geodesically, and every solve prints a round-trip-safe prior_quat_xyzw line.

THE APEX OFFSET — measured, refined, or CALCULATED (solve_offset)
────────────────────────────────────────────────────────────────
The offset a (board→apex, board frame) can be jointly estimated with the
extrinsic (MAP: free a regularised toward your measured value):
  p_cam_i = board_R_i · a + board_t_i.  Because the board rotates between poses,
a is separable from the constant extrinsic translation.
  • measured well → keep offset_prior_sigma tight; the solve barely moves it.
  • measured badly → the solve repairs it (verified: 44 mm error → ~15 mm,
    translation error 44 → 24 mm).
  • can't measure → seed 0 with a loose prior and it CALCULATES the offset
    (114 mm → ~15 mm) — provided you include the poses that make it observable.
Observability caveat: a is only visible where the radar's cross-range noise is
smaller than the offset, i.e. at CLOSE range (1.5–3 m) with HIGH board tilt
(±45–55°). At long range it is swamped; the solve then just returns your prior.
The reported apex 1σ tells you which happened, and the debug overlay confirms
the apex visually regardless.

WHY THE CAMERA SIDE USES THE BOARD, NOT DEPTH
─────────────────────────────────────────────
A bare corner reflector is specular — stereo/ToF depth on bare metal is
unreliable, and a hand-click on it is ~10 px noisy. The ChArUco board pose is
metric (from the known square size), sub-mm, and fully automatic. The apex
offset is measured once, in the board frame (see README).

RADAR SIDE (inspired by the click-based v8 tool, improved)
──────────────────────────────────────────────────────────
  • BACKGROUND SUBTRACTION — pool N frames of the static scene (rig absent); a
    live point is "new" only if far from every background point. Kills clutter.
  • GATING — range window, optional |doppler| window (the held-still reflector
    is ~0 doppler, so moving people are rejected), and — once an extrinsic
    exists — proximity to the camera-predicted apex.
  • HIGHEST-SNR SELECTION — among the surviving points, take argmax(SNR). The
    trihedral is engineered to be the strongest reflector, so this is the apex.
    (No clustering: one bright, background-subtracted point is the target.)

VALIDATION (carried over from v8)
─────────────────────────────────
  • Leave-one-out cross-validation (honest generalisation on a small set).
  • Per-axis SIGNED residual bias (RMS hides a constant lean; a mean residual
    on an axis exposes leftover range bias or a small axis error).
  • Reprojection error in pixels (project the radar-predicted apex through K).
  • Point-set condition number (warns on collinear/planar poses).
  • Range scale/bias diagnostic (radar range vs camera range).
  • Optional tape-measured |t| baseline check.
  • LIVE reprojection overlay published as an image: the whole radar cloud
    projected onto the camera feed via the solved X. Dots staying glued to the
    reflector everywhere in the FoV is the single most trustworthy check.

Coordinate conventions
──────────────────────
  radar  : X=forward, Y=left,  Z=up     (automotive)
  camera : X=right,   Y=down,  Z=forward (optical / pinhole)
  Raw radar points are fed straight into Kabsch, so the solved R absorbs the
  full ~90° frame difference AND the mounting rotation — no separate remap
  step to get wrong. A large-looking R (≈90°) is therefore EXPECTED and correct.

Control (std_msgs/Empty topics)
───────────────────────────────
  ~/background  pool background frames (rig OUT of scene)
  ~/capture     grab one pose now (manual mode / on demand)
  ~/solve       force a solve + full validation report
  ~/reset       clear all captures
  ~/save        write YAML/JSON now

Deps:  numpy scipy opencv-contrib-python  +  ROS2 rclpy cv_bridge sensor_msgs_py
"""
import json
import warnings
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import message_filters
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Empty
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge

try:
    from sensor_msgs_py import point_cloud2 as pc2
    _HAVE_PC2 = True
except Exception:
    _HAVE_PC2 = False


_DICT_MAP = {
    'DICT_4X4_50': cv2.aruco.DICT_4X4_50, 'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
    'DICT_4X4_250': cv2.aruco.DICT_4X4_250, 'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
    'DICT_5X5_100': cv2.aruco.DICT_5X5_100, 'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
    'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
}


def build_board(sx, sy, square, marker, dict_name):
    dname = _DICT_MAP.get(dict_name, cv2.aruco.DICT_4X4_50)
    d = (cv2.aruco.getPredefinedDictionary(dname)
         if hasattr(cv2.aruco, "getPredefinedDictionary")
         else cv2.aruco.Dictionary_get(dname))
    if hasattr(cv2.aruco, "CharucoBoard"):
        try:
            b = cv2.aruco.CharucoBoard((sx, sy), square, marker, d)
        except Exception:
            b = cv2.aruco.CharucoBoard_create(sx, sy, square, marker, d)
    else:
        b = cv2.aruco.CharucoBoard_create(sx, sy, square, marker, d)
    new = hasattr(cv2.aruco, "CharucoDetector")
    det = cv2.aruco.CharucoDetector(b) if new else None
    obj = b.getChessboardCorners() if hasattr(b, "getChessboardCorners") else b.chessboardCorners
    return d, b, det, new, obj


# ─────────────── maths ───────────────
def kabsch(P, Q):
    """Rigid transform with q = R p + t, least squares (Kabsch/Umeyama, no scale).
    P,Q: (N,3) source(radar), target(camera). Returns R,t."""
    muP, muQ = P.mean(0), Q.mean(0)
    H = (P - muP).T @ (Q - muQ)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1., 1., float(np.sign(np.linalg.det(Vt.T @ U.T)))])
    R = Vt.T @ D @ U.T
    t = muQ - R @ muP
    return R, t


def rms_3d(P, Q, R, t):
    pred = (P @ R.T) + t
    return float(np.sqrt(((pred - Q) ** 2).sum(1).mean()))


def cart_to_raz(p):
    """radar Cartesian (X=fwd,Y=left,Z=up) → (range, azimuth, elevation)."""
    r = float(np.linalg.norm(p))
    if r < 1e-9:
        return np.array([0.0, 0.0, 0.0])
    return np.array([r, np.arctan2(p[1], p[0]), np.arcsin(np.clip(p[2] / r, -1, 1))])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _cam_apex(Rb, tb, a):
    """Apex in the camera frame for board pose (Rb,tb) and board-frame offset a."""
    return (Rb @ a) + tb


# ─────────────── rotation readout (NEVER trust euler here) ───────────────
# This rig's extrinsic is a ~90° frame swap, which puts it ON the singularity of
# the 'xyz' euler convention (|pitch| = 90°). There, scipy zeroes the third angle
# and a ONE DEGREE physical change swings the printed triple by NINETY degrees:
#     [-90,-90,0]  --(+1° about camera X)-->  [0,-89,-90]
# Nothing is wrong with the rotation; the *representation* has blown up. Reading
# rpy as "the answer" — or worse, pasting a printed triple back in as
# prior_rpy_deg — turns that display artefact into a real, large error. So every
# rotation is reported as quaternion + axis-angle + an explicit axis mapping, and
# rotations are only ever COMPARED geodesically.
GIMBAL_WARN_DEG = 8.0
_CAM_AXES = ('X(right)', 'Y(down)', 'Z(fwd)')
_RADAR_AXES = ('X(fwd)', 'Y(left)', 'Z(up)')


def geodesic_deg(Ra, Rb):
    """Angle of the single rotation taking Ra to Rb — the only meaningful
    'how far apart are these two rotations' number."""
    return float(np.degrees(np.linalg.norm(Rot.from_matrix(Ra.T @ Rb).as_rotvec())))


def axis_mapping(R):
    """Where each RADAR axis points in CAMERA coordinates, as readable text.

    This is the readout to trust: exact, singularity-free, and checkable against
    the physical mounting by eye. For a radar sitting beside the camera facing
    the same way you should see fwd→+Z; if 'up' comes out as camera +Y (down),
    the radar is mounted rolled 180°."""
    out = []
    for j, name in enumerate(_RADAR_AXES):
        v = R[:, j]
        k = int(np.argmax(np.abs(v)))
        sign = '+' if v[k] >= 0 else '-'
        out.append(f"radar {name:<8} -> camera {sign}{_CAM_AXES[k]:<9} "
                   f"[{v[0]:+.3f} {v[1]:+.3f} {v[2]:+.3f}]")
    return out


def near_gimbal_lock(R, tol_deg=GIMBAL_WARN_DEG):
    """(is_near, pitch_deg) for the 'xyz' euler readout of R. R = Rz·Ry·Rx, so
    R[2,0] = -sin(pitch) and the convention is singular at |pitch| = 90°."""
    pitch = float(np.degrees(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))))
    return bool(abs(abs(pitch) - 90.0) < tol_deg), pitch


def robust_ml_calibrate(P_radar, board_R, board_t, apex0,
                        sig_r, sig_az, sig_el, use_elevation=True,
                        solve_offset=True, offset_prior_sigma=0.03,
                        R_prior=None, t_prior=None,
                        rot_prior_sigma=None, t_prior_sigma=None,
                        huber=1.5, reject_sigma=4.0, max_iter=5, init=None):
    """
    Maximum-likelihood extrinsic in the radar's MEASUREMENT space, optionally
    jointly estimating the reflector apex offset (board frame).

    A radar measures range precisely but angle poorly, with cross-range error
    that grows with range — so isotropic Cartesian Kabsch is biased. We predict
    each radar measurement from the camera apex via the current (R,t[,a]),
    convert to (range,az,el), and minimise residuals weighted by the real per-
    component sigmas. Huber loss + iterative sigma-gating reject bad matches.

        p_cam_i = board_R_i · a + board_t_i        (a = apex offset, board frame)
        predicted radar pt = Rᵀ (p_cam_i − t)      (X = T_cam_radar: p_cam = R q + t)

    solve_offset=True adds a (3 params) with a Gaussian PRIOR at apex0 of width
    offset_prior_sigma (MAP). This never hurts a good hand-measurement, repairs a
    bad one, and can recover the offset from scratch given close-range, high-tilt
    poses (offset is unobservable at long range where cross-range noise ≫ offset).
    Set offset_prior_sigma<=0 for a free (un-regularised) offset.

    Returns dict: R, t, apex, cov(6×6 on [rotvec,t]), cov_data(6×6, prior rows
    EXCLUDED), apex_sigma(3), inlier_mask, rms_sigma, n_in, cost, solved_offset.

    `init=(R0,t0)` overrides the starting guess — used by calibrate_extrinsic to
    try more than one basin instead of trusting whichever one the prior sits in.
    """
    P = np.asarray(P_radar, float)
    Rb = np.asarray(board_R, float); Tb = np.asarray(board_t, float)
    a0 = np.asarray(apex0, float)
    k = 3 if use_elevation else 2
    prior = solve_offset and offset_prior_sigma and offset_prior_sigma > 0
    ext_rot_prior = R_prior is not None and rot_prior_sigma and rot_prior_sigma > 0
    ext_t_prior = t_prior is not None and t_prior_sigma and t_prior_sigma > 0

    def unpack(x):
        R = Rot.from_rotvec(x[:3]).as_matrix(); t = x[3:6]
        a = x[6:9] if solve_offset else a0
        return R, t, a

    def residuals(x, idx):
        R, t, a = unpack(x)
        out = []
        for i in idx:
            pr = R.T @ (_cam_apex(Rb[i], Tb[i], a) - t)
            rp = cart_to_raz(pr); rm = cart_to_raz(P[i])
            out.append((rm[0] - rp[0]) / sig_r); out.append(_wrap(rm[1] - rp[1]) / sig_az)
            if use_elevation:
                out.append(_wrap(rm[2] - rp[2]) / sig_el)
        if prior:
            out.extend(list((x[6:9] - a0) / offset_prior_sigma))
        # extrinsic priors (MAP): pin poorly-observed DOFs to a known mounting
        if ext_rot_prior:
            dr = Rot.from_matrix(R_prior.T @ R).as_rotvec()   # geodesic rotation error
            out.extend(list(dr / rot_prior_sigma))
        if ext_t_prior:
            out.extend(list((t - np.asarray(t_prior, float)) / t_prior_sigma))
        return np.array(out)

    def per_point_sigma(x, idx):
        R, t, a = unpack(x)
        s = []
        for i in idx:
            pr = R.T @ (_cam_apex(Rb[i], Tb[i], a) - t)
            rp = cart_to_raz(pr); rm = cart_to_raz(P[i])
            d = [(rm[0] - rp[0]) / sig_r, _wrap(rm[1] - rp[1]) / sig_az]
            if use_elevation:
                d.append(_wrap(rm[2] - rp[2]) / sig_el)
            s.append(np.linalg.norm(d) / np.sqrt(k))
        return np.array(s)

    # init: caller-supplied, else the extrinsic prior, else Cartesian Kabsch
    # (apex fixed at a0). NOTE: initialising from the prior alone is what lets a
    # WRONG prior go unchallenged — calibrate_extrinsic always tries Kabsch too.
    if init is not None:
        R0 = np.asarray(init[0], float); t0 = np.asarray(init[1], float)
    elif ext_rot_prior and ext_t_prior:
        R0 = R_prior; t0 = np.asarray(t_prior, float)
    else:
        Q0 = np.array([_cam_apex(Rb[i], Tb[i], a0) for i in range(len(P))])
        R0, t0 = kabsch(P, Q0)
    x = np.concatenate([Rot.from_matrix(R0).as_rotvec(), t0] + ([a0] if solve_offset else []))
    allidx = np.arange(len(P)); mask = np.ones(len(P), bool)
    sol = None; fit_mask = mask.copy()
    for _ in range(max_iter):
        fit_mask = mask.copy()                      # the set sol.jac corresponds to
        sol = least_squares(residuals, x, args=(allidx[fit_mask],),
                            method='trf', loss='huber', f_scale=huber, max_nfev=6000)
        x = sol.x
        pn = per_point_sigma(x, allidx)
        new = pn < reject_sigma
        # Keep a floor of inliers — but never ask for more points than exist.
        # (The old code indexed np.sort(pn)[3] unconditionally, so a 3-capture
        # solve, which ~/solve explicitly permits, died with an IndexError.)
        floor = min(len(P), max(4, int(0.5 * len(P))))
        if new.sum() < floor:
            keep_n = min(len(P), max(4, int(0.6 * len(P))))
            new = pn <= np.sort(pn)[keep_n - 1]
        if np.array_equal(new, mask):
            break
        mask = new
    R, t, a = unpack(x)

    def _cov(J):
        try:
            return np.linalg.pinv(J.T @ J)
        except Exception:
            return np.full((len(x), len(x)), np.nan)

    # Two covariances, because they answer different questions:
    #   cov      — posterior, priors included: how well do we know the answer?
    #   cov_data — DATA ONLY: is this DOF actually observable from the poses?
    # Reporting only the former lets a tight prior masquerade as data-derived
    # confidence (a 15° prior alone shrinks the rotation 1σ from 6.8° to 5.3°),
    # which is exactly how an under-constrained rotation hides.
    full = _cov(sol.jac)
    n_data = k * int(fit_mask.sum())
    full_data = _cov(sol.jac[:n_data]) if n_data < sol.jac.shape[0] else full
    apex_sigma = np.sqrt(np.clip(np.diag(full)[6:9], 0, None)) if solve_offset else np.zeros(3)
    rms_sigma = float(np.sqrt((per_point_sigma(x, allidx[mask]) ** 2).mean()))
    return {'R': R, 't': t, 'apex': a, 'cov': full[:6, :6], 'cov_data': full_data[:6, :6],
            'apex_sigma': apex_sigma, 'inlier_mask': mask, 'rms_sigma': rms_sigma,
            'n_in': int(mask.sum()), 'cost': float(sol.cost),
            'solved_offset': bool(solve_offset)}


def calibrate_extrinsic(P, Rb, Tb, apex0, sig_r, sig_az, sig_el, use_elevation,
                        solve_offset, offset_prior_sigma,
                        R_prior, t_prior, rot_prior_sigma, t_prior_sigma,
                        huber, reject_sigma):
    """Solve twice — once from the DATA ALONE, once with the extrinsic prior —
    and report how far apart the two rotations are.

    The prior exists to stabilise DOFs the poses don't constrain, and it does
    that well. But a wrong prior is invisible to the residual: a prior 35° off,
    at σ=15°, still fits at 0.53σ while dragging the answer 15° away from truth.
    The old code made that worse by using the prior as the ONLY initialisation,
    so the data never got a chance to contradict it.

    So: fit the data on its own from a Kabsch start, then fit with the prior from
    BOTH starts and keep the lower-cost basin. The geodesic gap between the two
    answers is the diagnostic — small means prior and data agree, large means one
    of them is wrong and neither should be trusted yet.

    Returns (best, data_only, gap_deg)."""
    common = dict(use_elevation=use_elevation, solve_offset=solve_offset,
                  offset_prior_sigma=offset_prior_sigma,
                  huber=huber, reject_sigma=reject_sigma)
    free = robust_ml_calibrate(P, Rb, Tb, apex0, sig_r, sig_az, sig_el, **common)
    if R_prior is None or not rot_prior_sigma or rot_prior_sigma <= 0:
        return free, free, 0.0
    best = None
    for init in ((R_prior, t_prior), (free['R'], free['t'])):
        r = robust_ml_calibrate(P, Rb, Tb, apex0, sig_r, sig_az, sig_el,
                                R_prior=R_prior, t_prior=t_prior,
                                rot_prior_sigma=rot_prior_sigma,
                                t_prior_sigma=t_prior_sigma, init=init, **common)
        if best is None or r['cost'] < best['cost']:
            best = r
    return best, free, geodesic_deg(free['R'], best['R'])


def loo_cross_val(P, board_R, board_t, apex, sigmas, use_elevation,
                  R_prior=None, t_prior=None, rot_prior_sigma=None, t_prior_sigma=None):
    """Leave-one-out in MEASUREMENT space with the apex FIXED at the solved value
    (cheap, honest): refit extrinsic on N−1, predict the held-out radar
    measurement, score its error in sigma units. Returns (rms_sigma, max_sigma).

    The refits MUST carry the same priors as the real solve. Without them each
    fold is a different (unregularised) estimator, so on an under-constrained rig
    the folds scatter and LOO reads far worse than the residual — which looks
    like bad data but is really just a mismatched control."""
    n = len(P)
    if n < 5:
        return None
    sig_r, sig_az, sig_el = sigmas
    errs = []
    idx = np.arange(n)
    for i in range(n):
        m = idx != i
        r = robust_ml_calibrate(P[m], board_R[m], board_t[m], apex,
                                sig_r, sig_az, sig_el, use_elevation,
                                solve_offset=False, max_iter=3,
                                R_prior=R_prior, t_prior=t_prior,
                                rot_prior_sigma=rot_prior_sigma,
                                t_prior_sigma=t_prior_sigma)
        pr = r['R'].T @ (_cam_apex(board_R[i], board_t[i], apex) - r['t'])
        rp = cart_to_raz(pr); rm = cart_to_raz(P[i])
        d = [(rm[0] - rp[0]) / sig_r, _wrap(rm[1] - rp[1]) / sig_az]
        if use_elevation:
            d.append(_wrap(rm[2] - rp[2]) / sig_el)
        errs.append(float(np.linalg.norm(d) / np.sqrt(len(d))))
    errs = np.array(errs)
    return float(np.sqrt((errs ** 2).mean())), float(errs.max())


def condition_number(P):
    Pc = P - P.mean(0)
    if len(Pc) < 3:
        return float('inf')
    return float(np.linalg.cond(Pc))


def project(pt_cam, K, D):
    if pt_cam[2] <= 0:
        return None
    uv, _ = cv2.projectPoints(pt_cam.reshape(1, 3), np.zeros(3), np.zeros(3), K, D)
    return int(round(uv[0, 0, 0])), int(round(uv[0, 0, 1]))


# ─────────────── default parameters ───────────────
# One place for every parameter default. The two entry scripts
# (radar_camera_calib_static.py / _dynamic.py) pass a small `overrides` dict on
# top of this; CLI  -p name:=value  still overrides everything. Types matter to
# ROS 2 (int vs float), so keep 9 as int and 0.02 as float, etc.
DEFAULTS = {
    # camera
    'image_topic': '/zed/zed_node/left/image_rect_color',
    'info_topic': '/zed/zed_node/left/camera_info',
    # board (ChArUco) — must match the printed board
    'squares_x': 9, 'squares_y': 7, 'square_len': 0.020, 'marker_len': 0.015,
    'dictionary': 'DICT_4X4_50', 'min_corners': 6, 'max_reproj_px': 1.5,
    # planar pose ambiguity: reject a board pose when IPPE's two hypotheses are
    # within this reprojection-error ratio of each other (<=0 disables the guard)
    'pnp_ambiguity_ratio': 1.2,
    # 'auto' (decide from the topic name) | 'true' | 'false' — see _info
    'rectified_input': 'auto',
    # reflector apex offset in BOARD frame (m) + its prior width
    'reflector_offset_x': 0.10, 'reflector_offset_y': 0.23, 'reflector_offset_z': -0.05,
    'solve_offset': True, 'offset_prior_sigma_m': 0.05,
    # radar (IWR6843ISK 3DPC points_all: x,y,z,doppler,intensity)
    'radar_topic': '/radar1/radar/points_all',
    'pc_field_x': 'x', 'pc_field_y': 'y', 'pc_field_z': 'z',
    'pc_field_snr': 'intensity', 'pc_field_doppler': 'doppler', 'select_by': 'snr',
    'min_range': 0.5, 'max_range': 2.5,
    'max_abs_doppler': -1.0,   # >0 keep |dop|<= (STILL reflector); <=0 off
    'min_abs_doppler': -1.0,   # >0 keep |dop|>= (MOVING reflector); <=0 off
    'gate_radius': 0.5, 'range_gate_margin_m': 0.5,
    # background subtraction
    'bg_accum_frames': 15, 'bg_match_dist': 0.2, 'require_background': False,
    # radar range correction (ingest)
    'radar_range_scale': 1.0, 'radar_range_bias_m': 0.0,
    # radar measurement-noise model (drives ML weighting)
    'sigma_range_m': 0.05, 'sigma_az_deg': 3.0, 'sigma_el_deg': 10.0,
    'force_2d_radar': False, 'huber_f_scale': 1.5, 'reject_sigma': 4.0,
    # extrinsic prior (radar-in-camera): ON, this rig's mounting.
    # prior_quat_xyzw WINS over prior_rpy_deg when its norm is non-zero. Prefer
    # it: this rig's mounting is a ~90° frame swap, i.e. exactly the gimbal-lock
    # singularity of the rpy convention, where the euler triple is ambiguous and
    # a 1° change swings it by 90°. Every solve prints a ready-to-paste
    # prior_quat_xyzw line — use that, never the printed rpy.
    'use_extrinsic_prior': True,
    'prior_t_xyz': [0.207, 0.016, 0.020], 'prior_rpy_deg': [-90.0, -90.0, 0.0],
    'prior_quat_xyzw': [0.0, 0.0, 0.0, 0.0],   # zero norm = unset, fall back to rpy
    'prior_t_sigma_m': 0.05, 'prior_rot_sigma_deg': 15.0,
    # Warn (and fail the verdict) when the data-only and prior-pulled rotations
    # disagree by more than this — one of the two is wrong. Calibrated on
    # synthetic sweeps: a CORRECT prior lands at 1.4–1.7°, 10°-wrong at ~2.9°,
    # 35°-or-worse at 5.3–9.0°. 5° separates them.
    'prior_disagree_warn_deg': 5.0,
    # a DOF counts as unobservable when its data-only 1σ exceeds these
    'unobs_rot_sigma_deg': 10.0, 'unobs_t_sigma_mm': 100.0,
    # strict-capture gate
    'min_snr': 100.0,
    # capture / convergence
    'capture_mode': 'continuous',    # 'continuous' | 'auto' | 'manual'
    'stable_window': 12, 'stable_std': 0.01, 'stable_std_radar': 0.08,
    'min_baseline': 0.10, 'min_points': 20, 'sync_slop': 0.08,
    # validation thresholds / verdict
    'val_pass_reproj_px': 200.0, 'val_pass_3d_mm': 150.0, 'val_pass_bias_mm': 50.0,
    'measured_baseline_m': -1.0, 'baseline_tol_m': 0.03,
    # frames / output / display
    'parent_frame': 'zed_left_camera_optical_frame', 'child_frame': 'radar1_link',
    'camera_name': 'zed_left', 'radar_name': 'radar1', 'output_path': '',
    'publish_tf': True, 'debug_image': True,
    'debug_image_topic': '/radar_camera_calib/debug_image',
    'show_window': True, 'radar_watchdog_s': 3.0,
}


class RadarCameraCalib(Node):
    def __init__(self, overrides=None):
        super().__init__('radar_camera_calib')
        params = dict(DEFAULTS)
        if overrides:
            params.update(overrides)
        for name, value in params.items():        # CLI -p still overrides these
            self.declare_parameter(name, value)

        g = lambda n: self.get_parameter(n).value
        self.image_topic = g('image_topic')
        self.rectified_input = str(g('rectified_input')).lower()
        self.pnp_ratio = g('pnp_ambiguity_ratio')
        self.min_corners = int(g('min_corners')); self.max_reproj = g('max_reproj_px')
        self.dict, self.board, self.det, self.new_api, self.obj = build_board(
            int(g('squares_x')), int(g('squares_y')),
            g('square_len'), g('marker_len'), g('dictionary'))
        self.apex_board = np.array([g('reflector_offset_x'),
                                    g('reflector_offset_y'),
                                    g('reflector_offset_z')], float)
        self.apex_prior = self.apex_board.copy()          # prior centre (measured)
        self.solve_offset = bool(g('solve_offset'))
        self.offset_prior_sigma = g('offset_prior_sigma_m')
        self.bridge = CvBridge(); self.K = None; self.D = None

        self.radar_topic = g('radar_topic')
        self.fx, self.fy, self.fz = g('pc_field_x'), g('pc_field_y'), g('pc_field_z')
        self.fsnr, self.fdop = g('pc_field_snr'), g('pc_field_doppler')
        self.select_by = g('select_by')
        self.min_range = g('min_range'); self.max_range = g('max_range')
        self.max_abs_doppler = g('max_abs_doppler')
        self.min_abs_doppler = g('min_abs_doppler')
        self.gate_radius = g('gate_radius')
        self.range_gate_margin = g('range_gate_margin_m')
        self.bg_accum_frames = int(g('bg_accum_frames')); self.bg_match_dist = g('bg_match_dist')
        self.require_background = bool(g('require_background'))
        self.range_scale = g('radar_range_scale'); self.range_bias = g('radar_range_bias_m')
        self.sig_r = g('sigma_range_m')
        self.sig_az = np.radians(g('sigma_az_deg')); self.sig_el = np.radians(g('sigma_el_deg'))
        self.force_2d = bool(g('force_2d_radar'))
        self.huber = g('huber_f_scale'); self.reject_sigma = g('reject_sigma')
        self.use_ext_prior = bool(g('use_extrinsic_prior'))
        self.prior_R = self._prior_rotation(g('prior_quat_xyzw'), g('prior_rpy_deg'))
        self.prior_t = np.array(g('prior_t_xyz'), float) if self.use_ext_prior else None
        self.prior_t_sigma = g('prior_t_sigma_m')
        self.prior_rot_sigma = np.radians(g('prior_rot_sigma_deg'))
        self.prior_disagree = g('prior_disagree_warn_deg')
        self.unobs_rot = np.radians(g('unobs_rot_sigma_deg'))
        self.unobs_t = g('unobs_t_sigma_mm') / 1000.0
        self.min_snr = g('min_snr')

        self.capture_mode = g('capture_mode')
        self.stable_window = int(g('stable_window')); self.stable_std = g('stable_std')
        self.stable_std_radar = g('stable_std_radar')
        self.min_baseline = g('min_baseline'); self.min_points = int(g('min_points'))

        self.val_px = g('val_pass_reproj_px'); self.val_3d = g('val_pass_3d_mm')
        self.val_bias = g('val_pass_bias_mm')
        self.measured_baseline = g('measured_baseline_m'); self.baseline_tol = g('baseline_tol_m')

        self.parent_frame, self.child_frame = g('parent_frame'), g('child_frame')
        self.camera_name, self.radar_name = g('camera_name'), g('radar_name')
        op = g('output_path')
        self.output_path = op if op else f"extrinsic_{self.camera_name}__{self.radar_name}.yaml"
        self.publish_tf = bool(g('publish_tf'))
        self.want_debug = bool(g('debug_image'))
        self.show_window = bool(g('show_window'))
        self.window_name = 'radar_camera_calib — apex (green=matched) | reflector overlay'
        self._last_dbg = None; self._win_ok = None
        self.watchdog_s = g('radar_watchdog_s')

        # state
        self.win = []                 # rolling [(p_cam, p_radar), ...] for stability
        self.captures = []            # accepted [(p_cam, p_radar, snr, doppler), ...]
        self.n_ambiguous = 0          # board poses dropped as planar-ambiguous
        self.last_capture_cam = None
        self.manual_capture_req = False
        self.bg_radar = None          # (M,3) pooled background points (raw radar frame)
        self.bg_accum = None          # accumulation buffer while pooling
        self.X = None; self.rms = None
        self.last_radar_stamp = 0.0

        self.create_subscription(CameraInfo, g('info_topic'), self._info, qos_profile_sensor_data)
        img = message_filters.Subscriber(self, Image, g('image_topic'),
                                         qos_profile=qos_profile_sensor_data)
        radar = message_filters.Subscriber(self, PointCloud2, self.radar_topic,
                                           qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer([img, radar], 30, g('sync_slop'))
        self.sync.registerCallback(self._pair)
        # raw radar tap for background pooling + watchdog
        self.create_subscription(PointCloud2, self.radar_topic, self._radar_raw, qos_profile_sensor_data)

        self.create_subscription(Empty, '~/background', lambda _: self._start_background(), 1)
        self.create_subscription(Empty, '~/capture',    lambda _: self._on_capture(), 1)
        self.create_subscription(Empty, '~/solve',      lambda _: self._solve(force=True), 1)
        self.create_subscription(Empty, '~/reset',      lambda _: self._reset(), 1)
        self.create_subscription(Empty, '~/save',       lambda _: self._save_now(), 1)

        self.tf_static = tf2_ros.StaticTransformBroadcaster(self)
        self.dbg_pub = (self.create_publisher(Image, g('debug_image_topic'), 1)
                        if self.want_debug else None)
        self.create_timer(1.0, self._watchdog)
        if self.show_window:
            self.create_timer(0.05, self._gui)      # ~20 Hz native window refresh

        self.get_logger().info(
            f"[radar_camera_calib] OpenCV {cv2.__version__}, {'new' if self.new_api else 'old'} aruco\n"
            f"  camera : {g('image_topic')}  (parent/optical = {self.parent_frame})\n"
            f"  radar  : {self.radar_topic}  (child = {self.child_frame})\n"
            f"           fields x/y/z={self.fx}/{self.fy}/{self.fz} snr={self.fsnr} doppler={self.fdop}\n"
            f"           select_by={self.select_by}  range [{self.min_range},{self.max_range}] m\n"
            + (f"           range correction scale={self.range_scale} bias={self.range_bias} m\n"
               if (self.range_scale != 1.0 or self.range_bias != 0.0) else "")
            + f"  board  : {int(g('squares_x'))}x{int(g('squares_y'))} sq {g('square_len')} "
            f"mk {g('marker_len')} {g('dictionary')}\n"
            f"  apex_in_board (m): {self.apex_board.tolist()}\n"
            f"  capture: {self.capture_mode}  min_points {self.min_points}  "
            f"min_baseline {self.min_baseline} m\n"
            f"  output -> {self.output_path}\n"
            f"  1) rig OUT of scene → publish Empty on ~/background\n"
            f"  2) bring rig in, move it around the shared FoV, pausing at each spot\n"
            f"  3) auto-captures when still+moved; ~/solve, ~/save, ~/reset any time")

    # ── prior rotation / intrinsics / watchdog ──
    def _prior_rotation(self, quat, rpy):
        """Build the prior rotation, preferring the quaternion.

        prior_rpy_deg is a trap for this rig: the mounting is a ~90° frame swap,
        which is the gimbal-lock singularity of the 'xyz' euler convention. There
        the triple is not unique, scipy zeroes the third angle, and a 1° change
        rewrites two of the three numbers by 90°. Pasting a printed rpy back in
        as the prior therefore injects a genuinely wrong prior. The quaternion
        has no such singularity."""
        if not self.use_ext_prior:
            return None
        q = np.array(quat, float)
        if q.shape == (4,) and np.linalg.norm(q) > 1e-6:
            R = Rot.from_quat(q / np.linalg.norm(q)).as_matrix()
            self.get_logger().info("extrinsic prior taken from prior_quat_xyzw")
            return R
        R = Rot.from_euler('xyz', rpy, degrees=True).as_matrix()
        locked, pitch = near_gimbal_lock(R)
        if locked:
            qq = Rot.from_matrix(R).as_quat()
            self.get_logger().warn(
                f"prior_rpy_deg {list(rpy)} sits at GIMBAL LOCK (pitch {pitch:+.1f}°). "
                f"The euler triple is ambiguous there — a 1° change rewrites it by 90°. "
                f"Pin the prior with the quaternion instead:\n"
                f"    -p prior_quat_xyzw:=\"[{qq[0]:.6f},{qq[1]:.6f},{qq[2]:.6f},{qq[3]:.6f}]\"")
        return R

    def _info(self, msg):
        if self.K is not None:
            return
        K = np.array(msg.k).reshape(3, 3)
        D = np.array(msg.d) if len(msg.d) else np.zeros(5)
        Pm = np.array(msg.p).reshape(3, 4) if len(msg.p) == 12 else None
        rect = ('rect' in self.image_topic if self.rectified_input == 'auto'
                else self.rectified_input in ('true', '1', 'yes'))
        # On a RECTIFIED image the valid intrinsics are P[:3,:3] with ZERO
        # distortion. Feeding the raw k/d of a rectified stream into solvePnP
        # bends the board pose, and board-rotation error is multiplied by the
        # ~25 cm apex offset before it ever reaches the extrinsic solve.
        if rect and Pm is not None and Pm[0, 0] > 0:
            if np.abs(D).max() > 1e-6:
                self.get_logger().warn(
                    f"'{self.image_topic}' looks rectified but camera_info carries "
                    f"non-zero distortion {np.round(D, 4).tolist()} — using P with zero D. "
                    f"Pass rectified_input:=false if the image really is raw.")
            K = Pm[:3, :3].copy(); D = np.zeros(5)
        self.K = K; self.D = D
        self.get_logger().info(
            f"intrinsics locked ({msg.width}x{msg.height}) "
            f"fx {K[0,0]:.1f} fy {K[1,1]:.1f}  "
            f"{'rectified (P, D=0)' if rect else 'raw (K, D)'}")

    def _watchdog(self):
        if self.last_radar_stamp == 0.0:
            return
        age = self.get_clock().now().nanoseconds * 1e-9 - self.last_radar_stamp
        if age > self.watchdog_s:
            self.get_logger().warn(
                f"no radar on {self.radar_topic} for {age:.0f}s — wrong topic / node down?",
                throttle_duration_sec=5.0)

    # ── radar parsing ──
    def _radar_correct(self, xyz):
        if self.range_scale == 1.0 and self.range_bias == 0.0:
            return xyz
        xyz = xyz * float(self.range_scale)
        if self.range_bias != 0.0:
            r = np.linalg.norm(xyz, axis=1, keepdims=True)
            r = np.where(r < 1e-6, 1e-6, r)
            xyz = xyz - self.range_bias * (xyz / r)
        return xyz

    def _read_radar(self, msg):
        """Return (xyz Nx3 corrected, snr N, doppler N) or (None,None,None)."""
        if not _HAVE_PC2:
            return None, None, None
        names = [f.name for f in msg.fields]
        want = [self.fx, self.fy, self.fz]
        has_snr = self.fsnr in names; has_dop = self.fdop in names
        if has_snr: want.append(self.fsnr)
        if has_dop: want.append(self.fdop)
        arr = list(pc2.read_points(msg, field_names=want, skip_nans=True))
        if not arr:
            return None, None, None
        arr = np.array([tuple(a) for a in arr], float)
        xyz = self._radar_correct(arr[:, :3])
        col = 3
        snr = arr[:, col] if has_snr else np.zeros(len(arr)); col += has_snr
        dop = arr[:, col] if has_dop else np.zeros(len(arr))
        return xyz, snr, dop

    def _radar_raw(self, msg):
        self.last_radar_stamp = self.get_clock().now().nanoseconds * 1e-9
        if self.bg_accum is None:
            return
        xyz, _, _ = self._read_radar(msg)
        if xyz is not None:
            self.bg_accum.append(xyz)
            if len(self.bg_accum) >= self.bg_accum_frames:
                self.bg_radar = np.vstack(self.bg_accum)
                n = len(self.bg_radar); self.bg_accum = None
                self.get_logger().info(
                    f"background pooled: {self.bg_accum_frames} frames, {n} pts "
                    f"(match dist {self.bg_match_dist} m). Bring the rig in.")

    def _start_background(self):
        self.bg_accum = []
        self.get_logger().info(
            f"pooling {self.bg_accum_frames} background frames — keep the rig OUT of the scene…")

    def _select_radar(self, xyz, snr, dop, predicted=None, cam_range=None):
        """Gate the cloud, then pick the reflector return (highest SNR among the
        survivors). Gates, in order of strength:
          • range window [min_range, max_range]
          • optional |doppler| window (static reflector ≈ 0)
          • optional background subtraction
          • CAMERA-RANGE gate (bootstrap): keep points whose range matches the
            camera's board distance |p_cam| within range_gate_margin. Range is
            rotation-invariant, so this works BEFORE any extrinsic exists and
            rejects far clutter — "highest SNR AROUND THE BOARD".
          • once an extrinsic exists, a tight 3-D proximity gate to the predicted
            apex supersedes the range gate."""
        if xyz is None or len(xyz) == 0:
            return None, None, None, 0
        n0 = len(xyz)
        rng = np.linalg.norm(xyz, axis=1)
        keep = (rng >= self.min_range) & (rng <= self.max_range); n_rng = int(keep.sum())
        if self.max_abs_doppler > 0:
            keep &= (np.abs(dop) <= self.max_abs_doppler)
        if self.min_abs_doppler > 0:
            keep &= (np.abs(dop) >= self.min_abs_doppler)   # MOVING reflector only
        n_dop = int(keep.sum())
        if self.bg_radar is not None and len(self.bg_radar):
            diff = xyz[:, None, :] - self.bg_radar[None, :, :]
            mind = np.sqrt((diff ** 2).sum(2)).min(1)
            keep &= (mind > self.bg_match_dist)
        n_bg = int(keep.sum())
        # Reliable bootstrap: range-around-board (rotation-invariant) — ALWAYS on.
        if cam_range is not None and self.range_gate_margin > 0:
            keep &= (np.abs(rng - cam_range) <= self.range_gate_margin)
        n_cam = int(keep.sum())
        # Optional tighten around the predicted apex (from solve OR prior). If the
        # prior is imperfect and this would empty the set, KEEP the range-gated set.
        if predicted is not None:
            tight = keep & (np.linalg.norm(xyz - predicted, axis=1) <= self.gate_radius)
            if tight.any():
                keep = tight
        n_gated = int(keep.sum())

        def _diag(reason):
            self.get_logger().info(
                f"[gate] {reason}: total {n0} → range[{self.min_range},{self.max_range}] {n_rng}"
                + (f" → dop {n_dop}" if self.max_abs_doppler > 0 else "")
                + (f" → bg(>{self.bg_match_dist}m) {n_bg}" if self.bg_radar is not None else "")
                + (f" → cam±{self.range_gate_margin}m {n_cam}" if cam_range is not None else "")
                + f" → final {n_gated}"
                + (f"  (cam_range {cam_range:.2f}m)" if cam_range is not None else ""),
                throttle_duration_sec=1.5)

        if n_gated == 0:
            _diag("REJECTED-ALL")
            return None, None, None, 0
        xg, sg, dg = xyz[keep], snr[keep], dop[keep]
        if self.select_by == 'snr' and np.any(np.isfinite(sg)) and sg.max() > 0:
            idx = int(np.argmax(sg))             # ← highest-SNR survivor = the trihedral
        elif predicted is not None:
            idx = int(np.argmin(np.linalg.norm(xg - predicted, axis=1)))
        else:
            idx = int(np.argmin(np.linalg.norm(xg, axis=1)))
        if self.min_snr > 0 and sg[idx] < self.min_snr:
            _diag(f"best snr {sg[idx]:.0f} < min_snr {self.min_snr:.0f}")
            return None, None, None, n_gated     # too weak → likely mis-associated, skip
        return xg[idx], float(sg[idx]), float(dg[idx]), n_gated

    # ── camera: board pose → apex in camera frame ──
    def _board_pose(self, objp, cc):
        """ChArUco pose, with the planar two-fold ambiguity handled.

        A planar target always has two poses that project almost identically.
        SOLVEPNP_ITERATIVE silently returns one of them, and at this rig's scale
        (160×120 mm board, fx≈500) that is ~3° of board-rotation noise at 2 m
        with ~10% outright flips (>15°) — measured. Every one of those degrees is
        amplified by the 25 cm apex offset before it reaches the solver, and a
        flip is a 14 cm gross outlier.

        IPPE returns BOTH hypotheses with their reprojection errors. When the two
        are too close to call the pose is genuinely ambiguous, so we drop the
        frame instead of gambling; otherwise we polish the winner.
        Returns (rvec, tvec, ambiguous)."""
        op = np.ascontiguousarray(objp, np.float32).reshape(-1, 1, 3)
        ip = np.ascontiguousarray(cc, np.float32).reshape(-1, 1, 2)
        if self.pnp_ratio > 0 and len(op) >= 4:
            try:
                n, rvs, tvs, rep = cv2.solvePnPGeneric(
                    op, ip, self.K, self.D, flags=cv2.SOLVEPNP_IPPE)
                if n >= 1:
                    e = np.asarray(rep, float).ravel()
                    if n >= 2 and e[0] > 1e-9 and (e[1] / e[0]) < self.pnp_ratio:
                        return None, None, True
                    ok, rvec, tvec = cv2.solvePnP(
                        op, ip, self.K, self.D, rvs[0].copy(), tvs[0].copy(),
                        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
                    return (rvec, tvec, False) if ok else (rvs[0], tvs[0], False)
            except cv2.error:
                pass            # IPPE is fussy about degenerate sets — fall through
        ok, rvec, tvec = cv2.solvePnP(op, ip, self.K, self.D,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        return (rvec, tvec, False) if ok else (None, None, False)

    def _apex_in_camera(self, gray):
        if self.new_api:
            cc, cid, _, _ = self.det.detectBoard(gray)
        else:
            mc, mids, _ = cv2.aruco.detectMarkers(gray, self.dict)
            if mids is None or len(mids) == 0:
                return None, None, 0, None
            _, cc, cid = cv2.aruco.interpolateCornersCharuco(mc, mids, gray, self.board)
        n = 0 if cid is None else len(cid)
        if n < self.min_corners:
            return None, None, n, None
        objp = self.obj[cid.flatten()]
        rvec, tvec, ambiguous = self._board_pose(objp, cc)
        if ambiguous:
            self.n_ambiguous += 1
            self.get_logger().info(
                f"board pose AMBIGUOUS ({n} corners) — frame skipped. The board is "
                f"too small/flat in the image to pin its rotation; move CLOSER or "
                f"TILT it more (expect this past ~1.5 m with a 160 mm board). "
                f"{self.n_ambiguous} so far; relax with pnp_ambiguity_ratio "
                f"(now {self.pnp_ratio}, 0 disables the guard).",
                throttle_duration_sec=3.0)
            return None, None, n, None
        if rvec is None:
            return None, None, n, None
        proj, _ = cv2.projectPoints(objp, rvec, tvec, self.K, self.D)
        reproj = float(cv2.norm(cc, proj, cv2.NORM_L2) / len(proj))
        if reproj > self.max_reproj:
            return None, reproj, n, None
        R = cv2.Rodrigues(rvec)[0]
        p_cam = (R @ self.apex_board + tvec[:, 0])
        return p_cam, reproj, n, (rvec, tvec)

    # ── synced pair ──
    def _pair(self, a_img, radar_msg):
        if self.K is None:
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(a_img, 'bgr8')
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}"); return
        p_cam, reproj, n, pose = self._apex_in_camera(gray)
        xyz, snr, dop = self._read_radar(radar_msg)
        # Predict the reflector's radar location and search only AROUND it:
        #   p_cam already = R_board·(measured apex offset) + t_board, so |p_cam|
        #   and the predicted radar point both bake in the OFFSET PRIOR. Once an
        #   extrinsic is solved we use it; before that, the EXTRINSIC PRIOR (if
        #   given) predicts it too — so selection is tight from the first frame.
        #   The offset and extrinsic are still fully optimised in _solve.
        predicted = None
        if p_cam is not None:
            if self.X is not None:
                predicted = self.X[:3, :3].T @ (p_cam - self.X[:3, 3])       # camera → radar
            elif self.use_ext_prior and self.prior_R is not None:
                predicted = self.prior_R.T @ (p_cam - self.prior_t)          # prior-predicted
        cam_range = float(np.linalg.norm(p_cam)) if p_cam is not None else None
        p_radar, snr_i, dop_i, n_gated = self._select_radar(xyz, snr, dop, predicted, cam_range)

        self._publish_debug(bgr, pose, p_cam, xyz, n, reproj,
                            p_radar is not None, n_gated, p_radar)

        if p_cam is None:
            self.get_logger().info(f"no board (n={n}, reproj={reproj})", throttle_duration_sec=2.0)
            return
        if self.require_background and self.bg_radar is None:
            self.get_logger().info("waiting for ~/background (require_background=True)",
                                   throttle_duration_sec=3.0)
            return
        if p_radar is None:
            self.get_logger().info(f"no gated radar return (of {0 if xyz is None else len(xyz)})",
                                   throttle_duration_sec=2.0)
            return

        Rb = cv2.Rodrigues(pose[0])[0]; tb = pose[1][:, 0]

        # CONTINUOUS mode: no stillness required. Sweep the (moving) reflector
        # through the FoV; capture the current frame whenever it has moved
        # min_baseline from the last capture. A tiny 3-frame average tames the
        # per-frame radar noise without needing the rig to stop. Pair this with
        # a moving reflector + min_abs_doppler (or the points_dynamic topic) so
        # Doppler isolates it from static clutter.
        if self.capture_mode == 'continuous':
            self.win.append((p_cam, p_radar, snr_i, dop_i, Rb, tb))
            if len(self.win) > 3:
                self.win.pop(0)
            moved = (self.last_capture_cam is None or
                     np.linalg.norm(p_cam - self.last_capture_cam) >= self.min_baseline)
            if moved:
                self._accept(force=True)
            else:
                self.get_logger().info(
                    f"sweep… cam {p_cam.round(3).tolist()} radar {p_radar.round(3).tolist()} "
                    f"snr {snr_i:.0f} captures {len(self.captures)} (move {self.min_baseline*100:.0f}cm for next)",
                    throttle_duration_sec=0.8)
            return

        self.win.append((p_cam, p_radar, snr_i, dop_i, Rb, tb))
        if len(self.win) > self.stable_window:
            self.win.pop(0)
        stable = False
        if len(self.win) >= self.stable_window:
            cams = np.array([w[0] for w in self.win]); rads = np.array([w[1] for w in self.win])
            cstd = float(np.linalg.norm(cams.std(0))); rstd = float(np.linalg.norm(rads.std(0)))
            stable = (cstd < self.stable_std and rstd < self.stable_std_radar)
            self.get_logger().info(
                f"apex cam {p_cam.round(3).tolist()} radar {p_radar.round(3).tolist()} "
                f"snr {snr_i:.1f} still? {stable} (cam σ {cstd*1000:.0f}mm) captures {len(self.captures)}",
                throttle_duration_sec=1.0)

        if self.capture_mode == 'auto' and stable:
            self._accept()
        elif self.manual_capture_req and len(self.win) >= max(3, self.stable_window // 2):
            self.manual_capture_req = False
            self._accept(force=True)

    def _accept(self, force=False):
        p_cam = np.mean([w[0] for w in self.win], 0)
        p_radar = np.mean([w[1] for w in self.win], 0)
        snr_i = float(np.mean([w[2] for w in self.win]))
        dop_i = float(np.mean([w[3] for w in self.win]))
        # board pose held still during capture: mean translation, averaged rotation
        tb = np.mean([w[5] for w in self.win], 0)
        Rb = Rot.from_matrix(np.array([w[4] for w in self.win])).mean().as_matrix()
        if (not force and self.last_capture_cam is not None):
            d = float(np.linalg.norm(p_cam - self.last_capture_cam))
            if d < self.min_baseline:
                self.get_logger().info(
                    f"steady but only {d*100:.0f} cm from last capture "
                    f"(need {self.min_baseline*100:.0f} cm) — MOVE the rig to a NEW spot",
                    throttle_duration_sec=2.0)
                return
        self.captures.append({'p_radar': p_radar, 'snr': snr_i, 'dop': dop_i,
                              'Rb': Rb, 'tb': tb})
        self.last_capture_cam = p_cam; self.win.clear()
        self.get_logger().info(
            f"*** CAPTURED #{len(self.captures)}  cam {p_cam.round(3).tolist()}  "
            f"radar {p_radar.round(3).tolist()}  snr {snr_i:.1f} ***")
        self._solve()

    def _on_capture(self):
        self.manual_capture_req = True
        self.get_logger().info("manual capture requested — hold the rig still")

    # ── solve + validation ──
    def _solve(self, force=False):
        need = 3 if force else self.min_points
        if len(self.captures) < max(3, need):
            if force:
                self.get_logger().warn(f"need ≥3 captures (have {len(self.captures)})")
            return
        P = np.array([c['p_radar'] for c in self.captures])       # radar measurements
        Rb = np.array([c['Rb'] for c in self.captures])           # board rotations
        Tb = np.array([c['tb'] for c in self.captures])           # board translations

        # Detect a 2-D radar: one that reports z == 0 for EVERY point. That is
        # the only thing that justifies dropping the elevation residual.
        # The old test used std(|z|/r) > 0.01, which measures how much elevation
        # DIVERSITY the poses happened to have — so a genuine 3-D radar swept
        # horizontally got silently demoted to 2-D, throwing away real elevation
        # measurements. Measured cost of that mistake: rotation error 3.6° → 5.6°
        # and the rotation 1σ doubles. Thin elevation spread is a collection
        # problem (reported below in RADAR spread), not a sensor capability.
        radar_is_2d = float(np.abs(P[:, 2]).max()) < 1e-3
        use_el = (not self.force_2d) and not radar_is_2d

        # ── the accurate estimator: measurement-space ML + robust rejection,
        #    optionally jointly refining the apex offset (MAP toward measured).
        #    Solved BOTH with and without the extrinsic prior so a wrong prior
        #    cannot quietly own the answer (see calibrate_extrinsic). ──
        r, r_free, prior_gap = calibrate_extrinsic(
            P, Rb, Tb, self.apex_prior, self.sig_r, self.sig_az, self.sig_el,
            use_el, self.solve_offset, self.offset_prior_sigma,
            self.prior_R, self.prior_t, self.prior_rot_sigma, self.prior_t_sigma,
            self.huber, self.reject_sigma)
        R, t = r['R'], r['t']; mask = r['inlier_mask']; cov = r['cov']
        self.apex_board = r['apex']                    # use refined offset downstream
        self.X = np.eye(4); self.X[:3, :3] = R; self.X[:3, 3] = t
        # camera apex per capture, using the (possibly refined) offset
        Q = np.array([(_cam_apex(Rb[i], Tb[i], self.apex_board)) for i in range(len(P))])
        Pin, Qin = P[mask], Q[mask]
        self.rms = rms_3d(Pin, Qin, R, t)             # in-sample 3-D RMS (inliers), for overlay/text

        cond = condition_number(Pin)
        span = np.linalg.svd(Pin - Pin.mean(0), compute_uv=False)
        planar = span[2] < max(1e-3, 0.02 * span[0])
        q = Rot.from_matrix(R).as_quat()
        with warnings.catch_warnings():      # we detect and explain the lock ourselves
            warnings.simplefilter('ignore')
            rpy = Rot.from_matrix(R).as_euler('xyz', degrees=True)
        tmag = float(np.linalg.norm(t))
        n_out = int((~mask).sum())

        # diversity guard: the extrinsic is only observable if the RADAR points
        # span range + azimuth (+ elevation). Warn loudly when they don't.
        raz = np.array([cart_to_raz(p) for p in Pin])
        rng_span = float(raz[:, 0].max() - raz[:, 0].min())
        az_span = float(np.degrees(raz[:, 1].max() - raz[:, 1].min()))
        el_span = float(np.degrees(raz[:, 2].max() - raz[:, 2].min()))
        low_div = rng_span < 0.20 or az_span < 20.0
        div_msg = (f"  RADAR spread: range {rng_span*100:.0f} cm, az {az_span:.0f}°, el {el_span:.0f}°"
                   + ("   !! TOO CLUSTERED — move the rig NEAR↔FAR and LEFT↔RIGHT; "
                      "extrinsic under-constrained until this grows" if low_div else "  ✓"))

        # Per-DOF 1σ, inflated by the residual. pinv(JᵀJ) alone assumes the noise
        # model is exactly right; when the fit sits at 2.3σ the real uncertainty
        # is ~2.3× bigger, so reporting the raw number flatters a bad solve.
        # DATA-ONLY covariance drives the observability call: with the prior rows
        # included, the prior's own width reads as data-derived confidence.
        infl = max(1.0, r['rms_sigma'])
        dsig = np.sqrt(np.clip(np.diag(cov), 0, None)) * infl
        dsig_data = np.sqrt(np.clip(np.diag(r['cov_data']), 0, None)) * infl
        rot_sig_deg = np.degrees(dsig[:3]); t_sig_mm = dsig[3:6] * 1000
        rot_sig_data = np.degrees(dsig_data[:3]); t_sig_data = dsig_data[3:6] * 1000
        unobs = [nm for nm, s in zip(('rot_x', 'rot_y', 'rot_z'), dsig_data[:3])
                 if s > self.unobs_rot]
        unobs += [nm for nm, s in zip(('t_x', 't_y', 't_z'), dsig_data[3:6])
                  if s > self.unobs_t]

        locked, pitch = near_gimbal_lock(R)
        lines = [
            f"\n=== T_{self.parent_frame}_{self.child_frame}  (camera ← radar) ===",
            f"  method   : measurement-space ML ({'3-D' if use_el else '2-D, no elevation'}), "
            f"Huber+reject   inliers {r['n_in']}/{len(P)}" + (f"  ({n_out} rejected)" if n_out else ""),
            f"  captures {len(self.captures)}   in-sample RMS {self.rms*1000:.1f} mm   "
            f"residual {r['rms_sigma']:.2f} σ   cond {cond:.1f}",
            f"  xyz (m) : {t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}   |t| {tmag*100:.1f} cm",
            f"  quat    : {q[0]:+.6f} {q[1]:+.6f} {q[2]:+.6f} {q[3]:+.6f}   "
            f"(axis-angle {np.degrees(np.linalg.norm(Rot.from_matrix(R).as_rotvec())):.2f}°)",
            "  ROTATION (read THIS, not rpy):",
        ]
        lines += [f"      {s}" for s in axis_mapping(R)]
        lines += [
            f"  rpy(deg): {rpy[0]:+.2f} {rpy[1]:+.2f} {rpy[2]:+.2f}"
            + ("   !! AT GIMBAL LOCK (pitch %+.1f°) — this triple is NOT unique and "
               "jumps ~90° between solves. It is a display artefact, not a rotation "
               "error. Compare rotations with the axis map or the quaternion."
               % pitch if locked else ""),
            f"  1σ rot  : {rot_sig_deg[0]:.2f} {rot_sig_deg[1]:.2f} {rot_sig_deg[2]:.2f} deg   "
            f"1σ t: {t_sig_mm[0]:.1f} {t_sig_mm[1]:.1f} {t_sig_mm[2]:.1f} mm  (posterior)",
            f"  1σ data : {rot_sig_data[0]:.2f} {rot_sig_data[1]:.2f} {rot_sig_data[2]:.2f} deg   "
            f"1σ t: {t_sig_data[0]:.1f} {t_sig_data[1]:.1f} {t_sig_data[2]:.1f} mm  "
            f"(priors EXCLUDED — what the poses alone pin down)",
            div_msg,
        ]
        # A wrong prior fits as well as a right one, so the residual can't catch
        # it. Only the data-vs-prior disagreement can.
        if self.prior_R is not None:
            lines.append(
                f"  prior   : data-only vs prior-pulled rotation differ by {prior_gap:.2f}° "
                f"(prior σ {np.degrees(self.prior_rot_sigma):.0f}°)"
                + ("   !! THE PRIOR AND THE DATA DISAGREE — one of them is wrong. "
                   "Check the axis map above against the physical mounting, then either "
                   "fix prior_quat_xyzw or set use_extrinsic_prior:=false and collect "
                   "wider azimuth." if prior_gap > self.prior_disagree else "  ✓ consistent"))
            lines.append(f"  data-only rotation (no prior) → "
                         + " | ".join(axis_mapping(r_free['R'])[0:1]))
        lines.append(
            "  paste-back prior (round-trip safe, unlike rpy):\n"
            f"      -p use_extrinsic_prior:=true "
            f"-p prior_quat_xyzw:=\"[{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f}]\" "
            f"-p prior_t_xyz:=\"[{t[0]:.4f},{t[1]:.4f},{t[2]:.4f}]\"")
        if unobs:
            lines.append("  !! WEAK/UNOBSERVABLE dof: " + ", ".join(unobs)
                         + ("  — 2-D radar can't see out-of-plane rotation or height; "
                            "fix those from CAD" if not use_el else
                            "  — add pose diversity (range/azimuth/height)"))
        if self.solve_offset:
            a = self.apex_board; asig = r['apex_sigma'] * 1000
            moved = np.linalg.norm(a - self.apex_prior) * 1000
            weak = np.any(asig > 30)
            lines.append(
                f"  apex off: [{a[0]:+.3f} {a[1]:+.3f} {a[2]:+.3f}] m  "
                f"1σ [{asig[0]:.0f} {asig[1]:.0f} {asig[2]:.0f}] mm  "
                f"(moved {moved:.0f} mm from measured)"
                + ("  !! offset weakly observed — trusting your prior; add CLOSE-range "
                   "HIGH-TILT poses to calculate it" if weak else "  ✓ data-determined"))
        loo = loo_cross_val(P[mask], Rb[mask], Tb[mask], self.apex_board,
                            (self.sig_r, self.sig_az, self.sig_el), use_el,
                            R_prior=self.prior_R, t_prior=self.prior_t,
                            rot_prior_sigma=self.prior_rot_sigma,
                            t_prior_sigma=self.prior_t_sigma)
        if loo is not None:
            loo_s, loo_max = loo
            lines.append(f"  LOO CV  : {loo_s:.2f} σ (max {loo_max:.2f})"
                         + ("  !! ≫1 — poses too few/clustered or noise underestimated"
                            if loo_s > 2.5 else ""))
        if planar:
            lines.append("  !! radar inliers PLANAR/collinear — vary height & range")
        elif cond > 200:
            lines.append(f"  !! high condition ({cond:.0f}) — spread poses more in X/Y/Z")

        # Cartesian cross-checks (inliers) for the human-readable verdict
        res = (Pin @ R.T + t) - Qin
        bias_mm = res.mean(0) * 1000
        errs_px = []
        for i in range(len(Pin)):
            uv_t = project(Qin[i], self.K, self.D); uv_p = project(Pin[i] @ R.T + t, self.K, self.D)
            if uv_t and uv_p:
                errs_px.append(float(np.linalg.norm(np.subtract(uv_p, uv_t))))
        mean_px = float(np.mean(errs_px)) if errs_px else float('nan')
        mean_3d = float(np.linalg.norm(res, axis=1).mean()) * 1000
        lines.append(f"  signed residual (pred−cam) mm: X {bias_mm[0]:+.0f} Y {bias_mm[1]:+.0f} "
                     f"Z {bias_mm[2]:+.0f}   reproj {mean_px:.1f} px   mean 3-D {mean_3d:.1f} mm")

        # range diagnostic (camera range vs radar range, inliers)
        cam_r = np.linalg.norm(Qin, axis=1); rad_r = np.linalg.norm(Pin, axis=1)
        if len(Pin) >= 2:
            A = np.vstack([rad_r, np.ones_like(rad_r)]).T
            (a, b), *_ = np.linalg.lstsq(A, cam_r, rcond=None)
            if abs(a - 1) > 0.03 or abs(b) > 0.05:
                lines.append(f"  range fit: cam_r = {a:.3f}·radar_r {b:+.3f} m  → "
                             f"try radar_range_scale={1/a:.4f} bias={-b:+.4f}")

        if self.measured_baseline > 0:
            d = abs(tmag - self.measured_baseline)
            lines.append(f"  baseline: |t| {tmag*100:.1f} cm vs measured "
                         f"{self.measured_baseline*100:.1f} cm → Δ {d*100:.1f} cm "
                         f"[{'OK' if d <= self.baseline_tol else 'MISMATCH'}]")
        checks = [("reproj_px", mean_px, self.val_px), ("3d_mm", mean_3d, self.val_3d),
                  ("bias_mm", float(np.abs(bias_mm).max()), self.val_bias)]
        if self.prior_R is not None:
            checks.append(("prior_gap_deg", prior_gap, self.prior_disagree))
        verdict = all(v <= lim for _, v, lim in checks) and not unobs
        lines.append("  VERDICT : " + ("✔ GOOD  " if verdict else "✗ SUSPECT  ")
                     + "  ".join(f"{k} {v:.1f}/{lim:.0f}[{'P' if v<=lim else 'F'}]"
                                 for k, v, lim in checks))
        self.get_logger().info("\n".join(lines))

        if self.publish_tf:
            self._broadcast()
        self._save({'mean_px': mean_px, 'mean_3d': mean_3d, 'bias_mm': bias_mm, 'cond': cond,
                    'loo': loo, 'planar': planar, 'verdict': verdict, 'use_el': use_el,
                    'rot_sig_deg': rot_sig_deg, 't_sig_mm': t_sig_mm, 'unobs': unobs,
                    'rot_sig_data_deg': rot_sig_data, 't_sig_data_mm': t_sig_data,
                    'prior_gap_deg': prior_gap, 'R_free': r_free['R'], 'gimbal': locked,
                    'n_in': r['n_in'], 'n_out': n_out, 'rms_sigma': r['rms_sigma'],
                    'apex': self.apex_board, 'apex_sigma': r['apex_sigma'],
                    'apex_prior': self.apex_prior, 'solved_offset': r['solved_offset']})

    def _reset(self):
        self.captures.clear(); self.win.clear(); self.last_capture_cam = None
        self.X = None; self.rms = None
        self.apex_board = self.apex_prior.copy()      # drop the refined offset
        self.get_logger().info("reset — captures cleared (background kept)")

    def _save_now(self):
        if self.X is not None:
            self._solve(force=True)
        else:
            self.get_logger().warn("nothing to save yet")

    # ── outputs ──
    def _broadcast(self):
        m = TransformStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.parent_frame; m.child_frame_id = self.child_frame
        X = self.X
        m.transform.translation.x = float(X[0, 3]); m.transform.translation.y = float(X[1, 3])
        m.transform.translation.z = float(X[2, 3])
        q = Rot.from_matrix(X[:3, :3]).as_quat()
        (m.transform.rotation.x, m.transform.rotation.y,
         m.transform.rotation.z, m.transform.rotation.w) = map(float, q)
        self.tf_static.sendTransform(m)

    def _save(self, m):
        X = self.X; Xinv = np.linalg.inv(X)
        q = Rot.from_matrix(X[:3, :3]).as_quat(); rpy = Rot.from_matrix(X[:3, :3]).as_euler('xyz', degrees=True)
        qi = Rot.from_matrix(Xinv[:3, :3]).as_quat()
        data = {
            'parent_frame': self.parent_frame, 'child_frame': self.child_frame,
            'camera_name': self.camera_name, 'radar_name': self.radar_name,
            'method': 'measurement_space_ML_robust',
            'elevation_used': bool(m['use_el']),
            'n_captures': len(self.captures), 'n_inliers': m['n_in'], 'n_rejected': m['n_out'],
            'verdict_pass': bool(m['verdict']), 'planar_warning': bool(m['planar']),
            'unobservable_dof': m['unobs'],
            'in_sample_rms_mm': float(self.rms) * 1000,
            'residual_rms_sigma': float(m['rms_sigma']),
            'loo_cv_rms_sigma': (m['loo'][0] if m['loo'] else None),
            'sigma_1_rot_deg_xyz': [float(v) for v in m['rot_sig_deg']],
            'sigma_1_t_mm_xyz': [float(v) for v in m['t_sig_mm']],
            # priors excluded — what the poses alone actually determine
            'sigma_1_rot_deg_xyz_data_only': [float(v) for v in m['rot_sig_data_deg']],
            'sigma_1_t_mm_xyz_data_only': [float(v) for v in m['t_sig_data_mm']],
            'prior_data_rotation_gap_deg': float(m['prior_gap_deg']),
            'rpy_at_gimbal_lock': bool(m['gimbal']),
            'radar_axes_in_camera': axis_mapping(X[:3, :3]),
            'radar_noise_sigma_range_m': self.sig_r,
            'radar_noise_sigma_az_deg': float(np.degrees(self.sig_az)),
            'radar_noise_sigma_el_deg': float(np.degrees(self.sig_el)),
            'mean_reproj_px': float(m['mean_px']), 'mean_3d_mm': float(m['mean_3d']),
            'residual_bias_mm_xyz': [float(v) for v in m['bias_mm']],
            'condition_number': float(m['cond']),
            'radar_range_scale': self.range_scale, 'radar_range_bias_m': self.range_bias,
            # T_cam_radar : p_cam = R p_radar + t
            'T_cam_radar_translation': [float(v) for v in X[:3, 3]],
            'T_cam_radar_quaternion_xyzw': [float(v) for v in q],
            # rpy is DISPLAY ONLY — this rig's ~90° extrinsic sits at the 'xyz'
            # euler singularity, where the triple is not unique. Consume the
            # quaternion; never round-trip the rpy back in as a prior.
            'T_cam_radar_rpy_deg': [float(v) for v in rpy],
            # inverse
            'T_radar_cam_translation': [float(v) for v in Xinv[:3, 3]],
            'T_radar_cam_quaternion_xyzw': [float(v) for v in qi],
            'apex_offset_in_board_m': [float(v) for v in m['apex']],
            'apex_offset_solved': bool(m['solved_offset']),
            'apex_offset_1sigma_mm_xyz': [float(v) for v in m['apex_sigma'] * 1000],
            'apex_offset_measured_prior_m': [float(v) for v in m['apex_prior']],
            'static_tf_cmd': (f"ros2 run tf2_ros static_transform_publisher "
                              f"{X[0,3]:.6f} {X[1,3]:.6f} {X[2,3]:.6f} "
                              f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} "
                              f"{self.parent_frame} {self.child_frame}"),
            'stamp': self.get_clock().now().nanoseconds}
        try:
            with open(self.output_path, 'w') as f:
                for k, v in data.items():
                    f.write(f"{k}: {v}\n")
            with open(self.output_path.replace('.yaml', '.json'), 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"save failed: {e}")

    def _publish_debug(self, bgr, pose, p_cam, radar_xyz, n, reproj, got_radar,
                       n_gated, p_radar=None):
        if self.dbg_pub is None and not self.show_window:
            return
        try:
            h, w = bgr.shape[:2]
            # Project the WHOLE radar cloud into the image so you can SEE where the
            # radar points land vs the reflector. Use the solved extrinsic if we
            # have one, else the extrinsic prior. Yellow = generic radar point.
            Xr = self.X
            if Xr is None and self.use_ext_prior and self.prior_R is not None:
                Xr = np.eye(4); Xr[:3, :3] = self.prior_R; Xr[:3, 3] = self.prior_t
            if Xr is not None and radar_xyz is not None and len(radar_xyz):
                Pc = (radar_xyz @ Xr[:3, :3].T) + Xr[:3, 3]
                for p in Pc:
                    uv = project(p, self.K, self.D)
                    if uv and 0 <= uv[0] < w and 0 <= uv[1] < h:
                        cv2.circle(bgr, uv, 3, (0, 220, 220), -1)   # yellow dots
            # Highlight the SELECTED reflector point (magenta) with range + SNR,
            # projected the same way — check it lands on the real reflector.
            if p_radar is not None and Xr is not None:
                pc = Xr[:3, :3] @ np.asarray(p_radar) + Xr[:3, 3]
                uv = project(pc, self.K, self.D)
                if uv and 0 <= uv[0] < w and 0 <= uv[1] < h:
                    cv2.circle(bgr, uv, 10, (255, 0, 255), 2)
                    cv2.putText(bgr, f"radar r={np.linalg.norm(p_radar):.2f}m",
                                (uv[0] + 12, uv[1] + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            if pose is not None:
                rvec, tvec = pose
                cv2.drawFrameAxes(bgr, self.K, self.D, rvec, tvec, 0.05)
                uv = project(p_cam, self.K, self.D)
                if uv:
                    # reticle at the current apex estimate — aim the reflector here
                    col = (0, 255, 0) if got_radar else (0, 165, 255)
                    cv2.circle(bgr, uv, 12, col, 2)
                    cv2.line(bgr, (uv[0] - 18, uv[1]), (uv[0] + 18, uv[1]), col, 1)
                    cv2.line(bgr, (uv[0], uv[1] - 18), (uv[0], uv[1] + 18), col, 1)
                    tag = "apex (matched)" if got_radar else "apex (no radar)"
                    if self.solve_offset and self.X is None:
                        tag += " — offset unsolved, will shift"
                    cv2.putText(bgr, tag, (uv[0] + 14, uv[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            l1 = f"corners {n}  reproj {reproj if reproj else 0:.2f}px  gated {n_gated}  captures {len(self.captures)}"
            if self.rms is not None:
                l1 += f"  RMS {self.rms*1000:.0f}mm"
            l2 = ("BG: none (optional — range-gated around board)" if self.bg_radar is None
                  else "BG: set")
            cv2.putText(bgr, l1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(bgr, l2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 200, 255) if self.bg_radar is None else (0, 255, 0), 2)
            self._last_dbg = bgr
            if self.dbg_pub is not None:
                self.dbg_pub.publish(self.bridge.cv2_to_imgmsg(bgr, 'bgr8'))
        except Exception as e:
            self.get_logger().warn(f"debug image: {e}", throttle_duration_sec=5.0)

    def _gui(self):
        """Native OpenCV window (opt-in). Runs in the executor thread alongside
        spin; needs a display (X11). Degrades to the debug topic if none."""
        if not self.show_window or self._last_dbg is None or self._win_ok is False:
            return
        try:
            if self._win_ok is None:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self._win_ok = True
            cv2.imshow(self.window_name, self._last_dbg)
            cv2.waitKey(1)
        except Exception as e:
            self._win_ok = False
            self.get_logger().warn(
                f"show_window failed ({e}) — no display? Use "
                f"'rqt_image_view {self.get_parameter('debug_image_topic').value}' instead.")


def main(overrides=None):
    rclpy.init(); node = RadarCameraCalib(overrides)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.X is not None:
            node.get_logger().info(f"Final extrinsic in {node.output_path}")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()

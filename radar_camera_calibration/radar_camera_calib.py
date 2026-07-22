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
  • Kabsch provides the initial guess.
  • 2-D radar (no elevation) is auto-detected and the elevation residual is
    dropped — but then out-of-plane rotation and height are UNOBSERVABLE; the
    covariance readout flags exactly which DOFs are weak.

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
  • SELECTION (select_by) — among the surviving points, identify the reflector:
      'snr'     : take argmax(SNR); the trihedral is the strongest reflector.
      'cluster' : group the survivors (connected components at cluster_eps) and
                  take the SNR-WEIGHTED CENTROID of the reflector blob (the blob
                  holding the brightest return, or nearest the predicted apex).
                  More robust than a lone argmax spike — averages the blob's
                  returns down and rejects an isolated bright clutter point.
      'nearest' : nearest the predicted apex / radar origin.
  • DOPPLER↔MOTION CONSISTENCY (moving / hand-held rig) — while you sweep, the
    reflector is rigidly tied to the board, so its radar radial velocity must
    equal d|range|/dt from the CAMERA (v_pred = (p_cam−t_ext)·ṗ_cam/|p_cam−t_ext|,
    rotation-free). Hand/arm/body move at a DIFFERENT radial rate, so this
    separates the reflector from moving clutter where a static |doppler| gate
    cannot. Keeps only doppler-matching survivors, then argmax(SNR), with a
    fallback + auto-learned sign (use_doppler_consistency).

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
from datetime import datetime, timezone
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


def robust_ml_calibrate(P_radar, board_R, board_t, apex0,
                        sig_r, sig_az, sig_el, use_elevation=True,
                        solve_offset=True, offset_prior_sigma=0.03,
                        R_prior=None, t_prior=None,
                        rot_prior_sigma=None, t_prior_sigma=None,
                        huber=1.5, reject_sigma=4.0, reject_axis_sigma=0.0, max_iter=5):
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

    Returns dict: R, t, apex, cov(6×6 on [rotvec,t]), apex_sigma(3), inlier_mask,
    rms_sigma, n_in, solved_offset(bool).
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

    def per_point_axis_max(x, idx):
        """Largest single-axis |normalized residual| per point. Catches a bad
        match that spreads its error across two axes so its RMS (per_point_sigma)
        stays under reject_sigma — e.g. a multipath ghost wrong in range AND
        elevation but clean in azimuth."""
        R, t, a = unpack(x)
        s = []
        for i in idx:
            pr = R.T @ (_cam_apex(Rb[i], Tb[i], a) - t)
            rp = cart_to_raz(pr); rm = cart_to_raz(P[i])
            d = [abs(rm[0] - rp[0]) / sig_r, abs(_wrap(rm[1] - rp[1])) / sig_az]
            if use_elevation:
                d.append(abs(_wrap(rm[2] - rp[2])) / sig_el)
            s.append(max(d))
        return np.array(s)

    # init: prefer the extrinsic prior (robust when data is under-constrained),
    # else Cartesian Kabsch (apex fixed at a0)
    if ext_rot_prior and ext_t_prior:
        R0 = R_prior; t0 = np.asarray(t_prior, float)
    else:
        Q0 = np.array([_cam_apex(Rb[i], Tb[i], a0) for i in range(len(P))])
        R0, t0 = kabsch(P, Q0)
    x = np.concatenate([Rot.from_matrix(R0).as_rotvec(), t0] + ([a0] if solve_offset else []))
    allidx = np.arange(len(P)); mask = np.ones(len(P), bool)
    sol = None
    for _ in range(max_iter):
        sol = least_squares(residuals, x, args=(allidx[mask],),
                            method='trf', loss='huber', f_scale=huber, max_nfev=6000)
        x = sol.x
        pn = per_point_sigma(x, allidx)
        new = pn < reject_sigma
        if reject_axis_sigma and reject_axis_sigma > 0:      # opt-in per-axis cap
            new &= per_point_axis_max(x, allidx) < reject_axis_sigma
        if new.sum() < max(4, int(0.5 * len(P))):
            keep_n = max(4, int(0.6 * len(P)))
            new = pn <= np.sort(pn)[keep_n - 1]
        if np.array_equal(new, mask):
            break
        mask = new
    R, t, a = unpack(x)
    try:
        full = np.linalg.pinv(sol.jac.T @ sol.jac)
    except Exception:
        full = np.full((len(x), len(x)), np.nan)
    cov6 = full[:6, :6]
    apex_sigma = np.sqrt(np.clip(np.diag(full)[6:9], 0, None)) if solve_offset else np.zeros(3)
    rms_sigma = float(np.sqrt((per_point_sigma(x, allidx[mask]) ** 2).mean()))
    return {'R': R, 't': t, 'apex': a, 'cov': cov6, 'apex_sigma': apex_sigma,
            'inlier_mask': mask, 'rms_sigma': rms_sigma, 'n_in': int(mask.sum()),
            'solved_offset': bool(solve_offset)}


def loo_cross_val(P, board_R, board_t, apex, sigmas, use_elevation):
    """Leave-one-out in MEASUREMENT space with the apex FIXED at the solved value
    (cheap, honest): refit extrinsic on N−1, predict the held-out radar
    measurement, score its error in sigma units. Returns (rms_sigma, max_sigma)."""
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
                                solve_offset=False, max_iter=3)
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


def cluster_points(xyz, eps, min_size):
    """Group nearby radar points into clusters (single-linkage / connected
    components at distance `eps`), keeping only clusters of at least `min_size`
    points. No sklearn: the gated set is small, so an O(N²) union-find is ample
    and deterministic. Returns a list of index arrays (one per cluster), largest
    first. Used to identify the reflector as a compact bright BLOB rather than a
    single (noisy) highest-SNR spike."""
    xyz = np.asarray(xyz, float)
    n = len(xyz)
    if n == 0:
        return []
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    D = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= eps:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = [np.array(g) for g in groups.values() if len(g) >= min_size]
    out.sort(key=len, reverse=True)
    return out


# targets that make ROTATION (and, via the offset, translation) well-observed
DIVERSITY_TARGETS = {'pitch': 40.0, 'roll': 30.0, 'yaw': 40.0,       # board orientation spread, deg
                     'az': 40.0, 'el': 15.0, 'range': 0.30}          # radar point spread, deg / deg / m


def pose_diversity(Rb_list, P_list):
    """How diverse the collected STATIC poses are — the quantity that decides
    whether rotation can be recovered accurately (good |t|, bad R is the classic
    single-reflector failure). Two independent needs:

      • BOARD ORIENTATION spread (pitch/roll/yaw, in the camera optical frame:
        pitch=about X/right, yaw=about Y/down, roll=about Z/forward) — makes the
        apex OFFSET observable (board tilt), which in turn tightens translation.
      • RADAR POINT spread (azimuth/elevation/range) — the lever arm that makes
        the EXTRINSIC ROTATION observable (rot error ≈ cross-range_noise / extent).

    Returns {name: (value, target, ok)} for pitch/roll/yaw/az/el/range plus 'n'.
    Robust near gimbal lock: orientation spread is measured as the rotvec spread
    of each pose about the mean pose (small-angle roll/pitch/yaw of the board)."""
    out = {'n': len(Rb_list)}
    if len(Rb_list) >= 2:
        Rs = Rot.from_matrix(np.asarray(Rb_list))
        dv = np.degrees((Rs * Rs.mean().inv()).as_rotvec())          # (N,3): about cam X,Y,Z
        span = dv.max(0) - dv.min(0)
        for nm, i in (('pitch', 0), ('yaw', 1), ('roll', 2)):
            tgt = DIVERSITY_TARGETS[nm]; v = float(span[i])
            out[nm] = (v, tgt, v >= tgt)
    else:
        for nm in ('pitch', 'yaw', 'roll'):
            out[nm] = (0.0, DIVERSITY_TARGETS[nm], False)
    P = np.asarray(P_list, float)
    if len(P) >= 2:
        raz = np.array([cart_to_raz(p) for p in P])
        for nm, i, scale in (('range', 0, 1.0), ('az', 1, np.degrees(1)), ('el', 2, np.degrees(1))):
            tgt = DIVERSITY_TARGETS[nm]; v = float((raz[:, i].max() - raz[:, i].min()) * scale)
            out[nm] = (v, tgt, v >= tgt)
    else:
        for nm in ('range', 'az', 'el'):
            out[nm] = (0.0, DIVERSITY_TARGETS[nm], False)
    return out


class RadarCameraCalib(Node):
    def __init__(self, default_overrides=None):
        super().__init__('radar_camera_calib')
        self._defaults = dict(default_overrides or {})   # profile presets (static/dynamic scripts)
        self._param_names = []                            # recorded for the session dump

        def dp(name, val):
            self._param_names.append(name)
            return self.declare_parameter(name, self._defaults.get(name, val))
        # --- camera ---
        dp('image_topic', '/zed/zed_node/left/image_rect_color')
        dp('info_topic',  '/zed/zed_node/left/camera_info')
        # --- board (ChArUco) ---
        dp('squares_x', 9); dp('squares_y', 7)
        dp('square_len', 0.020); dp('marker_len', 0.015)
        dp('dictionary', 'DICT_4X4_50')
        dp('min_corners', 8); dp('max_reproj_px', 1.5)
        # --- reflector apex offset in the BOARD frame (metres) ---
        #   Best measured to ~cm; but the solver can also REFINE or fully
        #   CALCULATE it (see solve_offset). These values seed / prior-anchor it.
        dp('reflector_offset_x', 0.0)
        dp('reflector_offset_y', 0.0)
        dp('reflector_offset_z', 0.0)
        # jointly estimate the apex offset (MAP: free offset regularised toward
        # the measured value). Never worse than fixing it; repairs a bad measure;
        # can solve it from scratch given CLOSE-range, HIGH-TILT poses.
        dp('solve_offset', True)
        # prior width on the offset (m). Tight if you measured well; large/<=0 to
        # let the data drive it (set ~0.10 if you did NOT measure the offset).
        dp('offset_prior_sigma_m', 0.03)
        # --- radar (/points_all: x,y,z,snr,doppler) ---
        dp('radar_topic', '/points_all')
        dp('pc_field_x', 'x'); dp('pc_field_y', 'y'); dp('pc_field_z', 'z')
        dp('pc_field_snr', 'snr'); dp('pc_field_doppler', 'doppler')
        dp('select_by', 'snr')          # 'snr' | 'nearest' | 'cluster' (blob centroid)
        # 'cluster' selection: group the gated returns and take the SNR-weighted
        # centroid of the reflector blob (robust to a single noisy SNR spike).
        dp('cluster_eps', 0.15)         # m; points within this distance join one cluster
        dp('min_cluster_size', 2)       # min points to accept a cluster (else fall back to SNR)
        # cluster AROUND the estimated apex: once an extrinsic (solve) or extrinsic
        # prior predicts the apex, cluster only points within this radius of it.
        dp('cluster_apex_radius', 0.30) # m; window around the predicted apex to cluster in
        dp('cluster_strict', False)     # True → if NO cluster forms within that radius of the
                                        # predicted apex, REJECT the capture (no fall back to SNR)
        dp('min_range', 0.3); dp('max_range', 20.0)
        dp('max_abs_doppler', -1.0)     # >0 → keep |doppler| below this (still rig ≈0); <=0 disables
        dp('min_abs_doppler', -1.0)     # >0 → keep |doppler| ABOVE this (MOVING reflector); <=0 disables
        # --- Doppler ↔ motion consistency (for a MOVING / hand-held rig) ---
        #   While you sweep, the reflector is RIGIDLY tied to the board, so its
        #   radar radial velocity MUST equal the rate of change of the camera's
        #   range to the apex. Your hand / arm / body move too, but at a DIFFERENT
        #   radial velocity, so this separates the reflector from moving clutter
        #   far better than a static |doppler|≈0 gate ever can. Rotation-free:
        #       v_pred = (p_cam − t_ext) · ṗ_cam / |p_cam − t_ext|
        #   (needs only the small baseline t_ext, and ≈ d|p_cam|/dt even without it).
        #   This is the primary lever for "identifying the correct feature" while moving.
        dp('use_doppler_consistency', False)
        dp('doppler_match_tol', 0.30)   # m/s; keep radar pts whose doppler matches v_pred within this
        dp('doppler_sign', 'auto')      # 'auto' | '1' | '-1' — TI radial-velocity sign convention
        dp('min_motion_mps', 0.05)      # only apply the doppler gate when |ṗ_cam| exceeds this
        dp('gate_radius', 1.0)          # m; once X exists, radar pt must be within this of predicted apex
        # bootstrap gate BEFORE any extrinsic: keep radar pts whose range matches
        # the camera's board distance |p_cam| within this margin, then take the
        # highest SNR among them ("highest SNR around the board"). Rejects far
        # clutter without needing a background snapshot. <=0 disables.
        dp('range_gate_margin_m', 1.0)
        # --- background subtraction ---
        dp('bg_accum_frames', 15)       # radar frames pooled on ~/background
        dp('bg_match_dist', 0.2)        # m; live pt is "new" if farther than this from all bg pts
        dp('require_background', False) # if True, refuse to capture until background pooled
        # --- radar range correction (applied at ingest, before everything) ---
        dp('radar_range_scale', 1.0)    # multiply xyz (fixes proportional/units error)
        dp('radar_range_bias_m', 0.0)   # subtract along radial dir (fixes constant offset)
        # --- radar measurement noise (drives the ML solver's weighting) ---
        #   range is precise, angle is not; cross-range error ≈ range·sigma_az.
        #   Set from your radar's spec / resolution; relative sizes matter most.
        dp('sigma_range_m', 0.05)
        dp('sigma_az_deg', 2.0)
        dp('sigma_el_deg', 5.0)         # elevation is usually the worst (few el antennas)
        dp('force_2d_radar', False)     # True → ignore elevation entirely (2-D radar)
        dp('huber_f_scale', 1.5)        # robust-loss knee, in sigma units
        dp('reject_sigma', 4.0)         # drop a match whose RMS-across-axes residual exceeds this (sigma)
        dp('reject_axis_sigma', 0.0)    # opt-in (0=off): also drop a match if ANY single axis exceeds this (sigma) — catches multipath ghosts that hide under the RMS gate
        # --- extrinsic prior (MAP): pin poorly-observed DOFs to a known mounting.
        #     Give a rough measured/CAD radar-in-camera pose; the solve is
        #     regularised toward it AND initialised from it. Tight sigma = trust
        #     the prior; loose = let the data move it. Disabled by default. ---
        dp('use_extrinsic_prior', False)
        dp('prior_t_xyz', [0.0, 0.0, 0.0])       # radar position in camera frame (m)
        dp('prior_rpy_deg', [0.0, 0.0, 0.0])     # radar orientation in camera frame (xyz euler)
        dp('prior_t_sigma_m', 0.05)              # translation prior width
        dp('prior_rot_sigma_deg', 10.0)          # rotation prior width
        # --- strict-capture gate: reject a capture whose reflector SNR is below
        #     this (weak returns are the ones most likely mis-associated). 0=off ---
        dp('min_snr', 0.0)
        # --- capture / convergence ---
        dp('capture_mode', 'auto')      # 'auto' | 'manual'
        dp('stable_window', 12)
        dp('stable_std', 0.01)          # m; CAMERA (board) jitter to call it "still"
        dp('stable_std_radar', 0.08)    # m; RADAR jitter allowed (angular noise is cm-dm;
                                        # window-averaging beats it down — don't set this tight)
        dp('min_baseline', 0.15)        # m; min move between auto-captures
        dp('min_points', 6)
        dp('sync_slop', 0.06)
        # A moving, hand-held rig makes correspondence errors that static capture
        # does not: (1) camera/radar time mismatch × speed, (2) the people-counting
        # tracker lags a moving target. Both scale with hand speed. These two gates
        # keep those errors small by capturing only when well-aligned and slow.
        dp('max_sync_dt', -1.0)         # s; drop image/radar pairs whose stamps differ by more than this. <=0 off
        dp('max_capture_speed', -1.0)   # m/s; in continuous mode, skip captures while |ṗ_cam| exceeds this. <=0 off
        # --- validation thresholds / verdict ---
        dp('val_pass_reproj_px', 20.0)
        dp('val_pass_3d_mm', 150.0)
        dp('val_pass_bias_mm', 50.0)
        dp('measured_baseline_m', -1.0) # >0 → compare |t| against tape measure
        dp('baseline_tol_m', 0.03)
        # --- frames / output ---
        dp('parent_frame', 'zed_left_camera_optical_frame')
        dp('child_frame',  'radar_link')
        dp('camera_name', 'zed_left'); dp('radar_name', 'radar')
        dp('output_path', '')
        dp('publish_tf', True)
        dp('debug_image', True)
        dp('debug_image_topic', '/radar_camera_calib/debug_image')
        dp('show_window', False)        # True → pop a native OpenCV window (needs a display)
        # live diversity HUD: bars for board pitch/roll/yaw + radar az/el/range spread,
        # green when each crosses the target that makes rotation (& translation)
        # well-observed. On by default in the STATIC profile. See pose_diversity().
        dp('show_diversity_hud', False)
        dp('radar_watchdog_s', 3.0)

        g = lambda n: self.get_parameter(n).value
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
        self.cluster_eps = g('cluster_eps'); self.min_cluster_size = int(g('min_cluster_size'))
        self.cluster_apex_radius = g('cluster_apex_radius'); self.cluster_strict = bool(g('cluster_strict'))
        self.min_range = g('min_range'); self.max_range = g('max_range')
        self.max_abs_doppler = g('max_abs_doppler')
        self.min_abs_doppler = g('min_abs_doppler')
        self.use_dop_consistency = bool(g('use_doppler_consistency'))
        self.doppler_match_tol = g('doppler_match_tol')
        _ds = str(g('doppler_sign')).strip().lower()
        self.doppler_sign = 0.0 if _ds == 'auto' else float(_ds)  # 0.0 → learn from data
        self._dop_sign_vote = 0.0
        self.min_motion_mps = g('min_motion_mps')
        self.gate_radius = g('gate_radius')
        self.range_gate_margin = g('range_gate_margin_m')
        self.bg_accum_frames = int(g('bg_accum_frames')); self.bg_match_dist = g('bg_match_dist')
        self.require_background = bool(g('require_background'))
        self.range_scale = g('radar_range_scale'); self.range_bias = g('radar_range_bias_m')
        self.sig_r = g('sigma_range_m')
        self.sig_az = np.radians(g('sigma_az_deg')); self.sig_el = np.radians(g('sigma_el_deg'))
        self.force_2d = bool(g('force_2d_radar'))
        self.huber = g('huber_f_scale'); self.reject_sigma = g('reject_sigma')
        self.reject_axis_sigma = g('reject_axis_sigma')
        self.use_ext_prior = bool(g('use_extrinsic_prior'))
        self.prior_R = (Rot.from_euler('xyz', g('prior_rpy_deg'), degrees=True).as_matrix()
                        if self.use_ext_prior else None)
        self.prior_t = np.array(g('prior_t_xyz'), float) if self.use_ext_prior else None
        self.prior_t_sigma = g('prior_t_sigma_m')
        self.prior_rot_sigma = np.radians(g('prior_rot_sigma_deg'))
        self.min_snr = g('min_snr')

        self.capture_mode = g('capture_mode')
        self.stable_window = int(g('stable_window')); self.stable_std = g('stable_std')
        self.stable_std_radar = g('stable_std_radar')
        self.min_baseline = g('min_baseline'); self.min_points = int(g('min_points'))
        self.max_sync_dt = g('max_sync_dt'); self.max_capture_speed = g('max_capture_speed')

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
        self.show_diversity_hud = bool(g('show_diversity_hud'))
        self.window_name = 'radar_camera_calib — apex (green=matched) | reflector overlay'
        self._last_dbg = None; self._win_ok = None
        self.watchdog_s = g('radar_watchdog_s')

        # state
        self.win = []                 # rolling [(p_cam, p_radar), ...] for stability
        self.captures = []            # accepted [(p_cam, p_radar, snr, doppler), ...]
        self.last_capture_cam = None
        self._prev_apex = None; self._prev_apex_t = None   # camera-side apex velocity (radial-Doppler predict)
        self.manual_capture_req = False
        self.bg_radar = None          # (M,3) pooled background points (raw radar frame)
        self.bg_accum = None          # accumulation buffer while pooling
        self.X = None; self.rms = None
        self._rot_sig_deg = None       # per-axis rotation 1σ from the last solve (for the HUD)
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

    # ── intrinsics / watchdog ──
    def _info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.D = np.array(msg.d) if len(msg.d) else np.zeros(5)
            self.get_logger().info(f"intrinsics locked ({msg.width}x{msg.height})")

    @staticmethod
    def _stamp_s(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

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

    def _select_radar(self, xyz, snr, dop, predicted=None, cam_range=None, v_pred=None):
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
        # Doppler ↔ motion consistency: among the survivors, keep only those whose
        # radial velocity matches the camera-predicted v_pred. This rejects moving
        # clutter (hand/arm/body) that a static |doppler| gate cannot, because that
        # clutter moves at a DIFFERENT radial rate than the rigidly-fixed reflector.
        # Falls back to the full survivor set if the match would empty it (so a bad
        # v_pred or sign never rejects everything). Only active while actually moving.
        dop_applied = False
        if (self.use_dop_consistency and v_pred is not None
                and abs(v_pred) >= self.min_motion_mps and self.doppler_match_tol > 0):
            sign = self.doppler_sign or (1.0 if self._dop_sign_vote >= 0 else -1.0)
            keepd = np.abs(sign * dg - v_pred) <= self.doppler_match_tol
            if keepd.any():
                xg, sg, dg = xg[keepd], sg[keepd], dg[keepd]; dop_applied = True
        # Pick the reflector return → (point, snr, doppler).
        #   'cluster' : group the survivors and take the SNR-weighted CENTROID of
        #               the reflector blob (robust to a single noisy SNR spike).
        #   'snr'     : the single highest-SNR survivor.
        #   'nearest' : nearest the predicted apex, else nearest the radar origin.
        sel = None
        if self.select_by == 'cluster':
            sel = self._pick_cluster(xg, sg, dg, predicted)   # None → fall through / strict-reject
            if sel is None and self.cluster_strict and predicted is not None:
                _diag(f"strict cluster: no reflector cluster within "
                      f"{self.cluster_apex_radius}m of the predicted apex")
                return None, None, None, n_gated              # strict → reject this capture
        if sel is None:
            if self.select_by != 'nearest' and np.any(np.isfinite(sg)) and sg.max() > 0:
                idx = int(np.argmax(sg))         # highest-SNR survivor = the trihedral
            elif predicted is not None:
                idx = int(np.argmin(np.linalg.norm(xg - predicted, axis=1)))
            else:
                idx = int(np.argmin(np.linalg.norm(xg, axis=1)))
            sel = (xg[idx], float(sg[idx]), float(dg[idx]))
        p_sel, snr_sel, dop_sel = sel
        if self.min_snr > 0 and snr_sel < self.min_snr:
            _diag(f"best snr {snr_sel:.0f} < min_snr {self.min_snr:.0f}")
            return None, None, None, n_gated     # too weak → likely mis-associated, skip
        # auto-learn the Doppler sign convention from the chosen reflector point
        if (self.use_dop_consistency and self.doppler_sign == 0.0 and v_pred is not None
                and abs(v_pred) >= self.min_motion_mps):
            self._dop_sign_vote += float(dop_sel) * v_pred
        if self.use_dop_consistency and v_pred is not None:
            self.get_logger().info(
                f"[dop] v_pred {v_pred:+.2f} m/s  sel doppler {dop_sel:+.2f}  "
                f"tol {self.doppler_match_tol:.2f}  {'filtered' if dop_applied else 'off/fallback'}",
                throttle_duration_sec=1.0)
        return np.asarray(p_sel, float), float(snr_sel), float(dop_sel), n_gated

    def _pick_cluster(self, xg, sg, dg, predicted):
        """Cluster the gated survivors and return (centroid, snr, doppler) of the
        reflector blob, or None if none qualifies (caller then falls back to the
        SNR pick, or — in cluster_strict mode with a predicted apex — rejects).

        When an extrinsic/prior PREDICTS the apex, cluster ONLY points within
        cluster_apex_radius of it ("cluster around the estimated apex"), and take
        the cluster nearest the prediction. Without a prediction, use the cluster
        holding the brightest return. The representative point is the SNR-WEIGHTED
        CENTROID — averaging the blob beats a single argmax spike for angular noise."""
        xs, ss, ds = xg, sg, dg
        if predicted is not None and self.cluster_apex_radius > 0:
            near = np.linalg.norm(xg - predicted, axis=1) <= self.cluster_apex_radius
            if near.any():
                xs, ss, ds = xg[near], sg[near], dg[near]     # cluster around the estimated apex
            elif self.cluster_strict:
                return None                                   # strict: nothing near the apex
            # non-strict & nothing near → cluster over the whole gated set (lenient)
        clusters = cluster_points(xs, self.cluster_eps, self.min_cluster_size)
        if not clusters:
            return None
        if predicted is not None:
            best = min(clusters, key=lambda c: float(np.linalg.norm(xs[c].mean(0) - predicted)))
        else:
            seed = int(np.argmax(ss)) if (np.any(np.isfinite(ss)) and ss.max() > 0) else 0
            best = next((c for c in clusters if seed in set(c.tolist())), None)
            if best is None:
                best = max(clusters, key=lambda c: float(np.nanmax(ss[c])))
        w = np.clip(np.nan_to_num(ss[best]), 1e-6, None); w = w / w.sum()
        p = (xs[best] * w[:, None]).sum(0)
        return p, float(np.nanmax(ss[best])), float(np.median(ds[best]))

    # ── camera: board pose → apex in camera frame ──
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
        ok, rvec, tvec = cv2.solvePnP(objp, cc, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
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
        # Hand-held rigs move; a large image↔radar stamp gap turns into a
        # correspondence error (Δt × speed). Drop badly-aligned pairs outright.
        t_img = self._stamp_s(a_img); t_rad = self._stamp_s(radar_msg)
        if self.max_sync_dt > 0 and abs(t_img - t_rad) > self.max_sync_dt:
            self.get_logger().info(
                f"skip pair: img/radar Δt {abs(t_img - t_rad)*1000:.0f} ms "
                f"> max_sync_dt {self.max_sync_dt*1000:.0f} ms", throttle_duration_sec=2.0)
            return
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
        # Camera-side apex velocity → predicted radar radial velocity (Doppler).
        # Rotation-free: v_pred = (p_cam − t_ext)·ṗ_cam / |p_cam − t_ext|.
        v_pred = None; speed = None
        if p_cam is not None:
            if self._prev_apex is not None and self._prev_apex_t is not None:
                dt = t_img - self._prev_apex_t
                if 1e-3 < dt < 0.5:
                    vdot = (p_cam - self._prev_apex) / dt
                    speed = float(np.linalg.norm(vdot))
                    t_ext = (self.X[:3, 3] if self.X is not None
                             else (self.prior_t if (self.use_ext_prior and self.prior_t is not None)
                                   else np.zeros(3)))
                    d = p_cam - t_ext; nd = float(np.linalg.norm(d))
                    if nd > 1e-6:
                        v_pred = float(d @ vdot / nd)
            self._prev_apex = p_cam.copy(); self._prev_apex_t = t_img
        p_radar, snr_i, dop_i, n_gated = self._select_radar(xyz, snr, dop, predicted, cam_range, v_pred)

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
            too_fast = (self.max_capture_speed > 0 and speed is not None
                        and speed > self.max_capture_speed)
            if too_fast:
                self.get_logger().info(
                    f"sweep too fast ({speed:.2f} m/s > {self.max_capture_speed:.2f}) — "
                    f"slow down / pause briefly for a clean capture", throttle_duration_sec=1.0)
            elif moved:
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

        # auto-detect 2-D radar: no meaningful elevation spread → drop el residual
        rr = np.linalg.norm(P, axis=1); rr = np.where(rr < 1e-6, 1e-6, rr)
        el_spread = float(np.std(np.abs(P[:, 2]) / rr))
        use_el = (not self.force_2d) and el_spread > 0.01

        # ── the accurate estimator: measurement-space ML + robust rejection,
        #    optionally jointly refining the apex offset (MAP toward measured) ──
        r = robust_ml_calibrate(P, Rb, Tb, self.apex_prior,
                                self.sig_r, self.sig_az, self.sig_el, use_elevation=use_el,
                                solve_offset=self.solve_offset,
                                offset_prior_sigma=self.offset_prior_sigma,
                                R_prior=self.prior_R, t_prior=self.prior_t,
                                rot_prior_sigma=self.prior_rot_sigma,
                                t_prior_sigma=self.prior_t_sigma,
                                huber=self.huber, reject_sigma=self.reject_sigma,
                                reject_axis_sigma=self.reject_axis_sigma)
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
        q = Rot.from_matrix(R).as_quat(); rpy = Rot.from_matrix(R).as_euler('xyz', degrees=True)
        tmag = float(np.linalg.norm(t))
        n_out = int((~mask).sum())

        # diversity guard: the extrinsic is only observable if the RADAR points
        # span range + azimuth + ELEVATION. Rotation error ≈ cross-range_noise /
        # cloud_extent, so a thin/planar cloud gives good translation but BAD
        # rotation — the classic single-reflector signature. Warn per-axis.
        raz = np.array([cart_to_raz(p) for p in Pin])
        rng_span = float(raz[:, 0].max() - raz[:, 0].min())
        az_span = float(np.degrees(raz[:, 1].max() - raz[:, 1].min()))
        el_span = float(np.degrees(raz[:, 2].max() - raz[:, 2].min()))
        # elevation must span too, else out-of-plane rotation stays under-constrained
        low_div = rng_span < 0.20 or az_span < 20.0 or (use_el and el_span < 10.0)
        div_msg = (f"  RADAR spread: range {rng_span*100:.0f} cm, az {az_span:.0f}°, el {el_span:.0f}°"
                   + ("   !! TOO CLUSTERED — spread NEAR↔FAR, LEFT↔RIGHT and UP↔DOWN; "
                      "rotation stays under-constrained (good |t|, bad R) until this grows"
                      if low_div else "  ✓"))

        # per-DOF 1-sigma from the covariance (rotvec rad → deg, t m → mm).
        # rot 1σ is in the CAMERA optical frame (X=right, Y=down, Z=forward); each
        # weak rotation axis is fixed by a specific spatial spread of the points:
        #   rot_x (pitch) ← vertical (UP↔DOWN) + range (NEAR↔FAR) spread
        #   rot_y (yaw)   ← horizontal (LEFT↔RIGHT) + range (NEAR↔FAR) spread
        #   rot_z (roll)  ← spread ACROSS the image (LEFT↔RIGHT *and* UP↔DOWN)
        dsig = np.sqrt(np.clip(np.diag(cov), 0, None))
        rot_sig_deg = np.degrees(dsig[:3]); t_sig_mm = dsig[3:6] * 1000
        self._rot_sig_deg = rot_sig_deg                # expose to the diversity HUD
        _rot_fix = {'rot_x': 'move UP↔DOWN + NEAR↔FAR', 'rot_y': 'move LEFT↔RIGHT + NEAR↔FAR',
                    'rot_z': 'spread poses ACROSS the image (LEFT↔RIGHT and UP↔DOWN)'}
        weak_rot = [nm for nm, s in zip(('rot_x', 'rot_y', 'rot_z'), dsig[:3]) if s > 0.3]
        unobs = list(weak_rot)
        unobs += [nm for nm, s in zip(('t_x', 't_y', 't_z'), dsig[3:6]) if s > 0.2]

        lines = [
            f"\n=== T_{self.parent_frame}_{self.child_frame}  (camera ← radar) ===",
            f"  method   : measurement-space ML ({'3-D' if use_el else '2-D, no elevation'}), "
            f"Huber+reject   inliers {r['n_in']}/{len(P)}" + (f"  ({n_out} rejected)" if n_out else ""),
            f"  captures {len(self.captures)}   in-sample RMS {self.rms*1000:.1f} mm   "
            f"residual {r['rms_sigma']:.2f} σ   cond {cond:.1f}",
            f"  xyz (m) : {t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}   |t| {tmag*100:.1f} cm",
            f"  quat    : {q[0]:+.4f} {q[1]:+.4f} {q[2]:+.4f} {q[3]:+.4f}",
            f"  rpy(deg): {rpy[0]:+.2f} {rpy[1]:+.2f} {rpy[2]:+.2f}",
            f"  1σ rot  : {rot_sig_deg[0]:.2f} {rot_sig_deg[1]:.2f} {rot_sig_deg[2]:.2f} deg   "
            f"1σ t: {t_sig_mm[0]:.1f} {t_sig_mm[1]:.1f} {t_sig_mm[2]:.1f} mm",
            div_msg,
        ]
        if unobs:
            lines.append("  !! WEAK/UNOBSERVABLE dof: " + ", ".join(unobs)
                         + ("  — 2-D radar can't see out-of-plane rotation or height; "
                            "fix those from CAD or an extrinsic prior" if not use_el else
                            "  — add pose diversity (range/azimuth/height)"))
            for nm in weak_rot:                        # axis-specific rotation fix
                lines.append(f"     {nm} weak → {_rot_fix[nm]} (more lever arm shrinks rot 1σ)")
            if not self.use_ext_prior:
                lines.append("     tip: set use_extrinsic_prior:=true (rough mounting rpy/xyz) to "
                             "pin the weak rotation DOF while the data refines the rest")
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
                            (self.sig_r, self.sig_az, self.sig_el), use_el)
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
        # split the 3-D error into per-axis RMS (mm) — shows WHERE the error lives
        # (camera frame X=right, Y=down, Z=forward/range); a big Z is a range error,
        # a big X/Y is cross-range (the radar's weak angular direction).
        rms_xyz_mm = np.sqrt((res ** 2).mean(0)) * 1000
        lines.append(f"  signed residual (pred−cam) mm: X {bias_mm[0]:+.0f} Y {bias_mm[1]:+.0f} "
                     f"Z {bias_mm[2]:+.0f}   reproj {mean_px:.1f} px   mean 3-D {mean_3d:.1f} mm")
        lines.append(f"  3-D error RMS  mm: X {rms_xyz_mm[0]:.1f}  Y {rms_xyz_mm[1]:.1f}  "
                     f"Z {rms_xyz_mm[2]:.1f}   (X/Y = cross-range, Z = range)")

        # range diagnostic (camera range vs radar range, inliers)
        cam_r = np.linalg.norm(Qin, axis=1); rad_r = np.linalg.norm(Pin, axis=1)
        if len(Pin) >= 2:
            A = np.vstack([rad_r, np.ones_like(rad_r)]).T
            (a, b), *_ = np.linalg.lstsq(A, cam_r, rcond=None)
            if abs(a - 1) > 0.03 or abs(b) > 0.05:
                # rad_r is ALREADY corrected by the current scale/bias, so the fit
                # slope a is the RESIDUAL scale error — the goal is a ≈ 1.00. The
                # suggestion must fold in the scale already applied:
                #   corrected = scale·raw − bias ;  cam = a·corrected + b
                #   ⇒ new_scale = a·scale ,  new_bias = a·bias − b
                # (the old "1/a" ignored the current scale and, iterated, drove the
                #  scale the wrong way — over-scaling leaks straight into t_z/depth.)
                new_scale = a * self.range_scale
                new_bias = a * self.range_bias - b
                lines.append(f"  range fit: cam_r = {a:.3f}·radar_r {b:+.3f} m (want a≈1) → "
                             f"set radar_range_scale={new_scale:.4f} bias={new_bias:+.4f}")

        if self.measured_baseline > 0:
            d = abs(tmag - self.measured_baseline)
            lines.append(f"  baseline: |t| {tmag*100:.1f} cm vs measured "
                         f"{self.measured_baseline*100:.1f} cm → Δ {d*100:.1f} cm "
                         f"[{'OK' if d <= self.baseline_tol else 'MISMATCH'}]")
        checks = [("reproj_px", mean_px, self.val_px), ("3d_mm", mean_3d, self.val_3d),
                  ("bias_mm", float(np.abs(bias_mm).max()), self.val_bias)]
        verdict = all(v <= lim for _, v, lim in checks) and not unobs
        lines.append("  VERDICT : " + ("✔ GOOD  " if verdict else "✗ SUSPECT  ")
                     + "  ".join(f"{k} {v:.1f}/{lim:.0f}[{'P' if v<=lim else 'F'}]"
                                 for k, v, lim in checks))
        self.get_logger().info("\n".join(lines))

        if self.publish_tf:
            self._broadcast()
        self._save({'mean_px': mean_px, 'mean_3d': mean_3d, 'rms_xyz_mm': rms_xyz_mm,
                    'bias_mm': bias_mm, 'cond': cond,
                    'loo': loo, 'planar': planar, 'verdict': verdict, 'use_el': use_el,
                    'rot_sig_deg': rot_sig_deg, 't_sig_mm': t_sig_mm, 'unobs': unobs,
                    'n_in': r['n_in'], 'n_out': n_out, 'rms_sigma': r['rms_sigma'],
                    'apex': self.apex_board, 'apex_sigma': r['apex_sigma'],
                    'apex_prior': self.apex_prior, 'solved_offset': r['solved_offset']})

    def _reset(self):
        self.captures.clear(); self.win.clear(); self.last_capture_cam = None
        self.X = None; self.rms = None; self._rot_sig_deg = None
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
            'radar_noise_sigma_range_m': self.sig_r,
            'radar_noise_sigma_az_deg': float(np.degrees(self.sig_az)),
            'radar_noise_sigma_el_deg': float(np.degrees(self.sig_el)),
            'mean_reproj_px': float(m['mean_px']), 'mean_3d_mm': float(m['mean_3d']),
            'error_3d_rms_mm_xyz': [float(v) for v in m['rms_xyz_mm']],
            'residual_bias_mm_xyz': [float(v) for v in m['bias_mm']],
            'condition_number': float(m['cond']),
            'radar_range_scale': self.range_scale, 'radar_range_bias_m': self.range_bias,
            # T_cam_radar : p_cam = R p_radar + t
            'T_cam_radar_translation': [float(v) for v in X[:3, 3]],
            'T_cam_radar_quaternion_xyzw': [float(v) for v in q],
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
            # full session record: params + every capture (with board pose) + result.
            # Lets you re-solve or audit the run offline (see sessions/solve_from_poses.py).
            sess_path = self.output_path.replace('.yaml', '_session.json')
            with open(sess_path, 'w') as f:
                json.dump(self._session_dict(data), f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"save failed: {e}")

    def _param_snapshot(self):
        """All declared parameters and their current values, as a plain dict."""
        out = {}
        for n in self._param_names:
            try:
                v = self.get_parameter(n).value
                out[n] = list(v) if isinstance(v, (list, tuple)) else v
            except Exception:
                pass
        return out

    def _session_dict(self, result):
        """Reproducible session record: ISO time, params, per-capture poses (radar
        point + camera apex + full board pose, so an offline solve can reproduce the
        joint offset estimation), and the solved result/metrics."""
        caps = []
        for c in self.captures:
            Rb = np.asarray(c['Rb'], float); tb = np.asarray(c['tb'], float)
            p_cam = (Rb @ self.apex_board + tb)
            caps.append({
                'p_radar': [float(v) for v in c['p_radar']],
                'p_cam': [float(v) for v in p_cam],
                'board_R_quat_xyzw': [float(v) for v in Rot.from_matrix(Rb).as_quat()],
                'board_t': [float(v) for v in tb],
                'snr': float(c['snr']), 'doppler': float(c['dop'])})
        return {
            'iso_time_utc': datetime.now(timezone.utc).isoformat(),
            'stamp_ns': int(self.get_clock().now().nanoseconds),
            'node': 'radar_camera_calib',
            'parent_frame': self.parent_frame, 'child_frame': self.child_frame,
            'apex_offset_in_board_m': [float(v) for v in self.apex_board],
            'n_captures': len(self.captures),
            'params': self._param_snapshot(),
            'captures': caps,
            'result': result}

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
            if self.show_diversity_hud:
                self._draw_diversity_hud(bgr)
            self._last_dbg = bgr
            if self.dbg_pub is not None:
                self.dbg_pub.publish(self.bridge.cv2_to_imgmsg(bgr, 'bgr8'))
        except Exception as e:
            self.get_logger().warn(f"debug image: {e}", throttle_duration_sec=5.0)

    def _draw_diversity_hud(self, bgr):
        """Live 'is my pose set diverse enough for accurate ROTATION?' cue.
        Six bars — board PITCH/ROLL/YAW spread (offset observability) and radar
        AZ/EL/RANGE spread (extrinsic-rotation lever arm) — each turns green when
        it crosses the target in DIVERSITY_TARGETS. Also shows the measured rot 1σ
        from the last solve. Green verdict = rotation is observable; good |t| with
        red bars is exactly the 'translation fine, rotation bad' trap."""
        h, w = bgr.shape[:2]
        div = pose_diversity([c['Rb'] for c in self.captures],
                             [c['p_radar'] for c in self.captures])
        rows = [('PITCH', 'pitch', 'deg'), ('ROLL', 'roll', 'deg'), ('YAW', 'yaw', 'deg'),
                ('AZ', 'az', 'deg'), ('EL', 'el', 'deg'), ('RANGE', 'range', 'm')]
        pw, rh = 250, 22
        x0 = max(10, w - pw - 12); y0 = 78
        panel_h = rh * (len(rows) + 2) + 18
        ov = bgr.copy()
        cv2.rectangle(ov, (x0 - 10, y0 - 26), (x0 + pw, y0 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.45, bgr, 0.55, 0, bgr)
        cv2.putText(bgr, f"POSE DIVERSITY  n={div['n']}", (x0, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y = y0 + 12
        all_ok = div['n'] >= max(3, self.min_points)
        for label, key, unit in rows:
            v, tgt, ok = div[key]
            all_ok = all_ok and ok
            frac = 0.0 if tgt <= 0 else min(1.0, max(0.0, v / tgt))
            bx = x0 + 62; bw = pw - 132
            col = (0, 200, 0) if ok else (0, 140, 255)
            cv2.putText(bgr, label, (x0, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            cv2.rectangle(bgr, (bx, y - 6), (bx + bw, y + 6), (70, 70, 70), -1)
            cv2.rectangle(bgr, (bx, y - 6), (bx + int(bw * frac), y + 6), col, -1)
            txt = f"{v:.0f}/{tgt:.0f}" if unit == 'deg' else f"{v:.2f}/{tgt:.2f}"
            cv2.putText(bgr, txt, (bx + bw + 6, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
            y += rh
        if self._rot_sig_deg is not None:
            rs = self._rot_sig_deg; rmax = float(np.max(rs))
            rc = (0, 200, 0) if rmax < 1.5 else ((0, 140, 255) if rmax < 4.0 else (0, 0, 255))
            cv2.putText(bgr, f"rot 1sig {rs[0]:.1f}/{rs[1]:.1f}/{rs[2]:.1f} deg",
                        (x0, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, rc, 1)
            all_ok = all_ok and rmax < 1.5
            y += rh
        verdict = "READY - rotation observable" if all_ok else "KEEP MOVING - tilt & spread more"
        cv2.putText(bgr, verdict, (x0, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (0, 220, 0) if all_ok else (0, 140, 255), 1)

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


def main(default_overrides=None):
    rclpy.init(); node = RadarCameraCalib(default_overrides)
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

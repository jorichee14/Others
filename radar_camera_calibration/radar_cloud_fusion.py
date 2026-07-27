#!/usr/bin/env python3
"""
Two-radar POINT-CLOUD fusion → camera projection → validation
=============================================================
Fuse the FULL point clouds of two calibrated IWR6843 radars into a single,
denser, less-noisy cloud in the ZED camera frame, project it onto the image, and
continuously validate that the fusion is actually correct (not just prettier).

This is the cloud-level sibling of `radar_fusion_reflector.py`: that node tracks
one corner reflector with a Kalman filter; THIS node fuses every detection so you
get a better *scene* cloud for perception, not a single target.

Why fusing the two clouds gives a BETTER cloud
──────────────────────────────────────────────
A single IWR6843 is anisotropic: range is sharp (~cm), one angle is moderate
(azimuth), the other is poor (elevation, few antennas). We mounted radar2 rolled
~90° vs radar1, so their weak axes are PERPENDICULAR in the camera frame:
  • radar1 : sharp horizontal, soft vertical
  • radar2 : sharp vertical,   soft horizontal
So where BOTH radars see the same physical point, an inverse-covariance
(information-form BLUE) merge is tight on *both* axes — radar1 supplies the
horizontal, radar2 the vertical. That is a genuine accuracy gain, not just more
dots. Points seen by only one radar are still kept, flagged n_radars=1, and
carry that radar's honest (anisotropic) covariance so downstream code can weight
them correctly.

The pipeline
────────────
  1. INGEST   each radar cloud, apply its calibrated range_scale, gate by
              range/SNR, transform into the camera frame with T_cam_radar, and
              attach a 3×3 anisotropic covariance per point (radar_cov_cam).
  2. ASSOCIATE the two clouds by Mahalanobis distance under (C1+C2); optimal
              (Hungarian) assignment, gated at a 3-DOF χ² threshold. A matched
              pair is the SAME physical target seen by both radars.
  3. FUSE     each matched pair with the information filter
                  C_f = (C1⁻¹ + C2⁻¹)⁻¹ ,  p_f = C_f (C1⁻¹p1 + C2⁻¹p2)
              → one point, tight on every axis. Unmatched points pass through as
              single-radar points. Optional temporal voxel merge (`accum_s`)
              accumulates+averages sweeps for a denser, steadier cloud (static
              scenes only — it smears moving targets, so it is OFF by default).
  4. PROJECT  the fused cloud onto the ZED image, coloured by depth, each point
              drawn with its 1σ uncertainty ELLIPSE (the 3-D covariance pushed
              through the projection Jacobian). Confirmed (2-radar) points are
              filled; single-radar points are hollow.
  5. VALIDATE every frame, and report rolling stats:
                • χ² consistency of matched pairs — should average ≈3 (3 DOF).
                  ≫3 ⇒ the extrinsics or the noise model are wrong, NOT a better
                  cloud. This is the headline correctness check.
                • covariance shrink — fused σ vs the better single radar; <1
                  means fusion genuinely tightened the estimate.
                • match rate, in-frame coverage, point counts.
              Published on `fused_cloud_topic` (PointCloud2: x,y,z,intensity,
              n_radars,sigma_mm) and drawn on `debug_image_topic`; a one-line
              VALID/CHECK verdict is logged.

Extrinsics default to the FINALISED values for this rig
(sessions/2026-07-22_zed_radar1_radar2_final.md); override with params.

Run (ROS):   ros2 run wicoms_utils radar_cloud_fusion -p ...params...
Validate offline (no ROS, proves the math on synthetic data using these exact
extrinsics):
             python3 radar_cloud_fusion.py --selftest
"""
import sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_LSA = True
except Exception:
    _HAVE_LSA = False

# ── ROS + OpenCV are only needed for the live node; guard them so `--selftest`
#    runs on a bare machine with just numpy+scipy. ──────────────────────────────
try:
    import cv2
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
    from cv_bridge import CvBridge
    from sensor_msgs_py import point_cloud2 as pc2
    _HAVE_ROS = True
except Exception:
    _HAVE_ROS = False


# ─────────────────────────────────────────────────────────────────────────────
#  Pure-numpy core — no ROS, fully unit-testable (see selftest at the bottom).
# ─────────────────────────────────────────────────────────────────────────────
def radar_cov_cam(q, R, sr, saz, sel):
    """3×3 measurement covariance of a radar point in the CAMERA frame. Local
    basis: radial, azimuth-tangent (horizontal in the radar XY), elevation-tangent.
    Cross-range std grows with range (r·σ_angle). `q` is the point in the RADAR
    frame, `R` the extrinsic rotation R_cam_radar. Returns (cov3x3, blind_vec)."""
    q = np.asarray(q, float)
    r = float(np.linalg.norm(q))
    if r < 1e-6:
        r = 1e-6
    er = q / r
    eaz = np.array([-q[1], q[0], 0.0])                 # radar X=fwd, Y=left → az tangent
    n = np.linalg.norm(eaz); eaz = eaz / n if n > 1e-6 else np.array([0., 1., 0.])
    eel = np.cross(er, eaz)
    n = np.linalg.norm(eel); eel = eel / n if n > 1e-6 else np.array([0., 0., 1.])
    S = (sr ** 2) * np.outer(er, er) \
        + (r * saz) ** 2 * np.outer(eaz, eaz) \
        + (r * sel) ** 2 * np.outer(eel, eel)
    return R @ S @ R.T, R @ eel * (r * sel)


def ext_cov(p_cam, sig_t, sig_rot):
    """Covariance contributed by the EXTRINSIC's own uncertainty, in the camera
    frame. A translation 1σ `sig_t` (m) displaces the point isotropically; a
    rotation 1σ `sig_rot` (rad) rotates it about the camera origin, so it displaces
    the point by ≈ range·σ_rot PERPENDICULAR to the line of sight. Folding this in
    is what calibrates the cross-radar χ² back toward 3 on real data — without it,
    matched pairs look farther apart than the pure sensor-noise model predicts
    (the extrinsic's few-degree / few-cm error is unmodelled)."""
    p_cam = np.asarray(p_cam, float)
    r = float(np.linalg.norm(p_cam))
    if r < 1e-6:
        return (sig_t ** 2) * np.eye(3)
    er = p_cam / r
    perp = np.eye(3) - np.outer(er, er)               # project onto the line-of-sight ⊥ plane
    return (sig_t ** 2) * np.eye(3) + (r * sig_rot) ** 2 * perp


def _inv3(C):
    """Robust 3×3 inverse with a tiny diagonal jitter fallback (covariances from
    a near-degenerate geometry can be numerically singular)."""
    try:
        return np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return np.linalg.inv(C + np.eye(3) * (np.trace(C) * 1e-6 + 1e-12))


def fuse_pair(p1, C1, p2, C2):
    """Information-form BLUE of two 3-D estimates of the SAME point.
    C_f = (C1⁻¹+C2⁻¹)⁻¹,  p_f = C_f(C1⁻¹p1 + C2⁻¹p2). Optimal & unbiased when
    the two errors are independent (they are — different sensors)."""
    I1, I2 = _inv3(C1), _inv3(C2)
    Cf = _inv3(I1 + I2)
    pf = Cf @ (I1 @ p1 + I2 @ p2)
    return pf, Cf


def maha2(dp, Csum):
    """Squared Mahalanobis distance dpᵀ (Csum)⁻¹ dp."""
    return float(dp @ _inv3(Csum) @ dp)


def associate(P1, C1, P2, C2, gate_chi2):
    """Match points across the two clouds by Mahalanobis distance under the
    combined covariance (C1+C2), optimally (Hungarian) when scipy is present,
    greedily otherwise. A pair is accepted only if its χ² ≤ gate_chi2 (3 DOF).

    Returns (matches, un1, un2):
      matches = list of (i, j, d2)   indices into P1, P2 and their χ²
      un1, un2 = index lists of unmatched points in each cloud."""
    n1, n2 = len(P1), len(P2)
    if n1 == 0 or n2 == 0:
        return [], list(range(n1)), list(range(n2))
    D = np.full((n1, n2), np.inf)
    for i in range(n1):
        for j in range(n2):
            d2 = maha2(P1[i] - P2[j], C1[i] + C2[j])
            if d2 <= gate_chi2:
                D[i, j] = d2
    matches = []
    if _HAVE_LSA:
        big = 1e6
        cost = np.where(np.isfinite(D), D, big)
        ri, cj = linear_sum_assignment(cost)
        for i, j in zip(ri, cj):
            if np.isfinite(D[i, j]):
                matches.append((int(i), int(j), float(D[i, j])))
    else:                                              # greedy nearest fallback
        order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
        u1, u2 = set(), set()
        for i, j in order:
            if not np.isfinite(D[i, j]):
                break
            if i in u1 or j in u2:
                continue
            matches.append((int(i), int(j), float(D[i, j])))
            u1.add(i); u2.add(j)
    m1 = {i for i, _, _ in matches}; m2 = {j for _, j, _ in matches}
    un1 = [i for i in range(n1) if i not in m1]
    un2 = [j for j in range(n2) if j not in m2]
    return matches, un1, un2


def fuse_clouds(P1, C1, s1, P2, C2, s2, gate_chi2, require_both=False):
    """Fuse two camera-frame clouds (points, per-point covariances, per-point SNR).
    Returns a list of fused points: dict(p, C, n, snr). Matched pairs are BLUE-
    merged (n=2); unmatched points pass through (n=1) unless require_both."""
    matches, un1, un2 = associate(P1, C1, P2, C2, gate_chi2)
    out, chi2s, shrink = [], [], []
    for i, j, d2 in matches:
        pf, Cf = fuse_pair(P1[i], C1[i], P2[j], C2[j])
        out.append({'p': pf, 'C': Cf, 'n': 2, 'snr': max(s1[i], s2[j])})
        chi2s.append(d2)
        best = min(np.trace(C1[i]), np.trace(C2[j]))
        if best > 0:
            shrink.append(np.trace(Cf) / best)
    if not require_both:
        for i in un1:
            out.append({'p': P1[i], 'C': C1[i], 'n': 1, 'snr': s1[i]})
        for j in un2:
            out.append({'p': P2[j], 'C': C2[j], 'n': 1, 'snr': s2[j]})
    stats = {'n_match': len(matches), 'n1': len(P1), 'n2': len(P2),
             'chi2': np.array(chi2s), 'shrink': np.array(shrink)}
    return out, stats


def voxel_merge(points, voxel_m):
    """Temporal/spatial denoise: bucket accumulated points into a voxel grid and
    information-merge the members of each occupied voxel into one point. Averages
    down random angular noise and de-duplicates overlapping sweeps. STATIC scenes
    only (it smears motion). `points` = list of dict(p,C,n,snr)."""
    if voxel_m <= 0 or not points:
        return points
    buckets = {}
    for pt in points:
        key = tuple(np.floor(pt['p'] / voxel_m).astype(int))
        buckets.setdefault(key, []).append(pt)
    merged = []
    for members in buckets.values():
        I = np.zeros((3, 3)); b = np.zeros(3); n = 0; snr = 0.0
        for m in members:
            Ii = _inv3(m['C']); I += Ii; b += Ii @ m['p']
            n = max(n, m['n']); snr = max(snr, m['snr'])
        C = _inv3(I); p = C @ b
        merged.append({'p': p, 'C': C, 'n': n, 'snr': snr})
    return merged


def project_cov_2d(p, C, K):
    """Push a 3-D covariance through the linearised pinhole projection to a 2×2
    image covariance at pixel of p. Ignores distortion (ZED rect feed D≈0) — used
    only to draw the uncertainty ellipse. Returns (u,v,cov2x2) or None behind cam."""
    X, Y, Z = p
    if Z <= 1e-3:
        return None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    J = np.array([[fx / Z, 0.0, -fx * X / (Z * Z)],
                  [0.0, fy / Z, -fy * Y / (Z * Z)]])
    return fx * X / Z + cx, fy * Y / Z + cy, J @ C @ J.T


def cov_ellipse(cov2, nsig=1.0):
    """(major_len, minor_len, angle_deg) of the nσ ellipse of a 2×2 covariance."""
    w, V = np.linalg.eigh(cov2)
    w = np.clip(w, 0, None)
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    return nsig * np.sqrt(w[1]), nsig * np.sqrt(w[0]), ang


# ─────────────────────────────────────────────────────────────────────────────
#  Live ROS node
# ─────────────────────────────────────────────────────────────────────────────
if _HAVE_ROS:
    def project(pt_cam, K, D):
        if pt_cam[2] <= 0:
            return None
        uv, _ = cv2.projectPoints(pt_cam.reshape(1, 3), np.zeros(3), np.zeros(3), K, D)
        return int(round(uv[0, 0, 0])), int(round(uv[0, 0, 1]))

    class RadarCloudFusion(Node):
        def __init__(self):
            super().__init__('radar_cloud_fusion')
            dp = self.declare_parameter
            dp('image_topic', '/zed/zed_node/left/image_rect_color')
            dp('info_topic',  '/zed/zed_node/left/camera_info')
            dp('radar1_topic', '/radar1/radar/points_all')
            dp('radar2_topic', '/radar2/radar/points_all')
            dp('pc_field_x', 'x'); dp('pc_field_y', 'y'); dp('pc_field_z', 'z')
            dp('pc_field_snr', 'intensity')
            dp('camera_frame', 'zed_left_camera_optical_frame')
            dp('use_info_frame', True)    # publish in CameraInfo's frame_id (the true
            #                               left optical frame the points live in);
            #                               falls back to camera_frame until info arrives
            # per-radar extrinsics T_cam_radar — FINAL values for this rig
            # (sessions/2026-07-22_zed_radar1_radar2_final.md)
            dp('r1_t_xyz', [0.2368, 0.0190, -0.0542])
            dp('r1_quat_xyzw', [-0.4995, 0.6007, -0.4224, -0.4596])
            dp('r2_t_xyz', [-0.1194, -0.0096, -0.0157])
            dp('r2_quat_xyzw', [0.7572, 0.0539, 0.6506, -0.0217])
            dp('r1_range_scale', 0.958); dp('r2_range_scale', 0.967)
            # radar noise model (same chip) — sets each point's anisotropic covariance
            dp('sigma_range_m', 0.05); dp('sigma_az_deg', 3.0); dp('sigma_el_deg', 8.0)
            # extrinsic 1σ (from the calibration) — folded into each point's covariance
            # so the cross-radar χ² is calibrated (≈3). Defaults ≈ the final solve's σ.
            dp('r1_ext_sigma_t_m', 0.035); dp('r1_ext_sigma_rot_deg', 4.0)
            dp('r2_ext_sigma_t_m', 0.030); dp('r2_ext_sigma_rot_deg', 3.5)
            # gating
            dp('min_range', 0.3); dp('max_range', 8.0); dp('min_snr', 0.0)
            dp('max_points', 400)         # cap per cloud before O(n·m) association
            # fusion
            dp('assoc_gate_chi2', 7.815)  # 3-DOF 95% = 7.815 (99% = 11.345)
            dp('valid_chi2_max', 6.0)     # VALID if windowed mean χ² ≤ this (2× ideal)
            dp('stats_window', 300)       # report χ²/shrink over the last N matches (recent state)
            dp('require_both', False)     # True → publish ONLY 2-radar confirmed points
            dp('sync_s', 0.15)            # both clouds must be within this to cross-fuse
            dp('accum_s', 0.0)            # >0: accumulate+voxel-merge (STATIC scenes only)
            dp('voxel_m', 0.10)           # voxel size for the temporal merge
            # display / output
            dp('draw_ellipse', True)      # per-point 1σ projected uncertainty ellipse
            dp('point_radius', 4)
            dp('fused_cloud_topic', '/radar_fusion/cloud')
            dp('debug_image_topic', '/radar_fusion/cloud_image')
            dp('publish_cloud', True)
            dp('report_every_s', 2.0)
            dp('show_window', True)

            g = lambda n: self.get_parameter(n).value
            self.fx, self.fy, self.fz = g('pc_field_x'), g('pc_field_y'), g('pc_field_z')
            self.fsnr = g('pc_field_snr')
            self.camera_frame = g('camera_frame')
            self.use_info_frame = bool(g('use_info_frame')); self.info_frame = None
            self.R1 = Rot.from_quat(g('r1_quat_xyzw')).as_matrix(); self.t1 = np.array(g('r1_t_xyz'), float)
            self.R2 = Rot.from_quat(g('r2_quat_xyzw')).as_matrix(); self.t2 = np.array(g('r2_t_xyz'), float)
            self.s1 = g('r1_range_scale'); self.s2 = g('r2_range_scale')
            self.sr = g('sigma_range_m')
            self.saz = np.radians(g('sigma_az_deg')); self.sel = np.radians(g('sigma_el_deg'))
            self.ext_t = {1: g('r1_ext_sigma_t_m'), 2: g('r2_ext_sigma_t_m')}
            self.ext_rot = {1: np.radians(g('r1_ext_sigma_rot_deg')),
                            2: np.radians(g('r2_ext_sigma_rot_deg'))}
            self.min_range = g('min_range'); self.max_range = g('max_range'); self.min_snr = g('min_snr')
            self.max_points = int(g('max_points'))
            self.gate = g('assoc_gate_chi2'); self.require_both = bool(g('require_both'))
            self.valid_chi2_max = g('valid_chi2_max'); self.stats_window = int(g('stats_window'))
            self.sync_s = g('sync_s'); self.accum_s = g('accum_s'); self.voxel_m = g('voxel_m')
            self.draw_ellipse = bool(g('draw_ellipse')); self.prad = int(g('point_radius'))
            self.report_every = g('report_every_s')
            self.show_window = bool(g('show_window'))
            self.window = 'radar_cloud_fusion — fused scene cloud (filled=2-radar, hollow=1-radar)'
            self._win_ok = None

            self.bridge = CvBridge(); self.K = None; self.D = None
            self.latest = {1: None, 2: None}          # (P, C, snr, stamp) per radar in CAM frame
            self.accum = []                           # (stamp, list[dict]) for temporal merge
            self.fused = None                         # list[dict] for the overlay
            self.stats = None
            self._last_report = 0.0
            self._roll = {'chi2': [], 'shrink': [], 'match': [], 'total': []}
            self._last = None

            self.create_subscription(CameraInfo, g('info_topic'), self._info, qos_profile_sensor_data)
            self.create_subscription(Image, g('image_topic'), self._image, qos_profile_sensor_data)
            self.create_subscription(PointCloud2, g('radar1_topic'),
                                     lambda m: self._radar(m, 1), qos_profile_sensor_data)
            self.create_subscription(PointCloud2, g('radar2_topic'),
                                     lambda m: self._radar(m, 2), qos_profile_sensor_data)
            self.dbg_pub = self.create_publisher(Image, g('debug_image_topic'), 1)
            self.cloud_pub = (self.create_publisher(PointCloud2, g('fused_cloud_topic'), 1)
                              if bool(g('publish_cloud')) else None)
            if self.show_window:
                self.create_timer(0.05, self._gui)
            self.get_logger().info(
                f"[radar_cloud_fusion] r1 {g('radar1_topic')}  r2 {g('radar2_topic')}\n"
                f"  info-form BLUE fusion, χ²-gated assoc @ {self.gate:.2f}, "
                f"{'CONFIRMED-only' if self.require_both else 'keep single-radar'} "
                f"{'(voxel %.2fm/%.2fs accum)' % (self.voxel_m, self.accum_s) if self.accum_s > 0 else ''}")

        def _info(self, msg):
            if self.K is None:
                self.K = np.array(msg.k).reshape(3, 3)
                self.D = np.array(msg.d) if len(msg.d) else np.zeros(5)
                if msg.header.frame_id:
                    self.info_frame = msg.header.frame_id      # the true left optical frame
                frame = self._cloud_frame()
                self.get_logger().info(f"intrinsics locked ({msg.width}x{msg.height}); "
                                       f"fused cloud published in frame '{frame}'")

        def _cloud_frame(self):
            """Frame the fused cloud is published in: CameraInfo's frame_id (the
            left camera optical frame the points actually live in) when available
            and use_info_frame, else the camera_frame param."""
            if self.use_info_frame and self.info_frame:
                return self.info_frame
            return self.camera_frame

        def _now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def _ingest(self, msg, which):
            """Read one radar cloud → (P_cam, C_list, snr) after range_scale, gating,
            and the extrinsic transform. Covariances are anisotropic in the CAM frame."""
            names = [f.name for f in msg.fields]
            has_snr = self.fsnr in names
            want = [self.fx, self.fy, self.fz] + ([self.fsnr] if has_snr else [])
            arr = list(pc2.read_points(msg, field_names=want, skip_nans=True))
            if not arr:
                return None
            arr = np.array([tuple(a) for a in arr], float)
            scale = self.s1 if which == 1 else self.s2
            xyz = arr[:, :3] * float(scale)
            snr = arr[:, 3] if has_snr else np.zeros(len(arr))
            rng = np.linalg.norm(xyz, axis=1)
            keep = (rng >= self.min_range) & (rng <= self.max_range)
            if self.min_snr > 0:
                keep &= snr >= self.min_snr
            xyz, snr = xyz[keep], snr[keep]
            if not len(xyz):
                return None
            if len(xyz) > self.max_points:            # keep the brightest to bound cost
                idx = np.argsort(snr)[-self.max_points:]
                xyz, snr = xyz[idx], snr[idx]
            R, t = (self.R1, self.t1) if which == 1 else (self.R2, self.t2)
            P = (xyz @ R.T) + t
            # per-point covariance = radar measurement noise + this radar's extrinsic
            # uncertainty (calibrates the cross-radar χ²)
            st, srot = self.ext_t[which], self.ext_rot[which]
            C = [radar_cov_cam(q, R, self.sr, self.saz, self.sel)[0] + ext_cov(p, st, srot)
                 for q, p in zip(xyz, P)]
            return P, C, snr

        def _radar(self, msg, which):
            if not _HAVE_ROS or self.K is None:
                return
            ing = self._ingest(msg, which)
            if ing is None:
                return
            P, C, snr = ing
            now = self._now()
            self.latest[which] = (P, C, snr, now, msg.header.stamp)  # keep sensor stamp
            self._fuse(now)

        def _fuse(self, now):
            a, b = self.latest[1], self.latest[2]
            fresh = lambda d: d is not None and (now - d[3]) <= self.sync_s
            # stamp the fused cloud with the freshest contributing radar's sensor
            # time so TF lookups against the camera frame line up
            src = a if (fresh(a) and (b is None or a[3] >= b[3])) else b
            stamp = src[4] if src is not None else self.get_clock().now().to_msg()
            if fresh(a) and fresh(b):
                fused, stats = fuse_clouds(a[0], a[1], a[2], b[0], b[1], b[2],
                                           self.gate, self.require_both)
            elif not self.require_both and (fresh(a) or fresh(b)):
                d = a if fresh(a) else b                # one radar only → pass through
                fused = [{'p': d[0][i], 'C': d[1][i], 'n': 1, 'snr': d[2][i]}
                         for i in range(len(d[0]))]
                stats = {'n_match': 0, 'n1': len(a[0]) if a else 0,
                         'n2': len(b[0]) if b else 0,
                         'chi2': np.array([]), 'shrink': np.array([])}
            else:
                return
            if self.accum_s > 0:                        # temporal accumulate + voxel merge
                self.accum.append((now, fused))
                self.accum = [(t, f) for (t, f) in self.accum if now - t <= self.accum_s]
                merged = [pt for _, f in self.accum for pt in f]
                fused = voxel_merge(merged, self.voxel_m)
            self.fused = fused; self.stats = stats
            self._accumulate_stats(stats, len(fused))
            self._publish_cloud(fused, stamp)
            self._maybe_report(now)

        def _accumulate_stats(self, stats, n_total):
            for c in stats['chi2']:
                self._roll['chi2'].append(float(c))
            for s in stats['shrink']:
                self._roll['shrink'].append(float(s))
            self._roll['match'].append(stats['n_match'])
            self._roll['total'].append(n_total)
            for k in self._roll:                        # bound memory
                if len(self._roll[k]) > 2000:
                    self._roll[k] = self._roll[k][-2000:]

        def _maybe_report(self, now):
            if now - self._last_report < self.report_every:
                return
            self._last_report = now
            r = self._roll
            if not r['chi2']:
                self.get_logger().info("[validate] no 2-radar matches yet "
                                       "(check overlap, sync_s, or extrinsics)")
                return
            mean_chi2, frac_gate, med_shrink = self._window_stats()
            match_rate = float(np.mean(r['match'][-self.stats_window:]))
            # Verdict: matched pairs are 3-DOF, so E[χ²]=3 when extrinsics+σ are
            # right (extrinsic covariance is folded in, so this is calibrated). ≫
            # valid_chi2_max ⇒ miscalibration or too-small σ; cloud not trustworthy.
            ok = mean_chi2 <= self.valid_chi2_max and med_shrink <= 1.0
            self.get_logger().info(
                f"[validate] {'VALID' if ok else 'CHECK'} | "
                f"mean χ²={mean_chi2:.2f} (ideal 3) | within-gate {100*frac_gate:.0f}% | "
                f"fused σ shrink×{med_shrink:.2f} | ~{match_rate:.1f} pairs/frame")

        def _window_stats(self):
            """(mean χ², within-gate fraction, median shrink) over the last
            `stats_window` matches — reflects CURRENT state, not all history."""
            r = self._roll
            chi2 = np.array(r['chi2'][-self.stats_window:])
            shrink = np.array(r['shrink'][-self.stats_window:])
            if not len(chi2):
                return float('nan'), float('nan'), float('nan')
            return (float(np.mean(chi2)), float(np.mean(chi2 <= self.gate)),
                    float(np.median(shrink)) if len(shrink) else float('nan'))

        def _publish_cloud(self, fused, stamp):
            if self.cloud_pub is None or not fused:
                return
            fields = [
                PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name='n_radars',  offset=16, datatype=PointField.FLOAT32, count=1),
                PointField(name='sigma_mm',  offset=20, datatype=PointField.FLOAT32, count=1)]
            pts = []
            for pt in fused:
                sig_mm = float(np.sqrt(max(np.trace(pt['C']) / 3.0, 0.0)) * 1000.0)
                pts.append((float(pt['p'][0]), float(pt['p'][1]), float(pt['p'][2]),
                            float(pt['snr']), float(pt['n']), sig_mm))
            from std_msgs.msg import Header
            hdr = Header()
            hdr.frame_id = self._cloud_frame()        # parent = left camera optical frame
            hdr.stamp = stamp                          # sensor time of the freshest radar
            self.cloud_pub.publish(pc2.create_cloud(hdr, fields, pts))

        def _image(self, msg):
            if self.K is None:
                return
            try:
                bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            except Exception as e:
                self.get_logger().warn(f"cv_bridge: {e}"); return
            h, w = bgr.shape[:2]
            fused = self.fused or []
            n2 = sum(1 for p in fused if p['n'] == 2)
            in_img = 0
            # colour by depth over the gated range window (near=warm, far=cool)
            for pt in fused:
                p = pt['p']
                if p[2] <= 0:
                    continue
                uv = project(p, self.K, self.D)
                if not (uv and 0 <= uv[0] < w and 0 <= uv[1] < h):
                    continue
                in_img += 1
                z = float(np.clip((p[2] - self.min_range) /
                                  max(self.max_range - self.min_range, 1e-3), 0, 1))
                col = tuple(int(c) for c in cv2.applyColorMap(
                    np.uint8([[255 * (1 - z)]]), cv2.COLORMAP_JET)[0, 0])
                if self.draw_ellipse:
                    pc = project_cov_2d(p, pt['C'], self.K)
                    if pc is not None:
                        maj, minr, ang = cov_ellipse(pc[2], nsig=1.0)
                        maj = int(np.clip(maj, 1, 200)); minr = int(np.clip(minr, 1, 200))
                        cv2.ellipse(bgr, uv, (maj, minr), ang, 0, 360, col, 1)
                if pt['n'] == 2:
                    cv2.circle(bgr, uv, self.prad, col, -1)         # confirmed: filled
                else:
                    cv2.circle(bgr, uv, self.prad, col, 1)          # single-radar: hollow

            mean_chi2, _, med_shrink = self._window_stats()
            ok = (self._roll['chi2'] and mean_chi2 <= self.valid_chi2_max
                  and med_shrink <= 1.0)
            l1 = (f"fused {len(fused)}  (2-radar {n2}, 1-radar {len(fused)-n2})  "
                  f"in-img {in_img}")
            l2 = (f"VALIDATE: mean chi2 {mean_chi2:.2f}/3  shrink x{med_shrink:.2f}  "
                  f"{'VALID' if ok else 'CHECK'}")
            cv2.putText(bgr, l1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(bgr, l2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if ok else (0, 165, 255), 2)
            cv2.putText(bgr, "filled=2-radar (confirmed)  hollow=1-radar  ellipse=1sig",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            self._last = bgr
            try:
                self.dbg_pub.publish(self.bridge.cv2_to_imgmsg(bgr, 'bgr8'))
            except Exception:
                pass

        def _gui(self):
            if not self.show_window or self._last is None or self._win_ok is False:
                return
            try:
                if self._win_ok is None:
                    cv2.namedWindow(self.window, cv2.WINDOW_NORMAL); self._win_ok = True
                cv2.imshow(self.window, self._last); cv2.waitKey(1)
            except Exception as e:
                self._win_ok = False
                self.get_logger().warn(f"show_window failed ({e}) — use rqt_image_view "
                                       f"{self.get_parameter('debug_image_topic').value}")


# ─────────────────────────────────────────────────────────────────────────────
#  Offline validation / self-test — proves the fusion on synthetic data using the
#  rig's FINAL extrinsics, with NO ROS. Verifies three things:
#    1. the fused cloud is closer to ground truth than EITHER radar alone,
#    2. per-axis, radar1 wins horizontally & radar2 wins vertically (and fusion
#       inherits the best of both),
#    3. matched-pair χ² is statistically consistent (mean ≈ 3, 3 DOF).
#  Exit code 0 = all checks pass.
# ─────────────────────────────────────────────────────────────────────────────
def _selftest(seed=0, n_trials=300):
    rng = np.random.default_rng(seed)
    # FINAL extrinsics (sessions/2026-07-22_zed_radar1_radar2_final.md)
    R1 = Rot.from_quat([-0.4995, 0.6007, -0.4224, -0.4596]).as_matrix()
    t1 = np.array([0.2368, 0.0190, -0.0542])
    R2 = Rot.from_quat([0.7572, 0.0539, 0.6506, -0.0217]).as_matrix()
    t2 = np.array([-0.1194, -0.0096, -0.0157])
    sr, saz, sel = 0.05, np.radians(3.0), np.radians(8.0)

    def measure(p_cam, R, t):
        """Simulate one radar's noisy measurement of a camera-frame point and its
        modelled covariance: sample noise in the radar's (range,az,el) space."""
        q = R.T @ (p_cam - t)                          # true point in radar frame
        r = np.linalg.norm(q)
        er = q / r
        eaz = np.array([-q[1], q[0], 0.0]); eaz /= (np.linalg.norm(eaz) + 1e-12)
        eel = np.cross(er, eaz); eel /= (np.linalg.norm(eel) + 1e-12)
        qn = q + er * rng.normal(0, sr) \
               + eaz * rng.normal(0, r * saz) \
               + eel * rng.normal(0, r * sel)
        C = radar_cov_cam(q, R, sr, saz, sel)[0]
        return R @ qn + t, C

    e1 = {'x': [], 'y': [], 'z': [], 'n': []}
    e2 = {'x': [], 'y': [], 'z': [], 'n': []}
    ef = {'x': [], 'y': [], 'z': [], 'n': []}
    chi2_all = []
    for _ in range(n_trials):
        # a handful of well-separated targets in the shared FoV (2–5 m)
        k = rng.integers(3, 7)
        truth = np.column_stack([rng.uniform(-1.5, 1.5, k),      # X right
                                 rng.uniform(-1.0, 1.0, k),      # Y down
                                 rng.uniform(2.0, 5.0, k)])      # Z forward
        P1 = []; C1 = []; P2 = []; C2 = []
        for p in truth:
            m1, c1 = measure(p, R1, t1); P1.append(m1); C1.append(c1)
            m2, c2 = measure(p, R2, t2); P2.append(m2); C2.append(c2)
        P1, P2 = np.array(P1), np.array(P2)
        s = np.ones(k)
        fused, stats = fuse_clouds(P1, C1, s, P2, C2, s, gate_chi2=11.345)
        chi2_all.extend(stats['chi2'].tolist())
        # accuracy per matched target (index alignment holds here since 1:1)
        matches, _, _ = associate(P1, C1, P2, C2, 11.345)
        for i, j, _ in matches:
            pf, _ = fuse_pair(P1[i], C1[i], P2[j], C2[j])
            tp = truth[i]                               # i indexes truth (P1 built in order)
            for d, key in zip(P1[i] - tp, ('x', 'y', 'z')):
                e1[key].append(d)
            for d, key in zip(P2[j] - tp, ('x', 'y', 'z')):
                e2[key].append(d)
            for d, key in zip(pf - tp, ('x', 'y', 'z')):
                ef[key].append(d)

    rms = lambda a: float(np.sqrt(np.mean(np.square(a)))) if len(a) else float('nan')
    tot = lambda e: rms(np.concatenate([e['x'], e['y'], e['z']]))
    print("\n  Two-radar cloud fusion — synthetic validation (final extrinsics)")
    print("  ---------------------------------------------------------------")
    print(f"  matched pairs: {len(ef['x'])}   trials: {n_trials}")
    print(f"  {'axis':<8}{'radar1 RMS':>14}{'radar2 RMS':>14}{'FUSED RMS':>14}")
    for key, label in (('x', 'X (horiz)'), ('y', 'Y (vert)'), ('z', 'Z (range)')):
        print(f"  {label:<8}{rms(e1[key])*1000:>11.1f}mm{rms(e2[key])*1000:>11.1f}mm"
              f"{rms(ef[key])*1000:>11.1f}mm")
    print(f"  {'|3D|':<8}{tot(e1)*1000:>11.1f}mm{tot(e2)*1000:>11.1f}mm{tot(ef)*1000:>11.1f}mm")
    mean_chi2 = float(np.mean(chi2_all))
    print(f"\n  matched-pair mean χ² = {mean_chi2:.2f}   (ideal 3.0 for 3 DOF)")

    checks = []
    checks.append(("fused |3D| beats both radars",
                   tot(ef) < tot(e1) and tot(ef) < tot(e2)))
    checks.append(("radar1 sharper than radar2 horizontally (X)",
                   rms(e1['x']) < rms(e2['x'])))
    checks.append(("radar2 sharper than radar1 vertically (Y)",
                   rms(e2['y']) < rms(e1['y'])))
    checks.append(("fused X ≤ best single-radar X",
                   rms(ef['x']) <= min(rms(e1['x']), rms(e2['x'])) * 1.15))
    checks.append(("fused Y ≤ best single-radar Y",
                   rms(ef['y']) <= min(rms(e1['y']), rms(e2['y'])) * 1.15))
    checks.append(("matched-pair χ² consistent (2 ≤ mean ≤ 4.5)",
                   2.0 <= mean_chi2 <= 4.5))
    print()
    all_ok = True
    for name, ok in checks:
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok &= ok
    print(f"\n  {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}\n")
    return 0 if all_ok else 1


def main():
    if '--selftest' in sys.argv:
        sys.exit(_selftest())
    if not _HAVE_ROS:
        print("ROS 2 / OpenCV not available — run with --selftest for the offline "
              "validation, or install rclpy/cv_bridge/sensor_msgs to run the node.")
        sys.exit(1)
    rclpy.init(); node = RadarCloudFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

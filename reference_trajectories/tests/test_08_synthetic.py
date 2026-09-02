"""Synthetic end-to-end check of the mobile_1 workflow in 08_reference_traj.py:
room map, ground-truth camera trajectory, ZED-style odometry in a BODY child
frame with drift, Ouster-style scans in a lidar frame related to the camera by
a non-trivial T_lidar_camera, ChArUco sightings when a board is close.
Runs chain_lidar (lidar ICP vs odom) and run_arms with cloud_source='lidar'
and checks every trajectory against the truth."""
import sys, os, math
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import importlib
m = importlib.import_module("08_reference_traj")
Rt, inv, apply, exp_r, log_R = m.Rt, m.inv, m.apply, m.exp_r, m.log_R
rng = np.random.default_rng(0)

# ---------------- room map: 8 x 6 x 2.5 m box + a pillar (breaks symmetry)
def plane(p0, u, v, nu, nv):
    a, b = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv))
    return p0 + a.ravel()[:, None] * u + b.ravel()[:, None] * v
W, L, H = 8.0, 6.0, 2.5
n = 220
walls = [plane([0, 0, 0], [W, 0, 0], [0, 0, H], n, 80),
         plane([0, L, 0], [W, 0, 0], [0, 0, H], n, 80),
         plane([0, 0, 0], [0, L, 0], [0, 0, H], n, 80),
         plane([W, 0, 0], [0, L, 0], [0, 0, H], n, 80),
         plane([0, 0, 0], [W, 0, 0], [0, L, 0], n, 160),
         plane([0, 0, H], [W, 0, 0], [0, L, 0], n, 160),
         plane([5, 2, 0], [0.6, 0, 0], [0, 0, H], 30, 80),
         plane([5, 2.6, 0], [0.6, 0, 0], [0, 0, H], 30, 80),
         plane([5, 2, 0], [0, 0.6, 0], [0, 0, H], 30, 80),
         plane([5.6, 2, 0], [0, 0.6, 0], [0, 0, H], 30, 80)]
MAP = np.vstack(walls) + rng.normal(0, 0.005, (sum(len(w) for w in walls), 3))
REF = m.Reference(MAP, voxel=0.05, plane_voxel=0.4)

# ---------------- frames
# camera optical frame convention: z forward, x right, y down
# body/child frame: x forward, y left, z up.  X = T_child_cam
X = m.make_T_xyzq([-0.010, 0.060, 0.015, -0.5, 0.5, -0.5, 0.5])
# lidar sits 0.3 m above the camera, yawed 30 deg: T_lidar_camera = pose of cam in lidar
T_lc = Rt(Rot.from_euler("xyz", [0, 0, 30], degrees=True).as_matrix() @ X[:3, :3],
          np.array([0.12, -0.05, -0.30]))
T_cl_true = X @ inv(T_lc)         # T_child_lidar

# ---------------- ground-truth BODY trajectory: a loop in the room, 120 s at 20 Hz
T_END, HZ = 120.0, 20.0
ts = np.arange(0, T_END, 1.0 / HZ)
def body_pose(t):
    # rounded rectangle 1.5..6.5 x 1.2..4.8, one lap plus a dwell at start
    s = max(0.0, t - 8.0) / (T_END - 8.0)          # dwell 8 s facing a board
    ang = 2 * math.pi * s
    c = np.array([4.0 - 0.9, 3.0, 1.0])            # pillar-free centre
    x = 4.0 + 2.4 * math.cos(ang) - 0.9
    y = 3.0 + 1.6 * math.sin(ang)
    z = 1.0 + 0.02 * math.sin(3 * ang)
    yaw = ang + 0.6 * math.sin(5 * ang)
    pitch = 0.03 * math.sin(7 * ang); roll = 0.02 * math.cos(6 * ang)
    return Rt(Rot.from_euler("zyx", [yaw, pitch, roll]).as_matrix(), [x, y, z])
B_true = np.array([body_pose(t) for t in ts])                # T_map_child
C_true = m.compose_all(B_true, X)                             # T_map_cam
L_true = m.compose_all(C_true, inv(T_lc))                     # T_map_lidar

# ---------------- ZED odometry: T_odom_child with drift (yaw-rate bias + scale)
odom = [np.eye(4)]
for i in range(1, len(ts)):
    d = inv(B_true[i - 1]) @ B_true[i]
    w = log_R(d[:3, :3]); v = d[:3, 3]
    w = w + np.array([0, 0, 0.0025 / HZ])             # 0.14 deg/s yaw bias
    v = v * 1.02 + rng.normal(0, 0.0007, 3)          # 2 % scale + jitter
    odom.append(odom[-1] @ Rt(exp_r(w), v))
# a ZED jump: 2 m sideways and 6 deg of yaw at t = 45 s, then normal again
JUMP_I = int(45.0 * HZ)
J = Rt(exp_r([0, 0, math.radians(6)]), [0.0, 2.0, 0.0])
for i in range(JUMP_I, len(odom)):
    odom[i] = odom[JUMP_I - 1] @ J @ inv(odom[JUMP_I - 1]) @ odom[i]
oT = np.array(odom); ot = ts.copy()
# session anchor at the end of the dwell: T_map_cam at t=8 s (+ 5 mm noise)
t_anchor = 8.0
A = m.interp_traj(ts, C_true, np.array([t_anchor]))[0].copy()
A[:3, 3] += rng.normal(0, 0.005, 3)
T_o0 = m.interp_traj(ot, oT, np.array([t_anchor]))[0]
T_map_origin = A @ inv(T_o0 @ X)

# ---------------- lidar scans: map points within 10 m of the true lidar pose
tree = m.cKDTree(MAP)
def scan_at(i):
    p = L_true[i][:3, 3]
    idx = tree.query_ball_point(p, 10.0)
    Pm = MAP[rng.choice(idx, min(len(idx), 12000), replace=False)]
    Pl = apply(inv(L_true[i]), Pm) + rng.normal(0, 0.01, (len(Pm), 3))
    return Pl.astype(np.float32)
scans = ((ts[i], scan_at(i), None) for i in range(0, len(ts), 4))   # 5 Hz

track_l = dict(rate_hz=5.0, range_min=0.7, range_max=10.0, scan_voxel=0.10,
               keep_cloud_pts=3000, deskew=False, seed=os.environ.get("SEED", "lidar"))
print("== chain_lidar (seed=%s) ==" % track_l["seed"])
lts, lTs, RMS, NOBS, cl, n_rej, Q = m.chain_lidar(scans, ot, oT, T_map_origin,
                                                  T_cl_true, REF, track_l, log_every=0)
outd = os.environ.get("TEST_OUT", "/tmp/test_08_out"); os.makedirs(outd, exist_ok=True)
m.report_chain_quality(lts, Q, outd, "mobile_1_lidar", track_l["seed"])
big = [q for q in Q if q[5] > 0.5]
assert len(big) == 1 and abs(big[0][0] - 45.0) < 0.3, "ZED jump not localised: %r" % [(q[0], q[5]) for q in big]
Lt = m.interp_traj(ts, L_true, lts)
e_l, r_l = m.traj_gap(lTs, Lt)
To_l = m.compose_all(np.tile(T_map_origin, (len(lts), 1, 1)) @ m.interp_traj(ot, oT, lts), T_cl_true)
e_o, _ = m.traj_gap(To_l, Lt)
print("  scans %d rejected %d rms med %.2f cm" % (len(lts), n_rej, np.nanmedian(RMS) * 100))
print("  lidar ICP  vs truth: median %.1f cm max %.1f cm rot max %.2f deg"
      % (np.median(e_l) * 100, e_l.max() * 100, math.degrees(r_l.max())))
print("  odom-only  vs truth: median %.1f cm max %.1f cm"
      % (np.median(e_o) * 100, e_o.max() * 100))
m.report_gap("odom - lidar", lts, *m.traj_gap(lTs, To_l), m.path_length(lTs))
m.verify_odom_frames(lts, lTs, m.compose_all(lTs, T_lc), ot, oT, T_cl_true, X, Q, "child")
assert np.median(e_l) < 0.03 and e_l.max() < 0.10, "lidar chain off (max %.2f m)" % e_l.max()
assert e_o.max() > 0.3, "synthetic drift too small to test anything"
assert n_rej == 0, "%d unregistered scans" % n_rej

# ---------------- submap accumulation: 1 s windows of the same scans, stitched
# with the (jumping) odometry, registered as one cloud each
frames = [(ts[i], scan_at(i)[:3000]) for i in range(0, len(ts), 4)]
sub = m.build_submaps(frames, ot, oT, T_cl_true, 1.0, 0.10, 20000)
sts, sTs, _, _, _, s_rej, _ = m.chain_lidar(sub, ot, oT, T_map_origin, T_cl_true, REF,
                                            dict(track_l, min_pts=500), log_every=0)
es, _ = m.traj_gap(sTs, m.interp_traj(ts, L_true, sts))
print("  submap chain vs truth: median %.1f cm max %.1f cm, %d unregistered"
      % (np.median(es) * 100, es.max() * 100, s_rej))
assert np.median(es) < 0.03, "submap chain off"

# ---------------- boards: 2 instances of 'anchor' + 1 'rs_anchor'
def board_T(xyz, yaw_deg):
    # board z-axis pointing INTO the room (toward the camera looking at it)
    return Rt(Rot.from_euler("xyz", [-90, 0, yaw_deg], degrees=True).as_matrix(), xyz)
BM = {"anchor":    (board_T([7.95, 3.0, 1.0], 90), {"design": "anchor", "std_mm": 5}),
      "anchor_b":  (board_T([0.05, 3.0, 1.0], -90),   {"design": "anchor", "std_mm": 5}),
      "rs_anchor": (board_T([3.1, 5.95, 1.0], 180),  {"design": "rs_anchor", "std_mm": 5})}
sights = []
for i in range(0, len(ts), 2):              # 10 Hz image stream
    Tc = C_true[i]
    for name, (Tb, rec) in BM.items():
        T_cb = inv(Tc) @ Tb
        p = T_cb[:3, 3]
        if p[2] < 0.5 or p[2] > 3.0:            # in front, within 2 m
            continue
        if abs(p[0]) > 0.6 * p[2] or abs(p[1]) > 0.4 * p[2]:   # in the FOV
            continue
        if (Tc[:3, :3] @ np.array([0, 0, 1.0])) @ (Tb[:3, :3] @ np.array([0, 0, 1.0])) > -0.3:
            continue                            # must be facing the board
        n_ = Rt(exp_r(rng.normal(0, 0.004, 3)), rng.normal(0, 0.004, 3))
        sights.append((ts[i], rec["design"], T_cb @ n_))
print("\n%d synthetic sightings, boards seen: %s" % (
    len(sights), sorted({d for _, d, _ in sights})))
assert len(sights) > 20

# ---------------- arms with lidar clouds
track_a = dict(odom_sigma_t=0.003, odom_sigma_r=0.001, icp_pts=400, gn_iters=15,
               instance_radius=2.0)
cl_cam = [apply(inv(T_lc), c).astype(np.float32) for c in cl]
Ts_cam = m.compose_all(lTs, T_lc)
print("\n== run_arms (cloud_source=lidar) ==")
g = m.run_arms("mobile_1_zed", lts, Ts_cam, cl_cam, sights, ot, oT, X,
               T_map_origin, BM, sorted(BM), track_a, REF, 0.005, src="lidar",
               verbose=False)
Ct = m.interp_traj(ts, C_true, g["node_t"])
print("\n  vs TRUTH (camera frame):")
res = {}
for nm, T in [("odom only", g["odom_only"]), ("chained lidar", g["chained"])] + \
             sorted(g["arms"].items()):
    dt, dr = m.traj_gap(T, Ct)
    res[nm] = dt
    print("    %-14s median %5.1f cm  p95 %5.1f cm  max %5.1f cm | rot max %.2f deg"
          % (nm, np.median(dt) * 100, np.percentile(dt, 95) * 100, dt.max() * 100,
             math.degrees(dr.max())))
assert res["A_icp"].max() < 0.10, "arm A (odom + lidar map factors) off"
assert res["C_joint"].max() < 0.10, "arm C (joint) off"
assert np.median(res["B_boards"]) < np.median(res["odom only"]) * 0.5, \
    "arm B did not correct the odometry"
assert np.max(res["B_boards"]) < np.max(res["odom only"]) * 0.7
# the honest cells
print("\n  independent checks: arm A board resid %.1f cm (A never saw a board); "
      "arm B map rms %.2f cm (B never saw the map)"
      % (np.nanmedian(m.eval_board_resid(g["arms"]["A_icp"], g["res_nodes"], BM)) * 100,
         np.nanmedian(m.eval_map_rms(g["arms"]["B_boards"], g["clouds"], REF)) * 100))

# ---------------- particle filter: depth-like frames (90 deg cone, 5 m) in the
# BODY frame, the jumping odometry, the same board sightings as absolute fixes
grid = m.Grid2D(MAP, [0.3, 2.2], res=0.05)
pf_frames = []
for i in range(0, len(ts), 2):                       # 10 Hz
    Pm = MAP[rng.choice(len(MAP), 6000, replace=False)]
    Pc = apply(inv(C_true[i]), Pm)                   # camera optical frame
    ok = (Pc[:, 2] > 0.4) & (Pc[:, 2] < 5.0) & (np.abs(Pc[:, 0]) < Pc[:, 2]) & \
         (np.abs(Pc[:, 1]) < 0.7 * Pc[:, 2])
    Pb_ = apply(X, Pc[ok]) + rng.normal(0, 0.01, (ok.sum(), 3))   # body frame
    pf_frames.append((ts[i], Pb_.astype(np.float32)))
pft = np.array([f[0] for f in pf_frames])
bm = {}
for (st, dgn, T_cb) in sights:
    k = int(np.argmin(np.abs(pft - st)))
    if abs(pft[k] - st) > 0.03:
        continue
    name = [n for n, (Tb, r) in BM.items() if r["design"] == dgn and
            np.linalg.norm((C_true[int(np.argmin(np.abs(ts - st)))] @ T_cb)[:3, 3] - Tb[:3, 3]) < 1.0]
    if not name:
        continue
    T_mc = BM[name[0]][0] @ inv(T_cb) @ inv(X)
    bm[pft[k]] = m.level_parts(T_mc)[0]
print("\n== particle filter (%d frames, %d board fixes) ==" % (len(pf_frames), len(bm)))
prm = dict(particles=1500, scan_pts=200, slice_z=[0.3, 2.2])
Ts_pf_c, Qpf = m.pf_localise(pf_frames, ot, oT, X, T_map_origin, bm, grid, prm,
                             np.random.default_rng(1))
Ts_pf = m.compose_all(Ts_pf_c, X)
ep, rp = m.traj_gap(Ts_pf, m.interp_traj(ts, C_true, pft))
after = pft > 47.0
print("  PF vs truth: median %.1f cm  p95 %.1f cm  max %.1f cm | after the jump: "
      "median %.1f cm, max %.1f cm | yaw max %.1f deg"
      % (np.median(ep) * 100, np.percentile(ep, 95) * 100, ep.max() * 100,
         np.median(ep[after]) * 100, ep[after].max() * 100, math.degrees(rp.max())))
assert np.median(ep) < 0.15, "PF median off"
assert np.median(ep[after]) < 0.15, "PF did not recover after the jump"
assert ep[-1] < 0.20, "PF final pose off"

# ---------------- comparison table + plot through the real functions
results = {"mobile_1_lidar": dict(kind="lidar_icp", ts=lts, Ts=lTs, Ts_cam=Ts_cam,
                                  odom_only=To_l, odom_only_cam=m.compose_all(To_l, T_lc),
                                  clouds_cam=cl_cam, rms=float(np.nanmedian(RMS))),
           "mobile_1_zed": dict(kind="arms", ts=g["node_t"], Ts=g["arms"]["C_joint"],
                                arms=g["arms"], res_nodes=g["res_nodes"], bmap=BM,
                                odom_only=g["odom_only"], chained=g["chained"])}
outd = os.environ.get("TEST_OUT", "/tmp/test_08_out"); os.makedirs(outd, exist_ok=True)
m.compare_rig(results, T_lc, outd)
m.save_paths_png(results, REF, BM, outd, T_lc)
# two cloud sets in one graph (lidar at 2 cm + "depth" at 5 cm), only B and C
track_j = dict(track_a, arms_run=["B_boards", "C_depth", "C_joint"], joint_init="chained")
gj = m.run_arms("mobile_1_zed", lts, Ts_cam, None, sights, ot, oT, X, T_map_origin, BM,
                sorted(BM), track_j, REF, 0.005, src="lidar+depth", verbose=False,
                cloud_sets=[(lts, cl_cam, 0.02, "lidar"),
                            (lts + 0.05, cl_cam, 0.05, "depth")])
assert set(gj["arms"]) == {"B_boards", "C_depth", "C_joint"}, set(gj["arms"])
ed, _ = m.traj_gap(gj["arms"]["C_depth"], m.interp_traj(ts, C_true, gj["node_t"]))
# C_depth starts from B (depth ICP can only refine within its 10 cm gate), so
# where B smeared the jump it stays there: no worse than B, cm-level elsewhere
eb, _ = m.traj_gap(gj["arms"]["B_boards"], m.interp_traj(ts, C_true, gj["node_t"]))
assert ed.max() <= eb.max() * 1.2 and np.median(ed) < 0.05, \
    "C_depth off (median %.2f m, max %.2f m vs B max %.2f m)" % (np.median(ed), ed.max(), eb.max())
print("  C_depth (odom + boards + depth clouds) vs truth: median %.1f cm max %.1f cm"
      % (np.median(ed) * 100, ed.max() * 100))
ej, _ = m.traj_gap(gj["arms"]["C_joint"], m.interp_traj(ts, C_true, gj["node_t"]))
assert ej.max() < 0.10, "joint with two cloud sets off (%.2f m)" % ej.max()
print("  joint (lidar+depth clouds) vs truth: median %.1f cm max %.1f cm"
      % (np.median(ej) * 100, ej.max() * 100))
# a rig WITHOUT a lidar (mobile_2 style): the chained depth ICP is the reference
results2 = {"mobile_2_rs": dict(kind="arms", ts=g["node_t"], Ts=g["arms"]["C_joint"],
                                arms=g["arms"], res_nodes=g["res_nodes"], bmap=BM,
                                odom_only=g["odom_only"], chained=g["chained"],
                                cloud_source="depth", chained_label="depth ICP chained")}
m.compare_rig(results2, None, outd)
m.save_paths_png(results2, REF, BM, outd, None)
assert os.path.exists(os.path.join(outd, "paths_mobile_2.png"))
assert os.path.exists(os.path.join(outd, "compare_mobile_2.csv"))
print("\nALL SYNTHETIC CHECKS PASSED")

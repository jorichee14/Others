#!/usr/bin/env python3
"""Synthetic checks for 08_reference_traj.py - no bag, no ROS, no pipeline
modules needed (they are stubbed). Run from the directory holding the stage:

    python3 test_08_synthetic.py

  1. chain_icp in pure mode: the odometry never seeds/rescues a scan, and an
     injected 0.5 m odometry jump appears in the diagnostic column only.
  2. eval_map_stats: a corridor cloud scores the SAME rms 45 cm down the
     corridor - the DOF column is what exposes it.
  3. run_arms: the ingredient arms exist, odom_icp_boards and icp_boards
     recover a drifting+jumping odometry to < 5 cm of the truth.
  4. collect_methods: a failed depth chain is never replaced by a solved arm
     as the plotting reference."""
import sys, os, types, math, io, contextlib
import numpy as np
from scipy.spatial.transform import Rotation as Rot
_pc = types.ModuleType("pipeline_common")
_pc.R_to_q = lambda R: Rot.from_matrix(R).as_quat()
_pc.load_pipeline = lambda p: (_ for _ in ()).throw(RuntimeError("stub"))
_pb = types.ModuleType("pipeline_boards")
_pb.Board = type("Board", (), {}); _pb.read_bag = _pb.pick_intrinsics = lambda *a, **k: None
sys.modules["pipeline_common"] = _pc; sys.modules["pipeline_boards"] = _pb
import importlib.util
_here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s08", os.path.join(_here, "08_reference_traj.py"))
s08 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s08)
rng = np.random.default_rng(3)

def plane_pts(p0, u, v, nu, nv, step=0.04):
    a = np.arange(0, nu, step); b = np.arange(0, nv, step)
    A, B = np.meshgrid(a, b, indexing="ij")
    return p0 + A.reshape(-1, 1) * u + B.reshape(-1, 1) * v

def room_map(L=8.0, W=5.0, H=2.6):
    ex, ey, ez = np.eye(3)
    P = [plane_pts(np.array([0, 0, 0.]), ex, ey, L, W),          # floor
         plane_pts(np.array([0, 0, H]), ex, ey, L, W),           # ceiling
         plane_pts(np.array([0, 0, 0.]), ex, ez, L, H),          # wall y=0
         plane_pts(np.array([0, W, 0.]), ex, ez, L, H),          # wall y=W
         plane_pts(np.array([0, 0, 0.]), ey, ez, W, H),          # wall x=0
         plane_pts(np.array([L, 0, 0.]), ey, ez, W, H),          # wall x=L
         # some furniture so the room is not degenerate anywhere
         plane_pts(np.array([2.5, 1.0, 0.]), ey, ez, 1.2, 1.0),
         plane_pts(np.array([5.5, 3.0, 0.]), ex, ez, 1.0, 1.2)]
    return np.vstack(P)

def scan_at(T, MAP, rmax=6.0, n=1500, noise=0.005):
    Q = s08.apply(s08.inv(T), MAP)
    r = np.linalg.norm(Q, axis=1)
    Q = Q[(r > 0.4) & (r < rmax)]
    Q = Q[rng.choice(len(Q), min(n, len(Q)), replace=False)]
    return (Q + rng.normal(0, noise, Q.shape)).astype(np.float32)

MAP = room_map()
REF = s08.Reference(MAP, voxel=0.05, plane_voxel=0.4)

# truth: a loop inside the room, 80 poses at 10 Hz
N = 80; ts = 100.0 + np.arange(N) * 0.1
truth = []
for i in range(N):
    a = 2 * math.pi * i / N
    p = np.array([4.0 + 2.0 * math.cos(a), 2.5 + 1.2 * math.sin(a), 1.2])
    yaw = a + math.pi / 2
    truth.append(s08.Rt(s08.exp_r([0.02 * math.sin(3 * a), 0.02 * math.cos(2 * a), yaw]), p))
truth = np.array(truth)

# odometry: true increments + slow yaw drift + a 0.5 m jump at step 40
oT = [truth[0].copy()]
for i in range(1, N):
    Z = s08.inv(truth[i - 1]) @ truth[i]
    Zd = s08.Rt(Z[:3, :3] @ s08.exp_r([0, 0, math.radians(0.15)]), Z[:3, 3] * 1.01)
    if i == 40:
        Zd = s08.Rt(Zd[:3, :3], Zd[:3, 3] + np.array([0.5, 0, 0]))
    oT.append(oT[-1] @ Zd)
oT = np.array(oT); ot = ts.copy()
X = np.eye(4); T_cl = np.eye(4)
T_map_origin = truth[0] @ s08.inv(oT[0])         # anchor = truth at t0

print("\n#### 1. chain_icp PURE vs ODOM seeding")
scans = [(ts[i], scan_at(truth[i], MAP), None) for i in range(N)]
res = {}
for mode in ("icp", "odom"):
    trk = dict(seed=mode, rate_hz=0, range_min=0.3, range_max=7.0, scan_voxel=0.05,
               min_pts=300, deskew=False, max_shift=0.5, max_rot_deg=5.0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        c_ts, c_T, RMS, NOBS, cl, n_rej, Q = s08.chain_icp(iter(scans), ot, oT, T_map_origin,
                                                           T_cl, REF, trk, log_every=0)
    err = np.linalg.norm(c_T[:, :3, 3] - truth[:, :3, 3], axis=1)
    st = [q[7] for q in Q]
    from collections import Counter
    print("  seed=%-4s  err median %.2f cm  max %.2f cm  unreg %d  rms %.2f cm  statuses %s"
          % (mode, np.median(err) * 100, err.max() * 100, n_rej, np.nanmedian(RMS) * 100,
             dict(Counter(st))))
    res[mode] = (c_ts, c_T, cl, NOBS, Q)
    if mode == "icp":
        assert "odom" not in Counter(st) and "odom+wide" not in Counter(st), "odom seed leaked"
        assert err.max() < 0.05, "pure chain lost the truth"
assert res["icp"][4][40][5] > 0.3, "the odometry jump must show in the diagnostic column"
print("  step-40 odometry disagreement in the diagnostic column: %.2f m (jump was 0.50 m)" % res["icp"][4][40][5])

print("\n#### 2. eval_map_stats DOF: room vs corridor")
c_ts, c_T, cl, NOBS, Q = res["icp"]
clouds = {i: (cl[i], 0.02) for i in range(N)}
r_, inl, dof = s08.eval_map_stats(truth, clouds, REF)
print("  room:     rms %.2f cm  inlier %.0f%%  DOF median %d" % (np.nanmedian(r_) * 100, 100 * np.nanmean(inl), np.nanmedian(dof)))
assert np.nanmedian(dof) == 6
CORR = np.vstack([plane_pts(np.array([0, 0, 0.]), np.eye(3)[0], np.eye(3)[1], 20, 2),
                  plane_pts(np.array([0, 0, 0.]), np.eye(3)[0], np.eye(3)[2], 20, 2.6),
                  plane_pts(np.array([0, 2, 0.]), np.eye(3)[0], np.eye(3)[2], 20, 2.6)])
REFC = s08.Reference(CORR, voxel=0.05, plane_voxel=0.4)
Tc = s08.Rt(np.eye(3), [8.0, 1.0, 1.2])
Pc = scan_at(Tc, CORR, rmax=5.0)
Tslid = s08.Rt(np.eye(3), [8.45, 1.0, 1.2])           # 45 cm along the corridor
for lbl, T_ in (("true pose", Tc), ("slid 45 cm along corridor", Tslid)):
    r_, inl, dof = s08.eval_map_stats(np.array([T_]), {0: (Pc, 0.02)}, REFC)
    print("  corridor %-27s rms %.2f cm  inlier %.0f%%  DOF %d" % (lbl, r_[0] * 100, 100 * inl[0], dof[0]))
    assert dof[0] < 6
print("  -> same rms at the wrong pose: exactly the blindness the DOF column flags")

print("\n#### 3. run_arms with the ingredient arms")
# boards: two boards on the walls, sightings when within 1.5 m and facing
bmap = {"b1": (s08.Rt(s08.exp_r([0, 0, math.pi / 2]) @ s08.exp_r([-math.pi / 2, 0, 0]), [4.0, 0.0, 1.2]), dict(std_mm=3)),
        "b2": (s08.Rt(s08.exp_r([0, 0, -math.pi / 2]) @ s08.exp_r([-math.pi / 2, 0, 0]), [4.0, 5.0, 1.2]), dict(std_mm=3))}
sights = []
for i in range(N):
    for bn, (Tb, _) in bmap.items():
        d = np.linalg.norm(truth[i][:3, 3] - Tb[:3, 3])
        if d < 1.6:
            T_cb = s08.inv(truth[i]) @ Tb
            T_cb = s08.Rt(T_cb[:3, :3] @ s08.exp_r(rng.normal(0, 0.003, 3)), T_cb[:3, 3] + rng.normal(0, 0.004, 3))
            sights.append((ts[i], bn, T_cb))
print("  %d synthetic sightings" % len(sights))
trk = dict(odom_sigma_t=0.003, odom_sigma_r=0.001, icp_pts=300, gn_iters=15, instance_radius=2.0,
           odom_jump_check=True)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = s08.run_arms("rig_cam", c_ts, c_T, cl, sights, ot, oT, X, T_map_origin, bmap, ["b1", "b2"],
                     trk, REF, 0.003, src="lidar", outd=None, verbose=False, chain_nobs=np.array(NOBS))
out = buf.getvalue()
print("\n".join(l for l in out.split("\n") if "== evaluation" in l or "arm " in l[:6] or l.startswith("  odom") or l.startswith("  icp") or "freed" in l or "held out" in l or "DOF <" in l))
assert set(g["arms"]) == {"odom_icp", "odom_boards", "icp_boards", "odom_icp_boards"}, set(g["arms"])
node_t = g["node_t"]; tr_n = s08.interp_traj(ts, truth, node_t)
print("  error vs TRUTH (cm, median / max):")
for nm, Ts in [("odom", g["odom_only"]), ("icp", g["chained"])] + list(g["arms"].items()):
    e = np.linalg.norm(Ts[:, :3, 3] - tr_n[:, :3, 3], axis=1) * 100
    print("    %-16s %6.1f / %6.1f" % (nm, np.median(e), e.max()))
e_joint = np.linalg.norm(g["arms"]["odom_icp_boards"][:, :3, 3] - tr_n[:, :3, 3], axis=1)
e_ib = np.linalg.norm(g["arms"]["icp_boards"][:, :3, 3] - tr_n[:, :3, 3], axis=1)
e_od = np.linalg.norm(g["odom_only"][:, :3, 3] - tr_n[:, :3, 3], axis=1)
assert e_joint.max() < 0.05 and e_ib.max() < 0.05 and e_od.max() > 0.3
print("  OK: joint and icp_boards within 5 cm of truth everywhere; raw odom %.0f cm off at worst" % (e_od.max() * 100))

print("\n#### 4. collect_methods: failed depth chain must NOT promote an arm to reference")
fake = {"rig2_rs": dict(kind="arms", ts=node_t, Ts=g["arms"]["odom_icp_boards"], arms=g["arms"],
                        odom_only=g["odom_only"], chained=g["chained"], chained_label="depth ICP chained",
                        chain_ok=False, arm_clouds=g["arm_clouds"], res_nodes=g["res_nodes"], bmap=bmap)}
methods, has_ref = s08.collect_methods(fake, "rig2_rs")
print("  reference entry: %r" % methods[0][0]); print("  methods: %s" % [m[0] for m in methods[1:]])
assert has_ref and "odom only (baseline" in methods[0][0]
assert not any("A_icp" in m[0] or "(reference: depth chain failed)" in m[0] for m in methods)
fake["rig2_rs"]["chain_ok"] = True
methods, has_ref = s08.collect_methods(fake, "rig2_rs")
assert "depth ICP chained" in methods[0][0]
print("  chain_ok=True -> reference is the chain: %r" % methods[0][0])
print("\nALL TESTS PASSED")

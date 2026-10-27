#!/usr/bin/env python3
"""How accurate a run's trajectory is, against the pass's reference: the
stage 04 LiDAR track (data/processed/<date>/<pass>/odometry/reference_<tag>/
traj_<machine>_lidar.tum, T_map_lidar).

    python scoop_pipeline/slam/odometry/score.py <run dir> [--ref TUM]

run_odometry.sh runs it after every run; it can be run again on any run
folder (run.json names the bag, the robot and the frame of trajectory.tum).
The estimate is first put in the LiDAR frame (make_config.to_lidar: FAST-LIO2's
and OpenVINS's IMU, RTAB-Map's camera), then compared with
analysis/trajs.py, as check.py trajs compares the pass's own trajectories.

Writes <run dir>/accuracy.json and adds it to summary.json as "accuracy":
  ate               after the best rigid fit of the estimate's poses onto the
                    reference's (positions and orientations: a near-planar path
                    leaves roll and pitch free in a fit of positions alone, which
                    is kept as ate_position_only_fit): position cm and rotation
                    deg, rmse / median / p95 / max
  rpe_<d>s          the motion over d = 1, 10, 100 s against the reference's,
                    translation cm and rotation deg
  drift_<d>m_pct    the translation error over every d = 5, 10 m of travel, as
                    % of d (the KITTI-style odometry number)
  in_map            only for a run in map (run.json "world": "map", the
                    fast2lamaa_loc localizer): the error as it is, no fit and no
                    alignment at all: position cm and rotation deg, rmse /
                    median / p95 / max (what a consumer of the map pose sees)
  start_aligned     the estimate put on the reference at its first pose, no
                    fit: the error at the end (cm and % of the path), and its
                    max: drift as it accumulated
  path              path length of each from poses 1 s apart (pose-to-pose
                    jitter at 10 Hz would add length the robot did not travel),
                    their ratio, and the scale of the best similarity fit
                    (sim3_scale: the estimate's size against the reference's,
                    1.01 = 1 % too long)
  coverage          the share of the played span with an estimate pose no more
                    than 0.5 s away; the gaps longer than that
  jumps             steps faster than 3 m/s between consecutive poses (a hand-
                    pushed cart and a Scout Mini do not move that fast)
The reference is the pipeline's own output: an ATE near its uncertainty
(about 1-3 cm) is agreement with it, not accuracy beyond it.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, os.path.join(ROOT, "datasets"))

import make_config as MK                                            # noqa: E402
import trajs                                                        # noqa: E402
from pipeline_common import load_traj                               # noqa: E402

GAP_S = 0.5
JUMP_M_S = 3.0
DIST = (5.0, 10.0)


def default_ref(bag, machine):
    import swarm_slam_lidar as SL
    return SL.default_gt(bag, machine)


def drift(t, E, R, d):
    """Translation error over every d m of the reference's travel, % of d."""
    p = R[:, :3, 3]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    j = np.searchsorted(s, s + d)
    ok = j < len(s)
    i, j = np.flatnonzero(ok), j[ok]
    if not len(i):
        return None

    def rel(T):
        return np.einsum("nji,nj->ni", T[i, :3, :3], T[j, :3, 3] - T[i, :3, 3])
    return trajs.stats(np.linalg.norm(rel(E) - rel(R), axis=1) / d * 100.0)


def pose_fit(E, R, arm=1.0):
    """The rigid transform (R, t) putting the estimate's poses on the
    reference's: positions and, so that a near-planar path still fixes roll and
    pitch, a point `arm` m along each pose's three axes (Kabsch, no scale)."""
    def pts(T):
        return np.concatenate([T[:, :3, 3]] + [T[:, :3, 3] + arm * T[:, :3, k] for k in range(3)])
    return trajs.rigid_fit(pts(E), pts(R))


def sim3_scale(E, R):
    """The estimate's size against the reference's, from the best similarity
    fit of its positions onto them (Umeyama): 1.01 is 1 % too long."""
    a, b = E[:, :3, 3] - E[:, :3, 3].mean(0), R[:, :3, 3] - R[:, :3, 3].mean(0)
    U, D, Vt = np.linalg.svd(b.T @ a / len(a))
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return float((a ** 2).sum(1).mean() / np.trace(np.diag(D) @ S))


def every(t, T, step):
    """The poses at least `step` s apart: path length without the pose-to-pose jitter."""
    keep, last = [], -np.inf
    for k, ti in enumerate(t):
        if ti - last >= step:
            keep.append(k)
            last = ti
    return T[keep]


def score(t, E, tr, Tr, window=None, in_map=False):
    """Accuracy of (t, E) against the reference (tr, Tr), both T_world_lidar
    (in_map: the estimate's world is the reference's map, so also unaligned)."""
    lo, hi = max(t[0], tr[0]), min(t[-1], tr[-1])
    if hi <= lo:
        return {"error": "the estimate and the reference do not overlap in time"}
    c, _ = trajs.compare("estimate", "reference", t, E, tr, Tr)
    m = (t >= lo) & (t <= hi)
    ts, Es = t[m], E[m]
    Rs = trajs.interp(tr, Tr, ts)
    A = Rs[0] @ np.linalg.inv(Es[0])                    # the estimate on the reference at its start
    Ea = np.einsum("ij,njk->nik", A, Es)
    err = np.linalg.norm(Ea[:, :3, 3] - Rs[:, :3, 3], axis=1)
    Rf, tf = pose_fit(Es, Rs)
    Ef = np.einsum("ij,njk->nik", Rf, Es[:, :3, :])
    dp = np.linalg.norm(Ef[:, :, 3] + tf - Rs[:, :3, 3], axis=1)
    dR = trajs.rot_deg(np.einsum("nji,njk->nik", Ef[:, :, :3], Rs[:, :3, :3]))
    path_e, path_r = trajs.path_length(every(ts, Es, 1.0)), trajs.path_length(every(ts, Rs, 1.0))
    out = {"poses_scored": int(m.sum()), "span_s": round(float(ts[-1] - ts[0]), 2),
           "ate": {"position_cm": trajs.stats(dp * 100), "rotation_deg": trajs.stats(dR)},
           "ate_position_only_fit": {"position_cm": c["after_rigid_fit_cm"],
                                     "rotation_deg": c["after_rigid_fit_deg"]},
           "start_aligned": {"end_cm": round(float(err[-1] * 100), 2),
                             "end_pct_of_path": round(float(err[-1] / path_r * 100), 3) if path_r > 0 else None,
                             "max_cm": round(float(err.max() * 100), 2)},
           "path": {"estimate_m": round(path_e, 3), "reference_m": round(path_r, 3),
                    "ratio": round(path_e / path_r, 4) if path_r > 0 else None,
                    "sampled_s": 1.0, "sim3_scale": round(sim3_scale(Es, Rs), 4)}}
    if in_map:
        out["in_map"] = {"position_cm": trajs.stats(np.linalg.norm(Es[:, :3, 3] - Rs[:, :3, 3], axis=1) * 100),
                         "rotation_deg": trajs.stats(trajs.rot_deg(
                             np.einsum("nji,njk->nik", Es[:, :3, :3], Rs[:, :3, :3])))}
    for d in trajs.DELTAS:
        r = c["relative"]["%gs" % d]
        if r["translation_cm"] is not None:
            out["rpe_%gs" % d] = {"translation_cm": r["translation_cm"], "rotation_deg": r["rotation_deg"]}
    for d in DIST:
        v = drift(ts, Es, Rs, d)
        if v is not None:
            out["drift_%gm_pct" % d] = v
    # coverage of the played span, gaps, jumps
    w0, w1 = window if window else (lo, hi)
    tt = t[(t >= w0) & (t <= w1)]
    edges = np.concatenate([[w0], tt, [w1]])
    gaps = np.diff(edges)
    out["coverage"] = {"pct": round(float(100.0 * (1.0 - gaps[gaps > GAP_S].sum() / (w1 - w0))), 2)
                       if w1 > w0 else None,
                       "gaps_over_%gs" % GAP_S: [[round(float(edges[k] - w0), 2), round(float(gaps[k]), 2)]
                                                 for k in np.flatnonzero(gaps > GAP_S)][:20]}
    v = np.linalg.norm(np.diff(E[:, :3, 3], axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)
    big = np.flatnonzero(v > JUMP_M_S)
    out["jumps"] = {"over_m_s": JUMP_M_S, "count": int(len(big)),
                    "at_s": [round(float(t[k] - t[0]), 2) for k in big[:20]],
                    "max_m_s": round(float(v.max()), 2) if len(v) else None}
    return out


def run_score(run, ref=None):
    rj = json.load(open(os.path.join(run, "run.json")))
    tum = os.path.join(run, "trajectory.tum")
    if not os.path.exists(tum) or os.path.getsize(tum) == 0:
        return {"error": "no trajectory.tum"}
    ref = ref or rj.get("ref") or default_ref(rj["bag"], rj["machine"])
    if not ref or not os.path.exists(ref):
        return {"error": "no reference at %s (give --ref)" % ref}
    paths = MK.mc._paths(rj["bag"])
    T_fl, src = MK.to_lidar(paths, MK.mc.build_tf_static(paths), rj["machine"], rj["tum_frame"])
    t, T = load_traj(tum)
    E = np.einsum("nij,jk->nik", T, T_fl)                # T_odom_frame @ T_frame_lidar
    tr, Tr = load_traj(ref)
    window = None
    s = os.path.join(run, "summary.json")
    msgs = os.path.join(run, "messages.csv")
    if os.path.exists(msgs):
        import csv
        st = [int(r["stamp_ns"]) * 1e-9 for r in csv.DictReader(open(msgs))
              if r["topic"] == rj["key_topic"] and int(r["stamp_ns"]) > 0]
        if st:
            window = (min(st), max(st))
    out = score(t, E, tr, Tr, window, in_map=rj.get("world") == "map")
    out.update({"reference": ref, "estimate_frame": rj["tum_frame"], "to_lidar": src})
    json.dump(out, open(os.path.join(run, "accuracy.json"), "w"), indent=1)
    if os.path.exists(s):
        summ = json.load(open(s))
        summ["accuracy"] = out
        json.dump(summ, open(s, "w"), indent=1)
    return out


def report(a):
    if "error" in a:
        return "  accuracy: %s" % a["error"]
    f = lambda d, k="rmse": "-" if not d else "%.2f" % d[k]                     # noqa: E731
    lines = ["  ATE %s cm / %s deg rmse (median %s cm, max %s cm) over %d poses, %.1f s"
             % (f(a["ate"]["position_cm"]), f(a["ate"]["rotation_deg"]), f(a["ate"]["position_cm"], "median"),
                f(a["ate"]["position_cm"], "max"), a["poses_scored"], a["span_s"])]
    if "in_map" in a:
        lines.append("  in map, no alignment: %s cm / %s deg rmse (median %s cm, p95 %s cm, max %s cm)"
                     % (f(a["in_map"]["position_cm"]), f(a["in_map"]["rotation_deg"]),
                        f(a["in_map"]["position_cm"], "median"), f(a["in_map"]["position_cm"], "p95"),
                        f(a["in_map"]["position_cm"], "max")))
    rpe = ", ".join("%s %s cm / %s deg" % (k[4:], f(v["translation_cm"]), f(v["rotation_deg"]))
                    for k, v in a.items() if k.startswith("rpe_"))
    if rpe:
        lines.append("  RPE (rmse) %s" % rpe)
    dr = ", ".join("%s m %s %%" % (k[len("drift_"):-len("m_pct")], f(v, "median"))
                   for k, v in a.items() if k.startswith("drift_"))
    if dr:
        lines.append("  drift (median) per %s" % dr)
    sa, p = a["start_aligned"], a["path"]
    lines.append("  from the start: %.1f cm off at the end (%s %% of %.1f m), %.1f cm at most; path ratio %s, "
                 "scale %s" % (sa["end_cm"], sa["end_pct_of_path"], p["reference_m"], sa["max_cm"], p["ratio"],
                               p["sim3_scale"]))
    cv, j = a["coverage"], a["jumps"]
    lines.append("  coverage %s %%, %d gaps > %g s; %d jumps > %g m/s"
                 % (cv["pct"], len(cv["gaps_over_%gs" % GAP_S]), GAP_S, j["count"], j["over_m_s"]))
    lines.append("  (against %s; estimate in %s, to the LiDAR by %s)"
                 % (os.path.basename(a["reference"]), a["estimate_frame"], a["to_lidar"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--ref", default=None, help="T_map_lidar TUM (default: stage 04's for the run's robot)")
    a = ap.parse_args()
    try:
        out = run_score(a.run, a.ref)
    except (MK.Refused, MK.mc.ConvertError) as e:
        out = {"error": str(e)}
    print(report(out))
    if "error" in out:
        sys.exit(1)


if __name__ == "__main__":
    main()

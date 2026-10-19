#!/usr/bin/env python
"""Self-test for analysis/method_figures.py (check.py method) on a synthetic
pass: a room 8 x 5 m, a robot driving two laps 1 m inside its walls, the
reference poses true and the BA input drifting (up to 6 cm, 0.6 deg); each
1 s keyframe holds the wall points within 3 m in its own frame, as 02 caches
them; the TSDF map is the walls at 0.6 cm noise, the union at 2.5 cm. The
spot must be a wall seen on both laps, its keyframes thinner after BA than
before, the TSDF thinner than the union, and every figure written.

    python scoop_pipeline/tests/test_method_figures.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RNG = np.random.default_rng(5)
W, H, Z = 8.0, 5.0, 2.5


def walls(step=0.02):
    """Points of the four walls (and the floor, coarser)."""
    s = np.arange(0, 1, step / max(W, H))
    zs = np.arange(0.1, Z, step)
    out = []
    for (x0, y0), (x1, y1) in (((0, 0), (W, 0)), ((W, 0), (W, H)), ((W, H), (0, H)), ((0, H), (0, 0))):
        L = np.hypot(x1 - x0, y1 - y0)
        a = np.arange(0, L, step)
        A, Zz = np.meshgrid(a, zs)
        out.append(np.stack([x0 + (x1 - x0) * A.ravel() / L, y0 + (y1 - y0) * A.ravel() / L, Zz.ravel()], 1))
    fx, fy = np.meshgrid(np.arange(0, W, 0.05), np.arange(0, H, 0.05))
    out.append(np.stack([fx.ravel(), fy.ravel(), np.zeros(fx.size)], 1))
    del s
    return np.vstack(out)


def path(laps=2, speed=0.5, rate=10.0):
    """Poses (t, 4x4) around the room 1 m inside its walls."""
    corners = np.array([[1, 1], [W - 1, 1], [W - 1, H - 1], [1, H - 1], [1, 1]], float)
    seg = np.diff(corners, axis=0)
    L = np.linalg.norm(seg, axis=1)
    per = L.sum()
    t = np.arange(0, laps * per / speed, 1 / rate)
    s = (t * speed) % per
    T = np.tile(np.eye(4), (len(t), 1, 1))
    cum = np.concatenate([[0], np.cumsum(L)])
    for i in range(4):
        m = (s >= cum[i]) & (s < cum[i + 1])
        f = (s[m] - cum[i]) / L[i]
        T[m, :2, 3] = corners[i] + f[:, None] * seg[i]
        h = np.arctan2(seg[i, 1], seg[i, 0])
        T[m, :2, :2] = [[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]]
    T[:, 2, 3] = 0.5
    return t, T


def drift(t, T):
    """The BA input: the true path with a slow offset and heading error."""
    D = T.copy()
    f = np.sin(2 * np.pi * t / t[-1] * 1.5)
    yaw = np.radians(0.6) * f
    for i in range(len(t)):
        c, s = np.cos(yaw[i]), np.sin(yaw[i])
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        D[i, :3, :3] = Rz @ T[i, :3, :3]
        D[i, :3, 3] = T[i, :3, 3] + np.array([0.06 * f[i], -0.04 * f[i], 0])
    return D


def write_tum(path_, t, T):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(T[:, :3, :3]).as_quat()
    np.savetxt(path_, np.column_stack([t, T[:, :3, 3], q]), fmt="%.6f")


def main():
    tmp = tempfile.mkdtemp()
    f = []
    try:
        G = walls()
        wall = G[:, 2] > 0
        M = G + np.where(wall[:, None], RNG.normal(0, 0.006, G.shape) * [[1, 1, 0]], 0)
        U = np.vstack([G + RNG.normal(0, 0.025, G.shape) * [[1, 1, 0]], G[::3] + [0.05, 0.05, 0]])
        np.save(os.path.join(tmp, "tsdf.npy"), M)
        np.save(os.path.join(tmp, "union.npy"), U)
        t, T = path()
        Ti = drift(t, T)
        write_tum(os.path.join(tmp, "ref.txt"), t, T)
        write_tum(os.path.join(tmp, "input.txt"), t, Ti)
        sys.path.insert(0, os.path.join(ROOT, "analysis"))
        from method_figures import keyframe_plan
        _, anchor = keyframe_plan(t, 1.0)
        thin = G[wall][::4]
        clouds = []
        for k in anchor:
            near = np.linalg.norm(thin[:, :2] - T[k, :2, 3], axis=1) < 3.0
            p = thin[near] + RNG.normal(0, 0.006, (near.sum(), 3)) * [[1, 1, 0]]
            clouds.append(((p - T[k, :3, 3]) @ T[k, :3, :3]).astype(np.float32))   # the sensor's frame
        np.savez(os.path.join(tmp, "ref_keyframes.npz"), points=np.vstack(clouds),
                 sizes=np.array([len(c) for c in clouds]), key="{}")
        cfg = os.path.join(tmp, "pipeline_config.json")
        json.dump({"dataset": {"bag": os.path.join(tmp, "none.mcap"), "out_dir": tmp},
                   "02_bundle_adjust": {"kf_period": 1.0}}, open(cfg, "w"))
        out = os.path.join(tmp, "figs")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "check.py"), "method", cfg, "--out", out,
                            "--tsdf", os.path.join(tmp, "tsdf.npy"), "--union", os.path.join(tmp, "union.npy"),
                            "--ref", os.path.join(tmp, "ref.txt"), "--input", os.path.join(tmp, "input.txt"),
                            "--keyframes", os.path.join(tmp, "ref_keyframes.npz"), "--probes", "120", "--gap", "10"],
                           capture_output=True, text=True)
        if r.returncode:
            f.append("check.py method failed:\n" + r.stdout[-2000:] + r.stderr[-3000:])
        else:
            sp = json.load(open(os.path.join(out, "method_spot.json")))
            th = sp["thickness"]
            if sp["visits"] < 2:
                f.append("spot seen on %d visit(s), want 2 laps" % sp["visits"])
            c = np.array(sp["centre"])
            if min(c[0], W - c[0], c[1], H - c[1]) > 0.05:
                f.append("spot %s is not on a wall" % c)
            if not th["keyframes_after_cm"] < th["keyframes_before_cm"]:
                f.append("keyframes not thinner after BA: %s" % th)
            if not th["tsdf_cut_cm"] < th["union_cut_cm"]:
                f.append("TSDF not thinner than the union: %s" % th)
            if not 1.0 < th["map_patch_cm"] < 3.5:
                f.append("TSDF patch %.1f cm, want about 3.29 x 0.6 = 2.0" % th["map_patch_cm"])
            for name in ("method_overview", "method_thickness", "method_views", "method_fusion", "method"):
                for ext in (".png", ".pdf"):
                    if not os.path.isfile(os.path.join(out, name + ext)):
                        f.append("no " + name + ext)
            if os.environ.get("KEEP_FIGS"):
                shutil.copytree(out, os.environ["KEEP_FIGS"], dirs_exist_ok=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "method figures ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()

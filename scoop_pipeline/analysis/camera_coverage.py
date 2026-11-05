#!/usr/bin/env python3
"""Fixed cameras re-aimed for the most floor seen by two of them or more:
each camera keeps its place, only its pan and tilt change.

    python3 scoop_pipeline/analysis/camera_coverage.py <cameras.yaml> <out dir>
        --map <room map .ply/.pcd> --mesh <room mesh .ply> --room YAW XMIN XMAX YMIN YMAX
        [--intrinsics <file or folder>] [--intrinsics-all <one camera's file>]
        [--hfov 90] [--vfov 60] [--size 1280x720]
        [--cell 0.15] [--heights 0.2,1.0,1.6] [--min-heights 2] [--all-floor]
        [--max-range 10] [--min-ppm 50] [--margin 0.02]
        [--pan-step 5] [--tilt 5,85,5] [--max-change 0] [--restarts 20]
        [--weight-one 0.1] [--exact 0]

    e.g. ED305 (place_cameras.py's cameras, the cleaned room and its mesh, crop_room's frame):
    python3 scoop_pipeline/analysis/camera_coverage.py $R/cameras_ed305/cameras.yaml $R/coverage_ed305 \\
        --map $R/mapping/ed305_room_ceil.ply --mesh $R/mapping/ed305_mesh.ply \\
        --room 0.51 -0.03 11.21 -0.18 6.52

The model (the visual-sensor-network coverage problem; Mavrinac & Chen,
IJCV 2013; Hoerster & Lienhart, VSSN 2006):

Targets. The floor (a plane fitted to the map's lowest strong level, so a
tilted map does not matter) in --cell squares inside the room (--room, as
crop_room.py gives it, 0.1 m in from the walls). Free floor only (no map
point 0.15..1.8 m over it: furniture, counters), unless --all-floor. Each
cell a person-high column: points at --heights over the floor.

Intrinsics. Per camera from --intrinsics (the ED305 intrinsics json:
K, image_size, dist k1 k2 p1 p2 k3; a ROS camera_info yaml; a folder of
cam_<N>_<session>.json, the latest session each), else --intrinsics-all's
(one camera's, for all), else a distortion-free --hfov x --vfov at --size.
The lens distortion counts: a barrel lens sees wider than its fx says.

Seen. Camera c with aim a sees a point when: it is in the image (projected
through the lens, --margin of the size in from the edges), in front,
within --max-range, at --min-ppm pixels per metre or more (fx / depth), and
nothing of --mesh lies between (a ray from the point to the camera: the
camera may sit in the wall or ceiling it hangs on by up to 0.25 m). A cell is
seen by c when --min-heights of its points are.

Aims. Each camera's place as calibrated; its optical axis at every pan
(--pan-step) x tilt below the horizon (--tilt from,to,step degrees), the
image upright (no roll), plus its aim as it is now. --max-change > 0 keeps
the new aims within that many degrees of the present ones.

Objective. One aim per camera, maximising
    #cells seen by >= 2 cameras + --weight-one x #cells seen by >= 1,
the same weight for every cell. Solved by coordinate ascent (each camera in
turn takes its best aim given the others, until none changes) from the
present aims and from --restarts random ones; with --exact S > 0 also as an
integer program (scipy's HiGHS, S seconds) over the best aims of each camera,
which also bounds how far from the optimum the answer can be.

<out dir>:
  coverage.json          before and after: cells seen by 0 / 1 / 2 / 3+ cameras, per
                         camera its pan / tilt now and new and the turn between
  cameras_aimed.yaml     the cameras with their new aims (place_cameras.py's format)
  coverage_before.png    the floor from above: red 0, orange 1, green 2, dark green 3+,
  coverage_after.png     grey not a target (furniture); each camera a dot, its aim a line
Needs open3d, scipy, opencv.
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "processing"))


# ---------------------------------------------------------------- frames
def T_of(t, q):
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T


def rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def aim_rotation(pan, tilt, up=np.array([0.0, 0.0, 1.0])):
    """Camera -> map rotation (columns: x right, y down, z forward), the
    optical axis at pan (deg, from +x) and tilt (deg, + up), no roll."""
    p, t = np.radians(pan), np.radians(tilt)
    z = np.array([np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), np.sin(t)])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.c_[x, y, z]


def pan_tilt(R_mc):
    z = R_mc[:, 2]
    return float(np.degrees(np.arctan2(z[1], z[0]))), float(np.degrees(np.arcsin(np.clip(z[2], -1, 1))))


def axis_angle(Ra, Rb):
    """Degrees between two optical axes."""
    return float(np.degrees(np.arccos(np.clip(Ra[:, 2] @ Rb[:, 2], -1, 1))))


# ---------------------------------------------------------------- intrinsics
def parse_intrinsics(e):
    """dict W H fx fy cx cy D (and r_max) from one camera's calibration: the
    ED305 intrinsics json (K / camera_matrix, image_size / img_w img_h, dist /
    dist_coeffs / distortion), a ROS camera_info yaml, or a flat fx fy cx cy."""
    import re as _re
    if "camera_matrix" in e and isinstance(e["camera_matrix"], dict):          # ROS camera_info yaml
        k = e["camera_matrix"]["data"]
        K = np.array(k, float).reshape(3, 3)
        W, H = int(e["image_width"]), int(e["image_height"])
        D = (e.get("distortion_coefficients") or {}).get("data") or []
    elif "K" in e or "camera_matrix" in e or "k" in e:
        K = np.array(e.get("K") or e.get("camera_matrix") or e.get("k"), float).reshape(3, 3)
        if "image_size" in e:
            W, H = (int(v) for v in e["image_size"])
        else:
            W = int(e.get("img_w") or e.get("width") or e.get("image_width"))
            H = int(e.get("img_h") or e.get("height") or e.get("image_height"))
        D = e.get("dist") or e.get("dist_coeffs") or e.get("distortion") or e.get("D") or e.get("d") or []
    else:
        K = np.array([[e["fx"], 0, e["cx"]], [0, e["fy"], e["cy"]], [0, 0, 1.0]])
        W, H = int(e["width"]), int(e["height"])
        D = e.get("dist") or e.get("distortion") or []
    out = {"W": W, "H": H, "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]),
           "cy": float(K[1, 2]), "D": np.array(D, float).ravel()[:5]}
    out["r_max"] = edge_radius(out)
    return out


def edge_radius(k):
    """The largest normalised radius the image reaches (its edges undistorted),
    a little over: past it a distortion polynomial can fold back into view."""
    import cv2
    W, H = k["W"], k["H"]
    e = np.r_[np.c_[np.linspace(0, W - 1, 50), np.zeros(50)], np.c_[np.linspace(0, W - 1, 50), np.full(50, H - 1)],
              np.c_[np.zeros(50), np.linspace(0, H - 1, 50)], np.c_[np.full(50, W - 1), np.linspace(0, H - 1, 50)]]
    K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]])
    D = k["D"] if len(k["D"]) else np.zeros(5)
    u = cv2.undistortPoints(e.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)
    return float(np.linalg.norm(u, axis=1).max() * 1.05)


def fov_deg(k):
    """(horizontal, vertical) field of view, distortion included."""
    import cv2
    K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]])
    D = k["D"] if len(k["D"]) else np.zeros(5)
    p = np.array([[0, k["cy"]], [k["W"] - 1, k["cy"]], [k["cx"], 0], [k["cx"], k["H"] - 1]], np.float64)
    u = cv2.undistortPoints(p.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    return (float(np.degrees(np.arctan(-u[0, 0]) + np.arctan(u[1, 0]))),
            float(np.degrees(np.arctan(-u[2, 1]) + np.arctan(u[3, 1]))))


def load_intrinsics(path):
    """{camera name: intrinsics} from a file (one camera's, or a table of
    them) or a folder of files (cam_<N>[_<session>].json / .yaml; the
    latest session of each camera)."""
    import glob
    import re as _re
    path = os.path.expanduser(path)
    read = lambda f: json.load(open(f)) if f.endswith(".json") else yaml.safe_load(open(f))   # noqa: E731
    files = sorted(glob.glob(os.path.join(path, "*.json")) + glob.glob(os.path.join(path, "*.y*ml"))) \
        if os.path.isdir(path) else [path]
    best = {}
    for f in files:
        try:
            e = read(f)
        except Exception:
            continue
        if not isinstance(e, dict):
            continue
        one = any(k in e for k in ("K", "camera_matrix", "k", "fx"))
        if not one:                                                    # a table: name -> calibration
            for name, v in e.items():
                if isinstance(v, dict):
                    best[str(name)] = (0, parse_intrinsics(v), f)
            continue
        m = _re.search(r"cam_?(\d+)(?:_(\d+))?", os.path.basename(f))
        cid = e.get("camera_id", e.get("camera"))
        if cid is None and m:
            cid = int(m.group(1))
        if cid is None:
            continue
        session = e.get("session", int(m.group(2)) if m and m.group(2) else 0)
        name = "cam_%d" % int(cid)
        if name not in best or session >= best[name][0]:
            best[name] = (session, parse_intrinsics(e), f)
    return {k: (v[1], v[2]) for k, v in best.items()}


def pinhole(hfov, vfov, size):
    W, H = size
    k = {"W": W, "H": H, "fx": (W / 2) / np.tan(np.radians(hfov) / 2), "fy": (H / 2) / np.tan(np.radians(vfov) / 2),
         "cx": W / 2 - 0.5, "cy": H / 2 - 0.5, "D": np.zeros(0)}
    k["r_max"] = edge_radius(k)
    return k


# ---------------------------------------------------------------- floor
def floor_plane(P):
    """(a, b, c) of z = a + b x + c y: the floor, robustly, from the lowest
    strong height peak."""
    z = P[:, 2]
    h, e = np.histogram(z, np.arange(z.min(), z.max() + 0.02, 0.02))
    hs = np.convolve(h, np.ones(3) / 3, mode="same")
    c = 0.5 * (e[:-1] + e[1:])
    low = c < c[0] + 0.5 * (c[-1] - c[0])
    pk = [i for i in np.flatnonzero(low) if 0 < i < len(hs) - 1 and hs[i] >= hs[i - 1] and hs[i] > hs[i + 1]]
    top = max(hs[i] for i in pk)
    f = c[min(i for i in pk if hs[i] >= 0.3 * top)]
    sel = np.abs(z - f) < 0.08
    abc = np.array([f, 0.0, 0.0])
    for tol in (0.08, 0.04, 0.03):
        A = np.c_[np.ones(sel.sum()), P[sel, :2]]
        abc, *_ = np.linalg.lstsq(A, z[sel], rcond=None)
        sel = np.abs(z - (abc[0] + P[:, :2] @ abc[1:])) < tol
    return abc


def floor_targets(P, room, abc, cell, heights, all_floor=False, inset=0.1, occ_lo=0.15, occ_hi=1.8, occ_min=3):
    """Cells inside the room (in its turned frame), free unless all_floor.
    -> (cell centres xy in the map (n,2), target points (n, len(heights), 3),
    grid shape, cell index (i, j) per centre, free mask over the grid)."""
    yaw, x0, x1, y0, y1 = room
    Rz = rot_z(-yaw)
    Pr = P @ Rz.T
    xs = np.arange(x0 + inset + cell / 2, x1 - inset, cell)
    ys = np.arange(y0 + inset + cell / 2, y1 - inset, cell)
    hp = P[:, 2] - (abc[0] + P[:, :2] @ abc[1:])
    occ = np.zeros((len(xs), len(ys)), int)
    m = (hp > occ_lo) & (hp < occ_hi)
    i = np.floor((Pr[m, 0] - (x0 + inset)) / cell).astype(int)
    j = np.floor((Pr[m, 1] - (y0 + inset)) / cell).astype(int)
    ok = (i >= 0) & (i < len(xs)) & (j >= 0) & (j < len(ys))
    np.add.at(occ, (i[ok], j[ok]), 1)
    free = occ < occ_min
    use = np.ones_like(free) if all_floor else free
    I, J = np.nonzero(use)
    cr = np.c_[xs[I], ys[J]]
    cm = cr @ Rz[:2, :2]                                              # back to the map: R(yaw)
    zf = abc[0] + cm @ abc[1:]
    T = np.stack([np.c_[cm, zf + h] for h in heights], 1)
    return cm, T, (len(xs), len(ys)), np.c_[I, J], free


# ---------------------------------------------------------------- visibility
def unoccluded(scene, T, o, max_range, slack=0.25):
    """Points of T (n,3) with a clear line to o (nothing of the mesh nearer
    than their distance less slack), within max_range."""
    import open3d as o3d
    v = o - T
    d = np.linalg.norm(v, axis=1)
    u = v / np.maximum(d, 1e-9)[:, None]
    rays = np.c_[T + 0.02 * u, u].astype(np.float32)
    t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    return (t >= d - 0.02 - slack) & (d <= max_range)


def seen_by_aims(T, clear, o, Rs, K, min_ppm, margin, min_heights, chunk=128):
    """(n_aims, n_cells) bool: cell seen by the camera at o with each aim;
    K: parse_intrinsics' dict (lens distortion applied)."""
    W, H, fx, fy, cx, cy = K["W"], K["H"], K["fx"], K["fy"], K["cx"], K["cy"]
    D = np.r_[K["D"], np.zeros(5)][:5]
    k1, k2, p1, p2, k3 = D
    n, k, _ = T.shape
    V = (T - o).reshape(-1, 3)
    ok = clear.reshape(-1)
    out = np.zeros((len(Rs), n), bool)
    mu, mv = margin * W, margin * H
    for s in range(0, len(Rs), chunk):
        R = np.asarray(Rs[s:s + chunk])                               # (a, 3, 3) camera -> map
        C = np.einsum("nj,ajk->ank", V, R)                            # map -> camera: V @ R
        z = C[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = C[..., 0] / z, C[..., 1] / z
            r2 = x * x + y * y
            g = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
            xd = x * g + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            yd = y * g + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            u = fx * xd + cx
            v = fy * yd + cy
            ppm = fx * g / z                                          # local scale at the point
            vis = ((z > 0.1) & (r2 <= K["r_max"] ** 2) & (u >= mu) & (u < W - mu) & (v >= mv) & (v < H - mv)
                   & (ppm >= min_ppm) & ok[None])
        out[s:s + chunk] = vis.reshape(len(R), n, k).sum(2) >= min_heights
    return out


# ---------------------------------------------------------------- the choice
def score(n, w1):
    return float((n >= 2).sum() + w1 * (n >= 1).sum())


def ascend(S, start, w1, change=None, max_sweeps=50, rng=None):
    """Coordinate ascent: each camera in turn its best aim given the others.
    S: list per camera of (n_aims, n_cells) bool. -> aims, score."""
    aims = list(start)
    n = sum(S[c][aims[c]].astype(np.int32) for c in range(len(S)))
    order = list(range(len(S)))
    for _ in range(max_sweeps):
        moved = False
        if rng is not None:
            rng.shuffle(order)
        for c in order:
            rest = n - S[c][aims[c]]
            g2 = (rest >= 2).sum() + (S[c] & (rest == 1)[None]).sum(1)
            g1 = (rest >= 1).sum() + (S[c] & (rest == 0)[None]).sum(1)
            g = g2 + w1 * g1
            if change is not None:
                g = g - 1e-6 * change[c]                              # ties: the smallest turn
            b = int(np.argmax(g))
            if g[b] > g[aims[c]] + 1e-9:
                aims[c], moved = b, True
            n = rest + S[c][aims[c]]
        if not moved:
            break
    return aims, score(n, w1)


def exact(S, w1, keep, time_limit, log=print):
    """The integer program over each camera's `keep` aims (index lists):
    -> (aims, objective, dual bound) or None."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import csr_matrix, vstack, hstack, identity
    nc = S[0].shape[1]
    cols, owner = [], []
    for c, ks in enumerate(keep):
        for a in ks:
            cols.append(S[c][a])
            owner.append((c, a))
    A = csr_matrix(np.array(cols, np.float64).T)                      # cells x chosen aims
    nx = A.shape[1]
    # variables: x (nx), y2 (nc), y1 (nc); maximise y2 + w1 y1
    cost = np.r_[np.zeros(nx), -np.ones(nc), -w1 * np.ones(nc)]
    I = identity(nc, format="csr")
    Z = csr_matrix((nc, nc))
    c2 = hstack([-A, 2 * I, Z])                                       # 2 y2 <= sum a x
    c1 = hstack([-A, Z, I])                                           # y1 <= sum a x
    rows = []
    for c in range(len(S)):
        r = np.zeros(nx + 2 * nc)
        r[[k for k, (cc, _) in enumerate(owner) if cc == c]] = 1
        rows.append(r)
    one = csr_matrix(np.array(rows))
    cons = [LinearConstraint(vstack([c2, c1]), -np.inf, 0), LinearConstraint(one, 1, 1)]
    res = milp(cost, constraints=cons, integrality=np.r_[np.ones(nx), np.ones(2 * nc)],
               bounds=Bounds(0, 1), options={"time_limit": float(time_limit), "disp": False})
    if res.x is None:
        log("  exact: no solution in %.0f s (%s)" % (time_limit, res.message))
        return None
    x = res.x[:nx] > 0.5
    aims = [None] * len(S)
    for k in np.flatnonzero(x):
        c, a = owner[k]
        aims[c] = a
    bound = getattr(res, "mip_dual_bound", None)
    return aims, -float(res.fun), (-float(bound) if bound is not None else None)


# ---------------------------------------------------------------- pictures
def picture(path, shape, ij, n, free, cams, room, cell, title, px=60):
    """The floor from above (the room's turned frame), coloured by count."""
    import cv2
    yaw, x0, x1, y0, y1 = room
    gw, gh = shape
    img = np.full((gh * int(px * cell), gw * int(px * cell), 3), 255, np.uint8)
    s = int(px * cell)
    col = {0: (60, 60, 230), 1: (40, 160, 245), 2: (90, 200, 90), 3: (40, 130, 40)}
    for i in range(gw):
        for j in range(gh):
            if not free[i, j]:
                img[(gh - 1 - j) * s:(gh - j) * s, i * s:(i + 1) * s] = (190, 190, 190)
    for (i, j), k in zip(ij, n):
        img[(gh - 1 - j) * s:(gh - j) * s, i * s:(i + 1) * s] = col[min(int(k), 3)]
    pad = 60
    canvas = np.full((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad, 3), 255, np.uint8)
    canvas[pad:pad + img.shape[0], pad:pad + img.shape[1]] = img
    Rz = rot_z(-yaw)
    inset = 0.1

    def to_px(xy):
        r = Rz[:2, :2] @ xy
        return (int(pad + (r[0] - (x0 + inset)) * px), int(pad + img.shape[0] - (r[1] - (y0 + inset)) * px))
    for name, o, R in cams:
        p = to_px(o[:2])
        d = R[:2, 2]
        if np.linalg.norm(d) > 1e-6:
            q = to_px(o[:2] + 0.8 * d / np.linalg.norm(d))
            cv2.arrowedLine(canvas, p, q, (0, 0, 0), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(canvas, p, 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(canvas, name.replace("cam_", ""), (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (pad, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    lx = pad
    for k, lab in ((0, "0"), (1, "1"), (2, "2"), (3, "3+")):
        cv2.rectangle(canvas, (lx, canvas.shape[0] - 35), (lx + 18, canvas.shape[0] - 17), col[k], -1)
        cv2.putText(canvas, lab, (lx + 22, canvas.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        lx += 60
    cv2.rectangle(canvas, (lx, canvas.shape[0] - 35), (lx + 18, canvas.shape[0] - 17), (190, 190, 190), -1)
    cv2.putText(canvas, "not a target", (lx + 22, canvas.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(path, canvas)


def counts(n):
    return {"0": int((n == 0).sum()), "1": int((n == 1).sum()), "2": int((n == 2).sum()), "3+": int((n >= 3).sum()),
            "share_2plus": round(float((n >= 2).mean()), 4), "share_1plus": round(float((n >= 1).mean()), 4)}


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("cameras")
    ap.add_argument("out")
    ap.add_argument("--map", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--room", type=float, nargs=5, required=True, metavar=("YAW", "XMIN", "XMAX", "YMIN", "YMAX"))
    ap.add_argument("--intrinsics", default=None, help="a calibration file, a table of them, or a folder")
    ap.add_argument("--intrinsics-all", default=None, help="one camera's calibration for every camera without its own")
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--vfov", type=float, default=60.0)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--cell", type=float, default=0.15)
    ap.add_argument("--heights", default="0.2,1.0,1.6")
    ap.add_argument("--min-heights", type=int, default=2)
    ap.add_argument("--all-floor", action="store_true")
    ap.add_argument("--max-range", type=float, default=10.0)
    ap.add_argument("--min-ppm", type=float, default=50.0)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--pan-step", type=float, default=5.0)
    ap.add_argument("--tilt", default="5,85,5", help="degrees below the horizon: from,to,step")
    ap.add_argument("--max-change", type=float, default=0.0)
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--weight-one", type=float, default=0.1)
    ap.add_argument("--exact", type=float, default=0.0, help="seconds for the integer program (0: none)")
    ap.add_argument("--exact-aims", type=int, default=40, help="per camera, for the integer program")
    a = ap.parse_args(argv)
    import open3d as o3d
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    doc = yaml.safe_load(open(os.path.expanduser(a.cameras)))
    cams = [(t["child"], T_of(t["translation"], t["rotation_xyzw"])) for t in doc["transforms"]]
    size = tuple(int(v) for v in a.size.split("x"))
    table = load_intrinsics(a.intrinsics) if a.intrinsics else {}
    fallback = None
    if a.intrinsics_all:
        f = load_intrinsics(a.intrinsics_all)
        if len(f) != 1:
            raise SystemExit("--intrinsics-all: one camera's calibration expected, %d found" % len(f))
        fallback = list(f.values())[0]
    Ks = {}
    for name, _ in cams:
        if name in table:
            Ks[name], src = table[name][0], os.path.basename(table[name][1])
        elif fallback is not None:
            Ks[name], src = fallback[0], "%s (--intrinsics-all)" % os.path.basename(fallback[1])
        else:
            Ks[name], src = pinhole(a.hfov, a.vfov, size), "--hfov %.0f / --vfov %.0f, no distortion" % (a.hfov, a.vfov)
        h, v = fov_deg(Ks[name])
        print("  %-7s %dx%d, field of view %.1f x %.1f deg  (%s)" % (name, Ks[name]["W"], Ks[name]["H"], h, v, src))
    P = np.asarray(o3d.io.read_point_cloud(os.path.expanduser(a.map)).points)
    abc = floor_plane(P)
    print("floor: z = %.3f %+.4f x %+.4f y (tilt %.2f deg)" % (abc[0], abc[1], abc[2],
                                                                np.degrees(np.arctan(np.linalg.norm(abc[1:])))))
    heights = [float(v) for v in a.heights.split(",")]
    cm, T, shape, ij, free = floor_targets(P, tuple(a.room), abc, a.cell, heights, a.all_floor)
    print("targets: %d %s cells of %.2f m (%.1f m2), %d heights each %s, seen when %d are"
          % (len(cm), "floor" if a.all_floor else "free floor", a.cell, len(cm) * a.cell ** 2, len(heights),
             heights, a.min_heights))
    mesh = o3d.io.read_triangle_mesh(os.path.expanduser(a.mesh))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    t0, t1, ts = (float(v) for v in a.tilt.split(","))
    grid = [(p, -t) for t in np.arange(t0, t1 + 1e-9, ts) for p in np.arange(-180.0, 180.0, a.pan_step)]
    S, Rs_all, now_idx, change = [], [], [], []
    flat = T.reshape(-1, 3)
    for name, Tm in cams:
        o, R0 = Tm[:3, 3], Tm[:3, :3]
        Rs = [R0] + [aim_rotation(p, t) for p, t in grid]
        ch = np.array([axis_angle(R0, R) for R in Rs])
        if a.max_change > 0:
            keep = ch <= a.max_change
            Rs = [R for R, k in zip(Rs, keep) if k]
            ch = ch[keep]
        clear = unoccluded(scene, flat, o, a.max_range).reshape(T.shape[:2])
        S.append(seen_by_aims(T, clear, o, Rs, Ks[name], a.min_ppm, a.margin, a.min_heights))
        Rs_all.append(Rs)
        now_idx.append(0)
        change.append(ch)
        print("  %-7s at %s: %d aims; now sees %d cells, its best aim alone %d (%.0f %% of the line of sight)"
              % (name, np.round(o, 2).tolist(), len(Rs), int(S[-1][0].sum()), int(S[-1].sum(1).max()),
                 100 * clear.any(1).mean()))
    n0 = sum(S[c][0].astype(np.int32) for c in range(len(S)))
    best, best_sc = ascend(S, now_idx, a.weight_one, change)
    print("coordinate ascent from the present aims: %.1f" % best_sc)
    rng = np.random.default_rng(0)
    for r in range(a.restarts):
        start = [int(rng.integers(len(s))) for s in S]
        aims, sc = ascend(S, start, a.weight_one, change, rng=rng)
        if sc > best_sc + 1e-9:
            best, best_sc = aims, sc
            print("  restart %d: %.1f" % (r + 1, sc))
    bound = None
    if a.exact > 0:
        keep = []
        for c in range(len(S)):
            top = list(np.argsort(-S[c].sum(1))[:a.exact_aims])
            keep.append(sorted(set(top) | {best[c], 0}))
        ex = exact(S, a.weight_one, keep, a.exact)
        if ex is not None:
            aims, obj, bound = ex
            print("integer program (%d aims per camera): %.1f, bound %s" % (a.exact_aims, obj,
                                                                             "%.1f" % bound if bound else "-"))
            if obj > best_sc + 1e-9 and None not in aims:
                best, best_sc = aims, obj
    n1 = sum(S[c][best[c]].astype(np.int32) for c in range(len(S)))
    c0, c1 = counts(n0), counts(n1)
    area = a.cell ** 2
    print("\nfloor seen by >= 2 cameras: %.1f %% -> %.1f %%;  by >= 1: %.1f %% -> %.1f %%;  by none: %.1f m2 -> %.1f m2"
          % (100 * c0["share_2plus"], 100 * c1["share_2plus"], 100 * c0["share_1plus"], 100 * c1["share_1plus"],
             c0["0"] * area, c1["0"] * area))
    rows, transforms = [], []
    print("\n  camera   pan now -> new   tilt now -> new   turn   cells seen now -> new")
    for c, (name, Tm) in enumerate(cams):
        Rn = Rs_all[c][best[c]]
        p0, t0_ = pan_tilt(Tm[:3, :3])
        p1, t1_ = pan_tilt(Rn)
        turn = axis_angle(Tm[:3, :3], Rn)
        rows.append({"camera": name, "pan_now": round(p0, 1), "tilt_now": round(t0_, 1), "pan_new": round(p1, 1),
                     "tilt_new": round(t1_, 1), "turn_deg": round(turn, 1), "cells_now": int(S[c][0].sum()),
                     "cells_new": int(S[c][best[c]].sum()), "kept": best[c] == 0})
        print("  %-7s %7.1f -> %6.1f   %6.1f -> %6.1f   %5.1f   %d -> %d%s"
              % (name, p0, p1, t0_, t1_, turn, rows[-1]["cells_now"], rows[-1]["cells_new"],
                 "   (as it is)" if best[c] == 0 else ""))
        from scipy.spatial.transform import Rotation as Rot
        transforms.append({"parent": "map", "child": name, "translation": [round(float(v), 6) for v in Tm[:3, 3]],
                           "rotation_xyzw": [round(float(v), 6) for v in Rot.from_matrix(Rn).as_quat()]})
    with open(os.path.join(out, "cameras_aimed.yaml"), "w") as fh:
        fh.write("# the cameras re-aimed by camera_coverage.py (places as in %s)\n" % os.path.basename(a.cameras))
        yaml.safe_dump({"transforms": transforms}, fh, sort_keys=False)
    json.dump({"cameras": os.path.abspath(os.path.expanduser(a.cameras)), "cells": len(cm), "cell_m": a.cell,
               "heights": heights, "min_heights": a.min_heights, "floor_plane": abc.tolist(),
               "before": c0, "after": c1, "objective": {"before": score(n0, a.weight_one), "after": best_sc,
                                                        "bound": bound},
               "per_camera": rows}, open(os.path.join(out, "coverage.json"), "w"), indent=1)
    picture(os.path.join(out, "coverage_before.png"), shape, ij, n0, free, [(n, Tm[:3, 3], Tm[:3, :3]) for n, Tm in cams],
            tuple(a.room), a.cell, "now: %.0f %% of the floor seen by 2+ cameras" % (100 * c0["share_2plus"]))
    picture(os.path.join(out, "coverage_after.png"), shape, ij, n1, free,
            [(n, Tm[:3, 3], Rs_all[c][best[c]]) for c, (n, Tm) in enumerate(cams)],
            tuple(a.room), a.cell, "re-aimed: %.0f %% of the floor seen by 2+ cameras" % (100 * c1["share_2plus"]))
    print("\n-> %s (coverage.json, cameras_aimed.yaml, coverage_before.png, coverage_after.png)" % out)


if __name__ == "__main__":
    sys.exit(main())
